// C++ implementation of the pure-C OAK bridge over depthai-core v3.
// One device (USB or PoE) → Camera (CAM_A, RGB) + StereoDepth (CAM_B/CAM_C, aligned
// to CAM_A) + Sync, exposing a time-synced RGB888 + uint16-mm depth pair, PLUS an
// optional standalone on-device H.264 colour stream (a second camera output → encoder).
// All C++ exceptions are caught and surfaced as return codes + oak_last_error().

#include "oak_bridge.h"
#include "depthai/depthai.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static thread_local std::string g_err;
static void set_err(const std::string& m) { g_err = m; }

// depthai's getTimestamp() is on the host STEADY clock, synchronized across ALL connected devices, so
// multiple cameras share one timeline (essential for multi-camera alignment). getTimestampDevice() is
// per-device boot time — neither wall-clock nor comparable across cameras. We shift the steady stamp
// onto the system clock so the value forwarded downstream (publishers, recordings) is a real
// epoch time every camera agrees on. Shared by frames AND IMU reports so both land on ONE timeline —
// the whole point of the stereo+IMU modality, where the consumer interpolates inertial samples
// between image timestamps.
//
// The offset is computed once per batch rather than per sample: reading both clocks costs two
// clock_gettime calls each (hundreds/second at IMU rates) and, worse, lets the offset drift mid-batch
// so samples from a single transfer would disagree about what "now" was.
static std::chrono::nanoseconds steady_epoch_offset() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
         - duration_cast<nanoseconds>(steady_clock::now().time_since_epoch());
}

static uint64_t steady_to_epoch_ns(std::chrono::steady_clock::time_point t,
                                   std::chrono::nanoseconds offset) {
    using namespace std::chrono;
    return (uint64_t)(duration_cast<nanoseconds>(t.time_since_epoch()) + offset).count();
}

static uint64_t frame_epoch_ns(const std::shared_ptr<dai::ImgFrame>& f) {
    return steady_to_epoch_ns(f->getTimestamp(), steady_epoch_offset());
}

// Connect to an OAK by id (NULL/"" = first available), honouring the USB-speed cap below.
//
// Cap the USB link speed. The default (SUPER/USB3) boots the device into a USB3 descriptor; on a
// physical USB2 link the host then can't reconnect to the booted device → X_LINK_DEVICE_NOT_FOUND.
// Defaulting to HIGH (USB2) gives a stable link. Override with OAK_USB_SPEED=super on a USB3 cable.
// (Ignored for a PoE device, which connects over TCP/IP.)
//
// Device selection: a non-empty device_id picks a SPECIFIC camera by MxId (USB or PoE) or IP (PoE) —
// depthai's string ctor resolves the transport. Empty/NULL → first available device.
static std::shared_ptr<dai::Device> connect_device(const char* device_id) {
    dai::UsbSpeed usb_speed = dai::UsbSpeed::HIGH;
    if (const char* s = std::getenv("OAK_USB_SPEED")) {
        if (std::string(s) == "super" || std::string(s) == "SUPER") usb_speed = dai::UsbSpeed::SUPER;
    }
    if (device_id != nullptr && device_id[0] != '\0') {
        return std::make_shared<dai::Device>(std::string(device_id), usb_speed);
    }
    return std::make_shared<dai::Device>(usb_speed);
}



struct oak_device {
    std::shared_ptr<dai::Device> device;
    std::unique_ptr<dai::Pipeline> pipeline;
    std::shared_ptr<dai::MessageQueue> stereo_q; // Sync'd {left,right} MessageGroup (oak_open_stereo)
    std::shared_ptr<dai::MessageQueue> imu_q;    // IMUData batches, far faster than the frame rate
    // Keep the current frames alive so their buffers (handed out as raw spans) stay
    // valid until the next poll — zero-copy: no host repack.
    // The stereo pair is pinned via the two eye frames (NOT the MessageGroup): holding the group
    // alone would not keep the ImgFrame buffers alive once it hands out shared_ptrs.
    std::shared_ptr<dai::ImgFrame> cur_left;
    std::shared_ptr<dai::ImgFrame> cur_right;
    // IMU samples popped off the queue but not yet handed to the caller (see oak_poll_imu).
    std::vector<oak_imu_sample> imu_pending;
    bool has_stereo = false;  // stereo-pair pipeline running (oak_open_stereo)
    bool has_imu = false;     // on-board IMU running (optional even in stereo mode)
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
};


// STEREO+IMU modality: the two mono cameras as a Sync'd RGB888 pair + the IMU on its own queue.
// Shares no nodes with oak_open's colour/depth pipeline — deliberately a separate entry point so the
// working RGBD and H.264 paths cannot regress.
extern "C" oak_device* oak_open_stereo(const char* device_id, int width, int height,
                                       int fps, int imu_hz) {
    try {
        auto dev = std::make_unique<oak_device>();
        dev->width = width;
        dev->height = height;
        dev->device = connect_device(device_id);
        dev->pipeline = std::make_unique<dai::Pipeline>(dev->device);
        auto& pipeline = *dev->pipeline;

        // The stereo pair is the whole point of this modality — unlike depth (which oak_open silently
        // falls back from), a missing mono socket here has no meaningful degraded mode. Fail loudly.
        bool has_b = false, has_c = false;
        for (auto s : dev->device->getConnectedCameras()) {
            if (s == dai::CameraBoardSocket::CAM_B) has_b = true;
            else if (s == dai::CameraBoardSocket::CAM_C) has_c = true;
        }
        if (!has_b || !has_c) {
            set_err("device has no stereo pair (CAM_B/CAM_C) — oak_open_stereo needs both");
            return nullptr;
        }

        // Mono sensors requested as RGB888i: depthai replicates gray across 3 channels on-device, so
        // the frames land tightly packed in exactly the layout a 3-channel consumer (kornia
        // Image<u8,3>, and through it XFeat) wants — no host conversion, no repack.
        auto left = pipeline.create<dai::node::Camera>();
        left->build(dai::CameraBoardSocket::CAM_B);
        auto right = pipeline.create<dai::node::Camera>();
        right->build(dai::CameraBoardSocket::CAM_C);
        const std::pair<uint32_t, uint32_t> size((uint32_t)width, (uint32_t)height);
        auto* lo = left->requestOutput(size, dai::ImgFrame::Type::RGB888i,
                                       dai::ImgResizeMode::CROP, (float)fps,
                                       /*enableUndistortion=*/true);
        auto* ro = right->requestOutput(size, dai::ImgFrame::Type::RGB888i,
                                        dai::ImgResizeMode::CROP, (float)fps,
                                        /*enableUndistortion=*/true);

        // Sync node: emit {left,right} as ONE MessageGroup so the host never has to pair by timestamp.
        // The eyes are frame-locked by the shared stereo trigger, so the threshold only has to absorb
        // transport jitter — half a frame interval is generous.
        auto sync = pipeline.create<dai::node::Sync>();
        sync->setSyncThreshold(std::chrono::nanoseconds(
            fps > 0 ? (1000000000LL / fps) / 2 : 16000000LL));
        lo->link(sync->inputs["left"]);
        ro->link(sync->inputs["right"]);
        dev->stereo_q = sync->out.createOutputQueue(4, false);
        dev->has_stereo = true;

        // IMU is OPTIONAL: not every OAK carries one, and a failed IMU must not cost us the stereo
        // pair. Same degrade-don't-die discipline as device_has_stereo in oak_open.
        if (imu_hz > 0) try {
            auto imu = pipeline.create<dai::node::IMU>();
            imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, (uint32_t)imu_hz);
            imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, (uint32_t)imu_hz);
            // Batch a few reports per message (fewer, larger XLink transfers) but keep the batch
            // small enough that inertial data stays fresh relative to the frames.
            imu->setBatchReportThreshold(5);
            imu->setMaxBatchReports(20);
            dev->imu_q = imu->out.createOutputQueue(50, false);
            dev->has_imu = true;
        } catch (const std::exception&) {
            dev->has_imu = false;   // no IMU on this board — stereo still streams
        }

        pipeline.start();

        // Left (CAM_B) is the reference frame of a stereo rig, so oak_intrinsics reports ITS
        // intrinsics in this modality (CAM_A, the colour camera, isn't even in this pipeline).
        try {
            auto calib = dev->device->readCalibration();
            auto k = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, width, height);
            dev->fx = k[0][0]; dev->fy = k[1][1];
            dev->cx = k[0][2]; dev->cy = k[1][2];
        } catch (const std::exception&) { /* intrinsics stay zero — uncalibrated board */ }

        return dev.release();
    } catch (const std::exception& e) { set_err(e.what()); return nullptr; }
    catch (...) { set_err("unknown error in oak_open_stereo"); return nullptr; }
}

extern "C" int oak_has_stereo(const oak_device* dev) {
    return (dev && dev->has_stereo) ? 1 : 0;
}

extern "C" int oak_has_imu(const oak_device* dev) {
    return (dev && dev->has_imu) ? 1 : 0;
}

// Validate one eye and hand out its buffer zero-copy. The Rust side assumes TIGHT rows
// (stride == w*3), so verify that rather than trust it — same check as oak_poll's RGB.
static bool eye_span(const std::shared_ptr<dai::ImgFrame>& f, int w, int h,
                     const uint8_t** out) {
    if (!f) return false;
    if ((int)f->getWidth() != w || (int)f->getHeight() != h) return false;
    auto d = f->getData();
    const unsigned int stride = f->getStride();
    if (stride != 0 && stride != (unsigned)w * 3) return false;
    if (d.size() < (size_t)w * h * 3) return false;
    *out = d.data();
    return true;
}

extern "C" int oak_poll_stereo(oak_device* dev,
                               const uint8_t** left, const uint8_t** right,
                               int* width, int* height, int* len, uint64_t* ts_ns) {
    if (!dev) { set_err("null device"); return -1; }
    if (!left || !right || !width || !height || !len || !ts_ns) {
        set_err("null out pointer"); return -1;
    }
    if (!dev->stereo_q) return 0;   // not a stereo build
    try {
        bool timed_out = false;
        auto group = dev->stereo_q->get<dai::MessageGroup>(std::chrono::seconds(1), timed_out);
        if (timed_out || !group) return 0;

        auto l = group->get<dai::ImgFrame>("left");
        auto r = group->get<dai::ImgFrame>("right");
        if (!l || !r) { set_err("stereo group missing an eye"); return -1; }

        const int w = (int)l->getWidth(), h = (int)l->getHeight();
        if (w <= 0 || h <= 0) return 0;   // degenerate frame — skip, don't kill the stream
        // Pin BOTH eyes for the lifetime of the caller's spans (until the next poll).
        dev->cur_left = l;
        dev->cur_right = r;
        if (!eye_span(l, w, h, left) || !eye_span(r, w, h, right)) {
            set_err("stereo eye is not tightly packed RGB888 (stride != w*3) or eyes differ in size");
            return -1;
        }
        *width = w; *height = h;
        *len = w * h * 3;
        *ts_ns = frame_epoch_ns(l);
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_stereo"); return -1; }
}

// Retain/release for the current stereo eye. The handle is a heap-allocated copy of the
// shared_ptr, so the ImgFrame (and its pixel buffer) survives until the handle is freed —
// independently of dev->cur_left/cur_right being reassigned by the next poll.
extern "C" void* oak_stereo_retain(oak_device* dev, int eye) {
    if (!dev) { set_err("null device"); return nullptr; }
    try {
        const std::shared_ptr<dai::ImgFrame>& f = (eye == 0) ? dev->cur_left : dev->cur_right;
        if (!f) { set_err("no current stereo frame to retain"); return nullptr; }
        return new std::shared_ptr<dai::ImgFrame>(f);   // refcount++
    } catch (const std::exception& e) { set_err(e.what()); return nullptr; }
    catch (...) { set_err("unknown error in oak_stereo_retain"); return nullptr; }
}

extern "C" void oak_frame_release(void* handle) {
    if (!handle) return;
    delete static_cast<std::shared_ptr<dai::ImgFrame>*>(handle);   // refcount--
}

extern "C" int oak_poll_imu(oak_device* dev, oak_imu_sample* out, int max, int* n) {
    if (!dev) { set_err("null device"); return -1; }
    if (!out || !n) { set_err("null out pointer"); return -1; }
    *n = 0;
    if (!dev->has_imu || !dev->imu_q || max <= 0) return 0;
    try {
        // tryGet pops the batch destructively, so a batch that doesn't fit in the caller's array
        // CANNOT simply be left behind — it would be gone. Convert whole batches into a staging
        // buffer, hand out what fits, and keep the remainder (in order) for the next call.
        const auto offset = steady_epoch_offset();   // one clock pair for the whole drain
        while ((int)dev->imu_pending.size() < max) {
            auto data = dev->imu_q->tryGet<dai::IMUData>();
            if (!data) break;
            for (const auto& p : data->packets) {
                oak_imu_sample s;
                s.ts_ns = steady_to_epoch_ns(p.acceleroMeter.getTimestamp(), offset);
                s.ax = p.acceleroMeter.x; s.ay = p.acceleroMeter.y; s.az = p.acceleroMeter.z;
                s.gx = p.gyroscope.x;     s.gy = p.gyroscope.y;     s.gz = p.gyroscope.z;
                dev->imu_pending.push_back(s);
            }
        }
        const int take = std::min((int)dev->imu_pending.size(), max);
        std::copy(dev->imu_pending.begin(), dev->imu_pending.begin() + take, out);
        dev->imu_pending.erase(dev->imu_pending.begin(), dev->imu_pending.begin() + take);
        *n = take;
        return take > 0 ? 1 : 0;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_imu"); return -1; }
}


extern "C" int oak_intrinsics(const oak_device* dev, float* fx, float* fy, float* cx, float* cy) {
    if (!dev) { set_err("null device"); return -1; }
    if (!fx || !fy || !cx || !cy) { set_err("null out pointer"); return -1; }
    *fx = dev->fx; *fy = dev->fy; *cx = dev->cx; *cy = dev->cy;
    return 0;
}


// Recover a PoE OAK wedged in bootloader state (the "X_LINK_BOOTLOADER" signature that in-process
// oak_open retries can never clear): enumerate, find the target by IP/name or deviceId, and if it is
// sitting in a BOOTLOADER state, open+drop a DeviceBootloader — that reboots it back to UNBOOTED so the
// next oak_open can flash+run the pipeline fresh. `target` NULL = kick the first wedged device found.
// Returns 1 = a device was kicked (caller should wait ~8s for the reboot), 0 = nothing to kick (target
// not present, or present but not wedged), -1 = error (see oak_last_error). Blocking; call off the hot path.
extern "C" int oak_kick(const char* target) {
    try {
        std::string want = target ? target : "";
        auto infos = dai::Device::getAllAvailableDevices();
        for (const auto& info : infos) {
            if (!want.empty() && info.name != want && info.deviceId != want) continue;
            // The wedge: a PoE device stuck in the bootloader. A healthy device enumerates UNBOOTED /
            // BOOTED / FLASH_BOOTED and opens normally — kicking it would be a pointless reboot.
            if (info.state != X_LINK_BOOTLOADER) {
                if (!want.empty()) {
                    set_err("device '" + want + "' is present but not wedged (state "
                            + std::to_string((int)info.state) + ")");
                    return 0;
                }
                continue;
            }
            // Open+drop the bootloader connection: construction connects to the wedged firmware, and
            // destruction reboots the device to an unbooted state (the manual recovery, in-process).
            { dai::DeviceBootloader bl(info); }
            return 1;
        }
        set_err(want.empty() ? "no wedged device found" : ("device '" + want + "' not found"));
        return 0;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_kick"); return -1; }
}

extern "C" void oak_close(oak_device* dev) {
    if (!dev) return;
    try { if (dev->pipeline) dev->pipeline->stop(); } catch (...) {}
    // Gracefully close the XLink connection before destruction so the firmware
    // isn't torn down mid-stream (avoids a spurious crash-dump on USB2 disconnect).
    try { if (dev->device) dev->device->close(); } catch (...) {}
    delete dev;
}

extern "C" const char* oak_last_error(void) { return g_err.c_str(); }
