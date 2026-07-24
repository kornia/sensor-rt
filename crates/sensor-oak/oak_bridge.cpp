// C++ implementation of the pure-C OAK bridge over depthai-core v3.
// One device (USB or PoE) → the two mono cameras (CAM_B/CAM_C) through a Sync node,
// exposing a time-synced GRAY8 stereo pair, plus the on-board IMU on its own queue.
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

// H.264 target bitrate in kbps. The preset's default scales with resolution and is generous; 2000 kbps
// is comfortable for 640x360@30 and a big cut for a phone/Tailscale hop. Override with OAK_H264_KBPS.
static int h264_kbps() {
    if (const char* s = std::getenv("OAK_H264_KBPS")) {
        int v = std::atoi(s);
        if (v > 0) return v;
    }
    return 2000;
}

// True only if the device can actually produce aligned depth: it has BOTH stereo mono sockets
// (CAM_B + CAM_C) AND a readable factory calibration (a wiped/blank EEPROM reads back fx=0, which makes
// StereoDepth emit garbage/zero-scale depth). Used to auto-fall-back an uncalibrated or mono camera to
// the lean video-only pipeline instead of pulling raw RGB for a depth that would be unusable.
static bool device_has_stereo(const std::shared_ptr<dai::Device>& d, int w, int h) {
    try {
        bool has_b = false, has_c = false;
        for (auto s : d->getConnectedCameras()) {
            if (s == dai::CameraBoardSocket::CAM_B) has_b = true;
            else if (s == dai::CameraBoardSocket::CAM_C) has_c = true;
        }
        if (!has_b || !has_c) return false;
        auto k = d->readCalibration().getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, w, h);
        return k[0][0] > 0.0f;   // fx > 0 → real calibration present
    } catch (const std::exception&) { return false; }
}

struct oak_device {
    std::shared_ptr<dai::Device> device;
    std::unique_ptr<dai::Pipeline> pipeline;
    // STEREO+IMU modality (oak_open_stereo):
    std::shared_ptr<dai::MessageQueue> stereo_q; // Sync'd {left,right} MessageGroup
    std::shared_ptr<dai::MessageQueue> imu_q;    // IMUData batches, far faster than the frame rate
    // IMU samples popped off the queue but not yet handed to the caller (see oak_poll_imu).
    std::vector<oak_imu_sample> imu_pending;
    bool has_imu = false;     // on-board IMU running (optional even in stereo mode)
    // RGBD+H.264 modality (oak_open_rgbd): colour, depth, and video are DECOUPLED — each on its own
    // queue, pulled independently and paired downstream by timestamp. The cur_* frames pin the buffer
    // handed out as a raw span until the next poll of that stream (zero-copy: no host repack).
    std::shared_ptr<dai::MessageQueue> rgb_q;    // raw RGB888 colour, its own stream
    std::shared_ptr<dai::MessageQueue> depth_q;  // aligned uint16-mm depth, its own stream
    std::shared_ptr<dai::MessageQueue> video_q;  // H.264 bitstream, its own stream
    std::shared_ptr<dai::ImgFrame> cur_rgb;
    std::shared_ptr<dai::ImgFrame> cur_depth;
    std::shared_ptr<dai::ImgFrame> cur_video;
    std::vector<uint16_t> depth_repack;  // tightly-packed depth when the device row is stride-padded
    bool has_depth = false;   // StereoDepth running
    bool has_video = false;   // on-device H.264 colour stream running
    bool has_sync = false;    // colour(+depth) pipeline present; false = video-only fallback
    float fx = 0, fy = 0, cx = 0, cy = 0;
};

// Attach an NV12 output of `color` → a hardware H.264 encoder, handing the bitstream queue to `dev`.
// Shared by the video-only and decoupled RGBD paths so their encoder settings (BASELINE for
// Foxglove's decoder, ~4 keyframes/s for fast mid-stream join, OAK_H264_KBPS) can never drift apart.
// `fps` is clamped to >= 1 by the caller.
static void add_h264_encoder(dai::Pipeline& pipeline, const std::shared_ptr<dai::node::Camera>& color,
                             int width, int height, int fps, oak_device* dev) {
    auto* nv12_out = color->requestOutput(
        std::pair<uint32_t, uint32_t>((uint32_t)width, (uint32_t)height),
        dai::ImgFrame::Type::NV12, dai::ImgResizeMode::CROP, (float)fps, /*undistort=*/true);
    auto enc = pipeline.create<dai::node::VideoEncoder>();
    enc->setDefaultProfilePreset((float)fps, dai::VideoEncoderProperties::Profile::H264_BASELINE);
    enc->setKeyframeFrequency(std::max(fps / 4, 4));
    enc->setBitrateKbps(h264_kbps());
    nv12_out->link(enc->input);
    dev->video_q = enc->bitstream.createOutputQueue(30, false);
    dev->has_video = true;
}

// Read CAM_A factory intrinsics at the streamed size into `dev` (fx/fy/cx/cy). A wiped EEPROM or a
// device without CAM_A leaves them zero — fine for viewing, so the failure is swallowed.
static void read_rgb_intrinsics(oak_device* dev, int width, int height) {
    try {
        auto k = dev->device->readCalibration()
                     .getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, width, height);
        dev->fx = k[0][0]; dev->fy = k[1][1]; dev->cx = k[0][2]; dev->cy = k[1][2];
    } catch (const std::exception&) { /* intrinsics stay zero */ }
}

// STEREO+IMU modality: the two mono cameras as a Sync'd RGB888 pair + the IMU on its own queue.
// Shares no nodes with oak_open's colour/depth pipeline — deliberately a separate entry point so the
// working RGBD and H.264 paths cannot regress.
extern "C" oak_device* oak_open_stereo(const char* device_id, int width, int height,
                                       int fps, int imu_hz) {
    try {
        auto dev = std::make_unique<oak_device>();
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

        // CAM_B/CAM_C are MONOCHROME sensors, so they are requested as GRAY8 — one byte per pixel.
        // Asking for RGB888i would make depthai replicate the same gray value across three channels
        // on-device and then ship 3x the bytes over XLink for no information: 768 KB/eye at 640x400
        // instead of 256 KB. Consumers that need 3 channels (most models) expand it on the GPU, where
        // the copy is free next to the inference.
        auto left = pipeline.create<dai::node::Camera>();
        left->build(dai::CameraBoardSocket::CAM_B);
        auto right = pipeline.create<dai::node::Camera>();
        right->build(dai::CameraBoardSocket::CAM_C);
        const std::pair<uint32_t, uint32_t> size((uint32_t)width, (uint32_t)height);
        auto* lo = left->requestOutput(size, dai::ImgFrame::Type::GRAY8,
                                       dai::ImgResizeMode::CROP, (float)fps,
                                       /*enableUndistortion=*/true);
        auto* ro = right->requestOutput(size, dai::ImgFrame::Type::GRAY8,
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

        // IMU is OPTIONAL: not every OAK carries one, and a failed IMU must not cost us the stereo
        // pair — degrade rather than lose the stereo stream over a missing IMU.
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

extern "C" int oak_has_imu(const oak_device* dev) {
    return (dev && dev->has_imu) ? 1 : 0;
}

// RGBD + H.264 modality: CAM_A colour (RGB888) + StereoDepth aligned to it (uint16 mm) + an on-device
// H.264 colour stream, all DECOUPLED onto their own queues. Shares no nodes with oak_open_stereo — a
// separate entry point so neither modality can regress the other.
extern "C" oak_device* oak_open_rgbd(const char* device_id, int width, int height, int fps,
                                     int enable_h264, int enable_depth, int video_only) {
    try {
        auto dev = std::make_unique<oak_device>();
        dev->device = connect_device(device_id);
        dev->pipeline = std::make_unique<dai::Pipeline>(dev->device);
        auto& pipeline = *dev->pipeline;
        if (fps < 1) fps = 30;   // 0/negative fps would poison the encoder preset + requestOutput rate

        // Auto-fall-back to video-only when depth was requested but the device can't actually produce it
        // (mono camera, or wiped/blank calibration → fx=0). Pulling raw RGB over XLink for a "synced RGBD"
        // pair whose depth is garbage just caps the H.264 stream for nothing — build the lean video-only
        // pipeline instead. Policy: always ship compressed video; add RGBD only when depth works.
        bool want_video_only = (video_only != 0) ||
                               (enable_depth != 0 && !device_has_stereo(dev->device, width, height));

        // VIDEO-ONLY: just the H.264 encoder (CAM_A NV12 → encoder → queue). No RGB888/depth output, so
        // the device transmits ONLY the small H.264 bitstream (low-bandwidth viewing over USB2 / shared
        // gigabit). oak_poll_rgb/_depth yield nothing; the caller drains oak_poll_video.
        if (want_video_only) {
            auto color = pipeline.create<dai::node::Camera>();
            color->build(dai::CameraBoardSocket::CAM_A);
            add_h264_encoder(pipeline, color, width, height, fps, dev.get());
            dev->has_depth = false;
            dev->has_sync = false;
            pipeline.start();
            read_rgb_intrinsics(dev.get(), width, height);
            return dev.release();
        }

        // DECOUPLED build: raw-RGB and depth are SEPARATE streams (no on-device Sync node), each pulled at
        // its own rate + timestamped, so the consumer pairs them by timestamp. Depth runs at the mono/
        // stereo rate (OAK_DEPTH_FPS, default = fps); raw RGB — needed only for local compute — is pulled
        // at a LOW rate (OAK_RGB_FPS, default 10) to spare XLink. The H.264 video is a separate full-fps
        // output, unaffected by either.
        int dfps = (fps > 0 ? fps : 30);
        if (const char* s = std::getenv("OAK_DEPTH_FPS")) { int v = std::atoi(s); if (v >= 1) dfps = v; }
        if (fps > 0) dfps = std::min(dfps, fps);
        int rfps = 10;
        if (const char* s = std::getenv("OAK_RGB_FPS")) { int v = std::atoi(s); if (v >= 1) rfps = v; }
        if (fps > 0) rfps = std::min(rfps, fps);

        // Colour (CAM_A). Interleaved RGB888 (the ISP's native type, handed to the host zero-copy) as the
        // raw-RGB stream AND the depth-alignment reference; undistorted so depth aligns pixel-perfect and
        // the intrinsics are an exact pinhole.
        auto color = pipeline.create<dai::node::Camera>();
        color->build(dai::CameraBoardSocket::CAM_A);
        auto* rgb_out = color->requestOutput(
            std::pair<uint32_t, uint32_t>((uint32_t)width, (uint32_t)height),
            dai::ImgFrame::Type::RGB888i, dai::ImgResizeMode::CROP, (float)rfps,
            /*enableUndistortion=*/true);

        // StereoDepth aligned to the RGB OUTPUT (not just the CAM_A socket), so depth[u,v] matches
        // RGB[u,v] exactly — same CROP, same size, same intrinsics.
        std::shared_ptr<dai::node::StereoDepth> stereo;
        if (enable_depth) try {
            auto left = pipeline.create<dai::node::Camera>();
            left->build(dai::CameraBoardSocket::CAM_B);
            auto right = pipeline.create<dai::node::Camera>();
            right->build(dai::CameraBoardSocket::CAM_C);
            stereo = pipeline.create<dai::node::StereoDepth>();
            // ROBOTICS preset (depthai v3) is tuned for mobile-robot people/obstacle depth. Subpixel gives
            // ~8× finer disparity (removes the z-quantization that flickers a standing person's depth) but
            // ~halves the stereo FPS; OAK_SUBPIXEL=0 trades precision for rate. LR-check on for occlusion.
            stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::ROBOTICS);
            stereo->setLeftRightCheck(true);
            bool subpixel = true;
            if (const char* s = std::getenv("OAK_SUBPIXEL")) {
                subpixel = !(std::string(s) == "0" || std::string(s) == "false");
            }
            stereo->setSubpixel(subpixel);
            // Passive-stereo depth cleanup (no IR projector): SPATIAL edge-preserving hole-fill + TEMPORAL
            // averaging + THRESHOLD clamp to the useful range.
            {
                auto& pp = stereo->initialConfig->postProcessing;
                pp.spatialFilter.enable = true;
                pp.temporalFilter.enable = true;
                pp.thresholdFilter.minRange = 400;   // mm — drop closer than 0.4 m
                pp.thresholdFilter.maxRange = 8000;  // mm — and farther than 8 m
            }
            left->requestOutput(std::pair<uint32_t, uint32_t>(640, 400), std::nullopt,
                                dai::ImgResizeMode::CROP, (float)dfps)->link(stereo->left);
            right->requestOutput(std::pair<uint32_t, uint32_t>(640, 400), std::nullopt,
                                 dai::ImgResizeMode::CROP, (float)dfps)->link(stereo->right);
            rgb_out->link(stereo->inputAlignTo);   // align depth to the RGB output grid
            // Downscale the aligned depth ON-DEVICE before XLink. A room-scale point cloud doesn't need
            // per-RGB-pixel depth, and the full-res depth pull is the dominant XLink cost (it caps the
            // co-hosted H.264 on a PoE link). Default /2 → 1/4 the bytes; still aligned to the RGB grid,
            // so consumers scale coords by (rgb_w / depth_w).
            int ddiv = 2;
            if (const char* s = std::getenv("OAK_DEPTH_DIV")) { int v = std::atoi(s); if (v >= 1) ddiv = v; }
            // XLink requires EVEN depth dims — an odd width/height tears the device connection down
            // (X_LINK_ERROR, e.g. OAK_DEPTH_DIV=3 → 213x120). Round each down to even, floored at 2.
            stereo->setOutputSize(std::max(2, (width / ddiv) & ~1), std::max(2, (height / ddiv) & ~1));
            dev->has_depth = true;
        } catch (const std::exception&) {
            dev->has_depth = false;   // e.g. OAK-1: no stereo pair
        }

        // Each stream to its OWN non-blocking queue, pulled + published independently (consumer pairs by
        // timestamp). Frees depth from the rgb-pull rate.
        dev->rgb_q = rgb_out->createOutputQueue(4, false);
        if (dev->has_depth) dev->depth_q = stereo->depth.createOutputQueue(4, false);
        dev->has_sync = true; // has a colour(+depth) source to poll

        // Optional standalone H.264 colour stream: a SECOND CAM_A output (NV12, the encoder's input) →
        // hardware H.264 encoder → its own queue. BASELINE (no B-frames) for Foxglove's decoder; a
        // keyframe ~4×/s lets a viewer/recorder join mid-stream.
        if (enable_h264 != 0) {
            add_h264_encoder(pipeline, color, width, height, fps, dev.get());
        }

        pipeline.start();
        read_rgb_intrinsics(dev.get(), width, height);   // factory intrinsics of the aligned RGB camera

        return dev.release();
    } catch (const std::exception& e) { set_err(e.what()); return nullptr; }
    catch (...) { set_err("unknown error in oak_open_rgbd"); return nullptr; }
}

extern "C" int oak_has_depth(const oak_device* dev) { return (dev && dev->has_depth) ? 1 : 0; }
extern "C" int oak_has_video(const oak_device* dev) { return (dev && dev->has_video) ? 1 : 0; }
extern "C" int oak_has_sync(const oak_device* dev)  { return (dev && dev->has_sync)  ? 1 : 0; }

extern "C" int oak_poll_rgb(oak_device* dev, const uint8_t** rgb,
                            int* width, int* height, int* len, uint64_t* ts_ns) {
    if (!dev) { set_err("null device"); return -1; }
    if (!rgb || !width || !height || !len || !ts_ns) { set_err("null out pointer"); return -1; }
    if (!dev->rgb_q) return 0;   // video-only build: no colour queue to poll
    try {
        auto f = dev->rgb_q->tryGet<dai::ImgFrame>();
        if (!f) return 0;
        auto rd = f->getData();
        const int w = (int)f->getWidth(), h = (int)f->getHeight();
        const size_t npx = (size_t)w * h;
        const unsigned int stride = f->getStride();
        if (w <= 0 || h <= 0) return 0;   // degenerate frame — skip, don't kill the stream
        if ((stride != 0 && stride != (unsigned)w * 3) || rd.size() < npx * 3) {
            set_err("rgb frame is not tightly packed RGB888 (stride != w*3)"); return -1;
        }
        dev->cur_rgb = f;   // pin the buffer aliased out until the next poll
        *rgb = rd.data();
        *len = (int)(npx * 3);
        *width = w; *height = h;
        *ts_ns = frame_epoch_ns(f);
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_rgb"); return -1; }
}

extern "C" int oak_poll_depth(oak_device* dev, const uint16_t** depth_mm,
                              int* depth_w, int* depth_h, uint64_t* ts_ns) {
    if (!dev) { set_err("null device"); return -1; }
    if (!depth_mm || !depth_w || !depth_h || !ts_ns) { set_err("null out pointer"); return -1; }
    if (!dev->depth_q) return 0;
    try {
        auto d = dev->depth_q->tryGet<dai::ImgFrame>();
        if (!d) return 0;
        auto dd = d->getData();
        const int dw = (int)d->getWidth(), dh = (int)d->getHeight();
        if (dw <= 0 || dh <= 0) return 0;   // degenerate frame — skip, don't kill the stream
        const size_t row = (size_t)dw * sizeof(uint16_t);
        // A downscaled/aligned depth frame is often padded to a byte-alignment boundary
        // (stride > dw*2). Honor the stride by repacking row-by-row instead of dropping every such
        // frame — the old tight-only check silently left depth permanently empty when it was padded.
        unsigned int stride = d->getStride();
        if (stride == 0) stride = (unsigned)row;
        if (stride < row || dd.size() < (size_t)stride * dh) {
            return 0; // malformed — skip this frame rather than kill the stream
        }
        if (stride == row) {
            dev->cur_depth = d;   // tight → hand out zero-copy, pinned until the next poll
            *depth_mm = reinterpret_cast<const uint16_t*>(dd.data());
        } else {
            dev->cur_depth.reset();   // repacked into our own buffer; no need to pin the padded frame
            dev->depth_repack.resize((size_t)dw * dh);
            const uint8_t* base = dd.data();
            for (int y = 0; y < dh; ++y) {
                std::memcpy(dev->depth_repack.data() + (size_t)y * dw, base + (size_t)y * stride, row);
            }
            *depth_mm = dev->depth_repack.data();
        }
        *depth_w = dw; *depth_h = dh;
        *ts_ns = frame_epoch_ns(d);
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_depth"); return -1; }
}

extern "C" int oak_poll_video(oak_device* dev, const uint8_t** data, int* len, uint64_t* ts_ns) {
    if (!dev) { set_err("null device"); return -1; }
    if (!data || !len || !ts_ns) { set_err("null out pointer"); return -1; }
    if (!dev->video_q) return 0;
    try {
        auto frame = dev->video_q->tryGet<dai::ImgFrame>();
        if (!frame) return 0;
        dev->cur_video = frame;   // pin until the next poll
        auto d = frame->getData();
        *data = d.data();
        *len = (int)d.size();
        *ts_ns = frame_epoch_ns(frame);
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_video"); return -1; }
}

// Validate one eye and hand out its buffer zero-copy. GRAY8 is one byte per pixel, and the Rust
// side assumes TIGHT rows (stride == w), so verify that rather than trust it.
static bool eye_span(const std::shared_ptr<dai::ImgFrame>& f, int w, int h,
                     const uint8_t** out) {
    if (!f) return false;
    if ((int)f->getWidth() != w || (int)f->getHeight() != h) return false;
    auto d = f->getData();
    const unsigned int stride = f->getStride();
    if (stride != 0 && stride != (unsigned)w) return false;
    if (d.size() < (size_t)w * h) return false;
    *out = d.data();
    return true;
}

extern "C" int oak_poll_stereo(oak_device* dev,
                               const uint8_t** left, const uint8_t** right,
                               int* width, int* height, int* len, uint64_t* ts_ns,
                               void** l_hnd, void** r_hnd) {
    if (!dev) { set_err("null device"); return -1; }
    if (!left || !right || !width || !height || !len || !ts_ns || !l_hnd || !r_hnd) {
        set_err("null out pointer"); return -1;
    }
    try {
        bool timed_out = false;
        auto group = dev->stereo_q->get<dai::MessageGroup>(std::chrono::seconds(1), timed_out);
        if (timed_out || !group) return 0;

        auto l = group->get<dai::ImgFrame>("left");
        auto r = group->get<dai::ImgFrame>("right");
        if (!l || !r) { set_err("stereo group missing an eye"); return -1; }

        const int w = (int)l->getWidth(), h = (int)l->getHeight();
        if (w <= 0 || h <= 0) return 0;   // degenerate frame — skip, don't kill the stream
        if (!eye_span(l, w, h, left) || !eye_span(r, w, h, right)) {
            set_err("stereo eye is not tightly packed GRAY8 (stride != w) or eyes differ in size");
            return -1;
        }
        *width = w; *height = h;
        *len = w * h;
        *ts_ns = frame_epoch_ns(l);
        // Hand ownership of both eyes to the caller. Each handle is a heap-allocated copy of
        // the shared_ptr, so the pixel buffers live exactly as long as the caller keeps them —
        // no device-side "current frame" slot to re-address later, and no way to retain the
        // wrong frame.
        *l_hnd = new std::shared_ptr<dai::ImgFrame>(l);
        *r_hnd = new std::shared_ptr<dai::ImgFrame>(r);
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_stereo"); return -1; }
}

// Release a handle from oak_poll_stereo.
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
