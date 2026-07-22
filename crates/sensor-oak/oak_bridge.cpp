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

// H.264 target bitrate in kbps. The preset's default scales with resolution and is generous; 2000 kbps
// is comfortable for 640x360@30 and a big cut for a constrained network hop. Override with OAK_H264_KBPS.
static int h264_kbps() {
    if (const char* s = std::getenv("OAK_H264_KBPS")) {
        int v = std::atoi(s);
        if (v > 0) return v;
    }
    return 2000;
}

// True only if the device can actually produce aligned depth: it has BOTH stereo mono sockets
// (CAM_B + CAM_C) AND a readable factory calibration (a wiped/blank EEPROM reads back fx=0, which
// makes StereoDepth emit garbage/zero-scale depth). Used to auto-fall-back an uncalibrated or mono
// camera to the efficient video-only pipeline instead of the raw-RGB-pulling RGBD one.
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
    std::shared_ptr<dai::MessageQueue> rgb_q;    // raw RGB888, its OWN stream (for local compute), timestamped
    std::shared_ptr<dai::MessageQueue> depth_q;  // aligned depth, its OWN stream at the stereo rate, timestamped
    std::shared_ptr<dai::MessageQueue> video_q;  // H.264 bitstream (separate channel)
    std::shared_ptr<dai::MessageQueue> stereo_q; // Sync'd {left,right} MessageGroup (oak_open_stereo)
    std::shared_ptr<dai::MessageQueue> imu_q;    // IMUData batches, far faster than the frame rate
    // Keep the current frames alive so their buffers (handed out as raw spans) stay
    // valid until the next poll — zero-copy: no host repack.
    std::shared_ptr<dai::ImgFrame> cur_rgb;
    std::shared_ptr<dai::ImgFrame> cur_depth;
    std::shared_ptr<dai::ImgFrame> cur_video;
    // The stereo pair is pinned via the two eye frames (NOT the MessageGroup): holding the group
    // alone would not keep the ImgFrame buffers alive once it hands out shared_ptrs.
    std::shared_ptr<dai::ImgFrame> cur_left;
    std::shared_ptr<dai::ImgFrame> cur_right;
    // IMU samples popped off the queue but not yet handed to the caller (see oak_poll_imu).
    std::vector<oak_imu_sample> imu_pending;
    bool has_stereo = false;  // stereo-pair pipeline running (oak_open_stereo)
    bool has_imu = false;     // on-board IMU running (optional even in stereo mode)
    bool has_depth = false;
    bool has_video = false;   // on-device H.264 colour stream running
    bool has_sync = false;    // RGBD sync queue present (raw RGB+depth pulled over XLink); false = video-only
    int depth_w = 0, depth_h = 0; // depth output size (may be < RGB size: downscaled on-device before XLink)
    int width = 0, height = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
};

extern "C" oak_device* oak_open(const char* device_id, int width, int height, int fps,
                                int enable_h264, int enable_depth, int video_only) {
    try {
        auto dev = std::make_unique<oak_device>();
        dev->width = width;
        dev->height = height;

        dev->device = connect_device(device_id);
        dev->pipeline = std::make_unique<dai::Pipeline>(dev->device);
        auto& pipeline = *dev->pipeline;

        // Auto-fall-back to video-only when depth was requested but the device can't actually produce it
        // (mono camera, or wiped/blank calibration → fx=0). Pulling raw RGB over XLink for a "synced RGBD"
        // pair that has no usable depth just caps the H.264 stream for nothing — so build the lean
        // video-only pipeline instead. Policy: always ship compressed video; add RGBD only when depth works.
        bool want_video_only = (video_only != 0) ||
                               (enable_depth != 0 && !device_has_stereo(dev->device, width, height));

        // NO-STEREO path (flag still spelled `video_only`): H.264 encoder + a raw RGB888 output, and
        // no StereoDepth. Raw RGB is kept on purpose — these cameras feed ONBOARD COMPUTE, which needs
        // real frames, so oak_poll DOES yield frames here (RGB, no depth) and oak_poll_video runs
        // alongside. Costs w*h*3 per frame over XLink; that is an accepted trade, not an oversight.
        if (want_video_only) {
            auto color = pipeline.create<dai::node::Camera>();
            color->build(dai::CameraBoardSocket::CAM_A);
            // Raw RGB output (mirrors the full pipeline) so a NO-DEPTH camera still provides frames for
            // calibration snapshots + fusion colouring — not only the H.264 stream. The lean video-only
            // path omitted this purely to save XLink bandwidth (RGB888 is ~691KB/frame); it's worth it
            // for a camera you need to calibrate. Depth stays OFF (no stereo/calibration); oak_poll now
            // yields RGB and skips depth (has_depth=false, depth_q=null — already guarded there).
            auto* rgb_out = color->requestOutput(
                std::pair<uint32_t, uint32_t>((uint32_t)width, (uint32_t)height),
                dai::ImgFrame::Type::RGB888i, dai::ImgResizeMode::CROP, (float)fps,
                /*enableUndistortion=*/true);
            dev->rgb_q = rgb_out->createOutputQueue(4, false);
            dev->has_sync = true;
            auto* nv12_out = color->requestOutput(
                std::pair<uint32_t, uint32_t>((uint32_t)width, (uint32_t)height),
                dai::ImgFrame::Type::NV12, dai::ImgResizeMode::CROP, (float)fps, /*undistort=*/true);
            auto enc = pipeline.create<dai::node::VideoEncoder>();
            enc->setDefaultProfilePreset((float)fps, dai::VideoEncoderProperties::Profile::H264_BASELINE);
            enc->setKeyframeFrequency(fps > 0 ? std::max(fps / 4, 4) : 8); // ~4 keyframes/s → fast decoder start
            enc->setBitrateKbps(h264_kbps());
            nv12_out->link(enc->input);
            dev->video_q = enc->bitstream.createOutputQueue(30, false);
            dev->has_video = true;
            dev->has_depth = false;
            pipeline.start();
            try {
                auto calib = dev->device->readCalibration();
                auto k = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, width, height);
                dev->fx = k[0][0]; dev->fy = k[1][1]; dev->cx = k[0][2]; dev->cy = k[1][2];
            } catch (const std::exception&) { /* intrinsics stay zero — irrelevant for viewing */ }
            return dev.release();
        }

        // DECOUPLED build: depth and raw-RGB are SEPARATE streams (no on-device Sync node), each pulled at
        // its own rate + timestamped, so the consumer pairs them by timestamp (ApproximateTime). Depth is
        // no longer throttled to the rgb-pull rate — it runs at the mono/stereo rate (OAK_DEPTH_FPS, default
        // = capture fps). Raw RGB, needed only for local compute, is pulled at a LOW rate (OAK_RGB_FPS,
        // default 10) to spare XLink.
        int dfps = (fps > 0 ? fps : 30);
        if (const char* s = std::getenv("OAK_DEPTH_FPS")) { int v = std::atoi(s); if (v >= 1) dfps = v; }
        if (fps > 0) dfps = std::min(dfps, fps);
        // Raw-RGB rate (its own stream, for local compute). Low by default to spare XLink + ISP; the H.264
        // video is a separate 30fps output unaffected by this.
        int rfps = 10;
        if (const char* s = std::getenv("OAK_RGB_FPS")) { int v = std::atoi(s); if (v >= 1) rfps = v; }
        if (fps > 0) rfps = std::min(rfps, fps);

        // Colour (CAM_A). Interleaved RGB888 (the ISP's native type, handed to the host zero-copy) as the
        // raw-RGB stream + the depth-alignment reference. When H.264 is enabled we ALSO request an NV12
        // output (the encoder's input format) — a second ISP output, independent of the raw one.
        auto color = pipeline.create<dai::node::Camera>();
        color->build(dai::CameraBoardSocket::CAM_A);
        auto* rgb_out = color->requestOutput(
            std::pair<uint32_t, uint32_t>((uint32_t)width, (uint32_t)height),
            dai::ImgFrame::Type::RGB888i,
            dai::ImgResizeMode::CROP,
            (float)rfps,
            /*enableUndistortion=*/true);  // undistort RGB so depth aligns pixel-perfect
                                           // and the intrinsics are an exact pinhole

        // Stereo depth aligned to the RGB OUTPUT (not just the CAM_A socket), so the
        // depth map matches the RGB pixel grid exactly — same CROP, same size, same
        // intrinsics. The RVC2 (MyriadX) depth_align pattern: warp depth onto the
        // post-crop RGB frame, so depth[u,v] corresponds to RGB[u,v].
        std::shared_ptr<dai::node::StereoDepth> stereo;
        if (enable_depth) try {
            auto left = pipeline.create<dai::node::Camera>();
            left->build(dai::CameraBoardSocket::CAM_B);
            auto right = pipeline.create<dai::node::Camera>();
            right->build(dai::CameraBoardSocket::CAM_C);
            stereo = pipeline.create<dai::node::StereoDepth>();
            // ROBOTICS preset (depthai v3) is tuned for mobile-robot people/obstacle depth — it turns
            // on the post-processing the bare DEFAULT preset omits. Subpixel (3 fractional bits) gives
            // ~8× finer disparity, removing the depth quantization that makes a standing person's z
            // flicker frame-to-frame (which jitters the lift's z_ref). LR-check stays on for occlusion.
            stereo->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::ROBOTICS);
            stereo->setLeftRightCheck(true);
            // Subpixel gives ~8× finer disparity (removes z-quantization) but ~halves the stereo FPS
            // (RVC2: subpixel+LR ≈ 15fps@720p vs LR ≈ 60). OAK_SUBPIXEL=0 trades that precision for rate.
            bool subpixel = true;
            if (const char* s = std::getenv("OAK_SUBPIXEL")) {
                subpixel = !(std::string(s) == "0" || std::string(s) == "false");
            }
            stereo->setSubpixel(subpixel);
            // Passive-stereo depth cleanup (no IR projector on this OAK-D): SPATIAL edge-preserving
            // hole-fill + TEMPORAL averaging + THRESHOLD clamp to the useful range. Runs on the OAK's
            // stereo post-proc block.
            {
                auto& pp = stereo->initialConfig->postProcessing;
                pp.spatialFilter.enable = true;
                pp.temporalFilter.enable = true;
                pp.thresholdFilter.minRange = 400;   // mm — drop closer than 0.4 m
                pp.thresholdFilter.maxRange = 8000;  // mm — and farther than 8 m
            }
            // Mono pair at the capped depth rate (dfps, computed above) → bounds stereo → sync → the
            // raw-RGB pull frequency.
            left->requestOutput(std::pair<uint32_t, uint32_t>(640, 400), std::nullopt,
                                dai::ImgResizeMode::CROP, (float)dfps)->link(stereo->left);
            right->requestOutput(std::pair<uint32_t, uint32_t>(640, 400), std::nullopt,
                                 dai::ImgResizeMode::CROP, (float)dfps)->link(stereo->right);
            rgb_out->link(stereo->inputAlignTo);   // align depth to the RGB output grid
            // Downscale the aligned depth ON-DEVICE before it crosses XLink. A room-scale point cloud
            // doesn't need per-RGB-pixel depth, and the full-res depth pull is the dominant XLink cost
            // (it caps the co-hosted H.264 stream on a PoE link). Default /2 → 1/4 the depth bytes; the
            // map stays aligned to the RGB grid, so consumers just scale coords by (rgb_w / depth_w).
            int ddiv = 2;
            if (const char* s = std::getenv("OAK_DEPTH_DIV")) { int v = std::atoi(s); if (v >= 1) ddiv = v; }
            dev->depth_w = std::max(1, width / ddiv);
            dev->depth_h = std::max(1, height / ddiv);
            stereo->setOutputSize(dev->depth_w, dev->depth_h);
            dev->has_depth = true;
        } catch (const std::exception&) {
            dev->has_depth = false;   // e.g. OAK-1: no stereo pair
        }

        // DECOUPLED — NO Sync node. Raw RGB and depth each go to their OWN non-blocking output queue and
        // are pulled + published independently, each carrying its host-synced capture timestamp; the
        // consumer pairs them by timestamp (ApproximateTime). This frees depth from the rgb-pull rate: the
        // on-device Sync node emitted a bundled pair at the slower rate and its transport was capped by the
        // 691KB rgb, throttling depth to ~10fps. Depth stays aligned to the RGB grid (constant geometry via
        // inputAlignTo) but now flows at the full stereo rate.
        dev->rgb_q = rgb_out->createOutputQueue(4, false);
        if (dev->has_depth) {
            dev->depth_q = stereo->depth.createOutputQueue(4, false);
        }
        dev->has_sync = true; // "has an RGBD source to poll" (decoupled rgb + depth queues)

        // Optional standalone H.264 colour stream: a SECOND camera output (NV12, the
        // encoder's input format) → hardware H.264 encoder → its own output queue.
        // A periodic keyframe (1/sec) lets a viewer/recorder start mid-stream.
        if (enable_h264 != 0) {
            auto* nv12_out = color->requestOutput(
                std::pair<uint32_t, uint32_t>((uint32_t)width, (uint32_t)height),
                dai::ImgFrame::Type::NV12,
                dai::ImgResizeMode::CROP,
                (float)fps,
                /*enableUndistortion=*/true);
            auto enc = pipeline.create<dai::node::VideoEncoder>();
            // BASELINE profile → no B-frames, which Foxglove's CompressedVideo decoder requires
            // (it has no lookahead). The OAK emits Annex-B NAL units; a keyframe each second lets a
            // viewer/recorder join mid-stream.
            enc->setDefaultProfilePreset((float)fps, dai::VideoEncoderProperties::Profile::H264_BASELINE);
            enc->setKeyframeFrequency(fps > 0 ? std::max(fps / 4, 4) : 8); // ~4 keyframes/s → fast decoder start
            enc->setBitrateKbps(h264_kbps());
            nv12_out->link(enc->input);
            // Larger, non-blocking queue: the caller drains it each iteration. H.264 is
            // a dependent stream (P-frames need predecessors), so we keep depth rather
            // than letting a stall corrupt it — but never block the device pipeline.
            dev->video_q = enc->bitstream.createOutputQueue(30, false);
            dev->has_video = true;
        }

        pipeline.start();

        // Factory intrinsics of the (aligned) RGB camera at the streamed size.
        try {
            auto calib = dev->device->readCalibration();
            auto k = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, width, height);
            dev->fx = k[0][0]; dev->fy = k[1][1];
            dev->cx = k[0][2]; dev->cy = k[1][2];
        } catch (const std::exception&) { /* intrinsics stay zero */ }

        return dev.release();
    } catch (const std::exception& e) { set_err(e.what()); return nullptr; }
    catch (...) { set_err("unknown error in oak_open"); return nullptr; }
}

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

extern "C" int oak_has_depth(const oak_device* dev) {
    return (dev && dev->has_depth) ? 1 : 0;
}

extern "C" int oak_has_video(const oak_device* dev) {
    return (dev && dev->has_video) ? 1 : 0;
}

// 1 if the device runs the synced RGBD pipeline (oak_poll yields frames); 0 if it auto-fell-back to
// video-only (mono / uncalibrated) so the caller advertises only the H.264 stream and never polls sync.
extern "C" int oak_has_sync(const oak_device* dev) {
    return (dev && dev->has_sync) ? 1 : 0;
}

extern "C" int oak_intrinsics(const oak_device* dev, float* fx, float* fy, float* cx, float* cy) {
    if (!dev) { set_err("null device"); return -1; }
    if (!fx || !fy || !cx || !cy) { set_err("null out pointer"); return -1; }
    *fx = dev->fx; *fy = dev->fy; *cx = dev->cx; *cy = dev->cy;
    return 0;
}

extern "C" int oak_poll(oak_device* dev,
                        const uint8_t** rgb, const uint16_t** depth_mm,
                        int* width, int* height, int* rgb_len, uint64_t* ts_ns,
                        int* depth_w, int* depth_h) {
    if (!dev) { set_err("null device"); return -1; }
    if (!rgb || !depth_mm || !width || !height || !rgb_len || !ts_ns || !depth_w || !depth_h) {
        set_err("null out pointer"); return -1;
    }
    if (!dev->rgb_q) return 0;   // video-only build: no RGB(+depth) pipeline to poll
    try {
        // Synced-pair emulation over the DECOUPLED queues (the on-device Sync node is gone): block briefly
        // for the next RGB, then take the freshest depth already queued (non-blocking). Depth is aligned to
        // the RGB grid on-device (inputAlignTo) so newest-of-each is a valid pairing for the detection path;
        // callers needing exact host pairing use oak_poll_rgb/oak_poll_depth with the per-frame timestamps.
        bool timed_out = false;
        auto rgb_frame = dev->rgb_q->get<dai::ImgFrame>(std::chrono::seconds(1), timed_out);
        if (timed_out || !rgb_frame) return 0;

        // Keep the frame alive and hand out its buffer directly (zero host repack).
        // The raw RGB is always RGB888 — Rust assumes TIGHT rows (stride == w*3); verify stride + size.
        dev->cur_rgb = rgb_frame;
        auto rd = rgb_frame->getData();
        const int w = (int)rgb_frame->getWidth();
        const int h = (int)rgb_frame->getHeight();
        const size_t npx = (size_t)w * h;
        const unsigned int rgb_stride = rgb_frame->getStride();
        if ((rgb_stride != 0 && rgb_stride != (unsigned)w * 3) || rd.size() < npx * 3) {
            set_err("rgb frame is not tightly packed (stride != w*3)"); return -1;
        }
        *rgb = rd.data();
        *rgb_len = (int)(npx * 3);
        *width = w; *height = h;
        // Explicit ns conversion — don't assume the device clock's period is nano
        // (the Rust side divides ts by 1e9 for the Kalman dt).
        *ts_ns = frame_epoch_ns(rgb_frame); // host-synced epoch ns (aligned across all cameras)

        // Zero-copy depth: hand out the freshest aligned depth buffer directly (uint16 mm). The depth may
        // be a SMALLER grid than the RGB (downscaled on-device, see setOutputSize) but stays aligned to it —
        // so we return the depth's OWN dims and let the Rust/browser side scale coords by rgb_w/depth_w.
        dev->cur_depth.reset();   // don't pin last frame's depth on a dropout
        *depth_mm = nullptr;
        *depth_w = 0; *depth_h = 0;
        if (dev->has_depth && dev->depth_q) {
            if (auto d = dev->depth_q->tryGet<dai::ImgFrame>()) {
                auto dd = d->getData();
                const int dw = (int)d->getWidth(), dh = (int)d->getHeight();
                const unsigned int d_stride = d->getStride();
                const bool tight = (d_stride == 0 || d_stride == (unsigned)dw * sizeof(uint16_t));
                if (dw > 0 && dh > 0 && tight && dd.size() >= (size_t)dw * dh * sizeof(uint16_t)) {
                    dev->cur_depth = d;
                    *depth_mm = reinterpret_cast<const uint16_t*>(dd.data());
                    *depth_w = dw; *depth_h = dh;
                }
            }
        }
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll"); return -1; }
}

extern "C" int oak_poll_video(oak_device* dev,
                              const uint8_t** data, int* len, uint64_t* ts_ns) {
    if (!dev) { set_err("null device"); return -1; }
    if (!data || !len || !ts_ns) { set_err("null out pointer"); return -1; }
    if (!dev->has_video || !dev->video_q) return 0;
    try {
        // Non-blocking: return the next encoded frame if one is queued, else 0.
        auto frame = dev->video_q->tryGet<dai::ImgFrame>();
        if (!frame) return 0;
        dev->cur_video = frame;   // keep alive until the next call (buffer aliased out)
        auto d = frame->getData();
        *data = d.data();
        *len = (int)d.size();
        *ts_ns = frame_epoch_ns(frame); // host-synced epoch ns (same timeline as the synced RGBD)
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_video"); return -1; }
}

// Decoupled RGB poll: the next raw RGB888 frame from its own queue (non-blocking), aliased zero-copy +
// timestamped. 1 = frame, 0 = none ready, -1 = error. Drain in a loop until 0.
extern "C" int oak_poll_rgb(oak_device* dev,
                            const uint8_t** rgb, int* width, int* height, int* rgb_len, uint64_t* ts_ns) {
    if (!dev) { set_err("null device"); return -1; }
    if (!rgb || !width || !height || !rgb_len || !ts_ns) { set_err("null out pointer"); return -1; }
    if (!dev->rgb_q) return 0;
    try {
        auto f = dev->rgb_q->tryGet<dai::ImgFrame>();
        if (!f) return 0;
        dev->cur_rgb = f; // keep alive until the next call (buffer aliased out)
        auto rd = f->getData();
        const int w = (int)f->getWidth(), h = (int)f->getHeight();
        const size_t npx = (size_t)w * h;
        const unsigned int stride = f->getStride();
        if ((stride != 0 && stride != (unsigned)w * 3) || rd.size() < npx * 3) {
            set_err("rgb frame not tightly packed"); return -1;
        }
        *rgb = rd.data();
        *rgb_len = (int)(npx * 3);
        *width = w; *height = h;
        *ts_ns = frame_epoch_ns(f);
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_rgb"); return -1; }
}

// Decoupled depth poll: the next aligned uint16-mm depth frame from its own queue (non-blocking), aliased
// zero-copy + timestamped, with its own dims (may be < RGB size). 1 / 0 / -1. Drain in a loop until 0.
extern "C" int oak_poll_depth(oak_device* dev,
                              const uint16_t** depth_mm, int* depth_w, int* depth_h, uint64_t* ts_ns) {
    if (!dev) { set_err("null device"); return -1; }
    if (!depth_mm || !depth_w || !depth_h || !ts_ns) { set_err("null out pointer"); return -1; }
    if (!dev->depth_q) return 0;
    try {
        auto d = dev->depth_q->tryGet<dai::ImgFrame>();
        if (!d) return 0;
        auto dd = d->getData();
        const int dw = (int)d->getWidth(), dh = (int)d->getHeight();
        const unsigned int stride = d->getStride();
        const bool tight = (stride == 0 || stride == (unsigned)dw * sizeof(uint16_t));
        if (dw <= 0 || dh <= 0 || !tight || dd.size() < (size_t)dw * dh * sizeof(uint16_t)) {
            return 0; // skip a malformed frame rather than kill the stream
        }
        dev->cur_depth = d;
        *depth_mm = reinterpret_cast<const uint16_t*>(dd.data());
        *depth_w = dw; *depth_h = dh;
        *ts_ns = frame_epoch_ns(d);
        return 1;
    } catch (const std::exception& e) { set_err(e.what()); return -1; }
    catch (...) { set_err("unknown error in oak_poll_depth"); return -1; }
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
