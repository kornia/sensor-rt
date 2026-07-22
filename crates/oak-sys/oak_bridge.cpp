// C++ implementation of the pure-C OAK bridge over depthai-core v3.
// One device (USB or PoE) → Camera (CAM_A, RGB) + StereoDepth (CAM_B/CAM_C, aligned
// to CAM_A) + Sync, exposing a time-synced RGB888 + uint16-mm depth pair, PLUS an
// optional standalone on-device H.264 colour stream (a second camera output → encoder).
// All C++ exceptions are caught and surfaced as return codes + oak_last_error().

#include "oak_bridge.h"
#include "depthai/depthai.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static thread_local std::string g_err;
static void set_err(const std::string& m) { g_err = m; }

// Capture timestamp of a frame, in EPOCH nanoseconds (host wall clock). depthai's getTimestamp() is on
// the host STEADY clock and is synchronized across ALL connected devices — so multiple cameras share
// one timeline (essential for multi-camera alignment). getTimestampDevice() is per-device boot time,
// which is neither wall-clock nor comparable across cameras. We shift the steady timestamp onto the
// system clock once per frame so the value forwarded to ROS/Hiroz (and on to Foxglove/recordings) is a
// real epoch time that all cameras agree on.
static uint64_t frame_epoch_ns(const std::shared_ptr<dai::ImgFrame>& f) {
    using namespace std::chrono;
    auto steady_ns = duration_cast<nanoseconds>(f->getTimestamp().time_since_epoch());
    auto offset_ns = duration_cast<nanoseconds>(system_clock::now().time_since_epoch())
                   - duration_cast<nanoseconds>(steady_clock::now().time_since_epoch());
    return (uint64_t)(steady_ns + offset_ns).count();
}

// H.264 target bitrate in kbps. The preset's default scales with resolution and is generous; 2000 kbps
// is comfortable for 640x360@30 and a big cut for the phone/Tailscale hop. Override with OAK_H264_KBPS.
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
    // Keep the current frames alive so their buffers (handed out as raw spans) stay
    // valid until the next poll — zero-copy: no host repack.
    std::shared_ptr<dai::ImgFrame> cur_rgb;
    std::shared_ptr<dai::ImgFrame> cur_depth;
    std::shared_ptr<dai::ImgFrame> cur_video;
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

        // Cap the USB link speed. The default (SUPER/USB3) boots the device into a
        // USB3 descriptor; on a physical USB2 link the host then can't reconnect to
        // the booted device → X_LINK_DEVICE_NOT_FOUND. Defaulting to HIGH (USB2)
        // gives a stable link. Override with OAK_USB_SPEED=super on a USB3 cable.
        // (Ignored for a PoE device, which connects over TCP/IP.)
        dai::UsbSpeed usb_speed = dai::UsbSpeed::HIGH;
        if (const char* s = std::getenv("OAK_USB_SPEED")) {
            if (std::string(s) == "super" || std::string(s) == "SUPER") usb_speed = dai::UsbSpeed::SUPER;
        }

        // Device selection: a non-empty device_id picks a SPECIFIC camera by MxId
        // (USB or PoE) or IP (PoE) — depthai's string ctor resolves the transport.
        // Empty/NULL → first available device.
        if (device_id != nullptr && device_id[0] != '\0') {
            dev->device = std::make_shared<dai::Device>(std::string(device_id), usb_speed);
        } else {
            dev->device = std::make_shared<dai::Device>(usb_speed);
        }
        dev->pipeline = std::make_unique<dai::Pipeline>(dev->device);
        auto& pipeline = *dev->pipeline;

        // Auto-fall-back to video-only when depth was requested but the device can't actually produce it
        // (mono camera, or wiped/blank calibration → fx=0). Pulling raw RGB over XLink for a "synced RGBD"
        // pair that has no usable depth just caps the H.264 stream for nothing — so build the lean
        // video-only pipeline instead. Policy: always ship compressed video; add RGBD only when depth works.
        bool want_video_only = (video_only != 0) ||
                               (enable_depth != 0 && !device_has_stereo(dev->device, width, height));

        // VIDEO-ONLY: build just the H.264 encoder (CAM_A NV12 → encoder → queue). No RGB888 sync
        // output, no stereo — so the device transmits ONLY the small H.264 bitstream (low-bandwidth
        // viewing over USB2 / shared gigabit). oak_poll yields nothing; the caller drains oak_poll_video.
        if (want_video_only) {
            auto color = pipeline.create<dai::node::Camera>();
            color->build(dai::CameraBoardSocket::CAM_A);
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
