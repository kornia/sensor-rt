// C++ implementation of the pure-C OAK bridge over depthai-core v3.
// One device (USB or PoE) → the two mono cameras (CAM_B/CAM_C) through a Sync node,
// exposing a time-synced GRAY8 stereo pair, plus the on-board IMU on its own queue.
// All C++ exceptions are caught and surfaced as return codes + oak_last_error().

#include "oak_bridge.h"
#include "depthai/depthai.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
static bool device_has_stereo(const std::shared_ptr<dai::Device>& d,
                              const dai::CalibrationHandler& calib, int w, int h) {
    try {
        bool has_b = false, has_c = false;
        for (auto s : d->getConnectedCameras()) {
            if (s == dai::CameraBoardSocket::CAM_B) has_b = true;
            else if (s == dai::CameraBoardSocket::CAM_C) has_c = true;
        }
        if (!has_b || !has_c) return false;
        auto k = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, w, h);
        return k[0][0] > 0.0f;   // fx > 0 → real calibration present
    } catch (const std::exception&) { return false; }
}

struct oak_device {
    std::shared_ptr<dai::Device> device;
    std::unique_ptr<dai::Pipeline> pipeline;
    // On-board IMU, SHARED by both modalities (optional in each):
    std::shared_ptr<dai::MessageQueue> imu_q;    // IMUData batches, far faster than the frame rate
    // IMU samples popped off the queue but not yet handed to the caller (see oak_poll_imu).
    std::vector<oak_imu_sample> imu_pending;
    bool has_imu = false;     // on-board IMU running (optional in both modalities)
    // IMU-chip → camera-optical rotation (row-major) applied to every sample in oak_poll_imu.
    // Identity + imu_aligned=false when the calibration carries no IMU extrinsics (samples then
    // stay in the raw chip frame, which is axis-permuted vs the camera on most boards).
    float imu_rot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool imu_aligned = false;
    // STEREO+IMU modality (oak_open_stereo):
    std::shared_ptr<dai::MessageQueue> stereo_q; // Sync'd {left,right} MessageGroup
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
    // Full CAM_B/CAM_C calibration, read once at oak_open_stereo (readCalibration() is an RPC, and
    // a host rectifier needs these numbers before the first frame). `valid` stays 0 on the RGBD
    // modality and on a wiped EEPROM.
    oak_stereo_calib stereo_calib{};
};

// Attach an NV12 output of `color` → a hardware H.264 encoder, handing the bitstream queue to `dev`.
// Shared by the video-only and decoupled RGBD paths so their encoder settings (BASELINE for
// Foxglove's decoder, ~4 keyframes/s for fast mid-stream join, OAK_H264_KBPS) can never drift apart.
static void add_h264_encoder(dai::Pipeline& pipeline, const std::shared_ptr<dai::node::Camera>& color,
                             int width, int height, int fps, oak_device* dev) {
    if (fps <= 0) fps = 30;
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

// Attach the on-board IMU (ACCELEROMETER_RAW + GYROSCOPE_RAW at imu_hz) on its own queue, shared by
// both open paths. IMU is OPTIONAL: not every OAK carries one, and a missing IMU must not cost the
// image streams — so BEFORE creating the node (and thus before it can ever reach pipeline.start()),
// preflight with getConnectedIMU(): a board without one reports "NONE"/empty, and we skip the node
// (has_imu stays false, streams run on). The node setup itself is host-side property setters, which
// cannot throw — only the preflight RPC is guarded. No-op when imu_hz <= 0.
static void add_imu_node(dai::Pipeline& pipeline, oak_device* dev, int imu_hz) {
    if (imu_hz <= 0) return;
    std::string imu_name;
    try {
        imu_name = dev->device->getConnectedIMU();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sensor-oak: getConnectedIMU failed (%s) — skipping the IMU node\n",
                     e.what());
        return;
    }
    if (imu_name.empty() || imu_name == "NONE") {
        std::fprintf(stderr, "sensor-oak: no on-board IMU (getConnectedIMU=\"%s\") — "
                             "skipping the IMU node\n", imu_name.c_str());
        return;
    }
    auto imu = pipeline.create<dai::node::IMU>();
    imu->enableIMUSensor(dai::IMUSensor::ACCELEROMETER_RAW, (uint32_t)imu_hz);
    imu->enableIMUSensor(dai::IMUSensor::GYROSCOPE_RAW, (uint32_t)imu_hz);
    // Batch a few reports per message (fewer, larger XLink transfers) but keep the batch
    // small enough that inertial data stays fresh relative to the frames. 5 is also the
    // documented maxBatchReports ceiling (IMUProperties.hpp).
    imu->setBatchReportThreshold(5);
    imu->setMaxBatchReports(5);
    dev->imu_q = imu->out.createOutputQueue(50, false);
    dev->has_imu = true;
}

// Resolve the IMU-chip → camera-optical rotation from the device calibration so oak_poll_imu can
// report samples in the camera frame (what gyro priors / gravity alignment consume). Raw depthai
// reports are in the IMU chip frame, axis-permuted vs the camera by the board mounting. Falls back
// to identity (raw chip frame, imu_aligned=false) when the EEPROM has no IMU link or the stored
// matrix is not a proper rotation (wiped/unfilled calibration) — degrade, never abort, but say WHY
// on stderr so a tilted gravity gauge is diagnosable without a debugger.
//
// The gate is a real rotation test, not just a det check: det≈1 alone admits shears (a k=100 shear
// has det=1 and would turn 9.81 m/s² into ~981) and ±3% scales, and every IEEE compare with NaN is
// false, so a NaN-laced matrix sailed through `fabs(det-1) > 0.1`. Requirements, in order:
// all 9 entries finite; det > 0 (proper, not a reflection); R·Rᵀ = I to 1e-3 (orthonormal, which
// also pins |det| to 1); and not the EXACT identity — depthai stores identity as the "no
// calibration" sentinel, and a real chip→camera mounting is never a perfect identity.
static void read_imu_rotation(oak_device* dev, const dai::CalibrationHandler& calib,
                              dai::CameraBoardSocket socket) {
    if (!dev->has_imu) return;
    try {
        auto m = calib.getImuToCameraExtrinsics(socket);
        if (m.size() < 3 || m[0].size() < 3 || m[1].size() < 3 || m[2].size() < 3) {
            std::fprintf(stderr, "sensor-oak: IMU extrinsics rejected (matrix smaller than 3x3) — "
                                 "IMU samples stay in the raw chip frame\n");
            return;
        }
        float r[9];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r[i * 3 + j] = m[i][j];
        for (int i = 0; i < 9; ++i) {
            if (!std::isfinite(r[i])) {
                std::fprintf(stderr, "sensor-oak: IMU extrinsics rejected (non-finite entry) — "
                                     "IMU samples stay in the raw chip frame\n");
                return;
            }
        }
        const float det = r[0] * (r[4] * r[8] - r[5] * r[7])
                        - r[1] * (r[3] * r[8] - r[5] * r[6])
                        + r[2] * (r[3] * r[7] - r[4] * r[6]);
        if (det <= 0.0f) {
            std::fprintf(stderr, "sensor-oak: IMU extrinsics rejected (det %.3f <= 0, reflection "
                                 "or degenerate) — IMU samples stay in the raw chip frame\n", det);
            return;
        }
        // Orthonormality: max |(R·Rᵀ − I)| entry. Bounds scale AND shear at once; with det > 0
        // above this admits only proper rotations (to the EEPROM's float precision).
        float ortho_err = 0.0f;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const float dot = r[i * 3] * r[j * 3] + r[i * 3 + 1] * r[j * 3 + 1]
                                + r[i * 3 + 2] * r[j * 3 + 2];
                ortho_err = std::max(ortho_err, std::fabs(dot - (i == j ? 1.0f : 0.0f)));
            }
        }
        if (ortho_err > 1e-3f) {
            std::fprintf(stderr, "sensor-oak: IMU extrinsics rejected (not orthonormal, "
                                 "max |R*R^T - I| = %.5f) — IMU samples stay in the raw chip "
                                 "frame\n", ortho_err);
            return;
        }
        bool exact_identity = true;
        for (int i = 0; i < 9; ++i) {
            if (r[i] != (i % 4 == 0 ? 1.0f : 0.0f)) { exact_identity = false; break; }
        }
        if (exact_identity) {
            std::fprintf(stderr, "sensor-oak: IMU extrinsics are the exact identity (depthai's "
                                 "not-calibrated sentinel) — IMU samples stay in the raw chip "
                                 "frame\n");
            return;
        }
        std::copy(r, r + 9, dev->imu_rot);
        dev->imu_aligned = true;
    } catch (const std::exception& e) {
        // No IMU extrinsics in the EEPROM (or the calibration read failed) — raw chip frame.
        std::fprintf(stderr, "sensor-oak: no IMU extrinsics in EEPROM (%s) — "
                             "IMU samples stay in the raw chip frame\n", e.what());
    }
}

// Read CAM_A factory intrinsics at the streamed size into `dev` (fx/fy/cx/cy). A wiped EEPROM or a
// device without CAM_A leaves them zero — fine for viewing, so the failure is swallowed.
static void read_rgb_intrinsics(oak_device* dev, const dai::CalibrationHandler& calib,
                                int width, int height) {
    try {
        auto k = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_A, width, height);
        dev->fx = k[0][0]; dev->fy = k[1][1]; dev->cx = k[0][2]; dev->cy = k[1][2];
    } catch (const std::exception&) { /* intrinsics stay zero */ }
}

// Read the FULL CAM_B/CAM_C calibration into `dev` for a host stereo rectifier: per-eye intrinsics
// at the streamed size, per-eye distortion, and the CALIBRATED left->right extrinsic in METRES.
//
// Both depthai defaults on the extrinsics getter are traps we must not take: useSpecTranslation
// defaults to true (board design numbers, not the measured calibration) and the length unit
// defaults to centimetres. Either one silently rescales the entire reconstruction, so both are
// passed explicitly. The baseline is derived from this same extrinsic rather than from
// getBaselineDistance() (which has the identical two defaults) so rotation and baseline can never
// come from different sources.
//
// A wiped/blank EEPROM leaves valid = 0 with everything zeroed; the caller decides whether that is
// fatal (it is, for stereo VIO), so this does not fail the open.
static void read_stereo_calib(oak_device* dev, const dai::CalibrationHandler& calib,
                              int width, int height) {
    const auto lsock = dai::CameraBoardSocket::CAM_B;
    const auto rsock = dai::CameraBoardSocket::CAM_C;
    oak_stereo_calib c{};
    c.width = width;
    c.height = height;
    try {
        auto kl = calib.getCameraIntrinsics(lsock, width, height);
        auto kr = calib.getCameraIntrinsics(rsock, width, height);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                c.left_k[i * 3 + j] = kl[i][j];
                c.right_k[i * 3 + j] = kr[i][j];
            }
        }

        auto dl = calib.getDistortionCoefficients(lsock);
        auto dr = calib.getDistortionCoefficients(rsock);
        c.left_n_dist = (int)std::min<size_t>(dl.size(), 14);
        c.right_n_dist = (int)std::min<size_t>(dr.size(), 14);
        for (int i = 0; i < c.left_n_dist; ++i) c.left_dist[i] = dl[i];
        for (int i = 0; i < c.right_n_dist; ++i) c.right_dist[i] = dr[i];
        c.left_model = (int)calib.getDistortionModel(lsock);
        c.right_model = (int)calib.getDistortionModel(rsock);

        auto e = calib.getCameraExtrinsics(lsock, rsock, /*useSpecTranslation=*/false,
                                           dai::LengthUnit::METER);
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) c.t_left_right[i * 4 + j] = e[i][j];
        const float tx = c.t_left_right[3], ty = c.t_left_right[7], tz = c.t_left_right[11];
        c.baseline_m = std::sqrt(tx * tx + ty * ty + tz * tz);

        c.valid = 1;
    } catch (const std::exception& ex) {
        // Zero the partial read rather than hand out half a calibration: a rectifier built from a
        // valid K and a zero extrinsic produces NaN maps, which is far harder to diagnose than a
        // flat "no calibration".
        c = oak_stereo_calib{};
        c.width = width;
        c.height = height;
        set_err(std::string("stereo calibration unavailable: ") + ex.what());
    }
    dev->stereo_calib = c;
}

extern "C" int oak_stereo_calibration(const oak_device* dev, oak_stereo_calib* out) {
    if (!dev || !out) { set_err("null device or out pointer"); return -1; }
    if (!dev->stereo_calib.valid) {
        set_err("no stereo calibration (not a stereo device, or a wiped/blank EEPROM)");
        return -1;
    }
    *out = dev->stereo_calib;
    return 0;
}

// STEREO+IMU modality: the two mono cameras as a Sync'd GRAY8 pair + the IMU on its own queue.
// Shares no nodes with oak_open's colour/depth pipeline — deliberately a separate entry point so the
// working RGBD and H.264 paths cannot regress.
extern "C" oak_device* oak_open_stereo(const char* device_id, int width, int height,
                                       int fps, int imu_hz, int enable_h264) {
    try {
        auto dev = std::make_unique<oak_device>();
        dev->device = connect_device(device_id);
        dev->pipeline = std::make_unique<dai::Pipeline>(dev->device);
        auto& pipeline = *dev->pipeline;

        // The stereo pair is the whole point of this modality — unlike depth (which oak_open silently
        // falls back from), a missing mono socket here has no meaningful degraded mode. Fail loudly.
        bool has_a = false, has_b = false, has_c = false;
        for (auto s : dev->device->getConnectedCameras()) {
            if (s == dai::CameraBoardSocket::CAM_A) has_a = true;
            else if (s == dai::CameraBoardSocket::CAM_B) has_b = true;
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
        //
        // enableUndistortion is deliberately FALSE here (it is true on the RGBD colour path).
        // depthai's Camera node cannot rectify: ImageManip builds its map with
        // cv::initUndistortRectifyMap(..., R = cv::Mat(), ...) — an identity rectifying rotation —
        // so the two eyes come out undistorted but NOT row-aligned, which is useless to a stereo
        // matcher. A stereo consumer therefore rectifies on the host from
        // oak_stereo_calibration(), and feeding it pixels depthai had already undistorted would
        // apply the distortion correction TWICE, silently (the output frame carries cleared
        // distortion coefficients, so nothing downstream can tell). Raw in, host-rectified out.
        auto* lo = left->requestOutput(size, dai::ImgFrame::Type::GRAY8,
                                       dai::ImgResizeMode::CROP, (float)fps,
                                       /*enableUndistortion=*/false);
        auto* ro = right->requestOutput(size, dai::ImgFrame::Type::GRAY8,
                                        dai::ImgResizeMode::CROP, (float)fps,
                                        /*enableUndistortion=*/false);

        // Sync node: emit {left,right} as ONE MessageGroup so the host never has to pair by timestamp.
        // The eyes are frame-locked by the shared stereo trigger, so the threshold only has to absorb
        // transport jitter — half a frame interval is generous.
        auto sync = pipeline.create<dai::node::Sync>();
        sync->setSyncThreshold(std::chrono::nanoseconds(
            fps > 0 ? (1000000000LL / fps) / 2 : 16000000LL));
        lo->link(sync->inputs["left"]);
        ro->link(sync->inputs["right"]);
        dev->stereo_q = sync->out.createOutputQueue(4, false);

        // One EEPROM read shared by the IMU-extrinsics gate and the intrinsics below —
        // readCalibration() is an RPC per call, and a wiped EEPROM comes back as an empty
        // handler (its getters then throw, handled at each use) rather than throwing here.
        auto calib = dev->device->readCalibration();

        // Optional on-device H.264 of the COLOUR camera (CAM_A), viz-only: the encoder runs on
        // the device and only the ~OAK_H264_KBPS bitstream crosses the link, so it costs the
        // stereo pair nothing on the host. Same degrade rule as the IMU: a board without CAM_A
        // skips the stream (has_video stays false), it never costs the stereo pair.
        if (enable_h264) {
            if (has_a) {
                auto color = pipeline.create<dai::node::Camera>();
                color->build(dai::CameraBoardSocket::CAM_A);
                add_h264_encoder(pipeline, color, width, height, fps, dev.get());
            } else {
                std::fprintf(stderr,
                             "sensor-oak: no CAM_A on this board — skipping the H.264 viz stream\n");
            }
        }

        // IMU is OPTIONAL: not every OAK carries one — add_imu_node preflights with
        // getConnectedIMU() and skips the node on an IMU-less board, so a missing IMU never
        // costs the stereo pair (and never reaches pipeline.start()).
        add_imu_node(pipeline, dev.get(), imu_hz);
        // Left (CAM_B) is the stereo reference frame, so IMU samples are rotated into ITS
        // optical frame when the EEPROM carries the extrinsics (imu_aligned) — the same gate
        // as the RGBD modality, just a different reference socket.
        read_imu_rotation(dev.get(), calib, dai::CameraBoardSocket::CAM_B);

        pipeline.start();

        // Left (CAM_B) is the reference frame of a stereo rig, so oak_intrinsics reports ITS
        // intrinsics in this modality — never CAM_A's, whose only role here is the optional
        // viz-only H.264 stream.
        try {
            auto k = calib.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, width, height);
            dev->fx = k[0][0]; dev->fy = k[1][1];
            dev->cx = k[0][2]; dev->cy = k[1][2];
        } catch (const std::exception&) { /* intrinsics stay zero — uncalibrated board */ }

        // Full stereo calibration for a HOST rectifier (the pair above is raw). Read from the same
        // handler, so it costs no extra RPC. Failure is non-fatal: a stereo consumer will refuse to
        // start on oak_stereo_calibration() == -1, but a plain "two raw eyes" consumer still works.
        read_stereo_calib(dev.get(), calib, width, height);

        return dev.release();
    } catch (const std::exception& e) { set_err(e.what()); return nullptr; }
    catch (...) { set_err("unknown error in oak_open_stereo"); return nullptr; }
}

extern "C" int oak_has_imu(const oak_device* dev) {
    return (dev && dev->has_imu) ? 1 : 0;
}

extern "C" int oak_imu_aligned(const oak_device* dev) {
    return (dev && dev->imu_aligned) ? 1 : 0;
}

// RGBD + H.264 modality: CAM_A colour (RGB888) + StereoDepth aligned to it (uint16 mm) + an on-device
// H.264 colour stream, all DECOUPLED onto their own queues. Shares no nodes with oak_open_stereo — a
// separate entry point so neither modality can regress the other.
extern "C" oak_device* oak_open_rgbd(const char* device_id, int width, int height, int fps,
                                     int enable_h264, int enable_depth, int video_only,
                                     int imu_hz) {
    try {
        auto dev = std::make_unique<oak_device>();
        dev->device = connect_device(device_id);
        dev->pipeline = std::make_unique<dai::Pipeline>(dev->device);
        auto& pipeline = *dev->pipeline;
        if (fps < 1) fps = 30;   // 0/negative fps would poison the encoder preset + requestOutput rate

        // One EEPROM read shared by the stereo check, the IMU-extrinsics gate, and the intrinsics —
        // readCalibration() is an RPC per call, and a wiped EEPROM comes back as an empty handler
        // (its getters then throw, handled at each use) rather than throwing here.
        auto calib = dev->device->readCalibration();

        // Auto-fall-back to video-only when depth was requested but the device can't actually produce it
        // (mono camera, or wiped/blank calibration → fx=0). Pulling raw RGB over XLink for a "synced RGBD"
        // pair whose depth is garbage just caps the H.264 stream for nothing — build the lean video-only
        // pipeline instead. Policy: always ship compressed video; add RGBD only when depth works.
        bool want_video_only = (video_only != 0) ||
                               (enable_depth != 0 && !device_has_stereo(dev->device, calib, width, height));

        // VIDEO-ONLY: just the H.264 encoder (CAM_A NV12 → encoder → queue). No RGB888/depth output, so
        // the device transmits ONLY the small H.264 bitstream (low-bandwidth viewing over USB2 / shared
        // gigabit). oak_poll_rgb/_depth yield nothing; the caller drains oak_poll_video.
        if (want_video_only) {
            auto color = pipeline.create<dai::node::Camera>();
            color->build(dai::CameraBoardSocket::CAM_A);
            add_h264_encoder(pipeline, color, width, height, fps, dev.get());
            add_imu_node(pipeline, dev.get(), imu_hz);
            read_imu_rotation(dev.get(), calib, dai::CameraBoardSocket::CAM_A);
            dev->has_depth = false;
            dev->has_sync = false;
            pipeline.start();
            read_rgb_intrinsics(dev.get(), calib, width, height);
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
        add_imu_node(pipeline, dev.get(), imu_hz);
        read_imu_rotation(dev.get(), calib, dai::CameraBoardSocket::CAM_A);

        pipeline.start();
        read_rgb_intrinsics(dev.get(), calib, width, height);   // factory intrinsics of the aligned RGB camera

        // IR dot projector: passive stereo starves on texture-poor / dim scenes (single-digit
        // valid-depth %). Default 0.8 intensity; OAK_IR=0 disables (e.g. multi-cam cross-talk),
        // boards without a projector just return false. Set after start() — needs a live device.
        {
            float ir = 0.8f;
            if (const char* s = std::getenv("OAK_IR")) { ir = std::max(0.0f, std::min(1.0f, (float)std::atof(s))); }
            if (ir > 0.0f) dev->device->setIrLaserDotProjectorIntensity(ir);
        }

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
                // IMUPacket's acceleroMeter/gyroscope are VALUE members, default-initialised to
                // zeros with timestamp {0,0} — a packet missing one report would otherwise emit
                // gyro=(0,0,0) "not rotating" (or a boot-epoch stamp) as if it were real data.
                // A zero/default timestamp on either report marks such a hole: skip the sample.
                if ((p.acceleroMeter.timestamp.sec == 0 && p.acceleroMeter.timestamp.nsec == 0) ||
                    (p.gyroscope.timestamp.sec == 0 && p.gyroscope.timestamp.nsec == 0)) {
                    continue;
                }
                // NOTE: do NOT gate on accuracy == UNRELIABLE here — firmware does not
                // populate the accuracy field for the *_RAW streams (measured: every raw
                // report arrives UNRELIABLE, so the gate silenced the stream entirely).
                // The zero-timestamp gate below is the effective default-report guard.
                oak_imu_sample s;
                s.ts_ns = steady_to_epoch_ns(p.acceleroMeter.getTimestamp(), offset);
                const float ax = p.acceleroMeter.x, ay = p.acceleroMeter.y, az = p.acceleroMeter.z;
                const float gx = p.gyroscope.x,     gy = p.gyroscope.y,     gz = p.gyroscope.z;
                if (dev->imu_aligned) {
                    // chip frame → camera optical frame (see read_imu_rotation)
                    const float* R = dev->imu_rot;
                    s.ax = R[0] * ax + R[1] * ay + R[2] * az;
                    s.ay = R[3] * ax + R[4] * ay + R[5] * az;
                    s.az = R[6] * ax + R[7] * ay + R[8] * az;
                    s.gx = R[0] * gx + R[1] * gy + R[2] * gz;
                    s.gy = R[3] * gx + R[4] * gy + R[5] * gz;
                    s.gz = R[6] * gx + R[7] * gy + R[8] * gz;
                } else {
                    s.ax = ax; s.ay = ay; s.az = az;
                    s.gx = gx; s.gy = gy; s.gz = gz;
                }
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
