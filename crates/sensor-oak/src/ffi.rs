//! Raw FFI declarations for the depthai-core C shim (`oak_bridge.h`), which this
//! crate's `build.rs` compiles and links.
//!
//! Unsafe and pointer-based; the safe wrappers are [`OakSource`](crate::OakSource)
//! and friends, and nothing outside this crate touches these symbols. A pure-C ABI
//! over a C++ library — no C++ is ever visible to Rust (the `trt-sys` discipline).

use std::ffi::c_char;
use std::os::raw::c_int;

/// Opaque handle to an open OAK device + running pipeline.
#[repr(C)]
pub struct OakDevice {
    _private: [u8; 0],
}

/// One IMU reading: accelerometer (m/s²) + gyroscope (rad/s), stamped on the same
/// host-synced epoch timeline as the image frames. Mirrors `oak_imu_sample` in
/// `oak_bridge.h` — `#[repr(C)]`, and the field order must stay in lockstep with it.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct OakImuSample {
    pub ts_ns: u64,
    pub ax: f32,
    pub ay: f32,
    pub az: f32,
    pub gx: f32,
    pub gy: f32,
    pub gz: f32,
}

/// Full factory stereo calibration, cached at open. Mirrors `oak_stereo_calib` in `oak_bridge.h` —
/// `#[repr(C)]`, and the field order must stay in lockstep with it.
///
/// Row-major throughout; `t_left_right` is `X_right = T * X_left` with the translation in METRES,
/// taken from the *calibrated* (not board-spec) extrinsic.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OakStereoCalibRaw {
    pub width: c_int,
    pub height: c_int,
    pub left_k: [f32; 9],
    pub right_k: [f32; 9],
    pub left_dist: [f32; 14],
    pub right_dist: [f32; 14],
    pub left_n_dist: c_int,
    pub right_n_dist: c_int,
    pub left_model: c_int,
    pub right_model: c_int,
    pub t_left_right: [f32; 16],
    pub baseline_m: f32,
    pub valid: c_int,
}

impl Default for OakStereoCalibRaw {
    fn default() -> Self {
        // All-zero is exactly what the shim writes for "no calibration", so this doubles as a safe
        // out-param initialiser: a shim path that returns -1 without writing still leaves it
        // invalid rather than uninitialised.
        Self {
            width: 0,
            height: 0,
            left_k: [0.0; 9],
            right_k: [0.0; 9],
            left_dist: [0.0; 14],
            right_dist: [0.0; 14],
            left_n_dist: 0,
            right_n_dist: 0,
            left_model: 0,
            right_model: 0,
            t_left_right: [0.0; 16],
            baseline_m: 0.0,
            valid: 0,
        }
    }
}

extern "C" {

    /// Copy out the cached CAM_B/CAM_C calibration. 0 on success, -1 when unavailable.
    pub fn oak_stereo_calibration(dev: *const OakDevice, out: *mut OakStereoCalibRaw) -> c_int;

    /// Open the stereo (CAM_B/CAM_C) + IMU modality. See `oak_bridge.h`.
    pub fn oak_open_stereo(
        device_id: *const c_char,
        width: c_int,
        height: c_int,
        fps: c_int,
        imu_hz: c_int,
        enable_h264: c_int,
    ) -> *mut OakDevice;

    pub fn oak_has_imu(dev: *const OakDevice) -> c_int;

    /// 1 when IMU samples are calibration-rotated into the modality's reference camera optical
    /// frame (CAM_A on RGBD, CAM_B/left on stereo); 0 = raw IMU-chip frame. See `oak_bridge.h`.
    pub fn oak_imu_aligned(dev: *const OakDevice) -> c_int;

    /// Open the RGBD (CAM_A colour + aligned StereoDepth) + on-device H.264 modality. See `oak_bridge.h`.
    pub fn oak_open_rgbd(
        device_id: *const c_char,
        width: c_int,
        height: c_int,
        fps: c_int,
        enable_h264: c_int,
        enable_depth: c_int,
        video_only: c_int,
        imu_hz: c_int,
    ) -> *mut OakDevice;

    pub fn oak_has_depth(dev: *const OakDevice) -> c_int;
    pub fn oak_has_video(dev: *const OakDevice) -> c_int;
    pub fn oak_has_sync(dev: *const OakDevice) -> c_int;

    /// Next raw RGB888 colour frame (non-blocking); `rgb` aliases a buffer valid until the next call.
    pub fn oak_poll_rgb(
        dev: *mut OakDevice,
        rgb: *mut *const u8,
        width: *mut c_int,
        height: *mut c_int,
        len: *mut c_int,
        ts_ns: *mut u64,
    ) -> c_int;

    /// Next aligned uint16-mm depth frame (non-blocking); `depth_mm` aliases a buffer valid until the
    /// next call. Depth dims may be smaller than colour (downscaled on-device but aligned to it).
    pub fn oak_poll_depth(
        dev: *mut OakDevice,
        depth_mm: *mut *const u16,
        depth_w: *mut c_int,
        depth_h: *mut c_int,
        ts_ns: *mut u64,
    ) -> c_int;

    /// Next on-device H.264 access unit (non-blocking); `data` aliases a buffer valid until the next call.
    pub fn oak_poll_video(
        dev: *mut OakDevice,
        data: *mut *const u8,
        len: *mut c_int,
        ts_ns: *mut u64,
    ) -> c_int;

    pub fn oak_poll_stereo(
        dev: *mut OakDevice,
        left: *mut *const u8,
        right: *mut *const u8,
        width: *mut c_int,
        height: *mut c_int,
        len: *mut c_int,
        ts_ns: *mut u64,
        l_hnd: *mut *mut std::ffi::c_void,
        r_hnd: *mut *mut std::ffi::c_void,
    ) -> c_int;

    /// Release a retain handle from [`oak_poll_stereo`]. NULL is a no-op.
    pub fn oak_frame_release(handle: *mut std::ffi::c_void);

    pub fn oak_poll_imu(
        dev: *mut OakDevice,
        out: *mut OakImuSample,
        max: c_int,
        n: *mut c_int,
    ) -> c_int;

    pub fn oak_intrinsics(
        dev: *const OakDevice,
        fx: *mut f32,
        fy: *mut f32,
        cx: *mut f32,
        cy: *mut f32,
    ) -> c_int;

    /// Reboot a PoE OAK wedged in bootloader state so the next `oak_open` succeeds. `target` = IP/name
    /// or deviceId C string (NULL = first wedged device). 1 = kicked (wait ~8s), 0 = nothing to kick,
    /// -1 = error. Blocking.
    pub fn oak_kick(target: *const c_char) -> c_int;

    pub fn oak_close(dev: *mut OakDevice);

    pub fn oak_last_error() -> *const c_char;
}
