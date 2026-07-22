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

extern "C" {

    /// Open the stereo (CAM_B/CAM_C) + IMU modality. See `oak_bridge.h`.
    pub fn oak_open_stereo(
        device_id: *const c_char,
        width: c_int,
        height: c_int,
        fps: c_int,
        imu_hz: c_int,
    ) -> *mut OakDevice;

    pub fn oak_has_imu(dev: *const OakDevice) -> c_int;

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
