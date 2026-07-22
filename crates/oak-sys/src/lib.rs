//! Raw FFI declarations for the depthai-core C shim (`oak_bridge.h`).
//!
//! Unsafe and pointer-based — the safe wrapper lives in `vrt-oak`. Mirrors the
//! `trt-sys` pattern: a pure-C ABI over a C++ library, no C++ visible to Rust.

use std::ffi::c_char;
use std::os::raw::c_int;

/// Opaque handle to an open OAK device + running pipeline.
#[repr(C)]
pub struct oak_device {
    _private: [u8; 0],
}

extern "C" {
    pub fn oak_open(
        device_id: *const c_char,
        width: c_int,
        height: c_int,
        fps: c_int,
        enable_h264: c_int,
        enable_depth: c_int,
        video_only: c_int,
    ) -> *mut oak_device;

    pub fn oak_has_depth(dev: *const oak_device) -> c_int;

    pub fn oak_has_video(dev: *const oak_device) -> c_int;

    pub fn oak_has_sync(dev: *const oak_device) -> c_int;

    pub fn oak_poll_rgb(
        dev: *mut oak_device,
        rgb: *mut *const u8,
        width: *mut c_int,
        height: *mut c_int,
        rgb_len: *mut c_int,
        ts_ns: *mut u64,
    ) -> c_int;

    pub fn oak_poll_depth(
        dev: *mut oak_device,
        depth_mm: *mut *const u16,
        depth_w: *mut c_int,
        depth_h: *mut c_int,
        ts_ns: *mut u64,
    ) -> c_int;

    pub fn oak_intrinsics(
        dev: *const oak_device,
        fx: *mut f32,
        fy: *mut f32,
        cx: *mut f32,
        cy: *mut f32,
    ) -> c_int;

    pub fn oak_poll(
        dev: *mut oak_device,
        rgba: *mut *const u8,
        depth_mm: *mut *const u16,
        width: *mut c_int,
        height: *mut c_int,
        rgb_len: *mut c_int,
        ts_ns: *mut u64,
        depth_w: *mut c_int,
        depth_h: *mut c_int,
    ) -> c_int;

    pub fn oak_poll_video(
        dev: *mut oak_device,
        data: *mut *const u8,
        len: *mut c_int,
        ts_ns: *mut u64,
    ) -> c_int;

    /// Reboot a PoE OAK wedged in bootloader state so the next `oak_open` succeeds. `target` = IP/name
    /// or deviceId C string (NULL = first wedged device). 1 = kicked (wait ~8s), 0 = nothing to kick,
    /// -1 = error. Blocking.
    pub fn oak_kick(target: *const c_char) -> c_int;

    pub fn oak_close(dev: *mut oak_device);

    pub fn oak_last_error() -> *const c_char;
}
