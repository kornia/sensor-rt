//! OAK-D camera source: a time-synced **stereo pair + IMU**, plus factory
//! intrinsics. Safe wrapper over the bundled depthai-core C shim (`oak_bridge.h`),
//! which this crate's `build.rs` compiles and links.
//!
//! Scope is deliberately narrow right now: the colour/depth (RGB-D) and H.264
//! paths were removed while the stereo + inertial modality is the one under
//! development. See [`stereo`] for the pair and [`imu`] for the inertial stream —
//! they are drained independently, because the IMU reports far faster than frames.
//!
//! **Nothing here touches CUDA**: frames come out on the host and the consumer owns
//! any upload, so a process that only wants pixels builds no GPU stack.

use std::ffi::{CStr, CString};

/// Boxed error, `Send + Sync` so a source can be moved between threads.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Pinhole intrinsics of an OAK camera, in pixels, at the streamed resolution.
///
/// Defined here rather than pulled from an inference crate: this driver is a plain
/// producer and must not drag a TensorRT runtime into anything that merely wants
/// camera frames. Consumers that need a richer camera model (unprojection,
/// distortion) convert these four numbers into their own type — it is the same
/// `fx, fy, cx, cy` everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct OakIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

// Raw FFI over the depthai-core C shim (`oak_bridge.h`), built by this crate's
// build.rs. Private: unsafe pointer soup is an implementation detail, and every
// caller goes through the safe wrappers below.
mod ffi;
mod imu;
mod stereo;
pub use imu::ImuSample;
pub use stereo::OakStereoFrame;

/// OAK-D source. `open` then `next_frame()` in a loop (like `RtspSource`).
pub struct OakSource {
    dev: *mut ffi::OakDevice,
    width: u32,
    height: u32,
    seq: u64,
    intr: OakIntrinsics,
    has_stereo: bool,
    has_imu: bool,
    /// Reused staging buffer for `next_imu`, so draining inertial samples every
    /// frame costs no allocation once it has grown.
    imu_scratch: Vec<ffi::OakImuSample>,
}

// SAFETY: the device handle is used single-threaded by the owning loop; the CUDA
// stream is Arc-backed. (depthai's own threads live behind the C shim.)
unsafe impl Send for OakSource {}

impl OakSource {
    /// Wrap a device the shim has already opened, reading every capability back
    /// from it rather than assuming what the constructor asked for.
    ///
    /// Shared by `open_inner` and [`open_stereo`](OakSource::open_stereo): those
    /// two build genuinely different pipelines (they share no depthai nodes), but
    /// the "what did we actually get?" half is identical, and hard-coding it per
    /// constructor is how the two drift. Every `oak_has_*` accessor is the source of
    /// truth — e.g. `has_sync` is false when a mono/uncalibrated device auto-fell-back
    /// to a video-only pipeline, which the caller cannot predict.
    pub(crate) fn from_open_device(
        dev: *mut ffi::OakDevice,
        width: u32,
        height: u32,
    ) -> Result<Self, BoxError> {
        let (mut fx, mut fy, mut cx, mut cy) = (0.0f32, 0.0, 0.0, 0.0);
        unsafe { ffi::oak_intrinsics(dev, &mut fx, &mut fy, &mut cx, &mut cy) };
        Ok(Self {
            dev,
            width,
            height,
            seq: 0,
            intr: OakIntrinsics { fx, fy, cx, cy },
            has_stereo: unsafe { ffi::oak_has_stereo(dev) } != 0,
            has_imu: unsafe { ffi::oak_has_imu(dev) } != 0,
            imu_scratch: Vec::new(),
        })
    }

    /// Turn a device id into a C string (NULL for "first available"), kept alive
    /// by the caller across the `oak_open*` call.
    pub(crate) fn device_id_cstring(device: Option<&str>) -> Result<Option<CString>, BoxError> {
        device
            .map(CString::new)
            .transpose()
            .map_err(|e| format!("bad device id: {e}").into())
    }

    pub fn intrinsics(&self) -> OakIntrinsics {
        self.intr
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Reboot a PoE OAK that has wedged in bootloader state — the failure mode where a camera drops off and
/// no amount of in-process `open_video*` retrying recovers it (the firmware needs a bootloader-triggered
/// reboot). `target` = the camera's IP/name or deviceId (`None` = first wedged device found). Returns
/// `Ok(true)` if a device was kicked (wait ~8s for it to reboot before re-opening), `Ok(false)` if there
/// was nothing to kick (target absent or healthy), `Err` on a driver error. Blocking — call from the
/// reconnect path, not the drain loop.
pub fn kick_wedged_oak(target: Option<&str>) -> Result<bool, BoxError> {
    let c = target
        .map(CString::new)
        .transpose()
        .map_err(|e| format!("bad kick target: {e}"))?;
    let ptr = c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let rc = unsafe { ffi::oak_kick(ptr) };
    match rc {
        1 => Ok(true),
        0 => Ok(false),
        _ => {
            let e = unsafe { CStr::from_ptr(ffi::oak_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("oak_kick failed: {e}").into())
        }
    }
}

impl Drop for OakSource {
    fn drop(&mut self) {
        unsafe { ffi::oak_close(self.dev) };
    }
}
