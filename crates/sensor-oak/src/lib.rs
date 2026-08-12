//! OAK-D camera source: a time-synced **stereo pair + IMU**, or **RGBD + H.264**,
//! plus factory intrinsics. Safe wrapper over the bundled depthai-core C shim
//! (`oak_bridge.h`), which this crate's `build.rs` compiles and links.
//!
//! Two independent modalities behind one type: see [`stereo`] for the raw pair,
//! [`rgbd`] for colour/depth/video, and [`imu`] for the inertial stream available
//! alongside either — drained independently, because the IMU reports far faster
//! than frames.
//!
//! **Nothing here touches CUDA**: frames come out on the host and the consumer owns
//! any upload, so a process that only wants pixels builds no GPU stack.

use std::ffi::CString;

/// Turn an optional device id into a C string (`None` → NULL, "first available").
/// The caller keeps it alive across the FFI call.
pub(crate) fn device_id_cstring(device: Option<&str>) -> Result<Option<CString>, BoxError> {
    device
        .map(CString::new)
        .transpose()
        .map_err(|e| format!("bad device id: {e}").into())
}

/// The shim's last error for the calling thread, as a `BoxError`.
pub(crate) fn last_error(what: &str) -> BoxError {
    let e = unsafe { std::ffi::CStr::from_ptr(ffi::oak_last_error()) }
        .to_string_lossy()
        .into_owned();
    format!("{what} failed: {e}").into()
}

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
mod rgbd;
mod stereo;
pub use imu::ImuSample;
pub use stereo::OakStereoFrame;

/// OAK-D source: [`open_stereo`](Self::open_stereo), then [`next_stereo`](Self::next_stereo)
/// in a loop, draining [`next_imu`](Self::next_imu) alongside it.
pub struct OakSource {
    dev: *mut ffi::OakDevice,
    width: u32,
    height: u32,
    seq: u64,
    intr: OakIntrinsics,
    has_imu: bool,
    /// IMU samples are calibration-rotated into the modality's reference camera optical frame
    /// (CAM_A on RGBD, CAM_B/left on stereo; false = raw IMU-chip frame — see
    /// [`next_imu`](Self::next_imu)).
    imu_aligned: bool,
    /// Reused staging buffer for `next_imu`, so draining inertial samples every
    /// frame costs no allocation once it has grown.
    imu_scratch: Vec<ffi::OakImuSample>,
    // RGBD modality (oak_open_rgbd) capabilities, read back from the device after open. All false for a
    // stereo source. `has_sync` is false when the device auto-fell-back to video-only (mono/uncalibrated).
    has_depth: bool,
    has_video: bool,
    has_sync: bool,
}

// SAFETY: the device handle is used single-threaded by the owning loop; the CUDA
// stream is Arc-backed. (depthai's own threads live behind the C shim.)
unsafe impl Send for OakSource {}

impl OakSource {
    /// Wrap a device the shim has already opened (either modality), reading its
    /// capabilities back from it rather than assuming what the constructor asked
    /// for — the IMU may be absent on a given board, and `has_sync`/`has_depth`
    /// depend on the device's calibration, none of which the caller can predict.
    /// The RGBD-only getters simply report `false` on a stereo device, so one
    /// constructor serves both open paths.
    ///
    /// `width`/`height` are the *requested* size and are kept only so callers can size
    /// buffers before the first frame; each frame reports its own actual dimensions.
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
            has_imu: unsafe { ffi::oak_has_imu(dev) } != 0,
            imu_aligned: unsafe { ffi::oak_imu_aligned(dev) } != 0,
            imu_scratch: Vec::new(),
            has_depth: unsafe { ffi::oak_has_depth(dev) } != 0,
            has_video: unsafe { ffi::oak_has_video(dev) } != 0,
            has_sync: unsafe { ffi::oak_has_sync(dev) } != 0,
        })
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
    let c = device_id_cstring(target)?;
    let ptr = c.as_ref().map_or(std::ptr::null(), |s| s.as_ptr());
    let rc = unsafe { ffi::oak_kick(ptr) };
    match rc {
        1 => Ok(true),
        0 => Ok(false),
        _ => Err(last_error("oak_kick")),
    }
}

impl Drop for OakSource {
    fn drop(&mut self) {
        unsafe { ffi::oak_close(self.dev) };
    }
}
