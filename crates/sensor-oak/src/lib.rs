//! OAK-D camera source: a time-synced **stereo pair + IMU**, or **RGBD + H.264**,
//! plus factory intrinsics. Pure Rust on top of the [`depthai`] safe wrapper
//! (kornia/depthai-rs) — no C++ shim in this crate.
//!
//! Two independent modalities behind one type: see [`stereo`] for the raw pair,
//! [`rgbd`] for colour/depth/video, and [`imu`] for the inertial stream available
//! alongside either — drained independently, because the IMU reports far faster
//! than frames.
//!
//! Everything OAK-specific that is a *decision* — env knobs, the steady→epoch
//! clock shift, the IMU rotation gate, the calibration unit traps, the degrade
//! rules — lives in [`policy`] and [`graph`], unit-tested, instead of inside a
//! C++ shim. The `depthai` crate underneath is faithful and unopinionated.
//!
//! **Nothing here touches CUDA**: frames come out on the host and the consumer owns
//! any upload, so a process that only wants pixels builds no GPU stack.

use std::collections::VecDeque;

use depthai::{Device, ImgFrame, ImuData, MessageGroup, OutputQueue, Pipeline};

/// Boxed error, `Send + Sync` so a source can be moved between threads.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Pinhole intrinsics of an OAK camera, in pixels, at the **requested** resolution.
///
/// Which camera depends on the modality — CAM_A on RGBD, CAM_B (left) on stereo — and the values
/// are always the RAW factory ones: distorted, and on the stereo path unrectified, so they are
/// *not* the intrinsics a rectified stereo consumer should use (see
/// [`OakStereoCalib`]). A wiped EEPROM yields all zeros without an error, so check `fx > 0`.
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

mod calib;
mod graph;
mod imu;
mod policy;
mod rgbd;
mod stereo;
pub use calib::{OakCameraCalib, OakCameraModel, OakStereoCalib};
pub use imu::ImuSample;
pub use stereo::OakStereoFrame;

/// OAK-D source: [`open_stereo`](Self::open_stereo), then [`next_stereo`](Self::next_stereo)
/// in a loop, draining [`next_imu`](Self::next_imu) alongside it.
///
/// Field order matters for drop order: queues first, then the pipeline (stopped
/// in [`Drop`]), then the device (closed there too).
pub struct OakSource {
    // STEREO+IMU modality (open_stereo): Sync'd {left,right} MessageGroup.
    stereo_q: Option<OutputQueue<MessageGroup>>,
    // RGBD+H.264 modality (open_rgbd): colour, depth, and video are DECOUPLED — each
    // on its own queue, pulled independently and paired downstream by timestamp.
    rgb_q: Option<OutputQueue<ImgFrame>>,
    depth_q: Option<OutputQueue<ImgFrame>>,
    video_q: Option<OutputQueue<ImgFrame>>,
    // On-board IMU, SHARED by both modalities (optional in each). The `has_*`
    // accessors derive from these queue options: a present queue IS the capability.
    imu_q: Option<OutputQueue<ImuData>>,
    pipeline: Pipeline,
    device: Device,
    width: u32,
    height: u32,
    seq: u64,
    intr: OakIntrinsics,
    /// IMU-chip → camera-optical rotation (row-major) applied to every sample in
    /// `next_imu`. Identity + `imu_aligned = false` when the calibration carries no
    /// usable IMU extrinsics (samples then stay in the raw chip frame).
    imu_rot: [f32; 9],
    imu_aligned: bool,
    /// IMU samples popped off the queue but not yet handed to the caller.
    imu_pending: VecDeque<ImuSample>,
    /// Packets dropped by the zero-timestamp gate — surfaced so a half-rate IMU is
    /// visible instead of reading as "firmware delivers 98 Hz".
    imu_ts_skipped: u64,
    /// Tightly-packed depth when the device row is stride-padded.
    depth_repack: Vec<u16>,
    /// Full CAM_B/CAM_C calibration, read once at `open_stereo`. `Err(why)` on the
    /// RGBD modality and on a wiped EEPROM; surfaced by `stereo_calib()`.
    stereo_calib: Result<OakStereoCalib, String>,
}

impl OakSource {
    /// Assemble a source from an opened device + started pipeline and its queues.
    /// Capabilities are whatever queues the open path managed to create — the IMU
    /// may be absent on a given board, and `has_sync`/`has_depth` depend on the
    /// device's calibration, none of which the caller can predict.
    ///
    /// `width`/`height` are the *requested* size and are kept only so callers can size
    /// buffers before the first frame; each frame reports its own actual dimensions.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble(
        device: Device,
        pipeline: Pipeline,
        width: u32,
        height: u32,
        intr: OakIntrinsics,
        queues: Queues,
        imu: (Option<OutputQueue<ImuData>>, [f32; 9], bool),
        stereo_calib: Result<OakStereoCalib, String>,
    ) -> Self {
        Self {
            stereo_q: queues.stereo,
            rgb_q: queues.rgb,
            depth_q: queues.depth,
            video_q: queues.video,
            imu_q: imu.0,
            pipeline,
            device,
            width,
            height,
            seq: 0,
            intr,
            imu_rot: imu.1,
            imu_aligned: imu.2,
            imu_pending: VecDeque::new(),
            imu_ts_skipped: 0,
            depth_repack: Vec::new(),
            stereo_calib,
        }
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

/// The image queues an open path produced (all `None` where a modality does not
/// have that stream, or degraded).
#[derive(Default)]
pub(crate) struct Queues {
    pub(crate) stereo: Option<OutputQueue<MessageGroup>>,
    pub(crate) rgb: Option<OutputQueue<ImgFrame>>,
    pub(crate) depth: Option<OutputQueue<ImgFrame>>,
    pub(crate) video: Option<OutputQueue<ImgFrame>>,
}

/// Reboot a PoE OAK that has wedged in bootloader state — the failure mode where a camera drops off and
/// no amount of in-process `open_*` retrying recovers it (the firmware needs a bootloader-triggered
/// reboot). `target` = the camera's IP/name or deviceId (`None` = first wedged device found). Returns
/// `Ok(true)` if a device was kicked (wait ~8s for it to reboot before re-opening), `Ok(false)` if there
/// was nothing to kick (target absent or healthy), `Err` on a driver error. Blocking — call from the
/// reconnect path, not the drain loop.
pub fn kick_wedged_oak(target: Option<&str>) -> Result<bool, BoxError> {
    let infos = Device::all_available().map_err(|e| format!("enumerate devices failed: {e}"))?;
    for info in &infos {
        if let Some(t) = target {
            if info.name != t && info.device_id != t {
                continue;
            }
        }
        // The wedge: a PoE device stuck in the bootloader. A healthy device enumerates
        // UNBOOTED / BOOTED / FLASH_BOOTED and opens normally — kicking it would be a
        // pointless reboot.
        if info.state != depthai::DeviceState::Bootloader {
            if target.is_some() {
                return Ok(false);
            }
            continue;
        }
        // Open+drop the bootloader connection: construction connects to the wedged
        // firmware, and destruction reboots the device to an unbooted state (the
        // manual recovery, in-process).
        let bl = depthai::DeviceBootloader::open(info)
            .map_err(|e| format!("bootloader open failed: {e}"))?;
        drop(bl);
        return Ok(true);
    }
    Ok(false)
}

impl Drop for OakSource {
    fn drop(&mut self) {
        // Stop the pipeline, then gracefully close the XLink connection before the
        // handles are released, so the firmware isn't torn down mid-stream (avoids a
        // spurious crash-dump on USB2 disconnect). Errors are irrelevant on the way out.
        let _ = self.pipeline.stop();
        let _ = self.device.close();
    }
}
