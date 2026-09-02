//! OAK-D camera source: a time-synced **stereo pair + IMU**, or **RGBD + H.264**,
//! plus factory intrinsics. Pure Rust on top of the [`depthai`] safe wrapper
//! (kornia/depthai-rs).
//!
//! Two independent modalities behind one type: see [`stereo`] for the raw pair,
//! [`rgbd`] for colour/depth/video, and [`imu`] for the inertial stream available
//! alongside either — drained independently, because the IMU reports far faster
//! than frames.
//!
//! Everything OAK-specific that is a *decision* lives in Rust here: the pure
//! rules (`OAK_*` knobs, the steady→epoch clock shift, the IMU rotation gate,
//! depth sizing, stride repacks) in [`policy`] / [`rgbd`] with unit tests, and
//! the graph builders with their degrade rules in [`graph`]. The `depthai` crate
//! underneath is faithful and unopinionated.
//!
//! **Nothing here touches CUDA**: frames come out on the host and the consumer owns
//! any upload, so a process that only wants pixels builds no GPU stack.

use std::cell::Cell;
use std::collections::VecDeque;
use std::time::Duration;

use depthai::{
    Device, GateControl, ImgFrame, ImuData, ImuPacket, InputQueue, Message, MessageGroup,
    OutputQueue, Pipeline,
};

/// Boxed error, `Send + Sync` so a source can be moved between threads.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A depthai failure with what the driver was doing at the time.
#[derive(Debug, thiserror::Error)]
#[error("{what}: {source}")]
pub(crate) struct OakError {
    what: &'static str,
    #[source]
    source: depthai::DepthaiError,
}

/// `.ctx("what")` on a `depthai::Result` — the one way errors gain context here.
pub(crate) trait Ctx<T> {
    fn ctx(self, what: &'static str) -> Result<T, OakError>;
}

impl<T> Ctx<T> for depthai::Result<T> {
    fn ctx(self, what: &'static str) -> Result<T, OakError> {
        self.map_err(|source| OakError { what, source })
    }
}

/// Log a degrade decision (a capability the driver is continuing without, and why).
macro_rules! degrade {
    ($($t:tt)*) => {
        eprintln!("sensor-oak: {}", format_args!($($t)*))
    };
}

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

use calib::StereoCalibError;
use graph::{ImuAttach, Session};

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
    /// Control queue of the gate in front of the H.264 encoder (present iff `video_q`).
    video_gate: Option<InputQueue>,
    // On-board IMU, SHARED by both modalities (optional in each). The `has_*`
    // accessors derive from these queue options: a present queue IS the capability.
    imu_q: Option<OutputQueue<ImuData>>,
    pipeline: Pipeline,
    device: Device,
    width: u32,
    height: u32,
    seq: u64,
    intr: OakIntrinsics,
    /// Validated IMU-chip → camera-optical rotation (row-major) applied to every
    /// sample in `next_imu`; `None` = the calibration carries no usable IMU
    /// extrinsics and samples stay in the raw chip frame.
    imu_rot: Option<[f32; 9]>,
    /// IMU samples popped off the queue but not yet handed to the caller.
    imu_pending: VecDeque<ImuSample>,
    /// Reused per-batch staging buffer (no allocation in steady state).
    imu_packets: Vec<ImuPacket>,
    /// Packets dropped by the zero-timestamp gate — surfaced so a half-rate IMU is
    /// visible instead of reading as "firmware delivers 98 Hz".
    imu_ts_skipped: u64,
    /// Full CAM_B/CAM_C calibration, read once at `open_stereo`; surfaced by
    /// `stereo_calib()`.
    stereo_calib: Result<OakStereoCalib, StereoCalibError>,
    /// Per-queue "first poll failure logged" latch (a dying device would
    /// otherwise spam every drain).
    poll_warned: Cell<u8>,
}

/// Which queue a poll is for — its bit in `poll_warned`.
#[derive(Clone, Copy)]
pub(crate) enum Which {
    Stereo = 1,
    Rgb = 2,
    Depth = 4,
    Video = 8,
    Imu = 16,
}

impl Which {
    fn name(self) -> &'static str {
        match self {
            Which::Stereo => "stereo",
            Which::Rgb => "rgb",
            Which::Depth => "depth",
            Which::Video => "video",
            Which::Imu => "IMU",
        }
    }
}

/// The image queues an open path produced (all `None` where a modality does not
/// have that stream, or degraded).
#[derive(Default)]
pub(crate) struct Queues {
    pub(crate) stereo: Option<OutputQueue<MessageGroup>>,
    pub(crate) rgb: Option<OutputQueue<ImgFrame>>,
    pub(crate) depth: Option<OutputQueue<ImgFrame>>,
    pub(crate) video: Option<graph::H264>,
}

/// Everything an open path produced, beyond the session itself.
pub(crate) struct Built {
    pub(crate) queues: Queues,
    pub(crate) imu: ImuAttach,
    pub(crate) intr: OakIntrinsics,
    pub(crate) stereo_calib: Result<OakStereoCalib, StereoCalibError>,
}

impl OakSource {
    /// Assemble a source from an opened session (pipeline already started) and
    /// what its open path built. Capabilities are whatever queues that path managed
    /// to create — the IMU may be absent on a given board, and `has_sync`/`has_depth`
    /// depend on the device's calibration, none of which the caller can predict.
    ///
    /// `width`/`height` are the *requested* size and are kept only so callers can size
    /// buffers before the first frame; each frame reports its own actual dimensions.
    pub(crate) fn from_parts(session: Session, width: u32, height: u32, built: Built) -> Self {
        let (video_q, video_gate) = built
            .queues
            .video
            .map_or((None, None), |v| (Some(v.queue), Some(v.gate)));
        Self {
            stereo_q: built.queues.stereo,
            rgb_q: built.queues.rgb,
            depth_q: built.queues.depth,
            video_q,
            video_gate,
            imu_q: built.imu.queue,
            pipeline: session.pipeline,
            device: session.dev,
            width,
            height,
            seq: 0,
            intr: built.intr,
            imu_rot: built.imu.rot,
            imu_pending: VecDeque::new(),
            imu_packets: Vec::new(),
            imu_ts_skipped: 0,
            stereo_calib: built.stereo_calib,
            poll_warned: Cell::new(0),
        }
    }

    /// Pop from `q` (blocking up to `timeout`, or non-blocking when `None`).
    /// `Err(())` = a device/queue error (logged once per queue): the stream is
    /// unusable. `Ok(None)` = nothing queued / timed out.
    pub(crate) fn pop<M: Message>(
        &self,
        q: &OutputQueue<M>,
        which: Which,
        timeout: Option<Duration>,
    ) -> Result<Option<M>, ()> {
        let r = match timeout {
            Some(t) => q.get(t),
            None => q.try_get(),
        };
        r.map_err(|e| {
            let bit = which as u8;
            if self.poll_warned.get() & bit == 0 {
                self.poll_warned.set(self.poll_warned.get() | bit);
                degrade!("{} poll failed: {e}", which.name());
            }
        })
    }

    pub fn intrinsics(&self) -> OakIntrinsics {
        self.intr
    }

    /// Switch the on-device H.264 stream on or off at runtime. Off, the encoder
    /// idles and **no video bytes cross the link** (the colour camera keeps serving
    /// RGB/depth); on, encoding resumes at the next keyframe interval. Use it to
    /// spare a saturated PoE/USB2 link, or stream only while something is worth
    /// watching. `Err` when the source has no video stream
    /// ([`has_video`](Self::has_video)). Starts on unless `OAK_VIDEO_GATED=1`.
    pub fn set_video_streaming(&self, on: bool) -> Result<(), BoxError> {
        self.send_gate(GateControl::new(on, None, None))
    }

    /// Pass the next `frames` encoded frames (optionally throttled to `fps`), then
    /// switch the stream off again: the "record a clip on detection" pattern. A
    /// burst replaces any earlier burst still in progress.
    pub fn video_burst(&self, frames: u32, fps: Option<u32>) -> Result<(), BoxError> {
        self.send_gate(GateControl::open_for(frames, fps))
    }

    fn send_gate(&self, control: depthai::Result<GateControl>) -> Result<(), BoxError> {
        let gate = self
            .video_gate
            .as_ref()
            .ok_or("this source has no H.264 stream")?;
        gate.send(&control.ctx("video gate control")?)
            .ctx("video gate send")?;
        Ok(())
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
}

/// Reboot a PoE OAK that has wedged in bootloader state — the failure mode where a camera drops off and
/// no amount of in-process `open_*` retrying recovers it (the firmware needs a bootloader-triggered
/// reboot). `target` = the camera's IP/name or deviceId (`None` = first wedged device found). Returns
/// `Ok(true)` if a device was kicked (wait ~8s for it to reboot before re-opening), `Ok(false)` if there
/// was nothing to kick (target absent or healthy), `Err` on a driver error. Blocking — call from the
/// reconnect path, not the drain loop.
pub fn kick_wedged_oak(target: Option<&str>) -> Result<bool, BoxError> {
    let target = target.filter(|t| !t.is_empty()); // "" = any, like open_*
    let infos = Device::all_available().ctx("enumerate devices")?;
    // The wedge: a PoE device stuck in the bootloader. A healthy device enumerates
    // UNBOOTED / BOOTED / FLASH_BOOTED and opens normally — kicking it would be a
    // pointless reboot.
    let Some(info) = infos.iter().find(|i| {
        i.state == depthai::DeviceState::Bootloader
            && target.is_none_or(|t| i.name == t || i.device_id == t)
    }) else {
        return Ok(false);
    };
    // Open+drop the bootloader connection: construction connects to the wedged
    // firmware, and destruction reboots the device to an unbooted state (the manual
    // recovery, in-process).
    drop(depthai::DeviceBootloader::open(info).ctx("bootloader open")?);
    Ok(true)
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
