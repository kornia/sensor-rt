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
//! depth sizing) in [`policy`] with unit tests, the calibration readers with their
//! unit/spec traps in [`calib`], and the graph builders with their degrade rules
//! in [`graph`]. The `depthai` crate underneath is faithful and unopinionated.
//!
//! **Nothing here touches CUDA**: frames come out on the host and the consumer owns
//! any upload, so a process that only wants pixels builds no GPU stack.

use std::cell::Cell;
use std::collections::VecDeque;
use std::time::Duration;

use depthai::{
    Device, GateControl, ImgFrame, ImuData, ImuPacket, Message, MessageGroup, OutputQueue, Pipeline,
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
use graph::H264;

/// One output queue with its "first poll failure logged" latch (a dying device
/// would otherwise spam every drain).
pub(crate) struct Q<M: Message> {
    q: OutputQueue<M>,
    name: &'static str,
    warned: Cell<bool>,
}

impl<M: Message> Q<M> {
    pub(crate) fn new(q: OutputQueue<M>, name: &'static str) -> Self {
        Q {
            q,
            name,
            warned: Cell::new(false),
        }
    }

    /// Pop (blocking up to `timeout`, or non-blocking when `None`). `Err(())` = a
    /// device/queue error (logged once): the stream is unusable. `Ok(None)` =
    /// nothing queued / timed out.
    pub(crate) fn pop(&self, timeout: Option<Duration>) -> Result<Option<M>, ()> {
        let r = match timeout {
            Some(t) => self.q.get(t),
            None => self.q.try_get(),
        };
        r.map_err(|e| {
            if !self.warned.replace(true) {
                degrade!("{} poll failed: {e}", self.name);
            }
        })
    }
}

/// Normalise a frame's row pitch: `stride == 0` is how depthai reports "tight".
/// `None` when the frame is degenerate (zero-sized), its rows would overlap, or
/// the buffer is too short (the last row need not be padded).
pub(crate) fn row_pitch(row: usize, h: usize, stride: usize, len: usize) -> Option<usize> {
    if row == 0 || h == 0 {
        return None;
    }
    let stride = if stride == 0 { row } else { stride };
    (stride >= row && len >= (h - 1) * stride + row).then_some(stride)
}

/// The image queues an open path produced (`None` where a modality does not have
/// that stream, or degraded). A present queue IS the capability the `has_*`
/// accessors report.
#[derive(Default)]
pub(crate) struct Queues {
    pub(crate) stereo: Option<Q<MessageGroup>>,
    pub(crate) rgb: Option<Q<ImgFrame>>,
    pub(crate) depth: Option<Q<ImgFrame>>,
    pub(crate) video: Option<H264>,
}

/// OAK-D source: [`open_stereo`](Self::open_stereo), then [`next_stereo`](Self::next_stereo)
/// in a loop, draining [`next_imu`](Self::next_imu) alongside it.
pub struct OakSource {
    /// STEREO+IMU modality: Sync'd {left,right}. RGBD+H.264 modality: colour, depth
    /// and video, each on its own queue, paired downstream by timestamp. The IMU is
    /// optional in both.
    queues: Queues,
    imu_q: Option<Q<ImuData>>,
    pipeline: Pipeline,
    device: Device,
    /// The *requested* size, so callers can size buffers before the first frame;
    /// each frame reports its own actual dimensions.
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
}

impl OakSource {
    pub fn intrinsics(&self) -> OakIntrinsics {
        self.intr
    }

    /// Switch the on-device H.264 stream on or off at runtime. Off, the encoder
    /// idles and **no video bytes cross the link**; the colour camera keeps serving
    /// RGB/depth. `Err` when the source has no video stream
    /// ([`has_video`](Self::has_video)). Starts on; see `OAK_VIDEO_GATED`.
    pub fn set_video_streaming(&self, on: bool) -> Result<(), BoxError> {
        let control = GateControl::new(on, None, None).ctx("video gate control")?;
        self.video_control()?
            .send(&control)
            .ctx("video gate send")?;
        Ok(())
    }

    /// Pass the next `frames` encoded frames (optionally throttled to `fps`), then
    /// switch the stream off again: the "record a clip on detection" pattern.
    pub fn video_burst(&self, frames: u32, fps: Option<u32>) -> Result<(), BoxError> {
        let control = GateControl::open_for(frames, fps).ctx("video gate control")?;
        self.video_control()?
            .send(&control)
            .ctx("video gate send")?;
        Ok(())
    }

    fn video_control(&self) -> Result<&depthai::InputQueue, BoxError> {
        Ok(&self
            .queues
            .video
            .as_ref()
            .ok_or("this source has no H.264 stream")?
            .control)
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
    let target = policy::device_id(target);
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
        // Stop the pipeline, then gracefully close the XLink connection while every
        // queue handle is still alive, so the firmware isn't torn down mid-stream
        // (avoids a spurious crash-dump on USB2 disconnect). Errors are irrelevant on
        // the way out; field drop order does not matter after this.
        let _ = self.pipeline.stop();
        let _ = self.device.close();
    }
}
