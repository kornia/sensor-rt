//! OAK-D **stereo + IMU** modality: the two mono cameras (CAM_B = left, CAM_C =
//! right) as a time-synced GRAY8 pair, plus the on-board IMU.
//!
//! This is the raw stereo + inertial source for VIO / stereo-feature work — the
//! counterpart to the colour/depth path in [`crate`], not a variant of it. It
//! builds a completely separate device pipeline: no colour camera (unless the
//! H.264 viz stream is requested), no `StereoDepth`.
//!
//! **The frames are host-only and this source takes no CUDA stream.** That is
//! deliberate, and the crux of the design: a consumer that wants the two eyes to
//! overlap on the GPU needs them on *different* CUDA streams, which the source
//! cannot know — so it stays a plain producer (per the crate's architecture) and
//! hands out host spans. See `examples/oakd_xfeat_stereo`, which uploads left and
//! right onto two streams to keep both XFeat backbones in flight at once.

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use depthai::node::Sync as SyncNode;
use depthai::{CameraBoardSocket, ImgFrame, ImgFrameType, ImgResizeMode, MessageGroup};
use kornia_image::{Image, ImageSize};
use sensor_types::FrameMeta;

use crate::graph::{self, Session};
use crate::policy::Knobs;
use crate::{calib, policy, row_pitch, BoxError, Ctx, OakSource, Queues, Q};

/// One time-synced stereo pair, borrowed from the source.
///
/// Both spans are GRAY8 (`w*h`, tightly packed) — these are monochrome sensors, so
/// one byte per pixel is the whole signal — and are **valid only until the next
/// [`OakSource::next_stereo`]**. That applies to the borrowed *slices*
/// ([`left`](Self::left) / [`right`](Self::right)) only: the [`Image`]s from
/// [`left_image`](Self::left_image) / [`right_image`](Self::right_image) carry the frame's own
/// refcounted handle and outlive it, so there is no need to copy pixels out defensively. The `'a`
/// lifetime ties this frame to the `&mut OakSource` borrow, so the borrow checker forbids
/// pulling the next pair while this one is still held.
pub struct OakStereoFrame<'a> {
    /// The two eyes, validated as tight GRAY8 of equal size.
    left: ImgFrame,
    right: ImgFrame,
    meta: FrameMeta,
    _src: std::marker::PhantomData<&'a mut OakSource>,
}

impl OakStereoFrame<'_> {
    /// Left eye (CAM_B), GRAY8 `w*h`. The stereo reference frame — the
    /// intrinsics from [`OakSource::intrinsics`] belong to this camera.
    pub fn left(&self) -> &[u8] {
        gray8(&self.left)
    }
    /// Right eye (CAM_C), GRAY8 `w*h`, same dimensions as [`left`](Self::left).
    pub fn right(&self) -> &[u8] {
        gray8(&self.right)
    }
    /// Per-eye width (both eyes share it).
    pub fn width(&self) -> u32 {
        self.left.width()
    }
    /// Per-eye height (both eyes share it).
    pub fn height(&self) -> u32 {
        self.left.height()
    }
    /// Sequence + capture pts (of the left eye).
    pub fn meta(&self) -> &FrameMeta {
        &self.meta
    }

    /// Left eye as a host kornia grayscale [`Image`] — **zero copy**.
    ///
    /// The image borrows depthai's pixel buffer directly and shares this frame's
    /// refcounted handle, so unlike [`left`](Self::left) the result is NOT tied to the
    /// frame's lifetime: it stays valid across later polls and can be moved or
    /// buffered freely. The buffer is released once the frame and every image made
    /// from it are dropped.
    ///
    /// Read-only — the underlying storage refuses mutable slice access.
    pub fn left_image(&self) -> Result<Image<u8, 1>, BoxError> {
        eye_image(&self.left)
    }

    /// Right eye as a host [`Image`] — see [`left_image`](Self::left_image).
    pub fn right_image(&self) -> Result<Image<u8, 1>, BoxError> {
        eye_image(&self.right)
    }
}

/// The tight GRAY8 span of a validated eye.
fn gray8(f: &ImgFrame) -> &[u8] {
    &f.data()[..(f.width() * f.height()) as usize]
}

fn eye_image(frame: &ImgFrame) -> Result<Image<u8, 1>, BoxError> {
    // The keepalive is a clone of the very frame the pixels come from (a refcount
    // bump, no device call), so the two can never disagree.
    let keepalive: Arc<dyn Any + Send + Sync> = Arc::new(frame.clone());
    let size = ImageSize {
        width: frame.width() as usize,
        height: frame.height() as usize,
    };
    // SAFETY: the frame was validated as tight GRAY8 of `w*h` bytes before this
    // frame was built, the pixels are host memory, and `keepalive` shares the
    // frame's handle so the memory outlives the image.
    unsafe { Image::from_borrowed_host_readonly(size, gray8(frame).as_ptr(), keepalive) }
        .map_err(|e| format!("borrowed Image<u8,1>: {e}").into())
}

/// Whether an eye is tight GRAY8 with a full buffer. `Ok(false)` = a degenerate
/// (zero-sized) frame to skip; `Err` = not tightly packed or too short.
fn tight_gray8(f: &ImgFrame) -> Result<bool, String> {
    let (w, h) = (f.width() as usize, f.height() as usize);
    if w == 0 || h == 0 {
        return Ok(false);
    }
    match row_pitch(w, h, f.stride() as usize, f.data().len()) {
        None => Err("frame buffer is shorter than width * height".into()),
        Some(pitch) if pitch != w => Err(format!(
            "frame is not tightly packed (stride {pitch} != width {w})"
        )),
        Some(_) => Ok(true),
    }
}

/// Unpack a Sync group into a validated pair. `Ok(None)` = a degenerate (empty)
/// frame to skip; `Err` = the stream is unusable.
fn take_pair(group: &MessageGroup) -> Result<Option<(ImgFrame, ImgFrame)>, String> {
    let eye = |name: &str| -> Result<ImgFrame, String> {
        group
            .get::<ImgFrame>(name)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("stereo group missing the {name} eye"))
    };
    let (l, r) = (eye("left")?, eye("right")?);
    if (l.width(), l.height()) != (r.width(), r.height()) {
        return Err("stereo eyes differ in size".into());
    }
    let tl = tight_gray8(&l).map_err(|e| format!("left {e}"))?;
    let tr = tight_gray8(&r).map_err(|e| format!("right {e}"))?;
    Ok((tl && tr).then_some((l, r)))
}

impl OakSource {
    /// Open the **stereo + IMU** modality: a Sync'd left/right GRAY8 pair at
    /// `fps`, plus the IMU at `imu_hz` (accelerometer + gyroscope). `device`
    /// selects the camera exactly as in [`OakSource::open_rgbd`].
    ///
    /// The pair is **raw**: neither undistorted nor rectified. depthai's Camera
    /// node can only undistort (its rectifying rotation is hard-wired to
    /// identity), which no stereo matcher can use, and pre-undistorted pixels
    /// would then be silently double-corrected by a host rectifier. Pair this
    /// with [`stereo_calib`](OakSource::stereo_calib) and rectify on the host.
    ///
    /// Takes **no CUDA stream** — see the module docs: the consumer owns the
    /// upload, because it alone knows which stream each eye belongs on.
    ///
    /// The IMU is optional (`imu_hz = 0` skips it; a board without one degrades,
    /// see [`has_imu`](OakSource::has_imu)); samples come out in the **left (CAM_B)
    /// optical frame** when the EEPROM allows it, see
    /// [`imu_aligned`](OakSource::imu_aligned). A device with no CAM_B/CAM_C pair
    /// is an error — unlike depth, there is no meaningful degraded mode for a
    /// *stereo* source.
    ///
    /// `h264` additionally runs the on-device encoder over the COLOUR camera (viz stream,
    /// drained with [`next_video`](OakSource::next_video)); a board without CAM_A degrades
    /// to stereo-only ([`has_video`](OakSource::has_video) stays `false`).
    pub fn open_stereo(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        imu_hz: u32,
        h264: bool,
    ) -> Result<Self, BoxError> {
        Self::open_stereo_inner(device, width, height, fps, imu_hz, h264)
            .map_err(|e| format!("open_stereo failed: {e}").into())
    }

    fn open_stereo_inner(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        imu_hz: u32,
        h264: bool,
    ) -> Result<Self, BoxError> {
        let fps = policy::fps_or_default(fps);
        let imu_hz = policy::clamp_imu_hz(imu_hz);
        let s = Session::connect(device, Knobs::from_env())?;
        // The stereo pair is the whole point of this modality — unlike depth (which
        // open_rgbd silently falls back from), a missing mono socket here has no
        // meaningful degraded mode. Fail loudly.
        if !s.has(CameraBoardSocket::CamB) || !s.has(CameraBoardSocket::CamC) {
            return Err("device has no stereo pair (CAM_B/CAM_C) — open_stereo needs both".into());
        }
        let queues = build_stereo(&s, width, height, fps, h264).ctx("build stereo graph")?;
        // Full stereo calibration for a HOST rectifier (the pair is raw). Failure is
        // non-fatal: a stereo consumer will refuse to start on `stereo_calib()`, but
        // a plain "two raw eyes" consumer still works.
        let stereo_calib = calib::read_stereo_calib(&s.calib, width, height);
        // Left (CAM_B) is the reference frame of a stereo rig: the IMU is rotated
        // into it and `intrinsics()` reports it — never CAM_A, whose only role here
        // is the optional viz-only H.264 stream.
        Ok(s.finish(
            width,
            height,
            queues,
            imu_hz,
            CameraBoardSocket::CamB,
            stereo_calib,
        )?)
    }

    /// Pull the next time-synced stereo pair. Both eyes are borrowed from the
    /// device (zero-copy, no host repack) and valid until the next call.
    ///
    /// Blocks until a pair arrives, absorbing transient empty polls (device
    /// warmup). `None` means
    /// the stream ended: a device error, or ~5 s with no pair.
    pub fn next_stereo(&mut self) -> Option<OakStereoFrame<'_>> {
        let q = self.queues.stereo.as_ref()?;
        // Five 1 s polls (~5 s with no pair) before treating the stream as ended; a
        // device error ends it at once.
        let mut pair = None;
        for _ in 0..5 {
            let Some(group) = q.pop(Some(Duration::from_secs(1))).ok()? else {
                continue;
            };
            match take_pair(&group) {
                Ok(Some(p)) => {
                    pair = Some(p);
                    break;
                }
                Ok(None) => continue, // degenerate frame — skip, don't kill the stream
                Err(e) => {
                    degrade!("stereo stream ended: {e}");
                    return None;
                }
            }
        }
        let (left, right) = pair?;

        self.seq += 1;
        let meta = FrameMeta {
            seq: self.seq,
            pts_ns: Some(policy::frame_epoch_ns(&left)),
            source_id: None,
        };
        Some(OakStereoFrame {
            left,
            right,
            meta,
            _src: std::marker::PhantomData,
        })
    }
}

/// The stereo+IMU graph: two mono cameras as a Sync'd GRAY8 pair, plus the
/// optional CAM_A H.264 viz stream.
fn build_stereo(
    s: &Session,
    width: u32,
    height: u32,
    fps: u32,
    h264: bool,
) -> depthai::Result<Queues> {
    // CAM_B/CAM_C are MONOCHROME sensors, so they are requested as GRAY8 — one byte
    // per pixel. Asking for RGB888i would make depthai replicate the same gray value
    // across three channels on-device and then ship 3x the bytes over XLink for no
    // information. Consumers that need 3 channels expand it on the GPU, where the
    // copy is free next to the inference.
    //
    // undistort is deliberately FALSE here (true on the RGBD colour path): the
    // pair is host-rectified from `stereo_calib()`, and pre-undistorted pixels
    // would be corrected TWICE — see `open_stereo`.
    let mono = |socket| -> depthai::Result<_> {
        s.camera(socket)?.request_output(
            (width, height),
            Some(ImgFrameType::Gray8),
            ImgResizeMode::Crop,
            Some(fps as f32),
            Some(false),
        )
    };
    let lo = mono(CameraBoardSocket::CamB)?;
    let ro = mono(CameraBoardSocket::CamC)?;

    // Sync node: emit {left,right} as ONE MessageGroup so the host never has to pair
    // by timestamp. The eyes are frame-locked by the shared stereo trigger, so the
    // threshold only has to absorb transport jitter — half a frame interval is
    // generous.
    let sync = s.pipeline.create::<SyncNode>()?;
    sync.set_sync_threshold(Duration::from_secs(1) / (2 * fps))?;
    lo.link(&sync.input("left")?)?;
    ro.link(&sync.input("right")?)?;
    let stereo = Q::new(sync.out()?.create_output_queue(4, false)?, "stereo");

    // Optional on-device H.264 of the COLOUR camera (CAM_A), viz-only: the encoder
    // runs on the device and only the ~OAK_H264_KBPS bitstream crosses the link, so
    // it costs the stereo pair nothing on the host. Same degrade rule as the IMU: a
    // board without CAM_A skips the stream.
    let video = h264
        .then(|| graph::try_add_h264_encoder(s, None, width, height, fps, "stereo"))
        .flatten();
    Ok(Queues {
        stereo: Some(stereo),
        video,
        ..Default::default()
    })
}
