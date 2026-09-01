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

use depthai::node::{Camera, Sync as SyncNode};
use depthai::{CameraBoardSocket, ImgFrame, ImgFrameType, ImgResizeMode, Pipeline};
use kornia_image::Image;
use kornia_tensor::resource::MemoryDomain;
use kornia_tensor::{host_alloc, storage::TensorStorage, Tensor};
use sensor_types::FrameMeta;

use crate::{graph, policy, BoxError, OakSource, Queues};

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
    /// One per eye, owning the pixel buffers. Sharing one into a borrowed [`Image`]
    /// is a refcount bump, not a new device call.
    left: Arc<ImgFrame>,
    right: Arc<ImgFrame>,
    width: u32,
    height: u32,
    meta: FrameMeta,
    _src: std::marker::PhantomData<&'a mut OakSource>,
}

impl OakStereoFrame<'_> {
    /// Left eye (CAM_B), GRAY8 `w*h`. The stereo reference frame — the
    /// intrinsics from [`OakSource::intrinsics`] belong to this camera.
    pub fn left(&self) -> &[u8] {
        &self.left.data()[..(self.width * self.height) as usize]
    }
    /// Right eye (CAM_C), GRAY8 `w*h`, same dimensions as [`left`](Self::left).
    pub fn right(&self) -> &[u8] {
        &self.right.data()[..(self.width * self.height) as usize]
    }
    /// Per-eye width (both eyes share it).
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Per-eye height (both eyes share it).
    pub fn height(&self) -> u32 {
        self.height
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
        self.eye_image(&self.left)
    }

    /// Right eye as a host [`Image`] — see [`left_image`](Self::left_image).
    pub fn right_image(&self) -> Result<Image<u8, 1>, BoxError> {
        self.eye_image(&self.right)
    }

    fn eye_image(&self, keep: &Arc<ImgFrame>) -> Result<Image<u8, 1>, BoxError> {
        // Sharing the frame's own handle — a refcount bump, no device call, and no way
        // to end up holding a different frame than the one these pixels came from:
        // the pointer below comes from the very Arc that is the keepalive.
        let keepalive: Arc<dyn Any + Send + Sync> = keep.clone();
        let (w, h) = (self.width as usize, self.height as usize);
        let span = &keep.data()[..w * h];
        // SAFETY:
        //   - `span` points at the retained frame's pixels, non-null and `w*h` long
        //     (`check_tight_gray8` validated tight GRAY8 before this frame was built).
        //   - `keepalive` shares the frame's handle, so the memory outlives this storage.
        //   - Host memory: MemoryDomain::Host, and the OAK delivers frames to host RAM.
        let storage = unsafe {
            TensorStorage::from_borrowed_readonly(
                span.as_ptr(),
                span.len(),
                host_alloc(),
                MemoryDomain::Host,
                keepalive,
            )
        };
        // Row-major [H, W, 1]; tight rows, so these strides are exact.
        // (kornia's own `get_strides_from_shape` is not reachable from outside the crate,
        // which is why its v4l/gstreamer backends spell this out the same way.)
        let tensor = Tensor {
            storage,
            shape: [h, w, 1],
            strides: [w, 1, 1],
        };
        Image::try_from(tensor).map_err(|e| format!("borrowed Image<u8,1>: {e}").into())
    }
}

/// Validate one eye: GRAY8 is one byte per pixel, and the Rust side assumes TIGHT
/// rows (stride == w), so verify that rather than trust it. Returns the dims.
pub(crate) fn check_tight_gray8(l: &ImgFrame, r: &ImgFrame) -> Result<(u32, u32), String> {
    let (w, h) = (l.width(), l.height());
    if w == 0 || h == 0 {
        return Err("degenerate frame".into());
    }
    if (r.width(), r.height()) != (w, h) {
        return Err("eyes differ in size".into());
    }
    for (name, f) in [("left", l), ("right", r)] {
        let stride = f.stride();
        if stride != 0 && stride != w {
            return Err(format!(
                "{name} eye is not tightly packed GRAY8 (stride {stride} != w {w})"
            ));
        }
        if f.data().len() < (w * h) as usize {
            return Err(format!("{name} eye buffer is shorter than w*h"));
        }
    }
    Ok((w, h))
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
    /// The IMU is optional: the driver preflights with `connected_imu()` and only
    /// builds the IMU node when the board carries one, so an IMU-less board still
    /// streams stereo, with [`has_imu`](OakSource::has_imu) `false`. When the
    /// EEPROM carries valid IMU extrinsics, samples come out rotated into the
    /// **left (CAM_B) optical frame** — check
    /// [`imu_aligned`](OakSource::imu_aligned); otherwise they stay in the raw
    /// chip frame. A device with no CAM_B/CAM_C pair is an error — unlike depth,
    /// there is no meaningful degraded mode for a *stereo* source.
    ///
    /// `imu_hz = 0` skips the IMU node entirely.
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
        // A zero/negative rate would poison the encoder preset and the requestOutput
        // rate, failing the open for a reason nothing names.
        let fps = fps.max(1);
        let imu_hz = crate::imu::clamp_imu_hz(imu_hz);
        let dev = graph::connect_device(device)?;
        let pipeline = Pipeline::new(&dev).map_err(|e| format!("pipeline failed: {e}"))?;
        let cams = dev
            .connected_cameras()
            .map_err(|e| format!("getConnectedCameras failed: {e}"))?;

        // The stereo pair is the whole point of this modality — unlike depth (which
        // open_rgbd silently falls back from), a missing mono socket here has no
        // meaningful degraded mode. Fail loudly.
        let has_a = graph::has_socket(&cams, CameraBoardSocket::CamA);
        if !graph::has_socket(&cams, CameraBoardSocket::CamB)
            || !graph::has_socket(&cams, CameraBoardSocket::CamC)
        {
            return Err("device has no stereo pair (CAM_B/CAM_C) — open_stereo needs both".into());
        }

        let build = || -> depthai::Result<(Queues, depthai::CalibrationHandler)> {
            // CAM_B/CAM_C are MONOCHROME sensors, so they are requested as GRAY8 — one
            // byte per pixel. Asking for RGB888i would make depthai replicate the same
            // gray value across three channels on-device and then ship 3x the bytes
            // over XLink for no information. Consumers that need 3 channels expand it
            // on the GPU, where the copy is free next to the inference.
            let left = pipeline.create::<Camera>()?;
            left.build(CameraBoardSocket::CamB)?;
            let right = pipeline.create::<Camera>()?;
            right.build(CameraBoardSocket::CamC)?;
            // undistort is deliberately FALSE here (it is true on the RGBD colour path).
            // depthai's Camera node cannot rectify: its remap uses an identity
            // rectifying rotation, so the two eyes would come out undistorted but NOT
            // row-aligned, which is useless to a stereo matcher. A stereo consumer
            // rectifies on the host from `stereo_calib()`, and feeding it pixels
            // depthai had already undistorted would apply the correction TWICE,
            // silently. Raw in, host-rectified out.
            let lo = left.request_output(
                (width, height),
                Some(ImgFrameType::Gray8),
                ImgResizeMode::Crop,
                Some(fps as f32),
                Some(false),
            )?;
            let ro = right.request_output(
                (width, height),
                Some(ImgFrameType::Gray8),
                ImgResizeMode::Crop,
                Some(fps as f32),
                Some(false),
            )?;

            // Sync node: emit {left,right} as ONE MessageGroup so the host never has to
            // pair by timestamp. The eyes are frame-locked by the shared stereo trigger,
            // so the threshold only has to absorb transport jitter — half a frame
            // interval is generous.
            let sync = pipeline.create::<SyncNode>()?;
            sync.set_sync_threshold(Duration::from_nanos(1_000_000_000 / fps as u64 / 2))?;
            lo.link(&sync.input("left")?)?;
            ro.link(&sync.input("right")?)?;
            let stereo_q = sync.out()?.create_output_queue(4, false)?;

            // One EEPROM read shared by the IMU-extrinsics gate and the intrinsics —
            // readCalibration() is an RPC per call, and a wiped EEPROM comes back as
            // an empty handler (its getters then fail, handled at each use).
            let calib = dev.read_calibration()?;

            // Optional on-device H.264 of the COLOUR camera (CAM_A), viz-only: the
            // encoder runs on the device and only the ~OAK_H264_KBPS bitstream crosses
            // the link, so it costs the stereo pair nothing on the host. Same degrade
            // rule as the IMU: a board without CAM_A skips the stream.
            let video = if h264 && has_a {
                let color = pipeline.create::<Camera>()?;
                color.build(CameraBoardSocket::CamA)?;
                graph::try_add_h264_encoder(&pipeline, &color, width, height, fps, "stereo")
            } else {
                if h264 {
                    eprintln!("sensor-oak: no CAM_A on this board — skipping the H.264 viz stream");
                }
                None
            };
            Ok((
                Queues {
                    stereo: Some(stereo_q),
                    video,
                    ..Default::default()
                },
                calib,
            ))
        };
        let (queues, calib) = build().map_err(|e| format!("open_stereo failed: {e}"))?;

        // IMU is OPTIONAL: a missing IMU never costs the stereo pair (and never
        // reaches pipeline start). Left (CAM_B) is the stereo reference frame, so IMU
        // samples are rotated into ITS optical frame when the EEPROM carries the
        // extrinsics — same gate as RGBD, different reference socket.
        let imu = graph::attach_imu(&dev, &pipeline, imu_hz, &calib, CameraBoardSocket::CamB);

        pipeline
            .start()
            .map_err(|e| format!("open_stereo failed: pipeline start: {e}"))?;

        // Left (CAM_B) is the reference frame of a stereo rig, so `intrinsics()`
        // reports ITS intrinsics in this modality — never CAM_A's, whose only role
        // here is the optional viz-only H.264 stream.
        let intr = graph::read_intrinsics(&calib, CameraBoardSocket::CamB, width, height);
        // Full stereo calibration for a HOST rectifier (the pair above is raw). Read
        // from the same handler, so it costs no extra RPC. Failure is non-fatal: a
        // stereo consumer will refuse to start on `stereo_calib()`, but a plain "two
        // raw eyes" consumer still works.
        let stereo_calib = graph::read_stereo_calib(&calib, width, height);

        Ok(Self::assemble(
            dev,
            pipeline,
            width,
            height,
            intr,
            queues,
            imu,
            stereo_calib,
        ))
    }

    /// Whether the on-board IMU is running (so [`next_imu`](OakSource::next_imu)
    /// yields samples). `false` on a board with no IMU — degrade, don't abort.
    pub fn has_imu(&self) -> bool {
        self.imu_q.is_some()
    }

    /// Pull the next time-synced stereo pair. Both eyes are borrowed from the
    /// device (zero-copy, no host repack) and valid until the next call.
    ///
    /// Blocks until a pair arrives, absorbing transient empty polls (device
    /// warmup). `None` means
    /// the stream ended: a device error, or ~5 s with no pair.
    pub fn next_stereo(&mut self) -> Option<OakStereoFrame<'_>> {
        let q = self.stereo_q.as_ref()?;
        let mut tries = 0;
        let (left, right, w, h) = loop {
            let group = match q.get(Duration::from_secs(1)) {
                Ok(Some(g)) => Some(g),
                Ok(None) => None,
                Err(e) => {
                    eprintln!("sensor-oak: stereo poll failed: {e}");
                    return None; // device error → stream ended
                }
            };
            if let Some(group) = group {
                let eyes = (|| -> Result<(ImgFrame, ImgFrame), String> {
                    let l = group
                        .get::<ImgFrame>("left")
                        .map_err(|e| e.to_string())?
                        .ok_or("stereo group missing an eye")?;
                    let r = group
                        .get::<ImgFrame>("right")
                        .map_err(|e| e.to_string())?
                        .ok_or("stereo group missing an eye")?;
                    Ok((l, r))
                })();
                match eyes {
                    Ok((l, r)) => match check_tight_gray8(&l, &r) {
                        Ok((w, h)) => break (l, r, w, h),
                        // Degenerate frame — skip, don't kill the stream.
                        Err(e) if e == "degenerate frame" => {}
                        Err(e) => {
                            eprintln!("sensor-oak: {e}");
                            return None;
                        }
                    },
                    Err(e) => {
                        eprintln!("sensor-oak: {e}");
                        return None;
                    }
                }
            }
            tries += 1;
            if tries >= 5 {
                return None; // ~1s per poll → ~5s with no pair ⇒ treat as ended
            }
        };

        let ts =
            policy::steady_to_epoch_ns(left.timestamp_ns(), policy::steady_epoch_offset_cached());
        self.seq += 1;
        Some(OakStereoFrame {
            left: Arc::new(left),
            right: Arc::new(right),
            width: w,
            height: h,
            meta: FrameMeta {
                seq: self.seq,
                pts_ns: Some(ts),
                source_id: None,
            },
            _src: std::marker::PhantomData,
        })
    }
}
