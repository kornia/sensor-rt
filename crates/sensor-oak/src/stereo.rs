//! OAK-D **stereo + IMU** modality: the two mono cameras (CAM_B = left, CAM_C =
//! right) as a time-synced RGB888 pair, plus the on-board IMU.
//!
//! This is the raw stereo + inertial source for VIO / stereo-feature work — the
//! counterpart to the colour/depth path in [`crate`], not a variant of it. It
//! builds a completely separate device pipeline ([`crate::ffi::oak_open_stereo`]):
//! no colour camera, no `StereoDepth`, no encoder.
//!
//! **The frames are host-only and this source takes no [`CudaStream`].** That is
//! deliberate, and the crux of the design: a consumer that wants the two eyes to
//! overlap on the GPU needs them on *different* CUDA streams, which the source
//! cannot know — so it stays a plain producer (per the crate's architecture) and
//! hands out host spans. See `examples/oakd_xfeat_stereo`, which uploads left and
//! right onto two streams to keep both XFeat backbones in flight at once.
//!
//! [`CudaStream`]: cudarc::driver::CudaStream

use std::ffi::CStr;

use crate::BoxError;
use sensor_types::FrameMeta;

use crate::OakSource;

/// One IMU reading: accelerometer + gyroscope, sampled together.
///
/// `ts_ns` is on the **same host-synced epoch timeline as the image frames**, so
/// samples can be interpolated directly against a [`StereoFrame`]'s timestamp
/// without a clock alignment step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    /// Capture time, epoch nanoseconds — comparable to `StereoFrame::meta().pts_ns`.
    pub ts_ns: u64,
    /// Accelerometer, m/s².
    pub accel: [f32; 3],
    /// Gyroscope, rad/s.
    pub gyro: [f32; 3],
}

/// One time-synced stereo pair, borrowed from the source.
///
/// Both spans are RGB888 (`w*h*3`, tightly packed — the mono sensors are streamed
/// as 3-channel, so gray is already replicated and no conversion is needed) and
/// are **valid only until the next [`OakSource::next_stereo`]**. The `'a` lifetime
/// ties this frame to the `&mut OakSource` borrow, so the borrow checker forbids
/// pulling the next pair while this one is still held — the same contract, and
/// the same enforcement, as [`OakFrame`](crate::OakFrame).
pub struct StereoFrame<'a> {
    left: &'a [u8],
    right: &'a [u8],
    width: u32,
    height: u32,
    meta: FrameMeta,
    _src: std::marker::PhantomData<&'a mut OakSource>,
}

impl StereoFrame<'_> {
    /// Left eye (CAM_B), RGB888 `w*h*3`. The stereo reference frame — the
    /// intrinsics from [`OakSource::intrinsics`] belong to this camera.
    pub fn left(&self) -> &[u8] {
        self.left
    }
    /// Right eye (CAM_C), RGB888 `w*h*3`, same dimensions as [`left`](Self::left).
    pub fn right(&self) -> &[u8] {
        self.right
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
}

impl OakSource {
    /// Open the **stereo + IMU** modality: a Sync'd left/right RGB888 pair at
    /// `fps`, plus the IMU at `imu_hz` (accelerometer + gyroscope). `device`
    /// selects the camera exactly as in [`OakSource::open`].
    ///
    /// Takes **no CUDA stream** — see the module docs: the consumer owns the
    /// upload, because it alone knows which stream each eye belongs on.
    ///
    /// The IMU is optional: a board without one (or whose IMU fails to start)
    /// still streams stereo, with [`has_imu`](OakSource::has_imu) `false`. A
    /// device with no CAM_B/CAM_C pair is an error — unlike depth, there is no
    /// meaningful degraded mode for a *stereo* source.
    ///
    /// `imu_hz = 0` skips the IMU node entirely.
    pub fn open_stereo(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        imu_hz: u32,
    ) -> Result<Self, BoxError> {
        let id_c = Self::device_id_cstring(device)?;
        let id_ptr = id_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let dev = unsafe {
            crate::ffi::oak_open_stereo(
                id_ptr,
                width as i32,
                height as i32,
                fps as i32,
                imu_hz as i32,
            )
        };
        if dev.is_null() {
            let e = unsafe { CStr::from_ptr(crate::ffi::oak_last_error()) }
                .to_string_lossy()
                .into_owned();
            return Err(format!("oak_open_stereo failed: {e}").into());
        }
        // Capabilities (including whether the IMU actually started) are read back from
        // the device by `from_open_device`, not assumed here. In this modality the shim
        // reports CAM_B (left) intrinsics — the stereo reference camera.
        Self::from_open_device(dev, width, height)
    }

    /// Whether this source runs the stereo-pair pipeline (so
    /// [`next_stereo`](OakSource::next_stereo) yields frames).
    pub fn has_stereo(&self) -> bool {
        self.has_stereo
    }

    /// Whether the on-board IMU is running (so [`next_imu`](OakSource::next_imu)
    /// yields samples). `false` on a board with no IMU — degrade, don't abort.
    pub fn has_imu(&self) -> bool {
        self.has_imu
    }

    /// Pull the next time-synced stereo pair. Both eyes are borrowed from the
    /// device (zero-copy, no host repack) and valid until the next call.
    ///
    /// Blocks until a pair arrives, absorbing transient empty polls (device
    /// warmup) exactly like [`next_frame`](OakSource::next_frame). `None` means
    /// the stream ended: a device error, or ~5 s with no pair.
    pub fn next_stereo(&mut self) -> Option<StereoFrame<'_>> {
        if !self.has_stereo {
            return None;
        }
        let mut left: *const u8 = std::ptr::null();
        let mut right: *const u8 = std::ptr::null();
        let (mut w, mut h, mut len) = (0i32, 0i32, 0i32);
        let mut ts: u64 = 0;

        let mut tries = 0;
        loop {
            let rc = unsafe {
                crate::ffi::oak_poll_stereo(
                    self.dev, &mut left, &mut right, &mut w, &mut h, &mut len, &mut ts,
                )
            };
            if rc == 1 && !left.is_null() && !right.is_null() {
                break;
            }
            if rc < 0 {
                return None; // device error → stream ended
            }
            tries += 1;
            if tries >= 5 {
                return None; // ~1s per poll → ~5s with no pair ⇒ treat as ended
            }
        }
        let (w, h) = (w as u32, h as u32);
        if w == 0 || h == 0 {
            return None;
        }
        // The shim guarantees both eyes are tight RGB888 of these dims (it validates stride and
        // size and errors out otherwise), so one length covers both spans.
        let n = len.max(0) as usize;
        let left = unsafe { std::slice::from_raw_parts(left, n) };
        let right = unsafe { std::slice::from_raw_parts(right, n) };

        self.seq += 1;
        Some(StereoFrame {
            left,
            right,
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

    /// Drain queued IMU samples, appending them to `out` in capture order;
    /// returns how many were appended. Non-blocking.
    ///
    /// The IMU reports far faster than the frame rate (hundreds of Hz vs ~30),
    /// which is why it is drained separately rather than folded into the synced
    /// pair — bundling it per frame would throw away all but one sample. Call it
    /// once per frame (or more often); samples that don't fit `cap` stay queued
    /// in order for the next call, so nothing is dropped.
    ///
    /// Takes `&mut self` like every other poll, so drain it **outside** a held
    /// [`StereoFrame`].
    pub fn next_imu(&mut self, out: &mut Vec<ImuSample>, cap: usize) -> usize {
        if !self.has_imu || cap == 0 {
            return 0;
        }
        // Reused staging buffer: this runs every frame, and a fresh `vec![default; cap]`
        // would be a `cap * 32`-byte alloc + memset each time (16 KB at the usual
        // cap = 512) to receive the handful of samples that actually arrived.
        if self.imu_scratch.len() < cap {
            self.imu_scratch
                .resize(cap, crate::ffi::OakImuSample::default());
        }
        let mut n: i32 = 0;
        let rc = unsafe {
            crate::ffi::oak_poll_imu(self.dev, self.imu_scratch.as_mut_ptr(), cap as i32, &mut n)
        };
        if rc != 1 || n <= 0 {
            return 0;
        }
        let n = (n as usize).min(cap);
        out.extend(self.imu_scratch[..n].iter().map(|s| ImuSample {
            ts_ns: s.ts_ns,
            accel: [s.ax, s.ay, s.az],
            gyro: [s.gx, s.gy, s.gz],
        }));
        n
    }
}
