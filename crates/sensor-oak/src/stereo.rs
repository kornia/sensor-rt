//! OAK-D **stereo + IMU** modality: the two mono cameras (CAM_B = left, CAM_C =
//! right) as a time-synced GRAY8 pair, plus the on-board IMU.
//!
//! This is the raw stereo + inertial source for VIO / stereo-feature work — the
//! counterpart to the colour/depth path in [`crate`], not a variant of it. It
//! builds a completely separate device pipeline ([`crate::ffi::oak_open_stereo`]):
//! no colour camera, no `StereoDepth`, no encoder.
//!
//! **The frames are host-only and this source takes no CUDA stream.** That is
//! deliberate, and the crux of the design: a consumer that wants the two eyes to
//! overlap on the GPU needs them on *different* CUDA streams, which the source
//! cannot know — so it stays a plain producer (per the crate's architecture) and
//! hands out host spans. See `examples/oakd_xfeat_stereo`, which uploads left and
//! right onto two streams to keep both XFeat backbones in flight at once.
//!

use std::any::Any;
use std::ffi::c_void;
use std::sync::Arc;

use crate::{BoxError, OakSource};
use kornia_image::Image;
use kornia_tensor::resource::MemoryDomain;
use kornia_tensor::{host_alloc, storage::TensorStorage, Tensor};
use sensor_types::FrameMeta;

/// Owns one depthai frame's pixel buffer.
///
/// `oak_poll_stereo` hands out a retain handle per eye, so the buffer's lifetime is
/// the guard's — not "until the next poll". Holding one is what makes a borrowed
/// [`Image`] sound; dropping it releases the reference.
struct RetainedFrame(*mut c_void);

// SAFETY: the handle is an owned heap `shared_ptr<ImgFrame>` copy. depthai's control
// block is atomically refcounted, and nothing here dereferences the pointer outside
// the shim, so moving the guard between threads is sound.
unsafe impl Send for RetainedFrame {}
unsafe impl Sync for RetainedFrame {}

impl Drop for RetainedFrame {
    fn drop(&mut self) {
        // SAFETY: `self.0` came from `oak_stereo_retain` and is released exactly once.
        unsafe { crate::ffi::oak_frame_release(self.0) };
    }
}

/// One time-synced stereo pair, borrowed from the source.
///
/// Both spans are GRAY8 (`w*h`, tightly packed) — these are monochrome sensors, so
/// one byte per pixel is the whole signal — and are **valid only until the next
/// [`OakSource::next_stereo`]**. That applies to the borrowed *slices*
/// ([`left`](Self::left) / [`right`](Self::right)) only: the [`Image`]s from
/// [`left_image`](Self::left_image) / [`right_image`](Self::right_image) carry the frame's own
/// retain handle and outlive it, so there is no need to copy pixels out defensively. The `'a` lifetime
/// ties this frame to the `&mut OakSource` borrow, so the borrow checker forbids
/// pulling the next pair while this one is still held — the same contract, and
/// the same enforcement, as [`OakRgbFrame`](crate::OakRgbFrame).
pub struct OakStereoFrame<'a> {
    left: &'a [u8],
    right: &'a [u8],
    width: u32,
    height: u32,
    meta: FrameMeta,
    /// One per eye, owning the pixel buffers the spans point at. Sharing them into a
    /// borrowed [`Image`] is a refcount bump, not a new FFI call.
    keep_left: Arc<RetainedFrame>,
    keep_right: Arc<RetainedFrame>,
    _src: std::marker::PhantomData<&'a mut OakSource>,
}

impl OakStereoFrame<'_> {
    /// Left eye (CAM_B), GRAY8 `w*h`. The stereo reference frame — the
    /// intrinsics from [`OakSource::intrinsics`] belong to this camera.
    pub fn left(&self) -> &[u8] {
        self.left
    }
    /// Right eye (CAM_C), GRAY8 `w*h`, same dimensions as [`left`](Self::left).
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

    /// Left eye as a host kornia grayscale [`Image`] — **zero copy**.
    ///
    /// The image borrows depthai's pixel buffer directly and shares this frame's
    /// retain handle, so unlike [`left`](Self::left) the result is NOT tied to the
    /// frame's lifetime: it stays valid across later polls and can be moved or
    /// buffered freely. The buffer is released once the frame and every image made
    /// from it are dropped.
    ///
    /// Read-only — the underlying storage refuses mutable slice access.
    pub fn left_image(&self) -> Result<Image<u8, 1>, BoxError> {
        self.eye_image(&self.keep_left, self.left)
    }

    /// Right eye as a host [`Image`] — see [`left_image`](Self::left_image).
    pub fn right_image(&self) -> Result<Image<u8, 1>, BoxError> {
        self.eye_image(&self.keep_right, self.right)
    }

    fn eye_image(&self, keep: &Arc<RetainedFrame>, span: &[u8]) -> Result<Image<u8, 1>, BoxError> {
        // Sharing the frame's own handle — a refcount bump, no FFI call, and no way to
        // end up holding a different frame than the one these pixels came from.
        let keepalive: Arc<dyn Any + Send + Sync> = keep.clone();

        let (w, h) = (self.width as usize, self.height as usize);
        // SAFETY:
        //   - `span` points at the retained frame's pixels, non-null and `w*h` long
        //     (the shim validated tight GRAY8 before handing it out).
        //   - `keepalive` shares the frame's retain handle, so the memory outlives this storage.
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
        // Row-major [H, W, 1]; the shim guarantees tight rows, so these strides are exact.
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

impl OakSource {
    /// Open the **stereo + IMU** modality: a Sync'd left/right GRAY8 pair at
    /// `fps`, plus the IMU at `imu_hz` (accelerometer + gyroscope). `device`
    /// selects the camera exactly as in [`OakSource::open`].
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
    /// The IMU is optional: the shim preflights with `getConnectedIMU()` and only
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
        let id_c = crate::device_id_cstring(device)?;
        let id_ptr = id_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        // Same clamp as the RGBD path: the BNO086 gyro tops out at 400 Hz, and a wilder
        // rate makes the firmware's sensor-enable throw at pipeline start — which on this
        // path would take the whole STEREO open down over an IMU rate.
        let imu_hz = if imu_hz > 400 {
            eprintln!("sensor-oak: imu_hz {imu_hz} clamped to 400 (BNO086 gyro maximum)");
            400
        } else {
            imu_hz
        };
        let dev = unsafe {
            crate::ffi::oak_open_stereo(
                id_ptr,
                width as i32,
                height as i32,
                fps as i32,
                imu_hz as i32,
                h264 as i32,
            )
        };
        if dev.is_null() {
            return Err(crate::last_error("oak_open_stereo"));
        }
        // Capabilities (including whether the IMU actually started) are read back from
        // the device by `from_open_device`, not assumed here. In this modality the shim
        // reports CAM_B (left) intrinsics — the stereo reference camera.
        Self::from_open_device(dev, width, height)
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
    /// warmup). `None` means
    /// the stream ended: a device error, or ~5 s with no pair.
    pub fn next_stereo(&mut self) -> Option<OakStereoFrame<'_>> {
        let mut left: *const u8 = std::ptr::null();
        let mut right: *const u8 = std::ptr::null();
        let (mut w, mut h, mut len) = (0i32, 0i32, 0i32);
        let mut ts: u64 = 0;
        let (mut l_hnd, mut r_hnd) = (std::ptr::null_mut(), std::ptr::null_mut());

        let mut tries = 0;
        loop {
            let rc = unsafe {
                crate::ffi::oak_poll_stereo(
                    self.dev, &mut left, &mut right, &mut w, &mut h, &mut len, &mut ts, &mut l_hnd,
                    &mut r_hnd,
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
        // Own the handles immediately, so every early return below still releases them.
        let (keep_left, keep_right) = (
            Arc::new(RetainedFrame(l_hnd)),
            Arc::new(RetainedFrame(r_hnd)),
        );
        let (w, h) = (w as u32, h as u32);
        if w == 0 || h == 0 {
            return None;
        }
        // The shim guarantees both eyes are tight GRAY8 of these dims (it validates stride and
        // size and errors out otherwise), so one length covers both spans.
        let n = len.max(0) as usize;
        let left = unsafe { std::slice::from_raw_parts(left, n) };
        let right = unsafe { std::slice::from_raw_parts(right, n) };

        self.seq += 1;
        Some(OakStereoFrame {
            left,
            right,
            width: w,
            height: h,
            meta: FrameMeta {
                seq: self.seq,
                pts_ns: Some(ts),
                source_id: None,
            },
            keep_left,
            keep_right,
            _src: std::marker::PhantomData,
        })
    }
}
