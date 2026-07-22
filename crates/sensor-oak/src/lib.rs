//! OAK-D camera source: synced **host RGB + aligned depth (uint16 mm) + factory
//! intrinsics**, plus a stereo-pair + IMU modality (see [`stereo`]).
//! Safe wrapper over the bundled depthai-core C shim (`oak_bridge.h`).
//!
//! Same shape as the RTSP source: a `next_frame()` loop. The OAK computes stereo
//! depth on its own VPU, so there is no depth model on the host. **Nothing here
//! touches CUDA**: frames come out on the host and the consumer owns any upload.
//! 3D points are in the **camera frame**.

use std::ffi::{CStr, CString};

use kornia_image::{Image, ImageSize};
use sensor_types::FrameMeta;

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
mod depth;
mod ffi;
mod stereo;
pub use depth::OakDepthMap;
pub use stereo::{ImuSample, StereoFrame};

/// One synced OAK frame: RGB plus optional aligned depth, both on the host.
///
/// **Both buffers are borrowed from the source and only valid until the next
/// [`OakSource::next_frame`].** The `'a` lifetime ties this frame to the `&mut
/// OakSource` borrow, so the borrow checker forbids pulling the next frame while
/// this one (or anything it lent out) is still held — preventing use-after-free
/// of the borrowed device buffers.
pub struct OakFrame<'a> {
    // Borrowed from the source and valid only until the next poll, so they must NOT
    // be movable out of the frame.
    rgb_host: &'a [u8],
    meta: FrameMeta,
    width: u32,
    height: u32,
    depth: Option<OakDepthMap>,
    _src: std::marker::PhantomData<&'a mut OakSource>,
}

impl OakFrame<'_> {
    /// Host rgb span, borrowed — valid only while this frame is held. Always raw RGB888 (`w*h*3`,
    /// tightly packed). The compressed colour stream is separate — see [`OakSource::next_video`].
    pub fn rgb_host(&self) -> &[u8] {
        self.rgb_host
    }
    /// Frame metadata (sequence + capture pts).
    pub fn meta(&self) -> &FrameMeta {
        &self.meta
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    /// RGB as an owned **host** kornia [`Image`], ready to hand to CPU code or to
    /// push to the GPU with kornia's `to_cuda_image(&stream)`.
    ///
    /// This **copies** `w*h*3` bytes, because a kornia `Image` owns its buffer while
    /// [`rgb_host`](Self::rgb_host) only borrows the device's. For a per-frame hot
    /// loop, prefer `rgb_host()` straight into your own reused device buffer — that
    /// path stays copy-free on the host. Use this when the convenience is worth one
    /// memcpy (~2.7 MB at 1280x720).
    pub fn rgb_image(&self) -> Result<Image<u8, 3>, BoxError> {
        Image::new(
            ImageSize {
                width: self.width as usize,
                height: self.height as usize,
            },
            self.rgb_host.to_vec(),
        )
        .map_err(|e| format!("build host Image<u8,3>: {e}").into())
    }
    /// Aligned depth map, if the device has a stereo pair. Borrowed — same "valid only while this
    /// frame is held" contract as [`rgb_host`](Self::rgb_host).
    pub fn depth(&self) -> Option<&OakDepthMap> {
        self.depth.as_ref()
    }
}

/// OAK-D source. `open` then `next_frame()` in a loop (like `RtspSource`).
pub struct OakSource {
    dev: *mut ffi::OakDevice,
    h264: bool, // device also runs a standalone H.264 colour stream (next_video)
    width: u32,
    height: u32,
    seq: u64,
    intr: OakIntrinsics,
    has_depth: bool,
    has_sync: bool,
    // Stereo+IMU modality (`open_stereo`) — see `stereo.rs`. Both false on every other path.
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
    /// Open an OAK device and start an RGB(+aligned depth) pipeline. `device` selects the camera:
    /// `None` = first available; `Some(id)` = a specific MxId (USB or PoE) or IP (PoE).
    ///
    /// Takes **no CUDA stream**: this driver never touches the GPU. Frames arrive on the host —
    /// [`OakFrame::rgb_host`] (borrowed, copy-free) or [`OakFrame::rgb_image`] (owned) — and a
    /// consumer that wants them on the device owns that upload and the stream it runs on. That keeps
    /// the driver a plain producer and keeps CUDA out of processes that only want pixels.
    pub fn open(device: Option<&str>, width: u32, height: u32, fps: u32) -> Result<Self, BoxError> {
        Self::open_inner(device, width, height, fps, false, true, false)
    }

    /// Open **no-stereo**: the on-device H.264 encoder plus a raw RGB888 stream, and NO StereoDepth —
    /// for a camera with no stereo pair or no usable factory calibration, where the depth node would
    /// fail at runtime and take the pipeline down.
    ///
    /// The raw RGB output is deliberate even though H.264 alone would be cheaper on the link: this
    /// repo's cameras feed **onboard compute**, which needs actual frames (calibration snapshots,
    /// fusion colouring), not just a bitstream for viewing. So [`OakSource::next_frame`] DOES yield
    /// frames here — RGB with no depth — and [`OakSource::next_video`] drains the encoded stream
    /// alongside it. Budget for the raw stream: `w*h*3` bytes/frame over XLink.
    ///
    /// `device`: see [`OakSource::open`]. Takes NO CUDA stream — nothing on this path uploads to the
    /// GPU, so no CUDA context is created.
    pub fn open_video_only(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Self, BoxError> {
        Self::open_inner(device, width, height, fps, true, false, true)
    }

    /// Open host-only **plus an on-device hardware H.264 colour stream**: the synced [`OakFrame`] carries
    /// raw RGB888 (+ aligned depth when `depth`), and [`OakSource::next_video`] drains the separate H.264
    /// bitstream (efficient video for viewing / recording). Set `depth = false` for an
    /// **uncalibrated** camera — it skips the StereoDepth node, which would otherwise fail at runtime and
    /// crash the pipeline. `device`: see [`OakSource::open`]. Takes NO CUDA stream: the host-only + H.264
    /// path never uploads to the GPU (the decoupled RGB/depth polls hand out host copies), so the
    /// caller creates no CUDA context — saving ~200-300 MB RSS per camera process.
    pub fn open_video(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        depth: bool,
    ) -> Result<Self, BoxError> {
        Self::open_inner(device, width, height, fps, true, depth, false)
    }

    // Private funnel for the open_* variants; the trailing bools are all set by
    // those named constructors, not by end users.
    #[allow(clippy::too_many_arguments)]
    fn open_inner(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        h264: bool,
        depth: bool,
        video_only: bool,
    ) -> Result<Self, BoxError> {
        // device id → C string (NULL for "first available"); kept alive across the FFI call.
        let id_c = device
            .map(CString::new)
            .transpose()
            .map_err(|e| format!("bad device id: {e}"))?;
        let id_ptr = id_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let dev = unsafe {
            ffi::oak_open(
                id_ptr,
                width as i32,
                height as i32,
                fps as i32,
                h264 as i32,
                depth as i32,
                video_only as i32,
            )
        };
        if dev.is_null() {
            let e = unsafe { CStr::from_ptr(ffi::oak_last_error()) }
                .to_string_lossy()
                .into_owned();
            return Err(format!("oak_open failed: {e}").into());
        }
        Self::from_open_device(dev, width, height)
    }

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
            // Read back, not assumed: the shim may decline to build the encoder.
            h264: unsafe { ffi::oak_has_video(dev) } != 0,
            width,
            height,
            seq: 0,
            intr: OakIntrinsics { fx, fy, cx, cy },
            has_depth: unsafe { ffi::oak_has_depth(dev) } != 0,
            has_sync: unsafe { ffi::oak_has_sync(dev) } != 0,
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
    pub fn has_depth(&self) -> bool {
        self.has_depth
    }
    /// Whether the device runs the synced RGBD pipeline (so [`OakSource::next_frame`] yields frames).
    /// `false` means it auto-fell-back to video-only (mono / uncalibrated): drain [`OakSource::next_video`]
    /// only. Always check this after [`OakSource::open_video`] to pick the right drain loop.
    pub fn has_sync(&self) -> bool {
        self.has_sync
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Pull the next synced frame: the borrowed RGB888 span plus the borrowed
    /// aligned depth map, both valid only until the next call.
    ///
    /// **Zero copies.** The shim hands out depthai's RGB and depth buffers directly
    /// (no host repack, no depth copy) and this driver adds nothing on top. A
    /// consumer that wants the frame on the GPU uploads the span into its own
    /// buffer — see [`OakFrame::rgb_host`] — or takes an owned host image via
    /// [`OakFrame::rgb_image`] at the cost of one memcpy.
    pub fn next_frame(&mut self) -> Option<OakFrame<'_>> {
        let mut rgb: *const u8 = std::ptr::null();
        let mut depth: *const u16 = std::ptr::null();
        let (mut w, mut h) = (0i32, 0i32);
        let mut rgb_len: i32 = 0; // rgb byte length (w*h*3 for raw; jpeg size when encoded)
        let mut ts: u64 = 0;
        // Depth may be a smaller (on-device downscaled) grid than RGB — the shim reports its own dims.
        let (mut dw, mut dh) = (0i32, 0i32);
        // Block until a frame arrives, absorbing transient empty polls (rc==0, e.g.
        // device warmup) like RtspSource does. Only a hard error (rc<0) or a
        // persistent timeout ends the stream. Each poll has a ~2s shim timeout.
        let mut rc;
        let mut tries = 0;
        loop {
            rc = unsafe {
                ffi::oak_poll(
                    self.dev,
                    &mut rgb,
                    &mut depth,
                    &mut w,
                    &mut h,
                    &mut rgb_len,
                    &mut ts,
                    &mut dw,
                    &mut dh,
                )
            };
            if rc == 1 && !rgb.is_null() {
                break;
            }
            if rc < 0 {
                return None; // device error → stream ended
            }
            tries += 1;
            if tries >= 5 {
                return None; // oak_poll blocks up to 1s/call → ~5s with no frame ⇒ treat as ended
            }
        }
        let (w, h) = (w as u32, h as u32);
        if w == 0 || h == 0 {
            return None;
        } // guard a degenerate device frame
        let npx = w as usize * h as usize;

        // The shim's rgb span (zero-copy view into depthai's buffer), valid until the next poll —
        // `rgb_len` bytes: RGB888 (w*h*3) when raw, the JPEG bitstream length when encoded.
        let host = unsafe { std::slice::from_raw_parts(rgb, rgb_len.max(0) as usize) };
        self.seq += 1;
        let meta = FrameMeta {
            seq: self.seq,
            pts_ns: Some(ts),
            source_id: None,
        };

        // Raw frames must be RGB888 (w*h*3). If the shim ever hands back a different
        // length (e.g. an encoded stream), refuse the frame rather than hand out a
        // span whose documented shape is a lie.
        if host.len() != npx * 3 {
            return None;
        }

        // Zero-copy: borrow the OAK's aligned depth buffer (valid until next poll). Depth carries its OWN
        // dims (dw×dh) — it may be a downscaled grid aligned to the RGB, so consumers scale by w/dw.
        let depth = (!depth.is_null() && dw > 0 && dh > 0)
            .then(|| unsafe { OakDepthMap::borrowed(depth, dw as u32, dh as u32) });

        Some(OakFrame {
            rgb_host: host,
            meta,
            width: w,
            height: h,
            depth,
            _src: std::marker::PhantomData,
        })
    }

    /// Whether this source runs the on-device H.264 colour stream ([`OakSource::open_video`]).
    pub fn has_video(&self) -> bool {
        self.h264
    }

    /// Drain the next on-device **H.264** access unit, if one is ready (non-blocking). Returns the
    /// encoded bytes (copied out, so the caller may hold/publish them freely) + the capture timestamp
    /// in ns. `None` when no frame is queued or H.264 wasn't enabled. Call in a loop until `None` each
    /// iteration so the encoder queue never overflows — a dropped P-frame glitches the stream until the
    /// next keyframe (~1 s). Separate from [`OakSource::next_frame`]: drain it OUTSIDE a held frame
    /// (both borrow `&mut self`).
    pub fn next_video(&mut self) -> Option<(Vec<u8>, u64)> {
        if !self.h264 {
            return None;
        }
        let mut data: *const u8 = std::ptr::null();
        let mut len: i32 = 0;
        let mut ts: u64 = 0;
        let rc = unsafe { ffi::oak_poll_video(self.dev, &mut data, &mut len, &mut ts) };
        if rc == 1 && !data.is_null() && len > 0 {
            // Copy the bitstream out while the shim still pins it (valid until the next call).
            let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec();
            Some((bytes, ts))
        } else {
            None // rc==0 (none ready) or rc<0 (error) → nothing this poll
        }
    }

    /// DECOUPLED raw-RGB poll: the next RGB888 frame from its own queue (non-blocking), copied out with
    /// its dims + capture timestamp (ns). `None` when none is queued. Independent of [`OakSource::next_depth`]
    /// — pair them by timestamp on the consumer. Drain in a loop until `None`.
    pub fn next_rgb(&mut self) -> Option<(Vec<u8>, u32, u32, u64)> {
        let mut rgb: *const u8 = std::ptr::null();
        let (mut w, mut h, mut len, mut ts) = (0i32, 0i32, 0i32, 0u64);
        let rc =
            unsafe { ffi::oak_poll_rgb(self.dev, &mut rgb, &mut w, &mut h, &mut len, &mut ts) };
        if rc == 1 && !rgb.is_null() && len > 0 {
            let bytes = unsafe { std::slice::from_raw_parts(rgb, len as usize) }.to_vec();
            Some((bytes, w as u32, h as u32, ts))
        } else {
            None
        }
    }

    /// DECOUPLED depth poll: the next uint16-mm depth frame from its own queue (non-blocking) at the full
    /// stereo rate, copied out with its dims (may be < RGB size) + capture timestamp. `None` when none is
    /// queued. Drain in a loop until `None`.
    pub fn next_depth(&mut self) -> Option<(Vec<u16>, u32, u32, u64)> {
        let mut depth: *const u16 = std::ptr::null();
        let (mut dw, mut dh, mut ts) = (0i32, 0i32, 0u64);
        let rc = unsafe { ffi::oak_poll_depth(self.dev, &mut depth, &mut dw, &mut dh, &mut ts) };
        if rc == 1 && !depth.is_null() && dw > 0 && dh > 0 {
            let n = dw as usize * dh as usize;
            let vals = unsafe { std::slice::from_raw_parts(depth, n) }.to_vec();
            Some((vals, dw as u32, dh as u32, ts))
        } else {
            None
        }
    }

    /// A standalone handle to the on-device H.264 queue, for draining video from a **dedicated thread**
    /// while the main thread pulls synced RGBD (`next_frame`) concurrently. The two touch different
    /// depthai queues (video vs sync), which are independently thread-safe, so the encoded stream ships
    /// at full fps regardless of the slower/heavier RGBD pull. Returns `None` if this source has no H.264.
    ///
    /// SAFETY CONTRACT: the tap borrows the same device as the `OakSource`. The caller MUST stop using
    /// the tap before dropping/reopening the source (the device is freed on drop) — coordinate with a
    /// lock so no `VideoTap::next` runs during a reconnect.
    pub fn video_tap(&self) -> Option<VideoTap> {
        self.h264.then_some(VideoTap { dev: self.dev })
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

/// See [`OakSource::video_tap`]. Holds a raw device handle; `Send` so it can live in the video thread.
pub struct VideoTap {
    dev: *mut ffi::OakDevice,
}
// SAFETY: only ever used to call oak_poll_video (video queue), on one thread at a time, never
// concurrently with a source drop (the caller serialises via a lock — see `video_tap`).
unsafe impl Send for VideoTap {}

impl VideoTap {
    /// Drain the next H.264 access unit (non-blocking); see [`OakSource::next_video`].
    pub fn next(&self) -> Option<(Vec<u8>, u64)> {
        let mut data: *const u8 = std::ptr::null();
        let mut len: i32 = 0;
        let mut ts: u64 = 0;
        let rc = unsafe { ffi::oak_poll_video(self.dev, &mut data, &mut len, &mut ts) };
        if rc == 1 && !data.is_null() && len > 0 {
            let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec();
            Some((bytes, ts))
        } else {
            None
        }
    }
}

impl Drop for OakSource {
    fn drop(&mut self) {
        unsafe { ffi::oak_close(self.dev) };
    }
}
