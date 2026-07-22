//! OAK-D camera source: synced **RGB (device `VrtImage`) + aligned depth (CPU,
//! uint16 mm) + factory intrinsics**, the real-depth 3D detection source for the
//! `Box3DTracker`. Safe wrapper over the `oak-sys` C shim (depthai-core v3).
//!
//! Mirrors `vrt-gst`'s `RtspSource`: a `next_frame()` loop. The OAK computes
//! stereo depth on its own VPU, so there's no depth model on the Jetson. RGB is
//! uploaded H2D into a reused device buffer for RF-DETR; depth stays on the CPU
//! (sampled per-box — cheap). 3D points are in the **camera frame**.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use cudarc::driver::CudaStream;
use kornia_image::Image;
use kornia_tensor::Tensor;
use sensor_types::{DepthMap, FrameMeta};
use vrt::BoxError;
use vrt_types::CameraIntrinsics;

// The depth→3D bridge (`Lifter`/`SizePriors`) is sensor-agnostic and lives in
// `vrt-lift`; camera `CameraIntrinsics` are a `vrt` core type. This crate is just the
// OAK sensor driver and no longer depends on the tracker.

mod stereo;
pub use stereo::{ImuSample, StereoFrame};

/// One synced OAK frame: RGB on the device, optional aligned depth on the host.
///
/// **Both buffers are borrowed from the source and only valid until the next
/// [`OakSource::next_frame`].** The `'a` lifetime ties this frame to the `&mut
/// OakSource` borrow, so the borrow checker forbids pulling the next frame while
/// this one (or anything it lent out) is still held — preventing use-after-free
/// of the reused RGB device buffer and the borrowed depth buffer.
pub struct OakFrame<'a> {
    // All views are borrowed from the source and valid only until the next poll (so they must
    // NOT be movable out of the frame). `rgb_host` is the shim's RGB888 span (always present);
    // `rgb_device` is the H2D-uploaded image, built only on the upload (detection) path.
    rgb_host: &'a [u8],
    meta: FrameMeta,
    width: u32,
    height: u32,
    rgb_device: Option<&'a Image<u8, 3>>,
    depth: Option<DepthMap>,
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
    /// RGB as a device kornia [`Image`] (uploaded H2D) — present only when opened with
    /// [`OakSource::open`] (the detection path); `None` for the host-only
    /// [`OakSource::open_host`] path. Pair with [`OakFrame::meta`] for the stamp.
    pub fn rgb_device(&self) -> Option<&Image<u8, 3>> {
        self.rgb_device
    }
    /// Device RGB image. Panics if the source was opened host-only; kept for the detection
    /// examples that always upload. Borrowed — valid only while this frame is held.
    pub fn rgb(&self) -> &Image<u8, 3> {
        self.rgb_device
            .expect("OakFrame::rgb() needs OakSource::open (upload); use rgb_host()/rgb_device()")
    }
    /// Aligned depth map, if the device has a stereo pair. Borrowed — same "valid only while this
    /// frame is held" contract as [`rgb_host`](Self::rgb_host).
    pub fn depth(&self) -> Option<&DepthMap> {
        self.depth.as_ref()
    }
}

/// OAK-D source. `open` then `next_frame()` in a loop (like `RtspSource`).
pub struct OakSource {
    dev: *mut oak_sys::oak_device,
    stream: Option<Arc<CudaStream>>, // Some only on the upload (detection) path; None for pure encoders
    rgb_img: Option<Image<u8, 3>>,   // reused device RGB888 image; None when host-only
    upload: bool,                    // upload RGB H2D (detection) vs host-only (pure encoders)
    h264: bool, // device also runs a standalone H.264 colour stream (next_video)
    width: u32,
    height: u32,
    seq: u64,
    intr: CameraIntrinsics,
    has_depth: bool,
    has_sync: bool,
    // Stereo+IMU modality (`open_stereo`) — see `stereo.rs`. Both false on every other path.
    has_stereo: bool,
    has_imu: bool,
    /// Reused staging buffer for `next_imu`, so draining inertial samples every
    /// frame costs no allocation once it has grown.
    imu_scratch: Vec<oak_sys::oak_imu_sample>,
}

// SAFETY: the device handle is used single-threaded by the owning loop; the CUDA
// stream is Arc-backed. (depthai's own threads live behind the C shim.)
unsafe impl Send for OakSource {}

impl OakSource {
    /// Open an OAK device and start an RGB(+aligned depth) pipeline, uploading each RGB frame H2D into
    /// a reused device buffer (the detection path). `device` selects the camera: `None` = first
    /// available; `Some(id)` = a specific MxId (USB or PoE) or IP (PoE). `stream` is the shared CUDA
    /// stream.
    pub fn open(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        Self::open_inner(
            device,
            width,
            height,
            fps,
            Some(stream),
            true,
            false,
            true,
            false,
        )
    }

    /// Open host-only: skip the GPU alloc + the per-frame H2D. Frames expose [`OakFrame::rgb_host`]
    /// (the shim's raw RGB888 span) but not a device image — for nodes that never touch CUDA, so there
    /// are zero device copies. `device`: see [`OakSource::open`].
    pub fn open_host(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, BoxError> {
        Self::open_inner(
            device,
            width,
            height,
            fps,
            Some(stream),
            false,
            false,
            true,
            false,
        )
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
        Self::open_inner(device, width, height, fps, None, false, true, false, true)
    }

    /// Open host-only **plus an on-device hardware H.264 colour stream**: the synced [`OakFrame`] carries
    /// raw RGB888 (+ aligned depth when `depth`), and [`OakSource::next_video`] drains the separate H.264
    /// bitstream (efficient video for Foxglove / recording). For `flux-oak`. Set `depth = false` for an
    /// **uncalibrated** camera — it skips the StereoDepth node, which would otherwise fail at runtime and
    /// crash the pipeline. `device`: see [`OakSource::open`]. Takes NO CUDA stream: the host-only + H.264
    /// path never uploads to the GPU (the decoupled RGB/depth polls hand out host copies), so flux-oak
    /// creates no CUDA context — saving ~200-300 MB RSS per camera process.
    pub fn open_video(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        depth: bool,
    ) -> Result<Self, BoxError> {
        Self::open_inner(device, width, height, fps, None, false, true, depth, false)
    }

    // Private funnel for the open_* variants; the trailing bools are all set by
    // those named constructors, not by end users.
    #[allow(clippy::too_many_arguments)]
    fn open_inner(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        stream: Option<Arc<CudaStream>>,
        upload: bool,
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
            oak_sys::oak_open(
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
            let e = unsafe { CStr::from_ptr(oak_sys::oak_last_error()) }
                .to_string_lossy()
                .into_owned();
            return Err(format!("oak_open failed: {e}").into());
        }
        let rgb_img = if upload {
            let s = stream
                .as_ref()
                .ok_or("upload path requires a CUDA stream (use OakSource::open)")?;
            Some(alloc_rgb_image(s, width, height)?)
        } else {
            None
        };
        Self::from_open_device(dev, width, height, stream, rgb_img, upload, h264)
    }

    /// Wrap a device the shim has already opened, reading every capability back
    /// from it rather than assuming what the constructor asked for.
    ///
    /// Shared by `open_inner` and [`open_stereo`](OakSource::open_stereo): those
    /// two build genuinely different pipelines (they share no depthai nodes), but
    /// the "what did we actually get?" half is identical, and hard-coding it per
    /// constructor is how the two drift. The `oak_has_*` accessors are the source
    /// of truth — e.g. `has_sync` is false when a mono/uncalibrated device
    /// auto-fell-back to a video-only pipeline, which the caller cannot predict.
    pub(crate) fn from_open_device(
        dev: *mut oak_sys::oak_device,
        width: u32,
        height: u32,
        stream: Option<Arc<CudaStream>>,
        rgb_img: Option<Image<u8, 3>>,
        upload: bool,
        h264: bool,
    ) -> Result<Self, BoxError> {
        let (mut fx, mut fy, mut cx, mut cy) = (0.0f32, 0.0, 0.0, 0.0);
        unsafe { oak_sys::oak_intrinsics(dev, &mut fx, &mut fy, &mut cx, &mut cy) };
        Ok(Self {
            dev,
            stream,
            rgb_img,
            upload,
            h264,
            width,
            height,
            seq: 0,
            intr: CameraIntrinsics { fx, fy, cx, cy },
            has_depth: unsafe { oak_sys::oak_has_depth(dev) } != 0,
            has_sync: unsafe { oak_sys::oak_has_sync(dev) } != 0,
            has_stereo: unsafe { oak_sys::oak_has_stereo(dev) } != 0,
            has_imu: unsafe { oak_sys::oak_has_imu(dev) } != 0,
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

    pub fn intrinsics(&self) -> CameraIntrinsics {
        self.intr
    }
    /// The shared CUDA stream the RGB upload runs on. **Build the consuming model
    /// with this stream** (e.g. `RfDetr::new(engine, cam.cuda_stream(), …)`): the
    /// per-frame H2D upload is enqueued async on it, so a consumer on the same
    /// stream sees a completed transfer after the usual single per-frame sync,
    /// while a consumer on a *different* stream would race the upload.
    /// Only the upload path ([`OakSource::open`] / [`OakSource::open_host`]) carries a stream;
    /// panics on a video-only / decoupled source, which never touches the GPU.
    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        self.stream
            .clone()
            .expect("cuda_stream() is only valid on an upload OakSource (open/open_host)")
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

    /// Pull the next synced frame: RGB888 uploaded H2D into a reused device
    /// `VrtImage` (valid until the next call), plus the borrowed aligned depth map.
    ///
    /// This is the **only copy** in the OAK path: one host→device upload of the
    /// 3-channel RGB. The shim hands out depthai's RGB and depth buffers directly
    /// (no host repack, no depth copy), and the detector preprocessor consumes
    /// RGB888 natively (no 3→4 expansion). Both borrowed buffers are valid only
    /// until the next `next_frame`.
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
                oak_sys::oak_poll(
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

        // Detection path only: upload RGB888 H2D into the reused device image (tightly
        // packed RGB8 — exactly what kornia's Preprocessor + the vrt models consume).
        // Host-only callers skip this entirely (no alloc, no copy, no device image).
        let rgb_device = if self.upload {
            // The upload path requires raw RGB888 (w*h*3). If the shim ever hands
            // back a different length (e.g. an encoded stream), skip the upload
            // rather than copy a mismatched byte count into the device buffer.
            if host.len() != npx * 3 {
                return None;
            }
            // `upload` is only ever set by the stream-carrying constructors, so this is Some.
            let stream = self.stream.as_ref()?;
            let need = self
                .rgb_img
                .as_ref()
                .and_then(|i| i.as_cudaslice())
                .is_none_or(|s| s.len() != npx * 3);
            if need {
                self.rgb_img = Some(alloc_rgb_image(stream, w, h).ok()?);
            }
            let img = self.rgb_img.as_mut().unwrap();
            let dst = img.as_cudaslice_mut()?;
            stream.memcpy_htod(host, dst).ok()?;
            // Borrow the persistent image for this frame; valid until the next
            // next_frame() overwrites it (the `'a` borrow enforces that).
            self.rgb_img.as_ref()
        } else {
            None
        };

        // Zero-copy: borrow the OAK's aligned depth buffer (valid until next poll). Depth carries its OWN
        // dims (dw×dh) — it may be a downscaled grid aligned to the RGB, so consumers scale by w/dw.
        let depth = (!depth.is_null() && dw > 0 && dh > 0)
            .then(|| unsafe { DepthMap::borrowed(depth, dw as u32, dh as u32) });

        Some(OakFrame {
            rgb_host: host,
            meta,
            width: w,
            height: h,
            rgb_device,
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
        let rc = unsafe { oak_sys::oak_poll_video(self.dev, &mut data, &mut len, &mut ts) };
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
            unsafe { oak_sys::oak_poll_rgb(self.dev, &mut rgb, &mut w, &mut h, &mut len, &mut ts) };
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
        let rc =
            unsafe { oak_sys::oak_poll_depth(self.dev, &mut depth, &mut dw, &mut dh, &mut ts) };
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
    let rc = unsafe { oak_sys::oak_kick(ptr) };
    match rc {
        1 => Ok(true),
        0 => Ok(false),
        _ => {
            let e = unsafe { CStr::from_ptr(oak_sys::oak_last_error()) }
                .to_string_lossy()
                .into_owned();
            Err(format!("oak_kick failed: {e}").into())
        }
    }
}

/// See [`OakSource::video_tap`]. Holds a raw device handle; `Send` so it can live in the video thread.
pub struct VideoTap {
    dev: *mut oak_sys::oak_device,
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
        let rc = unsafe { oak_sys::oak_poll_video(self.dev, &mut data, &mut len, &mut ts) };
        if rc == 1 && !data.is_null() && len > 0 {
            let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec();
            Some((bytes, ts))
        } else {
            None
        }
    }
}

/// Build a zeroed, device-resident RGB888 kornia image of `w×h` (tight, 3 B/px) —
/// the layout kornia's `Preprocessor` and the `vrt` models consume, and the one the
/// OAK already hands us on the host, so uploads are a straight `memcpy_htod`.
///
/// Public so consumers that own their own upload (notably the stereo modality, which
/// deliberately takes no CUDA stream) allocate an identically-shaped destination
/// rather than restating the layout contract.
pub fn alloc_rgb_image(stream: &Arc<CudaStream>, w: u32, h: u32) -> Result<Image<u8, 3>, BoxError> {
    let slice = stream.alloc_zeros::<u8>(w as usize * h as usize * 3)?;
    let t = Tensor::from_cudaslice(slice, [h as usize, w as usize, 3], stream.clone());
    Image::try_from(t).map_err(|e| format!("build device Image<u8,3>: {e}").into())
}

impl Drop for OakSource {
    fn drop(&mut self) {
        unsafe { oak_sys::oak_close(self.dev) };
    }
}
