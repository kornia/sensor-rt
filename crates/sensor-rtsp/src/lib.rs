//! GStreamer RTSP source with Jetson NVMM hardware decode.
//!
//! Decodes via `nvv4l2decoder` and converts to RGBA in NVMM-backed memory, then
//! imports the DMA-BUF into CUDA and packs it into a tight RGB8 device image with
//! one on-GPU kernel (no host round-trip). For true zero-copy DMA-BUF sharing
//! (no pack), use [`RtspSource::next_nvmm`].
//!
//! ## Typical usage
//! ```no_run
//! use sensor_rtsp::RtspSource;
//!
//! // Share one CUDA stream with the model so the RGBA→RGB pack and inference
//! // are ordered. Each frame is a `Stamped<kornia_image::Image<u8, 3>>` (device).
//! let stream = vrt::Stream::new_standalone().unwrap().cuda_stream().clone();
//! let mut source = RtspSource::connect("rtsp://camera/stream", stream).unwrap();
//! while let Some(frame) = source.next_frame() {
//!     // let result = model.run(&frame.data).unwrap();  // frame.data: &Image<u8,3>
//! }
//! ```

pub mod mp4;

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::CudaStream;
use gstreamer::prelude::*;
use kornia_image::Image;
use kornia_tensor::{CudaKernel, Tensor};
use sensor_types::{FrameMeta, Stamped};

/// Errors from the GStreamer NVMM source.
#[derive(Debug, thiserror::Error)]
pub enum GstSourceError {
    #[error("GStreamer: {0}")]
    Glib(#[from] gstreamer::glib::Error),
    #[error("GStreamer state change: {0}")]
    StateChange(#[from] gstreamer::StateChangeError),
    #[error("pipeline setup: {0}")]
    Setup(&'static str),
    #[error("CUDA: {0}")]
    Cuda(String),
    #[error("stream ended before first frame — check RTSP URL and H.264 codec")]
    NoFirstFrame,
    #[error("cudaImportExternalMemory failed (fd={fd}, size={size}, err={code})")]
    NvmmImport { fd: i32, size: u64, code: i32 },
    #[error("invalid RTSP URL ({reason}): {url}")]
    BadUrl { reason: &'static str, url: String },
}

/// Redact userinfo (`user:pass@`) from an RTSP URL so credentials never reach
/// logs or error messages. Returns `scheme://host[:port]/path...` with the
/// `user:pass@` segment replaced by `***@` when present. Best-effort: if the
/// string doesn't look like a URL it is returned unchanged (it is never used
/// for anything but display).
fn redact_url(url: &str) -> String {
    // Split off scheme:// then look for userinfo before the first '/' of the
    // authority. userinfo is the segment up to and including the last '@' that
    // appears before the path.
    let (scheme, rest) = match url.split_once("://") {
        Some((s, r)) => (s, r),
        None => return url.to_string(),
    };
    // Authority ends at the first '/', '?' or '#'.
    let auth_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(auth_end);
    match authority.rsplit_once('@') {
        Some((_userinfo, hostport)) => format!("{scheme}://***@{hostport}{tail}"),
        None => url.to_string(),
    }
}

// ── CudaMemory ────────────────────────────────────────────────────────────────

/// RAII wrapper for a CUDA external-memory import of an NVMM DMA-BUF.
///
/// Calls `nvbuf_cuda_release` on drop.  Drop only after syncing any CUDA
/// stream that has used `dev_ptr`.
pub struct CudaMemory {
    pub dev_ptr: *mut c_void,
    ext_mem: *mut c_void,
}

unsafe impl Send for CudaMemory {}

impl Drop for CudaMemory {
    fn drop(&mut self) {
        unsafe {
            nvbuf_sys::nvbuf_cuda_release(self.ext_mem, self.dev_ptr);
        }
    }
}

// ── NvmmFrame ─────────────────────────────────────────────────────────────────

/// A single decoded NVMM RGBA frame from an RTSP stream.
///
/// `_keep_alive` holds whatever value must live for as long as `fd` is in use
/// (typically a GStreamer `Sample`) — erased to avoid a gstreamer dep on callers.
pub struct NvmmFrame {
    _keep_alive: Box<dyn Send + Sync + 'static>,
    /// DMA-BUF file descriptor — valid for the lifetime of this frame.
    pub fd: i32,
    /// Row pitch in bytes.
    pub pitch: u32,
    /// Total NVMM allocation size in bytes, required for `cudaImportExternalMemory`.
    pub size: u64,
    /// Capture presentation timestamp in nanoseconds (the GStreamer buffer PTS),
    /// if the decoder provided one.
    pub pts_ns: Option<u64>,
}

impl NvmmFrame {
    pub fn new(
        keep_alive: impl Send + Sync + 'static,
        fd: i32,
        pitch: u32,
        size: u64,
        pts_ns: Option<u64>,
    ) -> Self {
        Self {
            _keep_alive: Box::new(keep_alive),
            fd,
            pitch,
            size,
            pts_ns,
        }
    }

    /// Import this NVMM buffer into CUDA device memory.
    ///
    /// # Safety
    /// Calls `nvbuf_cuda_import`.  `self` must remain alive for the duration
    /// of any CUDA work using the returned `dev_ptr`.
    pub unsafe fn cuda_memory(&self) -> Result<CudaMemory, GstSourceError> {
        let mut ext_mem: *mut c_void = std::ptr::null_mut();
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let rc = nvbuf_sys::nvbuf_cuda_import(self.fd, self.size, &mut ext_mem, &mut dev_ptr);
        if rc != 0 {
            return Err(GstSourceError::NvmmImport {
                fd: self.fd,
                size: self.size,
                code: rc,
            });
        }
        Ok(CudaMemory { dev_ptr, ext_mem })
    }
}

/// A CPU RGBA snapshot for visualization: `(rgba_bytes, width, height)`.
pub type CpuFrame = (Vec<u8>, u32, u32);

// ── RtspSource ────────────────────────────────────────────────────────────────

/// RTSP source that delivers frames as NVMM RGBA using Jetson hardware decode.
///
/// # Pipeline
/// ```text
/// rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvidconv
///         → video/x-raw(memory:NVMM),format=RGBA → tee
///              ├→ appsink(NVMM)   [main inference path]
///              └→ nvvidconv → video/x-raw,format=RGBA → appsink(CPU)  [viz snapshot]
/// ```
pub struct RtspSource {
    pipeline: gstreamer::Pipeline,
    rx: mpsc::Receiver<NvmmFrame>,
    width: u32,
    height: u32,
    /// Stamped onto each frame's [`FrameMeta::source_id`] for multi-camera setups.
    source_id: u32,
    /// Latest CPU RGBA frame for visualization.  Updated asynchronously by GStreamer;
    /// take with `latest_cpu_frame()` and lock to read.
    cpu_frame: Arc<Mutex<Option<CpuFrame>>>,
    /// Shared CUDA stream the per-frame RGBA→RGB pack runs on (see [`RtspSource::cuda_stream`]).
    stream: Arc<CudaStream>,
    /// JIT kernel that packs the pitched NVMM RGBA import into a tight RGB8 image.
    pack: CudaKernel,
}

// Pack the decoder's NVMM RGBA (row-pitched, 4 B/px) into a tightly-packed RGB8
// image (3 B/px, stride = w*3) — the layout kornia's Preprocessor + the vrt models
// require. There is no zero-copy path (kornia is tight-RGB8-only), but this stays
// entirely on the GPU (no host round-trip).
const PACK_SRC: &str = r#"
extern "C" __global__ void rgba_pitch_to_rgb(
    const unsigned char* __restrict__ src, int src_pitch,
    unsigned char* __restrict__ dst, int w, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int x = i % w;
    int y = i / w;
    const unsigned char* s = src + y * src_pitch + x * 4; // RGBA, row-pitched
    dst[i * 3 + 0] = s[0];
    dst[i * 3 + 1] = s[1];
    dst[i * 3 + 2] = s[2];
}
"#;

/// Build a zeroed, device-resident tight RGB888 kornia image of `w×h`.
fn alloc_rgb_image(
    stream: &Arc<CudaStream>,
    w: u32,
    h: u32,
) -> Result<Image<u8, 3>, GstSourceError> {
    let slice = stream
        .alloc_zeros::<u8>(w as usize * h as usize * 3)
        .map_err(|e| GstSourceError::Cuda(format!("alloc rgb image: {e}")))?;
    let t = Tensor::from_cudaslice(slice, [h as usize, w as usize, 3], stream.clone());
    Image::try_from(t).map_err(|e| GstSourceError::Cuda(format!("build device Image<u8,3>: {e}")))
}

impl RtspSource {
    /// Open an RTSP stream and block until the first frame arrives.
    ///
    /// Frames are delivered at the camera's native resolution.
    /// Use [`connect_resized`] to have the VIC scaler downsize before CUDA.
    ///
    /// [`connect_resized`]: RtspSource::connect_resized
    pub fn connect(url: &str, stream: Arc<CudaStream>) -> Result<Self, GstSourceError> {
        Self::connect_internal(url, None, stream)
    }

    /// Open an RTSP stream and resize every frame to `(width, height)` in
    /// the GStreamer pipeline before it reaches CUDA.
    ///
    /// The resize is done by the Jetson VIC hardware scaler inside `nvvidconv`,
    /// so it costs no CUDA or CPU cycles.  `source.width()` / `source.height()`
    /// return the resized dimensions.
    pub fn connect_resized(
        url: &str,
        width: u32,
        height: u32,
        stream: Arc<CudaStream>,
    ) -> Result<Self, GstSourceError> {
        Self::connect_internal(url, Some((width, height)), stream)
    }

    fn connect_internal(
        url: &str,
        resize: Option<(u32, u32)>,
        stream: Arc<CudaStream>,
    ) -> Result<Self, GstSourceError> {
        gstreamer::init()?;

        // Reject anything that is not a well-formed rtsp(s) URL before it goes
        // anywhere near pipeline construction. The URL is NEVER interpolated
        // into a parsed pipeline string (see below) — it is passed to `rtspsrc`
        // as a typed `location` property — but we still validate so a malformed
        // URL fails fast with a typed error instead of inside GStreamer.
        if !(url.starts_with("rtsp://") || url.starts_with("rtsps://")) {
            return Err(GstSourceError::BadUrl {
                reason: "must start with rtsp:// or rtsps://",
                url: redact_url(url),
            });
        }
        if url
            .bytes()
            .any(|b| b.is_ascii_whitespace() || b.is_ascii_control())
        {
            return Err(GstSourceError::BadUrl {
                reason: "contains whitespace or control characters",
                url: redact_url(url),
            });
        }

        // Optional VIC resize: add width/height to nvvidconv output caps.
        // Two appsinks via tee: NVMM path for inference, CPU path for visualization.
        // leaky=upstream on both queues: if a branch falls behind it drops new frames
        // rather than blocking the decoder.
        let nvmm_caps = match resize {
            None => "video/x-raw(memory:NVMM),format=RGBA".to_string(),
            Some((w, h)) => format!("video/x-raw(memory:NVMM),format=RGBA,width={w},height={h}"),
        };

        // Build the downstream chain WITHOUT the URL: everything after `rtspsrc`
        // is static DSL with no user-controlled substring, so it cannot be used
        // to inject or redirect elements. `rtspsrc` itself is created as a real
        // element and the URL is set as a typed `location` property — it is
        // never parsed as pipeline DSL, fully removing the injection surface.
        let bin_str = format!(
            "rtph264depay name=depay ! h264parse ! \
             nvv4l2decoder enable-max-performance=1 disable-dpb=true ! \
             nvvidconv ! {nvmm_caps} ! tee name=t \
             t. ! queue max-size-buffers=2 leaky=upstream ! \
                  appsink name=sink max-buffers=1 drop=true sync=false \
             t. ! queue max-size-buffers=2 leaky=upstream ! \
                  nvvidconv ! video/x-raw,format=RGBA ! \
                  appsink name=sink_cpu max-buffers=1 drop=true sync=false"
        );

        let pipeline = gstreamer::Pipeline::default();

        // Force TCP-interleaved RTP (protocols=tcp). rtspsrc defaults to UDP-first (protocols=0x7);
        // on many consumer IP cameras (e.g. TP-Link Tapo) the UDP RTP sockets error out / never
        // receive — the RTP return path is blocked or the ports aren't reachable — so rtspsrc sits in
        // "doing receive with timeout" and the first frame never arrives, hanging `connect`'s initial
        // pull_sample forever. TCP interleaved carries RTP over the already-open RTSP TCP connection:
        // reliable (no packet loss on large keyframes), NAT/firewall-proof, and fine for LAN bitrates.
        let src = gstreamer::ElementFactory::make("rtspsrc")
            .property("location", url)
            .property("latency", 100u32)
            .property_from_str("protocols", "tcp")
            .build()
            .map_err(|_| GstSourceError::Setup("failed to create rtspsrc"))?;

        // Set up ONLY the video stream. A Tapo (and many IP cameras) also advertises an audio track
        // (e.g. PCMA); rtspsrc would expose a source pad for it that nothing downstream links, and the
        // unlinked pad tears down the whole pipeline ("streaming stopped, reason not-linked"). The
        // `select-stream` signal returns false to skip any RTP stream whose caps say media != "video",
        // so audio never enters the pipeline. Media unknown at select time → keep it (the pad-added
        // handler still links only a video pad, and falls back to the first pad if media is absent).
        src.connect("select-stream", false, |vals| {
            let keep = vals
                .get(2)
                .and_then(|v| v.get::<gstreamer::Caps>().ok())
                .and_then(|caps| {
                    caps.structure(0)
                        .and_then(|s| s.get::<String>("media").ok())
                })
                .is_none_or(|media| media == "video");
            Some(keep.to_value())
        });

        // Parse the static downstream chain into a bin. `ghost_unlinked_pads = TRUE` is REQUIRED: it
        // exposes `depay`'s (only) unlinked pad — its sink — as a GHOST pad named "sink" on the bin.
        // rtspsrc's dynamically-added src pad must link to that ghost pad, NOT to `depay`'s inner pad:
        // rtspsrc lives in the pipeline and depay lives inside this bin, so a direct pad link across the
        // bin boundary fails with "wrong hierarchy" and the stream never starts (reason not-linked).
        let bin = gstreamer::parse_bin_from_description(&bin_str, true)?;

        pipeline
            .add(&src)
            .map_err(|_| GstSourceError::Setup("add rtspsrc"))?;
        pipeline
            .add(&bin)
            .map_err(|_| GstSourceError::Setup("add bin"))?;

        // `rtspsrc` exposes its source pad only once the stream is described, so
        // link it to the depayloader on pad-added (same dynamic linking the old
        // `!` did implicitly).
        let bin_weak = bin.downgrade();
        src.connect_pad_added(move |_src, src_pad| {
            let Some(bin) = bin_weak.upgrade() else {
                return;
            };
            // The bin's ghost "sink" pad (proxying depay's sink) — link to THIS, not depay's inner pad,
            // or the cross-bin link fails "wrong hierarchy".
            let Some(sink_pad) = bin.static_pad("sink") else {
                return;
            };
            if sink_pad.is_linked() {
                return;
            }
            // Only link the video stream — ignore any audio/other media pads.
            // RTP pads carry caps `application/x-rtp, media=(string)video|audio`.
            // If media is absent (some servers describe it late), fall back to
            // linking, matching the old `!` which linked the first src pad.
            let is_video = src_pad
                .current_caps()
                .and_then(|c| c.structure(0).and_then(|s| s.get::<String>("media").ok()))
                .map(|m| m == "video")
                .unwrap_or(true);
            if is_video {
                let _ = src_pad.link(&sink_pad);
            }
        });

        // The appsinks live inside `bin` (the parsed downstream chain), so look
        // them up there — `Pipeline::by_name` does not recurse into child bins.
        let appsink = bin
            .by_name("sink")
            .ok_or(GstSourceError::Setup("no appsink"))?
            .dynamic_cast::<gstreamer_app::AppSink>()
            .map_err(|_| GstSourceError::Setup("element is not AppSink"))?;

        let appsink_cpu = bin
            .by_name("sink_cpu")
            .ok_or(GstSourceError::Setup("no sink_cpu"))?
            .dynamic_cast::<gstreamer_app::AppSink>()
            .map_err(|_| GstSourceError::Setup("sink_cpu is not AppSink"))?;

        let (frame_tx, frame_rx) = mpsc::sync_channel::<NvmmFrame>(2);
        let (dim_tx, dim_rx) = mpsc::sync_channel::<(u32, u32)>(1);
        let dims_sent = Arc::new(AtomicBool::new(false));

        appsink.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink
                        .pull_sample()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let buffer = sample.buffer().ok_or(gstreamer::FlowError::Error)?;
                    let caps = sample.caps().ok_or(gstreamer::FlowError::Error)?;
                    let st = caps.structure(0).ok_or(gstreamer::FlowError::Error)?;
                    let width = st
                        .get::<i32>("width")
                        .map_err(|_| gstreamer::FlowError::Error)?
                        as u32;
                    let height = st
                        .get::<i32>("height")
                        .map_err(|_| gstreamer::FlowError::Error)?
                        as u32;

                    let map = buffer
                        .map_readable()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let surf = map.as_slice().as_ptr() as *const c_void;

                    let fd = unsafe { nvbuf_sys::nvbuf_dmabuf_fd(surf) };
                    let pitch = unsafe { nvbuf_sys::nvbuf_pitch(surf) };
                    let size = unsafe { nvbuf_sys::nvbuf_data_size(surf) };
                    let layout = unsafe { nvbuf_sys::nvbuf_layout(surf) };
                    drop(map);

                    // Reject a malformed frame before it reaches the pack kernel:
                    // the kernel reads `y*pitch + x*4` per row, so a pitch below
                    // the minimum RGBA row stride (width*4) would read out of bounds.
                    if fd < 0 || size == 0 || layout != 0 || width == 0 || pitch < width * 4 {
                        return Err(gstreamer::FlowError::Error);
                    }

                    if !dims_sent.swap(true, Ordering::SeqCst) {
                        let _ = dim_tx.try_send((width, height));
                    }

                    // Capture PTS (camera/decoder timebase) before moving `sample`.
                    let pts_ns = buffer.pts().map(|t| t.nseconds());
                    let _ = frame_tx.try_send(NvmmFrame::new(sample, fd, pitch, size, pts_ns));
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        let cpu_frame: Arc<Mutex<Option<CpuFrame>>> = Arc::new(Mutex::new(None));
        let cpu_frame_cb = Arc::clone(&cpu_frame);

        appsink_cpu.set_callbacks(
            gstreamer_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink
                        .pull_sample()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let buffer = sample.buffer().ok_or(gstreamer::FlowError::Error)?;
                    let caps = sample.caps().ok_or(gstreamer::FlowError::Error)?;
                    let st = caps.structure(0).ok_or(gstreamer::FlowError::Error)?;
                    let w = st
                        .get::<i32>("width")
                        .map_err(|_| gstreamer::FlowError::Error)?
                        as u32;
                    let h = st
                        .get::<i32>("height")
                        .map_err(|_| gstreamer::FlowError::Error)?
                        as u32;
                    let map = buffer
                        .map_readable()
                        .map_err(|_| gstreamer::FlowError::Error)?;
                    let data = map.as_slice().to_vec();
                    drop(map);
                    if let Ok(mut g) = cpu_frame_cb.lock() {
                        *g = Some((data, w, h));
                    }
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        pipeline.set_state(gstreamer::State::Playing)?;

        let (width, height) = dim_rx.recv().map_err(|_| GstSourceError::NoFirstFrame)?;

        let pack = CudaKernel::compile(stream.context(), PACK_SRC, "rgba_pitch_to_rgb")
            .map_err(|e| GstSourceError::Cuda(format!("compile pack kernel: {e}")))?;

        Ok(Self {
            pipeline,
            rx: frame_rx,
            width,
            height,
            source_id: 0,
            cpu_frame,
            stream,
            pack,
        })
    }

    /// The shared CUDA stream the per-frame RGBA→RGB pack runs on. **Build the
    /// consuming model with this stream** so the pack and the model's inference
    /// are ordered on one stream (the frame's device image is valid after the
    /// pack, which `next_frame` syncs before returning).
    pub fn cuda_stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Set the `source_id` stamped onto every frame's [`FrameMeta`] (default `0`).
    /// Use distinct ids per camera so multi-source consumers can tell frames apart.
    pub fn with_source_id(mut self, id: u32) -> Self {
        self.source_id = id;
        self
    }

    /// Returns a shared handle to the latest CPU RGBA snapshot.
    ///
    /// Clone the `Arc` before moving the source into a `Pipeline`.  The GStreamer
    /// thread updates this every frame (overwriting old data); lock and `take()` to
    /// consume without holding the lock during PNG save.
    pub fn latest_cpu_frame(&self) -> Arc<Mutex<Option<CpuFrame>>> {
        Arc::clone(&self.cpu_frame)
    }
}

impl RtspSource {
    /// Pull the next decoded frame, import its NVMM DMA-BUF into CUDA, pack the
    /// pitched RGBA into a tight RGB8 device [`Image`], and return it [`Stamped`]
    /// with the capture timestamp (camera PTS) and `source_id`. The pack runs on
    /// [`RtspSource::cuda_stream`] and is synced before the transient NVMM import
    /// is released, so the returned image is self-owned and valid. Returns `None`
    /// at end-of-stream or on import/pack failure.
    pub fn next_frame(&mut self) -> Option<Stamped<Image<u8, 3>>> {
        let frame = self.rx.recv().ok()?;
        let mem = match unsafe { frame.cuda_memory() } {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[rtsp] NVMM import failed: {e}");
                return None;
            }
        };
        let (w, h) = (self.width, self.height);
        let meta = FrameMeta {
            pts_ns: frame.pts_ns, // camera capture timestamp
            source_id: Some(self.source_id),
            ..FrameMeta::default()
        };

        // Pack the pitched RGBA import → tight RGB8 device image on the shared stream.
        let mut img = alloc_rgb_image(&self.stream, w, h).ok()?;
        let src_raw = mem.dev_ptr as usize as CUdeviceptr;
        let (pitch_i, w_i, n) = (frame.pitch as i32, w as i32, (w * h) as i32);
        {
            let dst = img.as_cudaslice_mut()?;
            self.pack
                .launch_builder(&self.stream)
                .arg(&src_raw)
                .arg(&pitch_i)
                .arg(dst)
                .arg(&w_i)
                .arg(&n)
                .launch_1d(w * h)
                .ok()?;
        }
        // The kernel read the transient NVMM import; sync before it (and the
        // backing DMA-BUF frame) are dropped.
        self.stream.synchronize().ok()?;
        drop(mem);
        drop(frame);
        Some(Stamped::new(meta, img))
    }

    /// Pull the next decoded frame as a raw NVMM handle (DMA-BUF fd + metadata),
    /// WITHOUT importing it into CUDA. The returned [`NvmmFrame`] keeps the underlying
    /// GStreamer buffer (and thus `fd`) alive until it is dropped.
    ///
    /// Use this for zero-copy cross-process sharing: pass `fd` (with `size`) to another
    /// process via SCM_RIGHTS and let it `cudaImportExternalMemory` the same buffer. Hold
    /// the returned frame until consumers are done so the pool does not recycle it.
    pub fn next_nvmm(&mut self) -> Option<NvmmFrame> {
        self.rx.recv().ok()
    }
}

impl Drop for RtspSource {
    fn drop(&mut self) {
        let _ = self.pipeline.set_state(gstreamer::State::Null);
    }
}
