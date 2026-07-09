//! GStreamer RTSP source with Jetson NVMM hardware decode — **async, no hidden
//! sync**.
//!
//! Decodes via `nvv4l2decoder`, resizes with the VIC scaler (`nvvidconv`) to
//! RGBA in NVMM-backed memory, imports the DMA-BUF into CUDA, and copies the
//! hardware-pitched RGBA into a tight, device-resident `Image<u8, 3>` (RGB, alpha
//! dropped) with one
//! on-GPU kernel. [`RtspSource::next_frame`] only **enqueues** that copy on the
//! shared stream and returns an owned [`Frame`] — it never calls
//! `cudaStreamSynchronize`. The **caller owns the single sync** (VPI / TensorRT
//! model): run your model on the same stream, then sync once, then read.
//!
//! Frames come from a small **ring of device buffers**, so several can be in
//! flight (decode ∥ copy ∥ inference overlap). The transient NVMM imports are
//! reclaimed **lazily** via per-frame CUDA events (`cudaEventQuery`), never a
//! blocking sync on the hot path. For true zero-copy DMA-BUF sharing (no copy),
//! use [`RtspSource::next_nvmm`].
//!
//! ## Typical usage
//! ```no_run
//! use sensor_rtsp::RtspSource;
//! use cudarc::driver::CudaContext;
//!
//! let stream = CudaContext::new(0).unwrap().default_stream();
//! let mut source = RtspSource::connect("rtsp://camera/stream", stream).unwrap();
//! while let Some(frame) = source.next_frame() {
//!     // frame.image(): &Image<u8,3> (device RGB)  ·  frame.meta: FrameMeta
//!     // ... enqueue your model on the SAME stream (no sync here) ...
//!     stream.synchronize().unwrap(); // the caller's single sync
//! }
//! ```

pub mod stamp;

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use crate::stamp::FrameMeta;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaEvent, CudaStream};
use gstreamer::prelude::*;
use kornia_image::Image;
use kornia_tensor::{CudaKernel, Tensor};

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

// ── RtspSource ────────────────────────────────────────────────────────────────

/// RTSP source that delivers frames as device-resident RGBA using Jetson
/// hardware decode.
///
/// # Pipeline
/// ```text
/// rtspsrc → rtph264depay → h264parse → nvv4l2decoder → nvvidconv
///         → video/x-raw(memory:NVMM),format=RGBA → appsink(NVMM)
/// ```
pub struct RtspSource {
    pipeline: gstreamer::Pipeline,
    rx: mpsc::Receiver<NvmmFrame>,
    width: u32,
    height: u32,
    /// Stamped onto each frame's [`FrameMeta::source_id`] for multi-camera setups.
    source_id: u32,
    /// Shared CUDA stream the per-frame pitched→tight copy runs on (see
    /// [`RtspSource::cuda_stream`]).
    stream: Arc<CudaStream>,
    /// JIT kernel that copies the pitched NVMM RGBA import into a tight RGB image.
    pack: CudaKernel,
    /// Ring of device output buffers (see [`POOL_CAP`]); a [`Frame`] checks one
    /// out and returns it on drop — so several frames can be in flight.
    pool: Arc<Mutex<BufPool>>,
    /// NVMM imports awaiting their on-stream copy to finish; drained lazily by
    /// [`RtspSource::retire`] via a non-blocking event query (no host sync).
    inflight: VecDeque<InFlight>,
}

// Un-pitch + drop alpha in one pass: the decoder's NVMM RGBA (row-pitched,
// 4 B/px) → a tightly-packed **RGB** device image (stride = w*3). Fuses what used
// to be two device passes (un-pitch to RGBA, then RGBA→RGB) into one, so the
// source emits model-ready `Image<u8,3>`. Stays entirely on the GPU.
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
/// Allocated once per ring slot and reused for every frame's un-pitch copy.
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

/// Number of device output buffers in the ring — how many frames may be in
/// flight (checked out) at once before `next_frame`/`try_next` drop new frames.
const POOL_CAP: usize = 4;

/// A ring of reusable tight-RGB device buffers. A frame checks one out; it
/// returns to the ring when the [`Frame`] is dropped.
struct BufPool {
    free: Vec<Image<u8, 3>>,
    stream: Arc<CudaStream>,
    w: u32,
    h: u32,
    allocated: usize,
}

impl BufPool {
    fn checkout(&mut self) -> Option<Image<u8, 3>> {
        if let Some(b) = self.free.pop() {
            return Some(b);
        }
        if self.allocated < POOL_CAP {
            let b = alloc_rgb_image(&self.stream, self.w, self.h).ok()?;
            self.allocated += 1;
            return Some(b);
        }
        None // ring exhausted — all buffers are checked out by live Frames
    }

    fn checkin(&mut self, b: Image<u8, 3>) {
        self.free.push(b);
    }
}

/// A transient NVMM import + its GStreamer sample, held until the on-stream copy
/// that reads it has completed (tracked by `event`). Reclaimed lazily by
/// [`RtspSource::retire`] via a non-blocking `cudaEventQuery` — never a sync.
struct InFlight {
    _mem: CudaMemory, // released (cudaDestroyExternalMemory) on drop — declared first
    _nvmm: NvmmFrame, // keeps the GStreamer buffer / DMA-BUF fd alive
    event: Arc<CudaEvent>,
}

/// One decoded frame: an owned handle to a device-resident tight-RGB image
/// (drawn from the source's ring) plus provenance. Dropping it returns the
/// buffer to the ring. The pixel data is valid once the shared stream has passed
/// this frame's copy — read it after your caller-owned `stream.synchronize()`
/// (or wait [`Frame::event`] on another stream).
pub struct Frame {
    image: Option<Image<u8, 3>>,
    /// Frame provenance: sequence, capture PTS, source id.
    pub meta: FrameMeta,
    done: Arc<CudaEvent>,
    pool: Arc<Mutex<BufPool>>,
}

impl Frame {
    /// The device-resident tight-RGB image (valid after the caller's sync).
    pub fn image(&self) -> &Image<u8, 3> {
        self.image.as_ref().expect("frame image present until drop")
    }

    /// CUDA event recorded right after this frame's copy — wait it on another
    /// stream (`other.wait(frame.event())`) to order GPU work without a host sync.
    ///
    /// **Cross-stream caveat:** dropping the `Frame` immediately returns its buffer
    /// to the ring, where the next frame's copy (on the source stream) may
    /// overwrite it. If you read the image from a *different* stream, that stream's
    /// reads must be complete before the drop — sync it (or hold the `Frame`) first,
    /// or the recycle races your read. Same-stream consumers are always safe (stream
    /// order serialises the recycle behind their work).
    pub fn event(&self) -> &CudaEvent {
        &self.done
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        // Buffer returns to the ring. Safe for same-stream consumers (stream order
        // serialises the next copy behind their reads); cross-stream readers must
        // finish first — see [`Frame::event`].
        if let Some(img) = self.image.take() {
            if let Ok(mut pool) = self.pool.lock() {
                pool.checkin(img);
            }
        }
    }
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
        // Single NVMM appsink. leaky=upstream on the queue: if the consumer falls
        // behind it drops new frames rather than blocking the decoder.
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
             nvvidconv ! {nvmm_caps} ! \
             queue max-size-buffers=2 leaky=upstream ! \
             appsink name=sink max-buffers=1 drop=true sync=false"
        );

        let pipeline = gstreamer::Pipeline::default();

        let src = gstreamer::ElementFactory::make("rtspsrc")
            .property("location", url)
            .property("latency", 100u32)
            .build()
            .map_err(|_| GstSourceError::Setup("failed to create rtspsrc"))?;

        // Parse the static downstream chain into a bin. `depay`'s sink is the one
        // unlinked pad; a direct cross-bin-boundary link from `rtspsrc` does NOT
        // flow data, so `true` **auto-ghosts** that sink onto the bin (named after
        // its template → `"sink"`) and rtspsrc links to `bin.static_pad("sink")`.
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
        let pipeline_weak = pipeline.downgrade();
        src.connect_pad_added(move |_src, src_pad| {
            // RTP pads carry caps `application/x-rtp, media=(string)video|audio`.
            // If media is absent (some servers describe it late), treat as video —
            // matching the old `!` which linked the first src pad.
            let is_video = src_pad
                .current_caps()
                .and_then(|c| c.structure(0).and_then(|s| s.get::<String>("media").ok()))
                .map(|m| m == "video")
                .unwrap_or(true);

            // Link the first video stream to the bin's auto-ghosted sink pad.
            // Only bail out on a SUCCESSFUL link — if the link fails (e.g. a
            // non-H.264 codec the depayloader rejects) fall through and drain the
            // pad, else it stays unlinked and stalls the whole pipeline.
            if is_video {
                if let Some(sink_pad) = bin_weak.upgrade().and_then(|b| b.static_pad("sink")) {
                    if !sink_pad.is_linked() && src_pad.link(&sink_pad).is_ok() {
                        return;
                    }
                }
            }

            // Any other pad (audio, a second stream, or a video pad that wouldn't
            // link) MUST still be drained: an unlinked `rtspsrc` pad raises
            // "not-linked" and stalls the WHOLE pipeline, so no frames ever reach
            // the appsink. Sink it to a fakesink.
            if let Some(pipeline) = pipeline_weak.upgrade() {
                if let Ok(fake) = gstreamer::ElementFactory::make("fakesink")
                    .property("sync", false)
                    .property("async", false)
                    .build()
                {
                    if pipeline.add(&fake).is_ok() {
                        let _ = fake.sync_state_with_parent();
                        if let Some(fsink) = fake.static_pad("sink") {
                            let _ = src_pad.link(&fsink);
                        }
                    }
                }
            }
        });

        // The appsinks live inside `bin` (the parsed downstream chain), so look
        // them up there — `Pipeline::by_name` does not recurse into child bins.
        let appsink = bin
            .by_name("sink")
            .ok_or(GstSourceError::Setup("no appsink"))?
            .dynamic_cast::<gstreamer_app::AppSink>()
            .map_err(|_| GstSourceError::Setup("element is not AppSink"))?;

        let (frame_tx, frame_rx) = mpsc::sync_channel::<NvmmFrame>(2);
        let (dim_tx, dim_rx) = mpsc::sync_channel::<(u32, u32)>(1);
        // Packed (width<<32 | height) of the first valid frame; 0 = unset.
        let first_dims = Arc::new(AtomicU64::new(0));

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

                    // The first valid frame fixes the working dimensions (the pool
                    // + kernel are sized to them). Reject a later frame with
                    // different dims (mid-stream renegotiation under plain
                    // `connect()`) — feeding it would over-index the pack kernel.
                    let packed = ((width as u64) << 32) | height as u64;
                    match first_dims.compare_exchange(0, packed, Ordering::SeqCst, Ordering::SeqCst)
                    {
                        Ok(_) => {
                            let _ = dim_tx.try_send((width, height)); // first → publish
                        }
                        Err(prev) if prev != packed => return Err(gstreamer::FlowError::Error),
                        Err(_) => {} // same dims — ok
                    }

                    // Capture PTS (camera/decoder timebase) before moving `sample`.
                    let pts_ns = buffer.pts().map(|t| t.nseconds());
                    let _ = frame_tx.try_send(NvmmFrame::new(sample, fd, pitch, size, pts_ns));
                    Ok(gstreamer::FlowSuccess::Ok)
                })
                .build(),
        );

        pipeline.set_state(gstreamer::State::Playing)?;

        // Bounded wait: a dead/misbehaving endpoint (or a stream that never yields a
        // valid video frame) must fail with `NoFirstFrame`, not hang the caller.
        let (width, height) = dim_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .map_err(|_| GstSourceError::NoFirstFrame)?;

        let pack = CudaKernel::compile(stream.context(), PACK_SRC, "rgba_pitch_to_rgb")
            .map_err(|e| GstSourceError::Cuda(format!("compile pack kernel: {e}")))?;

        // Ring of tight-RGB output buffers, grown lazily up to POOL_CAP.
        let pool = Arc::new(Mutex::new(BufPool {
            free: Vec::new(),
            stream: stream.clone(),
            w: width,
            h: height,
            allocated: 0,
        }));

        Ok(Self {
            pipeline,
            rx: frame_rx,
            width,
            height,
            source_id: 0,
            stream,
            pack,
            pool,
            inflight: VecDeque::new(),
        })
    }

    /// The shared CUDA stream the per-frame pitched→tight copy is enqueued on.
    /// **Build the consuming model with this stream** so the copy and the model's
    /// inference are ordered on one stream — then a single `stream.synchronize()`
    /// completes both. `next_frame` never syncs it for you.
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
}

impl RtspSource {
    /// Pull the next decoded frame (**blocking** until one arrives), import its
    /// NVMM DMA-BUF into CUDA, and **enqueue** the pitched→tight RGB copy on the
    /// shared stream into a ring buffer. Returns an owned [`Frame`] — **no sync**.
    /// Run your model on [`RtspSource::cuda_stream`], then issue the single
    /// `stream.synchronize()` yourself before reading the pixels. Returns `None`
    /// at end-of-stream. If the ring is momentarily exhausted (all buffers held
    /// by live `Frame`s) the incoming frame is dropped and the next awaited.
    pub fn next_frame(&mut self) -> Option<Frame> {
        loop {
            self.retire();
            let nvmm = self.rx.recv().ok()?;
            match self.acquire(nvmm) {
                Some(frame) => return Some(frame),
                None => continue, // ring full or import failed — drop, await next
            }
        }
    }

    /// Non-blocking variant: return the next frame if one is already decoded,
    /// else `None`. Same async contract as [`next_frame`](Self::next_frame).
    pub fn try_next(&mut self) -> Option<Frame> {
        self.retire();
        let nvmm = self.rx.try_recv().ok()?;
        self.acquire(nvmm)
    }

    /// Return a checked-out ring buffer to the pool (used on `acquire` error paths
    /// so a transient CUDA failure can't permanently shrink the ring).
    fn recycle(&self, img: Image<u8, 3>) {
        if let Ok(mut pool) = self.pool.lock() {
            pool.checkin(img);
        }
    }

    /// Import + enqueue the copy for one NVMM frame into a ring buffer, record a
    /// completion event, and stash the import for lazy retire. `None` if the ring
    /// is exhausted or a CUDA op fails — in which case the buffer is returned to
    /// the ring and (once a kernel is enqueued) the stream is synced before the
    /// NVMM import is released.
    fn acquire(&mut self, nvmm: NvmmFrame) -> Option<Frame> {
        let mut img = self.pool.lock().ok()?.checkout()?; // None → ring full

        let mem = match unsafe { nvmm.cuda_memory() } {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[rtsp] NVMM import failed: {e}");
                self.recycle(img);
                return None;
            }
        };

        // Completion event, created BEFORE the launch: a failure here has enqueued
        // nothing, so `mem` (dropped on return) is not yet read by any kernel.
        let event = match self.stream.context().new_event(None) {
            Ok(e) => Arc::new(e),
            Err(e) => {
                eprintln!("[rtsp] cuda event create failed: {e}");
                self.recycle(img);
                return None;
            }
        };

        // Enqueue the pitched RGBA import → tight-RGB copy (async, no sync).
        let (w, h) = (self.width, self.height);
        let src_raw = mem.dev_ptr as usize as CUdeviceptr;
        let (pitch_i, w_i, n) = (nvmm.pitch as i32, w as i32, (w * h) as i32);
        let launched = img.as_cudaslice_mut().is_some_and(|dst| {
            self.pack
                .launch_builder(&self.stream)
                .arg(&src_raw)
                .arg(&pitch_i)
                .arg(dst)
                .arg(&w_i)
                .arg(&n)
                .launch_1d(w * h)
                .is_ok()
        });
        if !launched {
            eprintln!("[rtsp] pack kernel launch failed");
            self.recycle(img); // launch failed → kernel never ran → `mem` safe to drop
            return None;
        }

        // Record after the launch. If it fails, the kernel is already enqueued and
        // reading `mem`, so sync before `mem` drops to avoid a GPU use-after-free.
        if event.record(&self.stream).is_err() {
            eprintln!("[rtsp] cuda event record failed");
            let _ = self.stream.synchronize();
            self.recycle(img);
            return None;
        }

        let meta = FrameMeta {
            pts_ns: nvmm.pts_ns, // camera capture timestamp
            source_id: Some(self.source_id),
            ..FrameMeta::default()
        };
        self.inflight.push_back(InFlight {
            _mem: mem,
            _nvmm: nvmm,
            event: event.clone(),
        });
        Some(Frame {
            image: Some(img),
            meta,
            done: event,
            pool: self.pool.clone(),
        })
    }

    /// Reclaim NVMM imports whose copy has finished — **non-blocking**. Events
    /// complete in submission order on the single stream, so drain from the front
    /// while `cudaEventQuery` reports ready. Called at the top of every acquire.
    fn retire(&mut self) {
        use cudarc::driver::sys::CUresult;
        while let Some(front) = self.inflight.front() {
            match unsafe { cudarc::driver::result::event::query(front.event.cu_event()) } {
                // Copy done → drop _mem (release import) then _nvmm (unref sample).
                Ok(()) => {
                    self.inflight.pop_front();
                }
                // Still running (in submission order on one stream) → stop draining.
                Err(e) if e.0 == CUresult::CUDA_ERROR_NOT_READY => break,
                // A real query error will never clear; releasing the import now beats
                // an unbounded in-flight queue leaking DMA-BUF fds / device memory.
                Err(e) => {
                    eprintln!("[rtsp] event query error, releasing import: {e:?}");
                    self.inflight.pop_front();
                }
            }
        }
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
        // Ensure any in-flight copy has finished before the NVMM imports are
        // released (this is teardown, not the hot path — a blocking sync is fine).
        let _ = self.stream.synchronize();
        self.inflight.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::redact_url;
    use super::stamp::{FrameMeta, Stamped};

    #[test]
    fn redact_strips_credentials() {
        // userinfo removed, host/port/path kept
        assert_eq!(
            redact_url("rtsp://user:pass@192.168.1.147:554/stream1"),
            "rtsp://***@192.168.1.147:554/stream1"
        );
        // no credentials → unchanged
        assert_eq!(redact_url("rtsp://host:554/s"), "rtsp://host:554/s");
        // userinfo with no port/path
        assert_eq!(redact_url("rtsp://u:p@host"), "rtsp://***@host");
        // rtsps scheme
        assert_eq!(redact_url("rtsps://a:b@h/p"), "rtsps://***@h/p");
        // an '@' in the path must NOT be mistaken for userinfo
        assert_eq!(redact_url("rtsp://host/pa@th"), "rtsp://host/pa@th");
        // not a URL → returned as-is (never used except for display)
        assert_eq!(redact_url("garbage"), "garbage");
    }

    #[test]
    fn stamp_carries_and_maps_metadata() {
        let meta = FrameMeta {
            seq: 7,
            pts_ns: Some(42),
            source_id: Some(3),
        };
        let s: Stamped<&str> = meta.stamp("payload");
        assert_eq!(s.meta.seq, 7);
        assert_eq!(s.meta.pts_ns, Some(42));
        assert_eq!(s.data, "payload");

        // map transforms the value, keeps the metadata
        let mapped = s.map(|d| d.len());
        assert_eq!(mapped.data, 7);
        assert_eq!(mapped.meta.source_id, Some(3));
    }
}
