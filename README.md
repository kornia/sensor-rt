# sensor-rt

**RTSP camera source for NVIDIA Jetson Orin, written in Rust.**

Decodes RTSP/H.264 in hardware over NVMM (`nvv4l2decoder`), imports the DMA-BUF
into CUDA, and emits a **device-resident kornia `Image<u8,3>`** — ready to hand
to any GPU vision model. It's a plain producer: `next_frame()` in a loop, no
orchestration framework.

The source has **no algorithm dependency** — frame provenance travels in a small
vendored `stamp` type, so nothing here pulls in the model crates. Models
(e.g. [`vision-rt`](https://github.com/kornia/vision-rt)) consume these frames
from the application side.

**Target platform:** Jetson Orin (aarch64, SM87), JetPack 6.x, CUDA 12.6.

## Workspace

| Crate | Role |
|-------|------|
| `crates/nvbuf-sys` | FFI: Jetson `NvBufSurface` → CUDA device ptr from an NVMM DMA-BUF (`links = nvbufsurface`) |
| `crates/sensor-rtsp` | RTSP/H.264 source, NVMM → CUDA, emits a device `Image<u8,3>` (GStreamer) |

Examples live inside the `sensor-rtsp` crate (`crates/sensor-rtsp/examples/`):

| Example | Role |
|---------|------|
| `rtsp_grab` | Minimal, **vrt-free**: connect, grab a few frames, save one PNG |
| `rtsp_rfdetr` | RTSP → **RF-DETR** detection, device end-to-end (`source → model`, one sync/frame; dev-dep on the private [`vrt-rfdetr`](https://github.com/kornia/vision-rt)) |

## Usage

The API is **async — the caller owns the single sync** (VPI / TensorRT model):
`next_frame` only *enqueues* the NVMM→CUDA copy on the shared stream and returns
an owned `Frame`; you run your model on that same stream and sync once.

```rust
use cudarc::driver::CudaContext;
use sensor_rtsp::RtspSource;

let stream = CudaContext::new(0)?.default_stream();
let mut source = RtspSource::connect("rtsp://camera/stream", stream.clone())?;

while let Some(frame) = source.next_frame() {   // enqueues copy, NO sync
    // frame.image(): &Image<u8,3> (device RGB, model-ready)  ·  frame.meta: FrameMeta
    // ... enqueue your model on the SAME stream ...
    stream.synchronize()?;                       // the caller's one sync
    // ... read results ...
}                                                // Frame drops → buffer back to ring
```

`connect_resized(url, w, h, stream)` resizes on the Jetson VIC hardware scaler (no
CUDA/CPU cost). Each frame is imported from NVMM and copied — pitched RGBA → tight
**RGB** (alpha dropped in the same pass) — by one on-GPU kernel; **no hidden
`cudaStreamSynchronize`**. Frames come from a small **ring of device buffers** so
several can be in flight (decode ∥ copy ∥ inference); the transient NVMM imports
are reclaimed lazily via per-frame CUDA events (`cudaEventQuery`). `try_next()` is
the non-blocking variant. The output is a model-ready tight-RGB `Image<u8,3>` —
feed it straight to a detector on the same stream.

## Building

Native, **Jetson-only** — GStreamer + `libnvbufsurface` are system/JetPack (build
and runtime), not conda:

```bash
export CARGO_BUILD_JOBS=2               # 7.4 GB Orin OOMs on parallel native builds
cargo build -j2                         # library + nvbuf-sys (vrt-free)
cargo run --release -p sensor-rtsp --example rtsp_grab -- rtsp://<camera>/stream out.png

# rtsp_rfdetr is gated behind the `rfdetr-example` feature — it pulls the private
# vrt-rfdetr dep + its prebuilt HF engine (so default builds/tests stay vrt-free):
export CARGO_NET_GIT_FETCH_WITH_CLI=true
cargo run --release -p sensor-rtsp --example rtsp_rfdetr \
    --features rfdetr-example -- rtsp://<camera>/stream 0.5
```

CI on a hosted runner: `cargo fmt --all --check` + the pure-logic **unit tests**
(`NVBUF_STUB=1 cargo test -p sensor-rtsp --lib` — the nvbuf FFI is stubbed so the
crate builds with no Jetson libs). The full build/clippy/test is gated on a
self-hosted Jetson runner (needs real CUDA + NVMM).

## License

Apache-2.0
