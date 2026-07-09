# sensor-rt

**RTSP camera source for Jetson Orin** — hardware H.264 decode over NVMM →
device-resident kornia `Image<u8,3>`. A plain producer, no orchestration
framework. The source has **no dependency on the algorithm libraries**: frame
provenance rides in a small vendored `stamp` type, and models consume these
frames from the application side.

## Workspace layout

Flat `crates/` + `examples/`. GPU image/tensor types come from `kornia-rs`
(pinned git dep, `cudarc` feature); cudarc is 0.19 across the graph to match
kornia-rs (shared `CudaStream`/`CudaSlice` for zero-copy interop with downstream
models).

| Crate | lib | Role |
|-------|-----|------|
| `crates/nvbuf-sys` | `nvbuf_sys` | FFI: NvBufSurface → CUDA device ptr from NVMM DMA-BUF (`links = nvbufsurface`) |
| `crates/sensor-rtsp` | `sensor_rtsp` | RTSP/H.264 source, NVMM → CUDA, emits device `Image<u8,3>` |

## Architecture

`RtspSource` is a plain **async** producer: `next_frame()` → owned `Frame` (holds
a device-resident tight-**RGB** `Image<u8,3>` + `FrameMeta`). The pipeline decodes
to NVMM RGBA, VIC-resizes, and a single NVMM appsink hands each DMA-BUF to CUDA;
one on-GPU kernel un-pitches **and drops alpha** (pitched-RGBA → tight RGB) into a
ring buffer, so the frame is model-ready. `next_frame` only **enqueues** that copy
on the shared stream — **no hidden sync** (VPI/TRT model): the caller runs its
model on the same stream and issues the single `synchronize()`. A **ring of
`POOL_CAP` buffers** lets frames pipeline (decode ∥ copy ∥ inference); the
transient NVMM imports are retired **lazily** via per-frame CUDA events
(`cudaEventQuery`, non-blocking) — never a per-frame host sync. `try_next()` is
non-blocking.

The dynamic `rtspsrc` pad is linked to the downstream bin via an **auto-ghosted**
sink pad (`parse_bin_from_description(.., true)`) — a direct cross-bin link
silently drops all frames. Non-video pads (e.g. a camera's audio) are drained to a
`fakesink` so an unlinked pad can't stall the pipeline.

## Hard constraints

- **GStreamer + libnvbufsurface are system/JetPack** (build + runtime) — NOT conda.
- **Build cap `-j2`** (`CARGO_BUILD_JOBS=2`) — parallel native builds OOM the
  7.4 GB Orin.
- Native, Jetson-only: NVMM decode needs real CUDA + the Jetson Multimedia API
  headers (`/usr/src/jetson_multimedia_api`).

## Commands

```bash
export CARGO_BUILD_JOBS=2
cargo build -j2
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
```
