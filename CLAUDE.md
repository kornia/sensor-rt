# sensor-rt

Isolated **sensor drivers** for Jetson — RTSP/NVMM cameras and OAK-D depth —
extracted from vision-rt so that `vrt` stays pure algorithms. Every driver emits
host frames (RTSP emits a device-resident kornia `Image<u8,3>`; OAK emits a synced
stereo pair + IMU), consumed by the `vrt` models. The edge points one way:
`sensor-rt → vision-rt` (the **public upstream `kornia/vision-rt`**, pinned by
rev); `vrt` has no dependency back on sensors.

## Workspace layout

Flat `crates/` + `examples/`. `vrt`/`kornia` come from git (see root
`[workspace.dependencies]`); cudarc is 0.19 across the graph to match vision-rt
(shared `CudaStream`/`CudaSlice` for zero-copy interop).

| Crate | lib | Role |
|-------|-----|------|
| `crates/nvbuf-sys` | `nvbuf_sys` | FFI: NvBufSurface → CUDA device ptr from NVMM DMA-BUF (`links = nvbufsurface`) |
| `crates/sensor-rtsp` | `sensor_rtsp` | RTSP/H.264 source, NVMM → CUDA, emits device `Image<u8,3>` |
| `crates/sensor-oak` | `sensor_oak` | OAK-D **stereo pair + IMU** (`open_stereo`). Bundles the depthai C shim (no separate `-sys` crate); depends on NO inference runtime and never touches CUDA. RGB-D / H.264 paths removed for now — **`flux-oak` consumes those and must stay pinned to a pre-removal rev until it is reworked** |
| `crates/sensor-types` | `sensor_types` | Frame-timing leaf shared by every driver: `FrameMeta`, `Stamped<T>` (zero deps) |

## Architecture

Sensors are plain producers: `next_frame()` → `Stamped<Image<u8,3>>` (RTSP) or `next_stereo()` → an
`OakStereoFrame` lending both eyes as host **GRAY8** spans (CAM_B/CAM_C are mono —
consumers needing RGB expand on the GPU), or as zero-copy kornia
`Image`s that outlive the frame via a retained handle (OAK). RTSP frames are device-resident
and tightly packed RGB8 — the shape kornia's `Preprocessor` and the `vrt` models
consume. RTSP's NVMM path is RGBA + hardware-padded pitch, so it runs one on-GPU
pack kernel (RGBA-pitched → tight RGB8) — there is no zero-copy path into kornia's
tight-RGB8 `Image`, but the pack stays on-device (no host round-trip). OAK is
already tight RGB8 (zero extra copies).

## Hard constraints

- **GStreamer + libnvbufsurface are system/JetPack** (build + runtime) — NOT conda.
- **OAK-D needs the depthai prefix** under `vendor/depthai` (or `DEPTHAI_PREFIX=…`);
  runtime needs `LD_LIBRARY_PATH=…/vendor/depthai/lib` (libusb rpath). `sensor-oak`
  bakes an absolute rpath — rebuild from scratch if `vendor/` moves. The source is
  the **`vendor/depthai-core` git submodule** pinned to a release tag (`v3.7.1`):
  `git submodule update --init --recursive` then `pixi run depthai-build` to
  produce the prefix.
- **Upstream only**: all `vrt-*` deps come from the public `kornia/vision-rt`. Do
  NOT point them at a fork. Upstream deliberately has no
  `FrameMeta`/`Stamped` (producer concepts) — those live in `crates/sensor-types`.
  Device-shaped types (e.g. `OakIntrinsics`) belong to their driver crate:
  **driver crates must not depend on `vrt`**, so nothing that merely wants frames has
  to build TensorRT. Upstream model crates are **submit-only** (`alloc_result` +
  `submit` + an explicit `stream.synchronize()`); there is no `run()`.
- **Build cap**: `-j2` (`CARGO_BUILD_JOBS=2`) — parallel heavy builds OOM the 7.4 GB Orin.

## Commands

```bash
export CARGO_NET_GIT_FETCH_WITH_CLI=true CARGO_BUILD_JOBS=2
export DEPTHAI_PREFIX="$PWD/vendor/depthai"
export LD_LIBRARY_PATH=$DEPTHAI_PREFIX/lib
cargo build -j2
cargo fmt --all --check
```
