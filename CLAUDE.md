# sensor-rt

Isolated **sensor drivers** for Jetson — RTSP/NVMM cameras and OAK-D depth —
extracted from vision-rt so that `vrt` stays pure algorithms. Every driver emits
a device-resident kornia `Image<u8,3>` (plus `vrt::VrtDepthMap` + `Intrinsics`
for OAK), consumed directly by the `vrt` models. The edge points one way:
`sensor-rt → vision-rt` (a **private git dep**, pinned by rev); `vrt` has no
dependency back on sensors. Three-repo split: `vrt` ← `sensor-rt` ← `flux`.

## Workspace layout

Flat `crates/` + `examples/`. `vrt`/`kornia` come from git (see root
`[workspace.dependencies]`); cudarc is 0.19 across the graph to match vision-rt
(shared `CudaStream`/`CudaSlice` for zero-copy interop).

| Crate | lib | Role |
|-------|-----|------|
| `crates/nvbuf-sys` | `nvbuf_sys` | FFI: NvBufSurface → CUDA device ptr from NVMM DMA-BUF (`links = nvbufsurface`) |
| `crates/oak-sys` | `oak_sys` | FFI: C shim over depthai-core v3 (`links = depthai-core`; built from `vendor/`) |
| `crates/sensor-rtsp` | `sensor_rtsp` | RTSP/H.264 source, NVMM → CUDA, emits device `Image<u8,3>` |
| `crates/sensor-oak` | `sensor_oak` | OAK-D RGB + aligned depth → device `Image<u8,3>` + `VrtDepthMap` |

## Architecture

Sensors are plain producers: `next_frame()` → `Stamped<Image<u8,3>>` (RTSP) or an
`OakFrame` lending `&Image<u8,3>` + `&VrtDepthMap` (OAK). Frames are device-resident
and tightly packed RGB8 — the shape kornia's `Preprocessor` and the `vrt` models
consume. RTSP's NVMM path is RGBA + hardware-padded pitch, so it runs one on-GPU
pack kernel (RGBA-pitched → tight RGB8) — there is no zero-copy path into kornia's
tight-RGB8 `Image`, but the pack stays on-device (no host round-trip). OAK is
already tight RGB8 (zero extra copies).

## Hard constraints

- **GStreamer + libnvbufsurface are system/JetPack** (build + runtime) — NOT conda.
- **OAK-D needs the depthai prefix** under `vendor/depthai` (or `DEPTHAI_PREFIX=…`);
  runtime needs `LD_LIBRARY_PATH=…/vendor/depthai/lib` (libusb rpath). `oak-sys`
  bakes an absolute rpath — rebuild from scratch if `vendor/` moves. The source is
  the **`vendor/depthai-core` git submodule** pinned to a release tag (`v3.7.1`):
  `git submodule update --init --recursive` then `pixi run depthai-build` to
  produce the prefix.
- **Private git dep**: `export CARGO_NET_GIT_FETCH_WITH_CLI=true` so cargo clones
  the private vision-rt via the git credential helper.
- **Build cap**: `-j2` (`CARGO_BUILD_JOBS=2`) — parallel heavy builds OOM the 7.4 GB Orin.

## Commands

```bash
export CARGO_NET_GIT_FETCH_WITH_CLI=true CARGO_BUILD_JOBS=2
export DEPTHAI_PREFIX=/home/nvidia/sensor-rt/vendor/depthai
export LD_LIBRARY_PATH=$DEPTHAI_PREFIX/lib
cargo build -j2
cargo fmt --all --check
```
