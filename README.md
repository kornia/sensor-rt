# sensor-rt

Isolated **camera drivers for Jetson** — RTSP/H.264 over NVMM and OAK-D RGB-D —
that feed the [`vision-rt`](https://github.com/edgarriba/vision-rt) algorithm
crates. Drivers are plain producers: they emit a device-resident kornia
`Image<u8,3>` (plus depth / intrinsics for OAK), ready to hand to a `vrt` model.

The dependency edge points one way — `sensor-rt → vision-rt` — so `vrt` stays
pure algorithms with no sensor/GStreamer/depthai dependency. Part of the
three-repo split: `vision-rt` (algorithms) ← **sensor-rt** (drivers) ← `flux`
(publishing).

**Target platform:** Jetson Orin (aarch64), JetPack 6.x, CUDA 12.6.

## Workspace

| Crate | Role |
|-------|------|
| `crates/nvbuf-sys` | FFI: Jetson `NvBufSurface` → CUDA device ptr from an NVMM DMA-BUF (`links = nvbufsurface`) |
| `crates/oak-sys` | FFI: C shim over depthai-core v3 (`links = depthai-core`, built from `vendor/`) |
| `crates/sensor-rtsp` | RTSP/H.264 source, NVMM → CUDA, emits a device `Image<u8,3>` (GStreamer) |
| `crates/sensor-oak` | OAK-D RGB + aligned depth → device `Image<u8,3>` + `vrt::VrtDepthMap` |

## Usage

```rust
use sensor_oak::OakSource;
use vrt_rfdetr::RfDetr;

let mut cam = OakSource::open(Default::default())?;   // RGB + aligned depth
let mut detr = RfDetr::new(engine, cam.cuda_stream(), 0.5)?;

while let Some(frame) = cam.next_frame() {
    let dets = detr.run(frame.rgb())?;                // frame.rgb(): &Image<u8,3>, device-resident
    // frame.depth(): &VrtDepthMap  ·  frame.meta(): FrameMeta (seq / pts)
}
```

## Building

Native, Jetson-only. The image type + `vrt` come from vision-rt (a **private**
git dep), so cargo must fetch with the git CLI credentials:

```bash
export CARGO_NET_GIT_FETCH_WITH_CLI=true
cargo build -j2                       # -j2: the 7.4 GB Orin OOMs on parallel native builds
```

- **GStreamer + libnvbufsurface** are system/JetPack (build + runtime), not conda.
- **OAK-D** builds depthai-core from `vendor/depthai-core` — a git **submodule**
  pinned to a release tag (currently `v3.7.1`). Fetch it on a fresh checkout with
  `git clone --recursive …` (or `git submodule update --init --recursive`), then
  build the install prefix once: `pixi run depthai-build` (→ `vendor/depthai`).
  Runtime needs `LD_LIBRARY_PATH=…/vendor/depthai/lib` (libusb rpath); override
  the prefix with `DEPTHAI_PREFIX=…`.

CI runs `cargo fmt --all --check` on a hosted runner (fmt resolves no deps); the
real build/clippy/test job is gated on a self-hosted Jetson runner (it needs
CUDA, NVMM, depthai, and access to the private vision-rt dep).

## License

Apache-2.0
