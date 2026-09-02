# sensor-rt

Real-time **sensor drivers in Rust** — cameras and inertial units — that hand off
kornia images ready for inference.

Drivers here are plain producers. Each one owns a transport and a device, emits
frames (and where the hardware has one, an IMU stream), and stops there: no
inference, no orchestration, no opinion about what consumes it. That boundary is
the point — a process that only wants pixels should not build a model runtime, and
several of these crates depend on neither an inference stack nor CUDA.

Frames are kornia `Image` types, so they drop straight into
[`vision-rt`](https://github.com/kornia/vision-rt)'s models, into kornia's own
image ops, or into anything else that speaks the same types. The dependency edge
points one way — `sensor-rt → vision-rt` — so the algorithm crates stay free of
GStreamer, depthai, and driver concerns.

**Sources today:** RTSP/H.264 (hardware NVMM decode) and OAK-D (synced stereo pair
+ IMU). Adding one is a crate, not a fork.

**Current target:** Jetson Orin (aarch64), JetPack 6.x, CUDA 12.6 — the platform
the hardware paths are tuned and tested against, not a limit of the design.

## Workspace

| Crate | Role |
|-------|------|
| `crates/nvbuf-sys` | FFI: Jetson `NvBufSurface` → CUDA device ptr from an NVMM DMA-BUF (`links = nvbufsurface`) |
| `crates/sensor-types` | Frame-timing leaf shared by every driver: `FrameMeta`, `Stamped<T>` (zero deps) |
| `crates/sensor-rtsp` | RTSP/H.264 source, NVMM → CUDA, emits a device `Image<u8,3>` (GStreamer) |
| `crates/sensor-oak` | OAK-D **stereo pair + IMU** and **RGBD + H.264**; pure Rust on the [`depthai`](https://github.com/kornia/depthai-rs) safe wrapper (depthai-core v3) |

## Usage

```rust
use sensor_oak::{ImuSample, OakSource};

// 640x400 per eye @30 fps, IMU at 200 Hz. No CUDA stream: the driver never
// touches the GPU, so the consumer owns any upload.
let mut cam = OakSource::open_stereo(None, 640, 400, 30, 200, false)?;
let mut imu: Vec<ImuSample> = Vec::new();

loop {
    imu.clear();
    cam.next_imu(&mut imu, 512);          // drained separately: ~200 Hz vs ~30 Hz frames
    let Some(frame) = cam.next_stereo() else { break };

    // Borrowed, copy-free, valid until the next poll:
    let (l, r) = (frame.left(), frame.right());          // &[u8], GRAY8 w*h (mono sensors)
    // Or zero-copy kornia Images that OUTLIVE the frame (retained handle):
    let left_img = frame.left_image()?;                  // Image<u8,1>, host
    // frame.meta(): FrameMeta (seq / pts) — same epoch clock as ImuSample::ts_ns
}
```

## Building

Native, Jetson-only. All dependencies are public. depthai-core comes through
[kornia/depthai-rs](https://github.com/kornia/depthai-rs), which owns the pinned
source and the build:

```bash
export DEPTHAI_PREFIX=/path/to/depthai/prefix   # lib/cmake/depthai inside
cargo build -j2                                  # -j2: the 7.4 GB Orin OOMs on parallel native builds
```

- **GStreamer + libnvbufsurface** are system/JetPack (build + runtime), not conda.
- **OAK-D** links `libdepthai-core.so` from `DEPTHAI_PREFIX`. No prefix yet? Either
  build one with depthai-rs's `depthai-sys/scripts/build_depthai.sh` (once, ~30-60
  min on an Orin, OpenCV off), or let cargo do it:
  `cargo build -p sensor-oak --features vendored` (needs cmake/ninja/pkg-config,
  e.g. from `pixi shell`, and `CARGO_NET_GIT_FETCH_WITH_CLI=true` so cargo pulls
  depthai-rs's source submodule). depthai-sys bakes the prefix's rpath into every binary,
  so no `LD_LIBRARY_PATH` is needed; rebuild if the prefix moves.
  `DEPTHAI_SYS_SKIP_NATIVE=1` gives a check-only build without any prefix.

CI runs `cargo fmt --all --check` on a hosted runner (fmt resolves no deps); the
real build/clippy/test job is gated on a self-hosted Jetson runner (it needs
CUDA, NVMM, depthai, and access to the private vision-rt dep).

## License

Apache-2.0
