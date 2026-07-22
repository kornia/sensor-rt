# sensor-rt

Isolated **camera drivers for Jetson** — RTSP/H.264 over NVMM, and the OAK-D
stereo pair + IMU — feeding the [`vision-rt`](https://github.com/kornia/vision-rt)
algorithm crates. Drivers are plain producers: they emit kornia `Image<u8,3>`
frames and nothing else.

The dependency edge points one way — `sensor-rt → vision-rt` — so `vrt` stays
pure algorithms with no sensor/GStreamer/depthai dependency. The OAK driver goes
further and depends on **no inference runtime and no CUDA at all**, so a process
that only wants camera frames builds neither.

**Target platform:** Jetson Orin (aarch64), JetPack 6.x, CUDA 12.6.

## Workspace

| Crate | Role |
|-------|------|
| `crates/nvbuf-sys` | FFI: Jetson `NvBufSurface` → CUDA device ptr from an NVMM DMA-BUF (`links = nvbufsurface`) |
| `crates/sensor-types` | Frame-timing leaf shared by every driver: `FrameMeta`, `Stamped<T>` (zero deps) |
| `crates/sensor-rtsp` | RTSP/H.264 source, NVMM → CUDA, emits a device `Image<u8,3>` (GStreamer) |
| `crates/sensor-oak` | OAK-D **stereo pair + IMU**; bundles the depthai-core v3 C shim (`links = depthai-core`, built from `vendor/`) |

## Usage

```rust
use sensor_oak::{ImuSample, OakSource};

// 640x400 per eye @30 fps, IMU at 200 Hz. No CUDA stream: the driver never
// touches the GPU, so the consumer owns any upload.
let mut cam = OakSource::open_stereo(None, 640, 400, 30, 200)?;
let mut imu: Vec<ImuSample> = Vec::new();

loop {
    imu.clear();
    cam.next_imu(&mut imu, 512);          // drained separately: ~200 Hz vs ~30 Hz frames
    let Some(frame) = cam.next_stereo() else { break };

    // Borrowed, copy-free, valid until the next poll:
    let (l, r) = (frame.left(), frame.right());          // &[u8], RGB888 w*h*3
    // Or zero-copy kornia Images that OUTLIVE the frame (retained handle):
    let left_img = frame.left_image()?;                  // Image<u8,3>, host
    // frame.meta(): FrameMeta (seq / pts) — same epoch clock as ImuSample::ts_ns
}
```

## Building

Native, Jetson-only. All dependencies are public. depthai-core is a git submodule
built into a local prefix:

```bash
git submodule update --init --recursive
pixi run depthai-build                # produces vendor/depthai (gitignored)
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
