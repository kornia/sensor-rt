//! OAK-D stereo pair → XFeat on **both eyes concurrently** → left↔right matching.
//!
//! ```text
//!   OakSource::next_stereo()  →  StereoFrame (host RGB888 left + right, synced)
//!   upload left  → device Image on stream0        upload right → device Image on stream1
//!   xf_l.submit(left)   ─┐                        xf_r.submit(right)  ─┐   both async,
//!                        └── neither has synced yet ───────────────────┘   both in flight
//!   stream0.sync(); stream1.sync()
//!   Matcher::submit_match(left, right)  →  mutual-NN pairs, all on device
//! ```
//!
//! The two streams let both eyes be in flight at once. **Measured on an AGX Orin
//! at 640x400, that buys nothing:** ~12.3 ms mean GPU section either way (a
//! sequential submit+sync per eye measured 12.1 ms over 200 frames — marginally
//! *faster*). XFeat's backbone already saturates the SMs, so two of them
//! time-slice rather than overlap, and the second stream adds a little scheduling
//! overhead for no gain; the cheap parts (H2D uploads, NMS, top-K) are too small a
//! fraction to matter. The pipeline is camera-bound regardless: frames arrive
//! every 33 ms and the whole GPU section costs ~12 ms.
//!
//! Kept as the concurrent version anyway, because it is the right *shape* for a
//! multi-sensor front-end (it generalises to N cameras, and to workloads that
//! don't saturate the GPU) — just do not expect the streams to be a speedup here.
//! The real 2x lever would be a batch-2 engine (one `[2,3,h,w]` inference), which
//! XFeat cannot do today: its post-processing is fixed at batch 1.
//!
//! Usage:
//!   cargo run --release -p sensor-oak --example oakd_xfeat_stereo -- \
//!       models/xfeat/xfeat_backbone_fp16.engine --rrd /tmp/stereo.rrd
//!
//! Then `rerun /tmp/stereo.rrd`, or pass `--rrd-connect` to stream to a live viewer.
//! `--help` lists the rest (`--width`, `--fps`, `--imu-hz`, `--frames`, `--image-every`).

use std::sync::Arc;
use std::time::Instant;

use argh::FromArgs;
use sensor_oak::{ImuSample, OakSource};

#[derive(FromArgs)]
/// XFeat on both eyes of an OAK-D stereo pair, across two CUDA streams, matched on device.
struct Args {
    /// path to the XFeat backbone (.onnx builds and caches an engine; .engine is used as-is)
    #[argh(positional)]
    model: String,
    /// path for the rerun recording (default "oakd_xfeat_stereo.rrd")
    #[argh(option, default = "String::from(\"oakd_xfeat_stereo.rrd\")")]
    rrd: String,
    /// stream to a running rerun viewer over gRPC instead of writing a file
    #[argh(switch)]
    rrd_connect: bool,
    /// per-eye width (default 640)
    #[argh(option, default = "640")]
    width: u32,
    /// per-eye height (default 400)
    #[argh(option, default = "400")]
    height: u32,
    /// stereo pair rate (default 30)
    #[argh(option, default = "30")]
    fps: u32,
    /// imu report rate in Hz; 0 disables the IMU (default 200)
    #[argh(option, default = "200")]
    imu_hz: u32,
    /// stop after N frames; 0 runs until the stream ends (default 0)
    #[argh(option, default = "0")]
    frames: u64,
    /// log imagery every N frames; 0 disables images (counters and IMU always log)
    #[argh(option, default = "30")]
    image_every: u64,
}
use vrt::{BoxError, Engine, Logger, Runtime, Stream};
use vrt_xfeat::{Matcher, XFeat, XFeatParams, XFeatResult};

mod viz;
use viz::StereoViz;

fn main() -> Result<(), BoxError> {
    env_logger::init();

    let args: Args = argh::from_env();
    let (w, h, fps, imu_hz) = (args.width, args.height, args.fps, args.imu_hz);
    let max_frames = args.frames;
    let model_path = args.model.as_str();

    // ── camera ────────────────────────────────────────────────────────────────
    // Host-only by design: the source hands out spans, and *we* decide which CUDA
    // stream each eye lands on — which is the whole basis of the overlap below.
    let mut cam = OakSource::open_stereo(None, w, h, fps, imu_hz)?;
    println!(
        "OAK stereo up: {}×{} @{fps}  imu={}  (CAM_B fx={:.1})",
        cam.width(),
        cam.height(),
        cam.has_imu(),
        cam.intrinsics().fx,
    );

    // ── two CUDA streams on ONE context ───────────────────────────────────────
    // `new_standalone` creates the primary-context stream; the second comes from
    // that same context, so device pointers from either are mutually valid (they
    // are context-global) — a prerequisite for matching across the two eyes.
    let s0 = Stream::new_standalone()?.cuda_stream().clone();
    let s1 = s0.context().new_stream()?;

    // ── engine, shared by both extractors ─────────────────────────────────────
    let profile = vrt_hub::EngineProfile {
        input: Some((
            "image".into(),
            vec![1, 3, 240, 320],
            vec![1, 3, 640, 640],
            vec![1, 3, 1088, 1920],
        )),
        fp16: true,
        workspace_mb: 2048,
    };
    let engine_path =
        vrt_hub::EngineCache::default().resolve("xfeat-backbone", model_path, &profile)?;
    let logger = Logger::new(vrt::logger::Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;

    const TOP_K: usize = 4096;
    let params = XFeatParams::new(TOP_K, 0.05);
    // Upstream XFeat sizes the backbone per frame (floor-of-32 of the input) and
    // rescales keypoints back to original pixels itself — so there is no fixed
    // model-input size to configure here, and no model→frame scaling in the viz.
    println!(
        "XFeat: top_k={TOP_K}, backbone input {}×{} (floor-32 of the frame)",
        (cam.width() / 32) * 32,
        (cam.height() / 32) * 32,
    );

    // One Arc<Engine>, two extractors: TensorRT shares the engine's weights across
    // execution contexts, so the second costs activations only — not a second copy
    // of the model. Each binds to its own stream, and therefore its own scratch.
    let mut xf_l = XFeat::new(Arc::clone(&engine), s0.clone(), params.clone())?;
    let mut xf_r = XFeat::new(engine, s1.clone(), params)?;

    // Results are CALLER-allocated upstream, which is what makes multiple frames in
    // flight expressible: each eye owns its own output, so neither submission can
    // clobber the other's buffers before we sync.
    let mut res_l = XFeatResult::alloc(&s0, TOP_K)?;
    let mut res_r = XFeatResult::alloc(&s1, TOP_K)?;

    // The matcher is decoupled from postproc upstream; it runs on s0 and reads both
    // eyes' descriptors (see the sync note in the loop).
    let matcher = Matcher::new(s0.clone())?;
    let mut match_res = matcher.alloc_result(TOP_K)?;

    let viz = StereoViz::new(&args.rrd, args.rrd_connect, args.image_every)?;

    let mut imu: Vec<ImuSample> = Vec::new();
    let mut n = 0u64;
    let mut gpu_ms_total = 0.0f64;
    let t0 = Instant::now();

    loop {
        if max_frames > 0 && n >= max_frames {
            break;
        }
        // Drain inertial data outside the held frame (both borrow &mut cam).
        // Cleared first: only the newest sample is reported, so retaining the rest
        // would grow the Vec for the whole run (~19 MB/hour at 185 Hz) and would let
        // a frame whose drain came back empty print a stale earlier reading as if it
        // belonged to this frame. A consumer that actually integrates the samples
        // would keep them — and would bound the buffer itself.
        imu.clear();
        let imu_got = cam.next_imu(&mut imu, 512);
        let latest_imu = imu.last().copied();

        let Some(frame) = cam.next_stereo() else {
            println!("stereo stream ended after {n} frames");
            break;
        };
        n += 1;
        let pts_ms = frame.meta().pts_ns.map(|p| p as f64 / 1e6).unwrap_or(0.0);

        let t_gpu = Instant::now();

        // Host image -> device image, through kornia's own API. `to_cuda_image` does
        // the H2D in one `clone_htod` and hands back a device-resident `Image`, so the
        // frame never leaves the typed image world and this example needs no CUDA
        // allocation code of its own. Each eye uploads on its own stream.
        let img_l = frame.left_image()?.to_cuda_image(&s0)?;
        let img_r = frame.right_image()?.to_cuda_image(&s1)?;

        // Both submissions return immediately — nothing has synced yet, so the GPU
        // holds work from both streams at once and can interleave it.
        xf_l.submit(&img_l, &mut res_l)?;
        xf_r.submit(&img_r, &mut res_r)?;

        // ⚠ LOAD-BEARING, NOT DECORATIVE. `Stream::new_standalone` disables cudarc's
        // per-op event tracking, whose safety contract is "no buffer crosses
        // streams". We deliberately break that below: the match kernel is enqueued
        // on s0 but reads `res_r`, which was allocated and written on s1. These two
        // syncs are the *only* thing establishing that happens-before.
        //
        // Deleting `s1.synchronize()` does NOT reliably corrupt anything today, and
        // that is precisely why it is dangerous: the two eyes submit near-identical
        // workloads, so s1 has almost always drained by the time the ~12 ms
        // `s0.synchronize()` returns. The race is masked by a coincidence of timing,
        // not prevented. It would surface the moment the two streams diverge —
        // different per-eye resolutions, a heavier postproc on one side, or an extra
        // kernel added to just one path. Measured absence of corruption is not
        // evidence of safety here.
        s0.synchronize()?;
        s1.synchronize()?;

        // Mutual nearest-neighbour on device — descriptors never leave the GPU.
        // Enqueued on s0, then synced before reading the pairs back.
        matcher.submit_match(
            &res_l.descs,
            res_l.count(),
            &res_r.descs,
            res_r.count(),
            0.82,
            &mut match_res,
        )?;
        s0.synchronize()?;
        // NOTE: `pairs()` collects a fresh Vec every frame even though the pairs
        // themselves are only drawn 1 frame in `save_every`. Skipping it on the other
        // frames would need a count-only accessor that upstream `MatchResult` does not
        // expose — not worth a fork; ~16 KB/frame at TOP_K.
        let matches = match_res.pairs();

        let gpu_ms = t_gpu.elapsed().as_secs_f64() * 1e3;
        gpu_ms_total += gpu_ms;

        let imu_str = latest_imu.map_or_else(
            || "  imu: -".to_string(),
            |s| {
                format!(
                    "  imu+{imu_got} a=[{:+.2},{:+.2},{:+.2}] g=[{:+.3},{:+.3},{:+.3}]",
                    s.accel[0], s.accel[1], s.accel[2], s.gyro[0], s.gyro[1], s.gyro[2],
                )
            },
        );
        println!(
            "[{n:06}] pts={pts_ms:.1}ms  L={:5} R={:5} kpts  {:5} matches  gpu={gpu_ms:6.2}ms{imu_str}",
            res_l.len(),
            res_r.len(),
            matches.len(),
        );

        viz.log_frame(n, &frame, &res_l, &res_r, &matches, &imu)?;

        if n.is_multiple_of(100) {
            let fps_now = n as f64 / t0.elapsed().as_secs_f64();
            println!(
                "── {n} frames, {fps_now:.1} fps end-to-end, {:.2} ms mean GPU section",
                gpu_ms_total / n as f64
            );
        }
    }

    if n > 0 {
        let secs = t0.elapsed().as_secs_f64();
        println!("\n── {n} frames / {secs:.2}s ──");
        println!("end-to-end     : {:.2} fps", n as f64 / secs);
        println!("GPU section    : {:.2} ms mean", gpu_ms_total / n as f64);
    }
    Ok(())
}
