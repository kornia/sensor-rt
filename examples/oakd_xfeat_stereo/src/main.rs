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
//!   cargo run --release -p oakd_xfeat_stereo -- \
//!       models/xfeat/xfeat_backbone_fp16.engine [save_dir]
//!
//! Env: `OAK_W=640 OAK_H=400 OAK_FPS=30 OAK_IMU_HZ=200 OAK_FRAMES=0` (0 = forever),
//!      `OAK_SAVE_EVERY=30` (0 disables PNG output).

use std::sync::Arc;
use std::time::Instant;

use kornia_image::Image;
use sensor_oak::{alloc_rgb_image, ImuSample, OakSource};
use vrt::{BoxError, Engine, Logger, Runtime, Stream};
use vrt_xfeat::{Matcher, XFeat, XFeatParams, XFeatResult};

mod viz;
use viz::StereoMatchViz;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), BoxError> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let positional: Vec<&String> = args[1..].iter().filter(|a| !a.starts_with("--")).collect();
    if positional.is_empty() {
        eprintln!("Usage: oakd_xfeat_stereo <model.onnx|model.engine> [save_dir]");
        std::process::exit(1);
    }
    let model_path = positional[0];
    let save_dir = positional.get(1).map(|s| s.as_str()).unwrap_or(".");

    let w = env_u32("OAK_W", 640);
    let h = env_u32("OAK_H", 400);
    let fps = env_u32("OAK_FPS", 30);
    let imu_hz = env_u32("OAK_IMU_HZ", 200);
    let max_frames = env_u32("OAK_FRAMES", 0) as u64;
    let save_every = env_u32("OAK_SAVE_EVERY", 30) as u64;

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

    // Reused device buffers — one per eye, each on its own stream so the two uploads
    // are independent and can overlap. Sized lazily from the first frame's ACTUAL
    // dims: the device may hand back a different size than requested (CROP against
    // the sensor's native resolution), and `memcpy_htod` asserts dst >= src, so
    // sizing these from the *requested* dims would panic on a larger frame and
    // silently leave stale tail rows on a smaller one.
    let mut dev_l: Option<Image<u8, 3>> = None;
    let mut dev_r: Option<Image<u8, 3>> = None;

    let viz = StereoMatchViz::new(save_dir.to_string(), save_every);

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

        // (Re)allocate on the first frame, or if the device ever changes size.
        let (fw, fh) = (frame.width(), frame.height());
        let fits = |img: &Option<Image<u8, 3>>| {
            img.as_ref()
                .is_some_and(|i| i.width() as u32 == fw && i.height() as u32 == fh)
        };
        if !fits(&dev_l) {
            dev_l = Some(alloc_rgb_image(&s0, fw, fh)?);
            dev_r = Some(alloc_rgb_image(&s1, fw, fh)?);
        }
        let (img_l, img_r) = (dev_l.as_mut().unwrap(), dev_r.as_mut().unwrap());

        // Uploads: async on their own streams, so the right eye's copy is already
        // moving while the left eye's preprocess kernel launches.
        s0.memcpy_htod(
            frame.left(),
            img_l.as_cudaslice_mut().ok_or("left not device")?,
        )?;
        s1.memcpy_htod(
            frame.right(),
            img_r.as_cudaslice_mut().ok_or("right not device")?,
        )?;

        // Both submissions return immediately — nothing has synced yet, so the GPU
        // holds work from both streams at once and can interleave it.
        xf_l.submit(img_l, &mut res_l)?;
        xf_r.submit(img_r, &mut res_r)?;

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

        viz.draw_and_save(n, &frame, &res_l, &res_r, &matches);

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
