//! GStreamer RTSP → NVMM → XFeat keypoint detection in real-time.
//!
//! Plain loop over the algorithm libraries — no orchestration framework:
//! ```text
//!   RtspSource::next_frame()  →  Stamped<VrtImage>   (NVMM imported, camera PTS)
//!   XFeat::run(&img)          →  XFeatResult         (letterbox + backbone + top-K)
//!   KeypointViz::draw_and_save(...)                  (periodic PNG overlay)
//! ```
//! Orchestration (threads, Zenoh, microservices) is the application's job; this
//! example just calls the algorithms directly.
//!
//! Usage:
//!   cargo run --release -p rtsp_xfeat -- \
//!       models/xfeat/xfeat_backbone_fp16.engine rtsp://camera/stream [save_dir]

use sensor_rtsp::RtspSource;
use std::time::Instant;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime, Stream};
use vrt_xfeat::{XFeat, XFeatParams};

mod sink;
use sink::KeypointViz;

fn pad32(v: u32) -> u32 {
    v.div_ceil(32) * 32
}

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: rtsp_xfeat <model.onnx|model.engine> <rtsp_url> [save_dir]");
        eprintln!("  .onnx   — built on-device into ~/.cache/vision-rt/engines (one-time)");
        eprintln!("  .engine — used directly (must match this machine's TRT + GPU)");
        std::process::exit(1);
    }
    let (model_path, rtsp_url) = (&args[1], &args[2]);
    let save_dir = args.get(3).map(String::as_str).unwrap_or(".");

    // .onnx → versioned engine cache (build on first run); .engine → as-is.
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

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;

    // Hardware-resize to 1280×720 via VIC before any CUDA work.
    const RESIZE_W: u32 = 1280;
    const RESIZE_H: u32 = 720;
    // One shared stream: the RTSP RGBA→RGB pack and XFeat inference run on it.
    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let mut source = RtspSource::connect_resized(rtsp_url, RESIZE_W, RESIZE_H, stream.clone())?;
    let (src_w, src_h) = (source.width(), source.height());
    let (dst_w, dst_h) = (pad32(src_w), pad32(src_h));
    println!("Stream: resized to {src_w}×{src_h} (VIC) → model input {dst_w}×{dst_h}");

    let cpu_snap = source.latest_cpu_frame();
    // XFeat owns its letterbox preprocessor (source frames → dst_w×dst_h model input).
    let mut xfeat = XFeat::new(
        engine,
        stream.clone(),
        XFeatParams::new(4096, 0.05, dst_h as usize, dst_w as usize),
    )?;
    // The viz shares the stream to download device-resident keypoints when drawing.
    let viz = KeypointViz::new(cpu_snap, stream, save_dir.to_string(), dst_w, dst_h, 30);

    let mut n = 0u64;
    let t0 = Instant::now();
    while let Some(frame) = source.next_frame() {
        let result = xfeat.run(&frame.data)?;
        n += 1;
        // frame.meta carries the camera capture PTS; the app owns the seq counter.
        let pts_ms = frame.meta.pts_ns.map(|p| p as f64 / 1e6).unwrap_or(0.0);
        println!("[{n:06}] pts={pts_ms:.1}ms  | {} kpts", result.len());
        viz.draw_and_save(n, &result);

        if n.is_multiple_of(100) {
            let fps = n as f64 / t0.elapsed().as_secs_f64();
            println!("── {n} frames, {fps:.1} fps");
        }
    }

    Ok(())
}
