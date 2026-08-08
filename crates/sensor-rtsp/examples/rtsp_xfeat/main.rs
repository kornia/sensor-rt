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

use argh::FromArgs;
use sensor_rtsp::RtspSource;

#[derive(FromArgs)]
/// XFeat keypoints on a live RTSP/NVMM stream.
struct Args {
    /// XFeat backbone: .onnx is built on-device into the engine cache (one-time);
    /// .engine is used directly and must match this machine's TRT + GPU
    #[argh(positional)]
    model: String,
    /// RTSP URL to pull from
    #[argh(positional)]
    url: String,
    /// directory for the periodic keypoint PNGs (default ".")
    #[argh(option, default = "String::from(\".\")")]
    save_dir: String,
}
use std::time::Instant;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime, Stream};
use vrt_xfeat::{XFeat, XFeatParams};

mod sink;
use sink::KeypointViz;

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let args: Args = argh::from_env();
    let (model_path, rtsp_url, save_dir) = (&args.model, &args.url, args.save_dir.as_str());

    // .onnx → versioned engine cache (build on first run); .engine → as-is.
    // `inputs` is a Vec since upstream #18 (one profile per dynamic input).
    let profile = vrt_hub::EngineProfile {
        inputs: vec![(
            "image".into(),
            vec![1, 3, 240, 320],
            vec![1, 3, 640, 640],
            vec![1, 3, 1088, 1920],
        )],
        fp16: true,
        ..Default::default()
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
    println!(
        "Stream: resized to {src_w}×{src_h} (VIC); XFeat runs its backbone at {}×{} (floor-32)",
        (src_w / 32) * 32,
        (src_h / 32) * 32,
    );

    let cpu_snap = source.latest_cpu_frame();
    // Upstream XFeat sizes its backbone per frame (floor-of-32 of the input) and
    // rescales keypoints back to source pixels, so no model-input size is configured.
    let mut xfeat = XFeat::new(engine, stream.clone(), XFeatParams::new(4096, 0.05))?;
    // Submit-only API: caller owns the output buffer and the stream sync.
    let mut result = xfeat.alloc_result()?;
    // The viz shares the stream to download device-resident keypoints when drawing.
    let viz = KeypointViz::new(cpu_snap, save_dir.to_string(), 30);

    let mut n = 0u64;
    let t0 = Instant::now();
    while let Some(frame) = source.next_frame() {
        xfeat.submit(&frame.data, &mut result)?;
        stream.synchronize()?;
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
