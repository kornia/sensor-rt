//! GStreamer RTSP → NVMM → RF-DETR object detection in real-time.
//!
//! Plain loop over the algorithm libraries — no orchestration framework:
//! ```text
//!   RtspSource::next_frame()  →  Stamped<VrtImage>   (NVMM imported, camera PTS)
//!   RfDetr::run(&img)         →  Vec<Detection>      (stretch resize + TRT + decode, no NMS)
//! ```
//! The RF-DETR Small ONNX is downloaded from Hugging Face (sha256-pinned via
//! vrt-hub) and built into an fp16 TensorRT engine on first run (cached). Every
//! 30 frames the detections are drawn on the CPU snapshot and saved as a PNG.
//!
//! Usage:
//!   cargo run --release -p rtsp_rfdetr -- rtsp://camera/stream [save_dir]

use std::sync::{Arc, Mutex};
use std::time::Instant;

use kornia_image::{Image, ImageSize};
use kornia_io::png::write_image_png_rgba8;
use sensor_rtsp::{CpuFrame, RtspSource};
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime, Stream};
use vrt_rfdetr::{Detection, RfDetr};

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rtsp_rfdetr <rtsp_url> [save_dir]");
        std::process::exit(1);
    }
    let rtsp_url = &args[1];
    let save_dir = args.get(2).map(String::as_str).unwrap_or(".");

    // Download (HF, sha256-pinned) + build the fp16 engine on-device (cached).
    let det_model = std::env::var("RFDETR_MODEL").unwrap_or_else(|_| "rfdetr-medium".into());
    let onnx = vrt_hub::ModelHub::get(&det_model)?;
    let engine_path = vrt_hub::EngineCache::default().resolve(
        &det_model,
        &onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;

    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;
    let engine = Engine::from_file(runtime, &engine_path)?;

    // One shared stream: the RTSP RGBA→RGB pack and RF-DETR inference run on it.
    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let mut source = RtspSource::connect_resized(rtsp_url, 1280, 720, stream.clone())?;
    let (src_w, src_h) = (source.width(), source.height());
    println!("Stream: {src_w}×{src_h} | {det_model} (stretch)");
    let cpu_snap = source.latest_cpu_frame();

    let mut detr = RfDetr::new(engine, stream, 0.5)?;

    let mut n = 0u64;
    let t0 = Instant::now();
    while let Some(frame) = source.next_frame() {
        let dets = detr.run(&frame.data)?;
        n += 1;
        let pts_ms = frame.meta.pts_ns.map(|p| p as f64 / 1e6).unwrap_or(0.0);
        println!("[{n:06}] pts={pts_ms:.1}ms  | {} dets", dets.len());
        for d in dets.iter().take(5) {
            println!(
                "  class={:<3} score={:.2}  [{:.0},{:.0},{:.0},{:.0}]",
                d.class_id, d.score, d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]
            );
        }
        if n.is_multiple_of(30) {
            save_dets(&cpu_snap, src_w, src_h, &dets, save_dir, n);
        }
        if n.is_multiple_of(100) {
            println!(
                "── {n} frames, {:.1} fps",
                n as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

/// Draw detection boxes on the latest CPU snapshot and save a PNG.
fn save_dets(
    cpu_snap: &Arc<Mutex<Option<CpuFrame>>>,
    src_w: u32,
    src_h: u32,
    dets: &[Detection],
    save_dir: &str,
    seq: u64,
) {
    let Some((rgba, fw, fh)) = cpu_snap.lock().ok().and_then(|mut g| g.take()) else {
        eprintln!("[viz] no CPU frame yet at seq {seq}");
        return;
    };
    let mut buf = rgba;
    // boxes are in source pixels; scale to the snapshot's resolution.
    let sx = fw as f32 / src_w as f32;
    let sy = fh as f32 / src_h as f32;
    for d in dets {
        draw_rect(
            &mut buf,
            fw,
            fh,
            (d.bbox[0] * sx) as i32,
            (d.bbox[1] * sy) as i32,
            (d.bbox[2] * sx) as i32,
            (d.bbox[3] * sy) as i32,
            [50, 255, 50, 255],
        );
    }
    let path = format!("{save_dir}/rfdetr_{seq:06}.png");
    match Image::<u8, 4>::new(
        ImageSize {
            width: fw as usize,
            height: fh as usize,
        },
        buf,
    ) {
        Ok(img) => match write_image_png_rgba8(&path, &img) {
            Ok(()) => println!("[viz] saved {path}  ({} dets)", dets.len()),
            Err(e) => eprintln!("[viz] save failed: {e}"),
        },
        Err(e) => eprintln!("[viz] bad frame buffer: {e}"),
    }
}

/// Draw an axis-aligned rectangle outline (2px) into an interleaved RGBA buffer.
#[allow(clippy::too_many_arguments)]
fn draw_rect(buf: &mut [u8], w: u32, h: u32, x1: i32, y1: i32, x2: i32, y2: i32, color: [u8; 4]) {
    for t in 0..2 {
        draw_hline(buf, w, h, x1, x2, y1 + t, color);
        draw_hline(buf, w, h, x1, x2, y2 - t, color);
        draw_vline(buf, w, h, x1 + t, y1, y2, color);
        draw_vline(buf, w, h, x2 - t, y1, y2, color);
    }
}

fn put(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, color: [u8; 4]) {
    if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
        let p = (y as usize * w as usize + x as usize) * 4;
        buf[p..p + 4].copy_from_slice(&color);
    }
}
fn draw_hline(buf: &mut [u8], w: u32, h: u32, x1: i32, x2: i32, y: i32, c: [u8; 4]) {
    for x in x1.min(x2)..=x1.max(x2) {
        put(buf, w, h, x, y, c);
    }
}
fn draw_vline(buf: &mut [u8], w: u32, h: u32, x: i32, y1: i32, y2: i32, c: [u8; 4]) {
    for y in y1.min(y2)..=y1.max(y2) {
        put(buf, w, h, x, y, c);
    }
}
