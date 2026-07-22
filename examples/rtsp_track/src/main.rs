//! GStreamer RTSP → NVMM → RF-DETR → BoT-SORT tracking in real-time.
//!
//! Plain loop over the algorithm libraries:
//! ```text
//!   RtspSource::next_frame() → Stamped<VrtImage>   (NVMM, camera PTS)
//!   RfDetr::run(&img)        → Vec<Detection>      (GPU/TRT)
//!   Tracker::update(dets,dt) → Vec<TrackOut>       (Kalman + 2-stage assoc, CPU)
//! ```
//! `dt` comes from the camera PTS. Every `FRAME_STRIDE`-th frame the tracks are
//! drawn on the CPU snapshot (box colored by track id) and handed to a
//! background thread that H.264-encodes it into an `.mp4` via a GStreamer
//! `appsrc → videoconvert → x264enc → mp4mux` pipeline — so the encode never
//! blocks the capture/inference loop and every frame is kept (unlike PNG, which
//! is too slow to keep up). Set `FRAME_STRIDE=1` for a full-rate video.
//!
//! Env: `FRAME_STRIDE` (default 30), `VIDEO_FPS` (default 15),
//! `CAPTURE_FRAMES` (stop cleanly after N frames so the mp4 is finalized).
//!
//! Usage:
//!   cargo run --release -p rtsp_track -- rtsp://camera/stream [out_dir]

use std::sync::mpsc::{sync_channel, TrySendError};
use std::sync::Arc;
use std::time::Instant;

use kornia_image::Image;
use sensor_rtsp::{mp4, RtspSource};
use vrt::cudarc::driver::CudaStream;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime, Stream};
use vrt_reid::ReId;
use vrt_rfdetr::RfDetr;
use vrt_track::{Box2DTracker, Obs2D, TrackOut, TrackerConfig};

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();
    // GStreamer is initialized by RtspSource::connect and the mp4 writer.

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rtsp_track <rtsp_url> [save_dir]");
        std::process::exit(1);
    }
    let rtsp_url = &args[1];
    let save_dir = args.get(2).map(String::as_str).unwrap_or(".");

    // Download (HF, sha256-pinned) + build the fp16 engines on-device (cached).
    let logger = Logger::new(Severity::Warning)?;
    let runtime = Runtime::new(logger)?;

    // Detector model (default Medium; set RFDETR_MODEL=rfdetr-small for speed).
    let det_model = std::env::var("RFDETR_MODEL").unwrap_or_else(|_| "rfdetr-medium".into());
    let det_onnx = vrt_hub::ModelHub::get(&det_model)?;
    let det_path = vrt_hub::EngineCache::default().resolve(
        &det_model,
        &det_onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;
    let det_engine = Engine::from_file(runtime.clone(), &det_path)?;

    let reid_onnx = vrt_hub::ModelHub::get("osnet-reid")?;
    let reid_path = vrt_hub::EngineCache::default().resolve(
        "osnet-reid",
        &reid_onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;
    let reid_engine = Engine::from_file(runtime, &reid_path)?;

    // One shared stream: the RTSP RGBA→RGB pack, RF-DETR, and ReID run on it.
    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let mut source = RtspSource::connect_resized(rtsp_url, 1280, 720, stream.clone())?;
    let (src_w, src_h) = (source.width(), source.height());
    println!("Stream: {src_w}×{src_h} | {det_model} → OSNet ReID → BoT-SORT");

    let mut detr = RfDetr::new(det_engine, stream.clone(), 0.5)?;
    let mut reid = ReId::new(reid_engine, stream.clone())?;
    let mut tracker = Box2DTracker::new(TrackerConfig::default());

    // Save cadence: every FRAME_STRIDE-th frame (FRAME_STRIDE=1 = full rate).
    let stride: u64 = std::env::var("FRAME_STRIDE")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&s| s > 0)
        .unwrap_or(30);
    let video_fps: u64 = std::env::var("VIDEO_FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&f| f > 0)
        .unwrap_or(15);
    // Stop cleanly after N frames so the mp4 trailer is finalized (don't SIGKILL).
    let max_frames: Option<u64> = std::env::var("CAPTURE_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok());

    // H.264 encoding runs on a background thread so it never blocks capture. The
    // hot loop only draws boxes (cheap) and hands off the RGBA buffer; the worker
    // feeds a GStreamer appsrc→x264enc→mp4mux pipeline. x264enc easily keeps up
    // at the camera rate, so no frames are dropped (the bounded queue is just a
    // backpressure safety valve).
    let out_path = format!("{save_dir}/tracking.mp4");
    let (tx, rx) = sync_channel::<mp4::Frame>(8);
    let writer = mp4::spawn_writer(rx, out_path, video_fps);

    let mut n = 0u64;
    let mut dropped = 0u64;
    let t0 = Instant::now();
    let mut prev_pts: Option<u64> = None;
    while let Some(frame) = source.next_frame() {
        let dets = detr.run(&frame.data)?;
        let obs: Vec<Obs2D> = dets
            .iter()
            .map(|d| Obs2D {
                bbox: d.bbox,
                score: d.score,
                class_id: d.class_id,
            })
            .collect();

        // Appearance embeddings, aligned to `dets` (empty = not embedded). Scenes
        // here have few objects; with >batch detections you'd embed the top-batch
        // by score and leave the rest empty.
        let boxes: Vec<[f32; 4]> = dets.iter().map(|d| d.bbox).collect();
        let k = boxes.len().min(reid.batch());
        let mut embeds = vec![Vec::new(); dets.len()];
        for (i, e) in reid
            .embed(&frame.data, &boxes[..k])?
            .into_iter()
            .enumerate()
        {
            embeds[i] = e;
        }

        // dt from camera PTS (fall back to ~30fps for the first / missing stamp).
        let dt = match (prev_pts, frame.meta.pts_ns) {
            (Some(p), Some(c)) if c > p => (c - p) as f32 / 1e9,
            _ => 1.0 / 30.0,
        };
        prev_pts = frame.meta.pts_ns;

        let tracks = tracker.update(&obs, dt, Some(&embeds));
        n += 1;
        println!("[{n:06}] {} dets → {} tracks", dets.len(), tracks.len());
        for t in tracks.iter().take(6) {
            println!(
                "  id={:<3} class={:<3} score={:.2}  [{:.0},{:.0},{:.0},{:.0}]",
                t.id, t.class_id, t.score, t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]
            );
        }
        // Annotate the EXACT inference frame (D2H of frame.data) so boxes line up
        // with the pixels they were computed from, then hand it to the writer.
        if n.is_multiple_of(stride) {
            let job = annotate(&stream, &frame.data, &tracks);
            match tx.try_send(job) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => dropped += 1, // encoder behind; skip
                Err(TrySendError::Disconnected(_)) => break,
            }
        }
        if n.is_multiple_of(100) {
            println!(
                "── {n} frames, {:.1} fps  ({dropped} viz frames dropped)",
                n as f64 / t0.elapsed().as_secs_f64()
            );
        }
        if max_frames.is_some_and(|m| n >= m) {
            break;
        }
    }
    drop(tx); // close the channel so the writer sends EOS + finalizes
    let _ = writer.join();
    Ok(())
}

/// D2H the exact inference frame and draw each track's box (colored by id) on it.
/// Because this is the very surface the detector ran on, track boxes (already in
/// frame pixel coords) need no rescaling — they land exactly where they belong.
/// Returns the annotated tight-RGBA buffer + dims for the writer thread.
fn annotate(
    stream: &Arc<CudaStream>,
    img: &Image<u8, 3>,
    tracks: &[TrackOut],
) -> (Vec<u8>, u32, u32) {
    let (fw, fh) = (img.width() as u32, img.height() as u32);
    let mut buf = frame_to_host(stream, img);
    for t in tracks {
        draw_rect(
            &mut buf,
            fw,
            fh,
            t.bbox[0] as i32,
            t.bbox[1] as i32,
            t.bbox[2] as i32,
            t.bbox[3] as i32,
            id_color(t.id),
        );
    }
    (buf, fw, fh)
}

/// Copy a tight RGB8 device [`Image`] to a tight host **RGBA** buffer (`width*4`
/// stride, alpha = 255) so the CPU annotator can draw and the PNG writer can save.
fn frame_to_host(stream: &Arc<CudaStream>, img: &Image<u8, 3>) -> Vec<u8> {
    let (w, h) = (img.width(), img.height());
    let rgb = img
        .as_cudaslice()
        .and_then(|s| stream.clone_dtoh(s).ok())
        .unwrap_or_else(|| vec![0u8; w * h * 3]);
    let mut rgba = vec![0u8; w * h * 4];
    for i in 0..w * h {
        rgba[i * 4] = rgb[i * 3];
        rgba[i * 4 + 1] = rgb[i * 3 + 1];
        rgba[i * 4 + 2] = rgb[i * 3 + 2];
        rgba[i * 4 + 3] = 255;
    }
    rgba
}

/// Deterministic bright color per track id (so each id keeps its color).
fn id_color(id: u32) -> [u8; 4] {
    // Spread hues via a large odd multiplier; keep full saturation/value.
    let h = (id.wrapping_mul(2654435761) >> 8) as f32 % 360.0;
    let (r, g, b) = hsv_to_rgb(h, 1.0, 1.0);
    [r, g, b, 255]
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

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
