//! OAK-D 3D tracking → **MP4 demo video**.
//!
//! Same pipeline as `oakd_track3d` (OAK RGB+depth → RF-DETR → metric `Obs3D` →
//! `Box3DTracker`), but each frame is rendered into a side-by-side composite and
//! H.264-encoded to an `.mp4` (GStreamer `appsrc → x264enc → mp4mux`, on a
//! background thread — reused from `rtsp_track`):
//!
//! ```text
//!   ┌───────────────────────────┬──────────────────┐
//!   │ camera image              │  BEV (top-down)   │
//!   │  • detection boxes        │  • camera at base │
//!   │  • per-track id dot+range │  • id dots @ (x,z)│
//!   └───────────────────────────┴──────────────────┘
//! ```
//! Same id-colour links the two panels, so you watch tracked people move in the
//! image AND in metric top-down space. Record on the (headless) Jetson, play the
//! mp4 anywhere.
//!
//! Run (inside the pixi env for the depthai runtime libs):
//!   pixi run -- cargo run --release -p oakd_track3d_viz -- [out.mp4]
//! Env: RFDETR_CONF (0.4), CAPTURE_FRAMES (300), VIDEO_FPS (10).

use std::sync::mpsc::{sync_channel, TrySendError};
use std::sync::Arc;
use std::time::Instant;

use kornia_image::Image;
use sensor_oak::OakSource;
use sensor_rtsp::mp4;
use vrt::cudarc::driver::CudaStream;
use vrt::logger::Severity;
use vrt::{Engine, Intrinsics, Logger, Runtime, Stream};
use vrt_lift::Lifter;
use vrt_reid::ReId;
use vrt_rfdetr::RfDetr;
use vrt_track::Box3D;
use vrt_track::{Box3DTracker, Obs3D, TrackOut, TrackerConfig};

// BEV panel geometry (metres → pixels).
const BEV_W: u32 = 560;
const X_MAX: f32 = 4.0; // half-width of BEV in metres (x ∈ [-4, 4])
const Z_MAX: f32 = 6.0; // depth shown in BEV (z ∈ [0, 6])

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "oakd_track3d.mp4".into());
    let conf: f32 = env_f32("RFDETR_CONF", 0.4);
    let max_frames: u64 = env_u64("CAPTURE_FRAMES", 300);
    let video_fps: u64 = env_u64("VIDEO_FPS", 10);
    let (w, h, fps) = (1280u32, 720u32, 30u32);

    let det_model = std::env::var("RFDETR_MODEL").unwrap_or_else(|_| "rfdetr-medium".into());
    let onnx = vrt_hub::ModelHub::get(&det_model)?;
    let path = vrt_hub::EngineCache::default().resolve(
        &det_model,
        &onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;
    let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
    let engine = Engine::from_file(runtime.clone(), &path)?;

    // OSNet appearance ReID — gives the tracker an appearance bank so it can
    // re-acquire an id across detection gaps (the cause of the id churn).
    let reid_onnx = vrt_hub::ModelHub::get("osnet-reid")?;
    let reid_path = vrt_hub::EngineCache::default().resolve(
        "osnet-reid",
        &reid_onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;
    let reid_engine = Engine::from_file(runtime, &reid_path)?;

    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let mut detr = RfDetr::new(engine, stream.clone(), conf)?;
    let mut reid = ReId::new(reid_engine, stream.clone())?;
    let mut tracker = Box3DTracker::new(TrackerConfig::default());

    let mut src = OakSource::open(None, w, h, fps, stream.clone())?;
    let intr = src.intrinsics();
    let lifter = Lifter::new(intr);
    println!("recording {max_frames} frames → {out_path}  ({det_model}, conf={conf})");

    let comp_w = w + BEV_W;
    let (tx, rx) = sync_channel::<mp4::Frame>(8);
    let writer = mp4::spawn_writer(rx, out_path, video_fps);

    // Optional per-frame track log (frame,id,class,x,y,z,range) for diagnosing IDs.
    let mut csv = std::env::var("TRACK_CSV").ok().map(|p| {
        let mut f = std::fs::File::create(&p).expect("csv");
        use std::io::Write;
        let _ = writeln!(f, "frame,id,class,x,y,z,range");
        f
    });

    let mut n = 0u64;
    let t0 = Instant::now();
    let mut prev_pts: Option<u64> = None;
    while let Some(frame) = src.next_frame() {
        let dets = detr.run(frame.rgb())?;
        let depth = frame.depth();

        let obs: Vec<Obs3D> = match depth {
            Some(d) => dets
                .iter()
                .filter_map(|x| lifter.lift(d, &x.bbox, x.score, x.class_id))
                .collect(),
            None => Vec::new(),
        };

        // Appearance embeddings aligned to `dets` (top-batch by score; rest empty).
        // These let the tracker re-acquire ids across detection gaps.
        let boxes: Vec<[f32; 4]> = dets.iter().map(|d| d.bbox).collect();
        let k = boxes.len().min(reid.batch());
        let mut embeds = vec![Vec::new(); dets.len()];
        for (i, e) in reid
            .embed(frame.rgb(), &boxes[..k])?
            .into_iter()
            .enumerate()
        {
            embeds[i] = e;
        }

        // dt from camera PTS (real frame interval); fall back to ~fps for first frame.
        let dt = match (prev_pts, frame.meta().pts_ns) {
            (Some(p), Some(c)) if c > p => (c - p) as f32 / 1e9,
            _ => 1.0 / fps as f32,
        };
        prev_pts = frame.meta().pts_ns;
        let tracks = tracker.update(&obs, dt, Some(&embeds));

        if let Some(f) = csv.as_mut() {
            use std::io::Write;
            for t in &tracks {
                let c = t.bbox.center;
                let _ = writeln!(
                    f,
                    "{},{},{},{:.3},{:.3},{:.3},{:.3}",
                    n,
                    t.id,
                    t.class_id,
                    c[0],
                    c[1],
                    c[2],
                    range(c)
                );
            }
        }

        // Per-detection range for the image labels (independent of association).
        let det_ranges: Vec<Option<f32>> = dets
            .iter()
            .map(|d| depth.and_then(|dm| dm.sample_box(&d.bbox, 0.6)))
            .collect();

        let comp = compose(
            &stream,
            frame.rgb(),
            comp_w,
            h,
            &boxes,
            &det_ranges,
            &tracks,
            &intr,
        );

        n += 1;
        if n.is_multiple_of(30) {
            println!(
                "[{n:06}] {} dets → {} tracks  ({:.1} fps)",
                dets.len(),
                tracks.len(),
                n as f64 / t0.elapsed().as_secs_f64()
            );
        }
        match tx.try_send((comp, comp_w, h)) {
            Ok(()) | Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => break,
        }
        if n >= max_frames {
            break;
        }
    }
    drop(tx);
    let _ = writer.join();
    println!("done.");
    Ok(())
}

/// Build one side-by-side composite RGBA frame: [ camera image | BEV ].
#[allow(clippy::too_many_arguments)]
fn compose(
    stream: &Arc<CudaStream>,
    img: &Image<u8, 3>,
    comp_w: u32,
    comp_h: u32,
    det_boxes: &[[f32; 4]],
    det_ranges: &[Option<f32>],
    tracks: &[TrackOut<Box3D>],
    intr: &Intrinsics,
) -> Vec<u8> {
    let (sw, sh) = (img.width() as u32, img.height() as u32);
    let mut buf = vec![20u8; (comp_w * comp_h * 4) as usize]; // dark background
    for p in buf.chunks_exact_mut(4) {
        p[3] = 255;
    }

    // ── Left: camera image ───────────────────────────────────────────────────
    let rgba = frame_to_host_rgba(stream, img);
    blit(&mut buf, comp_w, &rgba, sw, sh, 0, 0);

    // Detection boxes (neutral) + per-detection range label.
    for (b, r) in det_boxes.iter().zip(det_ranges) {
        draw_rect(
            &mut buf,
            comp_w,
            comp_h,
            b[0] as i32,
            b[1] as i32,
            b[2] as i32,
            b[3] as i32,
            [180, 180, 180, 255],
        );
        if let Some(r) = r {
            draw_label(
                &mut buf,
                comp_w,
                comp_h,
                b[0] as i32,
                b[1] as i32 - 17,
                &fmt_m(*r),
                [230, 230, 230, 255],
            );
        }
    }
    // Each track as a projected 3D cuboid, coloured by id (links to the BEV).
    for t in tracks {
        let c = t.bbox.center;
        let col = id_color(t.id);
        draw_cuboid(
            &mut buf,
            comp_w,
            comp_h,
            c,
            t.bbox.size,
            t.bbox.yaw,
            intr,
            col,
        );
        // Label at the projected top-centre of the box (y is down, so −h/2).
        if let Some((lx, ly)) = project([c[0], c[1] - t.bbox.size[2] * 0.5, c[2]], intr) {
            draw_label(
                &mut buf,
                comp_w,
                comp_h,
                lx + 4,
                ly - 18,
                &fmt_id_m(t.id, range(c)),
                col,
            );
        }
    }

    // ── Right: BEV (top-down) ────────────────────────────────────────────────
    draw_bev(&mut buf, comp_w, comp_h, sw, tracks);
    buf
}

/// Draw the bird's-eye-view panel into the composite, starting at x = `ox`.
fn draw_bev(buf: &mut [u8], comp_w: u32, comp_h: u32, ox: u32, tracks: &[TrackOut<Box3D>]) {
    let (bx, bw, bh) = (ox as i32, BEV_W as i32, comp_h as i32);
    // Panel background.
    for y in 0..bh {
        for x in 0..bw {
            put(buf, comp_w, comp_h, bx + x, y, [32, 34, 40, 255]);
        }
    }
    let to_px = |x: f32, z: f32| -> (i32, i32) {
        let px = bx + ((x + X_MAX) / (2.0 * X_MAX) * bw as f32) as i32;
        let py = (bh as f32 - (z / Z_MAX) * bh as f32) as i32;
        (px, py)
    };
    // Range rings (distance lines) + labels, every metre.
    for zi in 1..=Z_MAX as i32 {
        let (_, py) = to_px(0.0, zi as f32);
        for x in 0..bw {
            put(buf, comp_w, comp_h, bx + x, py, [60, 64, 72, 255]);
        }
        draw_label(
            buf,
            comp_w,
            comp_h,
            bx + 4,
            py - 9,
            &fmt_m(zi as f32),
            [120, 130, 140, 255],
        );
    }
    // Centre line (x = 0) and camera marker at the base.
    let (cx0, _) = to_px(0.0, 0.0);
    for y in 0..bh {
        put(buf, comp_w, comp_h, cx0, y, [60, 64, 72, 255]);
    }
    fill_tri(buf, comp_w, comp_h, cx0, bh - 4, 7, [200, 200, 80, 255]);

    // Tracks: id-coloured footprint dot + range label.
    for t in tracks {
        let c = t.bbox.center;
        let (px, py) = to_px(c[0], c[2]);
        let col = id_color(t.id);
        fill_circle(buf, comp_w, comp_h, px, py, 9, col);
        draw_label(
            buf,
            comp_w,
            comp_h,
            px + 11,
            py - 6,
            &fmt_id_m(t.id, range(c)),
            col,
        );
    }
}

fn range(c: [f32; 3]) -> f32 {
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
}

/// Project a camera-frame point to integer image pixels (the pinhole model lives
/// on `vrt::Intrinsics`; this just rounds for drawing).
fn project(p: [f32; 3], intr: &Intrinsics) -> Option<(i32, i32)> {
    intr.project(p).map(|(u, v)| (u as i32, v as i32))
}

/// Draw a track's 3D box as a projected wireframe cuboid. Camera frame is x-right,
/// y-down, z-forward; `size` is `[l, w, h]` → x, z, y(vertical) extents; `yaw`
/// rotates the footprint about the vertical (y) axis.
#[allow(clippy::too_many_arguments)]
fn draw_cuboid(
    buf: &mut [u8],
    w: u32,
    h: u32,
    center: [f32; 3],
    size: [f32; 3],
    yaw: f32,
    intr: &Intrinsics,
    col: [u8; 4],
) {
    let (hx, hy, hz) = (size[0] * 0.5, size[2] * 0.5, size[1] * 0.5); // x=l, y=h, z=w
    let (sin, cos) = (yaw.sin(), yaw.cos());
    let mut pts = [None; 8];
    for (i, p) in pts.iter_mut().enumerate() {
        let sx = if i & 1 != 0 { 1.0 } else { -1.0 };
        let sy = if i & 2 != 0 { 1.0 } else { -1.0 };
        let sz = if i & 4 != 0 { 1.0 } else { -1.0 };
        let (dx, dy, dz) = (sx * hx, sy * hy, sz * hz);
        let rx = dx * cos + dz * sin; // rotate about vertical axis
        let rz = -dx * sin + dz * cos;
        *p = project([center[0] + rx, center[1] + dy, center[2] + rz], intr);
    }
    // 12 edges: corner pairs that differ in exactly one axis bit.
    for i in 0..8usize {
        for b in 0..3 {
            let j = i ^ (1 << b);
            if j > i {
                if let (Some(a), Some(c)) = (pts[i], pts[j]) {
                    draw_line(buf, w, h, a.0, a.1, c.0, c.1, col);
                }
            }
        }
    }
}
fn fmt_m(m: f32) -> String {
    format!("{m:.1}m")
}
fn fmt_id_m(id: u32, m: f32) -> String {
    format!("{id}:{m:.1}m")
}

// ── Image copy (Rgb8 device → tight RGBA host) ────────────────────────────────

/// D2H the OAK RGB888 device frame and expand to a tight RGBA host buffer.
fn frame_to_host_rgba(stream: &Arc<CudaStream>, img: &Image<u8, 3>) -> Vec<u8> {
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

/// Copy a tight RGBA image into the composite buffer at (ox, oy).
fn blit(dst: &mut [u8], dst_w: u32, src: &[u8], sw: u32, sh: u32, ox: u32, oy: u32) {
    for y in 0..sh {
        let drow = (((oy + y) * dst_w + ox) * 4) as usize;
        let srow = (y * sw * 4) as usize;
        dst[drow..drow + (sw * 4) as usize].copy_from_slice(&src[srow..srow + (sw * 4) as usize]);
    }
}

// ── Drawing primitives ────────────────────────────────────────────────────────

fn id_color(id: u32) -> [u8; 4] {
    let h = (id.wrapping_mul(2654435761) >> 8) as f32 % 360.0;
    let (r, g, b) = hsv_to_rgb(h, 0.9, 1.0);
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
fn put(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, c: [u8; 4]) {
    if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
        let p = (y as usize * w as usize + x as usize) * 4;
        buf[p..p + 4].copy_from_slice(&c);
    }
}
#[allow(clippy::too_many_arguments)]
fn draw_rect(buf: &mut [u8], w: u32, h: u32, x1: i32, y1: i32, x2: i32, y2: i32, c: [u8; 4]) {
    for t in 0..2 {
        for x in x1.min(x2)..=x1.max(x2) {
            put(buf, w, h, x, y1 + t, c);
            put(buf, w, h, x, y2 - t, c);
        }
        for y in y1.min(y2)..=y1.max(y2) {
            put(buf, w, h, x1 + t, y, c);
            put(buf, w, h, x2 - t, y, c);
        }
    }
}
/// 2px-thick Bresenham line.
#[allow(clippy::too_many_arguments)]
fn draw_line(buf: &mut [u8], w: u32, h: u32, x0: i32, y0: i32, x1: i32, y1: i32, c: [u8; 4]) {
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let mut err = dx + dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        put(buf, w, h, x, y, c);
        put(buf, w, h, x + 1, y, c);
        put(buf, w, h, x, y + 1, c);
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}
fn fill_circle(buf: &mut [u8], w: u32, h: u32, cx: i32, cy: i32, r: i32, c: [u8; 4]) {
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                put(buf, w, h, cx + dx, cy + dy, c);
            }
        }
    }
}
fn fill_tri(buf: &mut [u8], w: u32, h: u32, cx: i32, base_y: i32, s: i32, c: [u8; 4]) {
    for dy in 0..s {
        let half = s - dy;
        for dx in -half..=half {
            put(buf, w, h, cx + dx, base_y - dy, c);
        }
    }
}

// ── Tiny 3×5 bitmap font (digits + ':' '.' 'm') for labels ────────────────────

fn glyph(ch: char) -> [u8; 5] {
    // rows top→bottom, low 3 bits = left→right
    match ch {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        'm' => [0b000, 0b000, 0b111, 0b111, 0b101],
        _ => [0; 5],
    }
}
/// Draw `text` at (x,y) (scaled 3×) with a dark contrast backing.
#[allow(clippy::too_many_arguments)]
fn draw_label(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, text: &str, c: [u8; 4]) {
    const S: i32 = 3;
    let tw = text.len() as i32 * 4 * S;
    // backing
    for by in -1..(5 * S + 1) {
        for bx in -1..(tw + 1) {
            put(buf, w, h, x + bx, y + by, [0, 0, 0, 200]);
        }
    }
    let mut cx = x;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if bits & (1 << (2 - col)) != 0 {
                    for sy in 0..S {
                        for sx in 0..S {
                            put(buf, w, h, cx + col * S + sx, y + row as i32 * S + sy, c);
                        }
                    }
                }
            }
        }
        cx += 4 * S;
    }
}

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}
fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}
