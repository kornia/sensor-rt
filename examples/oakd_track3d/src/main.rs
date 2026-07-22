//! OAK-D real-depth 3D tracking:
//! ```text
//!   OakSource (RGB device + aligned depth mm) → RF-DETR (Jetson TRT, 2D boxes)
//!     → sample depth at each box → unproject (camera frame, metres) → Obs3D
//!     → Box3DTracker → 3D tracks with stable ids
//! ```
//! Depth is computed on the OAK's own VPU — no depth model on the Jetson. 3D
//! positions are in the camera frame (x right, y down, z forward).
//!
//! Run (via pixi, which provides the depthai-core runtime libs):
//!   pixi run oakd
//! or:  cargo run --release -p oakd_track3d

use std::time::Instant;

use sensor_oak::OakSource;
use vrt::logger::Severity;
use vrt::{Engine, Logger, Runtime, Stream};
use vrt_lift::Lifter;
use vrt_rfdetr::RfDetr;
use vrt_track::{Box3DTracker, Obs3D, TrackerConfig};

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let (w, h, fps) = (1280u32, 720u32, 30u32);

    // Detector engine (RF-DETR, default Medium) via the local model cache.
    let det_model = std::env::var("RFDETR_MODEL").unwrap_or_else(|_| "rfdetr-medium".into());
    let onnx = vrt_hub::ModelHub::get(&det_model)?;
    let path = vrt_hub::EngineCache::default().resolve(
        &det_model,
        &onnx.to_string_lossy(),
        &vrt_hub::EngineProfile::default(),
    )?;
    let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
    let engine = Engine::from_file(runtime, &path)?;

    let conf: f32 = std::env::var("RFDETR_CONF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.5);
    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let mut detr = RfDetr::new(engine, stream.clone(), conf)?;
    let mut tracker = Box3DTracker::new(TrackerConfig::default());

    let mut src = OakSource::open(None, w, h, fps, stream.clone())?;
    let intr = src.intrinsics();
    let lifter = Lifter::new(intr); // reusable depth→3D bridge (per-class size priors)
    println!(
        "OAK {}×{}  depth={}  intrinsics fx={:.1} fy={:.1} cx={:.1} cy={:.1}  | {det_model} → Box3DTracker",
        src.width(), src.height(), src.has_depth(), intr.fx, intr.fy, intr.cx, intr.cy);
    if !src.has_depth() {
        return Err("OAK device has no depth stream (need a stereo pair)".into());
    }

    let mut n = 0u64;
    let t0 = Instant::now();
    let mut prev_pts: Option<u64> = None;
    while let Some(frame) = src.next_frame() {
        let dets = detr.run(frame.rgb())?;
        let depth = frame.depth();

        // 2D boxes + aligned depth → 3D observations (camera frame) via the
        // reusable Lifter (robust per-box depth sampling + per-class size priors).
        let obs: Vec<Obs3D> = match depth {
            Some(depth) => dets
                .iter()
                .filter_map(|d| lifter.lift(depth, &d.bbox, d.score, d.class_id))
                .collect(),
            None => Vec::new(),
        };

        let dt = match (prev_pts, frame.meta().pts_ns) {
            (Some(p), Some(c)) if c > p => (c - p) as f32 / 1e9,
            _ => 1.0 / fps as f32,
        };
        prev_pts = frame.meta().pts_ns;

        let tracks = tracker.update(&obs, dt, None);
        n += 1;
        println!(
            "[{n:06}] {} dets → {} obs(3D) → {} tracks",
            dets.len(),
            obs.len(),
            tracks.len()
        );
        for t in tracks.iter().take(8) {
            let c = t.bbox.center;
            let range = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            println!(
                "  id={:<3} class={:<3} xyz=({:+.2},{:+.2},{:+.2}) m  range={:.2} m",
                t.id, t.class_id, c[0], c[1], c[2], range
            );
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
