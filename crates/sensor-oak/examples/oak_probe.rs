//! Stage-1 bring-up gate for the OAK-D path — NO detector, NO tracker.
//!
//! Proves: native shim ↔ device over USB, a synced RGBD frame arrives, factory
//! intrinsics read back, and depth is sane. Run inside the pixi env so the
//! depthai-core runtime libs resolve:
//!   pixi run -- cargo run -p sensor-oak --example oak_probe
//!
//! Optional: `OAK_W=640 OAK_H=400 OAK_FPS=15` to drop resolution (USB2 fallback).

use sensor_oak::OakSource;
use vrt::Stream;

fn main() -> Result<(), vrt::BoxError> {
    let w: u32 = std::env::var("OAK_W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1280);
    let h: u32 = std::env::var("OAK_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(720);
    let fps: u32 = std::env::var("OAK_FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    println!("opening OAK at {w}×{h}@{fps} ...");
    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let mut src = OakSource::open(None, w, h, fps, stream)?;
    let intr = src.intrinsics();
    println!(
        "device up: {}×{}  depth={}  intrinsics fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        src.width(),
        src.height(),
        src.has_depth(),
        intr.fx,
        intr.fy,
        intr.cx,
        intr.cy,
    );

    // Pull a handful of frames — the first few may be empty while the pipeline spins up.
    let mut got = 0u32;
    for attempt in 1..=60 {
        let Some(frame) = src.next_frame() else {
            continue;
        };
        got += 1;
        let img = frame.rgb();
        let cx = img.width() as u32 / 2;
        let cy = img.height() as u32 / 2;

        let (center_mm, valid_pct) = match frame.depth() {
            Some(d) => {
                let center = d.meters_at(cx, cy).map(|m| (m * 1000.0) as u32);
                (center, 100.0 * d.valid_fraction())
            }
            None => (None, 0.0),
        };

        println!(
            "frame {got:2} (attempt {attempt:2})  rgb {}×{}  pts={:?}  center_depth={}  depth_valid={:.1}%",
            img.width(), img.height(), frame.meta().pts_ns,
            center_mm.map_or("none".to_string(), |mm| format!("{mm} mm")),
            valid_pct,
        );

        if got >= 5 {
            break;
        }
    }

    if got == 0 {
        return Err(
            "no frames received — check USB cable/port (USB3) and device enumeration".into(),
        );
    }
    println!("Stage-1 OK: {got} synced frame(s) pulled.");
    Ok(())
}
