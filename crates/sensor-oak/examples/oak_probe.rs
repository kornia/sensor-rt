//! Stage-1 bring-up gate for the OAK-D path — NO detector, NO tracker.
//!
//! Proves: native shim ↔ device over USB, a synced RGBD frame arrives, factory
//! intrinsics read back, and depth is sane. Run inside the pixi env so the
//! depthai-core runtime libs resolve:
//!   pixi run -- cargo run -p sensor-oak --example oak_probe
//!
//! Optional: `--width 640 --height 400 --fps 15` to drop resolution (USB2 fallback).

use argh::FromArgs;
use sensor_oak::OakSource;

#[derive(FromArgs)]
/// Stage-1 bring-up gate for the OAK-D path: prove the shim talks to the device
/// and a synced RGB-D frame arrives.
struct Args {
    /// frame width (default 1280)
    #[argh(option, default = "1280")]
    width: u32,
    /// frame height (default 720)
    #[argh(option, default = "720")]
    height: u32,
    /// capture rate (default 30)
    #[argh(option, default = "30")]
    fps: u32,
}

fn main() -> Result<(), vrt::BoxError> {
    let args: Args = argh::from_env();
    let (w, h, fps) = (args.width, args.height, args.fps);

    println!("opening OAK at {w}×{h}@{fps} ...");
    let mut src = OakSource::open(None, w, h, fps)?;
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
        let cx = frame.width() / 2;
        let cy = frame.height() / 2;

        let (center_mm, valid_pct) = match frame.depth() {
            Some(d) => {
                let center = d.meters_at(cx, cy).map(|m| (m * 1000.0) as u32);
                (center, 100.0 * d.valid_fraction())
            }
            None => (None, 0.0),
        };

        println!(
            "frame {got:2} (attempt {attempt:2})  rgb {}×{}  pts={:?}  center_depth={}  depth_valid={:.1}%",
            frame.width(), frame.height(), frame.meta().pts_ns,
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
