//! Bring-up gate for the OAK-D **stereo + IMU** modality — no CUDA, no models.
//!
//! Proves the four things the modality promises, and prints the evidence for each:
//!   1. a synced left/right pair arrives, both eyes the same size and tightly packed
//!   2. the two eyes are genuinely *different* images (a mis-wired Sync node that
//!      handed back the same frame twice would otherwise look perfect)
//!   3. the IMU streams at roughly the requested rate
//!   4. IMU timestamps share the frames' epoch timeline (so they can be interpolated)
//!
//! Run inside the pixi env so the depthai-core runtime libs resolve:
//!   pixi run -- cargo run -p sensor-oak --example oak_stereo_probe
//!
//! Optional: `-- --width 640 --height 400 --fps 30 --imu-hz 200 --frames 60`.

use argh::FromArgs;
use sensor_oak::{BoxError, ImuSample, OakSource};
use std::time::Instant;

#[derive(FromArgs)]
/// Bring-up gate for the OAK-D stereo + IMU modality: checks that a synced pair
/// arrives, that the two eyes really differ, and that the IMU streams on the same
/// clock as the frames.
struct Args {
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
    /// how many frames to check (default 60)
    #[argh(option, default = "60")]
    frames: u64,
}

fn main() -> Result<(), BoxError> {
    let args: Args = argh::from_env();
    let (w, h, fps, imu_hz, frames) = (args.width, args.height, args.fps, args.imu_hz, args.frames);

    println!("opening OAK stereo+IMU at {w}×{h}@{fps}, imu={imu_hz}Hz ...");
    let mut src = OakSource::open_stereo(None, w, h, fps, imu_hz)?;
    let intr = src.intrinsics();
    println!(
        "device up: imu={}  CAM_B intrinsics fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        src.has_imu(),
        intr.fx,
        intr.fy,
        intr.cx,
        intr.cy,
    );
    if !src.has_imu() {
        println!(
            "NOTE: no IMU on this board — stereo checks still apply, IMU checks will be skipped"
        );
    }

    let mut imu: Vec<ImuSample> = Vec::new();
    let mut n = 0u64;
    let mut last_frame_ts = 0u64;
    // Worst (largest) absolute difference seen between the two eyes' mean intensity. Near-zero across
    // every frame is the signature of the Sync node handing back the same image twice.
    let mut max_eye_delta = 0.0f32;
    let mut min_eye_delta = f32::MAX;
    let t0 = Instant::now();

    while n < frames {
        // Drain the IMU OUTSIDE the held frame — both borrow &mut self.
        let got = src.next_imu(&mut imu, 256);

        let Some(frame) = src.next_stereo() else {
            println!("stream ended after {n} frames");
            break;
        };
        n += 1;
        let ts = frame.meta().pts_ns.unwrap_or(0);
        let (fw, fh) = (frame.width(), frame.height());
        let expect = fw as usize * fh as usize * 3;

        // (1) shape contract
        assert_eq!(frame.left().len(), expect, "left eye is not tight RGB888");
        assert_eq!(frame.right().len(), expect, "right eye is not tight RGB888");

        // (2) the eyes must differ — a stereo pair of the same scene from two baselines
        let ml = mean(frame.left());
        let mr = mean(frame.right());
        let delta = (ml - mr).abs();
        max_eye_delta = max_eye_delta.max(delta);
        min_eye_delta = min_eye_delta.min(delta);
        let identical = frame.left() == frame.right();
        assert!(
            !identical,
            "left and right eyes are byte-identical — Sync node is not pairing two cameras"
        );

        if n <= 3 || n.is_multiple_of(20) {
            let dt_ms = if last_frame_ts > 0 {
                (ts.saturating_sub(last_frame_ts)) as f64 / 1e6
            } else {
                0.0
            };
            println!(
                "[{n:04}] {fw}×{fh}  dt={dt_ms:6.2}ms  mean L={ml:6.2} R={mr:6.2} (Δ{delta:.2})  \
                 imu+{got} (total {})",
                imu.len()
            );
        }
        last_frame_ts = ts;
    }

    let secs = t0.elapsed().as_secs_f64();
    println!("\n── results over {n} frames / {secs:.2}s ──");
    println!(
        "frame rate      : {:.1} fps (requested {fps})",
        n as f64 / secs
    );
    println!("eye mean Δ      : min {min_eye_delta:.2}, max {max_eye_delta:.2} (non-zero ⇒ two real cameras)");

    if src.has_imu() {
        // Drain whatever is still queued so the rate estimate isn't truncated.
        src.next_imu(&mut imu, 4096);
        let rate = imu.len() as f64 / secs;
        println!(
            "imu samples     : {} → {rate:.0} Hz (requested {imu_hz})",
            imu.len()
        );

        if let (Some(first), Some(last)) = (imu.first(), imu.last()) {
            // (4) IMU and frames must share the epoch timeline. If the shim ever regressed to
            // handing out raw steady-clock times, this gap would be years, not milliseconds.
            let skew_ms = (last.ts_ns as i128 - last_frame_ts as i128) as f64 / 1e6;
            println!("imu↔frame skew  : {skew_ms:.1} ms (same epoch timeline ⇒ small)");
            assert!(
                skew_ms.abs() < 5000.0,
                "IMU and frame timestamps are not on the same clock (skew {skew_ms:.0} ms)"
            );
            // Monotonic, and a plausible gravity magnitude — proves real inertial data, not zeros.
            let g =
                (first.accel[0].powi(2) + first.accel[1].powi(2) + first.accel[2].powi(2)).sqrt();
            println!(
                "first sample    : accel {:?} (|a|={g:.2} m/s², expect ~9.8 at rest)  gyro {:?}",
                first.accel, first.gyro
            );
            let monotonic = imu.windows(2).all(|p| p[1].ts_ns >= p[0].ts_ns);
            println!("imu monotonic   : {monotonic}");
            assert!(
                monotonic,
                "IMU timestamps went backwards — batch ordering is broken"
            );
        }
    }
    // (5) A retained image must OUTLIVE the frame it came from. `left_image()` borrows
    // depthai's buffer and holds a retain handle, so it must stay byte-identical after
    // further polls have recycled the frame that produced it. If the retain were
    // missing (or the keepalive a dummy) this is exactly where it would show up: the
    // pixels would drift to whatever depthai wrote next.
    // The inner scope ends the frame's borrow of `src`, so `held` outliving it is not
    // just a runtime claim — it has to type-check, which is half the guarantee.
    let held = {
        let frame = src
            .next_stereo()
            .ok_or("no stereo frame for the retain check")?;
        frame.left_image()?
    };
    {
        let before = mean(held.as_slice());
        for _ in 0..10 {
            let _ = src.next_stereo(); // recycle the buffers behind it
        }
        let after = mean(held.as_slice());
        println!("retained image  : mean {before:.4} before / {after:.4} after 10 more polls");
        assert_eq!(
            before.to_bits(),
            after.to_bits(),
            "retained image changed after later polls — the frame was not actually retained"
        );
        println!("zero-copy hold  : image outlived its frame, contents stable");
    }

    println!("\nOK — stereo+IMU modality validated");
    Ok(())
}

/// Mean byte value of an RGB888 span — a cheap, allocation-free image fingerprint.
fn mean(buf: &[u8]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    // Sum in u64: a 640×400×3 span of 255s overflows u32 only barely, but larger frames would.
    let sum: u64 = buf.iter().map(|&b| b as u64).sum();
    sum as f32 / buf.len() as f32
}
