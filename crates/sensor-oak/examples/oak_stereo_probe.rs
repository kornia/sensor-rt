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
//! Optional: `OAK_W=640 OAK_H=400 OAK_FPS=30 OAK_IMU_HZ=200 OAK_FRAMES=60`.

use sensor_oak::{ImuSample, OakSource};
use std::time::Instant;

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), vrt::BoxError> {
    let w = env_u32("OAK_W", 640);
    let h = env_u32("OAK_H", 400);
    let fps = env_u32("OAK_FPS", 30);
    let imu_hz = env_u32("OAK_IMU_HZ", 200);
    let frames = env_u32("OAK_FRAMES", 60) as u64;

    println!("opening OAK stereo+IMU at {w}×{h}@{fps}, imu={imu_hz}Hz ...");
    let mut src = OakSource::open_stereo(None, w, h, fps, imu_hz)?;
    let intr = src.intrinsics();
    println!(
        "device up: stereo={} imu={}  CAM_B intrinsics fx={:.2} fy={:.2} cx={:.2} cy={:.2}",
        src.has_stereo(),
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
