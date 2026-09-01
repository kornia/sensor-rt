//! Probe the RGBD + H.264 modality: open, report capabilities, drain all three
//! queues plus the IMU for a few seconds and print rates + dims. The acceptance
//! check for `open_rgbd` after a driver change — compare its numbers before and
//! after.
//!
//! `cargo run --release --example oak_rgbd_probe -- [--device <id>] [--seconds 5] [--no-depth] [--video-only]`

use std::time::{Duration, Instant};

use argh::FromArgs;
use sensor_oak::{BoxError, ImuSample, OakSource};

#[derive(FromArgs)]
/// Drain the RGBD/H.264 modality and print per-stream statistics.
struct Args {
    /// device MxId or IP (default: first available)
    #[argh(option)]
    device: Option<String>,
    /// colour width (default 640)
    #[argh(option, default = "640")]
    width: u32,
    /// colour height (default 360)
    #[argh(option, default = "360")]
    height: u32,
    /// colour + encoder rate (default 30)
    #[argh(option, default = "30")]
    fps: u32,
    /// IMU rate, 0 disables (default 200)
    #[argh(option, default = "200")]
    imu_hz: u32,
    /// how long to drain (default 5)
    #[argh(option, default = "5")]
    seconds: u64,
    /// skip StereoDepth (for an uncalibrated camera)
    #[argh(switch)]
    no_depth: bool,
    /// video-only pipeline (open_rgbd_video)
    #[argh(switch)]
    video_only: bool,
}

fn main() -> Result<(), BoxError> {
    env_logger::init();
    let a: Args = argh::from_env();
    let t0 = Instant::now();
    let mut src = if a.video_only {
        OakSource::open_rgbd_video(a.device.as_deref(), a.width, a.height, a.fps, a.imu_hz)?
    } else {
        OakSource::open_rgbd(
            a.device.as_deref(),
            a.width,
            a.height,
            a.fps,
            !a.no_depth,
            a.imu_hz,
        )?
    };
    println!(
        "opened in {:.1}s: has_sync={} has_depth={} has_video={} has_imu={} imu_aligned={} intrinsics={:?}",
        t0.elapsed().as_secs_f32(),
        src.has_sync(),
        src.has_depth(),
        src.has_video(),
        src.has_imu(),
        src.imu_aligned(),
        src.intrinsics(),
    );

    let (mut rgb, mut depth, mut video, mut video_bytes, mut imu_n) =
        (0u32, 0u32, 0u32, 0usize, 0usize);
    let (mut rgb_dims, mut depth_dims) = ((0, 0), (0, 0));
    let (mut first_ts, mut last_ts) = (0u64, 0u64);
    let mut imu: Vec<ImuSample> = Vec::new();
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(a.seconds) {
        while let Some((bytes, w, h, ts)) = src.next_rgb() {
            rgb += 1;
            rgb_dims = (w, h);
            assert_eq!(bytes.len(), (w * h * 3) as usize);
            if first_ts == 0 {
                first_ts = ts;
            }
            last_ts = ts;
        }
        while let Some((vals, w, h, _ts)) = src.next_depth() {
            depth += 1;
            depth_dims = (w, h);
            assert_eq!(vals.len(), (w * h) as usize);
        }
        while let Some((bytes, _ts)) = src.next_video() {
            video += 1;
            video_bytes += bytes.len();
        }
        imu.clear();
        imu_n += src.next_imu(&mut imu, 512);
        std::thread::sleep(Duration::from_millis(2));
    }
    let s = a.seconds as f32;
    println!(
        "{s}s: rgb={rgb} ({:.1} fps, {}x{}) depth={depth} ({:.1} fps, {}x{}) video={video} ({:.1} fps, {} kB) imu={imu_n} ({:.0} Hz)",
        rgb as f32 / s,
        rgb_dims.0,
        rgb_dims.1,
        depth as f32 / s,
        depth_dims.0,
        depth_dims.1,
        video as f32 / s,
        video_bytes / 1024,
        imu_n as f32 / s,
    );
    if rgb > 1 {
        println!(
            "rgb timestamps span {:.2}s (epoch ns {first_ts}..{last_ts})",
            (last_ts - first_ts) as f64 / 1e9
        );
    }
    if src.has_depth() && depth_dims.0 > 0 {
        assert_eq!(depth_dims.0 % 2, 0, "depth width must be even (XLink)");
        assert_eq!(depth_dims.1 % 2, 0, "depth height must be even (XLink)");
    }
    Ok(())
}
