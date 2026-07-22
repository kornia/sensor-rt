//! Rerun logging for the stereo XFeat loop.
//!
//! Replaces the side-by-side PNG dump this example used to write. A PNG could only
//! show one frame in isolation; rerun puts everything on a shared timeline, which is
//! what the interesting questions here actually need:
//!
//!   * are the eyes genuinely different images, and do keypoints track the scene?
//!   * do matched keypoints line up row-wise (correct stereo disparity) or scatter?
//!   * does the IMU move when the camera moves — i.e. are frames and inertial data
//!     really on one clock?
//!
//! Everything is logged against a `frame` sequence timeline plus the camera's own
//! capture timestamp, so scrubbing lines the images up with the IMU trace.
//!
//! Writes an `.rrd` file by default (no viewer needed on the robot); open it later
//! with `rerun <file>`. `--rrd-connect` streams to a running viewer instead.

use rerun::{Points2D, RecordingStream, RecordingStreamBuilder};
use sensor_oak::{ImuSample, OakStereoFrame};
use vrt::BoxError;
use vrt_xfeat::XFeatResult;

/// Matched keypoints are drawn larger and in a distinct colour so they stand out
/// from the unmatched background set.
const MATCHED_COLOR: [u8; 3] = [40, 220, 90];
const UNMATCHED_COLOR: [u8; 3] = [90, 90, 200];

pub struct StereoViz {
    rec: RecordingStream,
    /// Log imagery every `image_every` frames; the IMU and counters log every frame
    /// regardless. Images dominate the .rrd size, and at 30 fps most are redundant.
    image_every: u64,
}

impl StereoViz {
    /// `rrd_path` writes a file; `connect` streams to a viewer on the default port.
    pub fn new(rrd_path: &str, connect: bool, image_every: u64) -> Result<Self, BoxError> {
        let builder = RecordingStreamBuilder::new("oakd_xfeat_stereo");
        let rec = if connect {
            builder.connect_grpc()?
        } else {
            builder.save(rrd_path)?
        };
        Ok(Self { rec, image_every })
    }

    /// Log one frame: both eyes, their keypoints, the match count, and whatever IMU
    /// samples arrived alongside.
    pub fn log_frame(
        &self,
        seq: u64,
        frame: &OakStereoFrame<'_>,
        res_l: &XFeatResult,
        res_r: &XFeatResult,
        matches: &[(usize, usize)],
        imu: &[ImuSample],
    ) -> Result<(), BoxError> {
        self.rec.set_time_sequence("frame", seq as i64);
        if let Some(pts) = frame.meta().pts_ns {
            self.rec.set_duration_secs("capture_time", pts as f64 / 1e9);
        }

        // Counters are cheap and the most useful thing to scrub against — a sudden
        // collapse in matches is the signature of a decalibrated or occluded eye.
        self.rec.log(
            "stats/keypoints_left",
            &rerun::Scalars::single(res_l.len() as f64),
        )?;
        self.rec.log(
            "stats/keypoints_right",
            &rerun::Scalars::single(res_r.len() as f64),
        )?;
        self.rec.log(
            "stats/matches",
            &rerun::Scalars::single(matches.len() as f64),
        )?;

        // IMU as time series. Logged per sample at its OWN timestamp, not smeared
        // across the frame, so the trace keeps its true ~200 Hz shape between frames.
        for s in imu {
            self.rec
                .set_duration_secs("capture_time", s.ts_ns as f64 / 1e9);
            for (axis, v) in ["x", "y", "z"].iter().zip(s.accel) {
                self.rec.log(
                    format!("imu/accel/{axis}"),
                    &rerun::Scalars::single(v as f64),
                )?;
            }
            for (axis, v) in ["x", "y", "z"].iter().zip(s.gyro) {
                self.rec.log(
                    format!("imu/gyro/{axis}"),
                    &rerun::Scalars::single(v as f64),
                )?;
            }
        }
        // Restore the frame's own time for the imagery below.
        if let Some(pts) = frame.meta().pts_ns {
            self.rec.set_duration_secs("capture_time", pts as f64 / 1e9);
        }

        if self.image_every == 0 || !seq.is_multiple_of(self.image_every) {
            return Ok(());
        }
        let res = [frame.width(), frame.height()];
        self.rec.log(
            "stereo/left/image",
            &rerun::Image::from_rgb24(frame.left().to_vec(), res),
        )?;
        self.rec.log(
            "stereo/right/image",
            &rerun::Image::from_rgb24(frame.right().to_vec(), res),
        )?;

        // Keypoints, with the matched subset highlighted. Logged in each eye's own
        // image space, so rerun overlays them on the right picture and a mis-scaled
        // coordinate shows up immediately as points sitting off the features.
        let (kl, kr) = (res_l.kpts_to_host()?, res_r.kpts_to_host()?);
        let mut matched_l = vec![false; kl.len() / 2];
        let mut matched_r = vec![false; kr.len() / 2];
        for &(i, j) in matches {
            if let Some(m) = matched_l.get_mut(i) {
                *m = true;
            }
            if let Some(m) = matched_r.get_mut(j) {
                *m = true;
            }
        }
        self.log_keypoints("stereo/left/keypoints", &kl, &matched_l)?;
        self.log_keypoints("stereo/right/keypoints", &kr, &matched_r)?;
        Ok(())
    }

    fn log_keypoints(&self, entity: &str, kpts: &[f32], matched: &[bool]) -> Result<(), BoxError> {
        // Upstream XFeat rescales keypoints back to source pixels, so these are frame
        // coordinates already — no model-space conversion.
        let points: Vec<(f32, f32)> = kpts.chunks_exact(2).map(|c| (c[0], c[1])).collect();
        let colors: Vec<rerun::Color> = matched
            .iter()
            .map(|&m| {
                let c = if m { MATCHED_COLOR } else { UNMATCHED_COLOR };
                rerun::Color::from_rgb(c[0], c[1], c[2])
            })
            .collect();
        let radii: Vec<f32> = matched.iter().map(|&m| if m { 2.0 } else { 1.0 }).collect();
        self.rec.log(
            entity,
            &Points2D::new(points).with_colors(colors).with_radii(radii),
        )?;
        Ok(())
    }
}
