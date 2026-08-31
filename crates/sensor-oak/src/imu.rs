//! On-board IMU: accelerometer + gyroscope, drained independently of the frames.
//!
//! Kept separate from the image path on purpose. The IMU reports at hundreds of Hz
//! against ~30 Hz frames, so bundling a sample per frame would throw almost all of
//! them away — the caller drains this at whatever rate it likes and pairs samples
//! with frames by timestamp.
//!
//! Both land on the SAME host-synced epoch timeline (the shim shifts depthai's
//! steady clock once per batch), so no clock alignment step is needed.

/// The BNO086 gyro tops out at 400 Hz; a wilder rate makes the firmware's sensor-enable
/// throw at `pipeline.start()`, which fails the WHOLE open — losing the imagery over an
/// IMU rate. Clamped on both open paths rather than surfaced as a device error.
pub(crate) fn clamp_imu_hz(imu_hz: u32) -> u32 {
    if imu_hz > 400 {
        eprintln!("sensor-oak: imu_hz {imu_hz} clamped to 400 (BNO086 gyro maximum)");
        return 400;
    }
    imu_hz
}

use crate::OakSource;

/// One IMU reading: accelerometer + gyroscope, sampled together.
///
/// `ts_ns` is on the **same host-synced epoch timeline as the image frames**, so
/// samples can be interpolated directly against the stereo frame's timestamp
/// without a clock alignment step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSample {
    /// Capture time, epoch nanoseconds — comparable to `meta().pts_ns` on a stereo frame.
    pub ts_ns: u64,
    /// Accelerometer, m/s².
    pub accel: [f32; 3],
    /// Gyroscope, rad/s.
    pub gyro: [f32; 3],
}

impl OakSource {
    /// Whether IMU samples are calibration-rotated into the modality's reference
    /// camera optical frame — CAM_A (colour) on RGBD, CAM_B (left) on stereo —
    /// because the EEPROM carries IMU extrinsics that pass the shim's rotation
    /// gate. `false` = raw IMU-chip frame, axis-permuted vs the camera by the
    /// board mounting — consumers doing gyro priors / gravity alignment should
    /// warn and expect a tilted gauge; the shim logs the reason (no extrinsics
    /// vs rejected matrix) to stderr at open.
    pub fn imu_aligned(&self) -> bool {
        self.imu_aligned
    }

    /// Drain queued IMU samples, appending them to `out` in capture order;
    /// returns how many were appended. Non-blocking.
    ///
    /// The IMU reports far faster than the frame rate (hundreds of Hz vs ~30),
    /// which is why it is drained separately rather than folded into the synced
    /// pair — bundling it per frame would throw away all but one sample. Call it
    /// once per frame (or more often); samples that don't fit `cap` stay queued
    /// in order for the next call, so nothing is dropped.
    ///
    /// Takes `&mut self` like every other poll, so drain it **outside** a held
    /// [`OakStereoFrame`].
    pub fn next_imu(&mut self, out: &mut Vec<ImuSample>, cap: usize) -> usize {
        if !self.has_imu || cap == 0 {
            return 0;
        }
        // Reused staging buffer: this runs every frame, and a fresh `vec![default; cap]`
        // would be a `cap * 32`-byte alloc + memset each time (16 KB at the usual
        // cap = 512) to receive the handful of samples that actually arrived.
        if self.imu_scratch.len() < cap {
            self.imu_scratch
                .resize(cap, crate::ffi::OakImuSample::default());
        }
        let mut n: i32 = 0;
        let rc = unsafe {
            crate::ffi::oak_poll_imu(self.dev, self.imu_scratch.as_mut_ptr(), cap as i32, &mut n)
        };
        if rc != 1 {
            // Surface the first shim failure instead of reading as a quiet sensor forever;
            // rc == -1 is an exception inside oak_poll_imu, not "nothing queued".
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "sensor-oak: oak_poll_imu failed: {}",
                    crate::last_error("oak_poll_imu")
                );
            }
            return 0;
        }
        if n <= 0 {
            return 0;
        }
        let n = (n as usize).min(cap);
        out.extend(self.imu_scratch[..n].iter().map(|s| ImuSample {
            ts_ns: s.ts_ns,
            accel: [s.ax, s.ay, s.az],
            gyro: [s.gx, s.gy, s.gz],
        }));
        n
    }
}
