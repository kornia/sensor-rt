//! On-board IMU: accelerometer + gyroscope, drained independently of the frames.
//!
//! Kept separate from the image path on purpose. The IMU reports at hundreds of Hz
//! against ~30 Hz frames, so bundling a sample per frame would throw almost all of
//! them away — the caller drains this at whatever rate it likes and pairs samples
//! with frames by timestamp.
//!
//! Both land on the SAME host-synced epoch timeline (depthai's steady clock shifted
//! once per drain batch), so no clock alignment step is needed.

use depthai::ImuPacket;

use crate::{policy, OakSource};

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

/// Convert one depthai packet into a sample on the epoch timeline, rotated by `rot`.
/// `None` = a hole: depthai's `IMUPacket` value-initialises its reports to zeros with a
/// `{0,0}` timestamp, so a packet missing one report would otherwise emit
/// gyro=(0,0,0) "not rotating" (or a boot-epoch stamp) as if it were real data.
///
/// NOTE: do NOT gate on `accuracy == Unreliable` — firmware does not populate the
/// accuracy field for the `*_RAW` streams (measured: every raw report arrives
/// UNRELIABLE, so that gate silenced the stream entirely). The zero-timestamp gate
/// is the effective default-report guard.
pub(crate) fn convert_packet(p: &ImuPacket, rot: &[f32; 9], offset: i128) -> Option<ImuSample> {
    if p.accelerometer.timestamp.is_zero() || p.gyroscope.timestamp.is_zero() {
        return None;
    }
    Some(ImuSample {
        ts_ns: policy::steady_to_epoch_ns(p.accelerometer.timestamp.as_nanos(), offset),
        // chip frame → camera optical frame (see graph::read_imu_rotation). `rot` is
        // identity unless a VALIDATED rotation replaced it, so applying it
        // unconditionally is correct on both paths.
        accel: policy::rotate(rot, p.accelerometer.xyz()),
        gyro: policy::rotate(rot, p.gyroscope.xyz()),
    })
}

impl OakSource {
    /// Whether IMU samples are calibration-rotated into the modality's reference
    /// camera optical frame — CAM_A (colour) on RGBD, CAM_B (left) on stereo —
    /// because the EEPROM carries IMU extrinsics that pass the rotation gate.
    /// `false` = raw IMU-chip frame, axis-permuted vs the camera by the board
    /// mounting — consumers doing gyro priors / gravity alignment should warn and
    /// expect a tilted gauge; the reason (no extrinsics vs rejected matrix) is logged
    /// to stderr at open.
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
    /// [`OakStereoFrame`](crate::OakStereoFrame).
    pub fn next_imu(&mut self, out: &mut Vec<ImuSample>, cap: usize) -> usize {
        let Some(q) = self.imu_q.as_ref() else {
            return 0;
        };
        if cap == 0 {
            return 0;
        }
        // A batch pops destructively, so one that doesn't fit in the caller's budget
        // CANNOT simply be left behind — convert whole batches into the pending
        // buffer, hand out what fits, and keep the remainder (in order).
        let offset = policy::steady_epoch_offset_now(); // one clock pair for the whole drain
        let mut packets: Vec<ImuPacket> = Vec::new();
        while self.imu_pending.len() < cap {
            let batch = match q.try_get() {
                Ok(Some(b)) => b,
                Ok(None) => break,
                Err(e) => {
                    // Surface the first failure instead of reading as a quiet sensor forever.
                    static WARNED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!("sensor-oak: IMU poll failed: {e}");
                    }
                    break;
                }
            };
            packets.clear();
            if let Err(e) = batch.packets_into(&mut packets) {
                eprintln!("sensor-oak: IMU batch unreadable: {e}");
                continue;
            }
            for p in &packets {
                match convert_packet(p, &self.imu_rot, offset) {
                    Some(s) => self.imu_pending.push_back(s),
                    None => {
                        self.imu_ts_skipped += 1;
                        if self.imu_ts_skipped == 1 || self.imu_ts_skipped.is_multiple_of(1000) {
                            eprintln!(
                                "sensor-oak: {} IMU packet(s) skipped (zero-timestamp report hole); \
                                 a count near half the requested rate means the gate, not the \
                                 firmware, sets your IMU rate",
                                self.imu_ts_skipped
                            );
                        }
                    }
                }
            }
        }
        let take = self.imu_pending.len().min(cap);
        out.extend(self.imu_pending.drain(..take));
        take
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use depthai::{ImuAccuracy, ImuRotationVector, ImuVecReport, RawTimestamp};

    fn report(xyz: [f32; 3], ts: RawTimestamp) -> ImuVecReport {
        ImuVecReport {
            x: xyz[0],
            y: xyz[1],
            z: xyz[2],
            timestamp: ts,
            timestamp_device: ts,
            sequence: 0,
            accuracy: ImuAccuracy::Unreliable, // what the firmware sends for *_RAW
        }
    }

    fn packet(acc: [f32; 3], gyr: [f32; 3], ts: RawTimestamp) -> ImuPacket {
        ImuPacket {
            accelerometer: report(acc, ts),
            gyroscope: report(gyr, ts),
            magnetic_field: report([0.0; 3], RawTimestamp::default()),
            rotation_vector: ImuRotationVector {
                i: 0.0,
                j: 0.0,
                k: 0.0,
                real: 1.0,
                accuracy_rad: 0.0,
                timestamp: RawTimestamp::default(),
                timestamp_device: RawTimestamp::default(),
                sequence: 0,
                accuracy: ImuAccuracy::Unreliable,
            },
        }
    }

    #[test]
    fn hole_gate_drops_zero_timestamp_reports_only() {
        let ts = RawTimestamp { sec: 10, nsec: 5 };
        let ok = packet([0.0, 0.0, 9.81], [0.1, 0.0, 0.0], ts);
        let s = convert_packet(&ok, &policy::IDENTITY, 1_000).unwrap();
        assert_eq!(s.ts_ns, 10_000_000_005 + 1_000);
        assert_eq!(s.accel, [0.0, 0.0, 9.81]);
        assert_eq!(s.gyro, [0.1, 0.0, 0.0]);

        let mut hole = ok;
        hole.gyroscope.timestamp = RawTimestamp::default();
        assert!(convert_packet(&hole, &policy::IDENTITY, 0).is_none());

        // Unreliable accuracy alone must NOT drop a sample.
        assert!(convert_packet(&ok, &policy::IDENTITY, 0).is_some());
    }

    #[test]
    fn rotation_is_applied_to_both_vectors() {
        // 90° about z: x -> y.
        let r = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let p = packet(
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            RawTimestamp { sec: 1, nsec: 0 },
        );
        let s = convert_packet(&p, &r, 0).unwrap();
        assert_eq!(s.accel, [0.0, 1.0, 0.0]);
        assert_eq!(s.gyro, [-2.0, 0.0, 0.0]);
    }

    #[test]
    fn clamp_caps_at_bno086_maximum() {
        assert_eq!(clamp_imu_hz(200), 200);
        assert_eq!(clamp_imu_hz(400), 400);
        assert_eq!(clamp_imu_hz(1000), 400);
    }
}
