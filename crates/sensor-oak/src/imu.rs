//! On-board IMU: accelerometer + gyroscope, drained independently of the frames.
//!
//! Kept separate from the image path on purpose. The IMU reports at hundreds of Hz
//! against ~30 Hz frames, so bundling a sample per frame would throw almost all of
//! them away — the caller drains this at whatever rate it likes and pairs samples
//! with frames by timestamp.
//!
//! Both land on the SAME host-synced epoch timeline (depthai's steady clock shifted
//! by one offset per drain), so no clock alignment step is needed.

use depthai::ImuPacket;

use crate::{policy, OakSource};

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

/// Convert one depthai packet into a sample on the epoch timeline, rotated into the
/// camera frame by `rot` when the calibration provided one.
/// `None` = a hole: depthai's `IMUPacket` value-initialises its reports to zeros with a
/// `{0,0}` timestamp, so a packet missing one report would otherwise emit
/// gyro=(0,0,0) "not rotating" (or a boot-epoch stamp) as if it were real data.
///
/// NOTE: do NOT gate on `accuracy == Unreliable` — firmware does not populate the
/// accuracy field for the `*_RAW` streams (measured: every raw report arrives
/// UNRELIABLE, so that gate silenced the stream entirely). The zero-timestamp gate
/// is the effective default-report guard.
pub(crate) fn convert_packet(
    p: &ImuPacket,
    rot: Option<&[f32; 9]>,
    offset: i128,
) -> Option<ImuSample> {
    if p.accelerometer.timestamp.is_zero() || p.gyroscope.timestamp.is_zero() {
        return None;
    }
    let frame = |v: [f32; 3]| rot.map_or(v, |r| policy::rotate(r, v));
    Some(ImuSample {
        ts_ns: policy::steady_to_epoch_ns(p.accelerometer.timestamp.as_nanos(), offset),
        accel: frame(p.accelerometer.xyz()),
        gyro: frame(p.gyroscope.xyz()),
    })
}

impl OakSource {
    /// Whether the on-board IMU is running (so [`next_imu`](Self::next_imu)
    /// yields samples). `false` on a board with no IMU — degrade, don't abort.
    pub fn has_imu(&self) -> bool {
        self.imu_q.is_some()
    }

    /// Whether IMU samples are calibration-rotated into the modality's reference
    /// camera optical frame — CAM_A (colour) on RGBD, CAM_B (left) on stereo —
    /// because the EEPROM carries IMU extrinsics that pass the rotation gate.
    /// `false` = raw IMU-chip frame, axis-permuted vs the camera by the board
    /// mounting — consumers doing gyro priors / gravity alignment should warn and
    /// expect a tilted gauge; the reason (no extrinsics vs rejected matrix) is logged
    /// to stderr at open.
    pub fn imu_aligned(&self) -> bool {
        self.imu_rot.is_some()
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
        let start = out.len();
        // Leftovers from an earlier over-budget batch go first, in order.
        let take = self.imu_pending.len().min(cap);
        out.extend(self.imu_pending.drain(..take));

        // One clock pair for the whole drain, taken only if a batch actually arrives.
        let mut offset: Option<i128> = None;
        let mut skipped = 0u64;
        while out.len() - start < cap {
            let Some(batch) = q.pop(None).ok().flatten() else {
                break;
            };
            let offset = *offset.get_or_insert_with(policy::steady_epoch_offset_now);
            // A batch pops destructively: convert all of it, hand out what fits, and
            // keep the remainder (in order) for the next call.
            self.imu_packets.clear();
            if let Err(e) = batch.packets_into(&mut self.imu_packets) {
                degrade!("IMU batch unreadable: {e}");
            }
            for p in &self.imu_packets {
                match convert_packet(p, self.imu_rot.as_ref(), offset) {
                    Some(s) if out.len() - start < cap => out.push(s),
                    Some(s) => self.imu_pending.push_back(s),
                    None => skipped += 1,
                }
            }
        }
        if skipped > 0 {
            self.note_ts_skipped(skipped);
        }
        out.len() - start
    }

    /// Log the running hole count: at the first one, then every 1000th.
    fn note_ts_skipped(&mut self, n: u64) {
        let before = self.imu_ts_skipped;
        self.imu_ts_skipped += n;
        if before == 0 || self.imu_ts_skipped / 1000 != before / 1000 {
            degrade!(
                "{} IMU packet(s) skipped (zero-timestamp report hole); a count near half the \
                 requested rate means the gate, not the firmware, sets your IMU rate",
                self.imu_ts_skipped
            );
        }
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
        let s = convert_packet(&ok, None, 1_000).unwrap();
        assert_eq!(s.ts_ns, 10_000_000_005 + 1_000);
        assert_eq!(s.accel, [0.0, 0.0, 9.81]);
        assert_eq!(s.gyro, [0.1, 0.0, 0.0]);

        let mut hole = ok;
        hole.gyroscope.timestamp = RawTimestamp::default();
        assert!(convert_packet(&hole, None, 0).is_none());

        // Unreliable accuracy alone must NOT drop a sample.
        assert!(convert_packet(&ok, None, 0).is_some());
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
        let s = convert_packet(&p, Some(&r), 0).unwrap();
        assert_eq!(s.accel, [0.0, 1.0, 0.0]);
        assert_eq!(s.gyro, [-2.0, 0.0, 0.0]);
    }
}
