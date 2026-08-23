//! Factory **stereo** calibration of the CAM_B/CAM_C pair, for a host rectifier.
//!
//! [`OakSource::open_stereo`](crate::OakSource::open_stereo) delivers a *raw* pair — depthai's
//! `Camera` node can only undistort, never rectify (it builds its remap with an identity
//! rectifying rotation; only `StereoDepth` applies R1/R2, and that forces the whole disparity
//! block). A stereo consumer therefore rectifies on the host, and this is the input it needs:
//! per-eye intrinsics at the streamed size, per-eye distortion, and the metric left→right
//! extrinsic.
//!
//! Two conventions are worth stating out loud, because depthai's own defaults get both wrong for
//! this purpose and neither error is visible in the output:
//!
//! * the translation is the **calibrated** one, not the board-design ("spec") one, and
//! * it is in **metres**, not depthai's default centimetres.
//!
//! [`OakStereoCalib::baseline_m`] is derived from the very same extrinsic as the rotation, so the
//! two can never come from different sources — unlike `getBaselineDistance()`, which carries the
//! identical pair of defaults independently.

use crate::{last_error, BoxError, OakSource};

/// Lens model the factory calibration was fitted with. Mirrors `dai::CameraModel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OakCameraModel {
    /// Brown-Conrady pinhole (`dai::CameraModel::Perspective`). The only model a
    /// polynomial-distortion rectifier can consume.
    Perspective,
    /// Fisheye (`dai::CameraModel::Fisheye`).
    Fisheye,
    /// A model this crate does not know, carried through as its raw discriminant so a consumer can
    /// refuse it explicitly rather than mistake it for a pinhole.
    Other(i32),
}

impl OakCameraModel {
    fn from_raw(v: i32) -> Self {
        match v {
            0 => Self::Perspective,
            1 => Self::Fisheye,
            other => Self::Other(other),
        }
    }
}

/// One eye's factory calibration at the streamed resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OakCameraCalib {
    /// Focal length x, pixels.
    pub fx: f64,
    /// Focal length y, pixels.
    pub fy: f64,
    /// Principal point x, pixels.
    pub cx: f64,
    /// Principal point y, pixels.
    pub cy: f64,
    /// Distortion coefficients in **OpenCV order**:
    /// `[k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, taux, tauy]`.
    ///
    /// Note the tangential terms `p1, p2` sit at indices **2 and 3**, between `k2` and `k3` —
    /// mapping this array onto a struct with fields ordered `k1..k6, p1, p2` by position instead of
    /// by name is a silent, plausible-looking distortion error. Entries past `n_dist` are zero.
    pub dist: [f64; 14],
    /// How many coefficients the EEPROM actually carried (the rest of `dist` is zero).
    pub n_dist: usize,
    /// The lens model these coefficients belong to.
    pub model: OakCameraModel,
}

impl OakCameraCalib {
    /// True when the thin-prism (`s1..s4`) and tilt (`taux, tauy`) terms are all zero, i.e. the
    /// calibration is representable by a plain Brown-Conrady `k1..k6, p1, p2` model.
    ///
    /// A rectifier that supports only those eight coefficients must check this: dropping non-zero
    /// prism/tilt terms silently biases every rectified pixel.
    pub fn is_brown_conrady(&self) -> bool {
        self.model == OakCameraModel::Perspective && self.dist[8..].iter().all(|v| *v == 0.0)
    }
}

/// Factory calibration of the whole stereo pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OakStereoCalib {
    /// Resolution the intrinsics are valid at — the size `open_stereo` was asked to stream.
    ///
    /// Compare it against a delivered frame's own dimensions before rectifying: depthai may crop
    /// to a size other than the request, and nothing else cross-checks the two.
    pub width: u32,
    /// See [`width`](Self::width).
    pub height: u32,
    /// Left eye, CAM_B — the stereo reference camera.
    pub left: OakCameraCalib,
    /// Right eye, CAM_C.
    pub right: OakCameraCalib,
    /// Rotation of `X_right = R * X_left + t`, **row-major** `r[row][col]`.
    ///
    /// Row-major matters: handing these rows to a column-major matrix constructor transposes the
    /// rotation, which for a rotation is exactly its inverse — a mirrored rig that still rectifies
    /// to something plausible-looking.
    pub r_left_right: [[f64; 3]; 3],
    /// Translation of `X_right = R * X_left + t`, **metres**.
    pub t_left_right: [f64; 3],
    /// `‖t_left_right‖`, metres. ~0.075 on an OAK-D-S2/PRO/Lite; a value near 7.5 or 0.000075
    /// means a unit conversion went wrong somewhere.
    pub baseline_m: f64,
}

impl OakSource {
    /// The factory calibration of the stereo pair, read once at
    /// [`open_stereo`](OakSource::open_stereo).
    ///
    /// # Errors
    /// On a device opened in the RGBD modality, or one whose EEPROM is wiped/blank. This is
    /// deliberately an error rather than the zeros [`intrinsics`](OakSource::intrinsics) returns:
    /// there is no useful degraded mode for a metric stereo consumer, and a zero baseline would
    /// otherwise reach a rectifier as `NaN` remap tables.
    pub fn stereo_calib(&self) -> Result<OakStereoCalib, BoxError> {
        let mut raw = crate::ffi::OakStereoCalibRaw::default();
        // SAFETY: `self.dev` is a live handle owned by this source, and `raw` is a valid,
        // fully-initialised out-param of exactly the type the shim writes.
        let rc = unsafe { crate::ffi::oak_stereo_calibration(self.dev, &mut raw) };
        if rc != 0 || raw.valid == 0 {
            return Err(last_error("oak_stereo_calibration"));
        }

        let eye = |k: &[f32; 9], d: &[f32; 14], n: i32, model: i32| OakCameraCalib {
            fx: k[0] as f64,
            fy: k[4] as f64,
            cx: k[2] as f64,
            cy: k[5] as f64,
            dist: std::array::from_fn(|i| d[i] as f64),
            n_dist: n.clamp(0, 14) as usize,
            model: OakCameraModel::from_raw(model),
        };

        // The 4x4 is row-major, so the rotation block is rows 0..3 cols 0..3 and the translation is
        // column 3 — indices 3, 7, 11.
        let e = &raw.t_left_right;
        let r_left_right = std::array::from_fn(|i| std::array::from_fn(|j| e[i * 4 + j] as f64));
        let t_left_right = [e[3] as f64, e[7] as f64, e[11] as f64];

        Ok(OakStereoCalib {
            width: raw.width.max(0) as u32,
            height: raw.height.max(0) as u32,
            left: eye(&raw.left_k, &raw.left_dist, raw.left_n_dist, raw.left_model),
            right: eye(
                &raw.right_k,
                &raw.right_dist,
                raw.right_n_dist,
                raw.right_model,
            ),
            r_left_right,
            t_left_right,
            baseline_m: raw.baseline_m as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The `#[repr(C)]` mirror must not drift from the C struct's layout. There is no way to ask the
    // C compiler here, so pin the sizes/offsets the header's field order implies: 2 ints + 18 + 28
    // floats + 4 ints + 16 floats + 1 float + 1 int, all 4-byte, no padding.
    #[test]
    fn raw_calib_layout_is_packed_as_the_header_declares() {
        let expect = 4 * (2 + 9 + 9 + 14 + 14 + 2 + 2 + 16 + 1 + 1);
        assert_eq!(std::mem::size_of::<crate::ffi::OakStereoCalibRaw>(), expect);
        assert_eq!(std::mem::align_of::<crate::ffi::OakStereoCalibRaw>(), 4);
    }

    // p1/p2 live at indices 2,3 (OpenCV order), so a calibration with only tangential terms must
    // NOT look like one with radial terms. Guards the mapping comment on `dist`.
    #[test]
    fn brown_conrady_gate_rejects_prism_and_tilt() {
        let mut c = OakCameraCalib {
            fx: 440.0,
            fy: 440.0,
            cx: 320.0,
            cy: 200.0,
            dist: [0.0; 14],
            n_dist: 14,
            model: OakCameraModel::Perspective,
        };
        c.dist[2] = 1e-3; // p1 — representable
        assert!(c.is_brown_conrady());
        c.dist[8] = 1e-6; // s1 — thin prism, NOT representable
        assert!(!c.is_brown_conrady());

        c.dist[8] = 0.0;
        c.model = OakCameraModel::Fisheye;
        assert!(!c.is_brown_conrady());
    }
}
