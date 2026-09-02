//! Factory **stereo** calibration of the CAM_B/CAM_C pair, for a host rectifier.
//!
//! [`OakSource::open_stereo`](crate::OakSource::open_stereo) delivers a *raw* pair — depthai's
//! `Camera` node can only undistort, never rectify (it builds its remap with an identity
//! rectifying rotation; only `StereoDepth` applies R1/R2, and that forces the whole disparity
//! block). A stereo consumer therefore rectifies on the host, and this is the input it needs:
//! per-eye intrinsics at the streamed size, per-eye distortion, and the metric left→right
//! extrinsic.
//!
//! Two conventions are worth stating out loud, because depthai's getters default them
//! inconsistently (`getCameraExtrinsics` → calibrated + centimetres,
//! `getBaselineDistance` → spec + centimetres) and neither error is visible in the output:
//!
//! * the translation is the **calibrated** one, not the board-design ("spec") one, and
//! * it is in **metres**, not depthai's default centimetres.
//!
//! [`OakStereoCalib::baseline_m`] is derived from the very same extrinsic as the rotation, so the
//! two can never come from different sources. The read itself lives in
//! `graph::read_stereo_calib`, where both choices are passed explicitly.

use crate::{BoxError, OakSource};

/// Why a stereo calibration is not available.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum StereoCalibError {
    #[error("not a stereo device (opened with open_rgbd)")]
    NotStereo,
    #[error("stereo calibration unavailable: {0}")]
    Unreadable(String),
    #[error("stereo calibration has a zero baseline (extrinsic present but blank)")]
    ZeroBaseline,
}

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
    pub(crate) fn from_depthai(m: depthai::CameraModel) -> Self {
        match m {
            depthai::CameraModel::Perspective => Self::Perspective,
            depthai::CameraModel::Fisheye => Self::Fisheye,
            other => Self::Other(other.to_raw()),
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
        self.stereo_calib.clone().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn camera_model_maps_unknowns_to_other() {
        assert_eq!(
            OakCameraModel::from_depthai(depthai::CameraModel::Perspective),
            OakCameraModel::Perspective
        );
        assert_eq!(
            OakCameraModel::from_depthai(depthai::CameraModel::Fisheye),
            OakCameraModel::Fisheye
        );
        assert_eq!(
            OakCameraModel::from_depthai(depthai::CameraModel::Equirectangular),
            OakCameraModel::Other(2)
        );
    }
}
