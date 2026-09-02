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
//! two can never come from different sources; [`read_stereo_calib`] passes both choices
//! explicitly. The other calibration readers (intrinsics, IMU rotation) live here too.

use depthai::{CalibrationHandler, CameraBoardSocket, LengthUnit};

use crate::{policy, BoxError, OakIntrinsics, OakSource};

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

/// Pinhole terms of a depthai 3x3.
fn pinhole(k: &[[f32; 3]; 3]) -> OakIntrinsics {
    OakIntrinsics {
        fx: k[0][0],
        fy: k[1][1],
        cx: k[0][2],
        cy: k[1][2],
    }
}

/// One socket's factory intrinsics at the streamed size. A wiped EEPROM or a
/// missing socket yields ZEROS — fine for viewing, so the failure is swallowed.
pub(crate) fn read_intrinsics(
    calib: &CalibrationHandler,
    socket: CameraBoardSocket,
    width: u32,
    height: u32,
) -> OakIntrinsics {
    calib
        .camera_intrinsics(socket, Some((width, height)))
        .map(|k| pinhole(&k))
        .unwrap_or_default()
}

/// Read the FULL CAM_B/CAM_C calibration for a host stereo rectifier: per-eye
/// intrinsics at the streamed size, per-eye distortion, and the CALIBRATED
/// left→right extrinsic in METRES.
///
/// depthai's own getters default the translation source and unit differently per
/// method (`getCameraExtrinsics` → calibrated/centimetres, `getBaselineDistance` →
/// spec/centimetres); either mismatch silently rescales the entire reconstruction,
/// so both are passed explicitly. A wiped/blank EEPROM yields an error; the caller
/// decides whether that is fatal (it is, for stereo VIO), so this does not fail
/// the open.
pub(crate) fn read_stereo_calib(
    calib: &CalibrationHandler,
    width: u32,
    height: u32,
) -> Result<OakStereoCalib, StereoCalibError> {
    let (l, r) = (CameraBoardSocket::CamB, CameraBoardSocket::CamC);
    let read = || -> depthai::Result<OakStereoCalib> {
        let kl = calib.camera_intrinsics(l, Some((width, height)))?;
        let kr = calib.camera_intrinsics(r, Some((width, height)))?;
        let dl = calib.distortion_coefficients(l)?;
        let dr = calib.distortion_coefficients(r)?;
        let ml = calib.distortion_model(l)?;
        let mr = calib.distortion_model(r)?;
        let e = calib.camera_extrinsics(l, r, false, LengthUnit::Meter)?;
        let t = [e[0][3] as f64, e[1][3] as f64, e[2][3] as f64];
        Ok(OakStereoCalib {
            width,
            height,
            left: eye(&kl, &dl, ml),
            right: eye(&kr, &dr, mr),
            r_left_right: std::array::from_fn(|i| std::array::from_fn(|j| e[i][j] as f64)),
            t_left_right: t,
            baseline_m: (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt(),
        })
    };
    let c = read().map_err(|e| StereoCalibError::Unreadable(e.to_string()))?;
    // A present-but-zero extrinsic passes every read above and reaches a rectifier
    // as NaN remap tables — the exact failure OakStereoCalib promises it cannot carry.
    if c.baseline_m <= 0.0 {
        return Err(StereoCalibError::ZeroBaseline);
    }
    Ok(c)
}

fn eye(k: &[[f32; 3]; 3], d: &[f32], model: depthai::CameraModel) -> OakCameraCalib {
    let p = pinhole(k);
    let mut dist = [0f64; 14];
    for (o, &v) in dist.iter_mut().zip(d) {
        *o = v as f64;
    }
    OakCameraCalib {
        fx: p.fx as f64,
        fy: p.fy as f64,
        cx: p.cx as f64,
        cy: p.cy as f64,
        dist,
        n_dist: d.len().min(14),
        model: OakCameraModel::from_depthai(model),
    }
}

/// The IMU-chip → camera-optical rotation from the device calibration, so
/// `next_imu` can report samples in the camera frame (what gyro priors / gravity
/// alignment consume). `None` (logged, with the reason) when the EEPROM has no IMU
/// link or the stored matrix is not a proper rotation: samples then stay in the
/// raw chip frame.
pub(crate) fn read_imu_rotation(
    calib: &CalibrationHandler,
    socket: CameraBoardSocket,
) -> Option<[f32; 9]> {
    let read = || -> Result<[f32; 9], String> {
        // Unit/spec do not matter for the rotation block, but they are mandatory in
        // the wrapper; pass the calibrated one for consistency.
        let t = calib
            .imu_to_camera_extrinsics(socket, false, LengthUnit::Meter)
            .map_err(|e| e.to_string())?;
        let r: [f32; 9] = std::array::from_fn(|i| t[i / 3][i % 3]);
        policy::validate_rotation(&r).map_err(String::from)?;
        Ok(r)
    };
    read()
        .map_err(|why| {
            degrade!("IMU extrinsics rejected ({why}) — IMU samples stay in the raw chip frame")
        })
        .ok()
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
