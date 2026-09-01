//! Pipeline-building helpers shared by both modalities: device connection,
//! the optional H.264 encoder, the optional IMU (with its extrinsics gate), and
//! the calibration readers. Each carries the *degrade* rule of the thing it
//! attaches, so that rule cannot drift between the two open paths.

use depthai::node::{Camera, Imu, VideoEncoder};
use depthai::{
    CalibrationHandler, CameraBoardSocket, Device, ImgFrame, ImgFrameType, ImgResizeMode, ImuData,
    ImuSensor, LengthUnit, OutputQueue, Pipeline, VideoEncoderProfile,
};

use crate::calib::{OakCameraCalib, OakCameraModel, OakStereoCalib};
use crate::policy;
use crate::{BoxError, OakIntrinsics};

/// Connect to an OAK by id (`None` = first available), honouring `OAK_USB_SPEED`.
pub(crate) fn connect_device(id: Option<&str>) -> Result<Device, BoxError> {
    let id = id.filter(|s| !s.is_empty());
    Device::open(id, Some(policy::usb_speed_from_env()))
        .map_err(|e| format!("open device failed: {e}").into())
}

/// Attach an NV12 output of `color` → a hardware H.264 encoder, returning the
/// bitstream queue. Shared by the video-only, decoupled RGBD and stereo-viz paths
/// so their encoder settings (BASELINE for Foxglove's decoder, ~4 keyframes/s for
/// fast mid-stream join, OAK_H264_KBPS) can never drift apart.
pub(crate) fn add_h264_encoder(
    pipeline: &Pipeline,
    color: &Camera,
    width: u32,
    height: u32,
    fps: u32,
) -> depthai::Result<OutputQueue<ImgFrame>> {
    let fps = fps.max(1);
    let nv12 = color.request_output(
        (width, height),
        Some(ImgFrameType::Nv12),
        ImgResizeMode::Crop,
        Some(fps as f32),
        Some(true),
    )?;
    let enc = pipeline.create::<VideoEncoder>()?;
    enc.set_default_profile_preset(fps as f32, VideoEncoderProfile::H264Baseline)?;
    enc.set_keyframe_frequency((fps / 4).max(4) as i32)?;
    enc.set_bitrate_kbps(policy::h264_kbps())?;
    nv12.link(&enc.input()?)?;
    enc.bitstream()?.create_output_queue(30, false)
}

/// The OPTIONAL attach: same encoder settings, but a board that rejects the colour
/// node or the requested NV12 output must DEGRADE (no video queue) rather than fail
/// the whole open. `ctx` names the modality in the log.
pub(crate) fn try_add_h264_encoder(
    pipeline: &Pipeline,
    color: &Camera,
    width: u32,
    height: u32,
    fps: u32,
    ctx: &str,
) -> Option<OutputQueue<ImgFrame>> {
    match add_h264_encoder(pipeline, color, width, height, fps) {
        Ok(q) => Some(q),
        Err(e) => {
            eprintln!(
                "sensor-oak: H.264 viz stream unavailable on the {ctx} path ({e}) — continuing without it"
            );
            None
        }
    }
}

/// Attach the on-board IMU (ACCELEROMETER_RAW + GYROSCOPE_RAW at `imu_hz`) on its
/// own queue. The IMU is OPTIONAL: not every OAK carries one, and a missing IMU must
/// not cost the image streams — so BEFORE creating the node (and thus before it can
/// ever reach `Pipeline::start`), preflight with `connected_imu()`: a board without
/// one reports `""`/`"NONE"` and we skip the node. `None` when `imu_hz == 0`.
pub(crate) fn add_imu_node(
    device: &Device,
    pipeline: &Pipeline,
    imu_hz: u32,
) -> Option<OutputQueue<ImuData>> {
    if imu_hz == 0 {
        return None;
    }
    let name = match device.connected_imu() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("sensor-oak: getConnectedIMU failed ({e}) — skipping the IMU node");
            return None;
        }
    };
    if name.is_empty() || name == "NONE" {
        eprintln!("sensor-oak: no on-board IMU (getConnectedIMU={name:?}) — skipping the IMU node");
        return None;
    }
    let build = || -> depthai::Result<OutputQueue<ImuData>> {
        let imu = pipeline.create::<Imu>()?;
        imu.enable_sensor(ImuSensor::AccelerometerRaw, imu_hz)?;
        imu.enable_sensor(ImuSensor::GyroscopeRaw, imu_hz)?;
        // Batch a few reports per message (fewer, larger XLink transfers) but keep
        // the batch small enough that inertial data stays fresh relative to the
        // frames. 5 is also the documented maxBatchReports ceiling.
        imu.set_batch_report_threshold(5)?;
        imu.set_max_batch_reports(5)?;
        imu.out()?.create_output_queue(50, false)
    };
    match build() {
        Ok(q) => Some(q),
        Err(e) => {
            eprintln!("sensor-oak: IMU node setup failed ({e}) — skipping the IMU node");
            None
        }
    }
}

/// Resolve the IMU-chip → camera-optical rotation from the device calibration so
/// `next_imu` can report samples in the camera frame (what gyro priors / gravity
/// alignment consume). Raw depthai reports are in the IMU chip frame, axis-permuted
/// vs the camera by the board mounting. Falls back to `None` (raw chip frame) when
/// the EEPROM has no IMU link or the stored matrix is not a proper rotation —
/// degrade, never abort, but say WHY on stderr so a tilted gravity gauge is
/// diagnosable without a debugger.
pub(crate) fn read_imu_rotation(
    calib: &CalibrationHandler,
    socket: CameraBoardSocket,
) -> Option<[f32; 9]> {
    let reject = |why: &str| {
        eprintln!(
            "sensor-oak: IMU extrinsics rejected ({why}) — IMU samples stay in the raw chip frame"
        );
    };
    // Unit/spec do not matter for the rotation block, but they are mandatory in
    // the wrapper; pass the calibrated one for consistency.
    let t = match calib.imu_to_camera_extrinsics(socket, false, LengthUnit::Meter) {
        Ok(t) => t,
        Err(e) => {
            // No IMU extrinsics in the EEPROM (or the read failed) — raw chip frame.
            reject(&e.to_string());
            return None;
        }
    };
    let r: [f32; 9] = std::array::from_fn(|i| t[i / 3][i % 3]);
    match policy::validate_rotation(&r) {
        Ok(()) => Some(r),
        Err(why) => {
            reject(why);
            None
        }
    }
}

/// Attach the IMU and load its extrinsics — ALWAYS both, in this order: the
/// rotation is only meaningful if the IMU actually started, and the reference
/// socket differs per modality (CAM_A on RGBD, CAM_B on stereo). Returns the queue,
/// the rotation to apply (identity when unaligned) and the `imu_aligned` flag.
pub(crate) fn attach_imu(
    device: &Device,
    pipeline: &Pipeline,
    imu_hz: u32,
    calib: &CalibrationHandler,
    socket: CameraBoardSocket,
) -> (Option<OutputQueue<ImuData>>, [f32; 9], bool) {
    let q = add_imu_node(device, pipeline, imu_hz);
    if q.is_none() {
        return (None, policy::IDENTITY, false);
    }
    match read_imu_rotation(calib, socket) {
        Some(r) => (q, r, true),
        None => (q, policy::IDENTITY, false),
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
    match calib.camera_intrinsics(socket, Some((width, height))) {
        Ok(k) => OakIntrinsics {
            fx: k[0][0],
            fy: k[1][1],
            cx: k[0][2],
            cy: k[1][2],
        },
        Err(_) => OakIntrinsics::default(),
    }
}

/// Is `socket` populated on this device?
pub(crate) fn has_socket(cams: &[CameraBoardSocket], socket: CameraBoardSocket) -> bool {
    cams.contains(&socket)
}

/// True only if the device can actually produce aligned depth: it has BOTH stereo
/// mono sockets (CAM_B + CAM_C) AND a readable factory calibration (a wiped/blank
/// EEPROM reads back fx=0, which makes StereoDepth emit garbage/zero-scale depth).
pub(crate) fn device_has_stereo(
    cams: &[CameraBoardSocket],
    calib: &CalibrationHandler,
    width: u32,
    height: u32,
) -> bool {
    if !has_socket(cams, CameraBoardSocket::CamB) || !has_socket(cams, CameraBoardSocket::CamC) {
        return false;
    }
    calib
        .camera_intrinsics(CameraBoardSocket::CamA, Some((width, height)))
        .map(|k| k[0][0] > 0.0)
        .unwrap_or(false)
}

/// Read the FULL CAM_B/CAM_C calibration for a host stereo rectifier: per-eye
/// intrinsics at the streamed size, per-eye distortion, and the CALIBRATED
/// left→right extrinsic in METRES.
///
/// depthai's own getters default the translation source and unit differently per
/// method (`getCameraExtrinsics` → calibrated/centimetres, `getBaselineDistance` →
/// spec/centimetres); either mismatch silently rescales the entire
/// reconstruction, so both are passed explicitly. The baseline is derived from
/// this same extrinsic so rotation and baseline can never come from different
/// sources.
///
/// A wiped/blank EEPROM yields `Err(reason)`; the caller decides whether that is
/// fatal (it is, for stereo VIO), so this does not fail the open.
pub(crate) fn read_stereo_calib(
    calib: &CalibrationHandler,
    width: u32,
    height: u32,
) -> Result<OakStereoCalib, String> {
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
    let c = read().map_err(|e| format!("stereo calibration unavailable: {e}"))?;
    // A present-but-zero extrinsic passes every read above and reaches a rectifier
    // as NaN remap tables — the exact failure OakStereoCalib promises it cannot carry.
    if c.baseline_m <= 0.0 {
        return Err("stereo calibration has a zero baseline (extrinsic present but blank)".into());
    }
    Ok(c)
}

fn eye(k: &[[f32; 3]; 3], d: &[f32], model: depthai::CameraModel) -> OakCameraCalib {
    let n_dist = d.len().min(14);
    let mut dist = [0f64; 14];
    for (i, v) in d.iter().take(n_dist).enumerate() {
        dist[i] = *v as f64;
    }
    OakCameraCalib {
        fx: k[0][0] as f64,
        fy: k[1][1] as f64,
        cx: k[0][2] as f64,
        cy: k[1][2] as f64,
        dist,
        n_dist,
        model: OakCameraModel::from_depthai(model),
    }
}
