//! Pipeline-building helpers shared by both modalities: the opened [`Session`],
//! camera/encoder/stereo/IMU attachers (each carrying the *degrade* rule of the
//! thing it attaches, so that rule cannot drift between the two open paths), and
//! the calibration readers.

use depthai::node::{Camera, Imu, Node, StereoDepth, VideoEncoder};
use depthai::NodeHandle;
use depthai::{
    CalibrationHandler, CameraBoardSocket, Device, ImgFrame, ImgFrameType, ImgResizeMode, ImuData,
    ImuSensor, LengthUnit, Output, OutputQueue, Pipeline, StereoPresetMode, VideoEncoderProfile,
};

use crate::calib::{OakCameraCalib, OakCameraModel, OakStereoCalib, StereoCalibError};
use crate::policy::{self, Knobs};
use crate::{Ctx, OakError, OakIntrinsics};

/// An opened device with everything both modalities need before building a graph:
/// the pipeline, the populated sockets, the factory calibration (one EEPROM RPC,
/// shared by the stereo check, the IMU-extrinsics gate and the intrinsics) and the
/// `OAK_*` knobs read once.
pub(crate) struct Session {
    pub dev: Device,
    pub pipeline: Pipeline,
    pub cams: Vec<CameraBoardSocket>,
    pub calib: CalibrationHandler,
    pub knobs: Knobs,
}

impl Session {
    /// Connect to an OAK by id (`None`/`""` = first available), honouring
    /// `OAK_USB_SPEED`.
    pub(crate) fn connect(id: Option<&str>) -> Result<Session, OakError> {
        let knobs = Knobs::from_env();
        let id = id.filter(|s| !s.is_empty());
        let dev = Device::open(id, Some(knobs.usb_speed)).ctx("open device")?;
        let pipeline = Pipeline::new(&dev).ctx("create pipeline")?;
        let cams = dev.connected_cameras().ctx("getConnectedCameras")?;
        // A wiped EEPROM comes back as an empty handler (its getters then fail,
        // handled at each use) rather than failing here.
        let calib = dev.read_calibration().ctx("readCalibration")?;
        Ok(Session {
            dev,
            pipeline,
            cams,
            calib,
            knobs,
        })
    }

    pub(crate) fn has(&self, socket: CameraBoardSocket) -> bool {
        self.cams.contains(&socket)
    }

    /// True only if the device can actually produce aligned depth: it has BOTH
    /// stereo mono sockets (CAM_B + CAM_C) AND a readable factory calibration (a
    /// wiped/blank EEPROM reads back fx=0, which makes StereoDepth emit
    /// garbage/zero-scale depth).
    pub(crate) fn can_do_depth(&self, width: u32, height: u32) -> bool {
        self.has(CameraBoardSocket::CamB)
            && self.has(CameraBoardSocket::CamC)
            && read_intrinsics(&self.calib, CameraBoardSocket::CamA, width, height).fx > 0.0
    }

    /// Create a `Camera` node bound to `socket`; a node whose build fails is removed
    /// again so it cannot poison `Pipeline::start`.
    pub(crate) fn camera(&self, socket: CameraBoardSocket) -> depthai::Result<Camera> {
        let cam = self.pipeline.create::<Camera>()?;
        if let Err(e) = cam.build(socket) {
            let _ = self.pipeline.remove(&cam);
            return Err(e);
        }
        Ok(cam)
    }

    /// Remove nodes an optional subgraph created before failing, so the degraded
    /// pipeline can still start.
    pub(crate) fn remove_all(&self, nodes: &[&NodeHandle]) {
        for n in nodes {
            let _ = self.pipeline.remove(*n);
        }
    }
}

/// Attach an NV12 output of `color` → a hardware H.264 encoder, returning the
/// bitstream queue. Shared by the video-only, decoupled RGBD and stereo-viz paths
/// so their encoder settings (BASELINE for Foxglove's decoder, ~4 keyframes/s for
/// fast mid-stream join, OAK_H264_KBPS) can never drift apart.
pub(crate) fn add_h264_encoder(
    s: &Session,
    color: &Camera,
    width: u32,
    height: u32,
    fps: u32,
) -> depthai::Result<OutputQueue<ImgFrame>> {
    let nv12 = color.request_output(
        (width, height),
        Some(ImgFrameType::Nv12),
        ImgResizeMode::Crop,
        Some(fps as f32),
        Some(true),
    )?;
    let enc = s.pipeline.create::<VideoEncoder>()?;
    enc.set_default_profile_preset(fps as f32, VideoEncoderProfile::H264Baseline)?;
    enc.set_keyframe_frequency((fps / 4).max(4) as i32)?;
    enc.set_bitrate_kbps(s.knobs.h264_kbps)?;
    nv12.link(&enc.input()?)?;
    enc.bitstream()?.create_output_queue(30, false)
}

/// The OPTIONAL attach: same encoder settings, but a board that rejects the NV12
/// output must DEGRADE (no video queue) rather than fail the whole open. `ctx`
/// names the modality in the log.
pub(crate) fn try_add_h264_encoder(
    s: &Session,
    color: &Camera,
    width: u32,
    height: u32,
    fps: u32,
    ctx: &str,
) -> Option<OutputQueue<ImgFrame>> {
    add_h264_encoder(s, color, width, height, fps)
        .map_err(|e| {
            degrade!("H.264 viz stream unavailable on the {ctx} path ({e}) — continuing without it")
        })
        .ok()
}

/// StereoDepth (CAM_B/CAM_C) aligned to `rgb_out`'s grid, downscaled on-device,
/// returning the depth queue. On failure every node it created is removed again.
pub(crate) fn add_stereo_depth(
    s: &Session,
    rgb_out: &Output<ImgFrame>,
    width: u32,
    height: u32,
    dfps: u32,
) -> depthai::Result<OutputQueue<ImgFrame>> {
    let left = s.camera(CameraBoardSocket::CamB)?;
    let right = match s.camera(CameraBoardSocket::CamC) {
        Ok(r) => r,
        Err(e) => {
            s.remove_all(&[left.handle()]);
            return Err(e);
        }
    };
    let stereo = match s.pipeline.create::<StereoDepth>() {
        Ok(n) => n,
        Err(e) => {
            s.remove_all(&[left.handle(), right.handle()]);
            return Err(e);
        }
    };
    let wire = || -> depthai::Result<OutputQueue<ImgFrame>> {
        // ROBOTICS preset (depthai v3) is tuned for mobile-robot people/obstacle
        // depth. Subpixel gives ~8× finer disparity (removes the z-quantization
        // that flickers a standing person's depth) but ~halves the stereo FPS;
        // OAK_SUBPIXEL=0 trades precision for rate. LR-check on for occlusion.
        stereo.set_default_profile_preset(StereoPresetMode::Robotics)?;
        stereo.set_left_right_check(true)?;
        stereo.set_subpixel(s.knobs.subpixel)?;
        // Passive-stereo depth cleanup (no IR projector): SPATIAL edge-preserving
        // hole-fill + TEMPORAL averaging + THRESHOLD clamp to the useful range
        // (0.4 m .. 8 m).
        stereo
            .post_processing()
            .set_spatial_filter_enable(true)?
            .set_temporal_filter_enable(true)?
            .set_threshold_filter(400, 8000)?;
        left.request_output(
            (640, 400),
            None,
            ImgResizeMode::Crop,
            Some(dfps as f32),
            None,
        )?
        .link(&stereo.left()?)?;
        right
            .request_output(
                (640, 400),
                None,
                ImgResizeMode::Crop,
                Some(dfps as f32),
                None,
            )?
            .link(&stereo.right()?)?;
        // Align depth to the RGB OUTPUT (not just the CAM_A socket), so depth[u,v]
        // matches RGB[u,v] exactly — same CROP, same size, same intrinsics.
        rgb_out.link(&stereo.input_align_to()?)?;
        // Downscale the aligned depth ON-DEVICE before XLink. A room-scale point
        // cloud doesn't need per-RGB-pixel depth, and the full-res depth pull is the
        // dominant XLink cost (it caps the co-hosted H.264 on a PoE link). Default
        // /2 → 1/4 the bytes; still aligned to the RGB grid, so consumers scale
        // coords by (rgb_w / depth_w).
        let (dw, dh) = policy::depth_output_size(width, height, s.knobs.depth_div);
        stereo.set_output_size(dw, dh)?;
        stereo.depth()?.create_output_queue(4, false)
    };
    wire().inspect_err(|_| s.remove_all(&[left.handle(), right.handle(), stereo.handle()]))
}

/// The OPTIONAL attach (e.g. OAK-1: no stereo pair): degrade to colour + video.
pub(crate) fn try_add_stereo_depth(
    s: &Session,
    rgb_out: &Output<ImgFrame>,
    width: u32,
    height: u32,
    dfps: u32,
) -> Option<OutputQueue<ImgFrame>> {
    add_stereo_depth(s, rgb_out, width, height, dfps)
        .map_err(|e| degrade!("StereoDepth unavailable ({e}) — continuing without depth"))
        .ok()
}

/// What [`attach_imu`] produced: the queue (absent = no IMU running) and the
/// validated chip→camera rotation (absent = samples stay in the raw chip frame).
#[derive(Default)]
pub(crate) struct ImuAttach {
    pub queue: Option<OutputQueue<ImuData>>,
    pub rot: Option<[f32; 9]>,
}

/// Attach the on-board IMU (ACCELEROMETER_RAW + GYROSCOPE_RAW at `imu_hz`) on its
/// own queue and load its extrinsics — ALWAYS both, in this order: the rotation is
/// only meaningful if the IMU actually started, and the reference socket differs
/// per modality (CAM_A on RGBD, CAM_B on stereo).
///
/// The IMU is OPTIONAL: not every OAK carries one, and a missing IMU must not cost
/// the image streams — so BEFORE creating the node (and thus before it can ever
/// reach `Pipeline::start`), preflight with `connected_imu()`: a board without one
/// reports `""`/`"NONE"` (or the query fails) and we skip the node. A board that
/// HAS an IMU but whose node cannot be built is a real device error and fails the
/// open, like any other node. Nothing when `imu_hz == 0`.
pub(crate) fn attach_imu(
    s: &Session,
    imu_hz: u32,
    socket: CameraBoardSocket,
) -> Result<ImuAttach, OakError> {
    if imu_hz == 0 {
        return Ok(ImuAttach::default());
    }
    let present = match s.dev.connected_imu() {
        Ok(name) if name.is_empty() || name == "NONE" => {
            degrade!("no on-board IMU (getConnectedIMU={name:?}) — skipping the IMU node");
            false
        }
        Ok(_) => true,
        Err(e) => {
            degrade!("getConnectedIMU failed ({e}) — skipping the IMU node");
            false
        }
    };
    if !present {
        return Ok(ImuAttach::default());
    }
    let queue = add_imu_node(s, imu_hz).ctx("IMU node")?;
    Ok(ImuAttach {
        queue: Some(queue),
        rot: read_imu_rotation(&s.calib, socket),
    })
}

fn add_imu_node(s: &Session, imu_hz: u32) -> depthai::Result<OutputQueue<ImuData>> {
    let imu = s.pipeline.create::<Imu>()?;
    imu.enable_sensor(ImuSensor::AccelerometerRaw, imu_hz)?;
    imu.enable_sensor(ImuSensor::GyroscopeRaw, imu_hz)?;
    // Batch a few reports per message (fewer, larger XLink transfers) but keep the
    // batch small enough that inertial data stays fresh relative to the frames. 5
    // is also the documented maxBatchReports ceiling.
    imu.set_batch_report_threshold(5)?;
    imu.set_max_batch_reports(5)?;
    imu.out()?.create_output_queue(50, false)
}

/// Resolve the IMU-chip → camera-optical rotation from the device calibration so
/// `next_imu` can report samples in the camera frame (what gyro priors / gravity
/// alignment consume). Raw depthai reports are in the IMU chip frame, axis-permuted
/// vs the camera by the board mounting. Falls back to `None` (raw chip frame) when
/// the EEPROM has no IMU link or the stored matrix is not a proper rotation —
/// degrade, never abort, but say WHY on stderr so a tilted gravity gauge is
/// diagnosable without a debugger.
fn read_imu_rotation(calib: &CalibrationHandler, socket: CameraBoardSocket) -> Option<[f32; 9]> {
    let reject = |why: &str| {
        degrade!("IMU extrinsics rejected ({why}) — IMU samples stay in the raw chip frame");
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

/// Read the FULL CAM_B/CAM_C calibration for a host stereo rectifier: per-eye
/// intrinsics at the streamed size, per-eye distortion, and the CALIBRATED
/// left→right extrinsic in METRES.
///
/// depthai's own getters default the translation source and unit differently per
/// method (`getCameraExtrinsics` → calibrated/centimetres, `getBaselineDistance` →
/// spec/centimetres); either mismatch silently rescales the entire reconstruction,
/// so both are passed explicitly. The baseline is derived from this same extrinsic
/// so rotation and baseline can never come from different sources.
///
/// A wiped/blank EEPROM yields an error; the caller decides whether that is fatal
/// (it is, for stereo VIO), so this does not fail the open.
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
    let mut dist = [0f64; 14];
    for (o, &v) in dist.iter_mut().zip(d) {
        *o = v as f64;
    }
    OakCameraCalib {
        fx: k[0][0] as f64,
        fy: k[1][1] as f64,
        cx: k[0][2] as f64,
        cy: k[1][2] as f64,
        dist,
        n_dist: d.len().min(14),
        model: OakCameraModel::from_depthai(model),
    }
}
