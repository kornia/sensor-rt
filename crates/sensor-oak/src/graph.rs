//! Pipeline-building helpers shared by both modalities: the opened [`Session`],
//! the camera/encoder/stereo/IMU attachers (each carrying the *degrade* rule of the
//! thing it attaches, so that rule cannot drift between the two open paths), and
//! [`Session::finish`], the one place a started pipeline becomes an [`OakSource`].

use std::collections::VecDeque;

use depthai::node::{Camera, Gate, Imu, Node, StereoDepth, VideoEncoder};
use depthai::NodeHandle;
use depthai::{
    CalibrationHandler, CameraBoardSocket, Device, GateControl, ImgFrame, ImgFrameType,
    ImgResizeMode, ImuData, ImuSensor, InputQueue, Output, Pipeline, StereoPresetMode,
    VideoEncoderProfile,
};

use crate::calib::{self, OakStereoCalib, StereoCalibError};
use crate::policy::{self, Knobs};
use crate::{Ctx, OakError, OakSource, Queues, Q};

/// An opened device with everything both modalities need before building a graph:
/// the pipeline, the populated sockets, the factory calibration (one EEPROM RPC,
/// shared by the stereo check, the IMU-extrinsics gate and the intrinsics) and the
/// `OAK_*` knobs.
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
    pub(crate) fn connect(id: Option<&str>, knobs: Knobs) -> Result<Session, OakError> {
        let dev = Device::open(policy::device_id(id), Some(knobs.usb_speed)).ctx("open device")?;
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
            && calib::read_intrinsics(&self.calib, CameraBoardSocket::CamA, width, height).fx > 0.0
    }

    /// Create a `Camera` node bound to `socket`; a node whose build fails is removed
    /// again so it cannot poison `Pipeline::start`.
    pub(crate) fn camera(&self, socket: CameraBoardSocket) -> depthai::Result<Camera> {
        let cam = self.pipeline.create::<Camera>()?;
        cam.build(socket)
            .inspect_err(|_| self.remove_all([cam.handle()]))?;
        Ok(cam)
    }

    /// Remove nodes an optional subgraph created before failing, so the degraded
    /// pipeline can still start.
    pub(crate) fn remove_all<'a>(&self, nodes: impl IntoIterator<Item = &'a NodeHandle>) {
        for n in nodes {
            let _ = self.pipeline.remove(n);
        }
    }

    /// The tail every open path shares, in the one order that is safe: attach the
    /// optional IMU (last, so a missing one never reaches `start`), start the
    /// pipeline, then read the intrinsics of `reference` — the camera the IMU is
    /// rotated into and `intrinsics()` reports (CAM_A on RGBD, CAM_B on stereo).
    pub(crate) fn finish(
        self,
        width: u32,
        height: u32,
        queues: Queues,
        imu_hz: u32,
        reference: CameraBoardSocket,
        stereo_calib: Result<OakStereoCalib, StereoCalibError>,
    ) -> Result<OakSource, OakError> {
        let imu = attach_imu(&self, imu_hz, reference)?;
        self.pipeline.start().ctx("pipeline start")?;
        let intr = calib::read_intrinsics(&self.calib, reference, width, height);
        let (imu_q, imu_rot) = imu.map_or((None, None), |(q, rot)| (Some(q), rot));
        Ok(OakSource {
            queues,
            imu_q,
            pipeline: self.pipeline,
            device: self.dev,
            width,
            height,
            seq: 0,
            intr,
            imu_rot,
            imu_pending: VecDeque::new(),
            imu_packets: Vec::new(),
            imu_ts_skipped: 0,
            stereo_calib,
        })
    }
}

/// The on-device H.264 path: the bitstream queue and the control queue of the
/// gate in front of the encoder (takes [`GateControl`]s).
pub(crate) struct H264 {
    pub(crate) queue: Q<ImgFrame>,
    pub(crate) control: InputQueue,
}

/// Attach an NV12 output of `color` → `Gate` → hardware H.264 encoder. Shared by
/// the video-only, decoupled RGBD and stereo-viz paths so their encoder settings
/// (BASELINE for Foxglove's decoder, ~4 keyframes/s for fast mid-stream join,
/// OAK_H264_KBPS) can never drift apart. The gate is always there (a parked
/// device thread + a host input-queue thread) so `set_video_streaming` works on
/// every source with video; `OAK_VIDEO_GATED` only picks its starting state.
pub(crate) fn add_h264_encoder(
    s: &Session,
    color: &Camera,
    width: u32,
    height: u32,
    fps: u32,
) -> depthai::Result<H264> {
    let nv12 = color.request_output(
        (width, height),
        Some(ImgFrameType::Nv12),
        ImgResizeMode::Crop,
        Some(fps as f32),
        Some(true),
    )?;
    let gate = s.pipeline.create::<Gate>()?;
    let enc = s.pipeline.create::<VideoEncoder>()?;
    let wire = || -> depthai::Result<H264> {
        // Same shape as the encoder's own input (3 deep, blocking) so the camera
        // is back-pressured exactly as it was when it fed the encoder directly,
        // instead of the gate's default 1-deep overwrite queue dropping frames.
        let gate_in = gate.input()?;
        gate_in.set_max_size(3)?;
        gate_in.set_blocking(true)?;
        nv12.link(&gate_in)?;
        if s.knobs.video_gated {
            gate.set_initial_config(&GateControl::close()?)?;
        }
        enc.set_default_profile_preset(fps as f32, VideoEncoderProfile::H264Baseline)?;
        enc.set_keyframe_frequency((fps / 4).max(4) as i32)?;
        enc.set_bitrate_kbps(s.knobs.h264_kbps)?;
        gate.output()?.cast::<ImgFrame>().link(&enc.input()?)?;
        Ok(H264 {
            queue: Q::new(enc.bitstream()?.create_output_queue(30, false)?, "video"),
            control: gate.input_control()?.create_input_queue(4, false)?,
        })
    };
    // A half-wired gate/encoder must not poison `Pipeline::start` on the degrade path.
    wire().inspect_err(|_| s.remove_all([gate.handle(), enc.handle()]))
}

/// The OPTIONAL attach: a board that rejects the NV12 output (or has no CAM_A at
/// all) must DEGRADE (no video queue) rather than fail the whole open. `color` is
/// the CAM_A node when the caller already has one, else it is created here. `ctx`
/// names the modality in the log.
pub(crate) fn try_add_h264_encoder(
    s: &Session,
    color: Option<&Camera>,
    width: u32,
    height: u32,
    fps: u32,
    ctx: &str,
) -> Option<H264> {
    if !s.has(CameraBoardSocket::CamA) {
        degrade!("no CAM_A on this board — skipping the H.264 viz stream on the {ctx} path");
        return None;
    }
    let attach = || -> depthai::Result<H264> {
        let own;
        let color = match color {
            Some(c) => c,
            None => {
                own = s.camera(CameraBoardSocket::CamA)?;
                &own
            }
        };
        add_h264_encoder(s, color, width, height, fps)
    };
    attach()
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
) -> depthai::Result<Q<ImgFrame>> {
    let mut made: Vec<NodeHandle> = Vec::new();
    let mut wire = || -> depthai::Result<Q<ImgFrame>> {
        let left = s.camera(CameraBoardSocket::CamB)?;
        made.push(left.handle().clone());
        let right = s.camera(CameraBoardSocket::CamC)?;
        made.push(right.handle().clone());
        let stereo = s.pipeline.create::<StereoDepth>()?;
        made.push(stereo.handle().clone());
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
        let mono = |cam: &Camera| {
            cam.request_output(
                (640, 400),
                None,
                ImgResizeMode::Crop,
                Some(dfps as f32),
                None,
            )
        };
        mono(&left)?.link(&stereo.left()?)?;
        mono(&right)?.link(&stereo.right()?)?;
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
        Ok(Q::new(
            stereo.depth()?.create_output_queue(4, false)?,
            "depth",
        ))
    };
    let r = wire();
    if r.is_err() {
        s.remove_all(&made);
    }
    r
}

/// Attach the on-board IMU (ACCELEROMETER_RAW + GYROSCOPE_RAW at `imu_hz`) on its
/// own queue and load its extrinsics — ALWAYS both, in this order: the rotation is
/// only meaningful if the IMU actually started, and the reference socket differs
/// per modality (CAM_A on RGBD, CAM_B on stereo). `None` = no IMU running.
///
/// The IMU is OPTIONAL: not every OAK carries one, and a missing IMU must not cost
/// the image streams — so BEFORE creating the node (and thus before it can ever
/// reach `Pipeline::start`), preflight with `connected_imu()`: a board without one
/// reports `""`/`"NONE"` (or the query fails) and we skip the node. A board that
/// HAS an IMU but whose node cannot be built is a real device error and fails the
/// open, like any other node. Nothing when `imu_hz == 0`.
/// A running IMU: its queue and the validated chip→camera rotation (`None` =
/// samples stay in the raw chip frame).
type ImuAttach = (Q<ImuData>, Option<[f32; 9]>);

fn attach_imu(
    s: &Session,
    imu_hz: u32,
    socket: CameraBoardSocket,
) -> Result<Option<ImuAttach>, OakError> {
    if imu_hz == 0 {
        return Ok(None);
    }
    match s.dev.connected_imu() {
        Ok(name) if name.is_empty() || name == "NONE" => {
            degrade!("no on-board IMU (getConnectedIMU={name:?}) — skipping the IMU node");
            return Ok(None);
        }
        Err(e) => {
            degrade!("getConnectedIMU failed ({e}) — skipping the IMU node");
            return Ok(None);
        }
        Ok(_) => {}
    }
    let queue = add_imu_node(s, imu_hz).ctx("IMU node")?;
    Ok(Some((queue, calib::read_imu_rotation(&s.calib, socket))))
}

fn add_imu_node(s: &Session, imu_hz: u32) -> depthai::Result<Q<ImuData>> {
    let imu = s.pipeline.create::<Imu>()?;
    imu.enable_sensor(ImuSensor::AccelerometerRaw, imu_hz)?;
    imu.enable_sensor(ImuSensor::GyroscopeRaw, imu_hz)?;
    // Batch a few reports per message (fewer, larger XLink transfers) but keep the
    // batch small enough that inertial data stays fresh relative to the frames. 5
    // is also the documented maxBatchReports ceiling.
    imu.set_batch_report_threshold(5)?;
    imu.set_max_batch_reports(5)?;
    Ok(Q::new(imu.out()?.create_output_queue(50, false)?, "IMU"))
}
