//! OAK-D **RGBD + H.264** modality: CAM_A colour (RGB888) + `StereoDepth` aligned to
//! it (uint16 mm) + an on-device H.264 colour stream.
//!
//! This is the colour/depth counterpart to the raw [`stereo`](crate::stereo) path —
//! the source a camera producer publishes from. The three outputs are
//! **decoupled**: colour, depth, and encoded video each come out of their own queue
//! ([`next_rgb`](OakSource::next_rgb) / [`next_depth`](OakSource::next_depth) /
//! [`next_video`](OakSource::next_video)), pulled independently and paired downstream
//! by their shared host-synced timestamps. That frees depth from the raw-RGB pull
//! rate and lets the small H.264 stream ship at full fps while the heavy raw RGBD is
//! decimated.
//!
//! **Host-only, no CUDA, no `vrt`.** Frames come out as owned host buffers (each
//! `next_*` copies its bytes out of the device frame); the consumer owns any GPU
//! upload. A producer that only encodes and republishes builds no GPU stack at all.

use depthai::node::{Camera, StereoDepth};
use depthai::{
    CameraBoardSocket, ImgFrame, ImgFrameType, ImgResizeMode, Pipeline, StereoPresetMode,
};

use crate::{graph, policy, BoxError, OakSource, Queues};

impl OakSource {
    /// Open an OAK in the RGBD + H.264 modality: CAM_A colour (RGB888) + an on-device H.264 colour
    /// stream, plus aligned `StereoDepth` when `depth` is set. `device`: `None` = first available;
    /// `Some(id)` = a specific MxId (USB or PoE) or IP (PoE).
    ///
    /// Set `depth = false` for an **uncalibrated** camera — the `StereoDepth` node would otherwise fail
    /// at runtime and crash the pipeline. Even with `depth = true` the device auto-falls-back to
    /// video-only if it can't actually produce depth (mono, or blank calibration); check
    /// [`has_sync`](Self::has_sync) after opening to pick the drain loop.
    ///
    /// `imu_hz > 0` also runs the on-board IMU (accel + gyro) on its own queue, drained with
    /// [`next_imu`](Self::next_imu) on the same host-epoch timeline as the frames; `0` disables it.
    /// The driver preflights with `connected_imu()` and only builds the IMU node when the board
    /// actually carries one, so an IMU-less board degrades ([`has_imu`](Self::has_imu) is `false`)
    /// without ever risking the image streams — never an error. Rates above 400 Hz (the BNO086
    /// gyro maximum) are clamped. When the EEPROM carries valid IMU extrinsics, samples come out
    /// in the CAM_A optical frame — check [`imu_aligned`](Self::imu_aligned); an absent or
    /// rejected calibration is logged to stderr with the reason.
    pub fn open_rgbd(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        depth: bool,
        imu_hz: u32,
    ) -> Result<Self, BoxError> {
        Self::open_rgbd_inner(device, width, height, fps, depth, false, imu_hz)
    }

    /// Open **video-only**: build ONLY the on-device H.264 encoder — no RGB888/depth output — so the
    /// device transmits just the small bitstream (low-bandwidth viewing over USB2 / a shared gigabit
    /// link, where raw RGBD would saturate it). [`next_rgb`](Self::next_rgb) / [`next_depth`](Self::next_depth)
    /// yield nothing; drain [`next_video`](Self::next_video). `device` and `imu_hz`: see
    /// [`open_rgbd`](Self::open_rgbd) — the IMU runs fine alongside the video-only pipeline.
    pub fn open_rgbd_video(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        imu_hz: u32,
    ) -> Result<Self, BoxError> {
        Self::open_rgbd_inner(device, width, height, fps, false, true, imu_hz)
    }

    fn open_rgbd_inner(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        depth: bool,
        video_only: bool,
        imu_hz: u32,
    ) -> Result<Self, BoxError> {
        let fps = policy::fps_or_default(fps);
        let imu_hz = crate::imu::clamp_imu_hz(imu_hz);
        // No open-retry here: the IMU is preflighted before the node is built, so an
        // IMU-less board already degrades inside ONE open. A failure at this point is a
        // real device error, and retrying (especially with `device = None`) could
        // silently bind a different physical camera on a multi-OAK rig.
        let dev = graph::connect_device(device)?;
        let pipeline = Pipeline::new(&dev).map_err(|e| format!("pipeline failed: {e}"))?;
        let cams = dev
            .connected_cameras()
            .map_err(|e| format!("getConnectedCameras failed: {e}"))?;
        // One EEPROM read shared by the stereo check, the IMU-extrinsics gate, and the
        // intrinsics.
        let calib = dev
            .read_calibration()
            .map_err(|e| format!("readCalibration failed: {e}"))?;

        // Auto-fall-back to video-only when depth was requested but the device can't
        // actually produce it (mono camera, or wiped/blank calibration → fx=0). Pulling
        // raw RGB over XLink for a "synced RGBD" pair whose depth is garbage just caps
        // the H.264 stream for nothing — build the lean video-only pipeline instead.
        // Policy: always ship compressed video; add RGBD only when depth works.
        let want_video_only =
            video_only || (depth && !graph::device_has_stereo(&cams, &calib, width, height));

        let queues = if want_video_only {
            Self::build_video_only(&pipeline, width, height, fps)?
        } else {
            Self::build_decoupled(&pipeline, &cams, width, height, fps, depth)?
        };

        let imu = graph::attach_imu(&dev, &pipeline, imu_hz, &calib, CameraBoardSocket::CamA);
        pipeline
            .start()
            .map_err(|e| format!("open_rgbd failed: pipeline start: {e}"))?;
        // Factory intrinsics of the aligned RGB camera.
        let intr = graph::read_intrinsics(&calib, CameraBoardSocket::CamA, width, height);

        if !want_video_only {
            // IR dot projector: passive stereo starves on texture-poor / dim scenes
            // (single-digit valid-depth %). Default 0.8 intensity; OAK_IR=0 disables
            // (e.g. multi-cam cross-talk), boards without a projector just return false.
            // Set after start() — needs a live device.
            let ir = policy::ir_intensity();
            if ir > 0.0 {
                // Ok(false) = no projector on this board (fine). Err = a real device
                // RPC failure: degrade, but say so.
                if let Err(e) = dev.set_ir_laser_dot_projector_intensity(ir, None) {
                    eprintln!("sensor-oak: IR dot-projector intensity failed ({e}) — continuing without it");
                }
            }
        }

        Ok(Self::assemble(
            dev,
            pipeline,
            width,
            height,
            intr,
            queues,
            imu,
            Err("no stereo calibration (not a stereo device — opened with open_rgbd)".into()),
        ))
    }

    /// VIDEO-ONLY: just the H.264 encoder (CAM_A NV12 → encoder → queue). No
    /// RGB888/depth output, so the device transmits ONLY the small H.264 bitstream.
    fn build_video_only(
        pipeline: &Pipeline,
        width: u32,
        height: u32,
        fps: u32,
    ) -> Result<Queues, BoxError> {
        let color = pipeline
            .create::<Camera>()
            .map_err(|e| format!("open_rgbd failed: colour camera: {e}"))?;
        color
            .build(CameraBoardSocket::CamA)
            .map_err(|e| format!("open_rgbd failed: colour camera: {e}"))?;
        // MANDATORY here — the stream is this modality's whole output — but this path
        // is also where an RGBD request LANDS when the board cannot do depth, so an
        // encoder failure must say that rather than surface as a bare device error.
        let video = graph::try_add_h264_encoder(pipeline, &color, width, height, fps, "video-only")
            .ok_or("H.264 encoder unavailable and this device cannot produce depth — nothing left to stream")?;
        // No rgb/depth queues on this path: has_sync + has_depth report false because
        // the queues are absent.
        Ok(Queues {
            video: Some(video),
            ..Default::default()
        })
    }

    /// DECOUPLED build: raw-RGB and depth are SEPARATE streams (no on-device Sync
    /// node), each pulled at its own rate + timestamped, so the consumer pairs them by
    /// timestamp. Depth runs at the mono/stereo rate (OAK_DEPTH_FPS, default = fps);
    /// raw RGB — needed only for local compute — is pulled at a LOW rate (OAK_RGB_FPS,
    /// default 10) to spare XLink. The H.264 video is a separate full-fps output,
    /// unaffected by either.
    fn build_decoupled(
        pipeline: &Pipeline,
        cams: &[CameraBoardSocket],
        width: u32,
        height: u32,
        fps: u32,
        depth: bool,
    ) -> Result<Queues, BoxError> {
        let dfps = policy::depth_fps(fps);
        let rfps = policy::rgb_fps(fps);

        let build = || -> depthai::Result<Queues> {
            // Colour (CAM_A). Interleaved RGB888 (the ISP's native type) as the raw-RGB
            // stream AND the depth-alignment reference; undistorted so depth aligns
            // pixel-perfect and the intrinsics are an exact pinhole.
            let color = pipeline.create::<Camera>()?;
            color.build(CameraBoardSocket::CamA)?;
            let rgb_out = color.request_output(
                (width, height),
                Some(ImgFrameType::Rgb888i),
                ImgResizeMode::Crop,
                Some(rfps as f32),
                Some(true),
            )?;

            // StereoDepth aligned to the RGB OUTPUT (not just the CAM_A socket), so
            // depth[u,v] matches RGB[u,v] exactly — same CROP, same size, same intrinsics.
            let mut depth_q = None;
            if depth {
                let stereo = (|| -> depthai::Result<StereoDepth> {
                    let left = pipeline.create::<Camera>()?;
                    left.build(CameraBoardSocket::CamB)?;
                    let right = pipeline.create::<Camera>()?;
                    right.build(CameraBoardSocket::CamC)?;
                    let stereo = pipeline.create::<StereoDepth>()?;
                    // ROBOTICS preset (depthai v3) is tuned for mobile-robot
                    // people/obstacle depth. Subpixel gives ~8× finer disparity (removes
                    // the z-quantization that flickers a standing person's depth) but
                    // ~halves the stereo FPS; OAK_SUBPIXEL=0 trades precision for rate.
                    // LR-check on for occlusion.
                    stereo.set_default_profile_preset(StereoPresetMode::Robotics)?;
                    stereo.set_left_right_check(true)?;
                    stereo.set_subpixel(policy::subpixel())?;
                    // Passive-stereo depth cleanup (no IR projector): SPATIAL
                    // edge-preserving hole-fill + TEMPORAL averaging + THRESHOLD clamp to
                    // the useful range (0.4 m .. 8 m).
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
                    rgb_out.link(&stereo.input_align_to()?)?; // align depth to the RGB output grid
                                                              // Downscale the aligned depth ON-DEVICE before XLink. A room-scale
                                                              // point cloud doesn't need per-RGB-pixel depth, and the full-res depth
                                                              // pull is the dominant XLink cost (it caps the co-hosted H.264 on a
                                                              // PoE link). Default /2 → 1/4 the bytes; still aligned to the RGB grid,
                                                              // so consumers scale coords by (rgb_w / depth_w).
                    let (dw, dh) = policy::depth_output_size(width, height, policy::depth_div());
                    stereo.set_output_size(dw, dh)?;
                    Ok(stereo)
                })();
                match stereo {
                    // The queue IS the capability: creating it is what makes has_depth true.
                    Ok(stereo) => depth_q = Some(stereo.depth()?.create_output_queue(4, false)?),
                    Err(e) => {
                        // e.g. OAK-1: no stereo pair — degrade to colour + video.
                        eprintln!(
                            "sensor-oak: StereoDepth unavailable ({e}) — continuing without depth"
                        );
                    }
                }
            }
            let _ = cams; // socket presence is already folded into `device_has_stereo`

            // Each stream to its OWN non-blocking queue, pulled + published
            // independently (consumer pairs by timestamp).
            let rgb_q = rgb_out.create_output_queue(4, false)?;

            // H.264 is always on in this modality — the whole point is the efficient
            // colour stream — but OPTIONAL in the sense that a board that rejects the
            // NV12 output must not cost the caller depth + RGB. No capability query
            // exists for the encoder the way connected_imu() preflights the IMU, so the
            // shared helper attempts and catches.
            let video = graph::try_add_h264_encoder(pipeline, &color, width, height, fps, "RGBD");
            Ok(Queues {
                rgb: Some(rgb_q),
                depth: depth_q,
                video,
                ..Default::default()
            })
        };
        build().map_err(|e| format!("open_rgbd failed: {e}").into())
    }

    /// Whether `StereoDepth` is running (so [`next_depth`](Self::next_depth) can yield aligned depth).
    pub fn has_depth(&self) -> bool {
        self.depth_q.is_some()
    }

    /// Whether the on-device H.264 colour stream is running (so [`next_video`](Self::next_video) yields).
    pub fn has_video(&self) -> bool {
        self.video_q.is_some()
    }

    /// Whether this device runs the colour(+depth) pipeline (so [`next_rgb`](Self::next_rgb) yields).
    /// `false` means it auto-fell-back to video-only (mono / uncalibrated): drain only
    /// [`next_video`](Self::next_video). Always check this after [`open_rgbd`](Self::open_rgbd).
    pub fn has_sync(&self) -> bool {
        self.rgb_q.is_some()
    }

    /// Decoupled raw-colour poll: the next RGB888 frame from its own queue (non-blocking), copied out
    /// with its dims + capture timestamp (ns). `None` when none is queued. Independent of
    /// [`next_depth`](Self::next_depth) — pair them by timestamp on the consumer. Drain in a loop until
    /// `None` each iteration.
    pub fn next_rgb(&mut self) -> Option<(Vec<u8>, u32, u32, u64)> {
        let q = self.rgb_q.as_ref()?;
        let f = poll(q, "rgb")?;
        let (w, h) = (f.width(), f.height());
        if w == 0 || h == 0 {
            return None; // degenerate frame — skip, don't kill the stream
        }
        let npx = (w as usize) * (h as usize);
        let stride = f.stride();
        if (stride != 0 && stride != w * 3) || f.data().len() < npx * 3 {
            eprintln!("sensor-oak: rgb frame is not tightly packed RGB888 (stride != w*3)");
            return None;
        }
        let bytes = f.data()[..npx * 3].to_vec();
        Some((bytes, w, h, epoch_ns(&f)))
    }

    /// Decoupled depth poll: the next aligned uint16-mm depth frame from its own queue (non-blocking) at
    /// the stereo rate, copied out with its dims (may be `<` colour size) + capture timestamp. `None`
    /// when none is queued. Drain in a loop until `None`.
    pub fn next_depth(&mut self) -> Option<(Vec<u16>, u32, u32, u64)> {
        let q = self.depth_q.as_ref()?;
        let d = poll(q, "depth")?;
        let (dw, dh) = (d.width(), d.height());
        if dw == 0 || dh == 0 {
            return None;
        }
        let ts = epoch_ns(&d);
        let vals = repack_depth(d.data(), dw, dh, d.stride(), &mut self.depth_repack)?;
        Some((vals, dw, dh, ts))
    }

    /// Drain the next on-device **H.264** access unit, if one is ready (non-blocking): the encoded bytes
    /// (copied out, so the caller may hold/publish them freely) + the capture timestamp in ns. `None`
    /// when no frame is queued or H.264 isn't running. Call in a loop until `None` each iteration so the
    /// encoder queue never overflows — a dropped P-frame glitches the stream until the next keyframe.
    pub fn next_video(&mut self) -> Option<(Vec<u8>, u64)> {
        let q = self.video_q.as_ref()?;
        let f = poll(q, "video")?;
        if f.data().is_empty() {
            return None;
        }
        Some((f.data().to_vec(), epoch_ns(&f)))
    }
}

fn poll(q: &depthai::OutputQueue<ImgFrame>, what: &str) -> Option<ImgFrame> {
    match q.try_get() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("sensor-oak: {what} poll failed: {e}");
            None
        }
    }
}

fn epoch_ns(f: &ImgFrame) -> u64 {
    policy::steady_to_epoch_ns(f.timestamp_ns(), policy::steady_epoch_offset_cached())
}

/// Copy a `RAW16` depth frame out as tightly packed `u16`s. A downscaled/aligned
/// depth frame is often padded to a byte-alignment boundary (stride > dw*2); honour
/// the stride by repacking row-by-row instead of dropping every such frame — a
/// tight-only check silently left depth permanently empty when it was padded.
/// `None` on a malformed frame (skip it rather than kill the stream).
pub(crate) fn repack_depth(
    data: &[u8],
    dw: u32,
    dh: u32,
    stride: u32,
    scratch: &mut Vec<u16>,
) -> Option<Vec<u16>> {
    let (dw, dh) = (dw as usize, dh as usize);
    let row = dw * 2;
    let stride = if stride == 0 { row } else { stride as usize };
    if stride < row || data.len() < stride * dh {
        return None;
    }
    scratch.clear();
    scratch.reserve(dw * dh);
    for y in 0..dh {
        let src = &data[y * stride..y * stride + row];
        scratch.extend(
            src.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]])),
        );
    }
    Some(scratch.clone())
}

#[cfg(test)]
mod tests {
    use super::repack_depth;

    #[test]
    fn repack_honours_padded_stride() {
        // 3x2 depth, rows padded to 8 bytes (stride 8, row 6).
        let mut data = Vec::new();
        for y in 0..2u16 {
            for x in 0..3u16 {
                data.extend_from_slice(&(y * 10 + x).to_le_bytes());
            }
            data.extend_from_slice(&[0xEE, 0xEE]); // padding
        }
        let mut scratch = Vec::new();
        let out = repack_depth(&data, 3, 2, 8, &mut scratch).unwrap();
        assert_eq!(out, vec![0, 1, 2, 10, 11, 12]);
        // Tight (stride 0 = "as reported") also works.
        let tight: Vec<u8> = [5u16, 6, 7, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(
            repack_depth(&tight, 2, 2, 0, &mut scratch).unwrap(),
            vec![5, 6, 7, 8]
        );
        // Short buffer → skip.
        assert!(repack_depth(&tight[..6], 2, 2, 0, &mut scratch).is_none());
        // Stride smaller than a row → skip.
        assert!(repack_depth(&tight, 2, 2, 2, &mut scratch).is_none());
    }
}
