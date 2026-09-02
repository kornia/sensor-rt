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

use depthai::{CameraBoardSocket, ImgFrameType, ImgResizeMode};

use crate::calib::StereoCalibError;
use crate::graph::{self, Session};
use crate::policy::Knobs;
use crate::{policy, row_pitch, BoxError, Ctx, OakSource, Queues, Q};

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
    /// [`next_imu`](Self::next_imu) on the same host-epoch timeline as the frames; `0` disables it,
    /// a board without one degrades (see [`has_imu`](Self::has_imu)), and samples come out in the
    /// CAM_A optical frame when the EEPROM allows it (see [`imu_aligned`](Self::imu_aligned)).
    pub fn open_rgbd(
        device: Option<&str>,
        width: u32,
        height: u32,
        fps: u32,
        depth: bool,
        imu_hz: u32,
    ) -> Result<Self, BoxError> {
        Self::open_rgbd_inner(device, width, height, fps, depth, false, imu_hz)
            .map_err(|e| format!("open_rgbd failed: {e}").into())
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
            .map_err(|e| format!("open_rgbd_video failed: {e}").into())
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
        let imu_hz = policy::clamp_imu_hz(imu_hz);
        // No open-retry here: the IMU is preflighted before the node is built, so an
        // IMU-less board already degrades inside ONE open. A failure at this point is a
        // real device error, and retrying (especially with `device = None`) could
        // silently bind a different physical camera on a multi-OAK rig.
        let s = Session::connect(device, Knobs::from_env())?;

        // Auto-fall-back to video-only when depth was requested but the device can't
        // actually produce it (mono camera, or wiped/blank calibration → fx=0). Pulling
        // raw RGB over XLink for a "synced RGBD" pair whose depth is garbage just caps
        // the H.264 stream for nothing — build the lean video-only pipeline instead.
        // Policy: always ship compressed video; add RGBD only when depth works.
        let want_video_only = video_only || (depth && !s.can_do_depth(width, height));
        let queues = if want_video_only {
            build_video_only(&s, width, height, fps)?
        } else {
            build_decoupled(&s, width, height, fps, depth).ctx("build RGBD graph")?
        };
        let ir = s.knobs.ir;
        let src = s.finish(
            width,
            height,
            queues,
            imu_hz,
            CameraBoardSocket::CamA,
            Err(StereoCalibError::NotStereo),
        )?;
        if !want_video_only && ir > 0.0 {
            // IR dot projector: passive stereo starves on texture-poor / dim scenes
            // (single-digit valid-depth %). Default 0.8 intensity; OAK_IR=0 disables
            // (e.g. multi-cam cross-talk), boards without a projector just return false.
            // Needs a live (started) device; an RPC failure here means the device is
            // already unhealthy, so it fails the open.
            src.device
                .set_ir_laser_dot_projector_intensity(ir, None)
                .ctx("IR dot projector")?;
        }
        Ok(src)
    }

    /// Whether `StereoDepth` is running (so [`next_depth`](Self::next_depth) can yield aligned depth).
    pub fn has_depth(&self) -> bool {
        self.queues.depth.is_some()
    }

    /// Whether the on-device H.264 colour stream is running (so [`next_video`](Self::next_video) yields).
    pub fn has_video(&self) -> bool {
        self.queues.video.is_some()
    }

    /// Whether this device runs the colour(+depth) pipeline (so [`next_rgb`](Self::next_rgb) yields).
    /// `false` means it auto-fell-back to video-only (mono / uncalibrated): drain only
    /// [`next_video`](Self::next_video). Always check this after [`open_rgbd`](Self::open_rgbd).
    pub fn has_sync(&self) -> bool {
        self.queues.rgb.is_some()
    }

    /// Decoupled raw-colour poll: the next RGB888 frame from its own queue (non-blocking), copied out
    /// with its dims + capture timestamp (ns). `None` when none is queued. Independent of
    /// [`next_depth`](Self::next_depth) — pair them by timestamp on the consumer. Drain in a loop until
    /// `None` each iteration.
    pub fn next_rgb(&mut self) -> Option<(Vec<u8>, u32, u32, u64)> {
        let f = self.queues.rgb.as_ref()?.pop(None).ok()??;
        let (w, h) = (f.width(), f.height());
        // Tight or row-padded, either way one copy out (the owned-Vec API's copy).
        let bytes = repack_rows(f.data(), w as usize * 3, h as usize, f.stride() as usize)?;
        Some((bytes, w, h, policy::frame_epoch_ns(&f)))
    }

    /// Decoupled depth poll: the next aligned uint16-mm depth frame from its own queue (non-blocking) at
    /// the stereo rate, copied out with its dims (may be `<` colour size) + capture timestamp. `None`
    /// when none is queued. Drain in a loop until `None`.
    pub fn next_depth(&mut self) -> Option<(Vec<u16>, u32, u32, u64)> {
        let d = self.queues.depth.as_ref()?.pop(None).ok()??;
        let (dw, dh) = (d.width(), d.height());
        let vals = repack_depth(d.data(), dw, dh, d.stride())?;
        Some((vals, dw, dh, policy::frame_epoch_ns(&d)))
    }

    /// Drain the next on-device **H.264** access unit, if one is ready (non-blocking): the encoded bytes
    /// (copied out, so the caller may hold/publish them freely) + the capture timestamp in ns. `None`
    /// when no frame is queued or H.264 isn't running. Call in a loop until `None` each iteration so the
    /// encoder queue never overflows — a dropped P-frame glitches the stream until the next keyframe.
    pub fn next_video(&mut self) -> Option<(Vec<u8>, u64)> {
        let f = self.queues.video.as_ref()?.queue.pop(None).ok()??;
        if f.data().is_empty() {
            return None;
        }
        Some((f.data().to_vec(), policy::frame_epoch_ns(&f)))
    }
}

/// VIDEO-ONLY: just the H.264 encoder (CAM_A NV12 → encoder → queue). No
/// RGB888/depth output, so the device transmits ONLY the small H.264 bitstream.
/// The encoder is MANDATORY here — the stream is this modality's whole output —
/// and this path is also where an RGBD request LANDS when the board cannot do
/// depth, so a failure says that rather than surfacing as a bare device error.
fn build_video_only(s: &Session, width: u32, height: u32, fps: u32) -> Result<Queues, BoxError> {
    let color = s.camera(CameraBoardSocket::CamA).ctx("colour camera")?;
    let video = graph::add_h264_encoder(s, &color, width, height, fps).map_err(|e| {
        format!("H.264 encoder unavailable and this device cannot produce depth ({e}) — nothing left to stream")
    })?;
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
    s: &Session,
    width: u32,
    height: u32,
    fps: u32,
    depth: bool,
) -> depthai::Result<Queues> {
    // Colour (CAM_A). Interleaved RGB888 (the ISP's native type) as the raw-RGB
    // stream AND the depth-alignment reference; undistorted so depth aligns
    // pixel-perfect and the intrinsics are an exact pinhole.
    let color = s.camera(CameraBoardSocket::CamA)?;
    let rgb_out = color.request_output(
        (width, height),
        Some(ImgFrameType::Rgb888i),
        ImgResizeMode::Crop,
        Some(s.knobs.rgb_fps(fps) as f32),
        Some(true),
    )?;
    // OPTIONAL (e.g. OAK-1: no stereo pair): degrade to colour + video.
    let depth_q = depth
        .then(|| {
            graph::add_stereo_depth(s, &rgb_out, width, height, s.knobs.depth_fps(fps))
                .map_err(|e| degrade!("StereoDepth unavailable ({e}) — continuing without depth"))
                .ok()
        })
        .flatten();
    // Each stream to its OWN non-blocking queue, pulled + published independently
    // (consumer pairs by timestamp). H.264 is always on in this modality, but a
    // board that rejects the NV12 output must not cost the caller depth + RGB.
    Ok(Queues {
        rgb: Some(Q::new(rgb_out.create_output_queue(4, false)?, "rgb")),
        depth: depth_q,
        video: graph::try_add_h264_encoder(s, Some(&color), width, height, fps, "RGBD"),
        ..Default::default()
    })
}

/// Copy `h` rows of `row` bytes out of a frame whose rows are `stride` bytes apart
/// (`stride == 0` = tight, as depthai reports it), into one tightly packed Vec —
/// the one copy the owned-Vec API mandates. A downscaled/aligned frame is often
/// padded to an alignment boundary; honouring the stride instead of dropping every
/// such frame is what keeps the stream alive. `None` on a malformed (zero-sized or
/// too short) frame: skip it rather than kill the stream.
fn repack_rows(data: &[u8], row: usize, h: usize, stride: usize) -> Option<Vec<u8>> {
    let stride = row_pitch(row, h, stride, data.len())?;
    if stride == row {
        return Some(data[..row * h].to_vec());
    }
    let mut out = Vec::with_capacity(row * h);
    for y in 0..h {
        out.extend_from_slice(&data[y * stride..y * stride + row]);
    }
    Some(out)
}

/// [`repack_rows`] for a `RAW16` depth frame, straight into little-endian `u16`
/// millimetres (one allocation, one pass).
fn repack_depth(data: &[u8], dw: u32, dh: u32, stride: u32) -> Option<Vec<u16>> {
    let (row, h) = (dw as usize * 2, dh as usize);
    let stride = row_pitch(row, h, stride as usize, data.len())?;
    let mut out = Vec::with_capacity(dw as usize * h);
    for y in 0..h {
        let line = &data[y * stride..y * stride + row];
        out.extend(
            line.chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]])),
        );
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::{repack_depth, repack_rows};

    #[test]
    fn repack_rows_accepts_unpadded_last_row() {
        // 2 rows of 3 bytes, stride 4, last row unpadded: 4 + 3 = 7 bytes suffice.
        let data = [1u8, 2, 3, 0xEE, 4, 5, 6];
        assert_eq!(repack_rows(&data, 3, 2, 4).unwrap(), vec![1, 2, 3, 4, 5, 6]);
        assert!(repack_rows(&data[..6], 3, 2, 4).is_none());
        assert!(repack_rows(&data, 3, 0, 4).is_none());
        // Tight input is a straight copy.
        assert_eq!(
            repack_rows(&data[..6], 3, 2, 0).unwrap(),
            vec![1, 2, 3, 0xEE, 4, 5]
        );
    }

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
        assert_eq!(
            repack_depth(&data, 3, 2, 8).unwrap(),
            vec![0, 1, 2, 10, 11, 12]
        );
        // Tight (stride 0 = "as reported") also works.
        let tight: Vec<u8> = [5u16, 6, 7, 8]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(repack_depth(&tight, 2, 2, 0).unwrap(), vec![5, 6, 7, 8]);
        // Short buffer → skip.
        assert!(repack_depth(&tight[..6], 2, 2, 0).is_none());
        // Stride smaller than a row → skip.
        assert!(repack_depth(&tight, 2, 2, 2).is_none());
    }
}
