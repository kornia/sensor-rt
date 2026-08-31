//! OAK-D **RGBD + H.264** modality: CAM_A colour (RGB888) + `StereoDepth` aligned to
//! it (uint16 mm) + an on-device H.264 colour stream.
//!
//! This is the colour/depth counterpart to the raw [`stereo`](crate::stereo) path —
//! the camera-producer source for the site (`flux-oak`). The three outputs are
//! **decoupled**: colour, depth, and encoded video each come out of their own queue
//! ([`next_rgb`](OakSource::next_rgb) / [`next_depth`](OakSource::next_depth) /
//! [`next_video`](OakSource::next_video)), pulled independently and paired downstream
//! by their shared host-synced timestamps. That frees depth from the raw-RGB pull
//! rate and lets the small H.264 stream ship at full fps while the heavy raw RGBD is
//! decimated.
//!
//! **Host-only, no CUDA, no `vrt`.** Frames come out as owned host buffers (the shim
//! pins the device buffer only until the next poll, so each `next_*` copies its bytes
//! out); the consumer owns any GPU upload. A `flux-oak`-style producer that only
//! encodes + republishes builds no GPU stack at all.

use crate::{device_id_cstring, ffi, last_error, BoxError, OakSource};

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
    /// The shim preflights with `getConnectedIMU()` and only builds the IMU node when the board
    /// actually carries one, so an IMU-less board degrades ([`has_imu`](Self::has_imu) is `false`)
    /// without ever risking the image streams — never an error. Rates above 400 Hz (the BNO086
    /// gyro maximum) are clamped. When the EEPROM carries valid IMU extrinsics, samples come out
    /// in the CAM_A optical frame — check [`imu_aligned`](Self::imu_aligned); an absent or
    /// rejected calibration is logged to stderr by the shim with the reason.
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
        let id_c = device_id_cstring(device)?;
        let id_ptr = id_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let imu_hz = crate::imu::clamp_imu_hz(imu_hz);
        // H.264 is always on in this modality — the whole point is the efficient colour stream.
        // No open-retry here: the shim preflights the IMU with getConnectedIMU() before building
        // the node, so an IMU-less board already degrades inside ONE open. A failure at this
        // point is a real device error, and retrying (especially with `device = None`) could
        // silently bind a different physical camera on a multi-OAK rig.
        let dev = unsafe {
            ffi::oak_open_rgbd(
                id_ptr,
                width as i32,
                height as i32,
                fps as i32,
                1,
                depth as i32,
                video_only as i32,
                imu_hz as i32,
            )
        };
        if dev.is_null() {
            return Err(last_error("oak_open_rgbd"));
        }
        Self::from_open_device(dev, width, height)
    }

    /// Whether `StereoDepth` is running (so [`next_depth`](Self::next_depth) can yield aligned depth).
    pub fn has_depth(&self) -> bool {
        self.has_depth
    }

    /// Whether the on-device H.264 colour stream is running (so [`next_video`](Self::next_video) yields).
    pub fn has_video(&self) -> bool {
        self.has_video
    }

    /// Whether this device runs the colour(+depth) pipeline (so [`next_rgb`](Self::next_rgb) yields).
    /// `false` means it auto-fell-back to video-only (mono / uncalibrated): drain only
    /// [`next_video`](Self::next_video). Always check this after [`open_rgbd`](Self::open_rgbd).
    pub fn has_sync(&self) -> bool {
        self.has_sync
    }

    /// Decoupled raw-colour poll: the next RGB888 frame from its own queue (non-blocking), copied out
    /// with its dims + capture timestamp (ns). `None` when none is queued. Independent of
    /// [`next_depth`](Self::next_depth) — pair them by timestamp on the consumer. Drain in a loop until
    /// `None` each iteration.
    pub fn next_rgb(&mut self) -> Option<(Vec<u8>, u32, u32, u64)> {
        let mut rgb: *const u8 = std::ptr::null();
        let (mut w, mut h, mut len, mut ts) = (0i32, 0i32, 0i32, 0u64);
        let rc =
            unsafe { ffi::oak_poll_rgb(self.dev, &mut rgb, &mut w, &mut h, &mut len, &mut ts) };
        if rc == 1 && !rgb.is_null() && len > 0 {
            // Copy the span out while the shim still pins it (valid only until the next poll).
            let bytes = unsafe { std::slice::from_raw_parts(rgb, len as usize) }.to_vec();
            Some((bytes, w as u32, h as u32, ts))
        } else {
            None
        }
    }

    /// Decoupled depth poll: the next aligned uint16-mm depth frame from its own queue (non-blocking) at
    /// the stereo rate, copied out with its dims (may be `<` colour size) + capture timestamp. `None`
    /// when none is queued. Drain in a loop until `None`.
    pub fn next_depth(&mut self) -> Option<(Vec<u16>, u32, u32, u64)> {
        let mut depth: *const u16 = std::ptr::null();
        let (mut dw, mut dh, mut ts) = (0i32, 0i32, 0u64);
        let rc = unsafe { ffi::oak_poll_depth(self.dev, &mut depth, &mut dw, &mut dh, &mut ts) };
        if rc == 1 && !depth.is_null() && dw > 0 && dh > 0 {
            let n = dw as usize * dh as usize;
            let vals = unsafe { std::slice::from_raw_parts(depth, n) }.to_vec();
            Some((vals, dw as u32, dh as u32, ts))
        } else {
            None
        }
    }

    /// Drain the next on-device **H.264** access unit, if one is ready (non-blocking): the encoded bytes
    /// (copied out, so the caller may hold/publish them freely) + the capture timestamp in ns. `None`
    /// when no frame is queued or H.264 isn't running. Call in a loop until `None` each iteration so the
    /// encoder queue never overflows — a dropped P-frame glitches the stream until the next keyframe.
    pub fn next_video(&mut self) -> Option<(Vec<u8>, u64)> {
        let mut data: *const u8 = std::ptr::null();
        let (mut len, mut ts) = (0i32, 0u64);
        let rc = unsafe { ffi::oak_poll_video(self.dev, &mut data, &mut len, &mut ts) };
        if rc == 1 && !data.is_null() && len > 0 {
            let bytes = unsafe { std::slice::from_raw_parts(data, len as usize) }.to_vec();
            Some((bytes, ts))
        } else {
            None
        }
    }
}
