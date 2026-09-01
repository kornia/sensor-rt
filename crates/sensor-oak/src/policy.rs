//! The OAK-specific **policy** this driver layers on the faithful `depthai`
//! wrapper: the `OAK_*` environment knobs, the steady→epoch clock shift, and the
//! IMU rotation gate. Pure functions where possible, so every decision is
//! unit-tested.

use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use depthai::{ImgFrame, Message, UsbSpeed};

// ---------------------------------------------------------------------------
// Frame rate
// ---------------------------------------------------------------------------

/// A zero frame rate means "default", never "1 fps": it would poison the encoder
/// preset, the requested output rate and the Sync threshold.
pub(crate) const DEFAULT_FPS: u32 = 30;

pub(crate) fn fps_or_default(fps: u32) -> u32 {
    if fps == 0 {
        DEFAULT_FPS
    } else {
        fps
    }
}

// ---------------------------------------------------------------------------
// Environment knobs (`OAK_*`), read once per open
// ---------------------------------------------------------------------------

/// The `OAK_*` environment knobs, parsed once at open and passed down, so the
/// graph builders are pure functions of their arguments.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Knobs {
    /// `OAK_USB_SPEED`: cap the USB link. Default HIGH (USB2): the SUPER default
    /// boots the device into a USB3 descriptor, and on a physical USB2 link the
    /// host then can't reconnect to the booted device (X_LINK_DEVICE_NOT_FOUND).
    /// `super` opts into USB3 on a USB3 cable. Ignored for PoE.
    pub usb_speed: UsbSpeed,
    /// `OAK_H264_KBPS`: encoder target bitrate. 2000 kbps is comfortable for
    /// 640x360@30 and a big cut for a phone/Tailscale hop.
    pub h264_kbps: i32,
    /// `OAK_DEPTH_FPS`: depth rate on the RGBD path (default = colour fps).
    pub depth_fps: Option<u32>,
    /// `OAK_RGB_FPS`: raw-RGB pull rate on the RGBD path (default 10) — raw colour
    /// is only needed for local compute and is the heaviest XLink stream.
    pub rgb_fps: Option<u32>,
    /// `OAK_DEPTH_DIV`: on-device downscale of the aligned depth (default /2 →
    /// 1/4 the bytes).
    pub depth_div: u32,
    /// `OAK_SUBPIXEL`: subpixel disparity (default on). `0`/`false` trades
    /// precision for rate.
    pub subpixel: bool,
    /// `OAK_IR`: IR dot-projector intensity, clamped to `0..=1` (default 0.8;
    /// 0 disables).
    pub ir: f32,
}

impl Knobs {
    pub(crate) fn from_env() -> Self {
        Self::parse(|k| std::env::var(k).ok())
    }

    /// Parse from any key → value source (tests pass a map).
    pub(crate) fn parse(get: impl Fn(&str) -> Option<String>) -> Self {
        let positive = |k: &str| {
            get(k)
                .and_then(|s| s.trim().parse::<i64>().ok())
                .filter(|&v| v >= 1)
        };
        Knobs {
            usb_speed: match get("OAK_USB_SPEED").as_deref() {
                Some("super") | Some("SUPER") => UsbSpeed::Super,
                _ => UsbSpeed::High,
            },
            h264_kbps: positive("OAK_H264_KBPS").map_or(2000, |v| v.min(i32::MAX as i64) as i32),
            depth_fps: positive("OAK_DEPTH_FPS").map(|v| v.min(u32::MAX as i64) as u32),
            rgb_fps: positive("OAK_RGB_FPS").map(|v| v.min(u32::MAX as i64) as u32),
            depth_div: positive("OAK_DEPTH_DIV").map_or(2, |v| v.min(u32::MAX as i64) as u32),
            subpixel: !matches!(get("OAK_SUBPIXEL").as_deref(), Some("0") | Some("false")),
            ir: get("OAK_IR")
                .and_then(|s| s.trim().parse::<f32>().ok())
                .map_or(0.8, |x| x.clamp(0.0, 1.0)),
        }
    }

    /// Depth rate on the RGBD path: the knob, else `fps`; never above `fps`.
    pub(crate) fn depth_fps(&self, fps: u32) -> u32 {
        self.depth_fps.unwrap_or(fps).min(fps).max(1)
    }

    /// Raw-RGB rate on the RGBD path: the knob, else 10; never above `fps`.
    pub(crate) fn rgb_fps(&self, fps: u32) -> u32 {
        self.rgb_fps.unwrap_or(10).min(fps).max(1)
    }
}

/// XLink requires EVEN depth dims — an odd width/height tears the device
/// connection down (X_LINK_ERROR, e.g. OAK_DEPTH_DIV=3 → 213x120). Round each down
/// to even, floored at 2.
pub(crate) fn depth_output_size(width: u32, height: u32, div: u32) -> (u32, u32) {
    let div = div.max(1);
    (((width / div) & !1).max(2), ((height / div) & !1).max(2))
}

// ---------------------------------------------------------------------------
// Clock: depthai's steady_clock -> epoch nanoseconds
// ---------------------------------------------------------------------------
//
// depthai's getTimestamp() is on the host STEADY clock, synchronized across ALL
// connected devices, so multiple cameras share one timeline. We shift it onto the
// system clock so the value forwarded downstream (publishers, recordings) is a real
// epoch time every camera agrees on. Frames and IMU reports both go through this
// so they land on ONE timeline.

/// `system_now - steady_now`, in ns (may be negative on a freshly booted host).
pub(crate) fn steady_epoch_offset_now() -> i128 {
    let steady = depthai::steady_now()
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0);
    let system = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as i128)
        .unwrap_or(0);
    system - steady
}

/// ONE offset for the whole process for frames: recomputing per frame lets the two
/// clocks' relative jitter separate a frame stamp from an IMU stamp taken at one
/// instant.
pub(crate) fn steady_epoch_offset_cached() -> i128 {
    static OFF: OnceLock<i128> = OnceLock::new();
    *OFF.get_or_init(steady_epoch_offset_now)
}

pub(crate) fn steady_to_epoch_ns(steady_ns: i64, offset: i128) -> u64 {
    (steady_ns as i128 + offset).clamp(0, u64::MAX as i128) as u64
}

/// A frame's capture time on the epoch timeline (frames use the process-wide
/// cached offset; the IMU drain uses a per-batch one).
pub(crate) fn frame_epoch_ns(f: &ImgFrame) -> u64 {
    steady_to_epoch_ns(f.timestamp_ns(), steady_epoch_offset_cached())
}

// ---------------------------------------------------------------------------
// IMU chip -> camera rotation gate
// ---------------------------------------------------------------------------

/// Validate a 3x3 row-major matrix as a proper rotation usable to rotate IMU
/// samples into the camera frame. Returns WHY it was rejected.
///
/// The gate is a real rotation test, not just a det check: det≈1 alone admits
/// shears (a k=100 shear has det=1 and would turn 9.81 m/s² into ~981) and ±3%
/// scales, and every IEEE compare with NaN is false, so a NaN-laced matrix sailed
/// through `fabs(det-1) > 0.1`. Requirements, in order: all 9 entries finite;
/// det > 0 (proper, not a reflection); R·Rᵀ = I to 1e-3 (orthonormal, which also
/// pins |det| to 1); and not the EXACT identity — depthai stores identity as the
/// "no calibration" sentinel, and a real chip→camera mounting is never a perfect
/// identity.
pub(crate) fn validate_rotation(r: &[f32; 9]) -> Result<(), &'static str> {
    if r.iter().any(|v| !v.is_finite()) {
        return Err("non-finite entry");
    }
    let det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
        + r[2] * (r[3] * r[7] - r[4] * r[6]);
    if det <= 0.0 {
        return Err("determinant <= 0: a reflection, not a rotation");
    }
    let mut ortho_err = 0f32;
    for i in 0..3 {
        for j in 0..3 {
            let dot =
                r[i * 3] * r[j * 3] + r[i * 3 + 1] * r[j * 3 + 1] + r[i * 3 + 2] * r[j * 3 + 2];
            let target = if i == j { 1.0 } else { 0.0 };
            ortho_err = ortho_err.max((dot - target).abs());
        }
    }
    if ortho_err > 1e-3 {
        return Err("not orthonormal: |R*Rt - I| exceeds 1e-3, so it scales or shears");
    }
    let exact_identity = r
        .iter()
        .enumerate()
        .all(|(i, &v)| v == if i % 4 == 0 { 1.0 } else { 0.0 });
    if exact_identity {
        return Err("exact identity: depthai's not-calibrated sentinel, not a measured pose");
    }
    Ok(())
}

/// Row-major 3x3 times a vector.
pub(crate) fn rotate(r: &[f32; 9], v: [f32; 3]) -> [f32; 3] {
    [
        r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
        r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
        r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    fn knobs(pairs: &[(&str, &str)]) -> Knobs {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        Knobs::parse(|k| map.get(k).map(|v| v.to_string()))
    }

    #[test]
    fn rotation_gate_rejects_each_reason() {
        let mut m = IDENTITY;
        assert_eq!(
            validate_rotation(&m),
            Err("exact identity: depthai's not-calibrated sentinel, not a measured pose")
        );
        m[0] = f32::NAN;
        assert_eq!(validate_rotation(&m), Err("non-finite entry"));
        let refl = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(validate_rotation(&refl)
            .unwrap_err()
            .starts_with("determinant <= 0"));
        let shear = [1.0, 100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(validate_rotation(&shear)
            .unwrap_err()
            .starts_with("not orthonormal"));
        let scale = [1.03, 0.0, 0.0, 0.0, 1.03, 0.0, 0.0, 0.0, 1.03];
        assert!(validate_rotation(&scale)
            .unwrap_err()
            .starts_with("not orthonormal"));
    }

    #[test]
    fn rotation_gate_accepts_a_real_mounting() {
        // 90° about z: x -> y, y -> -x — a typical board axis permutation.
        let r = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(validate_rotation(&r), Ok(()));
        assert_eq!(rotate(&r, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        // A slightly noisy rotation (EEPROM float precision) still passes.
        let r2 = [0.0, -1.0, 0.0002, 1.0, 0.0, 0.0, 0.0, 0.0001, 1.0];
        assert_eq!(validate_rotation(&r2), Ok(()));
    }

    #[test]
    fn depth_output_size_is_even_and_floored() {
        assert_eq!(depth_output_size(640, 360, 2), (320, 180));
        assert_eq!(depth_output_size(640, 360, 3), (212, 120)); // 213 -> 212
        assert_eq!(depth_output_size(2, 2, 100), (2, 2));
        assert_eq!(depth_output_size(640, 360, 0), (640, 360));
    }

    #[test]
    fn knobs_have_documented_defaults() {
        let k = knobs(&[]);
        assert_eq!(k.usb_speed, UsbSpeed::High);
        assert_eq!(k.h264_kbps, 2000);
        assert_eq!(k.depth_fps, None);
        assert_eq!(k.rgb_fps, None);
        assert_eq!(k.depth_div, 2);
        assert!(k.subpixel);
        assert_eq!(k.ir, 0.8);
        assert_eq!(k.depth_fps(30), 30);
        assert_eq!(k.rgb_fps(30), 10);
        assert_eq!(k.rgb_fps(5), 5);
    }

    #[test]
    fn knobs_parse_and_clamp() {
        let k = knobs(&[
            ("OAK_USB_SPEED", "super"),
            ("OAK_H264_KBPS", " 4000 "),
            ("OAK_DEPTH_FPS", "60"),
            ("OAK_RGB_FPS", "0"),
            ("OAK_DEPTH_DIV", "3"),
            ("OAK_SUBPIXEL", "false"),
            ("OAK_IR", "5"),
        ]);
        assert_eq!(k.usb_speed, UsbSpeed::Super);
        assert_eq!(k.h264_kbps, 4000);
        assert_eq!(k.depth_fps(30), 30); // never above fps
        assert_eq!(k.rgb_fps, None); // "0" is not a rate
        assert_eq!(k.depth_div, 3);
        assert!(!k.subpixel);
        assert_eq!(k.ir, 1.0);
        assert_eq!(knobs(&[("OAK_IR", "-1")]).ir, 0.0);
        assert_eq!(knobs(&[("OAK_IR", "abc")]).ir, 0.8);
        assert_eq!(
            knobs(&[("OAK_USB_SPEED", "nonsense")]).usb_speed,
            UsbSpeed::High
        );
    }

    #[test]
    fn zero_fps_means_thirty() {
        assert_eq!(fps_or_default(0), 30);
        assert_eq!(fps_or_default(15), 15);
    }

    #[test]
    fn epoch_conversion_clamps() {
        assert_eq!(steady_to_epoch_ns(100, 50), 150);
        assert_eq!(steady_to_epoch_ns(100, -500), 0);
    }
}
