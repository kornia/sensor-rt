//! The OAK-specific **policy** this driver layers on the faithful `depthai`
//! wrapper: environment knobs, the steady→epoch clock shift, and the IMU
//! rotation gate. Pure functions where possible, so every decision that used to
//! sit inside the C++ shim is unit-tested here.

use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use depthai::UsbSpeed;

// ---------------------------------------------------------------------------
// Environment knobs (same names, defaults and semantics as the old shim)
// ---------------------------------------------------------------------------

/// A zero/absent frame rate means "default", never "1 fps": it would poison the
/// encoder preset, the requested output rate and the Sync threshold.
pub(crate) const DEFAULT_FPS: u32 = 30;

pub(crate) fn fps_or_default(fps: u32) -> u32 {
    if fps == 0 {
        DEFAULT_FPS
    } else {
        fps
    }
}

/// `OAK_USB_SPEED`: cap the USB link. Default HIGH (USB2): the SUPER default boots
/// the device into a USB3 descriptor, and on a physical USB2 link the host then
/// can't reconnect to the booted device (X_LINK_DEVICE_NOT_FOUND). `super` opts
/// into USB3 on a USB3 cable. Ignored for PoE.
pub(crate) fn usb_speed_from_env() -> UsbSpeed {
    parse_usb_speed(std::env::var("OAK_USB_SPEED").ok().as_deref())
}

pub(crate) fn parse_usb_speed(v: Option<&str>) -> UsbSpeed {
    match v {
        Some("super") | Some("SUPER") => UsbSpeed::Super,
        _ => UsbSpeed::High,
    }
}

/// `OAK_H264_KBPS`: encoder target bitrate. 2000 kbps is comfortable for 640x360@30
/// and a big cut for a phone/Tailscale hop.
pub(crate) fn h264_kbps() -> i32 {
    parse_positive(std::env::var("OAK_H264_KBPS").ok().as_deref()).unwrap_or(2000)
}

/// `OAK_DEPTH_FPS`: depth rate on the RGBD path (default = colour fps, never above it).
pub(crate) fn depth_fps(fps: u32) -> u32 {
    clamp_rate(
        parse_positive(std::env::var("OAK_DEPTH_FPS").ok().as_deref()),
        fps,
        fps,
    )
}

/// `OAK_RGB_FPS`: raw-RGB pull rate on the RGBD path (default 10, never above fps) —
/// raw colour is only needed for local compute and is the heaviest XLink stream.
pub(crate) fn rgb_fps(fps: u32) -> u32 {
    clamp_rate(
        parse_positive(std::env::var("OAK_RGB_FPS").ok().as_deref()),
        10,
        fps,
    )
}

pub(crate) fn clamp_rate(requested: Option<i32>, default: u32, max: u32) -> u32 {
    let v = requested.map_or(default, |v| v as u32);
    v.min(max).max(1)
}

/// `OAK_DEPTH_DIV`: on-device downscale of the aligned depth (default /2 → 1/4 the bytes).
pub(crate) fn depth_div() -> u32 {
    parse_positive(std::env::var("OAK_DEPTH_DIV").ok().as_deref()).map_or(2, |v| v as u32)
}

/// XLink requires EVEN depth dims — an odd width/height tears the device
/// connection down (X_LINK_ERROR, e.g. OAK_DEPTH_DIV=3 → 213x120). Round each down
/// to even, floored at 2.
pub(crate) fn depth_output_size(width: u32, height: u32, div: u32) -> (u32, u32) {
    let div = div.max(1);
    (((width / div) & !1).max(2), ((height / div) & !1).max(2))
}

/// `OAK_SUBPIXEL`: subpixel disparity (default on). `0`/`false` trades precision for rate.
pub(crate) fn subpixel() -> bool {
    parse_bool_default_true(std::env::var("OAK_SUBPIXEL").ok().as_deref())
}

pub(crate) fn parse_bool_default_true(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false"))
}

/// `OAK_IR`: IR dot-projector intensity, clamped to `0..=1` (default 0.8; 0 disables).
pub(crate) fn ir_intensity() -> f32 {
    parse_ir(std::env::var("OAK_IR").ok().as_deref())
}

pub(crate) fn parse_ir(v: Option<&str>) -> f32 {
    v.and_then(|s| s.trim().parse::<f32>().ok())
        .map_or(0.8, |x| x.clamp(0.0, 1.0))
}

fn parse_positive(v: Option<&str>) -> Option<i32> {
    v.and_then(|s| s.trim().parse::<i32>().ok())
        .filter(|&x| x >= 1)
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

pub(crate) const IDENTITY: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_gate_rejects_each_reason() {
        let mut m = IDENTITY;
        assert_eq!(
            validate_rotation(&m),
            Err("exact identity: depthai's not-calibrated sentinel, not a measured pose")
        );
        m[0] = f32::NAN;
        assert_eq!(validate_rotation(&m), Err("non-finite entry"));
        // Reflection: flip one axis.
        let refl = [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(validate_rotation(&refl)
            .unwrap_err()
            .starts_with("determinant <= 0"));
        // Shear with det = 1.
        let shear = [1.0, 100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(validate_rotation(&shear)
            .unwrap_err()
            .starts_with("not orthonormal"));
        // 3% scale.
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
    fn env_parsers_have_the_shim_defaults() {
        assert_eq!(parse_usb_speed(None), UsbSpeed::High);
        assert_eq!(parse_usb_speed(Some("super")), UsbSpeed::Super);
        assert_eq!(parse_usb_speed(Some("nonsense")), UsbSpeed::High);
        assert!(parse_bool_default_true(None));
        assert!(!parse_bool_default_true(Some("0")));
        assert!(!parse_bool_default_true(Some("false")));
        assert_eq!(parse_ir(None), 0.8);
        assert_eq!(parse_ir(Some("5")), 1.0);
        assert_eq!(parse_ir(Some("-1")), 0.0);
        assert_eq!(parse_ir(Some("abc")), 0.8);
        assert_eq!(clamp_rate(None, 10, 30), 10);
        assert_eq!(clamp_rate(Some(60), 10, 30), 30);
        assert_eq!(clamp_rate(None, 30, 15), 15);
        assert_eq!(parse_positive(Some("0")), None);
        assert_eq!(parse_positive(Some(" 7 ")), Some(7));
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
