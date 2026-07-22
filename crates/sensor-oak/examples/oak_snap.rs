//! Diagnostic: grab one synced OAK frame (CPU side, straight from the shim) and
//! write THREE PNGs at the same resolution so we can verify RGB↔depth alignment:
//!   <out>_rgb.png     — the RGB image
//!   <out>_depth.png   — depth as grayscale (near=bright, far=dark, holes=black)
//!   <out>_overlay.png — 50/50 blend; if aligned, depth silhouettes sit exactly on RGB edges
//! Also prints RGB channel stats + depth coverage.
//!
//!   cargo run -p sensor-oak --example oak_snap                 # writes /tmp/oak_*.png
//!   cargo run -p sensor-oak --example oak_snap -- --out /tmp/foo

use argh::FromArgs;
use kornia_image::{Image, ImageSize};
use kornia_io::png::{write_image_png_gray8, write_image_png_rgb8};
use sensor_oak::{BoxError, OakSource};

#[derive(FromArgs)]
/// Grab one synced OAK frame and write RGB, depth, and overlay PNGs so RGB-depth
/// alignment can be checked by eye.
struct Args {
    /// output path prefix (default /tmp/oak)
    #[argh(option, default = "String::from(\"/tmp/oak\")")]
    out: String,
}

fn main() -> Result<(), BoxError> {
    let (w, h, fps) = (1280i32, 720i32, 30i32);
    let out = argh::from_env::<Args>().out;

    // The driver is host-only, so this diagnostic needs no CUDA context at all.
    let mut src = OakSource::open(None, w as u32, h as u32, fps as u32)?;

    let mut saved = false;
    for attempt in 1..=20 {
        let Some(frame) = src.next_frame() else {
            continue;
        };
        let (fw, fh) = (frame.width() as usize, frame.height() as usize);
        let npx = fw * fh;
        // RGB888 straight through — the PNG writer takes 3 channels, so there is no
        // reason to widen to RGBA.
        let rgb = frame.rgb_host();
        // Depth is aligned to the RGB grid but may be a SMALLER one (downscaled
        // on-device before transport), so resample it to RGB resolution by nearest
        // neighbour before overlaying — indexing it as if it were RGB-sized would
        // read the wrong pixels and make an aligned camera look misaligned.
        let (dims_match, dep): (bool, Vec<u16>) = match frame.depth() {
            None => (false, vec![0; npx]),
            Some(d) => {
                let (dw, dh) = (d.width() as usize, d.height() as usize);
                let src_mm = d.as_slice();
                let mut full = vec![0u16; npx];
                for y in 0..fh {
                    let sy = y * dh / fh;
                    for x in 0..fw {
                        full[y * fw + x] = src_mm[sy * dw + x * dw / fw];
                    }
                }
                (dw == fw && dh == fh, full)
            }
        };

        // RGB channel stats.
        let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
        for px in rgb.chunks_exact(3) {
            sr += px[0] as u64;
            sg += px[1] as u64;
            sb += px[2] as u64;
        }
        let n = npx as u64;
        let valid = dep.iter().filter(|&&v| v != 0).count();
        println!(
            "attempt {attempt:2}: {fw}×{fh}  meanRGB=({},{},{})  depth_valid={:.1}%  depth_dims_match_rgb={dims_match}",
            sr / n, sg / n, sb / n, 100.0 * valid as f32 / npx as f32,
        );

        if attempt >= 10 {
            let size = ImageSize {
                width: fw,
                height: fh,
            };

            // Colorize depth: near (300mm) bright → far (8000mm) dark; holes black.
            // Single channel — it is a grayscale ramp, so store and write it as one.
            let (near, far) = (300.0f32, 8000.0f32);
            let mut dgray = vec![0u8; npx];
            for (i, g) in dgray.iter_mut().enumerate() {
                let mm = dep[i] as f32;
                *g = if mm <= 0.0 {
                    0
                } else {
                    let t = ((mm - near) / (far - near)).clamp(0.0, 1.0);
                    (255.0 * (1.0 - t)) as u8
                };
            }
            // Overlay: RGB in green/blue, depth in red — misalignment shows as red fringes
            // offset from object edges.
            let mut ov = vec![0u8; npx * 3];
            for i in 0..npx {
                ov[i * 3] = dgray[i]; // depth → red
                ov[i * 3 + 1] = ((rgb[i * 3 + 1] as u16 + dgray[i] as u16) / 2) as u8; // rgb green blended
                ov[i * 3 + 2] = rgb[i * 3 + 2]; // rgb blue
            }

            let rgb_img = Image::<u8, 3>::new(size, rgb.to_vec())?;
            let dep_img = Image::<u8, 1>::new(size, dgray)?;
            let ov_img = Image::<u8, 3>::new(size, ov)?;
            write_image_png_rgb8(format!("{out}_rgb.png"), &rgb_img)?;
            write_image_png_gray8(format!("{out}_depth.png"), &dep_img)?;
            write_image_png_rgb8(format!("{out}_overlay.png"), &ov_img)?;
            println!("wrote {out}_rgb.png, {out}_depth.png, {out}_overlay.png");
            saved = true;
            break;
        }
    }

    // `src` closes the device on drop — no manual oak_close needed.
    if !saved {
        return Err("no frame captured".into());
    }
    Ok(())
}
