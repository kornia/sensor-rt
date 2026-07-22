//! Diagnostic: grab one synced OAK frame (CPU side, straight from the shim) and
//! write THREE PNGs at the same resolution so we can verify RGB↔depth alignment:
//!   <out>_rgb.png     — the RGB image
//!   <out>_depth.png   — colorized depth (grayscale: near=bright, far=dark, holes=black)
//!   <out>_overlay.png — 50/50 blend; if aligned, depth silhouettes sit exactly on RGB edges
//! Also prints RGB channel stats + depth coverage.
//!
//!   cargo run -p sensor-oak --example oak_snap            # writes /tmp/oak_*.png
//!   OAK_OUT=/tmp/foo cargo run -p sensor-oak --example oak_snap

use kornia_image::{Image, ImageSize};
use kornia_io::png::write_image_png_rgba8;
use sensor_oak::{BoxError, OakSource};

fn main() -> Result<(), BoxError> {
    let (w, h, fps) = (1280i32, 720i32, 30i32);
    let out = std::env::var("OAK_OUT").unwrap_or_else(|_| "/tmp/oak".into());

    // The driver is host-only, so this diagnostic needs no CUDA context at all.
    let mut src = OakSource::open(None, w as u32, h as u32, fps as u32)?;

    let mut saved = false;
    for attempt in 1..=20 {
        let Some(frame) = src.next_frame() else {
            continue;
        };
        let (fw, fh) = (frame.width() as usize, frame.height() as usize);
        let npx = fw * fh;
        // The shim hands out RGB888 (3 B/px); expand to RGBA for the PNG/stat code below.
        let rgb3 = frame.rgb_host();
        let mut rgb = vec![0u8; npx * 4];
        for i in 0..npx {
            rgb[i * 4] = rgb3[i * 3];
            rgb[i * 4 + 1] = rgb3[i * 3 + 1];
            rgb[i * 4 + 2] = rgb3[i * 3 + 2];
            rgb[i * 4 + 3] = 255;
        }
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
        for px in rgb.chunks_exact(4) {
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
            let (near, far) = (300.0f32, 8000.0f32);
            let mut dgray = vec![0u8; npx * 4];
            for i in 0..npx {
                let mm = dep[i] as f32;
                let g = if mm <= 0.0 {
                    0u8
                } else {
                    let t = ((mm - near) / (far - near)).clamp(0.0, 1.0);
                    (255.0 * (1.0 - t)) as u8
                };
                dgray[i * 4] = g;
                dgray[i * 4 + 1] = g;
                dgray[i * 4 + 2] = g;
                dgray[i * 4 + 3] = 255;
            }
            // Overlay: RGB in green/blue, depth in red — misalignment shows as red fringes
            // offset from object edges.
            let mut ov = vec![0u8; npx * 4];
            for i in 0..npx {
                ov[i * 4] = dgray[i * 4]; // depth → red
                ov[i * 4 + 1] = ((rgb[i * 4 + 1] as u16 + dgray[i * 4] as u16) / 2) as u8; // rgb green blended
                ov[i * 4 + 2] = rgb[i * 4 + 2]; // rgb blue
                ov[i * 4 + 3] = 255;
            }

            let rgb_img = Image::<u8, 4>::new(size, rgb)?;
            let dep_img = Image::<u8, 4>::new(size, dgray)?;
            let ov_img = Image::<u8, 4>::new(size, ov)?;
            write_image_png_rgba8(format!("{out}_rgb.png"), &rgb_img)?;
            write_image_png_rgba8(format!("{out}_depth.png"), &dep_img)?;
            write_image_png_rgba8(format!("{out}_overlay.png"), &ov_img)?;
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
