//! Side-by-side stereo match visualization: left | right, keypoints dotted, and a
//! line per mutual-NN match crossing the seam.
//!
//! This is the example's real evidence. Counts alone can't tell a working matcher
//! from a broken one — but on an unrectified OAK stereo pair, correct matches all
//! slope the same way (the disparity shifts every feature left-to-right by a
//! similar amount at similar depth), so a picture makes a bad matcher obvious at a
//! glance where "1200 matches" would not.

use kornia_image::{Image, ImageSize};
use kornia_io::png::write_image_png_rgba8;
use sensor_oak::StereoFrame;
use vrt_xfeat::XFeatResult;

/// Cap on match lines actually rendered — see the comment at the draw loop.
const MAX_MATCH_LINES: usize = 60;

pub struct StereoMatchViz {
    save_dir: String,
    /// Save one frame every `interval`; `0` disables saving.
    interval: u64,
}

impl StereoMatchViz {
    pub fn new(save_dir: String, interval: u64) -> Self {
        Self { save_dir, interval }
    }

    /// On every `interval`-th frame, render the pair with its matches and save
    /// `<save_dir>/xfeat_stereo_<seq>.png`. Viz failures are logged, never fatal —
    /// a PNG we couldn't write must not take down a camera loop.
    pub fn draw_and_save(
        &self,
        seq: u64,
        frame: &StereoFrame<'_>,
        res_l: &XFeatResult,
        res_r: &XFeatResult,
        matches: &[(usize, usize)],
    ) {
        if self.interval == 0 || !seq.is_multiple_of(self.interval) {
            return;
        }
        // Each result downloads on the stream it was allocated against.
        let (kl, kr) = match (res_l.kpts_to_host(), res_r.kpts_to_host()) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                eprintln!("[viz] keypoint D2H failed: {e}");
                return;
            }
        };

        let (fw, fh) = (frame.width(), frame.height());
        let (cw, ch) = (fw * 2, fh);
        let mut canvas = vec![0u8; cw as usize * ch as usize * 4];
        blit_rgb(&mut canvas, cw, frame.left(), fw, fh, 0);
        blit_rgb(&mut canvas, cw, frame.right(), fw, fh, fw);

        // Upstream XFeat already rescales keypoints from its floor-of-32 backbone
        // input back to original frame pixels, so these are frame coordinates
        // directly — the only adjustment is the right eye's horizontal offset in
        // the side-by-side canvas.
        let at = |k: &[f32], i: usize, x_off: u32| -> Option<(i32, i32)> {
            let (x, y) = (*k.get(i * 2)?, *k.get(i * 2 + 1)?);
            Some((x as i32 + x_off as i32, y as i32))
        };

        // All keypoints first, dim — so unmatched ones stay visible as context.
        for i in 0..kl.len() / 2 {
            if let Some((x, y)) = at(&kl, i, 0) {
                draw_dot(&mut canvas, cw, ch, x, y, 1, [90, 90, 200, 255]);
            }
        }
        for i in 0..kr.len() / 2 {
            if let Some((x, y)) = at(&kr, i, fw) {
                draw_dot(&mut canvas, cw, ch, x, y, 1, [90, 90, 200, 255]);
            }
        }

        // Then the matches. Every match line spans the full seam (~frame width), so
        // drawing all ~1000 of them paints the canvas solid green and hides exactly
        // what we came to check. Draw an evenly-spaced subset instead: the point is
        // to eyeball whether the lines are near-parallel and consistently sloped
        // (correct disparity) or fanned out at random (a broken matcher), and a few
        // dozen show that far better than a thousand.
        let stride = matches.len().div_ceil(MAX_MATCH_LINES).max(1);
        let mut drawn = 0;
        for &(i, j) in matches.iter().step_by(stride) {
            let (Some(a), Some(b)) = (at(&kl, i, 0), at(&kr, j, fw)) else {
                continue;
            };
            draw_line(&mut canvas, cw, ch, a, b, [40, 220, 90, 255]);
            draw_dot(&mut canvas, cw, ch, a.0, a.1, 2, [255, 60, 60, 255]);
            draw_dot(&mut canvas, cw, ch, b.0, b.1, 2, [255, 60, 60, 255]);
            drawn += 1;
        }

        let path = format!("{}/xfeat_stereo_{:06}.png", self.save_dir, seq);
        match Image::<u8, 4>::new(
            ImageSize {
                width: cw as usize,
                height: ch as usize,
            },
            canvas,
        ) {
            Ok(img) => match write_image_png_rgba8(&path, &img) {
                Ok(()) => println!(
                    "[viz] saved {path}  (L{} R{} kpts, {} matches, {drawn} lines drawn)",
                    res_l.len(),
                    res_r.len(),
                    matches.len()
                ),
                Err(e) => eprintln!("[viz] save failed: {e}"),
            },
            Err(e) => eprintln!("[viz] bad canvas: {e}"),
        }
    }
}

/// Copy a tightly packed RGB888 frame into an RGBA canvas at horizontal offset `x_off`.
fn blit_rgb(canvas: &mut [u8], cw: u32, src: &[u8], w: u32, h: u32, x_off: u32) {
    for y in 0..h as usize {
        for x in 0..w as usize {
            let s = (y * w as usize + x) * 3;
            let d = (y * cw as usize + x + x_off as usize) * 4;
            if s + 3 <= src.len() && d + 4 <= canvas.len() {
                canvas[d] = src[s];
                canvas[d + 1] = src[s + 1];
                canvas[d + 2] = src[s + 2];
                canvas[d + 3] = 255;
            }
        }
    }
}

/// Draw a filled disc of radius `r` into an interleaved RGBA buffer (`w`×`h`).
fn draw_dot(buf: &mut [u8], w: u32, h: u32, cx: i32, cy: i32, r: i32, color: [u8; 4]) {
    let (iw, ih) = (w as i32, h as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let (x, y) = (cx + dx, cy + dy);
                if x >= 0 && x < iw && y >= 0 && y < ih {
                    let p = (y as usize * w as usize + x as usize) * 4;
                    buf[p..p + 4].copy_from_slice(&color);
                }
            }
        }
    }
}

/// Bresenham-free DDA line — plenty for a debug overlay.
fn draw_line(buf: &mut [u8], w: u32, h: u32, a: (i32, i32), b: (i32, i32), color: [u8; 4]) {
    let steps = (b.0 - a.0).abs().max((b.1 - a.1).abs()).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = a.0 + ((b.0 - a.0) as f32 * t) as i32;
        let y = a.1 + ((b.1 - a.1) as f32 * t) as i32;
        if x >= 0 && x < w as i32 && y >= 0 && y < h as i32 {
            let p = (y as usize * w as usize + x as usize) * 4;
            buf[p..p + 4].copy_from_slice(&color);
        }
    }
}
