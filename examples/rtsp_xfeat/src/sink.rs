//! Keypoint visualization: draw XFeat keypoints on the camera snapshot and save
//! a PNG. A plain helper (no framework) the main loop calls periodically.

use std::sync::{Arc, Mutex};

use kornia_image::{Image, ImageSize};
use kornia_io::png::write_image_png_rgba8;
use sensor_rtsp::CpuFrame;
use vrt::CudaStream;
use vrt_xfeat::XFeatResult;

/// Draws keypoints onto the latest CPU snapshot and writes a PNG every
/// `interval` frames.
pub struct KeypointViz {
    /// Latest CPU RGBA snapshot from the `RtspSource` tee branch.
    cpu_snap: Arc<Mutex<Option<CpuFrame>>>,
    /// Shared stream — used to download the (device-resident) keypoints on demand.
    stream: Arc<CudaStream>,
    save_dir: String,
    /// Model input dims — keypoints are in model space and scale to frame space.
    dst_w: u32,
    dst_h: u32,
    /// Save one frame every `interval`; `0` disables saving.
    interval: u64,
}

impl KeypointViz {
    pub fn new(
        cpu_snap: Arc<Mutex<Option<CpuFrame>>>,
        stream: Arc<CudaStream>,
        save_dir: String,
        dst_w: u32,
        dst_h: u32,
        interval: u64,
    ) -> Self {
        Self {
            cpu_snap,
            stream,
            save_dir,
            dst_w,
            dst_h,
            interval,
        }
    }

    /// On every `interval`-th frame, draw `result`'s keypoints on the latest
    /// snapshot and save `<save_dir>/xfeat_<seq>.png`. Viz failures are logged,
    /// never fatal.
    pub fn draw_and_save(&self, seq: u64, result: &XFeatResult) {
        if self.interval == 0 || !seq.is_multiple_of(self.interval) {
            return;
        }
        let Some((rgba, fw, fh)) = self.cpu_snap.lock().ok().and_then(|mut g| g.take()) else {
            eprintln!("[viz] no CPU frame yet at seq {seq}");
            return;
        };
        // Keypoints live on the GPU — download the valid ones only when drawing.
        let kpts = match result.kpts_to_host(&self.stream) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("[viz] kpts D2H failed: {e}");
                return;
            }
        };

        let mut buf = rgba;
        // keypoint coords are in model space (dst_w × dst_h); scale to frame space
        let sx = fw as f32 / self.dst_w as f32;
        let sy = fh as f32 / self.dst_h as f32;
        for chunk in kpts.chunks_exact(2) {
            let cx = (chunk[0] * sx) as i32;
            let cy = (chunk[1] * sy) as i32;
            draw_dot(&mut buf, fw, fh, cx, cy, 4, [255, 50, 50, 255]); // red filled circle
            draw_dot(&mut buf, fw, fh, cx, cy, 2, [255, 255, 50, 255]); // yellow centre dot
        }

        let path = format!("{}/xfeat_{:06}.png", self.save_dir, seq);
        match Image::<u8, 4>::new(
            ImageSize {
                width: fw as usize,
                height: fh as usize,
            },
            buf,
        ) {
            Ok(img) => match write_image_png_rgba8(&path, &img) {
                Ok(()) => println!("[viz] saved {path}  ({} kpts)", result.count),
                Err(e) => eprintln!("[viz] save failed: {e}"),
            },
            Err(e) => eprintln!("[viz] bad frame buffer: {e}"),
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
