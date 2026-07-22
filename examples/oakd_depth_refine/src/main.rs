//! OAK-D → lingbot depth-**refinement** end-to-end benchmark:
//! ```text
//!   OakSource (RGB device + aligned depth mm) → resize depth to model size (m)
//!     → vrt_depth::DepthRefine (RGB + raw depth → refined depth on TRT) → time
//! ```
//! Unlike `oakd_track3d` (which trusts the OAK's on-VPU depth), this runs a
//! Jetson TRT depth model over the OAK's raw RGBD. It reports the per-frame
//! **inference** latency (`DepthRefine::run`, matches the synthetic `depth_bench`
//! since TRT latency is data-independent) and the **end-to-end** rate (capture +
//! depth resize/upload + refine), which is bounded by the camera/USB.
//!
//! Build the engine with lingbot-depth-trt's `export_trt.py` (fixed 480×640).
//!
//! Run (via pixi for the depthai-core runtime libs):
//!   cargo run --release -p oakd_depth_refine -- --engine lingbot.engine [--iters 100]

use std::time::Instant;

use kornia_tensor::Tensor;
use sensor_oak::OakSource;
use vrt::logger::Severity;
use vrt::{BoxError, CudaStream, Engine, Logger, Runtime, Stream};
use vrt_depth::DepthRefine;

use std::sync::Arc;

/// Nearest-neighbour resize of the OAK's `w×h` u16 millimetre depth map into a
/// device `[1,1,mh,mw]` f32 metre tensor (the shape `DepthRefine` expects).
/// Nearest (not bilinear) avoids blending metres across depth discontinuities;
/// the 0 = invalid sentinel is carried through as 0.0 (the model masks it).
fn depth_to_model_tensor(
    mm: &[u16],
    w: usize,
    h: usize,
    mh: usize,
    mw: usize,
    stream: &Arc<CudaStream>,
) -> Result<Tensor<f32, 4>, BoxError> {
    let mut out = vec![0.0f32; mh * mw];
    for r in 0..mh {
        let sy = (r * h / mh).min(h - 1);
        for c in 0..mw {
            let sx = (c * w / mw).min(w - 1);
            out[r * mw + c] = mm[sy * w + sx] as f32 * 1e-3;
        }
    }
    Ok(Tensor::<f32, 4>::from_shape_vec([1, 1, mh, mw], out)?.to_cuda(stream)?)
}

fn main() -> Result<(), vrt::BoxError> {
    env_logger::init();

    let mut engine_path: Option<String> = None;
    let mut iters = 100usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--engine" => engine_path = args.next(),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            _ => {}
        }
    }
    let Some(engine_path) = engine_path else {
        eprintln!("Usage: oakd_depth_refine --engine <path> [--iters N]");
        std::process::exit(1);
    };

    let (w, h, fps) = (1280u32, 720u32, 30u32);
    let stream = Stream::new_standalone()?.cuda_stream().clone();
    let runtime = Runtime::new(Logger::new(Severity::Warning)?)?;
    let engine = Engine::from_file(runtime, &engine_path)?;
    let mut refiner = DepthRefine::new(engine, stream.clone())?;
    let (mh, mw) = refiner.model_hw();

    let mut src = OakSource::open(None, w, h, fps, stream.clone())?;
    if !src.has_depth() {
        return Err("OAK device has no depth stream (need a stereo pair)".into());
    }
    println!(
        "OAK {}×{} depth={} → lingbot-depth @ {mh}×{mw}; warmup 10, measure {iters}",
        src.width(),
        src.height(),
        src.has_depth()
    );

    let (mut infer_ms, mut e2e_ms) = (Vec::with_capacity(iters), Vec::with_capacity(iters));
    let mut measured = 0usize;
    let mut warmup = 10i32;

    while measured < iters {
        let t_loop = Instant::now();
        let Some(frame) = src.next_frame() else { break };
        let Some(depth) = frame.depth() else { continue };

        let depth_t = depth_to_model_tensor(
            depth.as_slice(),
            depth.width() as usize,
            depth.height() as usize,
            mh,
            mw,
            &stream,
        )?;

        let t_infer = Instant::now();
        let _out = refiner.run(frame.rgb(), &depth_t)?;
        let infer = t_infer.elapsed().as_secs_f64() * 1000.0;
        let e2e = t_loop.elapsed().as_secs_f64() * 1000.0;

        if warmup > 0 {
            warmup -= 1;
            continue;
        }
        infer_ms.push(infer);
        e2e_ms.push(e2e);
        measured += 1;
    }

    if infer_ms.is_empty() {
        return Err("no frames captured from the OAK".into());
    }
    let stats = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        let p = |q: f64| v[((q * (n - 1) as f64).round() as usize).min(n - 1)];
        (mean, p(0.50), p(0.99))
    };
    let (im, ip50, ip99) = stats(&mut infer_ms);
    let (em, ep50, ep99) = stats(&mut e2e_ms);
    println!(
        "\nDepthRefine inference: mean {im:.2} ms  p50 {ip50:.2}  p99 {ip99:.2}  ({:.1} fps)",
        1000.0 / im
    );
    println!("end-to-end (capture+resize+refine): mean {em:.2} ms  p50 {ep50:.2}  p99 {ep99:.2}  ({:.1} fps)", 1000.0 / em);
    Ok(())
}
