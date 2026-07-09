//! Minimal RTSP source demo — no models, just the driver.
//!
//! Connects to an RTSP/H.264 stream, hardware-decodes over NVMM, pumps a few
//! frames, then copies the last device-resident RGBA frame to the host and saves
//! it as a PNG.
//!
//! Usage:
//!   cargo run --release -p sensor-rtsp --example rtsp_grab -- rtsp://<camera>/stream [out.png] [frames]

use cudarc::driver::CudaContext;
use kornia_io::png::write_image_png_rgb8;
use sensor_rtsp::RtspSource;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rtsp_grab <rtsp://url> [out.png] [frames]");
        std::process::exit(1);
    }
    let url = &args[1];
    let out = args.get(2).map(String::as_str).unwrap_or("frame.png");
    let frames: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30);

    let stream = CudaContext::new(0)?.default_stream();
    let mut source = RtspSource::connect_resized(url, 1280, 720, stream.clone())?;
    println!("connected {url} → {}x{}", source.width(), source.height());

    // Pump frames to warm the decoder; copy the last one (device RGBA → host).
    let mut host = None;
    let mut got = 0u64;
    while got < frames {
        let Some(frame) = source.next_frame() else {
            break;
        };
        got += 1;
        if got.is_multiple_of(10) {
            println!("  frame {got} (seq {})", frame.meta.seq);
        }
        if got == frames {
            // Device RGB → host (D2H is ordered after the frame's copy on the
            // shared stream); to_host_image completes the transfer.
            host = Some(frame.image().to_host_image(&stream)?);
        }
    }

    match host {
        Some(img) => {
            write_image_png_rgb8(out, &img)?;
            println!("saved {out}");
        }
        None => eprintln!("no frame captured"),
    }
    Ok(())
}
