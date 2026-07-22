//! Background H.264 MP4 writer: feed it RGBA frames over a channel and it encodes
//! them to a playable `.mp4` via a GStreamer `appsrc → videoconvert → x264enc →
//! mp4mux → filesink` pipeline on its own thread, so encoding never blocks the
//! capture/inference loop. Shared by the tracking examples.

use std::sync::mpsc::Receiver;
use std::thread::{self, JoinHandle};

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::AppSrc;

/// A frame for the writer: tight RGBA bytes (`width*height*4`) + its dimensions.
pub type Frame = (Vec<u8>, u32, u32);

/// Spawn the writer thread. Send `(rgba, w, h)` frames on the paired `SyncSender`;
/// drop it to flush EOS and finalize the mp4. The pipeline is built lazily on the
/// first frame (it needs the dimensions). Join the returned handle to wait for the
/// mp4 trailer to be written (do NOT just kill the process mid-record).
pub fn spawn_writer(rx: Receiver<Frame>, out_path: String, fps: u64) -> JoinHandle<()> {
    thread::spawn(move || run(rx, &out_path, fps))
}

fn run(rx: Receiver<Frame>, out_path: &str, fps: u64) {
    let _ = gst::init(); // idempotent; callers needn't init GStreamer themselves
    let frame_ns = 1_000_000_000 / fps.max(1);
    let mut pipe: Option<(gst::Pipeline, AppSrc)> = None;
    let mut idx = 0u64;

    for (buf, w, h) in rx {
        if pipe.is_none() {
            match build(out_path, w, h, fps) {
                Ok(p) => pipe = Some(p),
                // Report and stop the writer (draining the channel) rather than
                // killing the whole capture process from a worker thread.
                Err(e) => {
                    eprintln!("[mp4] pipeline build failed: {e}");
                    break;
                }
            }
        }
        let (_pipeline, appsrc) = pipe.as_ref().unwrap();
        let mut gbuf = gst::Buffer::from_slice(buf);
        {
            let b = gbuf.get_mut().unwrap();
            b.set_pts(gst::ClockTime::from_nseconds(idx * frame_ns));
            b.set_duration(gst::ClockTime::from_nseconds(frame_ns));
        }
        if appsrc.push_buffer(gbuf).is_err() {
            break;
        }
        idx += 1;
    }

    if let Some((pipeline, appsrc)) = pipe {
        let _ = appsrc.end_of_stream();
        if let Some(bus) = pipeline.bus() {
            let _ = bus.timed_pop_filtered(
                gst::ClockTime::from_seconds(10),
                &[gst::MessageType::Eos, gst::MessageType::Error],
            );
        }
        let _ = pipeline.set_state(gst::State::Null);
        println!("[mp4] wrote {idx} frames → {out_path}");
    }
}

fn build(
    out_path: &str,
    w: u32,
    h: u32,
    fps: u64,
) -> Result<(gst::Pipeline, AppSrc), gst::glib::Error> {
    let desc = format!(
        "appsrc name=src is-live=true format=time do-timestamp=false \
         ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency bitrate=8000 \
         ! mp4mux ! filesink location={out_path}"
    );
    let pipeline = gst::parse_launch(&desc)?
        .downcast::<gst::Pipeline>()
        .expect("not a pipeline");
    let appsrc = pipeline
        .by_name("src")
        .expect("no appsrc")
        .downcast::<AppSrc>()
        .expect("src not AppSrc");
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGBA")
        .field("width", w as i32)
        .field("height", h as i32)
        .field("framerate", gst::Fraction::new(fps as i32, 1))
        .build();
    appsrc.set_caps(Some(&caps));
    appsrc.set_format(gst::Format::Time);
    pipeline
        .set_state(gst::State::Playing)
        .map_err(|e| gst::glib::Error::new(gst::CoreError::StateChange, &e.to_string()))?;
    Ok((pipeline, appsrc))
}
