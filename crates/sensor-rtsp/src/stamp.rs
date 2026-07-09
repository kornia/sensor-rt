//! Frame provenance: small value types that travel with each decoded frame.
//!
//! This source is a pure driver — orchestration (message buses, publishing)
//! lives in the application. But *provenance* still needs to ride along with the
//! data: which frame, captured when, from which camera. `FrameMeta` carries that,
//! and `Stamped<T>` tags a value with it. (Vendored from vision-rt's `vrt::stamp`
//! so this crate has no dependency on the algorithm libraries.)

/// Provenance for one frame: a sequence number, an optional capture timestamp,
/// and an optional source identifier (for multi-camera setups).
#[derive(Debug, Clone, Default)]
pub struct FrameMeta {
    /// Monotonic frame counter (assigned by whoever drives the loop).
    pub seq: u64,
    /// Capture timestamp in nanoseconds (e.g. a camera PTS), if known.
    pub pts_ns: Option<u64>,
    /// Camera / stream identifier for multi-source setups.
    pub source_id: Option<u32>,
}

impl FrameMeta {
    /// Tag `data` with this metadata, producing a [`Stamped`] value.
    pub fn stamp<T>(&self, data: T) -> Stamped<T> {
        Stamped {
            meta: self.clone(),
            data,
        }
    }
}

/// A value tagged with the [`FrameMeta`] of the frame it came from.
///
/// Keeps provenance attached to data as it flows through the application — the
/// timestamp and source survive each hop.
#[derive(Debug, Clone)]
pub struct Stamped<T> {
    pub meta: FrameMeta,
    pub data: T,
}

impl<T> Stamped<T> {
    pub fn new(meta: FrameMeta, data: T) -> Self {
        Self { meta, data }
    }

    /// Transform the carried value, keeping the same metadata.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Stamped<U> {
        Stamped {
            meta: self.meta,
            data: f(self.data),
        }
    }
}
