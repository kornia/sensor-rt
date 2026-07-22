//! Sensor-side shared types for `sensor-rt`.
//!
//! These used to come from the `edgarriba/vision-rt` fork's `vrt` crate. The
//! workspace now tracks **upstream `kornia/vision-rt`**, which deliberately does
//! not carry them — and rightly so: frame timing metadata and a borrowed host
//! depth map are *producer* concepts, not inference-runtime ones. Upstream's
//! Scope is deliberately narrow: only what EVERY sensor emits — frame timing.
//! Anything specific to one device (OAK depth maps, OAK intrinsics) lives in that
//! device's own driver crate instead, so this stays a tiny leaf that costs nothing
//! to depend on.
//!
//! Kept dependency-free so every sensor crate can depend on it downward.

/// Per-frame metadata: a monotonically increasing sequence number plus the
/// sensor's own capture timestamp.
///
/// `pts_ns` is the **capture** time reported by the device (epoch nanoseconds
/// where the driver can put it on that timeline), not the time the host received
/// the frame — the distinction that makes multi-sensor fusion possible. `seq` is
/// owned by the producing source and counts frames it actually emitted, so gaps
/// are meaningful (a dropped frame).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameMeta {
    /// Frame sequence number, assigned by the source, starting at 1.
    pub seq: u64,
    /// Device capture timestamp in nanoseconds, when the source can supply one.
    pub pts_ns: Option<u64>,
    /// Which camera produced this frame, for multi-camera setups.
    pub source_id: Option<u32>,
}

/// A value carrying the [`FrameMeta`] of the frame it came from.
///
/// Keeps timing attached to the payload as it moves through a pipeline, so a
/// downstream stage never has to guess which frame a result belongs to.
#[derive(Debug, Clone, Copy)]
pub struct Stamped<T> {
    pub meta: FrameMeta,
    pub data: T,
}

impl<T> Stamped<T> {
    pub fn new(meta: FrameMeta, data: T) -> Self {
        Self { meta, data }
    }
}
