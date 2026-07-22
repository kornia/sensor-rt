//! Host depth map produced by the OAK's on-device stereo block.
//!
//! Lives in this crate rather than a shared one because it is an OAK-shaped type:
//! uint16 millimetres, borrowed straight out of the depthai frame, valid only
//! until the next poll. A driver that produced metric f32 or owned its buffer
//! would want a different type, and forcing both through one shared definition
//! would serve neither.

/// A **host** depth map in millimetres (`u16`, `0` = no valid measurement).
///
/// Host-resident on purpose: per-pixel / per-box sampling is cheap on the CPU,
/// and the data already lands in host RAM from USB depth sensors — uploading it
/// just to sample a few points would cost more than it saves.
///
/// Depth is expected **pixel-aligned to the image it will be sampled against**
/// (the producer's job), so [`meters_at`](Self::meters_at)`(u, v)` corresponds to
/// image pixel `(u, v)`. When the map is a smaller grid than the image (sensors
/// often downscale depth before transport), the consumer scales coordinates by
/// `image_width / depth_width`.
pub struct OakDepthMap {
    ptr: *const u16,
    width: u32,
    height: u32,
}

// SAFETY: `ptr` addresses host memory that the producer guarantees stays valid for
// this map's lifetime (e.g. "valid until the next frame").
unsafe impl Send for OakDepthMap {}

impl OakDepthMap {
    /// Borrow a producer's `width*height` u16 depth buffer — **zero-copy**.
    ///
    /// # Safety
    /// `ptr` must address at least `width*height` valid `u16`s that stay alive for
    /// as long as this map is used. Producers document the window (typically
    /// "valid until the next frame").
    pub unsafe fn borrowed(ptr: *const u16, width: u32, height: u32) -> Self {
        Self { ptr, width, height }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// The raw millimetre samples.
    pub fn as_slice(&self) -> &[u16] {
        // SAFETY: the constructors guarantee `ptr` covers width*height u16s for
        // this map's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.width as usize * self.height as usize) }
    }

    /// Fraction of pixels carrying a valid measurement, in `0.0..=1.0`.
    ///
    /// The single most useful health check on a passive-stereo depth stream: a
    /// sensor that is running but seeing a textureless wall returns frames that
    /// look structurally fine and are almost entirely zeros.
    pub fn valid_fraction(&self) -> f32 {
        let n = self.width as usize * self.height as usize;
        if n == 0 {
            return 0.0;
        }
        self.as_slice().iter().filter(|&&v| v != 0).count() as f32 / n as f32
    }

    /// Raw millimetre reading at `(u, v)`, or `None` when out of bounds or the
    /// pixel has no valid measurement (`0`).
    fn mm_at(&self, u: u32, v: u32) -> Option<u16> {
        if u >= self.width || v >= self.height {
            return None;
        }
        let mm = self.as_slice()[v as usize * self.width as usize + u as usize];
        (mm != 0).then_some(mm)
    }

    /// Depth at `(u, v)` in **metres**, or `None` when out of bounds or invalid.
    pub fn meters_at(&self, u: u32, v: u32) -> Option<f32> {
        self.mm_at(u, v).map(|mm| mm as f32 / 1000.0)
    }
}
