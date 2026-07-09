use std::os::raw::c_void;

#[cfg(not(nvbuf_stub))]
extern "C" {
    /// **DEPRECATED** — `dataPtr` is documented invalid for `NVBUF_MEM_SURFACE_ARRAY`
    /// (Jetson default).  Use `nvbuf_cuda_import` for all NVMM CUDA access.
    /// Only valid for `NVBUF_MEM_CUDA_UNIFIED` buffers.
    pub fn nvbuf_cuda_ptr(nvbuf_surface: *const c_void) -> *mut c_void;

    /// Width in pixels of the first surface.
    pub fn nvbuf_width(nvbuf_surface: *const c_void) -> u32;

    /// Height in pixels of the first surface.
    pub fn nvbuf_height(nvbuf_surface: *const c_void) -> u32;

    /// Row stride in bytes of the first surface.
    pub fn nvbuf_pitch(nvbuf_surface: *const c_void) -> u32;

    /// DMA-BUF file descriptor (-1 on error).  Valid for SURFACE_ARRAY / HANDLE.
    /// Caller retains ownership; do not close the returned FD.
    pub fn nvbuf_dmabuf_fd(nvbuf_surface: *const c_void) -> i32;

    /// Memory layout: 0 = pitch-linear, 1 = block-linear, -1 = error.
    /// Only pitch-linear (0) surfaces can be imported for linear CUDA access.
    pub fn nvbuf_layout(nvbuf_surface: *const c_void) -> i32;

    /// Total allocated bytes for the first surface.
    pub fn nvbuf_data_size(nvbuf_surface: *const c_void) -> u64;

    /// Import a DMA-BUF FD into CUDA as an external memory object.
    ///
    /// Returns 0 on success.  Fills `*ext_mem_out` and `*dev_ptr_out` with
    /// handles that **must** be released via `nvbuf_cuda_release` after the
    /// consuming CUDA stream is synced.
    ///
    /// Internally `dup()`s `fd` — caller retains ownership of the original.
    /// Must be called from a thread where the CUDA primary context is active.
    pub fn nvbuf_cuda_import(
        fd: i32,
        size: u64,
        ext_mem_out: *mut *mut c_void,
        dev_ptr_out: *mut *mut c_void,
    ) -> i32;

    /// Release handles returned by `nvbuf_cuda_import`.  Call only after sync.
    pub fn nvbuf_cuda_release(ext_mem: *mut c_void, dev_ptr: *mut c_void);
}

// Off-Jetson stub (`NVBUF_STUB=1`): same signatures, so the crate links without
// the native shim. These are NEVER called in stub builds — only pure-logic unit
// tests run there — so the bodies just return "error"/null placeholders.
#[cfg(nvbuf_stub)]
mod stub {
    use super::c_void;
    use std::ptr::null_mut;

    /// # Safety
    /// Stub — takes no action; signature matches the real FFI.
    pub unsafe fn nvbuf_cuda_ptr(_s: *const c_void) -> *mut c_void {
        null_mut()
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_width(_s: *const c_void) -> u32 {
        0
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_height(_s: *const c_void) -> u32 {
        0
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_pitch(_s: *const c_void) -> u32 {
        0
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_dmabuf_fd(_s: *const c_void) -> i32 {
        -1
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_layout(_s: *const c_void) -> i32 {
        -1
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_data_size(_s: *const c_void) -> u64 {
        0
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_cuda_import(
        _fd: i32,
        _size: u64,
        _ext_mem_out: *mut *mut c_void,
        _dev_ptr_out: *mut *mut c_void,
    ) -> i32 {
        -1
    }
    /// # Safety
    /// Stub.
    pub unsafe fn nvbuf_cuda_release(_ext_mem: *mut c_void, _dev_ptr: *mut c_void) {}
}

#[cfg(nvbuf_stub)]
pub use stub::*;
