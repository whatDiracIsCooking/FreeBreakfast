//! The raw C ABI of ffi/fb_gemm.h, declared by hand. Keep it in step with the
//! header: nothing checks the two against each other.

pub type Status = i32;
pub const FB_OK: Status = 0;
pub const FB_INVALID_ARGUMENT: Status = 1;
pub const FB_GPU_ERROR: Status = 2;
pub const FB_BLAS_ERROR: Status = 3;
pub const FB_FAILED: Status = 4;

pub type Op = i32;
pub const FB_OP_N: Op = 0;
pub const FB_OP_T: Op = 1;

/// Opaque: only ever behind a pointer.
#[repr(C)]
pub struct Resources {
    _opaque: [u8; 0],
}

/// Opaque: only ever behind a pointer.
#[repr(C)]
pub struct Buffer {
    _opaque: [u8; 0],
}

unsafe extern "C" {
    pub fn fb_last_error() -> *const std::ffi::c_char;

    pub fn fb_resources_new(dev_idx: i32, out: *mut *mut Resources) -> Status;
    pub fn fb_resources_free(res: *mut Resources);
    pub fn fb_resources_sync(res: *mut Resources) -> Status;

    pub fn fb_buffer_new(res: *mut Resources, len: usize, out: *mut *mut Buffer) -> Status;
    pub fn fb_buffer_free(buf: *mut Buffer);
    pub fn fb_buffer_len(buf: *const Buffer) -> usize;
    pub fn fb_buffer_upload(
        res: *mut Resources,
        buf: *mut Buffer,
        src: *const f64,
        len: usize,
    ) -> Status;
    pub fn fb_buffer_download(
        res: *mut Resources,
        buf: *const Buffer,
        dst: *mut f64,
        len: usize,
    ) -> Status;

    #[allow(clippy::too_many_arguments)]
    pub fn fb_dgemm(
        res: *mut Resources,
        op_a: Op,
        op_b: Op,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const Buffer,
        lda: i32,
        b: *const Buffer,
        ldb: i32,
        beta: f64,
        c: *mut Buffer,
        ldc: i32,
    ) -> Status;
}
