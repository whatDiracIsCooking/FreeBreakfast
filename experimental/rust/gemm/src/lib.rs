//! Safe Rust over free_breakfast's DeviceResources, its device buffers and an
//! fp64 GEMM, through the C ABI in ffi/fb_gemm.h.
//!
//! - [`Gpu`] owns one DeviceResources: a stream, its memory pool and a BLAS
//!   handle. It is neither `Send` nor `Sync`, which is DeviceResources' own
//!   rule: one driving thread at a time.
//! - [`DeviceBuffer`] is `len` doubles on the device, borrowed from its `Gpu`
//!   so it cannot outlive it, and freed on drop.
//! - [`Matrix`] is a column-major `rows x cols` view of a buffer, and
//!   [`gemm`] is `C = alpha op(A) op(B) + beta C` over three of them, shapes
//!   checked before anything reaches the device.
//!
//! A failed GPU creation or allocation aborts the process, as it does
//! everywhere in free_breakfast; every other failure is an [`Error`].

mod ffi;

use std::fmt;
use std::marker::PhantomData;
use std::ptr;

/// Why a call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The C side refused an argument: a null, a negative size, or a buffer
    /// too small for its matrix.
    InvalidArgument,
    /// The GPU runtime returned an error on a copy or a synchronize.
    Gpu,
    /// The BLAS call failed.
    Blas,
    /// The shapes passed to [`gemm`] do not line up; caught on the Rust side.
    Shape(String),
    /// A status the C side does not document.
    Unknown(i32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument => write!(f, "invalid argument"),
            Error::Gpu => write!(f, "GPU runtime error"),
            Error::Blas => write!(f, "BLAS error"),
            Error::Shape(what) => write!(f, "shape mismatch: {what}"),
            Error::Unknown(code) => write!(f, "unknown status {code}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn check(status: ffi::Status) -> Result<()> {
    match status {
        ffi::FB_OK => Ok(()),
        ffi::FB_INVALID_ARGUMENT => Err(Error::InvalidArgument),
        ffi::FB_GPU_ERROR => Err(Error::Gpu),
        ffi::FB_BLAS_ERROR => Err(Error::Blas),
        other => Err(Error::Unknown(other)),
    }
}

/// One device's DeviceResources. Dropping it frees the stream, pool and
/// handle; every [`DeviceBuffer`] borrows it, so none can outlive it.
pub struct Gpu {
    raw: *mut ffi::Resources, // a raw pointer: !Send and !Sync, on purpose
}

impl Gpu {
    /// DeviceResources on device `dev_idx`.
    pub fn new(dev_idx: i32) -> Result<Self> {
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer; on FB_OK it holds a live handle.
        check(unsafe { ffi::fb_resources_new(dev_idx, &mut raw) })?;
        Ok(Gpu { raw })
    }

    /// Block until everything queued on the stream has finished.
    pub fn sync(&self) -> Result<()> {
        // SAFETY: `self.raw` is live for as long as `self` is.
        check(unsafe { ffi::fb_resources_sync(self.raw) })
    }

    /// `len` uninitialised doubles on this device.
    pub fn alloc(&self, len: usize) -> Result<DeviceBuffer<'_>> {
        let mut raw = ptr::null_mut();
        // SAFETY: `self.raw` is live; `raw` is a valid out-pointer.
        check(unsafe { ffi::fb_buffer_new(self.raw, len, &mut raw) })?;
        Ok(DeviceBuffer {
            gpu: self,
            raw,
            _not_send: PhantomData,
        })
    }

    /// A device copy of `host`.
    pub fn upload(&self, host: &[f64]) -> Result<DeviceBuffer<'_>> {
        let mut buf = self.alloc(host.len())?;
        buf.copy_from(host)?;
        Ok(buf)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: the handle is live and no buffer still borrows it.
        unsafe { ffi::fb_resources_free(self.raw) }
    }
}

/// `len` doubles on the device, from its [`Gpu`]'s pool.
pub struct DeviceBuffer<'g> {
    gpu: &'g Gpu,
    raw: *mut ffi::Buffer,
    _not_send: PhantomData<*mut ()>,
}

impl DeviceBuffer<'_> {
    pub fn len(&self) -> usize {
        // SAFETY: `self.raw` is live for as long as `self` is.
        unsafe { ffi::fb_buffer_len(self.raw) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copy `host` into the buffer's start, and wait for it.
    pub fn copy_from(&mut self, host: &[f64]) -> Result<()> {
        // SAFETY: both handles are live; `host` is valid for host.len() reads,
        // and the C side checks host.len() against the buffer's length.
        check(unsafe { ffi::fb_buffer_upload(self.gpu.raw, self.raw, host.as_ptr(), host.len()) })
    }

    /// Copy the buffer's first `host.len()` doubles into `host`, and wait.
    pub fn copy_to(&self, host: &mut [f64]) -> Result<()> {
        // SAFETY: as copy_from, for host.len() writes.
        check(unsafe {
            ffi::fb_buffer_download(self.gpu.raw, self.raw, host.as_mut_ptr(), host.len())
        })
    }

    /// The whole buffer, on the host.
    pub fn to_vec(&self) -> Result<Vec<f64>> {
        let mut host = vec![0.0; self.len()];
        self.copy_to(&mut host)?;
        Ok(host)
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        // SAFETY: the handle is live; its Gpu outlives it by the borrow.
        unsafe { ffi::fb_buffer_free(self.raw) }
    }
}

/// A column-major `rows x cols` matrix on the device, leading dimension
/// `rows`.
pub struct Matrix<'g> {
    rows: usize,
    cols: usize,
    data: DeviceBuffer<'g>,
}

impl<'g> Matrix<'g> {
    /// A `rows x cols` matrix holding `host`, column-major.
    pub fn from_host(gpu: &'g Gpu, rows: usize, cols: usize, host: &[f64]) -> Result<Self> {
        if host.len() != rows * cols {
            return Err(Error::Shape(format!(
                "{} values for a {rows} x {cols} matrix",
                host.len()
            )));
        }
        Ok(Matrix {
            rows,
            cols,
            data: gpu.upload(host)?,
        })
    }

    /// A `rows x cols` matrix of zeros.
    pub fn zeros(gpu: &'g Gpu, rows: usize, cols: usize) -> Result<Self> {
        Self::from_host(gpu, rows, cols, &vec![0.0; rows * cols])
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// The matrix on the host, column-major.
    pub fn to_vec(&self) -> Result<Vec<f64>> {
        self.data.to_vec()
    }
}

/// Whether [`gemm`] takes a matrix as it is or transposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    N,
    T,
}

impl Op {
    fn raw(self) -> ffi::Op {
        match self {
            Op::N => ffi::FB_OP_N,
            Op::T => ffi::FB_OP_T,
        }
    }

    /// The shape of op(M) for an `rows x cols` M.
    fn apply(self, rows: usize, cols: usize) -> (usize, usize) {
        match self {
            Op::N => (rows, cols),
            Op::T => (cols, rows),
        }
    }
}

fn dim(n: usize, what: &str) -> Result<i32> {
    i32::try_from(n).map_err(|_| Error::Shape(format!("{what} = {n} overflows i32")))
}

/// `C = alpha op(A) op(B) + beta C`, queued on `gpu`'s stream; read C back
/// (or [`Gpu::sync`]) to wait for it. All three matrices must be `gpu`'s.
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    gpu: &Gpu,
    alpha: f64,
    op_a: Op,
    a: &Matrix<'_>,
    op_b: Op,
    b: &Matrix<'_>,
    beta: f64,
    c: &mut Matrix<'_>,
) -> Result<()> {
    for (name, m) in [("A", a), ("B", b)] {
        if !ptr::eq(m.data.gpu, gpu) {
            return Err(Error::Shape(format!("{name} belongs to another Gpu")));
        }
    }
    if !ptr::eq(c.data.gpu, gpu) {
        return Err(Error::Shape("C belongs to another Gpu".into()));
    }
    let (m, k) = op_a.apply(a.rows, a.cols);
    let (kb, n) = op_b.apply(b.rows, b.cols);
    if k != kb {
        return Err(Error::Shape(format!(
            "op(A) is {m} x {k} but op(B) is {kb} x {n}"
        )));
    }
    if (c.rows, c.cols) != (m, n) {
        return Err(Error::Shape(format!(
            "C is {} x {}, op(A) op(B) is {m} x {n}",
            c.rows, c.cols
        )));
    }
    // BLAS wants a leading dimension of at least 1, even for an empty matrix.
    let ld = |rows: usize| dim(rows.max(1), "leading dimension");
    // SAFETY: every handle is live and `gpu`'s; the shapes are checked above,
    // and the C side checks each buffer holds its matrix.
    check(unsafe {
        ffi::fb_dgemm(
            gpu.raw,
            op_a.raw(),
            op_b.raw(),
            dim(m, "m")?,
            dim(n, "n")?,
            dim(k, "k")?,
            alpha,
            a.data.raw,
            ld(a.rows)?,
            b.data.raw,
            ld(b.rows)?,
            beta,
            c.data.raw,
            ld(c.rows)?,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // [[1 2] [3 4]], column-major.
    const A: [f64; 4] = [1.0, 3.0, 2.0, 4.0];
    // [[5 6] [7 8]], column-major.
    const B: [f64; 4] = [5.0, 7.0, 6.0, 8.0];

    #[test]
    fn gemm_2x2_every_op() {
        let gpu = Gpu::new(0).unwrap();
        let a = Matrix::from_host(&gpu, 2, 2, &A).unwrap();
        let b = Matrix::from_host(&gpu, 2, 2, &B).unwrap();
        let cases = [
            (Op::N, Op::N, [19.0, 43.0, 22.0, 50.0]), // AB
            (Op::T, Op::N, [26.0, 38.0, 30.0, 44.0]), // A^T B
            (Op::N, Op::T, [17.0, 39.0, 23.0, 53.0]), // A B^T
            (Op::T, Op::T, [23.0, 34.0, 31.0, 46.0]), // A^T B^T
        ];
        for (op_a, op_b, want) in cases {
            let mut c = Matrix::zeros(&gpu, 2, 2).unwrap();
            gemm(&gpu, 1.0, op_a, &a, op_b, &b, 0.0, &mut c).unwrap();
            assert_eq!(c.to_vec().unwrap(), want, "{op_a:?} {op_b:?}");
        }
    }

    #[test]
    fn gemm_accumulates_with_beta() {
        let gpu = Gpu::new(0).unwrap();
        let a = Matrix::from_host(&gpu, 2, 2, &A).unwrap();
        let b = Matrix::from_host(&gpu, 2, 2, &B).unwrap();
        let mut c = Matrix::from_host(&gpu, 2, 2, &[1.0; 4]).unwrap();
        gemm(&gpu, 2.0, Op::N, &a, Op::N, &b, -1.0, &mut c).unwrap();
        assert_eq!(c.to_vec().unwrap(), [37.0, 85.0, 43.0, 99.0]);
    }

    #[test]
    fn gemm_rejects_mismatched_shapes() {
        let gpu = Gpu::new(0).unwrap();
        let a = Matrix::zeros(&gpu, 2, 3).unwrap();
        let b = Matrix::zeros(&gpu, 2, 2).unwrap();
        let mut c = Matrix::zeros(&gpu, 2, 2).unwrap();
        let err = gemm(&gpu, 1.0, Op::N, &a, Op::N, &b, 0.0, &mut c).unwrap_err();
        assert!(matches!(err, Error::Shape(_)), "{err}");
    }

    #[test]
    fn from_host_rejects_wrong_length() {
        let gpu = Gpu::new(0).unwrap();
        assert!(matches!(
            Matrix::from_host(&gpu, 2, 2, &[0.0; 3]),
            Err(Error::Shape(_))
        ));
    }

    #[test]
    fn buffer_round_trips() {
        let gpu = Gpu::new(0).unwrap();
        let host: Vec<f64> = (0..1000).map(f64::from).collect();
        let buf = gpu.upload(&host).unwrap();
        assert_eq!(buf.len(), host.len());
        assert_eq!(buf.to_vec().unwrap(), host);
    }
}
