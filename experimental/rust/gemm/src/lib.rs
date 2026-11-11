//! Safe Rust over a GPU stream, device buffers and an fp64 GEMM, through
//! WarpWraps' C ABI (the `wwr-sys` crate): no C++ of this tree's own in
//! between, and nothing here names CUDA or HIP. The backend is whichever the
//! linked `libwwr_capi_*.so` was built for.
//!
//! - [`Gpu`] owns a stream on one device and a BLAS handle bound to it. It is
//!   neither `Send` nor `Sync`: one driving thread at a time.
//! - [`DeviceBuffer`] is `len` doubles on the device, allocated and freed in
//!   stream order on its `Gpu`'s stream, which it borrows so it cannot
//!   outlive it.
//! - [`Matrix`] is a column-major `rows x cols` view of a buffer, and
//!   [`gemm`] is `C = alpha op(A) op(B) + beta C` over three of them, shapes
//!   checked before anything reaches the device.
//!
//! Host <-> device copies go through pinned host memory and a BLAS `dcopy`
//! on the stream, not `wwrMemcpyAsync`: the C ABI does not wrap that yet
//! (its `wwrMemcpyKind` argument is a by-value enum wwr's generator has not
//! registered; src/capi/README.md "Surface completeness"). Every failure is
//! an [`Error`] naming the call that failed; nothing aborts.

use std::ffi::c_void;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;

use wwr_sys::{
    wwr_blas_handle, wwr_blas_operation, wwr_blas_pointer_mode, wwr_blas_status, wwr_status,
    wwr_stream,
};

/// Why a call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A copy longer than the buffer it reads or writes.
    InvalidArgument,
    /// A GPU runtime call returned an error: which call, and its status.
    Gpu(&'static str, wwr_status),
    /// A BLAS call returned an error: which call, and its status.
    Blas(&'static str, wwr_blas_status),
    /// The shapes passed to [`gemm`] do not line up.
    Shape(String),
    /// A free_breakfast routine refused or failed, with its message (for a
    /// C ABI built beside this crate, such as rust/s_overlap's).
    Failed(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidArgument => write!(f, "invalid argument"),
            Error::Gpu(call, status) => write!(f, "{call}: {status:?}"),
            Error::Blas(call, status) => write!(f, "{call}: {status:?}"),
            Error::Shape(what) => write!(f, "shape mismatch: {what}"),
            Error::Failed(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn rt(call: &'static str, status: wwr_status) -> Result<()> {
    match status {
        wwr_status::WWR_STATUS_SUCCESS => Ok(()),
        other => Err(Error::Gpu(call, other)),
    }
}

fn blas(call: &'static str, status: wwr_blas_status) -> Result<()> {
    match status {
        wwr_blas_status::WWR_BLAS_STATUS_SUCCESS => Ok(()),
        other => Err(Error::Blas(call, other)),
    }
}

/// A stream on one device and a BLAS handle bound to it, with host pointer
/// mode for alpha and beta. Dropping it waits for the stream, then frees
/// both; every [`DeviceBuffer`] borrows it, so none can outlive it.
///
/// The device is made current on the creating thread, and not again: drive
/// two `Gpu`s on different devices from one thread and allocations follow
/// whichever was made last.
pub struct Gpu {
    device: i32,
    stream: wwr_stream, // raw pointers: !Send and !Sync, on purpose
    handle: wwr_blas_handle,
}

impl Gpu {
    /// A new stream and BLAS handle on device `dev_idx`.
    pub fn new(dev_idx: i32) -> Result<Self> {
        // SAFETY: plain runtime calls; each out-pointer is a valid local, set
        // only on success, and a partial Gpu is torn down below on failure.
        unsafe {
            rt("wwr_set_device", wwr_sys::wwr_set_device(dev_idx))?;
            let mut stream = ptr::null_mut();
            rt("wwr_stream_create", wwr_sys::wwr_stream_create(&mut stream))?;
            let mut handle = ptr::null_mut();
            if let Err(e) = blas("wwr_blas_create", wwr_sys::wwr_blas_create(&mut handle)) {
                wwr_sys::wwr_stream_destroy(stream);
                return Err(e);
            }
            let gpu = Gpu {
                device: dev_idx,
                stream,
                handle,
            };
            blas(
                "wwr_blas_set_stream",
                wwr_sys::wwr_blas_set_stream(handle, stream),
            )?;
            blas(
                "wwr_blas_set_pointer_mode",
                wwr_sys::wwr_blas_set_pointer_mode(
                    handle,
                    wwr_blas_pointer_mode::WWR_BLAS_POINTER_MODE_HOST,
                ),
            )?;
            Ok(gpu)
        }
    }

    /// The device index this `Gpu` was made on.
    pub fn device(&self) -> i32 {
        self.device
    }

    /// The stream everything here is queued on, for a C ABI that must order
    /// work against it. It stays owned by `self`.
    pub fn stream(&self) -> wwr_stream {
        self.stream
    }

    /// Block until everything queued on the stream has finished.
    pub fn sync(&self) -> Result<()> {
        // SAFETY: the stream is live for as long as `self` is.
        rt("wwr_stream_synchronize", unsafe {
            wwr_sys::wwr_stream_synchronize(self.stream)
        })
    }

    /// `len` uninitialised doubles on this device, allocated in stream order.
    pub fn alloc(&self, len: usize) -> Result<DeviceBuffer<'_>> {
        let mut raw: *mut c_void = ptr::null_mut();
        if len != 0 {
            let bytes = len
                .checked_mul(size_of::<f64>())
                .ok_or(Error::InvalidArgument)?;
            // SAFETY: `raw` is a valid out-pointer; the stream is live.
            rt("wwr_malloc_async", unsafe {
                wwr_sys::wwr_malloc_async(&mut raw, bytes, self.stream)
            })?;
        }
        Ok(DeviceBuffer {
            gpu: self,
            ptr: raw.cast(),
            len,
            _not_send: PhantomData,
        })
    }

    /// A device copy of `host`.
    pub fn upload(&self, host: &[f64]) -> Result<DeviceBuffer<'_>> {
        let mut buf = self.alloc(host.len())?;
        buf.copy_from(host)?;
        Ok(buf)
    }

    /// `y[..n] = x[..n]` on the stream, either side device memory or a
    /// pinned buffer's device view.
    fn dcopy(&self, n: usize, x: *const f64, y: *mut f64) -> Result<()> {
        let n = i64::try_from(n).map_err(|_| Error::InvalidArgument)?;
        // SAFETY: the caller passes n valid doubles on each side.
        blas("wwr_blas_dcopy_64", unsafe {
            wwr_sys::wwr_blas_dcopy_64(self.handle, n, x, 1, y, 1)
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: both are live and no buffer still borrows them. Errors on
        // teardown have nowhere to go.
        unsafe {
            wwr_sys::wwr_stream_synchronize(self.stream);
            wwr_sys::wwr_blas_destroy(self.handle);
            wwr_sys::wwr_stream_destroy(self.stream);
        }
    }
}

/// `len` doubles of pinned host memory, and the device's view of them: the
/// staging area of one host <-> device copy.
struct Pinned {
    host: *mut f64,
    device: *mut f64,
}

impl Pinned {
    fn new(len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<f64>())
            .ok_or(Error::InvalidArgument)?;
        let mut host: *mut c_void = ptr::null_mut();
        // SAFETY: valid out-pointers; flags 0 is the default on both
        // backends, and the allocation is freed on drop once it exists.
        unsafe {
            rt(
                "wwr_host_alloc",
                wwr_sys::wwr_host_alloc(&mut host, bytes, 0),
            )?;
            let mut pinned = Pinned {
                host: host.cast(),
                device: ptr::null_mut(),
            };
            let mut device: *mut c_void = ptr::null_mut();
            rt(
                "wwr_host_get_device_pointer",
                wwr_sys::wwr_host_get_device_pointer(&mut device, host, 0),
            )?;
            pinned.device = device.cast();
            Ok(pinned)
        }
    }
}

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: allocated by wwr_host_alloc; any copy using it was waited for.
        unsafe {
            wwr_sys::wwr_free_host(self.host.cast());
        }
    }
}

/// `len` doubles on the device, freed in stream order on drop.
pub struct DeviceBuffer<'g> {
    gpu: &'g Gpu,
    ptr: *mut f64,
    len: usize,
    _not_send: PhantomData<*mut ()>,
}

impl DeviceBuffer<'_> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device address, for a C ABI that reads the buffer. Null when empty.
    pub fn as_ptr(&self) -> *const f64 {
        self.ptr
    }

    /// The device address, for a C ABI that writes the buffer. Work queued on
    /// another stream must first wait for this buffer's [`Gpu::sync`].
    pub fn as_mut_ptr(&mut self) -> *mut f64 {
        self.ptr
    }

    /// Copy `host` into the buffer's start, and wait for it.
    pub fn copy_from(&mut self, host: &[f64]) -> Result<()> {
        if host.len() > self.len {
            return Err(Error::InvalidArgument);
        }
        if host.is_empty() {
            return Ok(());
        }
        let staging = Pinned::new(host.len())?;
        // SAFETY: the staging area holds host.len() doubles and is not yet in
        // use on the device.
        unsafe { ptr::copy_nonoverlapping(host.as_ptr(), staging.host, host.len()) };
        self.gpu.dcopy(host.len(), staging.device, self.ptr)?;
        self.gpu.sync()
    }

    /// Copy the buffer's first `host.len()` doubles into `host`, and wait.
    pub fn copy_to(&self, host: &mut [f64]) -> Result<()> {
        if host.len() > self.len {
            return Err(Error::InvalidArgument);
        }
        if host.is_empty() {
            return Ok(());
        }
        let staging = Pinned::new(host.len())?;
        self.gpu.dcopy(host.len(), self.ptr, staging.device)?;
        self.gpu.sync()?;
        // SAFETY: the copy into the staging area has finished.
        unsafe { ptr::copy_nonoverlapping(staging.host, host.as_mut_ptr(), host.len()) };
        Ok(())
    }

    /// The whole buffer, on the host.
    pub fn to_vec(&self) -> Result<Vec<f64>> {
        let mut host = vec![0.0; self.len];
        self.copy_to(&mut host)?;
        Ok(host)
    }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated on this stream, which outlives the buffer by
            // the borrow; the free is ordered after the work queued on it.
            unsafe {
                wwr_sys::wwr_free_async(self.ptr.cast(), self.gpu.stream);
            }
        }
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

    /// A `rows x cols` matrix over `data`, which must hold exactly that many
    /// values, column-major.
    pub fn from_buffer(rows: usize, cols: usize, data: DeviceBuffer<'g>) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::Shape(format!(
                "a buffer of {} for a {rows} x {cols} matrix",
                data.len()
            )));
        }
        Ok(Matrix { rows, cols, data })
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
    fn raw(self) -> wwr_blas_operation {
        match self {
            Op::N => wwr_blas_operation::WWR_BLAS_OP_N,
            Op::T => wwr_blas_operation::WWR_BLAS_OP_T,
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
    // SAFETY: the handle is live and `gpu`'s; each Matrix holds exactly
    // rows * cols doubles with leading dimension rows, and the shapes are
    // checked above. alpha and beta are host scalars (host pointer mode).
    blas("wwr_blas_dgemm", unsafe {
        wwr_sys::wwr_blas_dgemm(
            gpu.handle,
            op_a.raw(),
            op_b.raw(),
            dim(m, "m")?,
            dim(n, "n")?,
            dim(k, "k")?,
            &alpha,
            a.data.ptr,
            ld(a.rows)?,
            b.data.ptr,
            ld(b.rows)?,
            &beta,
            c.data.ptr,
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
    fn copy_rejects_overlong_host() {
        let gpu = Gpu::new(0).unwrap();
        let mut buf = gpu.alloc(2).unwrap();
        assert_eq!(buf.copy_from(&[0.0; 3]), Err(Error::InvalidArgument));
        assert_eq!(buf.copy_to(&mut [0.0; 3]), Err(Error::InvalidArgument));
    }

    #[test]
    fn buffer_round_trips() {
        let gpu = Gpu::new(0).unwrap();
        let host: Vec<f64> = (0..1000).map(f64::from).collect();
        let buf = gpu.upload(&host).unwrap();
        assert_eq!(buf.len(), host.len());
        assert_eq!(buf.to_vec().unwrap(), host);
    }

    #[test]
    fn empty_matrix_is_fine() {
        let gpu = Gpu::new(0).unwrap();
        let a = Matrix::zeros(&gpu, 0, 3).unwrap();
        let b = Matrix::zeros(&gpu, 3, 0).unwrap();
        let mut c = Matrix::zeros(&gpu, 0, 0).unwrap();
        gemm(&gpu, 1.0, Op::N, &a, Op::N, &b, 0.0, &mut c).unwrap();
        assert!(c.to_vec().unwrap().is_empty());
    }
}
