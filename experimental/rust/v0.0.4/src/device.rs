//! The device layer, written over the C ABI and nothing else of `rust/`:
//! the current device, a stream, an event, the copies, device slices, and
//! [`GpuResources`]: one GPU's stream with the BLAS and solver handles made
//! on it, reached only through that GPU's guard.
//!
//! - **A handle keeps its stream.** `SetStream` is called once, when the
//!   handle is made; the handle workspace is bound once, by the wrapper at
//!   the first acquisition after the final node allocated it.
//! - **A binding takes slices and checks them before it enqueues.** Every
//!   BLAS, solver and kernel call here is a method over [`DeviceSlice`]s
//!   that holds the lengths against the call's own formula and panics at
//!   the caller's line, with nothing enqueued.
//! - **The recipes take the guard.** [`GpuResources::trsm`] and
//!   [`GpuResources::cholesky`] put their diagonal kernels and their gemms
//!   on the one stream the guard has, so a recipe cannot be handed another.
//!
//! The kernel launches a step puts on a stream of its own are the free
//! functions of [`launch`].

use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use fb_core_error_handling::{Error, Result, Status};
use fb_kernel_abi::LaunchStatus;
use fb_recipes_sys::{FB_CHOLESKY_TILE, FB_TRSM_TILE};
use wwr_sys::{
    wwr_blas_fill_mode, wwr_blas_handle, wwr_blas_operation, wwr_blas_side_mode, wwr_blas_status,
    wwr_event, wwr_memcpy_kind, wwr_solver_eig_mode, wwr_solver_fill_mode, wwr_solver_handle,
    wwr_solver_status, wwr_status, wwr_stream,
};

pub use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

/// A runtime call's status as a [`Status`].
#[track_caller]
pub fn runtime_status(code: wwr_status) -> Status {
    if code as i32 == 0 {
        return Ok(());
    }
    Err(Error::gpu(format!("{code:?}")))
}

#[track_caller]
fn blas_status(what: &str, code: wwr_blas_status) -> Status {
    if code as i32 == 0 {
        return Ok(());
    }
    Err(Error::gpu(format!("{what}: {code:?}")))
}

#[track_caller]
fn solver_status(what: &str, code: wwr_solver_status) -> Status {
    if code as i32 == 0 {
        return Ok(());
    }
    Err(Error::gpu(format!("{what}: {code:?}")))
}

/// A kernel launch wrapper's status as a [`Status`], naming `what`.
#[track_caller]
pub fn launched(what: &str, status: LaunchStatus) -> Status {
    fb_kernel_abi::launch_status(what, status, |code, _| {
        Some(Err(Error::gpu(format!(
            "runtime error {code} at the launch"
        ))))
    })
}

/// The devices the runtime sees.
///
/// # Errors
/// The runtime's failure.
pub fn device_count() -> Result<i32> {
    let mut n = 0;
    // SAFETY: writes one local.
    runtime_status(unsafe { wwr_sys::wwr_get_device_count(&mut n) })?;
    Ok(n)
}

/// A device made current on this thread, and the one before it restored on
/// drop. Launches and library calls run on the calling thread's current
/// device, and a stolen task runs on whatever thread is free, so every
/// inner task that launches holds one.
pub struct ScopedDevice {
    before: i32,
}

impl ScopedDevice {
    /// Make `physical` current.
    ///
    /// # Panics
    /// If the runtime refuses: a device the run was not set up on.
    pub fn new(physical: i32) -> ScopedDevice {
        let mut before = 0;
        // SAFETY: plain runtime calls; `before` is a valid local.
        unsafe {
            runtime_status(wwr_sys::wwr_get_device(&mut before)).expect("the current device");
            runtime_status(wwr_sys::wwr_set_device(physical)).expect("the device is the run's");
        }
        ScopedDevice { before }
    }
}

impl Drop for ScopedDevice {
    fn drop(&mut self) {
        // SAFETY: restores a device that was current.
        let _ = unsafe { wwr_sys::wwr_set_device(self.before) };
    }
}

/// A device's free and total memory, with it current.
///
/// # Errors
/// The runtime's failure.
pub fn memory(physical: i32) -> Result<(u64, u64)> {
    let _current = ScopedDevice::new(physical);
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: a plain runtime call writing two locals.
    runtime_status(unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) })?;
    Ok((free as u64, total as u64))
}

/// `len` elements of device memory, readable while `self` is borrowed.
///
/// # Safety
/// `as_ptr` is `len` elements of live device memory (or null with the
/// budget's dry backing, which never runs).
pub unsafe trait DeviceSlice<T> {
    /// The first element.
    fn as_ptr(&self) -> *const T;
    /// The elements.
    fn len(&self) -> usize;
    /// Whether there are none.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A [`DeviceSlice`] its holder may write.
///
/// # Safety
/// As [`DeviceSlice`], and `&mut self` is the one user.
pub unsafe trait DeviceSliceMut<T>: DeviceSlice<T> {
    /// The first element.
    fn as_mut_ptr(&mut self) -> *mut T;
}

/// A borrowed run of a device slice.
pub struct Part<'a, T> {
    ptr: *const T,
    len: usize,
    life: PhantomData<&'a [T]>,
}

impl<T> Clone for Part<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Part<'_, T> {}

// SAFETY: a view of device memory, not tied to a host thread.
unsafe impl<T: Sync> Send for Part<'_, T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for Part<'_, T> {}

// SAFETY: a run inside the slice it was cut from.
unsafe impl<T> DeviceSlice<T> for Part<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

/// A borrowed, writable run of a device slice.
pub struct PartMut<'a, T> {
    ptr: *mut T,
    len: usize,
    life: PhantomData<&'a mut [T]>,
}

// SAFETY: a view of device memory, not tied to a host thread.
unsafe impl<T: Send> Send for PartMut<'_, T> {}

// SAFETY: a run inside the slice it was cut from.
unsafe impl<T> DeviceSlice<T> for PartMut<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: cut from a `&mut` borrow, so the one user.
unsafe impl<T> DeviceSliceMut<T> for PartMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<'a, T> PartMut<'a, T> {
    /// A writable run made from a raw region the caller owns for `'a`.
    ///
    /// # Safety
    /// `ptr` is `len` elements of device memory with no other user for
    /// `'a`.
    pub(crate) unsafe fn from_raw(ptr: *mut T, len: usize) -> PartMut<'a, T> {
        PartMut {
            ptr,
            len,
            life: PhantomData,
        }
    }
}

/// Elements `[offset, offset + len)` of `of`.
///
/// # Panics
/// If the run is not inside `of`.
#[track_caller]
pub fn part<T>(of: &impl DeviceSlice<T>, offset: usize, len: usize) -> Part<'_, T> {
    assert!(
        offset + len <= of.len(),
        "part: [{offset}, {}) is past the slice's {}",
        offset + len,
        of.len()
    );
    Part {
        ptr: of.as_ptr().wrapping_add(offset),
        len,
        life: PhantomData,
    }
}

/// Elements `[offset, offset + len)` of `of`, writable.
///
/// # Panics
/// If the run is not inside `of`.
#[track_caller]
pub fn part_mut<T>(of: &mut impl DeviceSliceMut<T>, offset: usize, len: usize) -> PartMut<'_, T> {
    assert!(
        offset + len <= of.len(),
        "part_mut: [{offset}, {}) is past the slice's {}",
        offset + len,
        of.len()
    );
    PartMut {
        ptr: of.as_mut_ptr().wrapping_add(offset),
        len,
        life: PhantomData,
    }
}

/// `of` cut at `mid`: two writable runs that do not meet.
///
/// # Panics
/// If `mid` is past `of`.
#[track_caller]
pub fn split_at_mut<T>(
    of: &mut impl DeviceSliceMut<T>,
    mid: usize,
) -> (PartMut<'_, T>, PartMut<'_, T>) {
    let len = of.len();
    assert!(mid <= len, "split_at_mut: {mid} is past the slice's {len}");
    let ptr = of.as_mut_ptr();
    (
        PartMut {
            ptr,
            len: mid,
            life: PhantomData,
        },
        PartMut {
            ptr: ptr.wrapping_add(mid),
            len: len - mid,
            life: PhantomData,
        },
    )
}

/// The stream syncs of the process so far, and the time blocked in them.
static SYNCS: AtomicU64 = AtomicU64::new(0);
static SYNC_NANOS: AtomicU64 = AtomicU64::new(0);

/// How many times a stream was synchronized so far, by any thread, and the
/// time the callers spent blocked there. A measurement, read as a
/// difference over a span in which one step runs.
pub fn sync_meter() -> (u64, std::time::Duration) {
    (
        SYNCS.load(Ordering::Relaxed),
        std::time::Duration::from_nanos(SYNC_NANOS.load(Ordering::Relaxed)),
    )
}

/// A stream on one device.
pub struct Stream {
    raw: wwr_stream,
    physical: i32,
}

// SAFETY: a stream handle may be used from any thread; the step graph and
// the GPU's lock order its users.
unsafe impl Send for Stream {}
// SAFETY: as above.
unsafe impl Sync for Stream {}

impl Stream {
    /// A stream on `physical`.
    ///
    /// # Errors
    /// The runtime's failure.
    pub fn new(physical: i32) -> Result<Stream> {
        let _current = ScopedDevice::new(physical);
        let mut raw: wwr_stream = std::ptr::null_mut();
        // SAFETY: writes one handle through a valid local, with the device
        // current.
        runtime_status(unsafe { wwr_sys::wwr_stream_create(&mut raw) })?;
        Ok(Stream { raw, physical })
    }

    /// The handle, for an extern call.
    pub fn raw(&self) -> wwr_stream {
        self.raw
    }

    /// The device the stream is on.
    pub fn physical(&self) -> i32 {
        self.physical
    }

    /// Block until everything enqueued has run.
    ///
    /// # Errors
    /// A fault of the enqueued work, surfaced here.
    pub fn sync(&self) -> Status {
        let _current = ScopedDevice::new(self.physical);
        let start = std::time::Instant::now();
        // SAFETY: a live stream.
        let status = runtime_status(unsafe { wwr_sys::wwr_stream_synchronize(self.raw) });
        SYNCS.fetch_add(1, Ordering::Relaxed);
        SYNC_NANOS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        status
    }

    /// Make everything enqueued after this wait for `event`.
    ///
    /// # Errors
    /// The runtime's failure.
    pub fn wait(&self, event: &Event) -> Status {
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: a live stream and a live event of one device.
        runtime_status(unsafe { wwr_sys::wwr_stream_wait_event(self.raw, event.raw, 0) })
    }

    /// Enqueue the zeroing of `dst`.
    ///
    /// # Errors
    /// The enqueue's failure.
    pub fn zero<T>(&self, dst: &mut impl DeviceSliceMut<T>) -> Status {
        self.fill(dst, 0)
    }

    /// Enqueue the setting of every byte of `dst` to `byte`.
    ///
    /// # Errors
    /// The enqueue's failure.
    pub fn fill<T>(&self, dst: &mut impl DeviceSliceMut<T>, byte: u8) -> Status {
        if dst.is_empty() {
            return Ok(());
        }
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: `dst` is its length of device memory, its holder's alone.
        runtime_status(unsafe {
            wwr_sys::wwr_memset_async(
                dst.as_mut_ptr().cast(),
                i32::from(byte),
                dst.len() * size_of::<T>(),
                self.raw,
            )
        })
    }

    fn copy(
        &self,
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: wwr_memcpy_kind,
    ) -> Status {
        if bytes == 0 {
            return Ok(());
        }
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: the callers below checked both ends hold `bytes`.
        runtime_status(unsafe { wwr_sys::wwr_memcpy_async(dst, src, bytes, kind, self.raw) })
    }

    /// Enqueue the copy of host `src` into the head of device `dst`. `src`
    /// is pinned: it must not be rewritten before the copy has run.
    ///
    /// # Errors
    /// The enqueue's failure.
    #[track_caller]
    pub fn upload<T: Copy>(&self, dst: &mut impl DeviceSliceMut<T>, src: &[T]) -> Status {
        assert!(dst.len() >= src.len(), "upload: the device end is shorter");
        self.copy(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            size_of_val(src),
            wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
        )
    }

    /// Enqueue the copy of the head of device `src` into host `dst`, read
    /// only after the copy has run.
    ///
    /// # Errors
    /// The enqueue's failure.
    #[track_caller]
    pub fn download<T: Copy>(&self, dst: &mut [T], src: &impl DeviceSlice<T>) -> Status {
        assert!(
            src.len() >= dst.len(),
            "download: the device end is shorter"
        );
        self.copy(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            size_of_val(dst),
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
        )
    }

    /// Enqueue the copy of device `src` into the head of device `dst`.
    ///
    /// # Errors
    /// The enqueue's failure.
    #[track_caller]
    pub fn copy_device<T: Copy>(
        &self,
        dst: &mut impl DeviceSliceMut<T>,
        src: &impl DeviceSlice<T>,
    ) -> Status {
        assert!(
            dst.len() >= src.len(),
            "copy_device: the destination is shorter"
        );
        self.copy(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            src.len() * size_of::<T>(),
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE,
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: created by wwr_stream_create on this device, destroyed
        // once.
        unsafe {
            let mut before = 0;
            let _ = wwr_sys::wwr_get_device(&mut before);
            let _ = wwr_sys::wwr_set_device(self.physical);
            let _ = wwr_sys::wwr_stream_destroy(self.raw);
            let _ = wwr_sys::wwr_set_device(before);
        }
    }
}

/// An event: a point on a stream another stream, or the host, waits for.
pub struct Event {
    raw: wwr_event,
    physical: i32,
}

// SAFETY: an event handle may be used from any thread.
unsafe impl Send for Event {}
// SAFETY: as above.
unsafe impl Sync for Event {}

impl Event {
    /// An event on `physical`.
    ///
    /// # Errors
    /// The runtime's failure.
    pub fn new(physical: i32) -> Result<Event> {
        let _current = ScopedDevice::new(physical);
        let mut raw: wwr_event = std::ptr::null_mut();
        // SAFETY: writes one handle through a valid local.
        runtime_status(unsafe { wwr_sys::wwr_event_create(&mut raw) })?;
        Ok(Event { raw, physical })
    }

    /// Mark the end of what `stream` holds now.
    ///
    /// # Errors
    /// The runtime's failure.
    pub fn record(&self, stream: &Stream) -> Status {
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: a live event and a live stream of one device.
        runtime_status(unsafe { wwr_sys::wwr_event_record(self.raw, stream.raw) })
    }

    /// Block the host until the recorded point has run.
    ///
    /// # Errors
    /// A fault of the work before it.
    pub fn sync(&self) -> Status {
        let _current = ScopedDevice::new(self.physical);
        // SAFETY: a live event.
        runtime_status(unsafe { wwr_sys::wwr_event_synchronize(self.raw) })
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: created by wwr_event_create, destroyed once.
        unsafe {
            let mut before = 0;
            let _ = wwr_sys::wwr_get_device(&mut before);
            let _ = wwr_sys::wwr_set_device(self.physical);
            let _ = wwr_sys::wwr_event_destroy(self.raw);
            let _ = wwr_sys::wwr_set_device(before);
        }
    }
}

/// The elements a column-major `rows × cols` matrix at leading dimension
/// `ld` touches.
fn extent(rows: i32, cols: i32, ld: i32) -> usize {
    if rows <= 0 || cols <= 0 {
        return 0;
    }
    ld as usize * (cols as usize - 1) + rows as usize
}

/// The elements a vector of `n` at stride `inc` touches.
fn vector(n: i32, inc: i32) -> usize {
    if n <= 0 {
        return 0;
    }
    inc as usize * (n as usize - 1) + 1
}

#[track_caller]
fn holds<T>(what: &str, name: &str, slice: &impl DeviceSlice<T>, need: usize) {
    assert!(
        slice.len() >= need,
        "{what}: {name} holds {} elements, short of the {need} the call touches",
        slice.len()
    );
}

/// One GPU's library resources: its stream, and the BLAS and dense-solver
/// handles made on that stream. On the MainBus inside a `Mutex`; a step
/// reaches them only through the guard the wrapper took, for its whole
/// `run`, so no handle is ever shared with work in flight: the previous
/// holder synchronized the stream before it released the lock.
pub struct GpuResources {
    gpu: usize,
    physical: i32,
    stream: Stream,
    blas: wwr_blas_handle,
    solver: wwr_solver_handle,
    workspace_bound: bool,
}

// SAFETY: the handles are used by the holder of the GPU's lock alone.
unsafe impl Send for GpuResources {}

const EIG_VECTORS: wwr_solver_eig_mode = wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR;
const EIG_LOWER: wwr_solver_fill_mode = wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER;
const LOWER: wwr_blas_fill_mode = wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;

impl GpuResources {
    /// GPU `gpu` of the run, on device `physical`: its stream, and a BLAS
    /// and a solver handle bound to it for the life of the process.
    ///
    /// # Errors
    /// The runtime's or a library's failure.
    pub fn new(gpu: usize, physical: i32) -> Result<GpuResources> {
        let stream = Stream::new(physical)?;
        let _current = ScopedDevice::new(physical);
        let mut blas: wwr_blas_handle = std::ptr::null_mut();
        let mut solver: wwr_solver_handle = std::ptr::null_mut();
        // SAFETY: each create writes one handle through a valid local with
        // the device current; each handle is then given the stream once.
        unsafe {
            blas_status("blas create", wwr_sys::wwr_blas_create(&mut blas))?;
            blas_status(
                "blas set_stream",
                wwr_sys::wwr_blas_set_stream(blas, stream.raw()),
            )?;
            solver_status("solver create", wwr_sys::wwr_solver_dn_create(&mut solver))?;
            solver_status(
                "solver set_stream",
                wwr_sys::wwr_solver_dn_set_stream(solver, stream.raw()),
            )?;
        }
        Ok(GpuResources {
            gpu,
            physical,
            stream,
            blas,
            solver,
            workspace_bound: false,
        })
    }

    /// The GPU's index in the run.
    pub fn gpu(&self) -> usize {
        self.gpu
    }

    /// The device it is on.
    pub fn physical(&self) -> i32 {
        self.physical
    }

    /// Make the GPU's device current on this thread.
    pub fn current(&self) -> ScopedDevice {
        ScopedDevice::new(self.physical)
    }

    /// The GPU's stream: library calls go there, and the holder's kernels
    /// and copies when it wants one order with no event.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Bind the handle workspace, once: the wrapper calls it at each
    /// acquisition, and only the first does anything.
    ///
    /// # Errors
    /// The library's refusal.
    pub(crate) fn bind_workspace(&mut self, workspace: *mut u8, bytes: usize) -> Status {
        if self.workspace_bound {
            return Ok(());
        }
        let _current = self.current();
        // SAFETY: the workspace is `bytes` of this GPU's device memory the
        // ledger keeps for the run; the caller holds the GPU's lock and no
        // call is in flight.
        blas_status("blas set_workspace", unsafe {
            wwr_sys::wwr_blas_set_workspace(self.blas, workspace.cast(), bytes)
        })?;
        self.workspace_bound = true;
        Ok(())
    }

    /// `C = alpha op(A) op(B) + beta C`.
    ///
    /// # Errors
    /// The library's failure.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn gemm(
        &mut self,
        ta: wwr_blas_operation,
        tb: wwr_blas_operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        b: &impl DeviceSlice<f64>,
        ldb: i32,
        beta: f64,
        c: &mut impl DeviceSliceMut<f64>,
        ldc: i32,
    ) -> Status {
        let (ra, ca) = if ta == OP_N { (m, k) } else { (k, m) };
        let (rb, cb) = if tb == OP_N { (k, n) } else { (n, k) };
        holds("gemm", "a", a, extent(ra, ca, lda));
        holds("gemm", "b", b, extent(rb, cb, ldb));
        holds("gemm", "c", c, extent(m, n, ldc));
        // SAFETY: checked above.
        unsafe {
            self.gemm_raw(
                ta,
                tb,
                m,
                n,
                k,
                alpha,
                a.as_ptr(),
                lda,
                b.as_ptr(),
                ldb,
                beta,
                c.as_mut_ptr(),
                ldc,
            )
        }
    }

    /// [`Self::gemm`] over pointers: the recipes' rows of one matrix.
    ///
    /// # Safety
    /// Each pointer holds the matrix the dimensions say, and `c` meets
    /// neither `a` nor `b`.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    unsafe fn gemm_raw(
        &mut self,
        ta: wwr_blas_operation,
        tb: wwr_blas_operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    ) -> Status {
        if m == 0 || n == 0 {
            return Ok(());
        }
        let _current = self.current();
        // SAFETY: the caller's.
        blas_status("gemm", unsafe {
            wwr_sys::wwr_blas_dgemm(
                self.blas, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc,
            )
        })
    }

    /// `batch` gemms at fixed strides: `C_i = alpha op(A_i) op(B_i) + beta
    /// C_i`.
    ///
    /// # Errors
    /// The library's failure.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn gemm_strided_batched(
        &mut self,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        stride_a: i64,
        b: &impl DeviceSlice<f64>,
        ldb: i32,
        stride_b: i64,
        beta: f64,
        c: &mut impl DeviceSliceMut<f64>,
        ldc: i32,
        stride_c: i64,
        batch: i32,
    ) -> Status {
        if batch == 0 {
            return Ok(());
        }
        let last = (batch - 1) as usize;
        holds(
            "gemm_strided_batched",
            "a",
            a,
            last * stride_a as usize + extent(m, k, lda),
        );
        holds(
            "gemm_strided_batched",
            "b",
            b,
            last * stride_b as usize + extent(k, n, ldb),
        );
        holds(
            "gemm_strided_batched",
            "c",
            c,
            last * stride_c as usize + extent(m, n, ldc),
        );
        let _current = self.current();
        // SAFETY: checked above.
        blas_status("gemm_strided_batched", unsafe {
            wwr_sys::wwr_blas_dgemm_strided_batched(
                self.blas,
                OP_N,
                OP_N,
                m,
                n,
                k,
                &alpha,
                a.as_ptr(),
                lda,
                stride_a,
                b.as_ptr(),
                ldb,
                stride_b,
                &beta,
                c.as_mut_ptr(),
                ldc,
                stride_c,
                batch,
            )
        })
    }

    /// `y = alpha op(A) x + beta y`.
    ///
    /// # Errors
    /// The library's failure.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn gemv(
        &mut self,
        t: wwr_blas_operation,
        m: i32,
        n: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        x: &impl DeviceSlice<f64>,
        beta: f64,
        y: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        let (nx, ny) = if t == OP_N { (n, m) } else { (m, n) };
        holds("gemv", "a", a, extent(m, n, lda));
        holds("gemv", "x", x, vector(nx, 1));
        holds("gemv", "y", y, vector(ny, 1));
        let _current = self.current();
        // SAFETY: checked above.
        blas_status("gemv", unsafe {
            wwr_sys::wwr_blas_dgemv(
                self.blas,
                t,
                m,
                n,
                &alpha,
                a.as_ptr(),
                lda,
                x.as_ptr(),
                1,
                &beta,
                y.as_mut_ptr(),
                1,
            )
        })
    }

    /// `C = alpha op(A) + beta op(B)`, all at leading dimension `m`.
    ///
    /// # Errors
    /// The library's failure.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn geam(
        &mut self,
        ta: wwr_blas_operation,
        tb: wwr_blas_operation,
        m: i32,
        n: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        beta: f64,
        b: &impl DeviceSlice<f64>,
        c: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        // A transposed operand is n × m at leading dimension n.
        let lda = if ta == OP_N { m } else { n };
        let ldb = if tb == OP_N { m } else { n };
        holds("geam", "a", a, (m * n) as usize);
        holds("geam", "b", b, (m * n) as usize);
        holds("geam", "c", c, (m * n) as usize);
        let _current = self.current();
        // SAFETY: checked above.
        blas_status("geam", unsafe {
            wwr_sys::wwr_blas_dgeam(
                self.blas,
                ta,
                tb,
                m,
                n,
                &alpha,
                a.as_ptr(),
                lda,
                &beta,
                b.as_ptr(),
                ldb,
                c.as_mut_ptr(),
                m,
            )
        })
    }

    /// `C = A diag(x)`, `n × n`.
    ///
    /// # Errors
    /// The library's failure.
    #[track_caller]
    pub fn scale_columns(
        &mut self,
        n: i32,
        a: &impl DeviceSlice<f64>,
        x: &impl DeviceSlice<f64>,
        c: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        holds("dgmm", "a", a, (n * n) as usize);
        holds("dgmm", "x", x, n as usize);
        holds("dgmm", "c", c, (n * n) as usize);
        let _current = self.current();
        // SAFETY: checked above.
        blas_status("dgmm", unsafe {
            wwr_sys::wwr_blas_ddgmm(
                self.blas,
                wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT,
                n,
                n,
                a.as_ptr(),
                n,
                x.as_ptr(),
                1,
                c.as_mut_ptr(),
                n,
            )
        })
    }

    /// syevd's work buffer at order `n`, in doubles: what a step declares
    /// as scratch for each eigensolve it makes.
    ///
    /// # Errors
    /// The query's failure.
    pub fn syevd_lwork(&self, n: usize) -> Result<usize> {
        if n == 0 {
            return Ok(0);
        }
        let _current = self.current();
        let mut lwork = 0;
        // SAFETY: a size query: it reads the dimensions and no array.
        solver_status("syevd_buffer_size", unsafe {
            wwr_sys::wwr_solver_dn_dsyevd_buffer_size(
                self.solver,
                EIG_VECTORS,
                EIG_LOWER,
                n as i32,
                std::ptr::null(),
                n as i32,
                std::ptr::null(),
                &mut lwork,
            )
        })?;
        // A work buffer is never null: a query of 0 still gets a slice.
        Ok((lwork as usize).max(1))
    }

    /// `A = V diag(w) Vᵀ` in place, the eigenvalues ascending, on the GPU's
    /// stream. `work` is a slice of the caller's scratch, declared from
    /// [`Self::syevd_lwork`]; `info` is the solver's flag, on the device.
    ///
    /// # Errors
    /// The query's or the solver's failure.
    ///
    /// # Panics
    /// If `work` is shorter than the size query at `n`, before anything is
    /// enqueued: the solver does not check it, and writes past a short one.
    #[track_caller]
    pub fn syevd(
        &mut self,
        n: usize,
        a: &mut impl DeviceSliceMut<f64>,
        w: &mut impl DeviceSliceMut<f64>,
        work: &mut impl DeviceSliceMut<f64>,
        info: &mut impl DeviceSliceMut<i32>,
    ) -> Status {
        let need = self.syevd_lwork(n)?;
        assert!(
            work.len() >= need,
            "syevd at n = {n}: the work buffer holds {} doubles, short of the size query's \
             {need}: the call is missing from the step's mem_budget",
            work.len()
        );
        holds("syevd", "a", a, n * n);
        holds("syevd", "w", w, n);
        holds("syevd", "info", info, 1);
        let _current = self.current();
        // SAFETY: checked above; `work` is at least the query.
        solver_status("syevd", unsafe {
            wwr_sys::wwr_solver_dn_dsyevd(
                self.solver,
                EIG_VECTORS,
                EIG_LOWER,
                n as i32,
                a.as_mut_ptr(),
                n as i32,
                w.as_mut_ptr(),
                work.as_mut_ptr(),
                work.len().min(i32::MAX as usize) as i32,
                info.as_mut_ptr(),
            )
        })
    }

    /// `A ← (A + Aᵀ)/2`, `n × n`, on the GPU's stream.
    ///
    /// # Errors
    /// The launch's failure.
    #[track_caller]
    pub fn symmetrize(&mut self, n: usize, a: &mut impl DeviceSliceMut<f64>) -> Status {
        holds("fb_symmetrize", "a", a, n * n);
        let _current = self.current();
        // SAFETY: checked above.
        let status = unsafe {
            fb_recipes_sys::fb_symmetrize(self.stream.raw(), n as i32, a.as_mut_ptr(), n as i32)
        };
        launched("fb_symmetrize", status)
    }

    /// Each of `data` replaced by `1/sqrt(x)`, or 0 at or below `floor`.
    ///
    /// # Errors
    /// The launch's failure.
    #[track_caller]
    pub fn inverse_sqrt(&mut self, data: &mut impl DeviceSliceMut<f64>, floor: f64) -> Status {
        let _current = self.current();
        // SAFETY: `data` is its length of device memory.
        let status = unsafe {
            fb_recipes_sys::fb_inverse_sqrt(
                self.stream.raw(),
                data.len() as u32,
                data.as_mut_ptr(),
                floor,
                std::ptr::null(),
            )
        };
        launched("fb_inverse_sqrt", status)
    }

    /// The DIIS coefficients of the `m` held residuals whose Gram is
    /// `gram` at leading dimension `ld`.
    ///
    /// # Errors
    /// The launch's failure.
    #[track_caller]
    pub fn diis_solve(
        &mut self,
        m: usize,
        ld: usize,
        newest: usize,
        gram: &impl DeviceSlice<f64>,
        coeff: &mut impl DeviceSliceMut<f64>,
        singular: &mut impl DeviceSliceMut<i32>,
    ) -> Status {
        holds("fb_diis_solve", "gram", gram, ld * (m - 1) + m);
        holds("fb_diis_solve", "coeff", coeff, m);
        holds("fb_diis_solve", "singular", singular, 1);
        let _current = self.current();
        // SAFETY: checked above.
        let status = unsafe {
            fb_recipes_sys::fb_diis_solve(
                self.stream.raw(),
                m as i32,
                ld as i32,
                newest as i32,
                gram.as_ptr(),
                coeff.as_mut_ptr(),
                singular.as_mut_ptr(),
            )
        };
        launched("fb_diis_solve", status)
    }

    /// The tree's blocked triangular solve, in place of a vendor `trsm`
    /// (`src/recipes/trsm`): `op(A) X = alpha B` on the left or `X op(A) =
    /// alpha B` on the right, `X` over `B`, `A` lower triangular. Each
    /// diagonal block by a kernel, the rest by one gemm, all on the GPU's
    /// stream and needing only the handle workspace.
    ///
    /// # Errors
    /// A launch's or the library's failure.
    #[allow(clippy::too_many_arguments, reason = "the BLAS signature")]
    #[track_caller]
    pub fn trsm_lower(
        &mut self,
        left: bool,
        trans: bool,
        m: i32,
        n: i32,
        alpha: f64,
        a: &impl DeviceSlice<f64>,
        lda: i32,
        b: &mut impl DeviceSliceMut<f64>,
        ldb: i32,
    ) -> Status {
        if m == 0 || n == 0 {
            return Ok(());
        }
        let k = if left { m } else { n };
        holds("trsm", "a", a, extent(k, k, lda));
        holds("trsm", "b", b, extent(m, n, ldb));
        // M is op(A) on the left and its transpose on the right; it is
        // lower for (left, N) and (right, T).
        let lower = left != trans;
        let a_trans = if left { trans } else { !trans };
        let a_op = if trans { OP_T } else { OP_N };
        let (block, order, la, lb) = (
            FB_TRSM_TILE as usize,
            k as usize,
            lda as usize,
            ldb as usize,
        );
        let ncols = if left { n } else { m };
        let blocks = order.div_ceil(block);
        let (pa, pb) = (a.as_ptr(), b.as_mut_ptr());
        for step in 0..blocks {
            let k0 = block * if lower { step } else { blocks - 1 - step };
            let jb = block.min(order - k0);
            let first = step == 0;
            let scale = if first { alpha } else { 1.0 };
            {
                let _current = self.current();
                // SAFETY: the triangle at (k0, k0) and the block's rows (on
                // the left) or columns (on the right) of B are inside the
                // extents checked above.
                let status = unsafe {
                    fb_recipes_sys::fb_trsm_diagonal_f64(
                        self.stream.raw(),
                        jb as i32,
                        ncols,
                        scale,
                        pa.wrapping_add(k0 + k0 * la),
                        lda,
                        a_trans,
                        !lower,
                        false,
                        pb.wrapping_add(if left { k0 } else { k0 * lb }),
                        ldb,
                        !left,
                    )
                };
                launched("fb_trsm_diagonal_f64", status)?;
            }
            let r0 = if lower { k0 + jb } else { 0 };
            let rows = if lower { order - r0 } else { k0 };
            if rows == 0 {
                continue;
            }
            let off = pa.wrapping_add(if left != trans {
                r0 + k0 * la
            } else {
                k0 + r0 * la
            });
            let beta = if first { alpha } else { 1.0 };
            let (jb, rows) = (jb as i32, rows as i32);
            // SAFETY: the solved block and the rest are disjoint rows (on
            // the left) or columns (on the right) of B, inside its extent;
            // `off` is inside A's.
            unsafe {
                if left {
                    // B(rest, :) = beta B(rest, :) - op(A)(rest, block) B(block, :)
                    self.gemm_raw(
                        a_op,
                        OP_N,
                        rows,
                        n,
                        jb,
                        -1.0,
                        off,
                        lda,
                        pb.wrapping_add(k0),
                        ldb,
                        beta,
                        pb.wrapping_add(r0),
                        ldb,
                    )?;
                } else {
                    // B(:, rest) = beta B(:, rest) - B(:, block) op(A)(block, rest)
                    self.gemm_raw(
                        OP_N,
                        a_op,
                        m,
                        rows,
                        jb,
                        -1.0,
                        pb.wrapping_add(k0 * lb),
                        ldb,
                        off,
                        lda,
                        beta,
                        pb.wrapping_add(r0 * lb),
                        ldb,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// The tree's blocked right-looking Cholesky (`src/recipes/cholesky`):
    /// `A = L Lᵀ` in the lower triangle of the `n × n` `a`, in place, on
    /// the GPU's stream. The diagonal tile by a kernel, the panel by
    /// [`Self::trsm_lower`], the trailing matrix by a syrk. `tile` is
    /// [`CHOLESKY_TILE`]² doubles of scratch; `flag` is zeroed here, and
    /// reads the 1-based leading minor that is not positive, or 0.
    ///
    /// # Errors
    /// A launch's or the library's failure.
    #[track_caller]
    pub fn cholesky(
        &mut self,
        n: usize,
        a: &mut impl DeviceSliceMut<f64>,
        tile: &mut impl DeviceSliceMut<f64>,
        flag: &mut impl DeviceSliceMut<i32>,
    ) -> Status {
        holds("cholesky", "a", a, n * n);
        holds("cholesky", "tile", tile, CHOLESKY_TILE * CHOLESKY_TILE);
        holds("cholesky", "flag", flag, 1);
        self.stream.zero(flag)?;
        let (pa, ni) = (a.as_mut_ptr(), n as i32);
        for j in (0..n).step_by(CHOLESKY_TILE) {
            let jb = CHOLESKY_TILE.min(n - j);
            {
                let _current = self.current();
                // SAFETY: the tile at (j, j) is inside the n × n matrix;
                // `tile` holds a full tile and `flag` one int.
                let status = unsafe {
                    fb_recipes_sys::fb_cholesky_diagonal(
                        self.stream.raw(),
                        jb as i32,
                        j as i32,
                        pa.wrapping_add(j * (n + 1)),
                        ni,
                        tile.as_mut_ptr(),
                        flag.as_mut_ptr(),
                    )
                };
                launched("fb_cholesky_diagonal", status)?;
            }
            let rest = n - j - jb;
            if rest == 0 {
                break;
            }
            let (jbi, resti) = (jb as i32, rest as i32);
            // The panel below the tile: X L11ᵀ = A21.
            // SAFETY: the panel is rows [j + jb, n) of columns [j, j + jb),
            // inside the matrix, and no other borrow of `a` is live.
            let mut panel = unsafe {
                PartMut::from_raw(pa.wrapping_add(j * (n + 1) + jb), extent(resti, jbi, ni))
            };
            let l11 = part(tile, 0, jb * jb);
            self.trsm_lower(false, true, resti, jbi, 1.0, &l11, jbi, &mut panel, ni)?;
            let _current = self.current();
            // The trailing matrix: A22 -= L21 L21ᵀ, lower triangle.
            // SAFETY: the panel and the trailing block are disjoint parts
            // of the matrix.
            blas_status("syrk", unsafe {
                wwr_sys::wwr_blas_dsyrk(
                    self.blas,
                    LOWER,
                    OP_N,
                    resti,
                    jbi,
                    &-1.0,
                    panel.as_ptr(),
                    ni,
                    &1.0,
                    pa.wrapping_add((j + jb) * (n + 1)),
                    ni,
                )
            })?;
        }
        Ok(())
    }
}

/// The order of the Cholesky recipe's tile.
pub const CHOLESKY_TILE: usize = FB_CHOLESKY_TILE as usize;

impl Drop for GpuResources {
    fn drop(&mut self) {
        let _current = self.current();
        // SAFETY: each handle was created here and is destroyed once; the
        // stream drops after them.
        unsafe {
            let _ = wwr_sys::wwr_solver_dn_destroy(self.solver);
            let _ = wwr_sys::wwr_blas_destroy(self.blas);
        }
    }
}

/// The kernel launches a step puts on a stream: one function per kernel
/// family, over slices, each holding its slices against the kernel's own
/// formula before anything is enqueued.
pub mod launch {
    use fb_core_error_handling::Status;
    use fb_integrals_sys::{
        fb_primitive_pair_class_view, fb_primitive_pair_records, fb_primitive_pair_source,
    };

    use super::{DeviceSlice, DeviceSliceMut, ScopedDevice, Stream, holds, launched};

    /// The f64 fields `fb_mmd_records` writes per record: zeta, P, PA, PB,
    /// s, ea, eb.
    pub const RECORD_F64_FIELDS: usize = 13;
    /// The i32 fields: off_a, off_b, rank_a, rank_b.
    pub const RECORD_I32_FIELDS: usize = 4;
    /// The fields of an aux record: x, y, z, α, `c (π/α)^{3/2}`.
    pub const AUX_FIELDS: usize = 5;
    /// The fields of a three-centre cell.
    pub const CELL_FIELDS: usize = 6;

    #[link(name = "fbx_ssperi3_kernels")]
    unsafe extern "C" {
        /// `df_batched/single_gpu_spherical/kernels/ssperi3_launch.cu`:
        /// class `(cls.la, cls.lb, lp)` of the fully spherical packed
        /// `(μν|Q)` into `eri3`, enqueued on `stream`.
        fn fbx_ssperi3_packed_class(
            stream: *mut std::ffi::c_void,
            cls: *const fb_primitive_pair_class_view,
            lp: i32,
            n_aux_rec: i32,
            aux: *const f64,
            n_cell: i32,
            cells: *const i32,
            n_col: i32,
            n_aux: i32,
            eri3: *mut f64,
        ) -> fb_kernel_abi::LaunchStatus;
    }

    /// The shell tables of one class's pairs, on the device: what
    /// [`mmd_records`] forms the class's records from.
    pub struct PairSource<'a, F, I, R, U> {
        /// The class's pairs.
        pub n_pairs: usize,
        /// The basis's shells and primitives.
        pub n_shell: usize,
        /// The primitives.
        pub n_prim: usize,
        /// exponent, coefficient (one per primitive), c_x, c_y, c_z (one
        /// per shell).
        pub f: &'a F,
        /// prim_begin (`n_shell + 1`), ao_offset, rank (`n_shell` each).
        pub i: &'a I,
        /// The class's rec_begin, `n_pairs + 1`.
        pub rec_begin: &'a R,
        /// The class's bra shells, then its ket shells.
        pub pairs: &'a U,
    }

    /// One class's records, `n_rec` per field: [`RECORD_F64_FIELDS`]
    /// segments in `f` and [`RECORD_I32_FIELDS`] in `i`, formed on `stream`
    /// from the shell tables.
    ///
    /// # Errors
    /// The launch's failure.
    #[track_caller]
    pub fn mmd_records(
        stream: &Stream,
        what: &str,
        n_rec: usize,
        src: &PairSource<
            '_,
            impl DeviceSlice<f64>,
            impl DeviceSlice<i32>,
            impl DeviceSlice<i32>,
            impl DeviceSlice<u32>,
        >,
        f: &mut impl DeviceSliceMut<f64>,
        i: &mut impl DeviceSliceMut<i32>,
    ) -> Status {
        let (n_shell, n_prim) = (src.n_shell, src.n_prim);
        holds(
            what,
            "the shell tables' doubles",
            src.f,
            2 * n_prim + 3 * n_shell,
        );
        holds(what, "the shell tables' ints", src.i, 3 * n_shell + 1);
        holds(what, "rec_begin", src.rec_begin, src.n_pairs + 1);
        holds(what, "the pairs", src.pairs, 2 * src.n_pairs);
        holds(what, "the records' doubles", f, RECORD_F64_FIELDS * n_rec);
        holds(what, "the records' ints", i, RECORD_I32_FIELDS * n_rec);
        let (sf, si, su) = (src.f.as_ptr(), src.i.as_ptr(), src.pairs.as_ptr());
        let source = fb_primitive_pair_source {
            n_pairs: src.n_pairs as i32,
            bra: su,
            ket: su.wrapping_add(src.n_pairs),
            rec_begin: src.rec_begin.as_ptr(),
            prim_begin: si,
            exponent: sf,
            coefficient: sf.wrapping_add(n_prim),
            c_x: sf.wrapping_add(2 * n_prim),
            c_y: sf.wrapping_add(2 * n_prim + n_shell),
            c_z: sf.wrapping_add(2 * n_prim + 2 * n_shell),
            ao_offset: si.wrapping_add(n_shell + 1),
            rank: si.wrapping_add(2 * n_shell + 1),
        };
        let (pf, pi) = (f.as_mut_ptr(), i.as_mut_ptr());
        let g = |k: usize| pf.wrapping_add(k * n_rec);
        let h = |k: usize| pi.wrapping_add(k * n_rec);
        let records = fb_primitive_pair_records {
            zeta: g(0),
            p_x: g(1),
            p_y: g(2),
            p_z: g(3),
            pa_x: g(4),
            pa_y: g(5),
            pa_z: g(6),
            pb_x: g(7),
            pb_y: g(8),
            pb_z: g(9),
            s: g(10),
            ea: g(11),
            eb: g(12),
            off_a: h(0),
            off_b: h(1),
            rank_a: h(2),
            rank_b: h(3),
        };
        let _current = ScopedDevice::new(stream.physical());
        // SAFETY: every source array and every output field was held
        // against its length above; the two structs are host values the
        // call reads before it returns.
        let status = unsafe {
            fb_integrals_sys::fb_mmd_records(stream.raw(), n_rec as i32, &source, &records)
        };
        launched(what, status)
    }

    /// What a class kernel reads of one class's records: the first eleven
    /// f64 fields.
    pub fn class_view(
        la: i32,
        lb: i32,
        n_rec: usize,
        f: &impl DeviceSlice<f64>,
    ) -> fb_primitive_pair_class_view {
        assert!(f.len() >= RECORD_F64_FIELDS * n_rec, "the class's records");
        let g = |k: usize| f.as_ptr().wrapping_add(k * n_rec);
        let null = std::ptr::null();
        fb_primitive_pair_class_view {
            la,
            lb,
            n_rec: n_rec as i32,
            zeta: g(0),
            p_x: g(1),
            p_y: g(2),
            p_z: g(3),
            pa_x: g(4),
            pa_y: g(5),
            pa_z: g(6),
            pb_x: g(7),
            pb_y: g(8),
            pb_z: g(9),
            s: g(10),
            ea: g(11),
            eb: g(12),
            bound: null,
            bound_ab: null,
            bound_ba: null,
            off_a: std::ptr::null(),
            off_b: std::ptr::null(),
            rank_a: std::ptr::null(),
            rank_b: std::ptr::null(),
        }
    }

    /// One aux momentum's cells of a pair batch through `ssperi3<la, lb,
    /// lp>`: `(μν|P)` with all three indices spherical, each cell's block
    /// at its destination in the `n_aux × n_col` batch `eri3`, aux fastest.
    /// `cells` is field-major, [`CELL_FIELDS`] × `n_cell`; the cells'
    /// destinations are the caller's, built from the same batch.
    ///
    /// # Errors
    /// The launch's failure.
    #[allow(clippy::too_many_arguments, reason = "the kernel's arguments")]
    #[track_caller]
    pub fn ssperi3(
        stream: &Stream,
        view: &fb_primitive_pair_class_view,
        lp: i32,
        n_aux_rec: usize,
        aux: &impl DeviceSlice<f64>,
        n_cell: usize,
        cells: &impl DeviceSlice<i32>,
        n_col: usize,
        n_aux: usize,
        eri3: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        let what = format!("ssperi3 ({}, {}, {lp})", view.la, view.lb);
        holds(&what, "the aux records", aux, AUX_FIELDS * n_aux_rec);
        holds(&what, "the cells", cells, CELL_FIELDS * n_cell);
        holds(&what, "the batch", eri3, n_col * n_aux);
        let _current = ScopedDevice::new(stream.physical());
        // SAFETY: held against the kernel's lengths above; the view's
        // fields are a class's records, live while the caller's borrow is.
        let status = unsafe {
            fbx_ssperi3_packed_class(
                stream.raw(),
                view,
                lp,
                n_aux_rec as i32,
                aux.as_ptr(),
                n_cell as i32,
                cells.as_ptr(),
                n_col as i32,
                n_aux as i32,
                eri3.as_mut_ptr(),
            )
        };
        launched(&what, status)
    }

    /// `rows` packed rows of B, `n_col` pair columns each, unpacked into
    /// their symmetric `n_ao × n_ao` squares.
    ///
    /// # Errors
    /// The launch's failure.
    #[allow(clippy::too_many_arguments, reason = "the kernel's arguments")]
    #[track_caller]
    pub fn packed_gather(
        stream: &Stream,
        n_ao: usize,
        n_col: usize,
        rows: usize,
        row_of: &impl DeviceSlice<i32>,
        col_of: &impl DeviceSlice<i32>,
        weights: &impl DeviceSlice<f64>,
        packed: &impl DeviceSlice<f64>,
        squares: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        let what = "fb_packed_gather";
        holds(what, "the columns' rows", row_of, n_col);
        holds(what, "the columns' cols", col_of, n_col);
        holds(what, "the weights", weights, n_col);
        holds(what, "the packed rows", packed, rows * n_col);
        holds(what, "the squares", squares, rows * n_ao * n_ao);
        let _current = ScopedDevice::new(stream.physical());
        // SAFETY: held against the kernel's lengths above.
        let status = unsafe {
            fb_integrals_sys::fb_packed_gather(
                stream.raw(),
                n_ao as i32,
                n_col as i32,
                rows as i32,
                row_of.as_ptr(),
                col_of.as_ptr(),
                weights.as_ptr(),
                packed.as_ptr(),
                squares.as_mut_ptr(),
            )
        };
        launched(what, status)
    }
}
