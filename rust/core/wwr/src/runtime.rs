//! wwr's `src/extension/runtime` (module `wwr.extension.runtime`): owning
//! wrappers for the runtime API's streams, events, memory pools and graphs,
//! built on [`handle`](super::handle).
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:stream`, `StreamWrapper<P_create, P_destroy, P_device>` | [`Stream`] |
//! | `:event`, `EventWrapper` | [`Event`] |
//! | `:mem_pool`, `MemPoolWrapper`, and the `wwrMemPoolProps` it fills | [`MemPool`], [`MemPoolProps`] |
//! | `:graph`, `GraphWrapper` | [`Graph`] |
//! | `:graph_exec`, `GraphExecWrapper` | [`GraphExec`] |
//! | `:device_handle`, `kit::DeviceHandle` | nothing: see below |
//! | `StreamView`, `EventView`, `GraphExecView` | the handle views [`view()`](super::handle::DeviceBoundHandle::view) returns |
//! | `wwrGetDeviceCount` | [`device_count`] |
//! | free_breakfast's `Device::stack_growth` (`src/core/wwr/device.cppm`) | [`stack_growth`], measured rather than computed |
//!
//! Each wrapper is a newtype over the handle it is built on, and derefs to
//! it, so `get()`, `view()` and `dev_idx()` are the handle's. Its
//! `HandleKind` is crate-private: the wrapper is the only way to make one.
//!
//! **A constructor aborts on failure**, logging at critical against its
//! caller: the tree's rule for every create (`docs/diagnostics.md`,
//! "Allocation is a planning-time error"), which the C++ tree spells
//! `CreatePolicy`. Each wrapper builds on one private fallible constructor
//! that returns the `Err` instead; a site that must recover from a failed
//! create is the reason to make that one public, and nothing else changes.
//! A wrapper is therefore always live, and no method has a case for one
//! that holds nothing.
//!
//! **The borrow-safe operations are methods on the owners**, and `unsafe`
//! free functions on the raw handle beside them. wwr's free functions take
//! a raw handle so one definition serves an owner, a view and a bare
//! handle; a destroyed handle is a use-after-free inside the runtime,
//! which only an owner's borrow rules out, so here the raw forms
//! ([`stream::sync`], [`stream::wait_event`], [`stream::begin_capture`],
//! [`event::record`], [`event::record_with_flags`], [`event::sync`],
//! [`graph_exec::launch`], [`graph_exec::upload`]) are `unsafe`, and
//! [`Stream::sync`], [`Event::record`], [`GraphExec::launch`] and the rest
//! call them safely. wwr overloads `sync` on `wwrStream_t` against
//! `wwrEvent_t`; the C ABI erases both to `*mut c_void`, so the raw forms
//! are told apart by module path, and the type system cannot.
//!
//! **Flags pass through untranslated.** The C ABI forwards a flags word to
//! the backend as it is, so the constants here ([`stream::NON_BLOCKING`],
//! [`event::DISABLE_TIMING`], [`event::BLOCKING_SYNC`]) are the vendors'
//! values, which CUDA and HIP share.
//!
//! **`kit::DeviceHandle`**, an owned stream and memory pool on one device,
//! is wwr's ready-made top rung of the device-handle ladder, for a buffer
//! that allocates from a pool. Neither the rung nor the bundle has a
//! counterpart: the arena allocates one plain block per plan
//! (`docs/memory.md`, "The three kinds"), and a [`Stream`] is a
//! [`DeviceHandleStream`](super::handle::DeviceHandleStream) on its own.

pub mod event;
pub mod graph;
pub mod graph_exec;
pub mod mem_pool;
pub mod stream;

pub use event::Event;
pub use graph::Graph;
pub use graph_exec::GraphExec;
pub use mem_pool::{MemPool, MemPoolProps};
pub use stream::Stream;

use fb_core_error_handling::Result;
use wwr_sys::wwr_status;

use crate::common::ScopedDeviceIndex;
use crate::gpu_status_at;

/// Whether the library this crate links was built for HIP: its CMake
/// build's backend, which `build.rs` reads.
const HIP: bool = matches!(env!("FB_WWR_BACKEND").as_bytes(), b"HIP");

/// What the kernels' stack can still take from device `dev_idx`'s memory,
/// in bytes, for a kernel of `per_thread` bytes of stack per thread to run:
/// the C++ `Device::stack_growth` (`../src/core/wwr/README.md`, "Kernel stacks
/// outside the pool"). CUDA holds a kernel's stack for every thread the
/// device can keep resident, from its first launch on, by raising the
/// device's stack limit and keeping it, so this is 0 once the limit is at
/// `per_thread` or past it. It is 0 under HIP, whose scratch is sized per
/// dispatch, and asked of no device there.
///
/// The C++ multiplies the limit's shortfall by the resident threads, read
/// from the device's properties. wwr's C ABI carries no device attribute
/// (its `src/capi/README.md`, `wwrDeviceGetAttribute`), so this measures
/// the product instead: it reads the free memory, raises the limit to
/// `per_thread`, which takes the stack at once, reads it again, and sets
/// the limit back, which returns it (the C++ README's measurements). Two
/// readings of a card every tenant shares, so another's allocation between
/// them moves the result: it is admission's (`docs/memory.md`,
/// "Admission"), never a run's, and is made before any kernel runs.
///
/// # Errors
/// The failed device switch or runtime call, among them raising the limit
/// on a card with no room left for the stack.
#[track_caller]
#[allow(
    clippy::disallowed_methods,
    reason = "admission counts the kernels' stack beside a device block (docs/memory.md, \"Admission\"); nothing is sized from it at run time"
)]
pub fn stack_growth(dev_idx: i32, per_thread: usize) -> Result<usize> {
    if HIP {
        return Ok(0);
    }
    let location = std::panic::Location::caller();
    let _device = ScopedDeviceIndex::try_new(dev_idx)?;
    let stack = wwr_sys::wwr_limit::WWR_LIMIT_STACK_SIZE;
    let mut limit = 0;
    // SAFETY: a plain runtime call writing one local.
    let status = unsafe { wwr_sys::wwr_device_get_limit(&mut limit, stack) };
    gpu_status_at(status, location)?;
    if limit >= per_thread {
        return Ok(0);
    }
    let free = || -> Result<usize> {
        let (mut free, mut total) = (0, 0);
        // SAFETY: a plain runtime call writing two locals.
        let status = unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) };
        gpu_status_at(status, location)?;
        Ok(free)
    };
    let before = free()?;
    // SAFETY: a plain runtime call; no kernel runs at admission.
    let status = unsafe { wwr_sys::wwr_device_set_limit(stack, per_thread) };
    gpu_status_at(status, location)?;
    let after = free();
    // SAFETY: as above, the limit the device had back.
    let status = unsafe { wwr_sys::wwr_device_set_limit(stack, limit) };
    gpu_status_at(status, location)?;
    Ok(before.saturating_sub(after?))
}

/// How many devices the runtime sees from this process, the count the
/// device indices `0..n` run over: `wwr_get_device_count`.
///
/// # Errors
/// The runtime's failure, among them the no-device status a machine with
/// no card, or no driver, returns.
#[track_caller]
pub fn device_count() -> Result<usize> {
    let mut count = 0;
    // SAFETY: a plain runtime call writing one local.
    let status = unsafe { wwr_sys::wwr_get_device_count(&mut count) };
    gpu_status_at(status, std::panic::Location::caller())?;
    // A count the runtime gives is never negative; one would read as none.
    Ok(usize::try_from(count).unwrap_or(0))
}

/// The first failure of a destroy that is two runtime calls, both made:
/// wwr synchronizes a stream or event before destroying it and reports
/// each. A `HandleKind` returns one status, so the sync's failure wins,
/// and the destroy still runs.
fn first_failure(sync: wwr_status, destroy: wwr_status) -> wwr_status {
    if sync == wwr_status::WWR_STATUS_SUCCESS {
        destroy
    } else {
        sync
    }
}

/// Device helpers for this crate's tests and, through the `test-support`
/// feature, for a dependent's.
#[cfg(any(test, feature = "test-support"))]
pub mod tests {
    use std::ffi::c_void;

    use wwr_sys::{wwr_memcpy_kind, wwr_status};

    use super::Stream;
    use crate::handle::DeviceHandleStream;

    use fb_core_error_handling::{Error, ErrorKind};

    /// Assert that `error` is a GPU failure whose status is named `name`,
    /// reported at `line` of `file`.
    ///
    /// # Panics
    /// If it is not: the assertion is the point.
    #[track_caller]
    pub fn assert_gpu_error_at(error: Error, name: &str, file: &str, line: u32) {
        assert_eq!(error.kind, ErrorKind::GpuRuntime);
        assert!(error.message.starts_with(name), "{}", error.message);
        assert_eq!((error.origin.file(), error.origin.line()), (file, line));
    }

    /// Assert that a raw runtime call succeeded.
    ///
    /// # Panics
    /// If `status` is not `WWR_STATUS_SUCCESS`.
    #[track_caller]
    pub fn ok(status: wwr_status) {
        assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
    }

    /// `n` bytes of device memory, freed on drop: a kernel library's test
    /// buffer, for a test with no arena. Its copies run on the test's own
    /// stream and are drained before they return, never the synchronous
    /// `wwr_memcpy`: that runs on the legacy stream, which a non-blocking
    /// [`Stream`] does not wait for, and from pageable memory it returns
    /// before its DMA lands (README.md, "Build and test").
    pub struct DeviceBytes(pub *mut c_void, usize);

    // SAFETY: the host never dereferences the pointer; the bytes are
    // reached only through runtime calls, which are thread-safe, and the
    // allocation is freed once, by the owner's drop.
    unsafe impl Send for DeviceBytes {}
    // SAFETY: as above; `&self` methods only read the bytes.
    unsafe impl Sync for DeviceBytes {}

    impl DeviceBytes {
        /// `n` uninitialized bytes, allocated by the synchronous
        /// `wwr_malloc`.
        ///
        /// # Panics
        /// If the allocation fails.
        #[allow(
            clippy::disallowed_methods,
            reason = "a test's own buffer, outside any plan: allocating is the budget's, and the test checks what is under it"
        )]
        pub fn new(n: usize) -> Self {
            let mut p = std::ptr::null_mut();
            // SAFETY: writes one pointer through a valid local.
            ok(unsafe { wwr_sys::wwr_malloc(&mut p, n) });
            DeviceBytes(p, n)
        }

        /// A new buffer holding `host`'s bytes, copied on `stream` and so
        /// ordered before the work queued on it after.
        pub fn upload<T: Copy>(host: &[T], stream: &Stream) -> Self {
            let device = Self::new(size_of_val(host));
            let kind = wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE;
            device.copy(device.0, host.as_ptr().cast(), kind, stream);
            device
        }

        /// The bytes, copied on `stream` after the work queued on it.
        pub fn read(&self, stream: &Stream) -> Vec<u8> {
            let mut host = vec![0u8; self.1];
            let kind = wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST;
            self.copy(host.as_mut_ptr().cast(), self.0, kind, stream);
            host
        }

        /// The one copy, all `self.1` bytes from `src` to `dst`, one of them
        /// this buffer and the other host memory, on `stream`, drained
        /// before it returns.
        #[allow(
            clippy::disallowed_methods,
            reason = "the test buffer's copy drains before it returns, so a test reads its result: the harness waits, not a task"
        )]
        fn copy(
            &self,
            dst: *mut c_void,
            src: *const c_void,
            kind: wwr_memcpy_kind,
            stream: &Stream,
        ) {
            if self.1 == 0 {
                return;
            }
            // SAFETY: both ranges are `self.1` bytes, the host one a live
            // local or borrow of the caller's that outlives the copy: the
            // stream is drained before this returns. `stream` is live.
            ok(unsafe { wwr_sys::wwr_memcpy_async(dst, src, self.1, kind, stream.stream()) });
            ok(stream.sync());
        }
    }

    impl Drop for DeviceBytes {
        fn drop(&mut self) {
            // SAFETY: allocated by `wwr_malloc`, freed once.
            ok(unsafe { wwr_sys::wwr_free(self.0) });
        }
    }

    // SAFETY: a live `wwr_malloc` allocation of `self.1` bytes on the
    // device current when it was made, 256-byte aligned, so as many whole
    // `T`s as fit; freed only on drop. A test using it on a handle of that
    // device, and writing it only through one `&mut`, keeps the rest.
    unsafe impl<T> crate::device_slice::DeviceSlice<T> for DeviceBytes {
        fn as_ptr(&self) -> *const T {
            self.0.cast_const().cast()
        }

        fn len(&self) -> usize {
            self.1 / size_of::<T>()
        }
    }

    // SAFETY: as above; `&mut self` makes the caller its only user.
    unsafe impl<T> crate::device_slice::DeviceSliceMut<T> for DeviceBytes {
        fn as_mut_ptr(&mut self) -> *mut T {
            self.0.cast()
        }
    }

    /// Device 0's stack limit.
    #[cfg(test)]
    fn stack_limit() -> usize {
        let mut limit = 0;
        let _device = crate::common::ScopedDeviceIndex::new(0);
        // SAFETY: a plain runtime call writing one local.
        ok(unsafe {
            wwr_sys::wwr_device_get_limit(&mut limit, wwr_sys::wwr_limit::WWR_LIMIT_STACK_SIZE)
        });
        limit
    }

    /// The runtime counts device 0, which every device case uses, and no
    /// index at or past the count can be made current. Needs a GPU.
    #[cfg(test)]
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn the_device_count_covers_device_0_and_ends_there() {
        let n = super::device_count().unwrap();
        assert!(n >= 1, "{n} devices");
        let past = i32::try_from(n).unwrap();
        assert!(crate::common::ScopedDeviceIndex::try_new(past).is_err());
        assert!(crate::common::ScopedDeviceIndex::try_new(past - 1).is_ok());
    }

    /// The stack term for the C++ tree's largest per-thread stack, 16,912
    /// bytes: exactly 0 under HIP, and on either backend the device's stack
    /// limit is what it was before the probe. Nothing is asserted of the
    /// CUDA term's size, two readings of a card other tenants share. Needs
    /// a GPU.
    #[cfg(test)]
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn the_stack_term_leaves_the_limit_as_it_was() {
        let limit = stack_limit();
        let growth = super::stack_growth(0, 16_912).unwrap();
        println!("device 0: stack limit {limit} B, stack term {growth} B");
        if super::HIP {
            assert_eq!(growth, 0, "HIP sizes scratch per dispatch");
        }
        assert_eq!(stack_limit(), limit, "the probe sets the limit back");
    }

    #[cfg(test)]
    #[test]
    fn first_failure_prefers_the_sync() {
        use super::first_failure;
        use wwr_status::*;
        assert_eq!(
            first_failure(WWR_STATUS_SUCCESS, WWR_STATUS_SUCCESS),
            WWR_STATUS_SUCCESS
        );
        assert_eq!(
            first_failure(WWR_STATUS_SUCCESS, WWR_STATUS_UNKNOWN),
            WWR_STATUS_UNKNOWN
        );
        assert_eq!(
            first_failure(WWR_STATUS_NOT_READY, WWR_STATUS_UNKNOWN),
            WWR_STATUS_NOT_READY
        );
    }
}
