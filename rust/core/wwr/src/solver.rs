//! wwr's `src/extension/solver` (module `wwr.extension.solver`): the dense
//! solver's status as an error code, and the owning dense-solver handle
//! bound to a stream.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `:solver_error`, the `wwrsolverStatus_t` specializations | `impl ErrorCode for wwr_solver_status` |
//! | `:solver_handle`, `SolverDnHandleWrapper<P_create, P_destroy, S, P_device>` | [`SolverHandle`], over [`Stream`] |
//! | `:solver_params`, the generic API's params object | not ported: the legacy int-based calls take none |
//! | `usual_fp`'s real half, the typed dispatch `syevd<T>`, `syevd_lwork<T>` over `wwrsolverDn{S,D}syevd` and its query | [`SolverScalar`], its `syevd_raw` and `syevd_lwork_raw` (`solver/syevd.rs`) |
//! | the same `syevd<T>`, `syevd_lwork<T>`, called with raw pointers | [`syevd`] over [`DeviceSliceMut`](crate::device_slice::DeviceSliceMut)s it checks, the work against the query; [`syevd_lwork`] |
//!
//! As [`BlasHandle`](crate::blas::BlasHandle) is: a [`StreamBoundHandle`]
//! made from a shared stream owner, created on its device, bound to its
//! stream, and holding the owner so the stream outlives it. Its constructor
//! aborts on failure.
//!
//! **The status has two values.** WarpWraps' C ABI maps every failure of
//! cuSOLVER or hipSOLVER to `WWR_SOLVER_STATUS_UNKNOWN`, so the vendor's
//! reason does not cross; a caller names the call that failed.
//!
//! **The typed calls**, one file per routine (`solver/syevd.rs`), are as
//! [`blas`](crate::blas)'s: [`SolverScalar`] is the `unsafe` raw layer,
//! sealed over the types the C ABI takes, and [`syevd`] the safe call over
//! it, every extent checked against its slice in every build.

mod syevd;

pub use syevd::{SolverScalar, syevd, syevd_lwork};

use std::ops::Deref;
use std::rc::Rc;

use wwr_sys::{wwr_solver_handle, wwr_solver_status, wwr_stream};

use crate::common::ErrorCode;
use crate::handle::{
    DeviceHandle, DeviceHandleStream, HandleKind, StreamBoundHandle, StreamBoundKind,
};
use crate::runtime::Stream;
use fb_core_error_handling::check_or_abort;

/// `wwrsolverStatus_t`, as the C ABI hands it back.
impl ErrorCode for wwr_solver_status {
    const SUCCESS: Self = wwr_solver_status::WWR_SOLVER_STATUS_SUCCESS;

    fn name(self) -> &'static str {
        use wwr_solver_status::*;
        match self {
            WWR_SOLVER_STATUS_SUCCESS => "WWR_SOLVER_STATUS_SUCCESS",
            WWR_SOLVER_STATUS_UNKNOWN => "WWR_SOLVER_STATUS_UNKNOWN",
        }
    }

    fn description(self) -> &'static str {
        use wwr_solver_status::*;
        match self {
            WWR_SOLVER_STATUS_SUCCESS => "no error",
            WWR_SOLVER_STATUS_UNKNOWN => "a solver error; the C ABI keeps no code of its own",
        }
    }
}

/// A dense-solver handle's create, destroy and stream bind.
pub struct SolverKind;

impl HandleKind for SolverKind {
    type Raw = wwr_solver_handle;
    type Error = wwr_solver_status;

    unsafe fn create(raw: *mut wwr_solver_handle) -> wwr_solver_status {
        // SAFETY: the caller hands a pointer valid for one write.
        unsafe { wwr_sys::wwr_solver_dn_create(raw) }
    }

    unsafe fn destroy(raw: wwr_solver_handle) -> wwr_solver_status {
        // SAFETY: `raw` is a live handle, destroyed once.
        unsafe { wwr_sys::wwr_solver_dn_destroy(raw) }
    }
}

impl StreamBoundKind for SolverKind {
    unsafe fn set_stream(raw: wwr_solver_handle, stream: wwr_stream) -> wwr_solver_status {
        // SAFETY: the caller's: both are live.
        unsafe { wwr_sys::wwr_solver_dn_set_stream(raw, stream) }
    }
}

/// An owned dense-solver handle bound to a stream it keeps alive:
/// `SolverDnHandleWrapper`. Every call through it enqueues on that stream.
///
/// Neither `Send` nor `Sync`, as [`BlasHandle`](crate::blas::BlasHandle).
pub struct SolverHandle(StreamBoundHandle<SolverKind, Stream>);

impl SolverHandle {
    /// A handle on `owner`'s device, bound to its stream. Aborts on
    /// failure, logging at critical against the caller.
    #[track_caller]
    pub fn new(owner: Rc<Stream>) -> Self {
        SolverHandle(check_or_abort(StreamBoundHandle::try_new(owner)))
    }
}

impl Deref for SolverHandle {
    type Target = StreamBoundHandle<SolverKind, Stream>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DeviceHandle for SolverHandle {
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

impl DeviceHandleStream for SolverHandle {
    /// The bound stream.
    fn stream(&self) -> wwr_stream {
        self.0.stream()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_status;
    use crate::runtime::tests::ok;

    #[test]
    fn every_status_has_a_name_and_a_description() {
        use wwr_solver_status::*;
        for s in [WWR_SOLVER_STATUS_SUCCESS, WWR_SOLVER_STATUS_UNKNOWN] {
            assert_eq!(s.name(), format!("{s:?}"));
            assert!(!s.description().is_empty());
        }
        assert!(gpu_status(WWR_SOLVER_STATUS_SUCCESS).is_ok());
        assert_eq!(
            gpu_status(WWR_SOLVER_STATUS_UNKNOWN).unwrap_err().message,
            "WWR_SOLVER_STATUS_UNKNOWN (a solver error; the C ABI keeps no code of its own)"
        );
    }

    /// Needs a GPU. The handle lands on the owner's device and the library
    /// reads back the owner's stream as the one it enqueues on.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[allow(
        clippy::disallowed_methods,
        reason = "a test reads its result back on the host: the harness waits, not a task"
    )]
    fn binds_the_owners_stream() {
        let owner = Rc::new(Stream::new(0));
        let solver = SolverHandle::new(owner.clone());
        assert_eq!(solver.dev_idx(), 0);
        assert_eq!(solver.stream(), owner.stream());
        let mut bound: wwr_stream = std::ptr::null_mut();
        // SAFETY: the handle is live; writes one stream through a local.
        let status = unsafe { wwr_sys::wwr_solver_dn_get_stream(solver.get(), &mut bound) };
        assert_eq!(status, wwr_solver_status::WWR_SOLVER_STATUS_SUCCESS);
        assert_eq!(bound, owner.stream());
        drop(solver);
        assert_eq!(Rc::strong_count(&owner), 1);
        ok(owner.sync());
    }
}
