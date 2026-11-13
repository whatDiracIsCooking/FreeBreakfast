//! `wwr.extension.handle:handle`: the owning RAII handle.

use std::mem::MaybeUninit;
use std::panic::Location;

use super::HandleView;
use crate::core::error_handling::Result;
use crate::core::wwr::common::{ErrorCode, SourceLocation};
use crate::core::wwr::{gpu_log_error_at, gpu_status_at};

/// What a kind of GPU handle is: its raw type, the status its calls return,
/// and how it is created and destroyed. It is CRTP's `Derived`, reduced to
/// the two hooks `BaseHandle` calls on it.
///
/// The error type is the kind's own, not looked up from the raw type. wwr
/// takes it from the create policy because on HIP every library handle is
/// `void*` and a raw-type table cannot tell them apart; a Rust kind is a
/// distinct type even where its `Raw` is shared, so it can carry its own.
pub trait HandleKind {
    /// The backend handle (`wwrStream_t`, `wwrblasHandle_t`, ...).
    type Raw: Copy;
    /// The status the create and destroy calls return.
    type Error: ErrorCode;

    /// Create a handle, writing it through `raw` on success.
    ///
    /// # Safety
    /// `raw` is valid for one write. On success the kind must have written a
    /// live handle there; on failure it may have written anything or nothing.
    unsafe fn create(raw: *mut Self::Raw) -> Self::Error;

    /// Destroy `raw`.
    ///
    /// # Safety
    /// `raw` is a live handle of this kind, and is never used again.
    unsafe fn destroy(raw: Self::Raw) -> Self::Error;
}

/// An owned GPU handle, destroyed when dropped: `BaseHandle`.
///
/// - **A handle is live for as long as it exists.** A create that fails is
///   an `Err` and yields no handle, so there is no empty state. wwr needs
///   one, a null sentinel for a pointer handle and a separate bool for an
///   integer one (`cufftHandle`), because its constructor cannot fail
///   without an exception; a Rust constructor returns `Result`.
/// - **Moves** are Rust's. Moving out leaves nothing behind to drop, so
///   wwr's move constructor and assignment, `release()` and the moved-from
///   state have no counterpart; assigning over a handle drops, and so
///   destroys, the one it held.
/// - **Custom creation** (`skip_default_create_t`, then `adopt`) is
///   [`BaseHandle::try_new_with`], handed the create to run in place of
///   `K::create`, or [`BaseHandle::adopt`] for a handle made elsewhere.
/// - **What a failure does** is the caller's: this layer reports a failed
///   create as an `Err`, and the wrappers above it abort on one, the tree's
///   rule. A failed destroy is logged at error ([`gpu_log_error_at`]).
pub struct BaseHandle<K: HandleKind> {
    raw: K::Raw,
}

impl<K: HandleKind> BaseHandle<K> {
    /// Create a handle with `K::create`; a failure is an `Err` at the
    /// caller's location.
    #[track_caller]
    pub fn try_new() -> Result<Self> {
        // SAFETY: `try_new_at` hands a pointer valid for one write.
        Self::try_new_at(|raw| unsafe { K::create(raw) }, Location::caller())
    }

    /// Create a handle with `create`, for a kind whose creation takes
    /// arguments `K::create` cannot carry. `create` is handed a pointer valid
    /// for one write, and must write a live handle there when it returns
    /// success.
    #[track_caller]
    pub fn try_new_with(create: impl FnOnce(*mut K::Raw) -> K::Error) -> Result<Self> {
        Self::try_new_at(create, Location::caller())
    }

    pub(crate) fn try_new_at(
        create: impl FnOnce(*mut K::Raw) -> K::Error,
        location: SourceLocation,
    ) -> Result<Self> {
        let mut raw = MaybeUninit::uninit();
        gpu_status_at(create(raw.as_mut_ptr()), location)?;
        // SAFETY: the create succeeded, which is when it promises to have
        // written a live handle.
        Ok(BaseHandle {
            raw: unsafe { raw.assume_init() },
        })
    }

    /// Own `raw`, a handle made elsewhere: wwr's `adopt`. It is destroyed
    /// with this.
    ///
    /// # Safety
    /// `raw` is a live handle of this kind that nothing else destroys.
    pub unsafe fn adopt(raw: K::Raw) -> Self {
        BaseHandle { raw }
    }

    /// The handle: wwr's `get()`. There is no `valid()`; see the type's doc.
    pub fn get(&self) -> K::Raw {
        self.raw
    }

    /// A copyable view of the handle, which cannot outlive `self`.
    pub fn view(&self) -> HandleView<'_, K::Raw> {
        HandleView::from_raw(self.raw)
    }
}

impl<K: HandleKind> Drop for BaseHandle<K> {
    fn drop(&mut self) {
        // SAFETY: `raw` is live and owned, and this runs once.
        gpu_log_error_at(unsafe { K::destroy(self.raw) }, Location::caller());
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{Fake, fake_destroyed, fake_fail_next_create};
    use super::*;
    use crate::core::error_handling::ErrorKind;

    type Handle = BaseHandle<Fake>;

    #[track_caller] // so a failure is reported at the test's line
    fn new() -> Handle {
        Handle::try_new().unwrap()
    }

    #[test]
    fn drop_destroys_once() {
        let before = fake_destroyed();
        let h = new();
        assert_eq!(h.get(), 1);
        drop(h);
        assert_eq!(fake_destroyed(), before + 1);
    }

    #[test]
    fn a_move_does_not_destroy() {
        let before = fake_destroyed();
        let h = new();
        let raw = h.get();
        let moved = h;
        assert_eq!(moved.get(), raw);
        assert_eq!(fake_destroyed(), before);
        drop(moved);
        assert_eq!(fake_destroyed(), before + 1);
    }

    #[test]
    fn assigning_over_destroys_the_old_handle() {
        let before = fake_destroyed();
        let mut h = new();
        assert_eq!(h.get(), 1);
        h = new();
        assert_eq!(fake_destroyed(), before + 1);
        drop(h);
        assert_eq!(fake_destroyed(), before + 2);
    }

    #[test]
    fn a_failed_create_is_an_error_at_the_caller_and_never_destroyed() {
        let before = fake_destroyed();
        fake_fail_next_create();
        let line = line!() + 1;
        let error = Handle::try_new().err().unwrap();
        assert_eq!(error.kind, ErrorKind::GpuRuntime);
        assert!(error.message.starts_with("WWR_STATUS_MEMORY_ALLOCATION"));
        assert_eq!((error.origin.file(), error.origin.line()), (file!(), line));
        assert_eq!(fake_destroyed(), before);
    }

    #[test]
    fn try_new_with_runs_the_given_create() {
        let h = Handle::try_new_with(|raw| {
            // SAFETY: `raw` is valid for one write.
            unsafe { *raw = 7 };
            wwr_sys::wwr_status::WWR_STATUS_SUCCESS
        })
        .unwrap();
        assert_eq!(h.get(), 7);
        assert_eq!(h.view().get(), 7);
    }

    #[test]
    fn adopt_destroys_the_adopted_handle() {
        let before = fake_destroyed();
        // SAFETY: the fake's handles are plain integers nothing else owns.
        let h = unsafe { Handle::adopt(9) };
        assert_eq!(h.get(), 9);
        drop(h);
        assert_eq!(fake_destroyed(), before + 1);
    }
}
