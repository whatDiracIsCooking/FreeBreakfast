//! `wwr.extension.handle:handle`: the owning RAII handle.

use std::mem::MaybeUninit;
use std::panic::Location;

use super::HandleView;
use crate::core::wwr::common::{ErrorCode, ErrorPolicy, SourceLocation, gpu_check_at};

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

/// Run `K::create` through `policy`, and yield the handle if it is live: the
/// default create path every constructor takes unless handed another.
pub(super) fn create_raw<K: HandleKind, P: ErrorPolicy<Error = K::Error>>(
    policy: &mut P,
    location: SourceLocation,
) -> Option<K::Raw> {
    let mut raw = MaybeUninit::uninit();
    // SAFETY: `raw` is a local, valid for one write; it is read only when
    // the create succeeded, which is when the kind promises it wrote it.
    let ok = gpu_check_at(unsafe { K::create(raw.as_mut_ptr()) }, policy, location);
    ok.then(|| unsafe { raw.assume_init() })
}

/// An owned GPU handle, destroyed when dropped: `BaseHandle`.
///
/// - **Liveness** is `Option<K::Raw>`: `None` after a create that failed
///   under a policy that returns. wwr needs a null sentinel for a pointer
///   handle and a separate bool for an integer one (`cufftHandle`); `Option`
///   is both, for one word more than a bare pointer.
/// - **Moves** are Rust's. Moving out leaves nothing behind to drop, so
///   wwr's move constructor and assignment, `release()` and the moved-from
///   state have no counterpart; assigning over a handle drops, and so
///   destroys, the one it held.
/// - **Custom creation** (`skip_default_create_t`, then `adopt`) is
///   [`BaseHandle::new_with`], handed the create to run in place of
///   `K::create`.
pub struct BaseHandle<K, PC, PD>
where
    K: HandleKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
{
    pub(super) raw: Option<K::Raw>,
    pub(super) policy_create: PC,
    pub(super) policy_destroy: PD,
}

impl<K, PC, PD> BaseHandle<K, PC, PD>
where
    K: HandleKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
{
    /// Create a handle with `K::create`, failures going to `policy_create`
    /// at the caller's location.
    #[track_caller]
    pub fn new(policy_create: PC, policy_destroy: PD) -> Self {
        Self::new_at(
            policy_create,
            policy_destroy,
            create_raw::<K, PC>,
            Location::caller(),
        )
    }

    /// Create a handle with `create`, for a kind whose creation takes
    /// arguments `K::create` cannot carry. It returns the handle if it made
    /// one, reporting through the create policy it is handed.
    #[track_caller]
    pub fn new_with(
        policy_create: PC,
        policy_destroy: PD,
        create: impl FnOnce(&mut PC, SourceLocation) -> Option<K::Raw>,
    ) -> Self {
        Self::new_at(policy_create, policy_destroy, create, Location::caller())
    }

    pub(super) fn new_at(
        mut policy_create: PC,
        policy_destroy: PD,
        create: impl FnOnce(&mut PC, SourceLocation) -> Option<K::Raw>,
        location: SourceLocation,
    ) -> Self {
        let raw = create(&mut policy_create, location);
        BaseHandle {
            raw,
            policy_create,
            policy_destroy,
        }
    }

    /// The handle, if this owns a live one: wwr's `get()` and `valid()`
    /// together.
    pub fn get(&self) -> Option<K::Raw> {
        self.raw
    }

    /// The policy that handled creation, for reading back what a stateful
    /// one recorded.
    pub fn create_policy(&self) -> &PC {
        &self.policy_create
    }

    /// The create policy, for a wrapper whose later calls report through it
    /// as construction did (a stream's `end_capture`).
    pub fn create_policy_mut(&mut self) -> &mut PC {
        &mut self.policy_create
    }

    /// The policy that handles destruction.
    pub fn destroy_policy(&self) -> &PD {
        &self.policy_destroy
    }

    /// A copyable view of the handle, which cannot outlive `self`; `None`
    /// when there is no live handle to view.
    pub fn view(&self) -> Option<HandleView<'_, K::Raw>> {
        self.raw.map(HandleView::from_raw)
    }
}

impl<K, PC, PD> Drop for BaseHandle<K, PC, PD>
where
    K: HandleKind,
    PC: ErrorPolicy<Error = K::Error>,
    PD: ErrorPolicy<Error = K::Error>,
{
    fn drop(&mut self) {
        if let Some(raw) = self.raw.take() {
            // SAFETY: `raw` is live and owned, and taken out, so it is
            // destroyed once.
            gpu_check_at(
                unsafe { K::destroy(raw) },
                &mut self.policy_destroy,
                Location::caller(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{Fake, fake_destroyed, fake_fail_next_create};
    use super::*;
    use crate::core::wwr::common::tests::Recording;
    use wwr_sys::wwr_status;

    type Handle = BaseHandle<Fake, Recording, Recording>;

    #[track_caller] // so a failure is reported at the test's line
    fn new() -> Handle {
        Handle::new(Recording::default(), Recording::default())
    }

    #[test]
    fn drop_destroys_once() {
        let before = fake_destroyed();
        let h = new();
        assert!(h.get().is_some());
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
        assert!(h.get().is_some());
        h = new();
        assert_eq!(fake_destroyed(), before + 1);
        drop(h);
        assert_eq!(fake_destroyed(), before + 2);
    }

    #[test]
    fn a_failed_create_is_reported_at_the_caller_and_never_destroyed() {
        let before = fake_destroyed();
        fake_fail_next_create();
        let line = line!() + 1;
        let h = new();
        assert_eq!(h.get(), None);
        assert!(h.view().is_none());
        assert_eq!(
            h.create_policy().0,
            [(wwr_status::WWR_STATUS_MEMORY_ALLOCATION, line)]
        );
        drop(h);
        assert_eq!(fake_destroyed(), before);
    }

    #[test]
    fn new_with_runs_the_given_create() {
        let h = Handle::new_with(Recording::default(), Recording::default(), |_, _| Some(7));
        assert_eq!(h.get(), Some(7));
        assert_eq!(h.view().map(|v| v.get()), Some(7));
    }
}
