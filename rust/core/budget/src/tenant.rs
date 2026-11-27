//! Tenants of an `Output`'s own buffers: a typed piece of a
//! [`PinnedBuf`](crate::PinnedBuf) or a [`DeviceBuf`](crate::DeviceBuf),
//! carved by a [`Span`] of a [`Layout`](crate::Layout) over the buffer's
//! bytes, as a lease's tenants are carved from a step's region, and handed
//! back by dropping it. A manager's ring carves the same bytes again on
//! every turn.
//!
//! **Each carve is recorded in the buffer, always on, and dropping the
//! tenant clears it.** A carve that meets a live one panics, so two
//! tenants never share a byte. A pinned tenant is the host end of a copy
//! and moves into the copy's token; a token dropped unsynced leaks it, so
//! its record stays: those bytes are never carved again, and the buffer is
//! never reached whole again, while the copy may still run.
//!
//! **A tenant owns a share of the allocation.** It holds the buffer's
//! allocation alive beside the buffer, so a leaked tenant leaks the
//! memory, never freeing bytes a copy may use, and a buffer dropped before
//! its tenants frees its memory with the last of them.
//!
//! **A buffer with a live tenant is not reached whole.** Carving takes the
//! buffer by `&mut`, so no borrow of the whole is alive at the carve, and
//! every later reach of the whole (its slice, its pointer as a copy's or a
//! kernel's operand) panics while a carve is recorded.

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use fb_core_device::{DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut};

use crate::buffers::Plain;
use crate::lease::Span;
use crate::ledger::locked;
use crate::raw::RawBuf;

/// One allocation of an `Output`'s buffer and the byte ranges carved from
/// it, shared by the buffer and its tenants.
pub(crate) struct Tenancy {
    pub(crate) raw: RawBuf,
    carved: Mutex<Vec<(usize, usize)>>,
}

impl Tenancy {
    /// `raw`, nothing carved.
    pub(crate) fn new(raw: RawBuf) -> Arc<Tenancy> {
        Arc::new(Tenancy {
            raw,
            carved: Mutex::new(Vec::new()),
        })
    }

    /// The allocation's first byte, for a reach of the whole buffer.
    ///
    /// # Panics
    /// If a tenant is carved, live or leaked, naming `what`.
    #[track_caller]
    pub(crate) fn base(&self, what: &str) -> *mut u8 {
        let carved = locked(&self.carved);
        if let Some(&(b, e)) = carved.first() {
            panic!("a {what} is reached whole while its bytes {b}..{e} are a tenant's");
        }
        self.raw.ptr
    }

    /// Record `span`'s bytes as carved; where they start.
    ///
    /// # Panics
    /// If they meet a tenant already carved, naming `what`.
    #[track_caller]
    fn carve<U>(&self, span: &Span<U>, what: &str) -> (*mut U, (usize, usize)) {
        let (begin, end) = (span.offset, span.end());
        let mut carved = locked(&self.carved);
        if span.len > 0 {
            if let Some(&(b, e)) = carved.iter().find(|&&(b, e)| begin < e && b < end) {
                panic!(
                    "a tenant of a {what} carves bytes {begin}..{end}, which meet the tenant \
                     carved at {b}..{e}"
                );
            }
            carved.push((begin, end));
        }
        // In range: the caller's contract keeps the span inside the bytes.
        (self.raw.ptr.wrapping_add(begin).cast(), (begin, end))
    }

    /// Clear the record of `range`.
    fn release(&self, range: (usize, usize)) {
        let mut carved = locked(&self.carved);
        if let Some(i) = carved.iter().position(|&r| r == range) {
            carved.swap_remove(i);
        }
    }
}

/// A carve of `span` from `tenancy`: the pointer, the length and the share.
#[track_caller]
pub(crate) fn carve<U>(tenancy: &Arc<Tenancy>, span: &Span<U>, what: &str) -> Carve<U> {
    let (ptr, range) = tenancy.carve(span, what);
    Carve {
        ptr,
        len: span.len,
        range,
        tenancy: Arc::clone(tenancy),
    }
}

/// What both tenants hold: their elements and their share of the
/// allocation, the record cleared on drop.
pub(crate) struct Carve<U> {
    ptr: *mut U,
    len: usize,
    range: (usize, usize),
    tenancy: Arc<Tenancy>,
}

impl<U> Drop for Carve<U> {
    fn drop(&mut self) {
        if self.len > 0 {
            self.tenancy.release(self.range);
        }
    }
}

/// A tenant of a [`PinnedBuf`](crate::PinnedBuf): a slice for the host,
/// and the host end of a copy, taken by value as the whole buffer is. Made
/// by [`PinnedBuf::tenant`](crate::PinnedBuf::tenant).
///
/// A token dropped unsynced leaks it, so its bytes stay carved and the
/// allocation stays alive:
///
/// ```no_run
/// use fb_core_budget::{DeviceBuf, Layout, PinnedBuf};
/// use fb_core_device::Stream;
///
/// fn leak(stream: &Stream, dst: &mut DeviceBuf<f64>, staging: &mut PinnedBuf<u8>) {
///     let mut layout = Layout::new();
///     let span = layout.span::<f64>(4);
///     let tenant = staging.tenant(&span);
///     drop(stream.upload(dst, tenant).unwrap()); // unsynced: leaked
///     let _again = staging.tenant(&span); // panics: the bytes are still carved
/// }
/// ```
pub struct PinnedTenant<U>(Carve<U>);

/// A tenant of a [`DeviceBuf`](crate::DeviceBuf): a typed run of its
/// device memory, apart from every other live tenant. Made by
/// [`DeviceBuf::tenant`](crate::DeviceBuf::tenant).
pub struct DeviceTenant<U>(Carve<U>);

impl<U> PinnedTenant<U> {
    pub(crate) fn new(carve: Carve<U>) -> PinnedTenant<U> {
        PinnedTenant(carve)
    }
}

impl<U> DeviceTenant<U> {
    pub(crate) fn new(carve: Carve<U>) -> DeviceTenant<U> {
        DeviceTenant(carve)
    }
}

// SAFETY: as `&mut [U]` moves when `U: Send`: the tenant's bytes are its
// alone (the record), and the share of the allocation is an `Arc` of
// `Send + Sync` parts.
unsafe impl<U: Send> Send for PinnedTenant<U> {}
// SAFETY: as `&mut [U]` is shared when `U: Sync`.
unsafe impl<U: Sync> Sync for PinnedTenant<U> {}
// SAFETY: device memory is not tied to a host thread, and the tenant is
// this one's alone.
unsafe impl<U: Send> Send for DeviceTenant<U> {}
// SAFETY: a `&DeviceTenant` hands out its address alone.
unsafe impl<U: Sync> Sync for DeviceTenant<U> {}

impl<U: Plain> Deref for PinnedTenant<U> {
    type Target = [U];
    fn deref(&self) -> &[U] {
        // SAFETY: `len` plain elements of the buffer's host memory, filled
        // when made, kept alive by the share, and this tenant's alone while
        // it lives (the record keeps tenants apart, and the whole buffer is
        // not reached while one is recorded).
        unsafe { std::slice::from_raw_parts(self.0.ptr, self.0.len) }
    }
}

impl<U: Plain> DerefMut for PinnedTenant<U> {
    fn deref_mut(&mut self) -> &mut [U] {
        // SAFETY: as `deref`, and `&mut self` is the one user.
        unsafe { std::slice::from_raw_parts_mut(self.0.ptr, self.0.len) }
    }
}

// SAFETY: `len` elements of host memory, reached only through this tenant
// (the record hands its bytes out once, and the whole buffer refuses to be
// reached while they are recorded). One never dropped keeps its record
// and its share, so the bytes stay valid and reached by no one for as long
// as the process lives.
unsafe impl<U: Plain> HostSlice<U> for PinnedTenant<U> {
    fn as_ptr(&self) -> *const U {
        self.0.ptr
    }
    fn len(&self) -> usize {
        self.0.len
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<U: Plain> HostSliceMut<U> for PinnedTenant<U> {
    fn as_mut_ptr(&mut self) -> *mut U {
        self.0.ptr
    }
}

// SAFETY: `len` elements of the buffer's device memory, kept alive by the
// share, apart from every other live tenant (the record) and from the
// whole buffer, which refuses to be reached while a tenant is recorded;
// only `as_mut_ptr`, through `&mut self`, writes it.
unsafe impl<U> DeviceSlice<U> for DeviceTenant<U> {
    fn as_ptr(&self) -> *const U {
        self.0.ptr
    }
    fn len(&self) -> usize {
        self.0.len
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<U> DeviceSliceMut<U> for DeviceTenant<U> {
    fn as_mut_ptr(&mut self) -> *mut U {
        self.0.ptr
    }
}
