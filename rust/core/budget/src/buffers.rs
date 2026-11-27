//! What a [`Budget`](crate::Budget) hands back for memory an `Output`
//! owns: a device buffer, a pinned buffer, a heap slab. Fixed size, freed
//! with the `Output`; only a `Budget` makes one.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use contracts::requires;
use fb_core_device::{DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut};
use fb_core_log::le;

use crate::lease::Span;
use crate::raw::RawBuf;
use crate::tenant::{DeviceTenant, PinnedTenant, Tenancy, carve};

mod sealed {
    pub trait Sealed {}
}

/// The element types host-visible budget memory is read as: plain data,
/// every bit pattern a value, so a NaN-filled or reused region is still
/// valid to hold. Sealed: a type with an invalid bit pattern cannot join.
pub trait Plain: sealed::Sealed + Copy + Send + Sync + 'static {}

macro_rules! plain {
    ($($t:ty),*) => {$(
        impl sealed::Sealed for $t {}
        impl Plain for $t {}
    )*};
}
plain!(f64, f32, i64, u64, i32, u32, u8);

/// `raw`'s first `len` elements, the length a buffer reports: none when
/// nothing was allocated, so an empty buffer's slice promises nothing.
fn allocated(raw: Option<&Arc<Tenancy>>, len: usize) -> usize {
    raw.map_or(0, |_| len)
}

/// Device memory a step's `Output` owns. Empty, of length 0, under a dry
/// or refused budget.
///
/// Its bytes can be carved into typed tenants ([`DeviceBuf::tenant`]),
/// several live at once, while the whole is not reached.
pub struct DeviceBuf<T> {
    pub(crate) raw: Option<Arc<Tenancy>>,
    pub(crate) len: usize,
    pub(crate) of: PhantomData<T>,
}

impl<T: Plain> DeviceBuf<T> {
    /// `raw`'s first `len` elements, or an empty buffer.
    pub(crate) fn new(raw: Option<RawBuf>, len: usize) -> DeviceBuf<T> {
        DeviceBuf {
            raw: raw.map(Tenancy::new),
            len,
            of: PhantomData,
        }
    }

    /// The bytes as host memory, under [`Backing::Host`](crate::Backing)
    /// only: how the device-free tests read and write a "device" buffer on
    /// no gpu.
    ///
    /// # Panics
    /// If a tenant of the buffer is live.
    #[track_caller]
    pub fn host(&mut self) -> Option<&mut [T]> {
        let t = self.raw.as_ref().filter(|t| t.raw.is_host())?;
        let ptr = t.base("device buffer");
        // SAFETY: `len` elements of zeroed host memory, a valid `T` at any
        // bit pattern (`Plain`), no tenant carved from them (`whole`), and
        // `&mut self` is the one user.
        Some(unsafe { std::slice::from_raw_parts_mut(ptr.cast(), self.len) })
    }

    /// The tenant `span` of the buffer's bytes, typed `U`: carved from a
    /// [`Layout`](crate::Layout) over them, as a lease's tenants are, and
    /// handed back by dropping it. Several are live at once, and the same
    /// span may be carved again once its tenant is dropped.
    ///
    /// # Panics
    /// If the span meets a live tenant of the buffer, or the buffer holds
    /// nothing (a dry or refused budget's).
    #[requires(
        le!(span.end(), self.len() * size_of::<T>()),
        "the span is inside the buffer's bytes"
    )]
    #[track_caller]
    pub fn tenant<U: Plain>(&mut self, span: &Span<U>) -> DeviceTenant<U> {
        let t = self
            .raw
            .as_ref()
            .expect("a span inside the bytes means an allocation");
        DeviceTenant::new(carve(t, span, "device buffer"))
    }
}

// SAFETY: `len()` elements of device memory (0 when not allocated), live
// while `self` is; only `as_mut_ptr`, through `&mut self`, writes it, and
// neither hands out the address while a tenant is carved (`whole`), so no
// tenant writes it meanwhile.
unsafe impl<T> DeviceSlice<T> for DeviceBuf<T> {
    #[track_caller]
    fn as_ptr(&self) -> *const T {
        self.raw.as_ref().map_or(std::ptr::null(), |t| {
            t.base("device buffer").cast_const().cast()
        })
    }
    fn len(&self) -> usize {
        allocated(self.raw.as_ref(), self.len)
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T> DeviceSliceMut<T> for DeviceBuf<T> {
    #[track_caller]
    fn as_mut_ptr(&mut self) -> *mut T {
        self.raw
            .as_ref()
            .map_or(std::ptr::null_mut(), |t| t.base("device buffer").cast())
    }
}

/// Page-locked host memory a step's `Output` owns, staged for one gpu,
/// filled with `0xFF` when made. Read and written as a slice, and the host
/// end of a copy ([`HostSlice`]). Empty under a dry or refused budget.
///
/// A copy takes it by value and holds it in its token until a sync of the
/// stream has covered the copy, so the host cannot touch the bytes while
/// the copy may still run:
///
/// ```compile_fail,E0382
/// use fb_core_budget::{DeviceBuf, PinnedBuf};
/// use fb_core_device::Stream;
///
/// fn rewrite(stream: &Stream, dst: &mut DeviceBuf<f64>, mut staged: PinnedBuf<f64>) {
///     let upload = stream.upload(dst, staged).unwrap();
///     staged[0] = 1.0; // the upload's token holds it
///     let _ = upload.sync();
/// }
/// ```
///
/// The token's sync hands it back:
///
/// ```no_run
/// use fb_core_budget::{DeviceBuf, PinnedBuf};
/// use fb_core_device::Stream;
/// use fb_core_error_handling::Result;
///
/// fn rewrite(
///     stream: &Stream,
///     dst: &mut DeviceBuf<f64>,
///     staged: PinnedBuf<f64>,
/// ) -> Result<PinnedBuf<f64>> {
///     let mut staged = stream.upload(dst, staged)?.sync()?;
///     staged[0] = 1.0;
///     Ok(staged)
/// }
/// ```
///
/// Its bytes can also be carved into typed tenants ([`PinnedBuf::tenant`]),
/// each the host end of a copy of its own, several in flight at once,
/// while the whole is not reached.
pub struct PinnedBuf<T> {
    pub(crate) raw: Option<Arc<Tenancy>>,
    pub(crate) len: usize,
    pub(crate) of: PhantomData<T>,
}

impl<T: Plain> PinnedBuf<T> {
    /// `raw`'s first `len` elements, or an empty buffer.
    pub(crate) fn new(raw: Option<RawBuf>, len: usize) -> PinnedBuf<T> {
        PinnedBuf {
            raw: raw.map(Tenancy::new),
            len,
            of: PhantomData,
        }
    }

    /// The tenant `span` of the buffer's bytes, typed `U`: carved from a
    /// [`Layout`](crate::Layout) over them, as a lease's tenants are, and
    /// handed back by dropping it, or by the sync of the copy it moved
    /// into. Several are live at once, and the same span may be carved
    /// again once its tenant is dropped; a copy's token dropped unsynced
    /// leaks the tenant, and its bytes stay carved.
    ///
    /// The carve takes the buffer by `&mut`, so no slice of the whole is
    /// alive beside a tenant:
    ///
    /// ```compile_fail,E0502
    /// use fb_core_budget::{Layout, PinnedBuf};
    ///
    /// fn alias(staging: &mut PinnedBuf<u8>) {
    ///     let whole: &[u8] = staging;
    ///     let mut tenant = staging.tenant(&Layout::new().span::<u8>(8));
    ///     tenant[0] = whole[0];
    /// }
    /// ```
    ///
    /// # Panics
    /// If the span meets a live tenant of the buffer, or the buffer holds
    /// nothing (a dry or refused budget's).
    #[requires(
        le!(span.end(), self.len() * size_of::<T>()),
        "the span is inside the buffer's bytes"
    )]
    #[track_caller]
    pub fn tenant<U: Plain>(&mut self, span: &Span<U>) -> PinnedTenant<U> {
        let t = self
            .raw
            .as_ref()
            .expect("a span inside the bytes means an allocation");
        PinnedTenant::new(carve(t, span, "pinned buffer"))
    }

    /// The whole buffer's first element.
    ///
    /// # Panics
    /// If a tenant of the buffer is live or leaked.
    #[track_caller]
    fn base(&self) -> *mut T {
        self.raw
            .as_ref()
            .map_or(std::ptr::NonNull::dangling().as_ptr(), |t| {
                t.base("pinned buffer").cast()
            })
    }
}

/// Reached whole, with no tenant carved.
///
/// # Panics
/// If a tenant of the buffer is live or leaked.
impl<T: Plain> Deref for PinnedBuf<T> {
    type Target = [T];
    #[track_caller]
    fn deref(&self) -> &[T] {
        let len = allocated(self.raw.as_ref(), self.len);
        // SAFETY: `len` elements of host memory filled when made (none,
        // from a dangling pointer, when not allocated), a valid `T` at any
        // bit pattern (`Plain`), borrowed with `self`, and no tenant carved
        // from them (`whole`): the carve takes `&mut self`, so none starts
        // while this borrow lives.
        unsafe { std::slice::from_raw_parts(self.base(), len) }
    }
}

impl<T: Plain> DerefMut for PinnedBuf<T> {
    #[track_caller]
    fn deref_mut(&mut self) -> &mut [T] {
        let len = allocated(self.raw.as_ref(), self.len);
        // SAFETY: as `deref`, and `&mut self` is the one user.
        unsafe { std::slice::from_raw_parts_mut(self.base(), len) }
    }
}

// SAFETY: `len()` elements of host memory, live while `self` is and
// reached only through it (`Deref` borrows it); one never dropped is never
// freed.
unsafe impl<T: Plain> HostSlice<T> for PinnedBuf<T> {
    fn as_ptr(&self) -> *const T {
        self.deref().as_ptr()
    }
    fn len(&self) -> usize {
        allocated(self.raw.as_ref(), self.len)
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T: Plain> HostSliceMut<T> for PinnedBuf<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.deref_mut().as_mut_ptr()
    }
}

/// Budgeted host heap: a boxed slice, so it has no `push`, `reserve` or
/// `extend`, and every page was touched when it was made. Only a
/// [`Budget`](crate::Budget) makes one. It is the host end of a copy too,
/// taken by value as a [`PinnedBuf`] is.
pub struct HeapSlab<T>(pub(crate) Box<[T]>);

impl<T> Deref for HeapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> DerefMut for HeapSlab<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

// SAFETY: the slice's elements, pageable memory, live while `self` is and
// reached only through it; one never dropped is never freed.
unsafe impl<T> HostSlice<T> for HeapSlab<T> {
    fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T> HostSliceMut<T> for HeapSlab<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }
}
