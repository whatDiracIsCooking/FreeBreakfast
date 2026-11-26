//! What a [`Budget`](crate::Budget) hands back for memory an `Output`
//! owns: a device buffer, a pinned buffer, a heap slab. Fixed size, freed
//! with the `Output`; only a `Budget` makes one.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use fb_core_device::{DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut};

use crate::raw::RawBuf;

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
fn allocated(raw: Option<&RawBuf>, len: usize) -> usize {
    raw.map_or(0, |_| len)
}

/// Device memory a step's `Output` owns. Empty, of length 0, under a dry
/// or refused budget ([`DeviceBuf::is_allocated`]).
pub struct DeviceBuf<T> {
    pub(crate) raw: Option<RawBuf>,
    pub(crate) len: usize,
    pub(crate) of: PhantomData<T>,
}

impl<T: Plain> DeviceBuf<T> {
    /// Whether the buffer was really allocated.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        self.raw.is_some()
    }

    /// The bytes as host memory, under [`Backing::Host`](crate::Backing)
    /// only: how the device-free tests read and write a "device" buffer on
    /// no gpu.
    pub fn host(&mut self) -> Option<&mut [T]> {
        let raw = self.raw.as_ref().filter(|r| r.is_host())?;
        // SAFETY: `len` elements of zeroed host memory, a valid `T` at any
        // bit pattern (`Plain`), and `&mut self` is the one user.
        Some(unsafe { std::slice::from_raw_parts_mut(raw.ptr.cast(), self.len) })
    }
}

// SAFETY: `len()` elements of device memory (0 when not allocated), live
// while `self` is; only `as_mut_ptr`, through `&mut self`, writes it.
unsafe impl<T> DeviceSlice<T> for DeviceBuf<T> {
    fn as_ptr(&self) -> *const T {
        self.raw
            .as_ref()
            .map_or(std::ptr::null(), |r| r.ptr.cast_const().cast())
    }
    fn len(&self) -> usize {
        allocated(self.raw.as_ref(), self.len)
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T> DeviceSliceMut<T> for DeviceBuf<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.raw
            .as_ref()
            .map_or(std::ptr::null_mut(), |r| r.ptr.cast())
    }
}

/// Page-locked host memory a step's `Output` owns, staged for one gpu,
/// filled with `0xFF` when made. Read and written as a slice, and the host
/// end of a copy ([`HostSlice`]). Empty under a dry or refused budget.
///
/// The copy's promise, that the host does not touch the bytes until the
/// copy has run, is the layer's rule and not this type's: a step's copies
/// are done at its step-end sync, and within a step the host waits on the
/// copy's event before it touches the bytes again
/// (`../experimental/rust/v0.0.4/README.md`, "Step scratch").
pub struct PinnedBuf<T> {
    pub(crate) raw: Option<RawBuf>,
    pub(crate) len: usize,
    pub(crate) of: PhantomData<T>,
}

impl<T: Plain> PinnedBuf<T> {
    /// Whether the buffer was really allocated.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        self.raw.is_some()
    }
}

impl<T: Plain> Deref for PinnedBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.raw {
            // SAFETY: `len` elements of host memory filled when made, a
            // valid `T` at any bit pattern (`Plain`), borrowed with `self`.
            Some(raw) => unsafe { std::slice::from_raw_parts(raw.ptr.cast(), self.len) },
            None => &[],
        }
    }
}

impl<T: Plain> DerefMut for PinnedBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        match &self.raw {
            // SAFETY: as `deref`, and `&mut self` is the one user.
            Some(raw) => unsafe { std::slice::from_raw_parts_mut(raw.ptr.cast(), self.len) },
            None => &mut [],
        }
    }
}

// SAFETY: `len()` elements of host memory, live while `self` is. That the
// host leaves them alone until an enqueued copy has run is the layer's
// sync rule (the type's doc), not the borrow's.
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
/// under [`PinnedBuf`]'s rule.
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

// SAFETY: the slice's elements, live while `self` is; pageable memory, and
// the host leaves it alone until an enqueued copy has run, under
// `PinnedBuf`'s rule.
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
