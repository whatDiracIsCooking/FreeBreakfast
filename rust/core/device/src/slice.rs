//! Device memory with its length, runs cut from it, and the host end of a
//! copy.

use std::marker::PhantomData;

use contracts::requires;
use fb_core_log::le;

/// Device memory to read: `len()` elements of `T` from `as_ptr()`.
///
/// # Safety
/// An implementor promises that, for as long as `self` is borrowed and
/// until the device work enqueued during that borrow has finished:
///
/// - `as_ptr()` is valid device memory for `len()` elements of `T`, on the
///   device of the stream the slice is used with;
/// - nothing writes it, unless through the slice's own
///   [`DeviceSliceMut::as_mut_ptr`].
///
/// Both methods return the same values on every call.
pub unsafe trait DeviceSlice<T> {
    /// The first element, on the device.
    fn as_ptr(&self) -> *const T;

    /// The length, in elements.
    fn len(&self) -> usize;

    /// Whether it has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Device memory to write: [`DeviceSlice`], and `as_mut_ptr()` the same
/// address to write through.
///
/// # Safety
/// [`DeviceSlice`]'s promise, and: for that same span nothing else reads or
/// writes the memory, so a call handed this slice is its only user.
pub unsafe trait DeviceSliceMut<T>: DeviceSlice<T> {
    /// The first element, on the device, to write through.
    fn as_mut_ptr(&mut self) -> *mut T;
}

/// Host memory a copy reads: `len()` elements of `T` from `as_ptr()`,
/// handed to the copy by value.
///
/// Not `[T]`, and never a reference: an asynchronous copy from pinned
/// memory runs after the call returns, and a borrow that ends with the
/// call, or with a token dropped early, says nothing about a copy that has
/// not run. A copy takes its host end by value and keeps it in the
/// [`Pending`](crate::Pending) it returns until a sync of its stream has
/// covered the copy, so nothing reaches the bytes meanwhile. The memory's
/// owner (the budget's pinned and heap buffers and a lease's pinned
/// tenants) makes the promise below.
///
/// # Safety
/// An implementor promises that `as_ptr()` is valid host memory for
/// `len()` elements of `T` for as long as the value lives, reached by no
/// one but through the value; and that a value never dropped (a token
/// dropped unsynced leaks it) leaves the memory valid and reached by no one
/// until the device work enqueued while it lived has finished. A reference
/// cannot promise it: the borrow ends when the reference is gone, leaked or
/// not. Both methods return the same values on every call.
pub unsafe trait HostSlice<T> {
    /// The first element.
    fn as_ptr(&self) -> *const T;

    /// The length, in elements.
    fn len(&self) -> usize;

    /// Whether it has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Host memory a copy writes: [`HostSlice`], and `as_mut_ptr()` the same
/// address to write through.
///
/// # Safety
/// [`HostSlice`]'s promise.
pub unsafe trait HostSliceMut<T>: HostSlice<T> {
    /// The first element, to write through.
    fn as_mut_ptr(&mut self) -> *mut T;
}

/// A borrowed run of a [`DeviceSlice`], to read, as `&slice[a..b]` borrows
/// from a slice. Made by [`part`].
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

// SAFETY: a part is a pointer and a length into device memory, which no
// host thread dereferences; sharing or moving it between threads moves the
// address alone, as `&[T]` may be when `T: Sync`.
unsafe impl<T: Sync> Send for Part<'_, T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for Part<'_, T> {}

// SAFETY: a part is `len` elements inside the slice it was cut from, which
// promised `DeviceSlice` over all of them (`part`'s contract); that slice is
// borrowed shared for `'a`, so nothing writes it through its own
// `as_mut_ptr`, which needs it exclusively, while the part lives. The
// slice's promise lasts until the work enqueued during its borrow has
// finished, and the part's borrow lies inside it.
unsafe impl<T> DeviceSlice<T> for Part<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// A borrowed run of a [`DeviceSliceMut`], to write, as `&mut slice[a..b]`
/// borrows from a slice. Made by [`part_mut`], and cut again the same way.
///
/// The parent is borrowed exclusively while a part lives, so neither it
/// nor a second part of it can be used beside the part:
///
/// ```compile_fail,E0499
/// use fb_core_device::{DeviceSliceMut, part_mut};
///
/// fn both(a: &mut impl DeviceSliceMut<f64>) {
///     let head = part_mut(a, 0, 1);
///     let _again = part_mut(a, 1, 1);
///     drop(head);
/// }
/// ```
pub struct PartMut<'a, T> {
    ptr: *mut T,
    len: usize,
    life: PhantomData<&'a mut [T]>,
}

// SAFETY: as `Part`'s, for an exclusive borrow, as `&mut [T]` moves when
// `T: Send`.
unsafe impl<T: Send> Send for PartMut<'_, T> {}

// SAFETY: a part is `len` elements inside the slice it was cut from, which
// promised `DeviceSlice` over all of them (the contract of `part_mut` or
// `split_at_mut`); the slice is borrowed exclusively for `'a`, and the
// promise lasts until the work enqueued during its borrow has finished.
unsafe impl<T> DeviceSlice<T> for PartMut<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }

    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: the parent is borrowed exclusively for `'a`, so nothing uses it
// while the part lives, and the two parts of one split cover disjoint
// elements.
unsafe impl<T> DeviceSliceMut<T> for PartMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// Elements `[offset, offset + len)` of `of`, to read.
///
/// ```
/// use fb_core_device::{DeviceSlice, part};
///
/// /// Column `j` of a column-major `n × n` matrix.
/// fn column(a: &impl DeviceSlice<f64>, n: usize, j: usize) -> usize {
///     part(a, j * n, n).len()
/// }
/// ```
#[requires(
    le!(offset, of.len()) && le!(len, of.len() - offset),
    "the run is inside the slice"
)]
#[track_caller]
pub fn part<T>(of: &impl DeviceSlice<T>, offset: usize, len: usize) -> Part<'_, T> {
    Part {
        // `wrapping_add`, not `add`: device memory is no allocation the
        // host's provenance rules know of, and the contract keeps the run
        // inside the slice.
        ptr: of.as_ptr().wrapping_add(offset),
        len,
        life: PhantomData,
    }
}

/// Elements `[offset, offset + len)` of `of`, to write.
///
/// ```
/// use fb_core_device::{DeviceSlice, DeviceSliceMut, part_mut};
///
/// /// The trailing `rest` elements of `a`.
/// fn tail(a: &mut impl DeviceSliceMut<f64>, rest: usize) -> usize {
///     let n = a.len();
///     part_mut(a, n - rest, rest).len()
/// }
/// ```
#[requires(
    le!(offset, of.len()) && le!(len, of.len() - offset),
    "the run is inside the slice"
)]
#[track_caller]
pub fn part_mut<T>(of: &mut impl DeviceSliceMut<T>, offset: usize, len: usize) -> PartMut<'_, T> {
    PartMut {
        // `wrapping_add`, as in `part`.
        ptr: of.as_mut_ptr().wrapping_add(offset),
        len,
        life: PhantomData,
    }
}

/// `of` cut at element `mid` into two writable parts, `[0, mid)` and
/// `[mid, len)`, which do not meet, as `<[T]>::split_at_mut`: how the
/// crate's own recipes reach two panels of one matrix at once.
#[requires(le!(mid, of.len()), "mid is within the slice")]
#[track_caller]
pub(crate) fn split_at_mut<T>(
    of: &mut impl DeviceSliceMut<T>,
    mid: usize,
) -> (PartMut<'_, T>, PartMut<'_, T>) {
    let len = of.len();
    let ptr = of.as_mut_ptr();
    let cut = |ptr, len| PartMut {
        ptr,
        len,
        life: PhantomData,
    };
    // `wrapping_add`, as in `part`.
    (cut(ptr, mid), cut(ptr.wrapping_add(mid), len - mid))
}
