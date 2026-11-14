//! Device memory with its length: what a safe call into a GPU library takes
//! in place of a raw pointer, so it can check the extent it was asked to
//! touch against the memory it was handed.
//!
//! No wwr counterpart: the C++ tree hands a library call raw pointers, as
//! wwr's wrappers take them, and trusts each extent. Here the check is what
//! lets [`blas::gemm`](crate::blas::gemm) be safe, and the crate that owns
//! the memory (the arena's `DeviceRef` and `DeviceMut`) is the one that
//! promises, once, in an `unsafe impl`, that a pointer and a length really
//! are device memory. The traits live here rather than beside those types
//! because the arena's device-free build links no wwr, and a call that
//! takes them does.

/// Device memory to read: `len()` elements of `T` from `as_ptr()`.
///
/// # Safety
/// An implementor promises that, for as long as `self` is borrowed and
/// until the device work enqueued on a stream during that borrow has
/// finished:
///
/// - `as_ptr()` is valid device memory for `len()` elements of `T`, on the
///   device of the handle the slice is used with;
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
