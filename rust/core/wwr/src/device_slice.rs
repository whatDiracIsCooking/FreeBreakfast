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

/// Part of a [`DeviceSliceMut`]: a run of its elements, borrowed from it
/// exclusively, as `&mut slice[a..b]` borrows from a slice. Made by
/// [`split_at_mut`], and split again the same way.
///
/// What it is for is a matrix whose blocks one library call reads and
/// another writes, the panels of a blocked factorization: each call takes
/// the part that holds its block, and parts never overlap, so the
/// non-overlap a call needs is decided from offsets (`docs/gpu_layer.md`,
/// "The safe wrapper").
pub struct DevicePart<'a, T> {
    ptr: *mut T,
    len: usize,
    marker: std::marker::PhantomData<&'a mut [T]>,
}

/// `slice` cut at element `mid` into two parts, `[0, mid)` and
/// `[mid, len)`, each its memory's only user while the parts live, as
/// `<[T]>::split_at_mut`.
///
/// ```
/// use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut, split_at_mut};
///
/// /// The trailing `rest` columns of an `n × n` matrix, apart from the rest.
/// fn trailing(a: &mut impl DeviceSliceMut<f64>, n: usize, rest: usize) -> usize {
///     let (_head, tail) = split_at_mut(a, n * (n - rest));
///     tail.len()
/// }
/// ```
///
/// # Panics
/// If `mid` is past `slice.len()`.
#[track_caller]
pub fn split_at_mut<T>(
    slice: &mut impl DeviceSliceMut<T>,
    mid: usize,
) -> (DevicePart<'_, T>, DevicePart<'_, T>) {
    let len = slice.len();
    assert!(
        mid <= len,
        "split_at_mut: mid is {mid}, past the slice's {len} elements"
    );
    let ptr = slice.as_mut_ptr();
    let part = |ptr, len| DevicePart {
        ptr,
        len,
        marker: std::marker::PhantomData,
    };
    // `wrapping_add`, not `add`: device memory is no allocation the host's
    // provenance rules know of, and `mid <= len` keeps it inside the slice.
    (part(ptr, mid), part(ptr.wrapping_add(mid), len - mid))
}

// SAFETY: a part is `len` elements inside the slice it was cut from, which
// promised `DeviceSlice` over all of them (`split_at_mut`'s `mid <= len`);
// that slice is borrowed exclusively for `'a`, so nothing else uses it while
// the part lives, and the two parts of one split cover disjoint elements.
// The slice's promise lasts until the work enqueued during its borrow has
// finished, and the part's borrow lies inside it.
unsafe impl<T> DeviceSlice<T> for DevicePart<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }

    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above: the part's elements are written by no one but through
// it, since the parent is borrowed exclusively and the other part holds
// other elements.
unsafe impl<T> DeviceSliceMut<T> for DevicePart<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host memory standing in for device memory: only the pointers and
    /// lengths are looked at, and nothing is dereferenced.
    struct Fake(Vec<f64>);

    // SAFETY: never handed to a call that dereferences it.
    unsafe impl DeviceSlice<f64> for Fake {
        fn as_ptr(&self) -> *const f64 {
            self.0.as_ptr()
        }
        fn len(&self) -> usize {
            self.0.len()
        }
    }

    // SAFETY: as above.
    unsafe impl DeviceSliceMut<f64> for Fake {
        fn as_mut_ptr(&mut self) -> *mut f64 {
            self.0.as_mut_ptr()
        }
    }

    /// The two parts tile the slice: the first at its start, the second at
    /// `mid`, their lengths summing to it; a part splits again inside
    /// itself; a split at either end leaves one part empty.
    #[test]
    fn the_parts_tile_the_slice() {
        let mut slice = Fake(vec![0.0; 10]);
        let base = slice.as_ptr() as usize;
        let at = |p: &DevicePart<'_, f64>| (p.as_ptr() as usize - base) / 8;
        let (mut head, tail) = split_at_mut(&mut slice, 4);
        assert_eq!((at(&head), head.len(), at(&tail), tail.len()), (0, 4, 4, 6));
        let (a, b) = split_at_mut(&mut head, 1);
        assert_eq!((at(&a), a.len(), at(&b), b.len()), (0, 1, 1, 3));
        let (empty, all) = split_at_mut(&mut slice, 0);
        assert_eq!((empty.len(), all.len()), (0, 10));
        let (all, empty) = split_at_mut(&mut slice, 10);
        assert_eq!((all.len(), empty.len(), at(&empty)), (10, 0, 10));
    }

    /// A cut past the end panics: a part would claim memory the slice never
    /// promised.
    #[test]
    #[should_panic(expected = "split_at_mut: mid is 11, past the slice's 10 elements")]
    fn a_cut_past_the_end_panics() {
        let mut slice = Fake(vec![0.0; 10]);
        let _ = split_at_mut(&mut slice, 11);
    }
}
