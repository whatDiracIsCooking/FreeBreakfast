//! wwr's `src/extension/memory_buffer` `:copy` (module
//! `wwr.extension.memory_buffer`): device memory copied to device memory on
//! a stream, between parts of rows.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `copy(dst, dst_offset, src, src_offset, count, wwrStream_t)`, async, and `validate_copy` | [`d2d`], over a [`DeviceSliceMut`] and a [`DeviceSlice`], the offsets the caller's parts and the count their common length, checked |
//! | the synchronous host overloads | nothing: host memory is a Rust slice |
//!
//! **Parts, not rows.** A whole row of a plan is copied by the arena's
//! `memcpy_d2d`, which names it by its id. This is for a run of elements
//! inside a row, a column of a ring buffer, which only a
//! [`split_at_mut`](crate::device_slice::split_at_mut) part names.
//!
//! **Safe for the same reason [`blas::gemm`](crate::blas::gemm) is**: each
//! side carries its length, the two are asserted equal, and the borrow
//! checker keeps the destination apart from the source.

use crate::device_slice::{DeviceSlice, DeviceSliceMut};
use crate::gpu_status;
use crate::handle::DeviceHandleStream;
use crate::runtime::Stream;
use fb_core_error_handling::Status;
use wwr_sys::wwr_memcpy_kind;

/// `src` copied into `dst` on `stream`, element for element: an
/// asynchronous device-to-device `wwrMemcpyAsync`, ordered after the work
/// queued on the stream before it. An empty pair enqueues nothing.
///
/// # Errors
/// The runtime's refusal, as a `GpuRuntime` error naming the status.
///
/// # Panics
/// If the two lengths differ, before anything is enqueued.
#[track_caller]
pub fn d2d<T>(
    dst: &mut impl DeviceSliceMut<T>,
    src: &impl DeviceSlice<T>,
    stream: &Stream,
) -> Status {
    assert_eq!(
        dst.len(),
        src.len(),
        "copy::d2d: the destination holds {} elements, the source {}",
        dst.len(),
        src.len()
    );
    if dst.is_empty() {
        return Ok(());
    }
    // No overflow: the slice promises `len()` elements of device memory,
    // so their bytes fit in an allocation's size.
    let bytes = dst.len() * size_of::<T>();
    let to = dst.as_mut_ptr();
    // SAFETY: `to` and `src` are each `bytes` of device memory, their
    // slices' whole extents, that stay allocated until the work enqueued
    // here finishes; `dst` has no other user, so it is apart from `src`
    // (DeviceSliceMut's and DeviceSlice's promises); `stream` is live.
    gpu_status(unsafe {
        wwr_sys::wwr_memcpy_async(
            to.cast(),
            src.as_ptr().cast(),
            bytes,
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE,
            stream.stream(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_slice::split_at_mut;
    use crate::runtime::tests::DeviceBytes;

    /// Needs a GPU. The second half of one buffer copied into the first
    /// half of another, as `u16`s: the destination's other half keeps its
    /// bytes. An empty pair is a successful no-op.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn copies_a_part_into_a_part_and_nothing_else() {
        let stream = Stream::new(0);
        let mut src = DeviceBytes::upload(&[1u8, 2, 3, 4, 5, 6, 7, 8], &stream);
        let mut dst = DeviceBytes::upload(&[0xFFu8; 8], &stream);
        {
            let (_, tail) = split_at_mut::<u16>(&mut src, 2);
            let (mut head, _) = split_at_mut::<u16>(&mut dst, 2);
            d2d(&mut head, &tail, &stream).unwrap();
        }
        assert_eq!(dst.read(&stream), [5, 6, 7, 8, 0xFF, 0xFF, 0xFF, 0xFF]);

        let (mut empty, nothing) = (DeviceBytes::new(0), DeviceBytes::new(0));
        d2d::<u16>(&mut empty, &nothing, &stream).unwrap();
    }

    /// Needs a GPU. Parts of different lengths panic before anything is
    /// enqueued.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    #[should_panic(expected = "copy::d2d: the destination holds 2 elements, the source 1")]
    fn parts_of_different_lengths_panic() {
        let stream = Stream::new(0);
        let (mut dst, src) = (DeviceBytes::new(16), DeviceBytes::new(8));
        let _ = d2d::<f64>(&mut dst, &src, &stream);
    }
}
