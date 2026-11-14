//! wwr's `src/extension/memory_buffer` `:memset` (module
//! `wwr.extension.memory_buffer`): device memory filled on a stream.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `memset(B &buf, int value, wwrStream_t)`, the whole buffer, async | [`zero`], over a [`DeviceSliceMut`], the value always 0 |
//! | the offset-and-count overload, and `validate_memset` | nothing: a sub-slice is the caller's slice |
//! | the synchronous host overloads | nothing: host memory is a Rust slice |
//!
//! **Zero only.** Every caller zeroes a row before a kernel accumulates
//! into it, so the byte value is not a parameter; a caller that needs
//! another is the reason to add one.
//!
//! **Safe for the same reason [`blas::gemm`](crate::blas::gemm) is**: the
//! slice carries its length, and the memset writes exactly that many
//! elements' bytes, so there is no extent to check against it.

use crate::device_slice::DeviceSliceMut;
use crate::gpu_status;
use crate::handle::DeviceHandleStream;
use crate::runtime::Stream;
use fb_core_error_handling::Status;

/// Every byte of `dst` set to zero on `stream`: `wwrMemsetAsync` over the
/// slice's `len()` elements, ordered after the work queued on the stream
/// before it. For the floats and integers a row holds, that is the value
/// 0. An empty slice enqueues nothing.
///
/// # Errors
/// The runtime's refusal, as a `GpuRuntime` error naming the status.
#[track_caller]
pub fn zero<T>(dst: &mut impl DeviceSliceMut<T>, stream: &Stream) -> Status {
    if dst.is_empty() {
        return Ok(());
    }
    // No overflow: the slice promises `len()` elements of device memory,
    // so their bytes fit in an allocation's size.
    let bytes = dst.len() * size_of::<T>();
    let ptr = dst.as_mut_ptr();
    // SAFETY: `ptr` is `bytes` of device memory, the slice's whole extent,
    // that stays allocated and unused by anything else until the work
    // enqueued here finishes (DeviceSliceMut's promise); `stream` is live.
    gpu_status(unsafe { wwr_sys::wwr_memset_async(ptr.cast(), 0, bytes, stream.stream()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tests::{DeviceBytes, ok};

    /// Needs a GPU. A buffer filled with `0xFF` reads back zero over the
    /// slice's whole elements and nothing past them: 20 bytes as `f64` are
    /// two elements, so the last four bytes keep their `0xFF`. An empty
    /// slice is a successful no-op.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn zeroes_the_slices_elements_and_nothing_else() {
        let stream = Stream::new(0);
        let mut bytes = DeviceBytes::upload(&[0xFFu8; 20], &stream);
        zero::<f64>(&mut bytes, &stream).unwrap();
        let mut want = [0u8; 20];
        want[16..].fill(0xFF);
        assert_eq!(bytes.read(&stream), want);

        let mut empty = DeviceBytes::new(0);
        zero::<f64>(&mut empty, &stream).unwrap();
        ok(stream.sync());
    }
}
