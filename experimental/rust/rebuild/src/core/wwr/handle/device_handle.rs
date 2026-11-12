//! `wwr.extension.handle:device_handle`: the capability ladder a device
//! buffer picks its allocation strategy from.
//!
//! | Trait | Adds | Unlocks |
//! |---|---|---|
//! | [`DeviceHandle`] | `dev_idx()` | synchronous `wwr_malloc` / `wwr_free` |
//! | [`DeviceHandleStream`] | `stream()` | `wwr_malloc_async` / `wwr_free_async` |
//! | [`DeviceHandlePool`] | `pool()` | `wwr_malloc_from_pool_async` |
//!
//! wwr's concepts are structural, so any type with the accessors qualifies;
//! a Rust type opts in by implementing the trait. wwr requires the accessors
//! `noexcept`, because a buffer reads them on its free path; these return
//! plain values and must not panic, for the same reason.

use wwr_sys::{wwr_mem_pool, wwr_stream};

/// A handle that knows its device: `device_handle`.
pub trait DeviceHandle {
    /// The index of the device the handle belongs to (-1 if unknown).
    fn dev_idx(&self) -> i32;
}

/// A [`DeviceHandle`] with a work stream: `device_handle_stream`.
pub trait DeviceHandleStream: DeviceHandle {
    /// The stream work on this handle is queued on.
    fn stream(&self) -> wwr_stream;
}

/// A [`DeviceHandleStream`] with a memory pool: `device_handle_pool`.
pub trait DeviceHandlePool: DeviceHandleStream {
    /// The pool allocations on this handle are drawn from.
    fn pool(&self) -> wwr_mem_pool;
}
