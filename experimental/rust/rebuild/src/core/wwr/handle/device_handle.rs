//! `wwr.extension.handle:device_handle`: what a handle knows about where
//! its work runs.
//!
//! | Trait | Adds |
//! |---|---|
//! | [`DeviceHandle`] | `dev_idx()` |
//! | [`DeviceHandleStream`] | `stream()` |
//!
//! In wwr this is the ladder a device buffer picks its allocation strategy
//! from, and its top rung, `device_handle_pool`, adds `pool()` for
//! `wwrMallocFromPoolAsync`. That rung has no counterpart: the arena
//! allocates one plain block per plan, never from a stream-ordered pool
//! (`docs/memory.md`, "The two kinds"). Here the two traits are what a
//! stream-bound handle needs of its stream owner.
//!
//! wwr's concepts are structural, so any type with the accessors qualifies;
//! a Rust type opts in by implementing the trait. wwr requires the accessors
//! `noexcept`; these return plain values and must not panic.

use wwr_sys::wwr_stream;

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
