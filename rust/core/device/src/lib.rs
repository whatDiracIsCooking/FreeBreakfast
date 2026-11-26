//! `core::device`: the runtime half of the v0.0.4 layer, written over the
//! C ABI (`wwr-sys`) and nothing else of `rust/`'s wrapper layer
//! (`docs/gpu_layer.md`, "The v0.0.4 runtime").
//!
//! - [`ScopedDevice`]: a device made current on this thread, the one
//!   before restored on drop; [`device_count`] and [`memory`], the devices
//!   the runtime sees and a device's free and total memory.
//! - [`Stream`]: a stream on one device, whose methods enqueue the copies
//!   (host to device, device to host, device to device) and the fills, and
//!   whose [`Stream::sync`] counts and times itself ([`sync_meter`]).
//! - [`Event`]: a point on a stream that another stream, or the host,
//!   waits for.
//! - [`DeviceSlice`] and [`DeviceSliceMut`]: device memory with its length,
//!   which every call here takes in place of a pointer, and [`Part`] and
//!   [`PartMut`], runs of one ([`part`], [`part_mut`], [`split_at_mut`]).
//!   [`HostSlice`] and [`HostSliceMut`] are the host end of a copy.
//! - [`GpuResources`]: one GPU's stream with a BLAS and a dense-solver
//!   handle made on it and the handle workspace, made inside a `Mutex` and
//!   reached only through its guard; its checked BLAS and solver calls, and
//!   the tree's [`GpuResources::trsm`] and [`GpuResources::cholesky`]
//!   recipes over them, on the guard's one stream.
//!
//! **Each call makes its stream's device current.** A stream and an event
//! remember their device, and every method holds a [`ScopedDevice`] on it
//! for the call, so a caller on any thread, with any device current, enqueues
//! on the right one.
//!
//! **A create aborts; everything else is a value.** Making a stream, an
//! event, a GPU's handles or a device current aborts at the caller's line when the runtime
//! refuses, the C++ tree's rule for a create (`docs/diagnostics.md`,
//! "Allocation is a planning-time error"); an enqueue, a sync and a query
//! return the runtime's failure as a [`Status`] or a
//! [`Result`]. A failed destroy is logged at
//! error (`docs/diagnostics.md`, "`Drop` cannot fail").

mod gpu;
#[cfg(test)]
mod gpu_tests;
mod slice;
mod stream;
#[cfg(test)]
mod tests;

use std::marker::PhantomData;
use std::panic::Location;

use fb_core_error_handling::{Error, Result, Status, check_or_abort};
use fb_core_log::{self as log, LogLevel};
use wwr_sys::wwr_status;

pub use gpu::{CHOLESKY_TILE, GpuResources, HANDLE_WORKSPACE_BYTES};
pub use slice::{
    DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut, Part, PartMut, part, part_mut,
    split_at_mut,
};
pub use stream::{Event, Stream, sync_meter};

/// A runtime call's status as a [`Status`], located at the caller: a
/// `GpuRuntime` error naming the status. Out of memory is no exception: an
/// `Allocation` error is a value only at admission (`docs/diagnostics.md`,
/// "Allocation is a planning-time error").
#[track_caller]
fn runtime_status(code: wwr_status) -> Status {
    match code {
        wwr_status::WWR_STATUS_SUCCESS => Ok(()),
        _ => Err(Error::gpu(format!("{code:?}"))),
    }
}

/// Log a failed step of a destroy at error, at `location`, and carry on:
/// what a `Drop` does with one, the crate's one error line.
fn log_destroy<T>(what: &str, result: &Result<T>, location: &Location<'_>) {
    if let Err(e) = result {
        log::log_at(LogLevel::Error, location, format_args!("{what}: {e}"));
    }
}

/// The devices the runtime sees.
///
/// # Errors
/// The runtime's failure, `WWR_STATUS_NO_DEVICE` on a host with none.
#[track_caller]
pub fn device_count() -> Result<i32> {
    let mut n = 0;
    // SAFETY: a plain runtime call writing one local.
    runtime_status(unsafe { wwr_sys::wwr_get_device_count(&mut n) })?;
    Ok(n)
}

/// Device `physical`'s free and total memory, in bytes, read with it
/// current: what the budget's default device cap is taken from, before
/// anything runs.
///
/// # Errors
/// The runtime's failure.
#[track_caller]
pub fn memory(physical: i32) -> Result<(u64, u64)> {
    let _current = ScopedDevice::new(physical);
    let (mut free, mut total) = (0usize, 0usize);
    #[allow(
        clippy::disallowed_methods,
        reason = "free memory read for the budget's cap, before anything runs: admission, not a width at run time"
    )]
    // SAFETY: a plain runtime call writing two locals.
    let status = unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) };
    runtime_status(status)?;
    Ok((free as u64, total as u64))
}

/// A device made current on this thread, and the one before it restored
/// when it drops. Launches and library calls run on the calling thread's
/// current device, and a step runs on whichever worker is free, so every
/// call of this crate that enqueues holds one on its stream's device.
///
/// Not `Send`: restoring on another thread would set that thread's device
/// and leave this one's switched.
///
/// ```compile_fail,E0277
/// fn on_another_thread<T: Send>(_: T) {}
/// on_another_thread(fb_core_device::ScopedDevice::new(0));
/// ```
pub struct ScopedDevice {
    before: i32,
    not_send: PhantomData<*const ()>,
}

impl ScopedDevice {
    /// Make `physical` current, aborting at the caller's line if the
    /// runtime refuses: a device the run was not set up on.
    #[track_caller]
    pub fn new(physical: i32) -> ScopedDevice {
        check_or_abort(Self::try_new(physical))
    }

    /// [`ScopedDevice::new`], the refusal as an `Err`; nothing was switched
    /// when it fails. A `Drop` uses it, which must not abort.
    #[track_caller]
    fn try_new(physical: i32) -> Result<ScopedDevice> {
        let mut before = 0;
        // SAFETY: a plain runtime call writing one local.
        runtime_status(unsafe { wwr_sys::wwr_get_device(&mut before) })?;
        // SAFETY: a plain runtime call; a bad index is a status.
        runtime_status(unsafe { wwr_sys::wwr_set_device(physical) })?;
        Ok(ScopedDevice {
            before,
            not_send: PhantomData,
        })
    }
}

impl Drop for ScopedDevice {
    /// Restore the device that was current, aborting if the runtime
    /// refuses: a thread left on the wrong device sends every later launch
    /// to the wrong card, which is worse than stopping.
    fn drop(&mut self) {
        // SAFETY: a plain runtime call, to a device that was current.
        check_or_abort(runtime_status(unsafe {
            wwr_sys::wwr_set_device(self.before)
        }));
    }
}
