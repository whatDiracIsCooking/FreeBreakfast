//! The raw calls behind [`Device`] and [`Pinned`], through `wwr-sys`: the
//! `device` feature.

use std::ffi::c_void;
use std::panic::Location;

use fb_core_error_handling::{Error, Status};
use fb_core_wwr::common::{ErrorCode, ScopedDeviceIndex};
use fb_core_wwr::{gpu_log_error_at, gpu_status_at};
use wwr_sys::wwr_status;

use super::{Device, Pinned, mem_available, sealed};

/// `cudaHostAllocPortable` / `hipHostMallocPortable`: both `0x1`.
const HOST_ALLOC_PORTABLE: u32 = 0x1;

/// A failed allocation's status, named and described.
fn alloc_result(status: wwr_status) -> Result<(), String> {
    if status == wwr_status::WWR_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{} ({})", status.name(), status.description()))
    }
}

impl sealed::Sealed for Device {
    const ALWAYS_FILL: bool = false;
    type Current = ScopedDeviceIndex;

    fn current(dev_idx: i32) -> ScopedDeviceIndex {
        ScopedDeviceIndex::new(dev_idx)
    }

    fn try_current(dev_idx: i32) -> Result<ScopedDeviceIndex, Error> {
        ScopedDeviceIndex::try_new(dev_idx)
    }

    unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> Result<(), String> {
        // SAFETY: the caller's pointer is valid for one write.
        alloc_result(unsafe { wwr_sys::wwr_malloc(ptr.cast::<*mut c_void>(), bytes) })
    }

    unsafe fn free(ptr: *mut u8, _: usize, location: &'static Location<'static>) {
        // SAFETY: the caller's allocation, freed once.
        gpu_log_error_at(unsafe { wwr_sys::wwr_free(ptr.cast()) }, location);
    }

    unsafe fn fill(ptr: *mut u8, bytes: usize, location: &'static Location<'static>) -> Status {
        // SAFETY: the caller's allocation of at least `bytes`.
        let status = unsafe { wwr_sys::wwr_memset(ptr.cast(), 0xFF, bytes) };
        gpu_status_at(status, location)?;
        // The memset runs on the legacy stream, which the run's
        // non-blocking streams do not wait for: finish it here.
        // SAFETY: a plain runtime call.
        gpu_status_at(unsafe { wwr_sys::wwr_device_synchronize() }, location)
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "a failed block allocation names the free memory (docs/memory.md, \"The two kinds\"); nothing is sized from it"
    )]
    fn free_memory() -> String {
        let (mut free, mut total) = (0, 0);
        // SAFETY: a plain runtime call writing two locals.
        let status = unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) };
        if status == wwr_status::WWR_STATUS_SUCCESS {
            format!("{free} bytes free of {total}")
        } else {
            "free memory unknown".to_owned()
        }
    }
}

impl sealed::Sealed for Pinned {
    const ALWAYS_FILL: bool = true;
    type Current = ScopedDeviceIndex;

    fn current(dev_idx: i32) -> ScopedDeviceIndex {
        ScopedDeviceIndex::new(dev_idx)
    }

    fn try_current(dev_idx: i32) -> Result<ScopedDeviceIndex, Error> {
        ScopedDeviceIndex::try_new(dev_idx)
    }

    unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> Result<(), String> {
        // SAFETY: the caller's pointer is valid for one write.
        alloc_result(unsafe { wwr_sys::wwr_host_alloc(ptr.cast(), bytes, HOST_ALLOC_PORTABLE) })
    }

    unsafe fn free(ptr: *mut u8, _: usize, location: &'static Location<'static>) {
        // SAFETY: the caller's allocation, freed once.
        gpu_log_error_at(unsafe { wwr_sys::wwr_free_host(ptr.cast()) }, location);
    }

    unsafe fn fill(ptr: *mut u8, bytes: usize, _: &'static Location<'static>) -> Status {
        // SAFETY: the caller's host allocation of at least `bytes`.
        unsafe { std::ptr::write_bytes(ptr, 0xFF, bytes) };
        Ok(())
    }

    fn free_memory() -> String {
        mem_available()
    }
}
