//! `wwr.extension.runtime:mem_pool`: the owning memory pool, and the
//! properties it is created from.

use std::ffi::c_void;
use std::ops::Deref;
use std::panic::Location;

use wwr_sys::{wwr_mem_pool, wwr_mem_pool_attr, wwr_status};

use crate::common::SourceLocation;
use crate::gpu_status_at;
use crate::handle::{DeviceBoundHandle, DeviceHandle, HandleKind};
use fb_core_error_handling::{Result, check_or_abort};

/// The release threshold a pool is made with unless told otherwise: 1 MiB,
/// wwr's `default_threshold`.
pub const DEFAULT_RELEASE_THRESHOLD: u64 = 1 << 20;

/// `cudaMemAllocationTypePinned` / `hipMemAllocationTypePinned`.
pub const ALLOCATION_PINNED: i32 = 0x1;
/// `cudaMemHandleTypeNone` / `hipMemHandleTypeNone`.
pub const HANDLE_TYPE_NONE: i32 = 0x0;
/// `cudaMemLocationTypeDevice` / `hipMemLocationTypeDevice`.
pub const LOCATION_DEVICE: i32 = 0x1;

/// `cudaMemLocation` / `hipMemLocation`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemLocation {
    pub kind: i32,
    pub id: i32,
}

/// `cudaMemPoolProps` / `hipMemPoolProps`, which the C ABI's
/// `wwr_mem_pool_create` takes as a bare `const void*`.
///
/// The ABI carries no layout for it, so this mirrors the vendors' own. The
/// two agree: 88 bytes, 8-aligned, `allocType` at 0, `handleTypes` at 4,
/// `location` at 8, `win32SecurityAttributes` at 16, `maxSize` at 24 and
/// zeroed reserved bytes after, measured against CUDA 13's
/// `driver_types.h` and ROCm's `hip_runtime_api.h`. CUDA spends two of
/// those reserved bytes on a `usage` field; zero is its default. The enum
/// fields are plain `i32`s holding the vendor values, which the two share.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemPoolProps {
    pub alloc_type: i32,
    pub handle_types: i32,
    pub location: MemLocation,
    pub win32_security_attributes: *mut c_void,
    pub max_size: usize,
    pub reserved: [u8; 56],
}

const _: () = assert!(std::mem::size_of::<MemPoolProps>() == 88);
const _: () = assert!(std::mem::align_of::<MemPoolProps>() == 8);

impl MemPoolProps {
    /// Pinned device memory on `dev_idx`, not shareable, no size cap: what
    /// wwr's `make_default_props` fills.
    pub fn device(dev_idx: i32) -> Self {
        MemPoolProps {
            alloc_type: ALLOCATION_PINNED,
            handle_types: HANDLE_TYPE_NONE,
            location: MemLocation {
                kind: LOCATION_DEVICE,
                id: dev_idx,
            },
            win32_security_attributes: std::ptr::null_mut(),
            max_size: 0,
            reserved: [0; 56],
        }
    }
}

/// A memory pool's destroy. Every constructor creates from a
/// [`MemPoolProps`] and then sets the release threshold, so none takes the
/// default create; the kind is not exported.
pub struct MemPoolKind;

impl HandleKind for MemPoolKind {
    type Raw = wwr_mem_pool;
    type Error = wwr_status;

    /// Unreachable: no constructor takes the default create path.
    unsafe fn create(_: *mut wwr_mem_pool) -> wwr_status {
        wwr_status::WWR_STATUS_NOT_SUPPORTED
    }

    unsafe fn destroy(raw: wwr_mem_pool) -> wwr_status {
        // SAFETY: `raw` is a live pool, destroyed once.
        unsafe { wwr_sys::wwr_mem_pool_destroy(raw) }
    }
}

/// An owned memory pool on one device: `MemPoolWrapper`. Every constructor
/// aborts on failure ([`super`] has why).
pub struct MemPool(DeviceBoundHandle<MemPoolKind>);

impl MemPool {
    /// A pool of pinned memory on `dev_idx`, releasing back to the OS above
    /// [`DEFAULT_RELEASE_THRESHOLD`].
    ///
    /// wwr's default create reads the current device back for the props,
    /// where its `(dev_idx, threshold)` constructor uses `dev_idx`; the two
    /// differ only when the select failed. Here both use `dev_idx`.
    #[track_caller]
    pub fn new(dev_idx: i32) -> Self {
        Self::with_threshold(dev_idx, DEFAULT_RELEASE_THRESHOLD)
    }

    /// [`MemPool::new`], with `release_threshold` bytes held before memory
    /// goes back to the OS.
    #[track_caller]
    pub fn with_threshold(dev_idx: i32, release_threshold: u64) -> Self {
        Self::with_props(&MemPoolProps::device(dev_idx), release_threshold)
    }

    /// A pool from `props`, on the device `props.location.id` names.
    #[track_caller]
    pub fn with_props(props: &MemPoolProps, release_threshold: u64) -> Self {
        check_or_abort(Self::try_create(
            props,
            release_threshold,
            Location::caller(),
        ))
    }

    /// The fallible constructor every public one aborts over.
    ///
    /// The release threshold is set after the create. If that fails, the
    /// pool is destroyed and the failure returned: a pool is never handed
    /// out with a threshold other than the one asked for.
    fn try_create(
        props: &MemPoolProps,
        release_threshold: u64,
        location: SourceLocation,
    ) -> Result<Self> {
        let pool = DeviceBoundHandle::try_new_at(
            props.location.id,
            // SAFETY: `raw` is valid for one write, and `props` is laid out
            // as both backends read it.
            |raw| unsafe {
                wwr_sys::wwr_mem_pool_create(raw, (props as *const MemPoolProps).cast())
            },
            location,
        )?;
        let mut threshold = release_threshold;
        // SAFETY: the pool is live; `threshold` is the attribute's 8 bytes.
        // A `u64` because the attribute is 64-bit on both backends and the
        // runtime reads 8 bytes through the pointer: wwr's note on the
        // 4-byte variable that read past itself and capped the threshold
        // at 4 GiB.
        gpu_status_at(
            unsafe {
                wwr_sys::wwr_mem_pool_set_attribute(
                    pool.get(),
                    wwr_mem_pool_attr::WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD,
                    (&mut threshold as *mut u64).cast(),
                )
            },
            location,
        )?;
        Ok(MemPool(pool))
    }

    /// The pool's `attr`, which is a 64-bit value for every attribute the
    /// C ABI names, or the failure.
    pub fn attribute(&self, attr: wwr_mem_pool_attr) -> std::result::Result<u64, wwr_status> {
        let mut value = 0u64;
        // SAFETY: the borrow keeps the pool alive; each named attribute is
        // 8 bytes on both backends, and `value` is 8 bytes.
        let status = unsafe {
            wwr_sys::wwr_mem_pool_get_attribute(self.0.get(), attr, (&mut value as *mut u64).cast())
        };
        if status == wwr_status::WWR_STATUS_SUCCESS {
            Ok(value)
        } else {
            Err(status)
        }
    }
}

impl Deref for MemPool {
    type Target = DeviceBoundHandle<MemPoolKind>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DeviceHandle for MemPool {
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::tests::assert_gpu_error_at;
    use wwr_mem_pool_attr::*;

    /// Needs a GPU. The threshold is read back whole, past 4 GiB.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn the_threshold_reaches_the_pool() {
        let pool = MemPool::new(0);
        assert_eq!(pool.dev_idx(), 0);
        assert_eq!(
            pool.attribute(WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD),
            Ok(DEFAULT_RELEASE_THRESHOLD)
        );
        let big = (5u64 << 30) + 3;
        let pool = MemPool::with_threshold(0, big);
        assert_eq!(pool.attribute(WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD), Ok(big));
    }

    // Allocating from a pool is tested in `tests/pool_allocation.rs`, in a
    // process of its own: see that file for why.

    /// Needs a GPU. A bad device fails the select, at the caller, and no
    /// pool is made.
    #[test]
    #[ignore = "tier 4: needs a GPU"]
    fn a_bad_device_is_an_error_at_the_caller() {
        let line = line!() + 1;
        let error = MemPool::try_create(&MemPoolProps::device(-1), 0, Location::caller())
            .err()
            .unwrap();
        assert_gpu_error_at(error, "WWR_STATUS_INVALID_DEVICE", file!(), line);
    }
}
