//! `wwr.extension.runtime:mem_pool`: the owning memory pool, and the
//! properties it is created from.

use std::ffi::c_void;
use std::ops::Deref;

use wwr_sys::{wwr_mem_pool, wwr_mem_pool_attr, wwr_status};

use super::create_with;
use crate::core::wwr::common::{ErrorPolicy, SourceLocation, gpu_check_at};
use crate::core::wwr::handle::{DeviceBoundHandle, DeviceHandle, HandleKind};

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

/// A memory pool's destroy. Every constructor creates through
/// [`MemPool::with_props`], which also sets the release threshold, so no
/// constructor takes the default create; the kind is not exported.
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

/// An owned memory pool on one device: `MemPoolWrapper<P_create, P_destroy,
/// P_device>`.
pub struct MemPool<PC, PD, PDev>(DeviceBoundHandle<MemPoolKind, PC, PD, PDev>)
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>;

impl<PC, PD, PDev> MemPool<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    /// A pool of pinned memory on `dev_idx`, releasing back to the OS above
    /// [`DEFAULT_RELEASE_THRESHOLD`].
    ///
    /// wwr's default create reads the current device back for the props,
    /// where its `(dev_idx, threshold)` constructor uses `dev_idx`; the two
    /// differ only when the select failed. Here both use `dev_idx`, so a bad
    /// index fails the pool's create too instead of landing it elsewhere.
    #[track_caller]
    pub fn new(dev_idx: i32, policy_create: PC, policy_destroy: PD, policy_device: PDev) -> Self {
        Self::with_threshold(
            dev_idx,
            DEFAULT_RELEASE_THRESHOLD,
            policy_create,
            policy_destroy,
            policy_device,
        )
    }

    /// [`MemPool::new`], with `release_threshold` bytes held before memory
    /// goes back to the OS.
    #[track_caller]
    pub fn with_threshold(
        dev_idx: i32,
        release_threshold: u64,
        policy_create: PC,
        policy_destroy: PD,
        policy_device: PDev,
    ) -> Self {
        Self::with_props(
            &MemPoolProps::device(dev_idx),
            release_threshold,
            policy_create,
            policy_destroy,
            policy_device,
        )
    }

    /// A pool from `props`, on the device `props.location.id` names.
    ///
    /// The release threshold is set after the create, through the create
    /// policy. If that fails the pool still exists and is still owned.
    #[track_caller]
    pub fn with_props(
        props: &MemPoolProps,
        release_threshold: u64,
        policy_create: PC,
        policy_destroy: PD,
        policy_device: PDev,
    ) -> Self {
        MemPool(DeviceBoundHandle::new_with(
            props.location.id,
            policy_create,
            policy_destroy,
            policy_device,
            |policy, location| {
                let pool = create_with(policy, location, |raw| {
                    // SAFETY: `raw` is valid for one write, and `props` is
                    // laid out as both backends read it.
                    unsafe {
                        wwr_sys::wwr_mem_pool_create(raw, (props as *const MemPoolProps).cast())
                    }
                })?;
                set_release_threshold(pool, release_threshold, policy, location);
                Some(pool)
            },
        ))
    }

    /// The pool's `attr`, which is a 64-bit value for every attribute the
    /// C ABI names, or the failure; `WWR_STATUS_INVALID_VALUE` if the pool
    /// owns nothing.
    pub fn attribute(&self, attr: wwr_mem_pool_attr) -> Result<u64, wwr_status> {
        let pool = self.0.get().ok_or(wwr_status::WWR_STATUS_INVALID_VALUE)?;
        let mut value = 0u64;
        // SAFETY: the borrow keeps the pool alive; each named attribute is
        // 8 bytes on both backends, and `value` is 8 bytes.
        let status = unsafe {
            wwr_sys::wwr_mem_pool_get_attribute(pool, attr, (&mut value as *mut u64).cast())
        };
        if status == wwr_status::WWR_STATUS_SUCCESS {
            Ok(value)
        } else {
            Err(status)
        }
    }
}

/// Set `pool`'s release threshold, reporting through `policy`.
///
/// The value is a `u64` because the attribute is 64-bit on both backends
/// and the runtime reads 8 bytes through the pointer: wwr's note on the
/// 4-byte variable that read past itself and capped the threshold at 4 GiB.
fn set_release_threshold<P: ErrorPolicy<Error = wwr_status>>(
    pool: wwr_mem_pool,
    mut threshold: u64,
    policy: &mut P,
    location: SourceLocation,
) {
    // SAFETY: `pool` is live; `threshold` is the attribute's 8 bytes.
    gpu_check_at(
        unsafe {
            wwr_sys::wwr_mem_pool_set_attribute(
                pool,
                wwr_mem_pool_attr::WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
        },
        policy,
        location,
    );
}

impl<PC, PD, PDev> Deref for MemPool<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    type Target = DeviceBoundHandle<MemPoolKind, PC, PD, PDev>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<PC, PD, PDev> DeviceHandle for MemPool<PC, PD, PDev>
where
    PC: ErrorPolicy<Error = wwr_status>,
    PD: ErrorPolicy<Error = wwr_status>,
    PDev: ErrorPolicy<Error = wwr_status>,
{
    fn dev_idx(&self) -> i32 {
        self.0.dev_idx()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::wwr::common::tests::Recording;
    use crate::core::wwr::runtime::tests::Abort;
    use wwr_mem_pool_attr::*;

    /// Needs a GPU. The threshold is read back whole, past 4 GiB.
    #[test]
    fn the_threshold_reaches_the_pool() {
        let pool = MemPool::new(0, Abort::new(), Abort::new(), Abort::new());
        assert_eq!(pool.dev_idx(), 0);
        assert_eq!(
            pool.attribute(WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD),
            Ok(DEFAULT_RELEASE_THRESHOLD)
        );
        let big = (5u64 << 30) + 3;
        let pool = MemPool::with_threshold(0, big, Abort::new(), Abort::new(), Abort::new());
        assert_eq!(pool.attribute(WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD), Ok(big));
    }

    // Allocating from a pool is tested in `tests/pool_allocation.rs`, in a
    // process of its own: see that file for why.

    /// Needs a GPU. A bad device fails the select and the create, each at
    /// the caller, and the pool owns nothing.
    #[test]
    fn a_bad_device_is_reported() {
        let line = line!() + 1;
        let pool = MemPool::new(
            -1,
            Recording::default(),
            Recording::default(),
            Recording::default(),
        );
        assert_eq!(pool.get(), None);
        assert_eq!(
            pool.device_policy().0,
            [(wwr_status::WWR_STATUS_INVALID_DEVICE, line)]
        );
        let created = &pool.create_policy().0;
        assert_eq!(created.len(), 1);
        assert_ne!(created[0].0, wwr_status::WWR_STATUS_SUCCESS);
        assert_eq!(created[0].1, line);
        assert_eq!(
            pool.attribute(WWR_MEM_POOL_ATTR_RELEASE_THRESHOLD),
            Err(wwr_status::WWR_STATUS_INVALID_VALUE)
        );
    }
}
