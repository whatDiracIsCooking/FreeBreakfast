//! Allocating from a memory pool, in a process of its own.
//!
//! **Why not beside the unit tests:** on the gfx1200 under ROCm 7.2.4,
//! `hipMallocFromPoolAsync` or `hipFreeAsync` on one thread, while other
//! threads synchronize, destroy or wait on their own streams and events,
//! intermittently deadlocks inside `libamdhip64`. Every thread ends up
//! blocked on a HIP-internal mutex. No frame of this crate holds a lock,
//! and the C ABI forwards each call straight to HIP. In the unit-test
//! binary it hung about one run in two; with the two pool-allocating tests
//! skipped, it ran ten times clean. A bare-HIP C reproducer of the same
//! calls on four threads has not hung yet, so the exact trigger is unknown.
//! It has not been seen on CUDA.
//!
//! Cargo runs each test binary as its own process, one after another, so
//! nothing here overlaps the unit tests; the lock keeps these two from
//! overlapping each other.

use std::sync::Mutex;

use fb_rebuild::core::wwr::common::kit::AbortPolicy;
use fb_rebuild::core::wwr::handle::{DeviceHandlePool, DeviceHandleStream};
use fb_rebuild::core::wwr::runtime::{MemPool, Stream, kit};
use wwr_sys::{wwr_mem_pool_attr::WWR_MEM_POOL_ATTR_USED_MEM_CURRENT, wwr_status};

type Abort = AbortPolicy<wwr_status>;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

#[track_caller]
fn ok(status: wwr_status) {
    assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
}

/// Needs a GPU. An allocation from the pool shows in its usage, and leaves
/// it when freed.
#[test]
fn allocations_come_from_the_pool() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let pool = MemPool::new(0, Abort::new(), Abort::new(), Abort::new());
    let s = Stream::new(0, Abort::new(), Abort::new(), Abort::new());
    let mut p = std::ptr::null_mut();
    // SAFETY: one pointer written through a valid local, freed below on the
    // same stream; the stream drains before each usage read.
    unsafe {
        ok(wwr_sys::wwr_malloc_from_pool_async(
            &mut p,
            4096,
            pool.get().unwrap(),
            s.stream(),
        ));
        ok(wwr_sys::wwr_stream_synchronize(s.stream()));
    }
    assert!(pool.attribute(WWR_MEM_POOL_ATTR_USED_MEM_CURRENT).unwrap() >= 4096);
    // SAFETY: as above.
    unsafe {
        ok(wwr_sys::wwr_free_async(p, s.stream()));
        ok(wwr_sys::wwr_stream_synchronize(s.stream()));
    }
    assert_eq!(pool.attribute(WWR_MEM_POOL_ATTR_USED_MEM_CURRENT), Ok(0));
}

/// Needs a GPU. A kit device handle allocates from its pool on its stream.
#[test]
fn the_kit_handle_allocates_from_its_pool() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let h = kit::DeviceHandle::<Abort, Abort, Abort>::new(0);
    let mut p = std::ptr::null_mut();
    // SAFETY: one pointer written through a valid local, freed on the same
    // stream before the handle drops.
    unsafe {
        ok(wwr_sys::wwr_malloc_from_pool_async(
            &mut p,
            256,
            h.pool(),
            h.stream(),
        ));
        ok(wwr_sys::wwr_stream_synchronize(h.stream()));
    }
    assert!(
        h.owned_pool()
            .attribute(WWR_MEM_POOL_ATTR_USED_MEM_CURRENT)
            .unwrap()
            >= 256
    );
    // SAFETY: as above.
    unsafe {
        ok(wwr_sys::wwr_free_async(p, h.stream()));
        ok(wwr_sys::wwr_stream_synchronize(h.stream()));
    }
}
