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
//! nothing here overlaps the unit tests. Add a lock before adding a second
//! test here: the harness runs a binary's tests in parallel.

use fb_rebuild::core::wwr::handle::DeviceHandleStream;
use fb_rebuild::core::wwr::runtime::{MemPool, Stream};
use wwr_sys::{wwr_mem_pool_attr::WWR_MEM_POOL_ATTR_USED_MEM_CURRENT, wwr_status};

#[track_caller]
fn ok(status: wwr_status) {
    assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
}

/// Needs a GPU. An allocation from the pool shows in its usage, and leaves
/// it when freed.
#[test]
fn allocations_come_from_the_pool() {
    let pool = MemPool::new(0);
    let s = Stream::new(0);
    let mut p = std::ptr::null_mut();
    // SAFETY: one pointer written through a valid local, freed below on the
    // same stream; the stream drains before each usage read.
    unsafe {
        ok(wwr_sys::wwr_malloc_from_pool_async(
            &mut p,
            4096,
            pool.get(),
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
