//! The allocation record (`docs/memory.md`, "The budget"): a global
//! allocator over the system's that records, and never refuses, what
//! the plain heap is asked for. A test binary installs it; the step
//! wrapper watches each `run` and fails a step that made one allocation
//! past [`RECORD_THRESHOLD`], and an execute's peak heap growth is the
//! budget table's unbudgeted row.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

/// One allocation inside a watched `run` past this many bytes, 1 MiB, is
/// large host data, which belongs in a `HeapSlab` on the bus: small host
/// scratch, a `Vec` that lives within a `run`, stays under it.
pub const RECORD_THRESHOLD: usize = 1 << 20;

/// Whether any allocation came through the record: a binary that does
/// not install it measures nothing.
static INSTALLED: AtomicBool = AtomicBool::new(false);
/// The bytes the record has handed out and not had back.
static LIVE: AtomicUsize = AtomicUsize::new(0);
/// The most `LIVE` has been since the last [`growth`] began.
static PEAK: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    /// Whether this thread is inside a [`AllocationRecord::watch`].
    static WATCHED: Cell<bool> = const { Cell::new(false) };
    /// The largest single allocation this thread made while watched.
    static LARGEST: Cell<usize> = const { Cell::new(0) };
}

/// The allocation record: the system allocator, with each allocation's
/// size noted. It records and never panics, since unwinding out of a
/// global allocator is undefined behaviour, and it sees Rust's heap
/// alone, not a C library's `malloc`. Installed by a binary:
///
/// ```
/// #[global_allocator]
/// static RECORD: fb_core_budget::AllocationRecord = fb_core_budget::AllocationRecord;
///
/// let ((), largest) = fb_core_budget::AllocationRecord::watch(|| {
///     drop(std::hint::black_box(vec![0u8; 2 << 20]));
/// });
/// assert_eq!(largest, Some(2 << 20));
/// let ((), largest) = fb_core_budget::AllocationRecord::watch(|| {
///     drop(std::hint::black_box(vec![0u8; 1024]));
/// });
/// assert_eq!(largest, None);
/// ```
pub struct AllocationRecord;

impl AllocationRecord {
    /// Run `f` with this thread watched: `f`'s value, and the largest
    /// single allocation past [`RECORD_THRESHOLD`] it made on this thread,
    /// if any (a growing `Vec`'s reallocation is one allocation of its new
    /// size). Another thread's allocations, a pool's or a fan-out's, are
    /// not seen; without the record installed, nothing is.
    pub fn watch<R>(f: impl FnOnce() -> R) -> (R, Option<usize>) {
        /// Restores the thread's watch as it was, `f` returning or not.
        struct Restore(bool, usize);
        impl Drop for Restore {
            fn drop(&mut self) {
                WATCHED.set(self.0);
                LARGEST.set(self.1);
            }
        }
        let restore = Restore(WATCHED.replace(true), LARGEST.replace(0));
        let value = f();
        let largest = LARGEST.get();
        drop(restore);
        (value, (largest > RECORD_THRESHOLD).then_some(largest))
    }

    /// Note an allocation of `size` bytes the system made.
    fn note(size: usize) {
        if !INSTALLED.load(Ordering::Relaxed) {
            INSTALLED.store(true, Ordering::Relaxed);
        }
        let live = LIVE.fetch_add(size, Ordering::Relaxed).wrapping_add(size);
        PEAK.fetch_max(live, Ordering::Relaxed);
        // `try_with`: neither cell has a destructor, so neither is ever
        // gone, but nothing here may panic.
        let _ = WATCHED.try_with(|watched| {
            if watched.get() {
                let _ = LARGEST.try_with(|largest| largest.set(largest.get().max(size)));
            }
        });
    }
}

// SAFETY: every method hands the call to `System` unchanged and returns
// what it returned; the bookkeeping beside it touches only atomics and
// constant-initialised thread-locals, allocates nothing and cannot panic.
unsafe impl GlobalAlloc for AllocationRecord {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `layout`, passed on.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            Self::note(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `layout`, passed on.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            Self::note(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator, so from `System`, with
        // `layout`: the caller's contract.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` came from this allocator, so from `System`, with
        // `layout`, and `new_size` is the caller's to keep valid.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            Self::note(new_size);
        }
        p
    }
}

/// How many [`growth`] measures are running. The first of a run of
/// overlapping ones resets `PEAK`, under this lock, and no later one does
/// until every one has ended, so no measure lowers the peak another is
/// reading.
static MEASURES: Mutex<usize> = Mutex::new(0);

/// Run `f` and return its value with the peak growth of the heap the
/// record saw during it, `None` without the record installed. Process
/// wide: another thread's allocations and frees meanwhile, a second
/// plan's included, count too. A measure overlapping another neither
/// resets nor lowers the other's peak: each reads the peak since the
/// first of the overlapping ones began, less its own start, so an
/// allocation made during it is never missed for a concurrent execute.
pub(crate) fn growth<R>(f: impl FnOnce() -> R) -> (R, Option<u64>) {
    /// Ends the measure, `f` returning or not.
    struct End;
    impl Drop for End {
        fn drop(&mut self) {
            *MEASURES.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        }
    }
    let base = {
        let mut running = MEASURES.lock().unwrap_or_else(PoisonError::into_inner);
        if *running == 0 {
            PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
        }
        *running += 1;
        LIVE.load(Ordering::Relaxed)
    };
    let end = End;
    let value = f();
    let peak = PEAK.load(Ordering::Relaxed);
    drop(end);
    let grown = INSTALLED
        .load(Ordering::Relaxed)
        .then(|| peak.saturating_sub(base) as u64);
    (value, grown)
}
