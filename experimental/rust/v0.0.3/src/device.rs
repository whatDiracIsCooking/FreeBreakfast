//! Streams and device resources: a step's stream on each card it spans
//! ([`CardLane`], the design's `ForkStream`), a card's lock
//! ([`CardResources`]), the bind the wrapper makes at every acquisition,
//! and the copies between host and device memory.
//!
//! There is no library stream. A step's library calls go on its own
//! stream, under the card's lock, which the wrapper holds for the step's
//! whole `run` and releases only after the step-end sync.
//!
//! **Where this departs from the design.** The design has one BLAS handle
//! per card, rebound to the holder's stream at every acquisition.
//! `fb_core_wwr::blas::BlasHandle` cannot be rebound: it keeps the
//! `Arc<Stream>` it was made on, and `fb_recipes_sys::trsm` launches its
//! diagonal kernels on that stream (`blas.owner()`), so a handle rebound
//! through the raw API would split one solve over two streams. So here a
//! BLAS handle is made per library-using step per card, on that step's own
//! stream, in budget mode. What is per card, as designed: the lock, and the
//! one fixed workspace every handle on the card is pointed at when its step
//! takes the lock. README.md, "What the prototype says to change", has it.

use std::sync::{Arc, Mutex};

use fb_core_error_handling::{Result, Status};
use fb_core_wwr::blas::BlasHandle;
use fb_core_wwr::common::{ScopedDeviceIndex, wwr_status_from_code};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::{gpu_status, gpu_status_at};
use fb_kernel_abi::LaunchStatus;
use wwr_sys::wwr_memcpy_kind;

/// A step's own stream on one card, and, if the step declared that it
/// calls the library there, its BLAS handle on that stream. Made by the
/// wrapper in budget mode, synchronized by the wrapper after every `run`.
pub struct CardLane {
    card: usize,
    physical: i32,
    stream: Arc<Stream>,
    blas: Option<BlasHandle>,
}

impl CardLane {
    /// The step's stream on logical card `card`, which is on device
    /// `physical`; with `library`, a BLAS handle on it.
    pub fn new(card: usize, physical: i32, library: bool) -> CardLane {
        let _current = ScopedDeviceIndex::new(physical);
        let stream = Arc::new(Stream::new(physical));
        CardLane {
            card,
            physical,
            blas: library.then(|| BlasHandle::new(Arc::clone(&stream))),
            stream,
        }
    }

    /// The logical card.
    pub fn card(&self) -> usize {
        self.card
    }

    /// Make the card's device current on this thread: every inner task that
    /// launches does, since it may run on any thread.
    pub fn current(&self) -> ScopedDeviceIndex {
        ScopedDeviceIndex::new(self.physical)
    }

    /// The stream: the step's kernels, copies and library calls go on it.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// The BLAS handle the wrapper bound for this `run`.
    ///
    /// # Panics
    /// If the step's `mem_budget` did not declare the library on this card.
    pub fn blas(&self) -> &BlasHandle {
        self.blas
            .as_ref()
            .unwrap_or_else(|| panic!("the step did not declare the library on card {}", self.card))
    }

    /// Whether the step declared the library on this card.
    pub fn calls_library(&self) -> bool {
        self.blas.is_some()
    }

    /// Block until everything enqueued on the stream has run. The wrapper
    /// calls it after `run`; a step calls it itself before it reads what it
    /// downloaded, or rewrites pinned bytes an enqueued copy still reads.
    ///
    /// # Errors
    /// A fault of the enqueued work, surfaced here.
    pub fn sync(&self) -> Status {
        let _current = self.current();
        gpu_status(self.stream.sync())
    }

    /// The acquisition's bind, in the one order that is safe on cuBLAS:
    /// `SetStream` to the step's stream, then `SetWorkspace` to the card's
    /// fixed workspace, every time (`SetStream` drops the workspace there).
    ///
    /// # Errors
    /// The library's refusal.
    pub(crate) fn bind(&self, workspace: *mut u8, bytes: usize) -> Status {
        let Some(blas) = &self.blas else {
            return Ok(());
        };
        let _current = self.current();
        // SAFETY: the handle and the stream are live and on this card. The
        // workspace is the card's, `bytes` of device memory the ledger
        // keeps for the run; the caller holds the card's lock, and the
        // previous holder synchronized before releasing it, so no call in
        // flight uses the workspace.
        unsafe {
            gpu_status(wwr_sys::wwr_blas_set_stream(
                blas.get(),
                self.stream.stream(),
            ))?;
            gpu_status(wwr_sys::wwr_blas_set_workspace(
                blas.get(),
                workspace.cast(),
                bytes,
            ))
        }
    }

    /// Set every byte of `bytes` at device pointer `ptr` to `0xFF`, on the
    /// stream: the NaN fill of a lease's device region.
    pub(crate) fn fill(&self, ptr: *mut u8, bytes: usize) -> Status {
        let _current = self.current();
        // SAFETY: the region is this step's scratch on this card, `bytes`
        // of device memory, and the stream is live.
        gpu_status(unsafe {
            wwr_sys::wwr_memset_async(ptr.cast(), 0xFF, bytes, self.stream.stream())
        })
    }
}

/// What the wrapper holds for a step: a [`CardLane`] per card it spans;
/// none for a host step.
pub struct Lane {
    /// The step's cards, ascending: gpu0 first.
    pub cards: Vec<CardLane>,
}

impl Lane {
    /// The step's lane on gpu0.
    ///
    /// # Panics
    /// For a host step, which has no stream.
    pub fn gpu0(&self) -> &CardLane {
        self.cards.first().expect("a host step has no stream")
    }
}

/// One per card, made before budget mode: the card's lock. The fixed BLAS
/// workspace it guards is the ledger's, allocated by the final node.
pub struct CardResources {
    /// The device the card is on. Two cards may share one.
    pub physical: i32,
    /// The lock: one library-using step at a time, for its whole `run`.
    pub lock: Mutex<()>,
}

/// Enqueue the copy of host `src` into device `dst` on `stream`. `src` is
/// pinned scratch: it must not be rewritten before a sync of that stream.
///
/// # Errors
/// The enqueue's failure.
pub fn upload<T: Copy>(dst: &mut impl DeviceSliceMut<T>, src: &[T], stream: &Stream) -> Status {
    assert!(dst.len() >= src.len(), "upload: the device end is shorter");
    if src.is_empty() {
        return Ok(());
    }
    // SAFETY: both ends hold `src.len()` elements; each is an `Output`'s
    // buffer or the running step's scratch, which outlives the enqueued
    // work: the wrapper synchronizes before the step's node finishes.
    gpu_status(unsafe {
        wwr_sys::wwr_memcpy_async(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            size_of_val(src),
            wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
            stream.stream(),
        )
    })
}

/// Enqueue the copy of the first `dst.len()` elements of device `src` into
/// host `dst` on `stream`. `dst` is read only after a sync of that stream.
///
/// # Errors
/// The enqueue's failure.
pub fn download<T: Copy>(dst: &mut [T], src: &impl DeviceSlice<T>, stream: &Stream) -> Status {
    assert!(
        src.len() >= dst.len(),
        "download: the device end is shorter"
    );
    if dst.is_empty() {
        return Ok(());
    }
    // SAFETY: as `upload`'s.
    gpu_status(unsafe {
        wwr_sys::wwr_memcpy_async(
            dst.as_mut_ptr().cast(),
            src.as_ptr().cast(),
            size_of_val(dst),
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
            stream.stream(),
        )
    })
}

/// A kernel launch wrapper's status as a [`Status`], naming `what`.
#[track_caller]
pub fn launched(what: &str, status: LaunchStatus) -> Status {
    fb_kernel_abi::launch_status(what, status, |code, at| {
        Some(gpu_status_at(wwr_status_from_code(code)?, at))
    })
}

/// A device's free and total memory, with it current: read once per device
/// at setup, for the caps.
///
/// # Errors
/// The runtime's failure.
pub fn memory(physical: i32) -> Result<(u64, u64)> {
    let _current = ScopedDeviceIndex::new(physical);
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: a plain runtime call writing two locals.
    gpu_status(unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) })?;
    Ok((free as u64, total as u64))
}
