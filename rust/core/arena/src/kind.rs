//! The three memory kinds a block can hold, the raw calls behind each, and
//! the element types a row can hold.
//!
//! The raw calls are the arena's one seam onto the runtime. With the
//! `device` feature (the default) [`Device`] and [`Pinned`] reach it through
//! `wwr-sys` (`device.rs`); without it they are still kinds, so tables, rows,
//! admission and the planner built on them compile and test with no library
//! linked, but a block of either of non-zero size aborts (`no_device.rs`).
//! [`Heap`] needs no runtime: its block is the global allocator's, in every
//! build.

use std::panic::Location;

use fb_core_error_handling::{Error, Status};

#[cfg(feature = "device")]
mod device;
#[cfg(not(feature = "device"))]
mod no_device;

/// The raw operations behind a block. The trait is public, so it can bound
/// [`Kind`], but it lives in a module nothing outside the arena can name:
/// only the arena allocates (`docs/programming_model.md`, "Memory: layouts
/// and arenas").
pub mod sealed {
    use super::{Error, Location, Status};

    pub trait Sealed {
        /// Whether the `0xFF` fill runs in release builds too. A host block
        /// is handed out as `&mut [T]`, which must not be uninitialised
        /// memory in any profile; a device block is only ever a pointer.
        const ALWAYS_FILL: bool;

        /// Held while the calls need the block's device current: a device
        /// guard, or `()` for a kind that needs none.
        type Current;

        /// Make device `dev_idx` current, aborting if it cannot be.
        fn current(dev_idx: i32) -> Self::Current;

        /// Make device `dev_idx` current, or say why it cannot be: for a
        /// `Drop`, which logs rather than aborts.
        ///
        /// # Errors
        /// The failed device switch.
        fn try_current(dev_idx: i32) -> Result<Self::Current, Error>;

        /// Allocate `bytes`, writing the base to `ptr`.
        ///
        /// # Errors
        /// The failure, named and described, for the abort's message.
        ///
        /// # Safety
        /// `ptr` is valid for one write.
        unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> Result<(), String>;

        /// Free an allocation [`Sealed::alloc`] made of `bytes`, logging a
        /// failure against `location`.
        ///
        /// # Safety
        /// `ptr` came from `alloc(.., bytes)` of this kind, freed once.
        unsafe fn free(ptr: *mut u8, bytes: usize, location: &'static Location<'static>);

        /// Set every byte to `0xFF` and wait for it.
        ///
        /// # Errors
        /// The failed call, located at `location`.
        ///
        /// # Safety
        /// `ptr` is an allocation of this kind of at least `bytes`.
        unsafe fn fill(ptr: *mut u8, bytes: usize, location: &'static Location<'static>) -> Status;

        /// The memory still free, for an allocation failure's message.
        fn free_memory() -> String;
    }
}

/// A memory kind: [`Device`], [`Pinned`] or [`Heap`]. Sealed; the kind is
/// in every table, row and block type, so a row of one kind cannot be asked
/// of a block of another.
pub trait Kind: sealed::Sealed + Send + Sync + 'static {
    /// How the kind is named in a table's text and an admission refusal.
    const NAME: &'static str;
}

/// A kind the host can read and write directly, whose rows are handed out
/// as slices: [`Pinned`] and [`Heap`].
pub trait HostKind: Kind {}

/// A host kind a GPU worker's copies stage through: [`Pinned`], and in the
/// device-free tests `PinnedStandIn` (the `test-support` feature). The
/// kind of a plan's per-worker host table, beside its device one; never
/// [`Heap`], whose rows are the plan's one table of their own, so no two
/// tables of a plan share a kind and a row of one cannot be asked of the
/// other (`docs/memory.md`, "The three kinds"). Sealed with the rest.
pub trait StagingKind: HostKind {}

/// Device memory on one GPU worker's card: a plain allocation, not a pool's
/// (`docs/memory.md`, "The three kinds").
#[derive(Debug)]
pub enum Device {}

/// Pinned host memory owned by one GPU worker, allocated portable so any
/// card's copy engine may read it (`docs/programming_model.md`, "Memory
/// kinds").
#[derive(Debug)]
pub enum Pinned {}

/// `MemAvailable` from `/proc/meminfo`, as the C++ `check_pinned_node`
/// reads it.
pub(crate) fn mem_available() -> String {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("MemAvailable:"))
                .map(|v| format!("MemAvailable {}", v.trim()))
        })
        .unwrap_or_else(|| "MemAvailable unknown".to_owned())
}

impl Kind for Device {
    const NAME: &'static str = "device";
}

impl Kind for Pinned {
    const NAME: &'static str = "pinned";
}

impl HostKind for Pinned {}

impl StagingKind for Pinned {}

/// Pageable host memory: one block per plan, from the global allocator,
/// for large flat host data no copy engine meets, sized at planning
/// (`docs/memory.md`, "The three kinds"). Only a CPU node holds a heap
/// row. The block keeps the pinned alignment, and is filled with `0xFF`
/// in every build, as pinned is, which also commits its pages when it is
/// made, so the memory admission counted is the memory it holds.
#[derive(Debug)]
pub enum Heap {}

impl Heap {
    fn layout(bytes: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(bytes, super::DEVICE_ALIGNMENT)
            .expect("a block's size fits a layout")
    }
}

impl sealed::Sealed for Heap {
    const ALWAYS_FILL: bool = true;
    type Current = ();

    fn current(_: i32) {}

    fn try_current(_: i32) -> Result<(), Error> {
        Ok(())
    }

    unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> Result<(), String> {
        // SAFETY: a block's size is never zero, so the layout is not empty.
        let base = unsafe { std::alloc::alloc(Self::layout(bytes)) };
        // SAFETY: the caller's pointer is valid for one write.
        unsafe { ptr.write(base) };
        if base.is_null() {
            Err("the global allocator returned null".to_owned())
        } else {
            Ok(())
        }
    }

    unsafe fn free(ptr: *mut u8, bytes: usize, _: &'static Location<'static>) {
        // SAFETY: allocated by `alloc` with this layout, freed once.
        unsafe { std::alloc::dealloc(ptr, Self::layout(bytes)) };
    }

    unsafe fn fill(ptr: *mut u8, bytes: usize, _: &'static Location<'static>) -> Status {
        // SAFETY: the caller's allocation of at least `bytes`.
        unsafe { std::ptr::write_bytes(ptr, 0xFF, bytes) };
        Ok(())
    }

    fn free_memory() -> String {
        mem_available()
    }
}

impl Kind for Heap {
    const NAME: &'static str = "heap";
}

impl HostKind for Heap {}

/// The device-free tests' stand-in for [`Pinned`]: a staging kind on the
/// global allocator, so a plan's per-worker host table is real memory with
/// no device, beside a [`Heap`] table of its own kind. Its tables print as
/// `pinned`, the kind it stands in for. Built for this crate's tests and,
/// through the `test-support` feature, for a dependent's.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
pub enum PinnedStandIn {}

#[cfg(any(test, feature = "test-support"))]
impl sealed::Sealed for PinnedStandIn {
    const ALWAYS_FILL: bool = true;
    type Current = ();

    fn current(_: i32) {}

    fn try_current(_: i32) -> Result<(), Error> {
        Ok(())
    }

    unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> Result<(), String> {
        // SAFETY: the caller's contract, which is the heap's.
        unsafe { <Heap as sealed::Sealed>::alloc(ptr, bytes) }
    }

    unsafe fn free(ptr: *mut u8, bytes: usize, location: &'static Location<'static>) {
        // SAFETY: allocated by `alloc` above, the heap's, with these bytes.
        unsafe { <Heap as sealed::Sealed>::free(ptr, bytes, location) }
    }

    unsafe fn fill(ptr: *mut u8, bytes: usize, location: &'static Location<'static>) -> Status {
        // SAFETY: the caller's allocation of at least `bytes`.
        unsafe { <Heap as sealed::Sealed>::fill(ptr, bytes, location) }
    }

    fn free_memory() -> String {
        mem_available()
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Kind for PinnedStandIn {
    const NAME: &'static str = "pinned";
}

#[cfg(any(test, feature = "test-support"))]
impl HostKind for PinnedStandIn {}

#[cfg(any(test, feature = "test-support"))]
impl StagingKind for PinnedStandIn {}

/// A type a row may hold.
///
/// # Safety
/// Every bit pattern of `size_of::<Self>()` bytes is a valid value, the
/// type has no padding and no drop glue, and its alignment divides
/// [`DEVICE_ALIGNMENT`](super::DEVICE_ALIGNMENT). A block's bytes are
/// filled with `0xFF` and then handed out as `[Self]`, and a row reused by
/// a later node is read with whatever an earlier one left, so any other
/// type would be undefined behaviour.
pub unsafe trait Element: Copy + Send + Sync + 'static {}

macro_rules! element {
    ($($t:ty),*) => {$(
        // SAFETY: a primitive integer or float: every bit pattern is a
        // value, with no padding and no drop glue, aligned to at most 8.
        unsafe impl Element for $t {}
    )*};
}
element!(u8, i32, u32, i64, u64, f32, f64);
