//! The two memory kinds a block can hold, the raw calls behind each, and the
//! element types a row can hold.

use std::ffi::c_void;

use wwr_sys::wwr_status;

/// The raw operations behind a block. The trait is public, so it can bound
/// [`Kind`], but it lives in a module nothing outside the arena can name:
/// only the arena allocates (`docs/programming_model.md`, "Memory: layouts
/// and arenas").
pub mod sealed {
    use wwr_sys::wwr_status;

    pub trait Sealed {
        /// Whether the calls need the block's device made current.
        const ON_DEVICE: bool;
        /// Whether the `0xFF` fill runs in release builds too. A host block
        /// is handed out as `&mut [T]`, which must not be uninitialised
        /// memory in any profile; a device block is only ever a pointer.
        const ALWAYS_FILL: bool;

        /// Allocate `bytes`, writing the base to `ptr`.
        ///
        /// # Safety
        /// `ptr` is valid for one write.
        unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> wwr_status;

        /// Free an allocation [`Sealed::alloc`] made of `bytes`.
        ///
        /// # Safety
        /// `ptr` came from `alloc(.., bytes)` of this kind, freed once.
        unsafe fn free(ptr: *mut u8, bytes: usize) -> wwr_status;

        /// Set every byte to `0xFF` and wait for it.
        ///
        /// # Safety
        /// `ptr` is an allocation of this kind of at least `bytes`.
        unsafe fn fill(ptr: *mut u8, bytes: usize) -> wwr_status;

        /// The memory still free, for an allocation failure's message.
        fn free_memory() -> String;
    }
}

/// A memory kind: [`Device`] or [`Pinned`]. Sealed; the kind is in every
/// table, row and block type, so a row of one kind cannot be asked of a
/// block of the other.
pub trait Kind: sealed::Sealed + Send + Sync + 'static {
    /// How the kind is named in a table's text and an admission refusal.
    const NAME: &'static str;
}

/// A kind the host can read and write directly, whose rows are handed out
/// as slices: [`Pinned`].
pub trait HostKind: Kind {}

/// Device memory on one GPU worker's card: a plain allocation, not a pool's
/// (`docs/memory.md`, "The two kinds").
#[derive(Debug)]
pub enum Device {}

/// Pinned host memory owned by one GPU worker, allocated portable so any
/// card's copy engine may read it (`docs/programming_model.md`, "Memory
/// kinds").
#[derive(Debug)]
pub enum Pinned {}

/// `cudaHostAllocPortable` / `hipHostMallocPortable`: both `0x1`.
const HOST_ALLOC_PORTABLE: u32 = 0x1;

impl sealed::Sealed for Device {
    const ON_DEVICE: bool = true;
    const ALWAYS_FILL: bool = false;

    unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> wwr_status {
        // SAFETY: the caller's pointer is valid for one write.
        unsafe { wwr_sys::wwr_malloc(ptr.cast::<*mut c_void>(), bytes) }
    }

    unsafe fn free(ptr: *mut u8, _: usize) -> wwr_status {
        // SAFETY: the caller's allocation, freed once.
        unsafe { wwr_sys::wwr_free(ptr.cast()) }
    }

    unsafe fn fill(ptr: *mut u8, bytes: usize) -> wwr_status {
        // SAFETY: the caller's allocation of at least `bytes`.
        let status = unsafe { wwr_sys::wwr_memset(ptr.cast(), 0xFF, bytes) };
        if status != wwr_status::WWR_STATUS_SUCCESS {
            return status;
        }
        // The memset runs on the legacy stream, which the run's
        // non-blocking streams do not wait for: finish it here.
        // SAFETY: a plain runtime call.
        unsafe { wwr_sys::wwr_device_synchronize() }
    }

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
    const ON_DEVICE: bool = true;
    const ALWAYS_FILL: bool = true;

    unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> wwr_status {
        // SAFETY: the caller's pointer is valid for one write.
        unsafe { wwr_sys::wwr_host_alloc(ptr.cast(), bytes, HOST_ALLOC_PORTABLE) }
    }

    unsafe fn free(ptr: *mut u8, _: usize) -> wwr_status {
        // SAFETY: the caller's allocation, freed once.
        unsafe { wwr_sys::wwr_free_host(ptr.cast()) }
    }

    unsafe fn fill(ptr: *mut u8, bytes: usize) -> wwr_status {
        // SAFETY: the caller's host allocation of at least `bytes`.
        unsafe { std::ptr::write_bytes(ptr, 0xFF, bytes) };
        wwr_status::WWR_STATUS_SUCCESS
    }

    fn free_memory() -> String {
        mem_available()
    }
}

/// `MemAvailable` from `/proc/meminfo`, as the C++ `check_pinned_node`
/// reads it.
fn mem_available() -> String {
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

/// Pageable host memory from the global allocator: a host kind that needs no
/// device, for the device-free tests of blocks and views. Built for this
/// crate's tests and, through the `test-support` feature, for a dependent's.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
pub enum Heap {}

#[cfg(any(test, feature = "test-support"))]
impl Heap {
    fn layout(bytes: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(bytes, super::DEVICE_ALIGNMENT)
            .expect("a block's size fits a layout")
    }
}

#[cfg(any(test, feature = "test-support"))]
impl sealed::Sealed for Heap {
    const ON_DEVICE: bool = false;
    const ALWAYS_FILL: bool = true;

    unsafe fn alloc(ptr: *mut *mut u8, bytes: usize) -> wwr_status {
        // SAFETY: a block's size is never zero, so the layout is not empty.
        let base = unsafe { std::alloc::alloc(Self::layout(bytes)) };
        // SAFETY: the caller's pointer is valid for one write.
        unsafe { ptr.write(base) };
        if base.is_null() {
            wwr_status::WWR_STATUS_MEMORY_ALLOCATION
        } else {
            wwr_status::WWR_STATUS_SUCCESS
        }
    }

    unsafe fn free(ptr: *mut u8, bytes: usize) -> wwr_status {
        // SAFETY: allocated by `alloc` with this layout, freed once.
        unsafe { std::alloc::dealloc(ptr, Self::layout(bytes)) };
        wwr_status::WWR_STATUS_SUCCESS
    }

    unsafe fn fill(ptr: *mut u8, bytes: usize) -> wwr_status {
        // SAFETY: the caller's allocation of at least `bytes`.
        unsafe { std::ptr::write_bytes(ptr, 0xFF, bytes) };
        wwr_status::WWR_STATUS_SUCCESS
    }

    fn free_memory() -> String {
        mem_available()
    }
}

#[cfg(any(test, feature = "test-support"))]
impl Kind for Heap {
    const NAME: &'static str = "heap";
}

#[cfg(any(test, feature = "test-support"))]
impl HostKind for Heap {}

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
