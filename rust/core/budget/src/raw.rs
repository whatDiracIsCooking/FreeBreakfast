//! The one place memory is taken and given back.

use std::ffi::c_void;

use fb_core_device::ScopedDevice;
use fb_core_error_handling::{Error, Status, check_or_abort};
use wwr_sys::wwr_status;

use crate::ALIGN;

/// Where an allocation lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mem {
    /// Device memory on a physical device.
    Device(i32),
    /// Page-locked host memory, portable.
    Pinned,
    /// The plain host heap, zeroed: a host-backed plan's every kind.
    Host,
}

/// A runtime call's status as a [`Status`], naming `what`.
#[track_caller]
fn status(code: wwr_status, what: &str) -> Status {
    match code {
        wwr_status::WWR_STATUS_SUCCESS => Ok(()),
        _ => Err(Error::gpu(format!("{what}: {code:?}"))),
    }
}

/// One allocation of at least a byte, freed on drop.
pub(crate) struct RawBuf {
    pub(crate) ptr: *mut u8,
    pub(crate) bytes: usize,
    pub(crate) mem: Mem,
}

// SAFETY: an allocation is not tied to a host thread; the step graph, and
// the borrows of the buffers that own one, order its users.
unsafe impl Send for RawBuf {}
// SAFETY: as above; `RawBuf` itself hands out only its address.
unsafe impl Sync for RawBuf {}

impl RawBuf {
    fn host_layout(bytes: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(bytes, ALIGN as usize).expect("a valid layout")
    }

    /// `bytes` of `mem`, aborting at the caller's line if the runtime or
    /// the allocator refuses: the budget already accepted them, so a
    /// refusal here is the machine's, past the margin. Pinned memory is
    /// filled with `0xFF`, so the host never reads it uninitialised.
    #[allow(
        clippy::disallowed_methods,
        reason = "the one site of the raw allocators: a Budget's buffer, a slab or a workspace, made in budget mode (docs/memory.md, \"The budget\")"
    )]
    #[track_caller]
    pub(crate) fn new(mem: Mem, bytes: usize) -> RawBuf {
        // INVARIANT: the ledger allocates only a charge of a byte or more.
        debug_assert!(bytes > 0, "an allocation holds a byte");
        let ptr = match mem {
            Mem::Device(physical) => {
                let _current = ScopedDevice::new(physical);
                let mut p: *mut c_void = std::ptr::null_mut();
                // SAFETY: writes one pointer through a valid local, with the
                // device current.
                let code = unsafe { wwr_sys::wwr_malloc(&mut p, bytes) };
                check_or_abort(status(code, &format!("wwr_malloc of {bytes} bytes")));
                p.cast::<u8>()
            }
            Mem::Pinned => {
                let mut p: *mut c_void = std::ptr::null_mut();
                // SAFETY: writes one pointer through a valid local.
                let code = unsafe { wwr_sys::wwr_host_alloc(&mut p, bytes, 0) };
                check_or_abort(status(code, &format!("wwr_host_alloc of {bytes} bytes")));
                let p = p.cast::<u8>();
                // SAFETY: `bytes` of host memory, just allocated and ours.
                unsafe { std::ptr::write_bytes(p, 0xFF, bytes) };
                p
            }
            Mem::Host => {
                // SAFETY: the layout has a size above zero.
                let p = unsafe { std::alloc::alloc_zeroed(Self::host_layout(bytes)) };
                if p.is_null() {
                    std::alloc::handle_alloc_error(Self::host_layout(bytes));
                }
                p
            }
        };
        RawBuf { ptr, bytes, mem }
    }

    /// Whether the bytes are host memory: pinned, or a host-backed plan's.
    pub(crate) fn is_host(&self) -> bool {
        !matches!(self.mem, Mem::Device(_))
    }
}

impl Drop for RawBuf {
    /// Free once; a failed free is logged at error and the drop carries on
    /// (`docs/diagnostics.md`, "`Drop` cannot fail").
    fn drop(&mut self) {
        let freed = match self.mem {
            Mem::Device(physical) => {
                let _current = ScopedDevice::new(physical);
                // SAFETY: allocated by wwr_malloc on this device, freed once.
                status(unsafe { wwr_sys::wwr_free(self.ptr.cast()) }, "wwr_free")
            }
            Mem::Pinned => {
                // SAFETY: allocated by wwr_host_alloc, freed once.
                let code = unsafe { wwr_sys::wwr_free_host(self.ptr.cast()) };
                status(code, "wwr_free_host")
            }
            Mem::Host => {
                // SAFETY: allocated with this layout, freed once.
                unsafe { std::alloc::dealloc(self.ptr, Self::host_layout(self.bytes)) };
                Ok(())
            }
        };
        if let Err(e) = freed {
            log::error!("a budget buffer's free failed: {e}");
        }
    }
}

/// Ask for huge pages under a fresh heap slab, before its pages are
/// touched. On the VM the glycine runs used, touching gly10's 7.3 GiB B
/// took 7.6 s at 4 KiB a page and 5.3 s with huge pages; on bare metal 2.7
/// s and 0.5 s (the prototype's `examples/touch_probe.rs`). Advice only:
/// where the kernel gives none the slab is touched a small page at a time.
pub(crate) fn huge_pages(ptr: *mut u8, bytes: usize) {
    #[cfg(target_os = "linux")]
    {
        const HUGE: usize = 2 << 20;
        const MADV_HUGEPAGE: i32 = 14;
        unsafe extern "C" {
            fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
        }
        let first = (ptr as usize).next_multiple_of(HUGE);
        let end = (ptr as usize + bytes) / HUGE * HUGE;
        if end > first {
            // SAFETY: the range is whole pages inside the slab's allocation,
            // and the advice changes nothing a program can read.
            unsafe { madvise(first as *mut c_void, end - first, MADV_HUGEPAGE) };
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (ptr, bytes);
}
