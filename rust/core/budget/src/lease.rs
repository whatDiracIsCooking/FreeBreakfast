//! A step's scratch for one `run`: the [`Layout`] `mem_budget` and `run`
//! share, the [`Lease`] the wrapper lends, and the tenants carved from it.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

use contracts::requires;
use fb_core_device::{DeviceSlice, DeviceSliceMut, HostSlice, HostSliceMut, Stream};
use fb_core_error_handling::Status;
use fb_core_log::{eq, le, lt};

use crate::Kind;
use crate::buffers::Plain;
use crate::ledger::{Ledger, locked};

/// One tenant of a step's scratch region: `len` elements at a byte offset.
/// Made by a [`Layout`], consumed by the [`Lease`] that carves it; neither
/// `Copy` nor `Clone`, so a tenant is carved once.
#[derive(Debug)]
pub struct Span<T> {
    pub(crate) offset: usize,
    pub(crate) len: usize,
    of: PhantomData<T>,
}

impl<T> Span<T> {
    /// The elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no element.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The byte past the tenant's last: no overflow, since its layout
    /// checked it.
    pub(crate) fn end(&self) -> usize {
        self.offset + self.len * size_of::<T>()
    }
}

/// A step's region of one slab, laid out: the one function `mem_budget`
/// and `run` share. `mem_budget` declares [`Layout::bytes`]; `run` carves
/// the spans.
#[derive(Debug, Default)]
pub struct Layout {
    bytes: usize,
}

impl Layout {
    /// An empty layout.
    #[must_use]
    pub fn new() -> Layout {
        Layout::default()
    }

    /// The next tenant: `len` elements of `T`, at the next 64-byte boundary.
    ///
    /// # Panics
    /// If the layout's end passes `isize::MAX` bytes.
    pub fn span<T>(&mut self, len: usize) -> Span<T> {
        let offset = self.bytes.next_multiple_of(64);
        let end = len
            .checked_mul(size_of::<T>())
            .and_then(|bytes| bytes.checked_add(offset))
            .filter(|&end| isize::try_from(end).is_ok());
        let Some(end) = end else {
            panic!("a layout of {len} elements past {offset} bytes passes isize::MAX");
        };
        self.bytes = end;
        Span {
            offset,
            len,
            of: PhantomData,
        }
    }

    /// The region's bytes: what `mem_budget` declares.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

/// One region of a lease: a step's bytes of one slab.
pub(crate) struct LeaseRegion {
    pub(crate) gpu: usize,
    pub(crate) kind: Kind,
    pub(crate) ptr: *mut u8,
    pub(crate) bytes: usize,
    /// Whether the bytes are host memory: pinned, or a host-backed plan's.
    pub(crate) host: bool,
}

/// A step's scratch for the length of one `run`: its region of each slab.
/// It borrows from the ledger, and what it carves borrows from it, so
/// nothing carved can be stored in an `Output`, which is `'static`:
///
/// ```compile_fail,E0521
/// use fb_core_budget::{HostPart, Layout, Lease};
/// fn keep(lease: &Lease<'_>, output: &mut Vec<HostPart<'static, f64>>) {
///     let mut layout = Layout::new();
///     output.push(lease.pinned(0, layout.span::<f64>(4)));
/// }
/// ```
///
/// The same with the borrow's own lifetime is a program:
///
/// ```no_run
/// use fb_core_budget::{HostPart, Layout, Lease};
/// fn carve<'a>(lease: &'a Lease<'_>, local: &mut Vec<HostPart<'a, f64>>) {
///     let mut layout = Layout::new();
///     local.push(lease.pinned(0, layout.span::<f64>(4)));
/// }
/// ```
///
/// A step holds one lease at a time ([`Ledger::lease`]), and two tenants
/// carved from it never meet: each carve is recorded, always on, so two
/// layouts over one region cannot hand out the same bytes twice.
pub struct Lease<'r> {
    pub(crate) ledger: &'r Ledger,
    pub(crate) step: String,
    pub(crate) regions: Vec<LeaseRegion>,
    /// The byte ranges carved so far, by gpu and kind.
    pub(crate) carved: Mutex<Vec<(usize, Kind, usize, usize)>>,
}

// SAFETY: a lease is addresses and a record behind a `Mutex`; a step's
// inner tasks carve disjoint tenants of it from several threads, and the
// record keeps them disjoint.
unsafe impl Send for Lease<'_> {}
// SAFETY: as above.
unsafe impl Sync for Lease<'_> {}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        locked(&self.ledger.leased).retain(|s| *s != self.step);
    }
}

impl Lease<'_> {
    fn region(&self, gpu: usize, kind: Kind) -> Option<&LeaseRegion> {
        self.regions.iter().find(|r| r.gpu == gpu && r.kind == kind)
    }

    /// The bytes the step declared on `gpu` for `kind`.
    #[must_use]
    pub fn bytes(&self, gpu: usize, kind: Kind) -> usize {
        self.region(gpu, kind).map_or(0, |r| r.bytes)
    }

    /// Set every byte of every host-addressable region to `0xFF`: NaN as an
    /// `f64`, -1 as an `i32`. The wrapper's test-mode fill, before any
    /// tenant is carved; the device regions of a real plan are filled on
    /// the step's stream.
    pub fn fill_host(&mut self) {
        for r in self.regions.iter().filter(|r| r.host) {
            // SAFETY: the region is `bytes` of host memory, this step's for
            // a run that has not started, and `&mut self` means no tenant
            // carved from it is alive.
            unsafe { std::ptr::write_bytes(r.ptr, 0xFF, r.bytes) };
        }
    }

    /// Enqueue the test-mode fill of the step's device region on `gpu` on
    /// `stream`, every byte `0xFF`, before any tenant is carved: the device
    /// half of [`Lease::fill_host`], which fills a host-backed plan's
    /// device regions itself, so nothing is enqueued for one. The wrapper
    /// syncs `stream` before the step's other streams may find the region.
    ///
    /// # Errors
    /// The enqueue's failure.
    #[requires(
        lt!(gpu, self.ledger.physical.len()),
        "gpu is a gpu of the run"
    )]
    #[requires(
        eq!(stream.physical(), self.ledger.physical[gpu]),
        "the stream is on the region's device"
    )]
    #[track_caller]
    pub fn fill_device(&mut self, gpu: usize, stream: &Stream) -> Status {
        let Some(r) = self.region(gpu, Kind::Device).filter(|r| !r.host) else {
            return Ok(());
        };
        // The step's whole region, a tenant of no one: `&mut self` means
        // nothing has been carved from it for this run.
        let mut whole = ScratchBuf::<u8> {
            ptr: r.ptr,
            len: r.bytes,
            host: false,
            life: PhantomData,
        };
        stream.fill(&mut whole, 0xFF)
    }

    /// Where tenant `span` of `gpu`'s `kind` starts, its bytes recorded as
    /// carved.
    ///
    /// # Panics
    /// If the span meets a tenant already carved from the lease.
    fn carve<T>(&self, gpu: usize, kind: Kind, span: &Span<T>) -> *mut T {
        if span.len == 0 {
            return std::ptr::NonNull::dangling().as_ptr();
        }
        let (begin, end) = (span.offset, span.end());
        let mut carved = locked(&self.carved);
        if let Some(&(_, _, b, e)) = carved
            .iter()
            .find(|&&(g, k, b, e)| g == gpu && k == kind && begin < e && b < end)
        {
            panic!(
                "step '{}' carves bytes {begin}..{end} of its {kind:?} scratch on gpu {gpu}, \
                 which meet the tenant already carved at {b}..{e}",
                self.step
            );
        }
        carved.push((gpu, kind, begin, end));
        let region = self.region(gpu, kind).expect("the span is within a region");
        // In range: the contract keeps the span inside the region.
        region.ptr.wrapping_add(begin).cast()
    }

    /// The tenant `span` of the step's device scratch on `gpu`.
    ///
    /// # Panics
    /// If the span meets a tenant already carved from this lease.
    #[requires(
        le!(span.end(), self.bytes(gpu, Kind::Device)),
        "a run carves what its mem_budget declared: the two come from one layout"
    )]
    #[track_caller]
    pub fn device<T: Plain>(&self, gpu: usize, span: Span<T>) -> ScratchBuf<'_, T> {
        let host = self.region(gpu, Kind::Device).is_some_and(|r| r.host);
        // The closure invariant: this region overlaps another step's only
        // if the DAG orders the two steps (`fb_core_pack::pack`), and a
        // finished node is finished device work (the wrapper's step-end
        // sync), so while this lease lives no other step touches these
        // bytes. Within the step, `carve` keeps the tenants apart.
        ScratchBuf {
            ptr: self.carve(gpu, Kind::Device, &span),
            len: span.len,
            host,
            life: PhantomData,
        }
    }

    /// The tenant `span` of the step's pinned scratch on `gpu`: a slice
    /// for the host, and the host end of a copy. Its contents on entry are
    /// undefined, and plain data.
    ///
    /// # Panics
    /// If the span meets a tenant already carved from this lease.
    #[requires(
        le!(span.end(), self.bytes(gpu, Kind::Pinned)),
        "a run carves what its mem_budget declared: the two come from one layout"
    )]
    #[track_caller]
    pub fn pinned<T: Plain>(&self, gpu: usize, span: Span<T>) -> HostPart<'_, T> {
        HostPart {
            ptr: self.carve(gpu, Kind::Pinned, &span),
            len: span.len,
            life: PhantomData,
        }
    }
}

/// A tenant of a step's device scratch, for the length of its `run`.
pub struct ScratchBuf<'a, T> {
    ptr: *mut T,
    len: usize,
    host: bool,
    life: PhantomData<&'a mut [T]>,
}

// SAFETY: device memory is not tied to a host thread, and the tenant is
// this one's alone.
unsafe impl<T: Send> Send for ScratchBuf<'_, T> {}

impl<T: Plain> ScratchBuf<'_, T> {
    /// The bytes as host memory, under [`Backing::Host`](crate::Backing)
    /// only.
    pub fn host(&mut self) -> Option<&mut [T]> {
        // SAFETY: host memory of a host-backed plan, `len` plain elements
        // (filled when made or by a run before), the tenant this one's
        // alone, and `&mut self` its one user.
        self.host
            .then(|| unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }
}

// SAFETY: `len` elements of the step's region, live while the lease is,
// apart from every other tenant (`Lease::carve`) and every unordered step's
// region (the closure).
unsafe impl<T> DeviceSlice<T> for ScratchBuf<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T> DeviceSliceMut<T> for ScratchBuf<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// A tenant of a step's pinned scratch, for the length of its `run`: a
/// host slice, and the host end of a copy, taken by value as a
/// [`PinnedBuf`](crate::PinnedBuf) is. A copy's token dropped unsynced
/// leaks it at no cost: its bytes stay carved, so no tenant meets them,
/// until the lease drops after the step-end sync.
pub struct HostPart<'a, T> {
    ptr: *mut T,
    len: usize,
    life: PhantomData<&'a mut [T]>,
}

// SAFETY: as `&mut [T]` moves when `T: Send`: the tenant is this one's.
unsafe impl<T: Send> Send for HostPart<'_, T> {}
// SAFETY: as `&mut [T]` is shared when `T: Sync`.
unsafe impl<T: Sync> Sync for HostPart<'_, T> {}

impl<T: Plain> Deref for HostPart<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `len` plain elements of host memory, filled when made,
        // this tenant's alone while the lease lives.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Plain> DerefMut for HostPart<'_, T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as `deref`, and `&mut self` is the one user.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

// SAFETY: `len` elements of host memory, live while the lease is and
// reached only through this tenant (`Lease::carve` hands its bytes out
// once). One never dropped leaves them carved, so reached by no one, until
// the lease drops, which the step wrapper does only after its step-end
// sync has finished the step's device work (`docs/memory.md`, "Step
// scratch").
unsafe impl<T: Plain> HostSlice<T> for HostPart<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T: Plain> HostSliceMut<T> for HostPart<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
