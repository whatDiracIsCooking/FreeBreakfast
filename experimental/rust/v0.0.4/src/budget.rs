//! The memory budget: the ledger of caps and counters, the [`Budget`] handle
//! a step's `mem_budget` allocates through, the buffers it hands back, and
//! step scratch: declared per step, packed by the final node, lent to `run`
//! as a [`Lease`].
//!
//! Each [`Budget`] call for memory a step's `Output` owns, in order: adds
//! the bytes to the kind's atomic counter and to the step's row of the
//! table; checks the counter against the cap, and sets the ledger's `over`
//! flag past it; allocates, unless the flag is set. [`Budget::scratch`] and
//! [`Budget::locks`] only record a row. [`Ledger::finish`], the budget
//! DAG's last node, packs each gpu and kind's scratch rows into one slab by
//! the DAG's closure ([`crate::pack`]), adds each packed peak and the fixed
//! BLAS-handle workspace to its counter, checks again, and either refuses
//! the whole plan with the table or allocates the slabs and the workspaces.
//!
//! No step holds a `Budget` in execute mode: `run` is never given one, and
//! the buffers' constructors are private to this module.

use std::ffi::c_void;
use std::fmt::Write as _;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use fb_core_error_handling::{Error, Status, check_or_abort};

use crate::device::{DeviceSlice, DeviceSliceMut, PartMut, ScopedDevice, runtime_status};

use crate::pack::{self, Packed, Region};

/// The BLAS handle's workspace, per gpu: a fixed row of the budget table.
pub const BLAS_WORKSPACE: u64 = 128 << 20;

/// What a scratch slab is aligned to, and every region inside it.
const ALIGN: u64 = 256;

/// The two kinds of step scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Device memory on one gpu.
    Device,
    /// Page-locked host memory, staged for one gpu.
    Pinned,
}

/// A column of the table: what a counter sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    Device(usize),
    Pinned(usize),
    Heap,
}

/// Where a ledger's bytes come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// Device and pinned memory through wwr.
    Real,
    /// Everything on the plain host heap, device and pinned kinds included:
    /// the core's tests run their steps with it, on no gpu.
    Host,
    /// Nothing is allocated: counted and checked only, every buffer empty.
    /// A plan budgeted so cannot be executed.
    Dry,
}

/// A cap as the config wrote it and as the plan is held to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cap {
    /// What the config gave; `None` for a defaulted cap.
    pub written: Option<u64>,
    /// What the plan is held against.
    pub used: u64,
}

impl Cap {
    /// Clamp a cap to what the machine has at init. A defaulted cap is the
    /// reading less `margin`; an explicit one is used as written, the user
    /// owning the margin, unless it is past the reading, and then it is the
    /// reading itself. With no reading (a dry budget on no gpu) the cap is
    /// as written, or none.
    pub fn resolve(written: Option<u64>, reading: Option<u64>, margin: u64) -> Cap {
        let used = match (written, reading) {
            (Some(w), Some(r)) => w.min(r),
            (Some(w), None) => w,
            (None, Some(r)) => r.saturating_sub(margin),
            (None, None) => u64::MAX,
        };
        Cap { written, used }
    }
}

/// Ask for huge pages under a fresh heap slab, before its pages are
/// touched. On the VM the glycine runs used, touching gly10's 7.3 GiB B took
/// 7.6 s at 4 KiB a page and 5.3 s with huge pages; on bare metal it takes
/// 2.7 s and 0.5 s (`examples/touch_probe.rs`). Advice
/// only: where the kernel gives none the slab is touched a small page at a
/// time, as before.
fn huge_pages(ptr: *mut u8, bytes: usize) {
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

/// The device margin of a defaulted cap: the larger of 512 MiB and 5% of
/// the gpu's total memory.
pub fn device_margin(total: u64) -> u64 {
    (512 << 20).max(total / 20)
}

/// The host margin of a defaulted cap: the larger of 1 GiB and 5% of the
/// available memory.
pub fn host_margin(available: u64) -> u64 {
    (1 << 30).max(available / 20)
}

/// The caps the counters are held against.
#[derive(Clone, Debug)]
pub struct Caps {
    /// Device memory, per gpu.
    pub device: Vec<Cap>,
    /// The host: heap and pinned bytes together.
    pub host: Cap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mem {
    /// Device memory on a physical device.
    Device(i32),
    Pinned,
    Host,
}

/// One allocation: the only place memory is taken and given back.
struct RawBuf {
    ptr: *mut u8,
    bytes: usize,
    mem: Mem,
}

// SAFETY: an allocation is not tied to a host thread; the step graph, not
// the type, orders its users.
unsafe impl Send for RawBuf {}
// SAFETY: as above.
unsafe impl Sync for RawBuf {}

impl RawBuf {
    fn host_layout(bytes: usize) -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(bytes, ALIGN as usize).expect("a valid layout")
    }

    fn new(mem: Mem, bytes: usize) -> RawBuf {
        assert!(bytes > 0, "an allocation holds a byte");
        let ptr = match mem {
            Mem::Device(physical) => {
                let _current = ScopedDevice::new(physical);
                let mut p: *mut c_void = std::ptr::null_mut();
                // SAFETY: writes one pointer through a valid local, with the
                // gpu current.
                check_or_abort(runtime_status(unsafe {
                    wwr_sys::wwr_malloc(&mut p, bytes)
                }));
                p.cast()
            }
            Mem::Pinned => {
                let mut p: *mut c_void = std::ptr::null_mut();
                // SAFETY: writes one pointer through a valid local.
                check_or_abort(runtime_status(unsafe {
                    wwr_sys::wwr_host_alloc(&mut p, bytes, 0)
                }));
                // Not zeroed: page-locking has already brought every page
                // in, and a second pass over it is time for nothing.
                p.cast()
            }
            Mem::Host => {
                // SAFETY: the layout has a size above zero.
                let p = unsafe { std::alloc::alloc_zeroed(Self::host_layout(bytes)) };
                assert!(!p.is_null(), "the host allocation failed");
                p
            }
        };
        RawBuf { ptr, bytes, mem }
    }
}

impl Drop for RawBuf {
    fn drop(&mut self) {
        match self.mem {
            Mem::Device(physical) => {
                let _current = ScopedDevice::new(physical);
                // SAFETY: allocated by wwr_malloc on this gpu, freed once.
                unsafe { wwr_sys::wwr_free(self.ptr.cast()) };
            }
            // SAFETY: allocated by wwr_host_alloc, freed once.
            Mem::Pinned => unsafe {
                wwr_sys::wwr_free_host(self.ptr.cast());
            },
            // SAFETY: allocated with this layout, freed once.
            Mem::Host => unsafe { std::alloc::dealloc(self.ptr, Self::host_layout(self.bytes)) },
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Row {
    step: &'static str,
    column: Column,
    bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct ScratchRow {
    step: &'static str,
    gpu: usize,
    kind: Kind,
    bytes: u64,
}

/// One packed slab: a gpu and kind's scratch.
struct Slab {
    gpu: usize,
    kind: Kind,
    /// The declaring steps, in name order, with their offsets in `packed`.
    steps: Vec<(&'static str, u64)>,
    packed: Packed,
    buf: Option<RawBuf>,
}

/// One step's region of a slab, for a test or a figure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionInfo {
    /// The declaring step.
    pub step: &'static str,
    /// The gpu.
    pub gpu: usize,
    /// Device or pinned.
    pub kind: Kind,
    /// The region's first byte in the slab.
    pub offset: u64,
    /// Its bytes, as declared and aligned.
    pub bytes: u64,
}

/// One slab's totals, for a test or a figure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlabInfo {
    /// The gpu.
    pub gpu: usize,
    /// Device or pinned.
    pub kind: Kind,
    /// The slab's size.
    pub peak: u64,
    /// The plain sum of the declarations.
    pub sum: u64,
    /// The heaviest set of mutually unordered declarations.
    pub bound: u64,
    /// The steps that make the bound.
    pub bound_steps: Vec<&'static str>,
}

/// The caps, the counters, the `over` flag, the per-step table and, after
/// the final node, the scratch slabs: one per run, held by the bus.
pub struct Ledger {
    caps: Caps,
    backing: Backing,
    /// Each gpu's physical device.
    physical: Vec<i32>,
    nan_fill: bool,
    device: Vec<AtomicU64>,
    pinned: Vec<AtomicU64>,
    heap: AtomicU64,
    over: AtomicBool,
    sealed: AtomicBool,
    allocations: AtomicU64,
    rows: Mutex<Vec<Row>>,
    scratch: Mutex<Vec<ScratchRow>>,
    locks: Mutex<Vec<(&'static str, LockId)>>,
    slabs: OnceLock<Vec<Slab>>,
    workspaces: OnceLock<Vec<Option<RawBuf>>>,
}

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Ledger {
    /// An empty ledger over `caps`, one gpu per entry of `physical` (the
    /// device each gpu is on; two gpus may name one device). With
    /// `nan_fill`, the wrapper fills each lease with NaN before `run`.
    pub fn new(caps: Caps, backing: Backing, physical: Vec<i32>, nan_fill: bool) -> Ledger {
        assert_eq!(caps.device.len(), physical.len(), "one device cap per gpu");
        let zeros = |n: usize| (0..n).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();
        let n = physical.len();
        Ledger {
            caps,
            backing,
            physical,
            nan_fill,
            device: zeros(n),
            pinned: zeros(n),
            heap: AtomicU64::new(0),
            over: AtomicBool::new(false),
            sealed: AtomicBool::new(false),
            allocations: AtomicU64::new(0),
            rows: Mutex::new(Vec::new()),
            scratch: Mutex::new(Vec::new()),
            locks: Mutex::new(Vec::new()),
            slabs: OnceLock::new(),
            workspaces: OnceLock::new(),
        }
    }

    /// Where the bytes come from.
    pub fn backing(&self) -> Backing {
        self.backing
    }

    /// The gpus.
    pub fn gpus(&self) -> usize {
        self.physical.len()
    }

    /// The caps.
    pub fn caps(&self) -> &Caps {
        &self.caps
    }

    /// Whether the wrapper fills each lease with NaN before `run`.
    pub fn nan_fill(&self) -> bool {
        self.nan_fill
    }

    /// Whether some charge went over its cap.
    pub fn over(&self) -> bool {
        self.over.load(Ordering::Acquire)
    }

    /// Whether [`Ledger::finish`] has run: nothing is budgeted after it.
    pub fn sealed(&self) -> bool {
        self.sealed.load(Ordering::Acquire)
    }

    /// The allocations this ledger has made: the same before and after an
    /// execute, since `run` has no `Budget`.
    pub fn allocations(&self) -> u64 {
        self.allocations.load(Ordering::Acquire)
    }

    /// The device bytes charged to `gpu` so far; after the final node, the
    /// packed scratch and the handle workspace are in it.
    pub fn device_total(&self, gpu: usize) -> u64 {
        self.device[gpu].load(Ordering::Acquire)
    }

    /// The pinned bytes charged for `gpu` so far.
    pub fn pinned_total(&self, gpu: usize) -> u64 {
        self.pinned[gpu].load(Ordering::Acquire)
    }

    /// The budgeted heap so far.
    pub fn heap_total(&self) -> u64 {
        self.heap.load(Ordering::Acquire)
    }

    fn host_total(&self) -> u64 {
        self.heap_total() + (0..self.gpus()).map(|c| self.pinned_total(c)).sum::<u64>()
    }

    /// The handle `step`'s `mem_budget` allocates through.
    pub fn budget(&self, step: &'static str) -> Budget<'_> {
        Budget { ledger: self, step }
    }

    /// The locks `step` declared, in rank order: every manager first, then
    /// the GPUs by index.
    pub fn locks_of(&self, step: &str) -> Vec<LockId> {
        let mut locks: Vec<LockId> = locked(&self.locks)
            .iter()
            .filter(|(s, _)| *s == step)
            .map(|&(_, l)| l)
            .collect();
        locks.sort_unstable();
        locks.dedup();
        locks
    }

    /// The fixed workspace of `gpu`'s BLAS handle, once the final node has
    /// allocated it: a pointer and its bytes.
    pub fn workspace(&self, gpu: usize) -> Option<(*mut u8, usize)> {
        let buf = self.workspaces.get()?.get(gpu)?.as_ref()?;
        Some((buf.ptr, buf.bytes))
    }

    fn assert_open(&self, step: &str) {
        assert!(
            !self.sealed(),
            "step '{step}' budgets after the budget DAG's final node"
        );
    }

    /// Count `bytes` against the column's cap. Whether to allocate.
    fn charge(&self, step: &'static str, column: Column, bytes: u64) -> bool {
        self.assert_open(step);
        {
            let mut rows = locked(&self.rows);
            match rows
                .iter_mut()
                .find(|r| r.step == step && r.column == column)
            {
                Some(row) => row.bytes += bytes,
                None => rows.push(Row {
                    step,
                    column,
                    bytes,
                }),
            }
        }
        let over = match column {
            Column::Device(c) => {
                self.device[c].fetch_add(bytes, Ordering::AcqRel) + bytes > self.caps.device[c].used
            }
            // The host is checked at every call, before the pages are
            // touched: heap and pinned bytes are one counter against it.
            Column::Pinned(c) => {
                self.pinned[c].fetch_add(bytes, Ordering::AcqRel);
                self.host_total() > self.caps.host.used
            }
            Column::Heap => {
                self.heap.fetch_add(bytes, Ordering::AcqRel);
                self.host_total() > self.caps.host.used
            }
        };
        if over {
            self.over.store(true, Ordering::Release);
        }
        !self.over() && self.backing != Backing::Dry && bytes > 0
    }

    fn mem(&self, gpu: usize, kind: Kind) -> Mem {
        match (self.backing, kind) {
            (Backing::Real, Kind::Device) => Mem::Device(self.physical[gpu]),
            (Backing::Real, Kind::Pinned) => Mem::Pinned,
            _ => Mem::Host,
        }
    }

    fn take(&self, mem: Mem, bytes: usize) -> RawBuf {
        self.allocations.fetch_add(1, Ordering::AcqRel);
        RawBuf::new(mem, bytes)
    }

    /// The budget DAG's final node, over the steps `names` and, for each,
    /// the steps it reads. Packs each gpu and kind's scratch rows by the
    /// closure, adds each packed peak and the handle workspace to its
    /// counter and makes the same check, holds the host total with the
    /// packed pinned scratch against the host cap, and then either refuses
    /// the plan (the error carries the whole table) or allocates the slabs
    /// and each gpu's workspace.
    ///
    /// # Errors
    /// The refusal.
    ///
    /// # Panics
    /// If called twice, or if a step outside `names` declared scratch.
    pub fn finish(&self, names: &[&'static str], preds: &[Vec<usize>]) -> Status {
        let reach = pack::closure(preds);
        let rows = locked(&self.scratch).clone();
        let mut slabs = Vec::new();
        for gpu in 0..self.gpus() {
            for kind in [Kind::Device, Kind::Pinned] {
                let mut of: Vec<&ScratchRow> = rows
                    .iter()
                    .filter(|r| r.gpu == gpu && r.kind == kind)
                    .collect();
                // Name order: the table and the offsets are the same
                // whatever order the steps declared in.
                of.sort_by_key(|r| r.step);
                let regions: Vec<Region> = of
                    .iter()
                    .map(|r| Region {
                        step: names
                            .iter()
                            .position(|n| *n == r.step)
                            .unwrap_or_else(|| panic!("step '{}' is not in the plan", r.step)),
                        name: r.step,
                        bytes: r.bytes,
                    })
                    .collect();
                let packed = pack::pack(&reach, &regions);
                let counter = match kind {
                    Kind::Device => &self.device[gpu],
                    Kind::Pinned => &self.pinned[gpu],
                };
                counter.fetch_add(packed.peak, Ordering::AcqRel);
                slabs.push(Slab {
                    gpu,
                    kind,
                    steps: of
                        .iter()
                        .zip(&packed.offsets)
                        .map(|(r, &o)| (r.step, o))
                        .collect(),
                    packed,
                    buf: None,
                });
            }
            let total = self.device[gpu].fetch_add(BLAS_WORKSPACE, Ordering::AcqRel);
            if total + BLAS_WORKSPACE > self.caps.device[gpu].used {
                self.over.store(true, Ordering::Release);
            }
        }
        if self.host_total() > self.caps.host.used {
            self.over.store(true, Ordering::Release);
        }
        self.sealed.store(true, Ordering::Release);
        let refused = self.over();
        let mut workspaces = Vec::new();
        if !refused && self.backing != Backing::Dry {
            for slab in &mut slabs {
                if slab.packed.peak > 0 {
                    slab.buf =
                        Some(self.take(self.mem(slab.gpu, slab.kind), slab.packed.peak as usize));
                }
            }
            for gpu in 0..self.gpus() {
                // A host-backed plan has no handle to give a workspace.
                workspaces.push(
                    (self.backing == Backing::Real)
                        .then(|| self.take(self.mem(gpu, Kind::Device), BLAS_WORKSPACE as usize)),
                );
            }
        }
        assert!(
            self.slabs.set(slabs).is_ok() && self.workspaces.set(workspaces).is_ok(),
            "the budget DAG's final node ran twice"
        );
        if refused {
            return Err(Error::allocation(format!(
                "the plan is over its memory budget; nothing was run\n{}",
                self.table()
            )));
        }
        log::info!("{}", self.table());
        Ok(())
    }

    /// Every step's scratch region, after the final node: by gpu, kind and
    /// step name.
    pub fn regions(&self) -> Vec<RegionInfo> {
        let mut out = Vec::new();
        for slab in self.slabs.get().map_or(&[][..], Vec::as_slice) {
            for (i, &(step, offset)) in slab.steps.iter().enumerate() {
                out.push(RegionInfo {
                    step,
                    gpu: slab.gpu,
                    kind: slab.kind,
                    offset,
                    bytes: self.declared(slab, i),
                });
            }
        }
        out
    }

    fn declared(&self, slab: &Slab, i: usize) -> u64 {
        let (step, _) = slab.steps[i];
        locked(&self.scratch)
            .iter()
            .find(|r| r.step == step && r.gpu == slab.gpu && r.kind == slab.kind)
            .map_or(0, |r| r.bytes)
    }

    /// Every slab's totals, after the final node.
    pub fn slabs(&self) -> Vec<SlabInfo> {
        self.slabs
            .get()
            .map_or(&[][..], Vec::as_slice)
            .iter()
            .map(|slab| SlabInfo {
                gpu: slab.gpu,
                kind: slab.kind,
                peak: slab.packed.peak,
                sum: slab.packed.sum,
                bound: slab.packed.bound,
                bound_steps: slab
                    .packed
                    .bound_set
                    .iter()
                    .map(|&i| slab.steps[i].0)
                    .collect(),
            })
            .collect()
    }

    /// `step`'s lease: its region of every slab, for the length of one
    /// `run`. The wrapper makes it; a step is handed a reference.
    ///
    /// # Panics
    /// Before the final node, or on a dry or refused budget.
    pub fn lease(&self, step: &str) -> Lease<'_> {
        let slabs = self.slabs.get().expect("a lease is made after budget mode");
        let mut regions = Vec::new();
        for slab in slabs {
            let Some(i) = slab.steps.iter().position(|(s, _)| *s == step) else {
                continue;
            };
            let bytes = self.declared(slab, i) as usize;
            if bytes == 0 {
                continue;
            }
            let buf = slab
                .buf
                .as_ref()
                .expect("a lease is made on an accepted budget with real bytes");
            regions.push(LeaseRegion {
                gpu: slab.gpu,
                kind: slab.kind,
                // In range: the packer placed the region inside the slab.
                ptr: buf.ptr.wrapping_add(slab.steps[i].1 as usize),
                bytes,
                host: !matches!(buf.mem, Mem::Device(_)),
            });
        }
        Lease {
            regions,
            life: PhantomData,
        }
    }

    /// The table: one row per step and column for what the `Output`s hold,
    /// each slab's scratch regions with the packed peak beside the plain
    /// sum and the lower bound, the handle workspace, the totals against
    /// the caps, each cap as written beside the value used. Rows in
    /// step-name order, so the text is the same every run.
    pub fn table(&self) -> String {
        let n = self.gpus();
        let mut columns: Vec<(String, Column)> = Vec::new();
        for c in 0..n {
            columns.push((format!("device {c}"), Column::Device(c)));
            columns.push((format!("pinned {c}"), Column::Pinned(c)));
        }
        columns.push(("heap".to_owned(), Column::Heap));
        let rows = locked(&self.rows);
        let mut steps: Vec<&'static str> = rows.iter().map(|r| r.step).collect();
        steps.sort_unstable();
        steps.dedup();
        let sealed = self.sealed();
        let state = match (sealed, self.over()) {
            (false, _) => "not finished",
            (true, false) => "accepted",
            (true, true) => "REFUSED",
        };
        let mut out = format!("memory budget: {state}\n");
        let line = |out: &mut String, name: &str, cell: &dyn Fn(Column) -> String| {
            let _ = write!(out, "{name:<22}");
            for (_, column) in &columns {
                let _ = write!(out, "{:>13}", cell(*column));
            }
            out.push('\n');
        };
        line(&mut out, "step", &|c| {
            columns.iter().find(|(_, k)| *k == c).unwrap().0.clone()
        });
        let mut outputs = vec![0u64; columns.len()];
        for step in &steps {
            line(&mut out, step, &|c| {
                rows.iter()
                    .find(|r| r.step == *step && r.column == c)
                    .map_or_else(|| "-".to_owned(), |r| bytes(r.bytes))
            });
            for (i, (_, c)) in columns.iter().enumerate() {
                outputs[i] += rows
                    .iter()
                    .find(|r| r.step == *step && r.column == *c)
                    .map_or(0, |r| r.bytes);
            }
        }
        let index = |c: Column| columns.iter().position(|(_, k)| *k == c).unwrap();
        line(&mut out, "outputs (sum)", &|c| bytes(outputs[index(c)]));
        let slab_of = |c: Column| {
            let (gpu, kind) = match c {
                Column::Device(gpu) => (gpu, Kind::Device),
                Column::Pinned(gpu) => (gpu, Kind::Pinned),
                Column::Heap => return None,
            };
            self.slabs
                .get()?
                .iter()
                .find(|s| s.gpu == gpu && s.kind == kind)
        };
        let slab_cell = |c: Column, f: &dyn Fn(&Slab) -> u64| match slab_of(c) {
            Some(slab) => bytes(f(slab)),
            None => "-".to_owned(),
        };
        if sealed {
            line(&mut out, "scratch (packed)", &|c| {
                slab_cell(c, &|s| s.packed.peak)
            });
            line(&mut out, "  plain sum", &|c| {
                slab_cell(c, &|s| s.packed.sum)
            });
            line(&mut out, "  lower bound", &|c| {
                slab_cell(c, &|s| s.packed.bound)
            });
            line(&mut out, "blas workspace", &|c| match c {
                Column::Device(_) => bytes(BLAS_WORKSPACE),
                _ => "-".to_owned(),
            });
        }
        line(&mut out, "total", &|c| {
            bytes(match c {
                Column::Device(gpu) => self.device_total(gpu),
                Column::Pinned(gpu) => self.pinned_total(gpu),
                Column::Heap => self.heap_total(),
            })
        });
        line(&mut out, "cap used", &|c| match c {
            Column::Device(gpu) => bytes(self.caps.device[gpu].used),
            _ => "-".to_owned(),
        });
        line(&mut out, "cap as written", &|c| match c {
            Column::Device(gpu) => self.caps.device[gpu]
                .written
                .map_or_else(|| "default".to_owned(), bytes),
            _ => "-".to_owned(),
        });
        let _ = writeln!(
            out,
            "host: heap + pinned {} against {} used ({} as written)",
            bytes(self.host_total()),
            bytes(self.caps.host.used),
            self.caps
                .host
                .written
                .map_or_else(|| "default".to_owned(), bytes)
        );
        if let Some(slabs) = self.slabs.get() {
            for slab in slabs.iter().filter(|s| !s.steps.is_empty()) {
                let kind = match slab.kind {
                    Kind::Device => "device",
                    Kind::Pinned => "pinned",
                };
                let bound: Vec<&str> = slab
                    .packed
                    .bound_set
                    .iter()
                    .map(|&i| slab.steps[i].0)
                    .collect();
                let _ = writeln!(
                    out,
                    "scratch {kind} {}: packed {}, plain sum {}, lower bound {} ({})",
                    slab.gpu,
                    bytes(slab.packed.peak),
                    bytes(slab.packed.sum),
                    bytes(slab.packed.bound),
                    bound.join(" + ")
                );
                for (i, (step, offset)) in slab.steps.iter().enumerate() {
                    let size = self.declared(slab, i);
                    let _ = writeln!(
                        out,
                        "  {step:<20}{:>13} at offset {}",
                        bytes(size),
                        bytes(*offset)
                    );
                }
            }
        }
        let locks = locked(&self.locks);
        let mut users: Vec<String> = locks
            .iter()
            .map(|(step, lock)| format!("{step}: {lock:?}"))
            .collect();
        users.sort();
        users.dedup();
        let _ = write!(
            out,
            "locks declared: {}",
            if users.is_empty() {
                "none".to_owned()
            } else {
                users.join(", ")
            }
        );
        out
    }
}

/// `n` bytes in the largest binary unit that keeps three digits.
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if n == u64::MAX {
        return "none".to_owned();
    }
    if n < 1024 {
        return format!("{n} B");
    }
    let mut value = n as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

/// What a step's `mem_budget` allocates through.
pub struct Budget<'a> {
    ledger: &'a Ledger,
    step: &'static str,
}

impl Budget<'_> {
    /// The gpus of the run.
    pub fn gpus(&self) -> usize {
        self.ledger.gpus()
    }

    /// The device cap `gpu` is held to: what a step's default limit is a
    /// fraction of.
    pub fn device_cap(&self, gpu: usize) -> u64 {
        self.ledger.caps.device[gpu].used
    }

    /// `len` elements of device memory on `gpu`, the step's `Output`'s for
    /// the run. Uninitialized.
    pub fn device<T: Copy>(&self, gpu: usize, len: usize) -> DeviceBuf<T> {
        let size = len * size_of::<T>();
        let raw = self
            .ledger
            .charge(self.step, Column::Device(gpu), size as u64)
            .then(|| self.ledger.take(self.ledger.mem(gpu, Kind::Device), size));
        DeviceBuf {
            raw,
            len,
            of: PhantomData,
        }
    }

    /// `len` elements of host heap that cannot grow, every page touched
    /// here: zeroed memory is handed out lazily, so an untouched slab would
    /// "succeed" without existing and fail at first touch mid-run. The host
    /// cap is checked before the slab is made.
    pub fn heap<T: Copy + Default>(&self, len: usize) -> HeapSlab<T> {
        let size = (len * size_of::<T>()) as u64;
        if !self.ledger.charge(self.step, Column::Heap, size) {
            return HeapSlab(Box::default());
        }
        let mut slab = vec![T::default(); len].into_boxed_slice();
        let stride = (4096 / size_of::<T>().max(1)).max(1);
        huge_pages(slab.as_mut_ptr().cast(), size as usize);
        for i in (0..len).step_by(stride) {
            // SAFETY: `i` is in range; the write is of the value already
            // there, volatile so the page is really touched.
            unsafe {
                let p = slab.as_mut_ptr().add(i);
                std::ptr::write_volatile(p, std::ptr::read(p));
            }
        }
        self.ledger.allocations.fetch_add(1, Ordering::AcqRel);
        HeapSlab(slab)
    }

    /// Declare `bytes` of step scratch on `gpu`: memory the step needs
    /// during its `run` and nothing reads afterwards. Records a row;
    /// charges and allocates nothing. The final node packs it, and `run` is
    /// lent the region. Calls for one gpu and kind add up.
    pub fn scratch(&self, gpu: usize, kind: Kind, bytes: usize) {
        self.ledger.assert_open(self.step);
        assert!(gpu < self.ledger.gpus(), "gpu {gpu} is not in the run");
        let bytes = (bytes as u64).div_ceil(ALIGN) * ALIGN;
        let mut rows = locked(&self.ledger.scratch);
        match rows
            .iter_mut()
            .find(|r| r.step == self.step && r.gpu == gpu && r.kind == kind)
        {
            Some(row) => row.bytes += bytes,
            None => rows.push(ScratchRow {
                step: self.step,
                gpu,
                kind,
                bytes,
            }),
        }
    }

    /// Declare the locks the step takes: the wrapper takes them in rank
    /// order before `run` and releases them after the step-end sync. A step
    /// that calls the library on a GPU declares that GPU's lock; a reader
    /// that calls a manager's methods declares the manager's.
    pub fn locks(&self, locks: impl IntoIterator<Item = LockId>) {
        self.ledger.assert_open(self.step);
        let mut held = locked(&self.ledger.locks);
        for lock in locks {
            if let LockId::Gpu(gpu) = lock {
                assert!(gpu < self.ledger.gpus(), "gpu {gpu} is not in the run");
            }
            held.push((self.step, lock));
        }
    }

    /// `len` elements of page-locked host memory an `Output` owns, staged
    /// for `gpu`: charged to that GPU's pinned column and held against the
    /// host.
    pub fn pinned<T: Plain>(&self, gpu: usize, len: usize) -> PinnedBuf<T> {
        let size = len * size_of::<T>();
        let raw = self
            .ledger
            .charge(self.step, Column::Pinned(gpu), size as u64)
            .then(|| self.ledger.take(self.ledger.mem(gpu, Kind::Pinned), size));
        PinnedBuf {
            raw,
            len,
            of: PhantomData,
        }
    }
}

/// Device memory a step's `Output` owns: fixed size, freed with it. Empty
/// (a null pointer of the asked length) under a dry or refused budget.
pub struct DeviceBuf<T> {
    raw: Option<RawBuf>,
    len: usize,
    of: PhantomData<T>,
}

impl<T> DeviceBuf<T> {
    /// Whether the buffer was really allocated.
    pub fn is_allocated(&self) -> bool {
        self.raw.is_some()
    }

    /// The bytes as host memory, under [`Backing::Host`] only: how the
    /// core's tests read and write a "device" buffer on no gpu.
    pub fn host(&mut self) -> Option<&mut [T]> {
        let raw = self.raw.as_ref().filter(|r| r.mem == Mem::Host)?;
        // SAFETY: `len` zero-initialized elements of plain host memory,
        // and `&mut self` is the one user.
        Some(unsafe { std::slice::from_raw_parts_mut(raw.ptr.cast(), self.len) })
    }
}

// SAFETY: `len` elements of device memory, live while `self` is.
unsafe impl<T> DeviceSlice<T> for DeviceBuf<T> {
    fn as_ptr(&self) -> *const T {
        self.raw
            .as_ref()
            .map_or(std::ptr::null(), |r| r.ptr.cast_const().cast())
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T> DeviceSliceMut<T> for DeviceBuf<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.raw
            .as_ref()
            .map_or(std::ptr::null_mut(), |r| r.ptr.cast())
    }
}

/// Budgeted host heap: a boxed slice, so it has no `push`, `reserve` or
/// `extend`. Only a [`Budget`] makes one.
pub struct HeapSlab<T>(Box<[T]>);

impl<T> Deref for HeapSlab<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.0
    }
}

impl<T> DerefMut for HeapSlab<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

/// The element types scratch is carved into: plain data, every bit pattern
/// a value, so a NaN-filled region is still valid to hold.
pub trait Plain: Copy + Send + Sync + 'static {}
impl Plain for f64 {}
impl Plain for i32 {}
impl Plain for u32 {}
impl Plain for u8 {}

/// One tenant of a step's scratch region: `len` elements at a byte offset.
/// Made by a [`Layout`], consumed by the [`Lease`] that carves it, so the
/// tenants of one layout cannot be handed out twice.
#[derive(Debug)]
pub struct Span<T> {
    offset: usize,
    len: usize,
    of: PhantomData<T>,
}

impl<T> Span<T> {
    /// The elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
    pub fn new() -> Layout {
        Layout::default()
    }

    /// The next tenant: `len` elements of `T`, at the next 64-byte boundary.
    pub fn span<T>(&mut self, len: usize) -> Span<T> {
        let offset = self.bytes.div_ceil(64) * 64;
        self.bytes = offset + len * size_of::<T>();
        Span {
            offset,
            len,
            of: PhantomData,
        }
    }

    /// The region's bytes: what `mem_budget` declares.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

struct LeaseRegion {
    gpu: usize,
    kind: Kind,
    ptr: *mut u8,
    bytes: usize,
    /// Whether the bytes are host memory: pinned, or a host-backed plan's.
    host: bool,
}

/// A step's scratch for the length of one `run`: its region of each slab.
/// It borrows from the ledger, and what it carves borrows from it, so
/// nothing carved can be stored in an `Output`, which is `'static`:
///
/// ```compile_fail,E0521
/// use fb_v04::budget::{Layout, Lease};
/// fn keep(lease: &Lease<'_>, output: &mut Vec<&'static mut [f64]>) {
///     let mut layout = Layout::new();
///     output.push(lease.pinned(0, layout.span::<f64>(4)));
/// }
/// ```
///
/// The same with the borrow's own lifetime is a program:
///
/// ```no_run
/// use fb_v04::budget::{Layout, Lease};
/// fn carve<'a>(lease: &'a Lease<'_>, local: &mut Vec<&'a mut [f64]>) {
///     let mut layout = Layout::new();
///     local.push(lease.pinned(0, layout.span::<f64>(4)));
/// }
/// ```
pub struct Lease<'r> {
    regions: Vec<LeaseRegion>,
    life: PhantomData<&'r Ledger>,
}

// SAFETY: a lease is addresses; a step's inner tasks carve disjoint spans
// of it from several threads.
unsafe impl Send for Lease<'_> {}
// SAFETY: as above.
unsafe impl Sync for Lease<'_> {}

impl Lease<'_> {
    fn region(&self, gpu: usize, kind: Kind) -> Option<&LeaseRegion> {
        self.regions.iter().find(|r| r.gpu == gpu && r.kind == kind)
    }

    /// The bytes the step declared on `gpu` for `kind`.
    pub fn bytes(&self, gpu: usize, kind: Kind) -> usize {
        self.region(gpu, kind).map_or(0, |r| r.bytes)
    }

    /// The step's whole region of `gpu`'s slab of `kind`: the wrapper's
    /// NaN fill takes it. Whether it is host memory comes with it.
    pub fn whole(&self, gpu: usize, kind: Kind) -> Option<(*mut u8, usize, bool)> {
        self.region(gpu, kind).map(|r| (r.ptr, r.bytes, r.host))
    }

    /// Every region as `(gpu, kind)`.
    pub fn kinds(&self) -> Vec<(usize, Kind)> {
        self.regions.iter().map(|r| (r.gpu, r.kind)).collect()
    }

    /// Set every byte of every host-addressable region to `0xFF`: NaN as an
    /// `f64`, -1 as an `i32`. The wrapper's test-mode fill; the device
    /// regions of a real plan are the bus's, on the step's stream.
    pub fn fill_host(&self) {
        for r in self.regions.iter().filter(|r| r.host) {
            // SAFETY: the region is `bytes` of host memory, this step's
            // for the run that has not started.
            unsafe { std::ptr::write_bytes(r.ptr, 0xFF, r.bytes) };
        }
    }

    fn carve<T>(&self, gpu: usize, kind: Kind, span: &Span<T>) -> *mut T {
        let held = self.bytes(gpu, kind);
        let end = span.offset + span.len * size_of::<T>();
        assert!(
            end <= held,
            "a step's run carves {end} bytes of its {kind:?} scratch on gpu {gpu}, but its \
             mem_budget declared {held}: the two must come from one layout"
        );
        if span.len == 0 {
            return std::ptr::NonNull::dangling().as_ptr();
        }
        let region = self.region(gpu, kind).expect("held is above zero");
        region.ptr.wrapping_add(span.offset).cast()
    }

    /// The tenant `span` of the step's device scratch on `gpu`.
    ///
    /// # Panics
    /// If the span ends past what `mem_budget` declared.
    pub fn device<T: Plain>(&self, gpu: usize, span: Span<T>) -> ScratchBuf<'_, T> {
        let host = self.region(gpu, Kind::Device).is_some_and(|r| r.host);
        // THE ONE UNSAFE OF THE ARENA. The closure invariant: this region
        // overlaps another step's only if the DAG orders the two steps
        // (`pack::pack`), and a finished node is finished device work (the
        // wrapper's step-end sync), so while this lease lives no other step
        // touches these bytes. Within the step, the spans of one `Layout`
        // are disjoint and each is consumed here.
        ScratchBuf {
            ptr: self.carve(gpu, Kind::Device, &span),
            len: span.len,
            host,
            life: PhantomData,
        }
    }

    /// The tenant `span` of the step's pinned scratch on `gpu`, as a host
    /// slice. Its contents on entry are undefined.
    ///
    /// # Panics
    /// If the span ends past what `mem_budget` declared.
    #[allow(
        clippy::mut_from_ref,
        reason = "the span is consumed: one slice per tenant"
    )]
    pub fn pinned<T: Plain>(&self, gpu: usize, span: Span<T>) -> &mut [T] {
        let ptr = self.carve(gpu, Kind::Pinned, &span);
        // SAFETY: as `device`'s: the closure invariant keeps other steps
        // off these bytes, the span is this layout's alone, and `T` is
        // plain data, valid at any bit pattern.
        unsafe { std::slice::from_raw_parts_mut(ptr, span.len) }
    }
}

/// A tenant of a step's device scratch, for the length of its `run`.
pub struct ScratchBuf<'a, T> {
    ptr: *mut T,
    len: usize,
    host: bool,
    life: PhantomData<&'a mut [T]>,
}

// SAFETY: device memory is not tied to a host thread.
unsafe impl<T: Send> Send for ScratchBuf<'_, T> {}

impl<T> ScratchBuf<'_, T> {
    /// The bytes as host memory, under [`Backing::Host`] only.
    pub fn host(&mut self) -> Option<&mut [T]> {
        // SAFETY: host memory of a host-backed plan, `len` plain elements,
        // and `&mut self` is the one user.
        self.host
            .then(|| unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) })
    }
}

// SAFETY: `len` elements of the step's region, live while the lease is.
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

/// A manager: a shared product with mutable state of its own, reached by
/// its readers under its lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ManagerId {
    /// The `EriManager`.
    Eri,
}

/// A lock, in the one global rank: every manager first, by its id, then
/// the GPUs by index. `Ord` is derived in that order, and the wrapper takes
/// a step's declared set ascending, so no step holds a higher lock while it
/// waits for a lower one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LockId {
    /// A manager's lock.
    Manager(ManagerId),
    /// A GPU's `GpuResources`.
    Gpu(usize),
}

/// Page-locked host memory an `Output` owns.
pub struct PinnedBuf<T> {
    raw: Option<RawBuf>,
    len: usize,
    of: PhantomData<T>,
}

impl PinnedBuf<u8> {
    /// The tenant `span` cut from this buffer's layout, for a copy
    /// enqueued on a stream. The manager's lock orders the buffer's users,
    /// not the type: the holder must not rewrite a tenant an enqueued copy
    /// still reads.
    ///
    /// # Panics
    /// If the tenant is past the buffer.
    #[allow(clippy::mut_from_ref, reason = "disjoint tenants of one layout")]
    pub fn tenant<U: Plain>(&self, span: &Span<U>) -> &mut [U] {
        assert!(
            span.offset + span.len * size_of::<U>() <= self.len,
            "past the pinned buffer"
        );
        if span.len == 0 {
            return &mut [];
        }
        let raw = self
            .raw
            .as_ref()
            .expect("an accepted budget with real bytes");
        // SAFETY: inside the buffer; a layout's spans are disjoint and
        // aligned to 64 bytes, and `U` is plain data.
        unsafe { std::slice::from_raw_parts_mut(raw.ptr.add(span.offset).cast::<U>(), span.len) }
    }
}

impl DeviceBuf<u8> {
    /// The tenant `span` cut from this buffer's layout, writable, for a
    /// manager that lays several tenants out in one buffer it owns. The
    /// manager's lock orders the buffer's users.
    ///
    /// # Panics
    /// If the tenant is past the buffer.
    pub fn tenant<U: Plain>(&self, span: &Span<U>) -> PartMut<'_, U> {
        assert!(
            span.offset + span.len * size_of::<U>() <= self.len,
            "past the device buffer"
        );
        let ptr = self
            .raw
            .as_ref()
            .map_or(std::ptr::null_mut(), |r| r.ptr.wrapping_add(span.offset));
        // SAFETY: inside the buffer; a layout's spans are disjoint, and
        // the manager's lock orders the buffer's users.
        unsafe { PartMut::from_raw(ptr.cast::<U>(), span.len) }
    }
}

impl<T> Clone for Span<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Span<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_are_printed_in_binary_units() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(bytes(3 << 30), "3.00 GiB");
    }

    #[test]
    fn a_cap_is_clamped_to_the_reading() {
        let gib = 1u64 << 30;
        // Defaulted: the reading less the margin.
        assert_eq!(Cap::resolve(None, Some(10 * gib), gib).used, 9 * gib);
        // Explicit and within the machine: as written.
        assert_eq!(
            Cap::resolve(Some(4 * gib), Some(10 * gib), gib).used,
            4 * gib
        );
        // Explicit and past it: the reading itself, no margin taken.
        assert_eq!(
            Cap::resolve(Some(64 * gib), Some(10 * gib), gib).used,
            10 * gib
        );
        assert_eq!(device_margin(10 * gib), 512 << 20);
        assert_eq!(device_margin(16 * gib), 16 * gib / 20);
        assert_eq!(host_margin(8 * gib), gib);
        assert_eq!(host_margin(64 * gib), 64 * gib / 20);
    }
}
