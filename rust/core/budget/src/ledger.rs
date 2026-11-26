//! The ledger: caps, counters, the `over` flag, the per-step table, the
//! final node and, after it, the scratch slabs; and the [`Budget`] handle
//! a step's `mem_budget` allocates through.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use contracts::{debug_requires, requires};
use fb_core_error_handling::{Error, Status};
use fb_core_log::{eq, le, lt};
use fb_core_pack::{self as pack, Packed, Region};

use crate::buffers::{DeviceBuf, HeapSlab, PinnedBuf, Plain};
use crate::lease::{Lease, LeaseRegion};
use crate::raw::{Mem, RawBuf, huge_pages};
use crate::{ALIGN, Backing, Caps, Kind, LockId, bytes};

/// The handle workspace per gpu, in the counters' unit.
const WORKSPACE: u64 = fb_core_device::HANDLE_WORKSPACE_BYTES as u64;

/// A column of the table: what a counter sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Column {
    Device(usize),
    Pinned(usize),
    Heap,
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
    /// The declaring steps, in name order, with their declared bytes and
    /// their offsets in `packed`.
    steps: Vec<(&'static str, u64, u64)>,
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
    pub(crate) physical: Vec<i32>,
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
    workspaces: Mutex<Vec<Option<DeviceBuf<u8>>>>,
    /// The steps holding a lease now.
    pub(crate) leased: Mutex<Vec<String>>,
    /// The last execute's peak heap growth, `u64::MAX` before one was
    /// measured.
    heap_growth: AtomicU64,
}

/// `m`'s guard, poisoned or not: every critical section here leaves its
/// vector whole.
pub(crate) fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Ledger {
    /// An empty ledger over `caps`, one gpu per entry of `physical` (the
    /// device each gpu is on; two gpus may name one device). With
    /// `nan_fill`, the wrapper fills each lease with NaN before `run`.
    #[debug_requires(eq!(caps.device.len(), physical.len()), "one device cap per gpu")]
    #[must_use]
    pub fn new(caps: Caps, backing: Backing, physical: Vec<i32>, nan_fill: bool) -> Ledger {
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
            workspaces: Mutex::new(Vec::new()),
            leased: Mutex::new(Vec::new()),
            heap_growth: AtomicU64::new(u64::MAX),
        }
    }

    /// Where the bytes come from.
    #[must_use]
    pub fn backing(&self) -> Backing {
        self.backing
    }

    /// The gpus.
    #[must_use]
    pub fn gpus(&self) -> usize {
        self.physical.len()
    }

    /// The caps.
    #[must_use]
    pub fn caps(&self) -> &Caps {
        &self.caps
    }

    /// Whether the wrapper fills each lease with NaN before `run`.
    #[must_use]
    pub fn nan_fill(&self) -> bool {
        self.nan_fill
    }

    /// Whether some charge went over its cap.
    #[must_use]
    pub fn over(&self) -> bool {
        self.over.load(Ordering::Acquire)
    }

    /// Whether [`Ledger::finish`] has run: nothing is budgeted after it.
    #[must_use]
    pub fn sealed(&self) -> bool {
        self.sealed.load(Ordering::Acquire)
    }

    /// The allocations this ledger has made: the same before and after an
    /// execute, since `run` has no `Budget`.
    #[must_use]
    pub fn allocations(&self) -> u64 {
        self.allocations.load(Ordering::Acquire)
    }

    /// The device bytes charged to `gpu` so far; after the final node, the
    /// packed scratch and the handle workspace are in it.
    #[must_use]
    pub fn device_total(&self, gpu: usize) -> u64 {
        self.device[gpu].load(Ordering::Acquire)
    }

    /// The pinned bytes charged for `gpu` so far.
    #[must_use]
    pub fn pinned_total(&self, gpu: usize) -> u64 {
        self.pinned[gpu].load(Ordering::Acquire)
    }

    /// The budgeted heap so far.
    #[must_use]
    pub fn heap_total(&self) -> u64 {
        self.heap.load(Ordering::Acquire)
    }

    fn host_total(&self) -> u64 {
        self.heap_total() + (0..self.gpus()).map(|c| self.pinned_total(c)).sum::<u64>()
    }

    /// Run `execute`, an execute of the plan over this ledger, and keep
    /// the plain heap's peak growth during it, as the
    /// [`AllocationRecord`](crate::AllocationRecord) saw it, for the
    /// table's unbudgeted row: nothing when the binary does not install
    /// the record. Process wide, so another plan's execute meanwhile
    /// counts too; but its start does not reset this one's: an execute
    /// that starts while this one runs leaves the peak it reads alone.
    ///
    /// ```
    /// use std::sync::mpsc::channel;
    /// use fb_core_budget::{Backing, Cap, Caps, Ledger};
    ///
    /// #[global_allocator]
    /// static RECORD: fb_core_budget::AllocationRecord = fb_core_budget::AllocationRecord;
    ///
    /// let ledger = || {
    ///     let cap = Cap { written: Some(1 << 30), used: 1 << 30 };
    ///     Ledger::new(Caps { device: vec![cap], host: cap }, Backing::Host, vec![0], false)
    /// };
    /// let (first, second) = (ledger(), ledger());
    /// let (freed, on_freed) = channel();
    /// let (ended, on_ended) = channel();
    /// std::thread::scope(|s| {
    ///     let first = &first;
    ///     s.spawn(move || {
    ///         first.measure_execute(|| {
    ///             drop(std::hint::black_box(vec![0u8; 512 << 10]));
    ///             freed.send(()).unwrap();
    ///             on_ended.recv().unwrap();
    ///         })
    ///     });
    ///     // The second execute begins after the first's allocation is
    ///     // freed, and ends before the first does.
    ///     on_freed.recv().unwrap();
    ///     second.measure_execute(|| ());
    ///     ended.send(()).unwrap();
    /// });
    /// assert!(first.heap_growth().unwrap() >= 512 << 10);
    /// ```
    pub fn measure_execute<R>(&self, execute: impl FnOnce() -> R) -> R {
        let (value, grown) = crate::record::growth(execute);
        if let Some(grown) = grown {
            self.heap_growth.store(grown, Ordering::Release);
        }
        value
    }

    /// The last execute's peak heap growth, if one was measured
    /// ([`Ledger::measure_execute`]).
    #[must_use]
    pub fn heap_growth(&self) -> Option<u64> {
        Some(self.heap_growth.load(Ordering::Acquire)).filter(|&g| g != u64::MAX)
    }

    /// The handle `step`'s `mem_budget` allocates through.
    #[must_use]
    pub fn budget(&self, step: &'static str) -> Budget<'_> {
        Budget { ledger: self, step }
    }

    /// The locks `step` declared, in rank order: every manager first, then
    /// the GPUs by index.
    #[must_use]
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

    /// `gpu`'s handle workspace, once the final node has allocated it on
    /// an accepted real plan, moved out for its `GpuResources` to own
    /// (`bind_workspace`); `None` before, after the first take, or on a
    /// host-backed, dry or refused plan.
    #[must_use]
    pub fn take_workspace(&self, gpu: usize) -> Option<DeviceBuf<u8>> {
        locked(&self.workspaces).get_mut(gpu)?.take()
    }

    /// # Panics
    /// After the final node.
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

    #[track_caller]
    fn take(&self, mem: Mem, bytes: usize) -> RawBuf {
        self.allocations.fetch_add(1, Ordering::AcqRel);
        RawBuf::new(mem, bytes)
    }

    /// The budget DAG's final node, over the steps `names` and, for each,
    /// the steps it reads (`preds[i]`, indices into `names`). Packs each
    /// gpu and kind's scratch rows by the closure, adds each packed peak
    /// and the handle workspace to its counter and makes the same check,
    /// holds the host total with the packed pinned scratch against the
    /// host cap, and then either refuses the plan or allocates the slabs
    /// and each gpu's workspace and logs the table at `info`.
    ///
    /// # Errors
    /// The refusal, an `Allocation` error carrying the whole table, the
    /// same whatever order the steps budgeted in.
    ///
    /// # Panics
    /// If called twice, if a step outside `names` declared scratch, or if
    /// the packer placed a region past its slab.
    #[debug_requires(eq!(names.len(), preds.len()), "one predecessor list per step")]
    #[track_caller]
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
                // Every region inside its slab, whatever the packer's
                // quality: a lease's slices rest on it.
                for (r, &offset) in of.iter().zip(&packed.offsets) {
                    let end = offset.checked_add(r.bytes);
                    assert!(
                        end.is_some_and(|end| end <= packed.peak),
                        "step '{}' was packed past its slab",
                        r.step
                    );
                }
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
                        .map(|(r, &o)| (r.step, r.bytes, o))
                        .collect(),
                    packed,
                    buf: None,
                });
            }
            let total = self.device[gpu].fetch_add(WORKSPACE, Ordering::AcqRel);
            if total + WORKSPACE > self.caps.device[gpu].used {
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
                    let peak = usize::try_from(slab.packed.peak).expect("a slab fits the host");
                    slab.buf = Some(self.take(self.mem(slab.gpu, slab.kind), peak));
                }
            }
            for gpu in 0..self.gpus() {
                // A host-backed plan has no handle to give a workspace.
                workspaces.push((self.backing == Backing::Real).then(|| {
                    let raw = self.take(self.mem(gpu, Kind::Device), WORKSPACE as usize);
                    DeviceBuf::new(Some(raw), WORKSPACE as usize)
                }));
            }
        }
        assert!(
            self.slabs.set(slabs).is_ok(),
            "the budget DAG's final node ran twice"
        );
        *locked(&self.workspaces) = workspaces;
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
    #[must_use]
    pub fn regions(&self) -> Vec<RegionInfo> {
        let mut out = Vec::new();
        for slab in self.slabs.get().map_or(&[][..], Vec::as_slice) {
            for &(step, bytes, offset) in &slab.steps {
                out.push(RegionInfo {
                    step,
                    gpu: slab.gpu,
                    kind: slab.kind,
                    offset,
                    bytes,
                });
            }
        }
        out
    }

    /// Every slab's totals, after the final node.
    #[must_use]
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
    /// Before the final node, on a dry or refused budget, or while `step`
    /// already holds a lease.
    #[must_use]
    pub fn lease(&self, step: &str) -> Lease<'_> {
        let slabs = self.slabs.get().expect("a lease is made after budget mode");
        let mut regions = Vec::new();
        for slab in slabs {
            let Some(&(_, bytes, offset)) = slab.steps.iter().find(|(s, ..)| *s == step) else {
                continue;
            };
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
                // In range: `finish` checked every region against its slab.
                ptr: buf.ptr.wrapping_add(offset as usize),
                bytes: bytes as usize,
                host: buf.is_host(),
            });
        }
        {
            let mut leased = locked(&self.leased);
            assert!(
                !leased.iter().any(|s| s == step),
                "step '{step}' already holds a lease"
            );
            leased.push(step.to_owned());
        }
        Lease {
            ledger: self,
            step: step.to_owned(),
            regions,
            carved: Mutex::new(Vec::new()),
        }
    }

    /// The table: one row per step and column for what the `Output`s hold,
    /// each slab's scratch regions with the packed peak beside the plain
    /// sum and the lower bound, the handle workspace, the totals against
    /// the caps, each cap as written beside the value used. Rows in
    /// step-name order, so the text is the same every run.
    #[must_use]
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
        let index = |c: Column| columns.iter().position(|(_, k)| *k == c);
        line(&mut out, "step", &|c| {
            index(c).map_or_else(String::new, |i| columns[i].0.clone())
        });
        let cell_of = |step: &str, c: Column| {
            rows.iter()
                .find(|r| r.step == step && r.column == c)
                .map(|r| r.bytes)
        };
        let mut outputs = vec![0u64; columns.len()];
        for step in &steps {
            line(&mut out, step, &|c| {
                cell_of(step, c).map_or_else(|| "-".to_owned(), bytes)
            });
            for (i, (_, c)) in columns.iter().enumerate() {
                outputs[i] += cell_of(step, *c).unwrap_or(0);
            }
        }
        line(&mut out, "outputs (sum)", &|c| {
            bytes(index(c).map_or(0, |i| outputs[i]))
        });
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
                Column::Device(_) => bytes(WORKSPACE),
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
        let _ = writeln!(
            out,
            "heap, unbudgeted: {}",
            self.heap_growth().map_or_else(
                || "not measured (no execute yet, or no allocation record)".to_owned(),
                |g| format!("the last execute's peak growth {}", bytes(g))
            )
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
                for &(step, size, offset) in &slab.steps {
                    let _ = writeln!(
                        out,
                        "  {step:<20}{:>13} at offset {}",
                        bytes(size),
                        bytes(offset)
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

/// What a step's `mem_budget` allocates through.
pub struct Budget<'a> {
    ledger: &'a Ledger,
    step: &'static str,
}

impl Budget<'_> {
    /// The gpus of the run.
    #[must_use]
    pub fn gpus(&self) -> usize {
        self.ledger.gpus()
    }

    /// The device cap `gpu` is held to: what a step's default limit is a
    /// fraction of.
    #[must_use]
    pub fn device_cap(&self, gpu: usize) -> u64 {
        self.ledger.caps.device[gpu].used
    }

    /// `len` elements of device memory on `gpu`, the step's `Output`'s for
    /// the run. Uninitialized on a real plan, zeroed on a host-backed one.
    ///
    /// # Panics
    /// After the final node.
    #[requires(
        le!(len, isize::MAX as usize / size_of::<T>().max(1)),
        "the buffer's bytes fit an allocation"
    )]
    #[debug_requires(lt!(gpu, self.gpus()), "the gpu is in the run")]
    #[track_caller]
    pub fn device<T: Plain>(&self, gpu: usize, len: usize) -> DeviceBuf<T> {
        let size = len * size_of::<T>();
        let raw = self
            .ledger
            .charge(self.step, Column::Device(gpu), size as u64)
            .then(|| self.ledger.take(self.ledger.mem(gpu, Kind::Device), size));
        DeviceBuf::new(raw, len)
    }

    /// `len` elements of page-locked host memory an `Output` owns, staged
    /// for `gpu`: charged to that GPU's pinned column and held against the
    /// host. Filled with `0xFF`.
    ///
    /// # Panics
    /// After the final node.
    #[requires(
        le!(len, isize::MAX as usize / size_of::<T>().max(1)),
        "the buffer's bytes fit an allocation"
    )]
    #[debug_requires(lt!(gpu, self.gpus()), "the gpu is in the run")]
    #[track_caller]
    pub fn pinned<T: Plain>(&self, gpu: usize, len: usize) -> PinnedBuf<T> {
        let size = len * size_of::<T>();
        let raw = self
            .ledger
            .charge(self.step, Column::Pinned(gpu), size as u64)
            .then(|| self.ledger.take(self.ledger.mem(gpu, Kind::Pinned), size));
        PinnedBuf::new(raw, len)
    }

    /// `len` elements of host heap that cannot grow, huge pages asked for
    /// and every page touched here: zeroed memory is handed out lazily, so
    /// an untouched slab would "succeed" without existing and fail at first
    /// touch mid-run. The host cap is checked before the slab is made, and
    /// a slab past it is never made: empty.
    ///
    /// # Panics
    /// After the final node.
    #[requires(
        le!(len, isize::MAX as usize / size_of::<T>().max(1)),
        "the slab's bytes fit an allocation"
    )]
    pub fn heap<T: Copy + Default>(&self, len: usize) -> HeapSlab<T> {
        let size = len * size_of::<T>();
        if !self.ledger.charge(self.step, Column::Heap, size as u64) {
            return HeapSlab(Box::default());
        }
        let mut slab = vec![T::default(); len].into_boxed_slice();
        huge_pages(slab.as_mut_ptr().cast(), size);
        let stride = (4096 / size_of::<T>().max(1)).max(1);
        for i in (0..len).step_by(stride) {
            let p: *mut T = &raw mut slab[i];
            // SAFETY: `p` is element `i` of the slab, in range; the write
            // is of the value already there, volatile so the page is
            // really touched.
            let value = unsafe { std::ptr::read(p) };
            // SAFETY: as above.
            unsafe { std::ptr::write_volatile(p, value) };
        }
        self.ledger.allocations.fetch_add(1, Ordering::AcqRel);
        HeapSlab(slab)
    }

    /// Declare `bytes` of step scratch on `gpu`: memory the step needs
    /// during its `run` and nothing reads afterwards. Records a row,
    /// rounded up to the slab's alignment; charges and allocates nothing.
    /// The final node packs it, and `run` is lent the region. Calls for one
    /// gpu and kind add up.
    ///
    /// # Panics
    /// After the final node.
    #[debug_requires(lt!(gpu, self.gpus()), "the gpu is in the run")]
    pub fn scratch(&self, gpu: usize, kind: Kind, bytes: usize) {
        self.ledger.assert_open(self.step);
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
    ///
    /// # Panics
    /// After the final node.
    #[debug_requires(
        locks.clone().into_iter().all(|l| match l {
            LockId::Gpu(gpu) => gpu < self.gpus(),
            LockId::Manager(_) => true,
        }),
        "every gpu lock is a gpu of the run"
    )]
    pub fn locks<I>(&self, locks: I)
    where
        I: IntoIterator<Item = LockId> + Clone,
    {
        self.ledger.assert_open(self.step);
        locked(&self.ledger.locks).extend(locks.into_iter().map(|lock| (self.step, lock)));
    }
}
