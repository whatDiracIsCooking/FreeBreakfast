//! The memory budget: the ledger of caps and counters, the [`Budget`] handle
//! a step's `mem_budget` allocates through, and the three buffer types it
//! hands back.
//!
//! Each [`Budget`] call, in order: adds the bytes to the kind's atomic
//! counter and to the step's row of the table; checks the counter against
//! the cap, and sets the ledger's `over` flag past it; allocates, unless the
//! flag is set. Workspace is the one kind that is a max and not a sum, and
//! nothing is handed back for it: [`Ledger::finish`], the budget DAG's last
//! node, adds each device's max to its counter, checks again, and either
//! refuses the whole plan with the table or binds the workspace.
//!
//! No step holds a `Budget` in execute mode: `run` is never given one, and
//! the buffers' constructors are private to this module.

use std::ffi::c_void;
use std::fmt::Write as _;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use fb_core_error_handling::{Error, Status, check_or_abort};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::gpu_status;

/// What a row of the table counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Device memory on one device: a sum.
    Device(usize),
    /// Page-locked host memory, staged for one device: a sum.
    Pinned(usize),
    /// Host heap: one sum for the node.
    Heap,
    /// Library workspace on one device: a max.
    Workspace(usize),
}

/// The caps the counters are held against, in bytes.
#[derive(Clone, Debug)]
pub struct Caps {
    /// Device memory, per device.
    pub device: Vec<u64>,
    /// Pinned memory, per device.
    pub pinned: Vec<u64>,
    /// The budgeted heap.
    pub heap: u64,
    /// What the host has: the heap and pinned totals together are held
    /// against it by the final node.
    pub host_available: u64,
}

/// Whether a ledger's allocations are real.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// Buffers are allocated: device and pinned memory through wwr.
    Real,
    /// Nothing is allocated: counted and checked only, every buffer empty.
    /// A plan budgeted so cannot be executed.
    Dry,
}

#[derive(Clone, Copy, Debug)]
struct Row {
    step: &'static str,
    kind: Kind,
    bytes: u64,
}

/// The caps, the counters, the `over` flag and the per-step table: one per
/// run, held by the bus.
pub struct Ledger {
    caps: Caps,
    backing: Backing,
    device: Vec<AtomicU64>,
    pinned: Vec<AtomicU64>,
    heap: AtomicU64,
    workspace: Vec<AtomicU64>,
    over: AtomicBool,
    sealed: AtomicBool,
    allocations: AtomicU64,
    rows: Mutex<Vec<Row>>,
}

impl Ledger {
    /// An empty ledger over `caps`; the device count is `caps.device`'s
    /// length.
    pub fn new(caps: Caps, backing: Backing) -> Ledger {
        assert_eq!(
            caps.device.len(),
            caps.pinned.len(),
            "one pinned cap per device"
        );
        let zeros = |n: usize| (0..n).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();
        let n = caps.device.len();
        Ledger {
            caps,
            backing,
            device: zeros(n),
            pinned: zeros(n),
            heap: AtomicU64::new(0),
            workspace: zeros(n),
            over: AtomicBool::new(false),
            sealed: AtomicBool::new(false),
            allocations: AtomicU64::new(0),
            rows: Mutex::new(Vec::new()),
        }
    }

    /// Whether the allocations are real.
    pub fn backing(&self) -> Backing {
        self.backing
    }

    /// Whether some charge went over its cap.
    pub fn over(&self) -> bool {
        self.over.load(Ordering::Acquire)
    }

    /// Whether [`Ledger::finish`] has run: nothing is budgeted after it.
    pub fn sealed(&self) -> bool {
        self.sealed.load(Ordering::Acquire)
    }

    /// The buffers this ledger has allocated: the same before and after an
    /// execute, since `run` has no `Budget`.
    pub fn allocations(&self) -> u64 {
        self.allocations.load(Ordering::Acquire)
    }

    /// The total charged to `kind` so far; a workspace's is its max.
    pub fn total(&self, kind: Kind) -> u64 {
        match kind {
            Kind::Device(d) => self.device[d].load(Ordering::Acquire),
            Kind::Pinned(d) => self.pinned[d].load(Ordering::Acquire),
            Kind::Heap => self.heap.load(Ordering::Acquire),
            Kind::Workspace(d) => self.workspace[d].load(Ordering::Acquire),
        }
    }

    /// The handle `step`'s `mem_budget` allocates through.
    pub fn budget(&self, step: &'static str) -> Budget<'_> {
        Budget { ledger: self, step }
    }

    fn record(&self, step: &'static str, kind: Kind, bytes: u64) {
        let mut rows = self.rows.lock().unwrap_or_else(|e| e.into_inner());
        match rows.iter_mut().find(|r| r.step == step && r.kind == kind) {
            Some(row) if matches!(kind, Kind::Workspace(_)) => row.bytes = row.bytes.max(bytes),
            Some(row) => row.bytes += bytes,
            None => rows.push(Row { step, kind, bytes }),
        }
    }

    /// Count `bytes` of a summed kind against its cap. Whether to allocate.
    fn charge(&self, step: &'static str, kind: Kind, bytes: u64) -> bool {
        assert!(
            !self.sealed(),
            "step '{step}' budgets after the budget DAG's final node"
        );
        let (counter, cap) = match kind {
            Kind::Device(d) => (&self.device[d], self.caps.device[d]),
            Kind::Pinned(d) => (&self.pinned[d], self.caps.pinned[d]),
            Kind::Heap => (&self.heap, self.caps.heap),
            Kind::Workspace(_) => unreachable!("a workspace is a max: Budget::workspace"),
        };
        self.record(step, kind, bytes);
        if counter.fetch_add(bytes, Ordering::AcqRel) + bytes > cap {
            self.over.store(true, Ordering::Release);
        }
        !self.over() && self.backing == Backing::Real && bytes > 0
    }

    /// The budget DAG's final node. Adds each device's workspace max to its
    /// counter and checks the cap again, holds the heap and pinned totals
    /// against the host, and then either refuses the plan (the error carries
    /// the whole table) or calls `bind(device, bytes)` for each device with a
    /// workspace, which allocates it and hands it to the handles.
    ///
    /// # Errors
    /// The refusal, or `bind`'s error.
    pub fn finish(&self, mut bind: impl FnMut(usize, u64) -> Status) -> Status {
        for d in 0..self.device.len() {
            let workspace = self.total(Kind::Workspace(d));
            let total = self.device[d].fetch_add(workspace, Ordering::AcqRel) + workspace;
            if total > self.caps.device[d] {
                self.over.store(true, Ordering::Release);
            }
        }
        let pinned: u64 = (0..self.pinned.len())
            .map(|d| self.total(Kind::Pinned(d)))
            .sum();
        if self.total(Kind::Heap) + pinned > self.caps.host_available {
            self.over.store(true, Ordering::Release);
        }
        self.sealed.store(true, Ordering::Release);
        if self.over() {
            return Err(Error::allocation(format!(
                "the plan is over its memory budget; nothing was run\n{}",
                self.table()
            )));
        }
        if self.backing == Backing::Real {
            for d in 0..self.device.len() {
                let workspace = self.total(Kind::Workspace(d));
                if workspace > 0 {
                    bind(d, workspace)?;
                    self.allocations.fetch_add(1, Ordering::AcqRel);
                }
            }
        }
        log::info!("memory budget\n{}", self.table());
        Ok(())
    }

    /// The table: one row per step, one column per kind, the workspace its
    /// own row, then the totals against the caps. Rows in step-name order,
    /// so the text is the same every run.
    pub fn table(&self) -> String {
        let n = self.device.len();
        let mut columns: Vec<(String, Kind)> = Vec::new();
        for d in 0..n {
            columns.push((format!("device {d}"), Kind::Device(d)));
        }
        for d in 0..n {
            columns.push((format!("pinned {d}"), Kind::Pinned(d)));
        }
        columns.push(("heap".to_owned(), Kind::Heap));
        let rows = self.rows.lock().unwrap_or_else(|e| e.into_inner());
        let mut steps: Vec<&'static str> = rows.iter().map(|r| r.step).collect();
        steps.sort_unstable();
        steps.dedup();
        let mut out = String::new();
        let _ = write!(out, "{:<24}", "step");
        for (name, _) in &columns {
            let _ = write!(out, "{name:>14}");
        }
        out.push('\n');
        let cell = |step: &str, kind: Kind| {
            rows.iter()
                .find(|r| r.step == step && r.kind == kind)
                .map_or_else(|| "-".to_owned(), |r| bytes(r.bytes))
        };
        for step in &steps {
            if !columns.iter().any(|(_, k)| cell(step, *k) != "-") {
                continue;
            }
            let _ = write!(out, "{step:<24}");
            for (_, kind) in &columns {
                let _ = write!(out, "{:>14}", cell(step, *kind));
            }
            out.push('\n');
        }
        // The workspace: who asked for the max, per device.
        let _ = write!(out, "{:<24}", "workspace (max)");
        for (_, kind) in &columns {
            let text = match kind {
                Kind::Device(d) => bytes(self.total(Kind::Workspace(*d))),
                _ => "-".to_owned(),
            };
            let _ = write!(out, "{text:>14}");
        }
        out.push('\n');
        for step in &steps {
            for d in 0..n {
                if let Some(r) = rows
                    .iter()
                    .find(|r| r.step == *step && r.kind == Kind::Workspace(d))
                {
                    let _ = writeln!(
                        out,
                        "  {:<22}asks {} of device {d}'s workspace",
                        step,
                        bytes(r.bytes)
                    );
                }
            }
        }
        let sealed = self.sealed();
        let _ = write!(out, "{:<24}", "total");
        for (_, kind) in &columns {
            // Before the final node the device counter lacks the workspace.
            let total = match kind {
                Kind::Device(d) if !sealed => self.total(*kind) + self.total(Kind::Workspace(*d)),
                _ => self.total(*kind),
            };
            let _ = write!(out, "{:>14}", bytes(total));
        }
        out.push('\n');
        let _ = write!(out, "{:<24}", "cap");
        for (_, kind) in &columns {
            let cap = match kind {
                Kind::Device(d) => self.caps.device[*d],
                Kind::Pinned(d) => self.caps.pinned[*d],
                _ => self.caps.heap,
            };
            let _ = write!(out, "{:>14}", bytes(cap));
        }
        out.push('\n');
        let _ = write!(
            out,
            "host: heap + pinned against {} available",
            bytes(self.caps.host_available)
        );
        out
    }
}

/// `n` bytes in the largest binary unit that keeps three digits.
fn bytes(n: u64) -> String {
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

/// What a step's `mem_budget` allocates through: every call charges the
/// step's row and the kind's counter, then allocates unless the plan is
/// already over.
pub struct Budget<'a> {
    ledger: &'a Ledger,
    step: &'static str,
}

impl Budget<'_> {
    /// `len` elements of device memory on `device`, the step's own for the
    /// run. Uninitialized.
    pub fn device<T: Copy>(&self, device: usize, len: usize) -> DeviceBuf<T> {
        let size = len * size_of::<T>();
        if !self
            .ledger
            .charge(self.step, Kind::Device(device), size as u64)
        {
            return DeviceBuf::empty(len);
        }
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: writes one pointer through a valid local; the wrapper
        // made `device` current.
        check_or_abort(gpu_status(unsafe { wwr_sys::wwr_malloc(&mut p, size) }));
        self.ledger.allocations.fetch_add(1, Ordering::AcqRel);
        DeviceBuf { ptr: p.cast(), len }
    }

    /// `len` elements of page-locked host memory, staged for `device`.
    /// Zeroed, so every page is touched here.
    pub fn pinned<T: Copy>(&self, device: usize, len: usize) -> PinnedBuf<T> {
        let size = len * size_of::<T>();
        if !self
            .ledger
            .charge(self.step, Kind::Pinned(device), size as u64)
        {
            return PinnedBuf::empty(len);
        }
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: writes one pointer through a valid local.
        check_or_abort(gpu_status(unsafe {
            wwr_sys::wwr_host_alloc(&mut p, size, 0)
        }));
        // SAFETY: `p` is `size` fresh bytes.
        unsafe { std::ptr::write_bytes(p.cast::<u8>(), 0, size) };
        self.ledger.allocations.fetch_add(1, Ordering::AcqRel);
        PinnedBuf { ptr: p.cast(), len }
    }

    /// `len` elements of host heap that cannot grow, every page touched
    /// here: zeroed memory is handed out lazily, so an untouched slab would
    /// "succeed" without existing and fail at first touch mid-run.
    pub fn heap<T: Copy + Default>(&self, len: usize) -> HeapSlab<T> {
        let size = (len * size_of::<T>()) as u64;
        if !self.ledger.charge(self.step, Kind::Heap, size) {
            return HeapSlab(Box::default());
        }
        let mut slab = vec![T::default(); len].into_boxed_slice();
        let stride = (4096 / size_of::<T>().max(1)).max(1);
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

    /// Declare a library call's workspace on `device`: a max over every
    /// step's declarations. Nothing comes back; the final node allocates
    /// the max once and hands it to the handles.
    pub fn workspace(&self, device: usize, bytes: usize) {
        assert!(
            !self.ledger.sealed(),
            "step '{}' budgets after the budget DAG's final node",
            self.step
        );
        self.ledger
            .record(self.step, Kind::Workspace(device), bytes as u64);
        self.ledger.workspace[device].fetch_max(bytes as u64, Ordering::AcqRel);
    }
}

/// Device memory a step owns: fixed size, freed with the step's `Output`.
/// Empty (a null pointer of the asked length) under a dry or refused budget.
pub struct DeviceBuf<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: a device allocation is not tied to a host thread; the task graph,
// not the type, orders its users.
unsafe impl<T: Send> Send for DeviceBuf<T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for DeviceBuf<T> {}

impl<T> DeviceBuf<T> {
    fn empty(len: usize) -> Self {
        DeviceBuf {
            ptr: std::ptr::null_mut(),
            len,
        }
    }

    /// Whether the buffer was really allocated.
    pub fn is_allocated(&self) -> bool {
        !self.ptr.is_null()
    }

    /// The workspace the final node allocates: outside any step's row, since
    /// the ledger has already counted it as the max.
    pub(crate) fn workspace(bytes: usize) -> DeviceBuf<T> {
        let len = bytes.div_ceil(size_of::<T>());
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: writes one pointer through a valid local.
        check_or_abort(gpu_status(unsafe {
            wwr_sys::wwr_malloc(&mut p, len * size_of::<T>())
        }));
        DeviceBuf { ptr: p.cast(), len }
    }
}

impl<T> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated by wwr_malloc, freed once.
            unsafe { wwr_sys::wwr_free(self.ptr.cast()) };
        }
    }
}

// SAFETY: `len` elements of device memory, live while `self` is.
unsafe impl<T> DeviceSlice<T> for DeviceBuf<T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above, and `&mut self` is the one user.
unsafe impl<T> DeviceSliceMut<T> for DeviceBuf<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// Page-locked host memory a step owns: fixed size, zeroed at allocation.
pub struct PinnedBuf<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: plain host memory.
unsafe impl<T: Send> Send for PinnedBuf<T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for PinnedBuf<T> {}

impl<T> PinnedBuf<T> {
    fn empty(len: usize) -> Self {
        PinnedBuf {
            ptr: std::ptr::null_mut(),
            len,
        }
    }

    /// The element count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The host pointer, for a copy to enqueue.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// The host pointer, for a copy to enqueue into.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    fn allocated(&self) {
        assert!(
            !self.ptr.is_null() || self.len == 0,
            "a pinned buffer of a dry or refused budget was touched"
        );
    }
}

impl<T> Deref for PinnedBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.allocated();
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `len` initialized elements, live while `self` is. A copy
        // enqueued into it has finished before a reader runs: the step that
        // enqueued it synchronized first.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T> DerefMut for PinnedBuf<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.allocated();
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: as above, and `&mut self` is the one user.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: allocated by wwr_host_alloc, freed once.
            unsafe { wwr_sys::wwr_free_host(self.ptr.cast()) };
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_are_printed_in_binary_units() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1536), "1.50 KiB");
        assert_eq!(bytes(3 << 30), "3.00 GiB");
    }
}
