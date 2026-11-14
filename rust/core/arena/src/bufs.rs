//! What a node is handed: its rows of one block, as slices on the host and
//! as pointers on the device.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::atomic::Ordering;

use super::block::Block;
use super::kind::{Device, Element, HostKind, Kind};
#[cfg(feature = "device")]
use super::table::Owner;
use super::table::{Access, RowId};

/// Node `pos`'s view of a [`Block`]: [`Block::partition`].
///
/// Asking for a row the node's layout did not name panics, as does asking
/// for one another live view holds in a conflicting mode: a writer is
/// alone, readers share. These checks are always on, because they are what
/// makes handing out `&mut [T]` from a shared block sound. Every borrow is
/// returned when the view drops.
///
/// **A stream node does not touch a pinned row from the host.** It hands
/// the row's pointer to a copy it enqueues, and that is all: the handoff
/// from a stream node to the next one on the same stream adds no wait,
/// because the stream orders the two nodes' device work, not a host write
/// against an earlier copy still in flight. Reading or writing pinned bytes
/// on the host is a CPU node's, which the executor starts only after the
/// events of the stream nodes before it.
pub struct Bufs<'b, K: Kind> {
    pub(super) block: &'b Block<K>,
    pub(super) pos: usize,
    /// Each row this view has borrowed, and whether as its writer.
    pub(super) held: RefCell<Vec<(usize, bool)>>,
}

impl<K: Kind> Bufs<'_, K> {
    /// The node this view is for.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Borrow `row` for this view, as its writer when `write`, and return
    /// its index and length.
    fn acquire<T: Element>(&self, row: RowId<K, T>, write: bool) -> (usize, usize) {
        let table = &self.block.table;
        let index = table.index(row);
        let info = &table.rows[index];
        let Some(access) = info.access(self.pos) else {
            panic!("node {}: {} is not in its layout", self.pos, info.name)
        };
        assert!(
            !write || access == Access::Write,
            "node {}: {} is read-only to it",
            self.pos,
            info.name
        );
        // A view may read a row as often as it likes, as `&` borrows may
        // repeat; a repeat that involves a write would alias a `&mut`.
        if let Some(&(_, held_write)) = self.held.borrow().iter().find(|&&(i, _)| i == index) {
            assert!(
                !write && !held_write,
                "node {}: {} is already borrowed by this view",
                self.pos,
                info.name
            );
            return (index, info.len);
        }
        let flag = &self.block.borrows[index];
        let taken = if write {
            flag.compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
                .is_ok()
        } else {
            flag.try_update(Ordering::Acquire, Ordering::Relaxed, |n| {
                (n >= 0).then_some(n + 1)
            })
            .is_ok()
        };
        assert!(
            taken,
            "node {}: {} is held by another node's view",
            self.pos, info.name
        );
        self.held.borrow_mut().push((index, write));
        (index, info.len)
    }
}

impl<K: Kind> Bufs<'_, K> {
    /// `row` borrowed for this view as [`Bufs::acquire`] borrows it, as its
    /// first element and length: what a copy enqueued on a stream reads or
    /// writes ([`super::memcpy`]). The borrow is held until the view drops.
    #[cfg(feature = "device")]
    pub(super) fn acquire_ptr<T: Element>(&self, row: RowId<K, T>, write: bool) -> (*mut T, usize) {
        let (index, len) = self.acquire(row, write);
        (self.block.row_ptr(index).cast::<T>(), len)
    }
}

impl<K: Kind> Drop for Bufs<'_, K> {
    fn drop(&mut self) {
        for &(index, write) in self.held.get_mut().iter() {
            let flag = &self.block.borrows[index];
            if write {
                flag.store(0, Ordering::Release);
            } else {
                flag.fetch_sub(1, Ordering::Release);
            }
        }
    }
}

impl<K: HostKind> Bufs<'_, K> {
    /// A row the node reads or writes, to read. As often as it likes.
    ///
    /// # Panics
    /// If `row` is not in the node's layout, or is held by a writer, this
    /// view included.
    pub fn get<T: Element>(&self, row: RowId<K, T>) -> &[T] {
        let (index, len) = self.acquire(row, false);
        if len == 0 {
            return &[];
        }
        let data = self.block.row_ptr(index).cast::<T>();
        // SAFETY: the row is `len` elements of `T` inside the live block,
        // aligned to DEVICE_ALIGNMENT, which `T`'s alignment divides
        // (`Element`); every byte was filled at allocation, and every bit
        // pattern is a `T`. The shared borrow is held until this view drops,
        // which the returned lifetime cannot outlive, so no writer exists.
        unsafe { std::slice::from_raw_parts(data, len) }
    }

    /// A row the node writes, to write. Once per row per view.
    ///
    /// # Panics
    /// If `row` is not written by the node, or is held by any other view,
    /// or this view already holds it.
    // `&self` hands out `&mut`: the row's borrow flag, taken exclusively,
    // is what makes it unique, as a RefCell's is.
    #[allow(clippy::mut_from_ref)]
    pub fn take_mut<T: Element>(&self, row: RowId<K, T>) -> &mut [T] {
        let (index, len) = self.acquire(row, true);
        if len == 0 {
            return &mut [];
        }
        let data = self.block.row_ptr(index).cast::<T>();
        // SAFETY: as in `get`, and the borrow is exclusive: no other view
        // holds this row, and this one cannot ask for it again.
        unsafe { std::slice::from_raw_parts_mut(data, len) }
    }
}

impl Bufs<'_, Device> {
    /// A row the node reads or writes, as a device pointer to read through.
    ///
    /// # Panics
    /// If `row` is not in the node's layout, or is held by a writer.
    pub fn get<T: Element>(&self, row: RowId<Device, T>) -> DeviceRef<'_, T> {
        let (index, len) = self.acquire(row, false);
        DeviceRef {
            ptr: self.block.row_ptr(index).cast::<T>().cast_const(),
            len,
            marker: PhantomData,
        }
    }

    /// A row the node writes, as a device pointer to write through. Once
    /// per row per view.
    ///
    /// # Panics
    /// If `row` is not written by the node, or is held by any other view,
    /// or this view already holds it.
    pub fn take_mut<T: Element>(&self, row: RowId<Device, T>) -> DeviceMut<'_, T> {
        let (index, len) = self.acquire(row, true);
        DeviceMut {
            ptr: self.block.row_ptr(index).cast::<T>(),
            len,
            marker: PhantomData,
        }
    }
}

/// A device row to read: a pointer and a length, with no `Deref`, since the
/// host cannot read it. Borrows the view it came from.
pub struct DeviceRef<'v, T> {
    ptr: *const T,
    len: usize,
    marker: PhantomData<&'v [T]>,
}

impl<T> Clone for DeviceRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for DeviceRef<'_, T> {}

impl<T> DeviceRef<'_, T> {
    /// The row's first element, on the device.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Its length, in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T> From<&'a DeviceMut<'_, T>> for DeviceRef<'a, T> {
    /// The row a writer holds, to read: as a `&mut` reborrows as `&`.
    fn from(row: &'a DeviceMut<'_, T>) -> Self {
        DeviceRef {
            ptr: row.as_ptr(),
            len: row.len(),
            marker: PhantomData,
        }
    }
}

/// A device row to write: a pointer and a length, with no `Deref`. Borrows
/// the view it came from, and is the row's only writer while it lives.
pub struct DeviceMut<'v, T> {
    ptr: *mut T,
    len: usize,
    marker: PhantomData<&'v mut [T]>,
}

impl<T> DeviceMut<'_, T> {
    /// The row's first element, on the device, to write through.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// The same, to read through.
    pub fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }

    /// Its length, in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it has no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// SAFETY: the row is `len` elements of `T` inside a live device block on
// its worker's device, aligned for `T` (`Element`), and the block outlives
// every task's device work. The view's shared borrow excludes every writer
// while it lives, and a later writer's work is ordered after this task's by
// the executor: the same stream, or an event on the task's stream.
#[cfg(feature = "device")]
unsafe impl<T> fb_core_wwr::device_slice::DeviceSlice<T> for DeviceRef<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as for `DeviceRef`, with the view's borrow exclusive: no other
// view reads or writes the row while it lives, and later work on it is
// ordered after this task's.
#[cfg(feature = "device")]
unsafe impl<T> fb_core_wwr::device_slice::DeviceSlice<T> for DeviceMut<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr.cast_const()
    }

    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above.
#[cfg(feature = "device")]
unsafe impl<T> fb_core_wwr::device_slice::DeviceSliceMut<T> for DeviceMut<'_, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

/// A device worker row held outside any node's view, as its writer, for as
/// long as the value lives, with the block it is in kept alive by it:
/// [`Block::hold`]. For memory a library keeps beyond any node, such as a
/// BLAS handle's workspace, which the handle owns by owning this.
///
/// It owns its memory rather than borrowing a view, so it is `'static`, and
/// it is the row's only user while it lives: a node's view asking for the
/// row panics, as for any row another view holds.
#[cfg(feature = "device")]
pub struct HeldRow<T> {
    block: std::sync::Arc<Block<Device>>,
    index: usize,
    len: usize,
    marker: PhantomData<fn() -> T>,
}

#[cfg(feature = "device")]
impl Block<Device> {
    /// Worker row `row`, held as its writer until the returned
    /// [`HeldRow`] drops, and the block kept alive until then.
    ///
    /// Only a worker row ([`TableBuilder::worker`]) can be held: its bytes
    /// are its own for the whole plan, where a node's row shares them with
    /// the rows before and after it.
    ///
    /// [`TableBuilder::worker`]: super::TableBuilder::worker
    ///
    /// # Panics
    /// If `row` is not of this block's table or not a worker row, or a
    /// view or another [`HeldRow`] holds it.
    pub fn hold<T: Element>(self: &std::sync::Arc<Self>, row: RowId<Device, T>) -> HeldRow<T> {
        let index = self.table.index(row);
        let info = &self.table.rows[index];
        assert!(
            info.owner == Owner::Worker,
            "{} is not a worker row, so it cannot be held",
            info.name
        );
        let taken = self.borrows[index]
            .compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok();
        assert!(taken, "{} is already held", info.name);
        HeldRow {
            block: std::sync::Arc::clone(self),
            index,
            len: info.len,
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "device")]
impl<T> HeldRow<T> {
    /// The device its block is on.
    pub fn dev_idx(&self) -> i32 {
        self.block.dev_idx()
    }
}

#[cfg(feature = "device")]
impl<T> Drop for HeldRow<T> {
    fn drop(&mut self) {
        self.block.borrows[self.index].store(0, Ordering::Release);
    }
}

// SAFETY: the row is `len` elements of `T` inside a device block on its
// worker's device (`dev_idx`), aligned for `T` (`Element`), which the `Arc`
// keeps allocated for as long as the value lives. A worker row's bytes
// overlap no other row's for the plan's whole life, and its writer flag,
// taken in `hold` and returned only in `drop`, keeps every view and every
// other `HeldRow` off it meanwhile. What the holder enqueued through it
// must have finished before it drops, and the holder that takes it by
// value is what promises so: a BLAS handle drops it only once its stream
// has drained (`fb_core_wwr::blas::BlasHandle::bind_workspace`).
#[cfg(feature = "device")]
unsafe impl<T> fb_core_wwr::device_slice::DeviceSlice<T> for HeldRow<T> {
    fn as_ptr(&self) -> *const T {
        self.block.row_ptr(self.index).cast::<T>().cast_const()
    }

    fn len(&self) -> usize {
        self.len
    }
}

// SAFETY: as above: the flag makes the value the row's only user.
#[cfg(feature = "device")]
unsafe impl<T> fb_core_wwr::device_slice::DeviceSliceMut<T> for HeldRow<T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.block.row_ptr(self.index).cast::<T>()
    }
}
