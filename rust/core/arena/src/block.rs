//! A block: one allocation of one kind, laid out by its table.

use std::cell::RefCell;
use std::panic::Location;
use std::ptr;
use std::sync::atomic::{AtomicIsize, AtomicU64, Ordering};

use super::DEVICE_ALIGNMENT;
use super::bufs::Bufs;
use super::kind::Kind;
use super::table::Table;
use fb_core_error_handling::{Error, ErrorKind, check_or_abort};
use log::error;

/// Every block [`Block::new`] has allocated in this process, of any kind.
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);

/// How many blocks this process has allocated, of every kind, since it
/// started: one per [`Block::new`] of a table with a non-zero total (a table
/// of zero bytes allocates nothing and is not counted), never decreased by a
/// drop. The arena is the only place the workspace allocates, so the count
/// is every allocation of ours, and only ours: unlike the device's free
/// memory, another tenant of the card cannot move it, and neither can a
/// library allocating behind the arena, which it does not see
/// (`docs/memory.md`, "What is checked"). A test reads it twice around what
/// must allocate nothing:
///
/// ```
/// let before = fb_core_arena::allocations();
/// // ... a loop's replay ...
/// assert_eq!(fb_core_arena::allocations(), before);
/// ```
#[must_use]
pub fn allocations() -> u64 {
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// One contiguous allocation of kind `K` on one GPU worker, sized to its
/// table's total, made in [`Block::new`] and freed when it drops
/// (`docs/memory.md`, "The two kinds").
///
/// Not an allocator: a node reaches its rows through
/// [`Block::partition`], and nothing allocates or frees between the two
/// ends. Each row carries a borrow flag, so the views two nodes hold at
/// once can never alias; a schedule that honours the table's
/// [`handoffs`](Table::handoffs) never trips one.
///
/// ```no_run
/// use fb_core_arena::{Block, Device, TableBuilder};
///
/// let mut b = TableBuilder::<Device>::new();
/// let s = b.output::<f64>(0, "S", 24 * 24);
/// let block = Block::new(0, b.build());
/// let bufs = block.partition(0, 0);
/// let s = bufs.take_mut(s);
/// assert_eq!(s.len(), 24 * 24);
/// // s.as_mut_ptr() goes to a launch wrapper.
/// ```
pub struct Block<K: Kind> {
    /// Null when the table's total is zero.
    pub(super) base: *mut u8,
    dev_idx: i32,
    pub(super) table: Table<K>,
    /// Per row and copy of it, row `i`'s copy `c` at `flags[i] + c`: 0
    /// free, `n > 0` shared by `n` views, -1 held by one writer.
    pub(super) borrows: Box<[AtomicIsize]>,
    /// Per row, the index of its first copy's flag in `borrows`.
    pub(super) flags: Box<[usize]>,
}

// SAFETY: the base pointer is what makes `Block` neither `Send` nor `Sync`.
// The allocation is not tied to a thread: the device ones are freed with
// the block's device made current again, and a pinned or heap one from any
// thread. Every access to its bytes goes through a `Bufs`, which takes the
// row's atomic borrow flag first, so views on different threads never
// alias mutably; the pointer itself is only read after `new`.
unsafe impl<K: Kind> Send for Block<K> {}
// SAFETY: as above.
unsafe impl<K: Kind> Sync for Block<K> {}

impl<K: Kind> Block<K> {
    /// Allocate `table`'s block on device `dev_idx` (made current for the
    /// call), and fill every byte with `0xFF`: a NaN in both float widths
    /// and -1 in every integer width, so a row read before it is written
    /// fails the first comparison. A pinned block is filled in every
    /// profile, a device block in debug builds only. A table of zero bytes
    /// allocates nothing.
    ///
    /// Aborts, as every create does, if the allocation or the fill fails,
    /// naming the bytes asked and the memory free, or if the base is not
    /// [`DEVICE_ALIGNMENT`]-aligned.
    #[track_caller]
    pub fn new(dev_idx: i32, table: Table<K>) -> Self {
        let location = Location::caller();
        let total = table.total();
        let mut base = ptr::null_mut();
        if total > 0 {
            let _device = K::current(dev_idx);
            // SAFETY: one call writing one pointer through a valid local.
            if let Err(failure) = unsafe { K::alloc(&mut base, total) } {
                check_or_abort::<()>(Err(Error {
                    kind: ErrorKind::Allocation,
                    message: format!(
                        "the {} block of {total} bytes on device {dev_idx}: {failure}; {}",
                        K::NAME,
                        K::free_memory()
                    ),
                    origin: location,
                    context: Vec::new(),
                }));
            }
            if base.addr() % DEVICE_ALIGNMENT != 0 {
                check_or_abort::<()>(Err(Error {
                    kind: ErrorKind::Allocation,
                    message: format!(
                        "the {} block on device {dev_idx} is at {base:p}, not \
                         {DEVICE_ALIGNMENT}-aligned",
                        K::NAME
                    ),
                    origin: location,
                    context: Vec::new(),
                }));
            }
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            if K::ALWAYS_FILL || cfg!(debug_assertions) {
                // SAFETY: the allocation just made, of `total` bytes.
                check_or_abort(unsafe { K::fill(base, total, location) });
            }
        }
        let flags: Box<[usize]> = (0..table.n_rows())
            .scan(0, |next, i| {
                let first = *next;
                *next += table.copies(i).0;
                Some(first)
            })
            .collect();
        let n_flags = (0..table.n_rows()).map(|i| table.copies(i).0).sum();
        let borrows = (0..n_flags).map(|_| AtomicIsize::new(0)).collect();
        Block {
            base,
            dev_idx,
            table,
            borrows,
            flags,
        }
    }

    /// The table the block was laid out by.
    pub fn table(&self) -> &Table<K> {
        &self.table
    }

    /// The device the block belongs to.
    pub fn dev_idx(&self) -> i32 {
        self.dev_idx
    }

    /// The rows of node `pos` in iteration `iteration` of the loop it is
    /// in, 0 outside every loop: what the orchestrator hands the node when
    /// it runs. A row is borrowed when the node first asks for it and
    /// returned when the [`Bufs`] drops.
    ///
    /// A counted loop's body row is reached in copy `iteration mod depth`
    /// of its region ([`TableBuilder::add_counted`]), each copy with a
    /// borrow flag of its own, so two iterations in flight hold their own
    /// bytes; every other row has one copy, whatever the iteration.
    ///
    /// [`TableBuilder::add_counted`]: super::TableBuilder::add_counted
    pub fn partition(&self, pos: usize, iteration: usize) -> Bufs<'_, K> {
        Bufs {
            block: self,
            pos,
            iteration,
            held: RefCell::new(Vec::new()),
        }
    }

    /// Row `index`'s copy `copy`, its first byte. Never dereferenced here.
    pub(super) fn row_ptr(&self, index: usize, copy: usize) -> *mut u8 {
        let (_, stride) = self.table.copies(index);
        self.base
            .wrapping_add(self.table.offsets[index] + copy * stride)
    }
}

impl<K: Kind> Drop for Block<K> {
    fn drop(&mut self) {
        if self.base.is_null() {
            return;
        }
        let location = Location::caller();
        let _device = K::try_current(self.dev_idx)
            .inspect_err(|e| error!("freeing the {} block: {e}", K::NAME))
            .ok();
        // SAFETY: `new`'s allocation of the table's total, freed once.
        unsafe { K::free(self.base, self.table.total(), location) };
    }
}
