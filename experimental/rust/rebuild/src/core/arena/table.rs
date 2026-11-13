//! The table: a plan's rows for one worker and one memory kind, who uses
//! each, and the offsets [`assign`] gave them.

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

use super::DEVICE_ALIGNMENT;
use super::assign::{Life, Row, assign};
use super::kind::{Element, Kind};

/// How a node uses a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Access {
    /// It only reads the row.
    Read,
    /// It writes the row, and may read it.
    Write,
}

/// Who a row belongs to, which is what fixes its lifetime
/// (`docs/memory.md`, "The table").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Owner {
    /// A node's scratch: its one user is the node.
    Task,
    /// A node's output: written by its producer, read by later nodes.
    Edge,
    /// The stream worker's: live for the whole plan.
    Worker,
}

pub(super) struct RowInfo {
    pub(super) name: String,
    pub(super) owner: Owner,
    /// In elements.
    pub(super) len: usize,
    pub(super) bytes: usize,
    /// `(position, access)` in the order declared. The producer's is first,
    /// and the positions after it are larger. Empty for a worker row.
    pub(super) users: Vec<(usize, Access)>,
}

impl RowInfo {
    pub(super) fn life(&self) -> Life {
        match self.owner {
            Owner::Worker => Life::Plan,
            Owner::Task | Owner::Edge => Life::Span {
                first: self.users[0].0,
                last: self.users.iter().map(|&(p, _)| p).max().unwrap_or(0),
            },
        }
    }

    /// How node `pos` uses this row: `None` if it is not a user. A worker
    /// row is written by any node.
    pub(super) fn access(&self, pos: usize) -> Option<Access> {
        match self.owner {
            Owner::Worker => Some(Access::Write),
            Owner::Task | Owner::Edge => {
                self.users.iter().find(|&&(p, _)| p == pos).map(|&(_, a)| a)
            }
        }
    }
}

/// A row of a [`Table`] of kind `K` holding elements of type `T`.
///
/// It names the table it came from, so handing it to another table panics
/// rather than reading some other row's bytes as `T`.
pub struct RowId<K, T> {
    pub(super) table: u64,
    pub(super) index: usize,
    marker: PhantomData<fn() -> (K, T)>,
}

impl<K, T> Clone for RowId<K, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, T> Copy for RowId<K, T> {}

impl<K, T> fmt::Debug for RowId<K, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RowId({}:{})", self.table, self.index)
    }
}

static NEXT_TABLE: AtomicU64 = AtomicU64::new(0);

/// Declares a table's rows, node by node, and then lays them out.
///
/// A node is named by its position, its rank in the plan the table serves:
/// the order the plan's nodes are added in, which is a topological order.
///
/// ```
/// use fb_rebuild::core::arena::{Device, TableBuilder};
///
/// let n = 24 * 24;
/// let mut b = TableBuilder::<Device>::new();
/// let s = b.output::<f64>(0, "S", n);
/// let j = b.scratch::<f64>(1, "J", n);
/// b.read(1, s);
/// let f = b.output::<f64>(2, "F", n);
/// let table = b.build();
/// // S and J are both live at node 1, so they cannot share bytes. At node
/// // 2 both are dead and F takes S's, so both of S's users precede it.
/// assert_ne!(table.offset(s), table.offset(j));
/// assert_eq!(table.offset(f), table.offset(s));
/// assert_eq!(table.handoffs(), vec![(0, 2), (1, 2)]);
/// ```
pub struct TableBuilder<K> {
    id: u64,
    rows: Vec<RowInfo>,
    marker: PhantomData<fn() -> K>,
}

impl<K: Kind> Default for TableBuilder<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Kind> TableBuilder<K> {
    /// An empty table.
    pub fn new() -> Self {
        TableBuilder {
            id: NEXT_TABLE.fetch_add(1, Ordering::Relaxed),
            rows: Vec::new(),
            marker: PhantomData,
        }
    }

    /// Scratch of node `pos`: `len` elements, written by it and used by no
    /// other node.
    #[must_use]
    pub fn scratch<T: Element>(&mut self, pos: usize, name: &str, len: usize) -> RowId<K, T> {
        self.push::<T>(name, Owner::Task, len, vec![(pos, Access::Write)])
    }

    /// An output of node `pos`: `len` elements, written by it. Its readers
    /// are added with [`TableBuilder::read`].
    #[must_use]
    pub fn output<T: Element>(&mut self, pos: usize, name: &str, len: usize) -> RowId<K, T> {
        self.push::<T>(name, Owner::Edge, len, vec![(pos, Access::Write)])
    }

    /// A row the stream worker holds for the whole plan, such as a library
    /// handle's workspace or a solver's `info` flag. Any node may write it.
    #[must_use]
    pub fn worker<T: Element>(&mut self, name: &str, len: usize) -> RowId<K, T> {
        self.push::<T>(name, Owner::Worker, len, Vec::new())
    }

    /// Node `pos` reads the output `row`.
    ///
    /// The plan must have an edge from the row's producer to `pos`: the
    /// table records the use, the plan orders it.
    ///
    /// # Panics
    /// If `row` is not an output of this table, or `pos` is not past its
    /// producer, or `pos` already uses it.
    pub fn read<T: Element>(&mut self, pos: usize, row: RowId<K, T>) {
        assert_eq!(row.table, self.id, "a row of another table");
        let info = &mut self.rows[row.index];
        assert!(
            info.owner == Owner::Edge,
            "{}: only an output is read by another node",
            info.name
        );
        let producer = info.users[0].0;
        assert!(
            pos > producer,
            "{}: node {pos} reads it, but node {producer} produces it",
            info.name
        );
        assert!(
            info.users.iter().all(|&(p, _)| p != pos),
            "{}: node {pos} already uses it",
            info.name
        );
        info.users.push((pos, Access::Read));
    }

    /// Lay the rows out.
    pub fn build(self) -> Table<K> {
        let rows: Vec<Row> = self
            .rows
            .iter()
            .map(|r| Row {
                bytes: r.bytes,
                align: DEVICE_ALIGNMENT,
                life: r.life(),
            })
            .collect();
        let assignment = assign(&rows);
        Table {
            id: self.id,
            rows: self.rows,
            offsets: assignment.offsets,
            total: assignment.total,
            marker: PhantomData,
        }
    }

    fn push<T: Element>(
        &mut self,
        name: &str,
        owner: Owner,
        len: usize,
        users: Vec<(usize, Access)>,
    ) -> RowId<K, T> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .unwrap_or_else(|| panic!("{name}: {len} elements overflow usize bytes"));
        let index = self.rows.len();
        self.rows.push(RowInfo {
            name: name.to_owned(),
            owner,
            len,
            bytes,
            users,
        });
        RowId {
            table: self.id,
            index,
            marker: PhantomData,
        }
    }
}

/// A laid-out table: every row's offset, and the block's total.
///
/// Its text ([`fmt::Display`]) is the committed memory golden: the rows,
/// their offsets, the peak and the handoffs.
pub struct Table<K> {
    id: u64,
    pub(super) rows: Vec<RowInfo>,
    pub(super) offsets: Vec<usize>,
    total: usize,
    marker: PhantomData<fn() -> K>,
}

/// The instant a table's live rows are largest: [`Table::peak`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peak<'t> {
    /// The position, or `None` when only worker rows are declared.
    pub position: Option<usize>,
    /// The bytes live there.
    pub bytes: usize,
    /// The rows live there, by name and bytes, in declaration order.
    pub rows: Vec<(&'t str, usize)>,
}

impl<K: Kind> Table<K> {
    /// The block's size in bytes.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The number of rows.
    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// `row`'s offset in the block, in bytes.
    ///
    /// # Panics
    /// If `row` is not of this table.
    pub fn offset<T: Element>(&self, row: RowId<K, T>) -> usize {
        self.offsets[self.index(row)]
    }

    /// `row`'s length, in elements.
    ///
    /// # Panics
    /// If `row` is not of this table.
    pub fn len<T: Element>(&self, row: RowId<K, T>) -> usize {
        self.rows[self.index(row)].len
    }

    pub(super) fn index<T: Element>(&self, row: RowId<K, T>) -> usize {
        assert_eq!(row.table, self.id, "a row of another table");
        row.index
    }

    /// The orderings that reusing bytes needs: `(u, v)` for every user `u`
    /// of a row whose bytes a later row takes, `v` being that later row's
    /// producer. Sorted, without repeats.
    ///
    /// The layout packs rows against the positions, but a plan runs a
    /// ready node whatever its rank, so the positions alone do not order
    /// two rows that share bytes. Every pair returned must therefore be an
    /// edge of the plan, or implied by its edges: the previous owner of an
    /// offset is a predecessor of the next (`docs/memory.md`,
    /// "Lifetimes").
    pub fn handoffs(&self) -> Vec<(usize, usize)> {
        let mut pairs = BTreeSet::new();
        for (i, a) in self.rows.iter().enumerate() {
            for (j, b) in self.rows.iter().enumerate().skip(i + 1) {
                if a.bytes == 0 || b.bytes == 0 {
                    continue;
                }
                let (oa, ob) = (self.offsets[i], self.offsets[j]);
                if oa >= ob + b.bytes || ob >= oa + a.bytes {
                    continue;
                }
                let (Life::Span { last: la, .. }, Life::Span { first: fb, .. }) =
                    (a.life(), b.life())
                else {
                    unreachable!("a plan-long row shares bytes with no row")
                };
                let (earlier, later) = if la < fb { (a, b) } else { (b, a) };
                let next = later.users[0].0;
                pairs.extend(earlier.users.iter().map(|&(u, _)| (u, next)));
            }
        }
        pairs.into_iter().collect()
    }

    /// The position at which the live rows' bytes are largest, the lowest
    /// on a tie. The total is at least this; alignment and the greedy
    /// layout's gaps are the difference.
    pub fn peak(&self) -> Peak<'_> {
        let positions: BTreeSet<usize> = self
            .rows
            .iter()
            .flat_map(|r| r.users.iter().map(|&(p, _)| p))
            .collect();
        let live_at = |pos: Option<usize>| {
            self.rows.iter().filter(move |r| match (r.life(), pos) {
                (Life::Plan, _) => true,
                (Life::Span { first, last }, Some(p)) => first <= p && p <= last,
                (Life::Span { .. }, None) => false,
            })
        };
        let bytes_at = |pos| live_at(pos).map(|r| r.bytes).sum::<usize>();
        // The largest bytes, and on a tie the lowest position.
        let position = positions
            .into_iter()
            .max_by_key(|&p| (bytes_at(Some(p)), Reverse(p)));
        Peak {
            position,
            bytes: bytes_at(position),
            rows: live_at(position)
                .map(|r| (r.name.as_str(), r.bytes))
                .collect(),
        }
    }
}

impl<K: Kind> fmt::Display for Table<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} table: {} rows, {} bytes",
            K::NAME,
            self.rows.len(),
            self.total
        )?;
        let peak = self.peak();
        match peak.position {
            Some(p) => writeln!(f, "peak: {} bytes at node {p}", peak.bytes)?,
            None => writeln!(f, "peak: {} bytes", peak.bytes)?,
        }
        writeln!(
            f,
            "{:>10} {:>10}  {:<6}  {:<7}  name: users (w writes, r reads)",
            "offset", "bytes", "owner", "life"
        )?;
        for (row, offset) in self.rows.iter().zip(&self.offsets) {
            let owner = match row.owner {
                Owner::Task => "task",
                Owner::Edge => "edge",
                Owner::Worker => "worker",
            };
            let life = match row.life() {
                Life::Plan => "plan".to_owned(),
                Life::Span { first, last } => format!("{first}-{last}"),
            };
            let users: Vec<String> = row
                .users
                .iter()
                .map(|&(p, a)| match a {
                    Access::Read => format!("{p}r"),
                    Access::Write => format!("{p}w"),
                })
                .collect();
            write!(
                f,
                "{offset:>10} {:>10}  {owner:<6}  {life:<7}  {}",
                row.bytes, row.name
            )?;
            if users.is_empty() {
                writeln!(f)?;
            } else {
                writeln!(f, ": {}", users.join(" "))?;
            }
        }
        let handoffs: Vec<String> = self
            .handoffs()
            .iter()
            .map(|(u, v)| format!("{u}->{v}"))
            .collect();
        writeln!(f, "handoffs: {}", handoffs.join(" "))
    }
}
