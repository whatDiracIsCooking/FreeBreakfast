//! The table: a plan's rows for one worker and one memory kind, who uses
//! each, and the offsets [`assign`] gave them.

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fmt;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU64, Ordering};

use contracts::debug_requires;

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
    /// and the positions after it are larger. A later `Write` is a loop's
    /// carrying writer ([`TableBuilder::write`]). Empty for a worker row.
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
/// use fb_core_arena::{Device, TableBuilder};
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
    /// Each loop's positions, in plan order ([`TableBuilder::add_loop`]).
    loops: Vec<RangeInclusive<usize>>,
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
            loops: Vec::new(),
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
    #[debug_requires(row.table == self.id, "a row of this table")]
    #[debug_requires(
        self.rows.get(row.index).is_some_and(|r| r.owner == Owner::Edge),
        "only an output is read by another node"
    )]
    #[debug_requires(
        self.rows.get(row.index).is_some_and(|r| r.users[0].0 < pos),
        "a row is read after its producer"
    )]
    #[debug_requires(
        self.rows
            .get(row.index)
            .is_some_and(|r| r.users.iter().all(|&(p, _)| p != pos)),
        "a node uses a row once"
    )]
    pub fn read<T: Element>(&mut self, pos: usize, row: RowId<K, T>) {
        self.rows[row.index].users.push((pos, Access::Read));
    }

    /// Node `pos` writes the output `row`, which a node before its loop
    /// produced: the row is carried from one iteration of the loop `pos` is
    /// in to the next (`docs/memory.md`, "Nested plans"). The body reads
    /// it before this write, and this node is its last user in the loop,
    /// so a body node reads, in each iteration, the value the iteration
    /// before wrote, or the producer's in the first. The row is live
    /// across the whole loop: no handoff crosses the iteration edge, and
    /// no body row takes its bytes.
    ///
    /// The plan must order every other user of the row against the write:
    /// [`Table::reads`] returns those pairs.
    #[debug_requires(row.table == self.id, "a row of this table")]
    #[debug_requires(
        self.rows.get(row.index).is_some_and(|r| r.owner == Owner::Edge),
        "only an output is carried"
    )]
    #[debug_requires(
        self.rows.get(row.index).is_some_and(|r| r.users[0].0 < pos),
        "a carried row is written after its producer"
    )]
    #[debug_requires(
        self.rows
            .get(row.index)
            .is_some_and(|r| r.users.iter().all(|&(p, _)| p != pos)),
        "a node uses a row once"
    )]
    pub fn write<T: Element>(&mut self, pos: usize, row: RowId<K, T>) {
        self.rows[row.index].users.push((pos, Access::Write));
    }

    /// A loop of the plan over `positions`: its body's nodes and, last, its
    /// decision, the whole replayed once per iteration
    /// (`docs/memory.md`, "Nested plans"). A row all of whose users lie in
    /// it is the body's: the body's rows are laid out as a table of their
    /// own, in a region the rest of the table holds for the loop's whole
    /// span, sized to the body's peak once. Every other row used in the
    /// loop, or live across it, is live for the whole loop, so no body row
    /// takes its bytes.
    #[debug_requires(positions.start() <= positions.end(), "a loop has a position")]
    #[debug_requires(
        self.loops.last().is_none_or(|l| l.end() < positions.start()),
        "loops are added in plan order, and none holds another"
    )]
    pub fn add_loop(&mut self, positions: RangeInclusive<usize>) {
        self.loops.push(positions);
    }

    /// Node `pos`'s rows: the same declarations with the position bound,
    /// which is what a task's `layout` is handed.
    pub fn node(&mut self, pos: usize) -> Layout<'_, K> {
        Layout {
            builder: self,
            pos,
            used: 0,
        }
    }

    /// Lay the rows out: each loop's body rows in a table of their own, at
    /// offsets inside the loop's region, and the rest, the regions among
    /// them, with [`assign`].
    #[debug_requires(
        self.carried().all(|(_, _, l)| l.is_some()),
        "a row is written again only inside a loop"
    )]
    #[debug_requires(
        self.carried()
            .all(|(r, _, l)| l.is_none_or(|l| r.users[0].0 < *l.start())),
        "a carried row is produced before its loop"
    )]
    #[debug_requires(
        self.carried().all(|(r, w, l)| l.is_none_or(|l| {
            r.users.iter().all(|&(p, _)| !l.contains(&p) || p <= w)
        })),
        "a carried row's writer is its last user in the loop"
    )]
    pub fn build(self) -> Table<K> {
        let body: Vec<Option<usize>> = self
            .rows
            .iter()
            .map(|r| match r.owner {
                Owner::Worker => None,
                Owner::Task | Owner::Edge => self
                    .loops
                    .iter()
                    .position(|l| r.users.iter().all(|(p, _)| l.contains(p))),
            })
            .collect();
        let lives: Vec<Life> = self
            .rows
            .iter()
            .zip(&body)
            .map(|(r, b)| match (r.life(), b) {
                (
                    Life::Span {
                        mut first,
                        mut last,
                    },
                    None,
                ) => {
                    for l in &self.loops {
                        if first <= *l.end() && *l.start() <= last {
                            first = first.min(*l.start());
                            last = last.max(*l.end());
                        }
                    }
                    Life::Span { first, last }
                }
                (life, _) => life,
            })
            .collect();
        let as_row = |i: usize| Row {
            bytes: self.rows[i].bytes,
            align: DEVICE_ALIGNMENT,
            life: lives[i],
        };
        // Each body laid out alone, from offset 0 of its region.
        let mut offsets = vec![0; self.rows.len()];
        let mut regions = Vec::with_capacity(self.loops.len());
        for (index, l) in self.loops.iter().enumerate() {
            let members: Vec<usize> = (0..self.rows.len())
                .filter(|&i| body[i] == Some(index))
                .collect();
            let local = assign(&members.iter().map(|&i| as_row(i)).collect::<Vec<_>>());
            for (&i, &o) in members.iter().zip(&local.offsets) {
                offsets[i] = o;
            }
            regions.push(Region {
                name: format!("loop {}-{}", l.start(), l.end()),
                positions: l.clone(),
                offset: 0,
                bytes: local.total,
            });
        }
        // The rest, each region a row live for its loop's span.
        let parent: Vec<usize> = (0..self.rows.len())
            .filter(|&i| body[i].is_none())
            .collect();
        let mut rows: Vec<Row> = parent.iter().map(|&i| as_row(i)).collect();
        rows.extend(regions.iter().map(|g| Row {
            bytes: g.bytes,
            align: DEVICE_ALIGNMENT,
            life: Life::Span {
                first: *g.positions.start(),
                last: *g.positions.end(),
            },
        }));
        let assignment = assign(&rows);
        for (&i, &o) in parent.iter().zip(&assignment.offsets) {
            offsets[i] = o;
        }
        for (g, &o) in regions.iter_mut().zip(&assignment.offsets[parent.len()..]) {
            g.offset = o;
        }
        for (i, b) in body.iter().enumerate() {
            if let Some(b) = *b {
                offsets[i] += regions[b].offset;
            }
        }
        Table {
            id: self.id,
            rows: self.rows,
            offsets,
            total: assignment.total,
            lives,
            body,
            regions,
            marker: PhantomData,
        }
    }

    /// Every write after a row's producer: the row, the writer's position,
    /// and the loop the write is in, if any.
    fn carried(&self) -> impl Iterator<Item = (&RowInfo, usize, Option<&RangeInclusive<usize>>)> {
        self.rows.iter().flat_map(move |r| {
            r.users
                .iter()
                .skip(1)
                .filter(|&&(_, a)| a == Access::Write)
                .map(move |&(w, _)| (r, w, self.loops.iter().find(|l| l.contains(&w))))
        })
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

/// One node's rows of one [`TableBuilder`]: [`TableBuilder::node`].
///
/// What a task's `layout` declares through, so it never names its own
/// position: its scratch, its outputs, and the outputs of earlier nodes it
/// reads.
pub struct Layout<'b, K> {
    builder: &'b mut TableBuilder<K>,
    pos: usize,
    used: usize,
}

impl<K: Kind> Layout<'_, K> {
    /// The node's position.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Scratch: `len` elements, written by this node and used by no other.
    #[must_use]
    pub fn scratch<T: Element>(&mut self, name: &str, len: usize) -> RowId<K, T> {
        self.used += 1;
        self.builder.scratch(self.pos, name, len)
    }

    /// An output: `len` elements, written by this node, read by later ones.
    #[must_use]
    pub fn output<T: Element>(&mut self, name: &str, len: usize) -> RowId<K, T> {
        self.used += 1;
        self.builder.output(self.pos, name, len)
    }

    /// This node writes the output `row`, which a node before its loop
    /// produced, as the row's last user in the loop: the value carried to
    /// the next iteration. Returns `row`. [`TableBuilder::write`]'s
    /// contracts hold here too.
    pub fn write<T: Element>(&mut self, row: RowId<K, T>) -> RowId<K, T> {
        self.used += 1;
        self.builder.write(self.pos, row);
        row
    }

    /// This node reads the output `row`, which an earlier node produced.
    /// Returns `row`, so a task's rows can be written `h: l.read(self.h)`.
    /// [`TableBuilder::read`]'s contracts hold here too.
    pub fn read<T: Element>(&mut self, row: RowId<K, T>) -> RowId<K, T> {
        self.used += 1;
        self.builder.read(self.pos, row);
        row
    }

    /// The rows declared or read through this layout so far.
    pub fn used(&self) -> usize {
        self.used
    }
}

/// A laid-out table: every row's offset, and the block's total.
///
/// Its text ([`fmt::Display`]) is the committed memory golden: the rows,
/// their offsets, the peak and the handoffs.
pub struct Table<K> {
    id: u64,
    pub(super) rows: Vec<RowInfo>,
    /// Per row, its offset in the block: a body row's inside its region.
    pub(super) offsets: Vec<usize>,
    total: usize,
    /// Per row, the span its bytes are held: a body row's within its
    /// loop, any other row used in or across a loop spanning it whole.
    pub(super) lives: Vec<Life>,
    /// Per row, the loop whose body it is, by index into `regions`.
    pub(super) body: Vec<Option<usize>>,
    /// Per loop, its region.
    pub(super) regions: Vec<Region>,
    marker: PhantomData<fn() -> K>,
}

/// A loop's region: the bytes the table holds for the loop's whole span,
/// in which its body's rows are laid out.
pub(super) struct Region {
    /// `loop first-last`, as admission names it.
    pub(super) name: String,
    pub(super) positions: RangeInclusive<usize>,
    pub(super) offset: usize,
    /// The body's own total: its peak, laid out.
    pub(super) bytes: usize,
}

/// The instant a table's live rows are largest: [`Table::peak`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peak<'t> {
    /// The position, or `None` when only worker rows are declared.
    pub position: Option<usize>,
    /// The bytes live there.
    pub bytes: usize,
    /// The rows live there, by name and bytes, in declaration order, then
    /// each loop's region live there, named `loop first-last`.
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
    ///
    /// Two rows of one loop's body share bytes within an iteration, and the
    /// pair orders them there; the next iteration starts after the loop's
    /// decision, which is after every body node, so nothing crosses the
    /// iteration edge. A pair from a body node to a node after the loop is
    /// returned from the loop's decision instead, which a node after a
    /// loop may follow and a body node, replayed, may not.
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
                    (self.lives[i], self.lives[j])
                else {
                    unreachable!("a plan-long row shares bytes with no row")
                };
                let (earlier, later) = if la < fb { (a, b) } else { (b, a) };
                let next = later.users[0].0;
                pairs.extend(earlier.users.iter().map(|&(u, _)| self.after_loop(u, next)));
            }
        }
        pairs.into_iter().collect()
    }

    /// The pair `(u, v)` as a plan may hold it: from the decision of the
    /// loop `u` is in when `v` is after that loop, else as it is.
    fn after_loop(&self, u: usize, v: usize) -> (usize, usize) {
        match self.regions.iter().find(|g| g.positions.contains(&u)) {
            Some(g) if v > *g.positions.end() => (*g.positions.end(), v),
            _ => (u, v),
        }
    }

    /// The orderings the data needs: `(producer, reader)` for every reader
    /// of every output, and for a row a loop carries
    /// ([`TableBuilder::write`]) every other user ordered against its
    /// writer, before it if lower in rank and after it if higher. A pair
    /// from a body node to a node after its loop is returned from the
    /// loop's decision, as in [`handoffs`](Table::handoffs). Sorted,
    /// without repeats. With the handoffs, every dependency the table asks
    /// of a plan.
    pub fn reads(&self) -> Vec<(usize, usize)> {
        let mut pairs = BTreeSet::new();
        for r in self.rows.iter().filter(|r| r.owner == Owner::Edge) {
            let producer = r.users[0].0;
            for &(p, _) in &r.users[1..] {
                pairs.insert(self.after_loop(producer, p));
            }
            for &(w, _) in r.users[1..].iter().filter(|&&(_, a)| a == Access::Write) {
                for &(u, _) in &r.users[1..] {
                    if u < w {
                        pairs.insert(self.after_loop(u, w));
                    } else if u > w {
                        pairs.insert(self.after_loop(w, u));
                    }
                }
            }
        }
        pairs.into_iter().collect()
    }

    /// The position at which the live rows' bytes are largest, the lowest
    /// on a tie, a loop's region counted whole, once, at each of its
    /// positions, and its body's rows not again. The total is at least
    /// this; alignment and the greedy layout's gaps are the difference.
    pub fn peak(&self) -> Peak<'_> {
        let positions: BTreeSet<usize> = self
            .rows
            .iter()
            .flat_map(|r| r.users.iter().map(|&(p, _)| p))
            .collect();
        // What the block holds outside the regions, then each region.
        let held = self
            .rows
            .iter()
            .zip(&self.lives)
            .zip(&self.body)
            .filter(|(_, b)| b.is_none())
            .map(|((r, &life), _)| (r.name.as_str(), r.bytes, life))
            .chain(self.regions.iter().map(|g| {
                let life = Life::Span {
                    first: *g.positions.start(),
                    last: *g.positions.end(),
                };
                (g.name.as_str(), g.bytes, life)
            }));
        let live_at = move |pos: Option<usize>| {
            held.clone().filter(move |&(_, _, life)| match (life, pos) {
                (Life::Plan, _) => true,
                (Life::Span { first, last }, Some(p)) => first <= p && p <= last,
                (Life::Span { .. }, None) => false,
            })
        };
        let bytes_at = |pos| live_at(pos).map(|(_, bytes, _)| bytes).sum::<usize>();
        // The largest bytes, and on a tie the lowest position.
        let position = positions
            .into_iter()
            .max_by_key(|&p| (bytes_at(Some(p)), Reverse(p)));
        Peak {
            position,
            bytes: bytes_at(position),
            rows: live_at(position)
                .map(|(name, bytes, _)| (name, bytes))
                .collect(),
        }
    }
}

impl<K> Table<K> {
    /// Row `i`'s line of the text.
    fn fmt_row(&self, f: &mut fmt::Formatter<'_>, i: usize) -> fmt::Result {
        let (row, offset) = (&self.rows[i], self.offsets[i]);
        let owner = match row.owner {
            Owner::Task => "task",
            Owner::Edge => "edge",
            Owner::Worker => "worker",
        };
        let life = match self.lives[i] {
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
            writeln!(f)
        } else {
            writeln!(f, ": {}", users.join(" "))
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
        let in_body =
            |body: Option<usize>| (0..self.rows.len()).filter(move |&i| self.body[i] == body);
        for i in in_body(None) {
            self.fmt_row(f, i)?;
        }
        for (index, g) in self.regions.iter().enumerate() {
            writeln!(
                f,
                "{}: region of {} bytes at {}, its body's rows:",
                g.name, g.bytes, g.offset
            )?;
            for i in in_body(Some(index)) {
                self.fmt_row(f, i)?;
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
