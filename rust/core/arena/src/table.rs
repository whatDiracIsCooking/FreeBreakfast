//! The table: a plan's rows for one worker and one memory kind, who uses
//! each, and the offsets [`assign`] gave them.

use std::cmp::Reverse;
use std::collections::BTreeSet;
use std::fmt;
use std::marker::PhantomData;
use std::ops::RangeInclusive;
use std::sync::atomic::{AtomicU64, Ordering};

use contracts::{debug_requires, requires};
use fb_core_log::eq;

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
    /// The GPU worker's, every lane of it: live for the whole plan.
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

/// A loop as declared to a [`TableBuilder`].
struct LoopDecl {
    positions: RangeInclusive<usize>,
    /// How many copies of the body's table the region holds: 1 for a
    /// decision loop, the depth for a counted one.
    copies: usize,
    /// Whether its last position is a decision, after every body node.
    decides: bool,
}

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
    /// Each loop, in plan order: its positions, its copies of the body,
    /// and whether it ends at a decision ([`TableBuilder::add_loop`],
    /// [`TableBuilder::add_counted`]).
    loops: Vec<LoopDecl>,
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

    /// A row the GPU worker holds for the whole plan, such as a library
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
    /// it, and in no loop it holds, is the body's: the body's rows are laid
    /// out as a table of their own, in a region the rest of the table holds
    /// for the loop's whole span, sized to the body's peak once. Every other
    /// row used in the loop, or live across it, is live for the whole loop,
    /// so no body row takes its bytes.
    ///
    /// Its body may hold counted loops, declared before it: each one's
    /// region is a row of this body's table, live over the counted loop's
    /// span, and the rows of this body live across it are live over the
    /// whole of it.
    #[debug_requires(positions.start() <= positions.end(), "a loop has a position")]
    #[debug_requires(
        self.loops.iter().all(|l| l.positions.end() < positions.start() || !l.decides),
        "a decision loop holds no decision loop"
    )]
    #[debug_requires(
        self.loops.iter().all(|l| {
            l.positions.end() < positions.start()
                || (positions.start() <= l.positions.start() && l.positions.end() < positions.end())
        }),
        "loops are added as they close, each holding an earlier one in its body whole or not at all"
    )]
    pub fn add_loop(&mut self, positions: RangeInclusive<usize>) {
        self.loops.push(LoopDecl {
            positions,
            copies: 1,
            decides: true,
        });
    }

    /// A counted loop of the plan over `positions`, its body's nodes, run
    /// a trip count fixed at planning with up to `depth` iterations in
    /// flight at once and no decision (`docs/memory.md`, "Nested plans").
    /// Its region is `depth` copies of the body's table, the body laid out
    /// once and repeated at a stride, and iteration `k` runs in copy
    /// `k mod depth` ([`Block::partition`](super::Block::partition)): the
    /// plan must start iteration `k` only after iteration `k - depth`, so
    /// that no two iterations hold one copy at once, and run each body
    /// node after itself in the iteration before.
    ///
    /// A row the body writes that is not the body's own, an accumulator
    /// carried across iterations ([`TableBuilder::write`]) or an output
    /// read after the loop, is used by one body node alone, since two
    /// iterations in flight would otherwise reach its one copy in no
    /// order; [`TableBuilder::build`] holds that.
    ///
    /// It may lie in a decision loop's body, declared before that loop
    /// ([`TableBuilder::add_loop`]): its region is then a row of that
    /// body's table.
    #[debug_requires(positions.start() <= positions.end(), "a loop has a position")]
    #[debug_requires(depth > 0, "a counted loop has a copy")]
    #[debug_requires(
        self.loops.last().is_none_or(|l| l.positions.end() < positions.start()),
        "loops are added as they close, and a counted loop holds no loop"
    )]
    pub fn add_counted(&mut self, positions: RangeInclusive<usize>, depth: usize) {
        self.loops.push(LoopDecl {
            positions,
            copies: depth,
            decides: false,
        });
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
    /// offsets inside the loop's region (a counted loop's region that
    /// table's copies), a counted loop's region inside a decision loop's
    /// laid out as a row of that loop's body, and the rest, the outermost
    /// regions among them, with [`assign`].
    #[debug_requires(
        self.loops.iter().filter(|l| !l.decides).all(|l| self.rows.iter().all(|r| {
            let inside = r.users.iter().filter(|(p, _)| l.positions.contains(p)).count();
            let writes = r.users.iter().any(|&(p, a)| a == Access::Write && l.positions.contains(&p));
            inside == r.users.len() || !writes || inside == 1
        })),
        "a row a counted loop writes and does not own has one user in its body"
    )]
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
        // Each loop's parent, the decision loop whose body holds it, which
        // was declared after it.
        let parent: Vec<Option<usize>> = (0..self.loops.len())
            .map(|i| {
                let p = &self.loops[i].positions;
                (i + 1..self.loops.len()).find(|&j| self.loops[j].positions.contains(p.start()))
            })
            .collect();
        // Whether loop `a` is loop `b` or holds it.
        let holds = |a: usize, b: usize| {
            let mut at = Some(b);
            while let Some(x) = at {
                if x == a {
                    return true;
                }
                at = parent[x];
            }
            false
        };
        // Each row's loop, the innermost holding every user: a child is
        // declared before its parent, so the first found.
        let body: Vec<Option<usize>> = self
            .rows
            .iter()
            .map(|r| match r.owner {
                Owner::Worker => None,
                Owner::Task | Owner::Edge => self
                    .loops
                    .iter()
                    .position(|l| r.users.iter().all(|(p, _)| l.positions.contains(p))),
            })
            .collect();
        // A row's span extended over every loop it overlaps but its own and
        // the loops holding that one.
        let lives: Vec<Life> = self
            .rows
            .iter()
            .zip(&body)
            .map(|(r, &b)| match r.life() {
                Life::Span {
                    mut first,
                    mut last,
                } => {
                    for (i, l) in self.loops.iter().enumerate() {
                        let mine = b.is_some_and(|b| holds(i, b));
                        let l = &l.positions;
                        if !mine && first <= *l.end() && *l.start() <= last {
                            first = first.min(*l.start());
                            last = last.max(*l.end());
                        }
                    }
                    Life::Span { first, last }
                }
                life => life,
            })
            .collect();
        let as_row = |i: usize| Row {
            bytes: self.rows[i].bytes,
            align: DEVICE_ALIGNMENT,
            life: lives[i],
        };
        let span = |l: &LoopDecl| Life::Span {
            first: *l.positions.start(),
            last: *l.positions.end(),
        };
        // Each body laid out alone, from offset 0 of its region, a child's
        // region one of its rows, and a counted loop's repeated at a
        // stride, the body's total aligned. A child comes before its
        // parent, so its region's bytes are known when the parent's body is
        // laid out; the offsets are made absolute below.
        let mut offsets = vec![0; self.rows.len()];
        let mut regions: Vec<Region> = Vec::with_capacity(self.loops.len());
        for (index, l) in self.loops.iter().enumerate() {
            let members: Vec<usize> = (0..self.rows.len())
                .filter(|&i| body[i] == Some(index))
                .collect();
            let children: Vec<usize> = (0..index).filter(|&c| parent[c] == Some(index)).collect();
            let mut rows: Vec<Row> = members.iter().map(|&i| as_row(i)).collect();
            rows.extend(children.iter().map(|&c| Row {
                bytes: regions[c].bytes,
                align: DEVICE_ALIGNMENT,
                life: span(&self.loops[c]),
            }));
            let local = assign(&rows);
            for (&i, &o) in members.iter().zip(&local.offsets) {
                offsets[i] = o;
            }
            for (&c, &o) in children.iter().zip(&local.offsets[members.len()..]) {
                regions[c].offset = o;
            }
            let stride = local.total.next_multiple_of(DEVICE_ALIGNMENT);
            regions.push(Region {
                name: format!("loop {}-{}", l.positions.start(), l.positions.end()),
                positions: l.positions.clone(),
                offset: 0,
                bytes: (l.copies - 1) * stride + local.total,
                copies: l.copies,
                stride,
                decides: l.decides,
                parent: parent[index],
            });
        }
        // The rest, each outermost region a row live for its loop's span.
        let outside: Vec<usize> = (0..self.rows.len())
            .filter(|&i| body[i].is_none())
            .collect();
        let outermost: Vec<usize> = (0..regions.len())
            .filter(|&g| parent[g].is_none())
            .collect();
        let mut rows: Vec<Row> = outside.iter().map(|&i| as_row(i)).collect();
        rows.extend(outermost.iter().map(|&g| Row {
            bytes: regions[g].bytes,
            align: DEVICE_ALIGNMENT,
            life: span(&self.loops[g]),
        }));
        let assignment = assign(&rows);
        for (&i, &o) in outside.iter().zip(&assignment.offsets) {
            offsets[i] = o;
        }
        for (&g, &o) in outermost.iter().zip(&assignment.offsets[outside.len()..]) {
            regions[g].offset = o;
        }
        // A region's offset made absolute, a parent's first.
        for g in (0..regions.len()).rev() {
            if let Some(p) = parent[g] {
                regions[g].offset += regions[p].offset;
            }
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
    /// and the innermost loop the write is in, if any.
    fn carried(&self) -> impl Iterator<Item = (&RowInfo, usize, Option<&RangeInclusive<usize>>)> {
        self.rows.iter().flat_map(move |r| {
            r.users
                .iter()
                .skip(1)
                .filter(|&&(_, a)| a == Access::Write)
                .map(move |&(w, _)| {
                    // A child is declared before its parent.
                    let l = self
                        .loops
                        .iter()
                        .map(|l| &l.positions)
                        .find(|l| l.contains(&w));
                    (r, w, l)
                })
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
    /// Per row, the innermost loop whose body it is, by index into
    /// `regions`.
    pub(super) body: Vec<Option<usize>>,
    /// Per loop, its region.
    pub(super) regions: Vec<Region>,
    marker: PhantomData<fn() -> K>,
}

/// A loop's region: the bytes the table holds for the loop's whole span,
/// in which its body's rows are laid out, once or, for a counted loop,
/// once per copy.
pub(super) struct Region {
    /// `loop first-last`, as admission names it.
    pub(super) name: String,
    pub(super) positions: RangeInclusive<usize>,
    pub(super) offset: usize,
    /// Every copy of the body: `copies - 1` strides and the body's own
    /// total, its peak laid out.
    pub(super) bytes: usize,
    /// The copies of the body's table: 1 but for a counted loop.
    pub(super) copies: usize,
    /// The bytes from one copy to the next: the body's total, aligned.
    pub(super) stride: usize,
    /// Whether the loop ends at a decision.
    pub(super) decides: bool,
    /// The decision loop whose body holds it, by index into `regions`:
    /// its region is then a row of that body's table.
    pub(super) parent: Option<usize>,
}

/// The instant a table's live rows are largest: [`Table::peak`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Peak<'t> {
    /// The position, or `None` when only worker rows are declared.
    pub position: Option<usize>,
    /// The bytes live there.
    pub bytes: usize,
    /// The rows live there, by name and bytes, in declaration order, then
    /// each outermost loop's region live there, named `loop first-last`.
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

    #[requires(eq!(row.table, self.id), "a row of this table")]
    pub(super) fn index<T: Element>(&self, row: RowId<K, T>) -> usize {
        row.index
    }

    /// Row `i`'s copies and the stride between them: a counted loop's body
    /// row has its region's, any other row one.
    pub(super) fn copies(&self, i: usize) -> (usize, usize) {
        match self.body[i] {
            Some(g) => (self.regions[g].copies, self.regions[g].stride),
            None => (1, 0),
        }
    }

    /// The orderings that reusing bytes needs: `(u, v)` for every user `u`
    /// of a row whose bytes a later row takes, `v` being that later row's
    /// producer. Sorted, without repeats.
    ///
    /// The layout packs rows against the positions, but a plan runs a
    /// ready node whatever its rank, and the lanes of a worker run at once,
    /// so the positions alone do not order two rows that share bytes. Every
    /// pair returned must therefore be an edge of the plan, or implied by
    /// its edges: the previous owner of an offset is a predecessor of the
    /// next (`docs/memory.md`, "Lifetimes"). Between two lanes that edge is
    /// an event on the device, so every reuse across lanes serializes them:
    /// the table trades concurrency for bytes, always.
    ///
    /// Two rows of one loop's body share bytes within an iteration, and the
    /// pair orders them there. Across the iteration edge: a decision
    /// loop's next iteration starts after its decision, which is after
    /// every body node; a counted loop's iteration `k` takes copy `k mod
    /// depth` after iteration `k - depth` has finished with it, which the
    /// plan orders. A pair from a body node to a node after a decision
    /// loop is returned from the loop's decision instead, which a node
    /// after a loop may follow and a body node, replayed, may not; after
    /// a counted loop, as it is, from the body node's last iteration.
    /// A row outside a counted loop's body is held against every copy of
    /// it, a row of the decision loop holding it among them; a pair from
    /// a body node of a counted loop to a later node of the decision loop
    /// holding it is from that node's last trip.
    pub fn handoffs(&self) -> Vec<(usize, usize)> {
        let mut pairs = BTreeSet::new();
        // A row's bytes, every copy of it from the first to the last.
        let extent = |i: usize| {
            let (copies, stride) = self.copies(i);
            (copies - 1) * stride + self.rows[i].bytes
        };
        for (i, a) in self.rows.iter().enumerate() {
            for (j, b) in self.rows.iter().enumerate().skip(i + 1) {
                if a.bytes == 0 || b.bytes == 0 {
                    continue;
                }
                // Two rows of one body share bytes copy for copy; any
                // other pair, over every copy of each.
                let (ea, eb) = if self.body[i].is_some() && self.body[i] == self.body[j] {
                    (a.bytes, b.bytes)
                } else {
                    (extent(i), extent(j))
                };
                let (oa, ob) = (self.offsets[i], self.offsets[j]);
                if oa >= ob + eb || ob >= oa + ea {
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
    /// decision loop `u` is in when `v` is after that loop, else as it is.
    /// A pair leaving only a counted loop, to a node of the decision loop
    /// holding it or after it, stays as it is: the plan runs it from the
    /// writer's last trip.
    fn after_loop(&self, u: usize, v: usize) -> (usize, usize) {
        match self
            .regions
            .iter()
            .find(|g| g.decides && g.positions.contains(&u))
        {
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
    /// on a tie, an outermost loop's region counted whole, once, at each of
    /// its positions, and its body's rows, and the regions of the loops it
    /// holds, not again. The total is at least
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
            .chain(self.regions.iter().filter(|g| g.parent.is_none()).map(|g| {
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
            write!(f, "{}: region of {} bytes at {}", g.name, g.bytes, g.offset)?;
            if !g.decides {
                write!(f, ", {} copies {} bytes apart", g.copies, g.stride)?;
            }
            if let Some(p) = g.parent {
                write!(f, ", in {}'s body", self.regions[p].name)?;
            }
            writeln!(f, ", its body's rows:")?;
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
