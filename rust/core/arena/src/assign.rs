//! The offset assigner: rows with sizes, alignments and lifetimes in, an
//! offset per row and the block's total out. Pure; no device.

use std::cmp::Reverse;

use contracts::debug_requires;

/// When a row's bytes are in use: for the whole plan, or from the position
/// of its first user to that of its last, both inclusive.
///
/// A position is a node's rank in the plan. Two spans that share a position
/// overlap, so a row that ends at a node and one that starts there never
/// share bytes: there is no in-place alias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Life {
    /// Live at every position: a worker's row.
    Plan,
    /// Live from `first` to `last`, inclusive.
    Span {
        /// The first user's position.
        first: usize,
        /// The last user's position.
        last: usize,
    },
}

impl Life {
    /// Whether the two are live at some common position.
    pub fn overlaps(self, other: Life) -> bool {
        match (self, other) {
            (
                Life::Span {
                    first: a0,
                    last: a1,
                },
                Life::Span {
                    first: b0,
                    last: b1,
                },
            ) => a0 <= b1 && b0 <= a1,
            _ => true,
        }
    }

    fn start(self) -> usize {
        match self {
            Life::Plan => 0,
            Life::Span { first, .. } => first,
        }
    }
}

/// One buffer to place: its bytes, the alignment its offset must honour and
/// its lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row {
    /// Its size. A row of zero bytes is placed at 0 and conflicts with
    /// nothing.
    pub bytes: usize,
    /// A power of two its offset is a multiple of.
    pub align: usize,
    /// When it is live.
    pub life: Life,
}

impl Row {
    /// Whether the two may not share bytes: both are non-empty and live at
    /// some common position.
    pub fn conflicts(&self, other: &Row) -> bool {
        self.bytes > 0 && other.bytes > 0 && self.life.overlaps(other.life)
    }
}

/// [`assign`]'s result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Assignment {
    /// Per row, in the order given, its offset in the block.
    pub offsets: Vec<usize>,
    /// The block's size: the largest end of a non-empty row, or 0.
    pub total: usize,
}

/// Give every row an offset such that no two conflicting rows
/// ([`Row::conflicts`]) share a byte, each offset a multiple of its row's
/// alignment.
///
/// Greedy with best fit, the static memory planning a tensor compiler
/// does, run in three orders and kept at the smallest total, the earlier
/// order on a tie: each row in turn goes into the smallest gap between the
/// placed rows it conflicts with that holds it, else past the last of them.
/// The orders, plan-long rows first in each, are by first position and then
/// the larger first; by first position and then the longer-lived first;
/// and the larger first whatever its position. No one order is smallest on
/// every committed golden, and the three together hold each golden's block
/// within 3% of its lower bound (`docs/memory.md`, "Lifetimes").
/// Deterministic in its input. The total is not minimal in general; the
/// schedule search in `docs/programming_model.md`, "Open questions", says
/// when that may matter.
#[debug_requires(
    rows.iter().all(|r| r.align.is_power_of_two()),
    "every alignment is a power of two"
)]
#[debug_requires(
    rows.iter().all(|r| match r.life {
        Life::Span { first, last } => first <= last,
        Life::Plan => true,
    }),
    "a span's first position is at most its last"
)]
pub fn assign(rows: &[Row]) -> Assignment {
    let span = |r: &Row| matches!(r.life, Life::Span { .. });
    let last = |r: &Row| match r.life {
        Life::Plan => usize::MAX,
        Life::Span { last, .. } => last,
    };
    // The first order, then each other where it is strictly smaller.
    let mut best = place(
        rows,
        &order(rows, |i, r| (r.life.start(), span(r), Reverse(r.bytes), i)),
    );
    for order in [
        order(rows, |i, r| {
            (
                r.life.start(),
                span(r),
                Reverse(last(r)),
                Reverse(r.bytes),
                i,
            )
        }),
        order(rows, |i, r| (span(r), Reverse(r.bytes), r.life.start(), i)),
    ] {
        let a = place(rows, &order);
        if a.total < best.total {
            best = a;
        }
    }
    best
}

/// The rows' indices sorted by `key`, given each index and its row.
fn order<K: Ord>(rows: &[Row], key: impl Fn(usize, &Row) -> K) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| key(i, &rows[i]));
    order
}

/// Place the rows greedily in `order`, each at [`fit`] among the placed rows
/// it conflicts with.
fn place(rows: &[Row], order: &[usize]) -> Assignment {
    let mut offsets = vec![0; rows.len()];
    let mut placed: Vec<usize> = Vec::with_capacity(rows.len());
    let mut taken: Vec<(usize, usize)> = Vec::new();
    for &i in order {
        let row = &rows[i];
        if row.bytes == 0 {
            continue;
        }
        taken.clear();
        taken.extend(
            placed
                .iter()
                .filter(|&&j| row.conflicts(&rows[j]))
                .map(|&j| (offsets[j], offsets[j] + rows[j].bytes)),
        );
        taken.sort_unstable();
        offsets[i] = fit(&taken, row.bytes, row.align);
        placed.push(i);
    }
    let total = rows
        .iter()
        .zip(&offsets)
        .filter(|(r, _)| r.bytes > 0)
        .map(|(r, &o)| o + r.bytes)
        .max()
        .unwrap_or(0);
    Assignment { offsets, total }
}

/// The offset for `bytes` at `align` among the byte ranges `taken`, sorted
/// by start and possibly overlapping one another: the smallest free gap
/// that holds it, the lowest on a tie, else the first aligned offset past
/// them all.
fn fit(taken: &[(usize, usize)], bytes: usize, align: usize) -> usize {
    let mut best: Option<(usize, usize)> = None;
    let mut cursor = 0;
    for &(start, end) in taken {
        if start > cursor {
            let offset = cursor.next_multiple_of(align);
            if offset + bytes <= start {
                let free = start - cursor;
                if best.is_none_or(|(f, _)| free < f) {
                    best = Some((free, offset));
                }
            }
        }
        cursor = cursor.max(end);
    }
    best.map_or_else(|| cursor.next_multiple_of(align), |(_, offset)| offset)
}
