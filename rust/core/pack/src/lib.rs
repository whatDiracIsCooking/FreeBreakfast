//! The scratch packer: the transitive closure of the step DAG, first fit
//! largest first, and the lower bound (`docs/memory.md`, "Step scratch").
//! Pure functions over indices and bytes: no device, no ledger.
//!
//! Ported from the v0.0.4 prototype's `src/pack.rs` as it is; the C++ tree
//! has no counterpart, since its steps allocate as they go.
//!
//! Two regions may share bytes only if the DAG orders their steps, one
//! reaching the other. That is the packer's constraint and not its
//! objective: any placement that keeps unordered regions apart is safe, so
//! a poorer packing costs bytes and never a collision.
//!
//! ```
//! use fb_core_pack::{closure, pack, Region};
//!
//! // a -> b, and c beside both.
//! let reach = closure(&[vec![], vec![0], vec![]]);
//! let regions = [
//!     Region { step: 0, name: "a", bytes: 8 },
//!     Region { step: 1, name: "b", bytes: 4 },
//!     Region { step: 2, name: "c", bytes: 2 },
//! ];
//! let packed = pack(&reach, &regions);
//! // b shares a's bytes; c, ordered against neither, goes above them.
//! assert_eq!(packed.offsets, [0, 0, 8]);
//! assert_eq!((packed.peak, packed.sum, packed.bound), (10, 14, 10));
//! assert_eq!(packed.bound_set, [0, 2]);
//! ```

#![forbid(unsafe_code)]

use contracts::debug_requires;

#[cfg(test)]
mod tests;

/// The steps a step reaches, as a bitset over step indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reach(Vec<u64>);

impl Reach {
    fn new(n: usize) -> Reach {
        Reach(vec![0; n.div_ceil(64)])
    }

    fn set(&mut self, i: usize) {
        self.0[i / 64] |= 1 << (i % 64);
    }

    /// Whether step `i` is in the set.
    #[must_use]
    pub fn has(&self, i: usize) -> bool {
        self.0[i / 64] >> (i % 64) & 1 == 1
    }

    fn union(&mut self, other: &Reach) {
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a |= b;
        }
    }
}

/// The transitive closure of the DAG whose step `i` reads the steps
/// `preds[i]`: `closure[i]` is every step a path leads to from `i`.
///
/// # Panics
/// If `preds` has a cycle, found once the topological order is built.
#[debug_requires(
    preds.iter().flatten().all(|&p| p < preds.len()),
    "every predecessor is a step of the graph"
)]
#[must_use]
pub fn closure(preds: &[Vec<usize>]) -> Vec<Reach> {
    let n = preds.len();
    let mut succs = vec![Vec::new(); n];
    let mut left: Vec<usize> = preds.iter().map(Vec::len).collect();
    for (i, of) in preds.iter().enumerate() {
        for &p in of {
            succs[p].push(i);
        }
    }
    let mut order: Vec<usize> = (0..n).filter(|&i| left[i] == 0).collect();
    let mut at = 0;
    while at < order.len() {
        for &s in &succs[order[at]] {
            left[s] -= 1;
            if left[s] == 0 {
                order.push(s);
            }
        }
        at += 1;
    }
    assert_eq!(order.len(), n, "the step graph has a cycle");
    let mut reach = vec![Reach::new(n); n];
    // Sinks first: a step reaches its successors and what they reach.
    for &i in order.iter().rev() {
        let mut mine = Reach::new(n);
        for &s in &succs[i] {
            mine.set(s);
            mine.union(&reach[s]);
        }
        reach[i] = mine;
    }
    reach
}

/// Whether the DAG orders `a` and `b`: one reaches the other.
#[must_use]
pub fn ordered(reach: &[Reach], a: usize, b: usize) -> bool {
    reach[a].has(b) || reach[b].has(a)
}

/// One step's scratch declaration on one device and kind.
#[derive(Clone, Debug)]
pub struct Region {
    /// The step's index in the closure.
    pub step: usize,
    /// The step's name: the tie-break, so the offsets do not depend on the
    /// order the steps were added in.
    pub name: &'static str,
    /// The declared bytes.
    pub bytes: u64,
}

/// A slab's packing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    /// Each region's offset, in the order the regions were given.
    pub offsets: Vec<u64>,
    /// The slab's size: the largest end.
    pub peak: u64,
    /// The plain sum of the declarations.
    pub sum: u64,
    /// The heaviest set of mutually unordered regions: no packing that
    /// keeps unordered regions apart is smaller.
    pub bound: u64,
    /// The regions (indices into the input) that make the bound.
    pub bound_set: Vec<usize>,
}

/// Pack `regions` into one slab: first fit, largest first and then by step
/// name, each region at the lowest offset clear of every region already
/// placed whose step the DAG does not order against its own.
#[debug_requires(
    regions.iter().all(|r| r.step < reach.len()),
    "every region's step is in the closure"
)]
#[debug_requires(
    regions.iter().try_fold(0u64, |s, r| s.checked_add(r.bytes)).is_some(),
    "the declarations' sum fits a u64, so no offset or end overflows"
)]
#[must_use]
pub fn pack(reach: &[Reach], regions: &[Region]) -> Packed {
    let order = largest_first(regions);
    let mut offsets = vec![0u64; regions.len()];
    let mut placed: Vec<usize> = Vec::new();
    for &r in &order {
        let bytes = regions[r].bytes;
        // The regions this one must stay clear of, by offset.
        let mut apart: Vec<(u64, u64)> = placed
            .iter()
            .filter(|&&p| !ordered(reach, regions[p].step, regions[r].step))
            .map(|&p| (offsets[p], offsets[p] + regions[p].bytes))
            .collect();
        apart.sort_unstable();
        let mut at = 0;
        for (begin, end) in apart {
            if at + bytes <= begin {
                break;
            }
            at = at.max(end);
        }
        offsets[r] = at;
        placed.push(r);
    }
    let peak = regions
        .iter()
        .zip(&offsets)
        .map(|(r, o)| o + r.bytes)
        .max()
        .unwrap_or(0);
    let (bound, bound_set) = lower_bound(reach, regions);
    Packed {
        offsets,
        peak,
        sum: regions.iter().map(|r| r.bytes).sum(),
        bound,
        bound_set,
    }
}

/// The regions' indices, largest first and then by step name.
fn largest_first(regions: &[Region]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by(|&a, &b| {
        (regions[b].bytes, regions[a].name).cmp(&(regions[a].bytes, regions[b].name))
    });
    order
}

/// The heaviest antichain of `regions`: the largest total of regions no
/// two of which the DAG orders, and the regions that make it, sorted. A
/// branch and bound; at a few dozen regions it is instant.
#[debug_requires(
    regions.iter().all(|r| r.step < reach.len()),
    "every region's step is in the closure"
)]
#[debug_requires(
    regions.iter().try_fold(0u64, |s, r| s.checked_add(r.bytes)).is_some(),
    "the declarations' sum fits a u64"
)]
#[must_use]
pub fn lower_bound(reach: &[Reach], regions: &[Region]) -> (u64, Vec<usize>) {
    let mut search = Search {
        reach,
        regions,
        order: largest_first(regions),
        chosen: Vec::new(),
        best: (0, Vec::new()),
    };
    search.go(0, 0, regions.iter().map(|r| r.bytes).sum());
    let mut best = search.best;
    best.1.sort_unstable();
    best
}

/// The branch and bound's state: the regions largest first, the antichain
/// being built and the heaviest found.
struct Search<'a> {
    reach: &'a [Reach],
    regions: &'a [Region],
    order: Vec<usize>,
    chosen: Vec<usize>,
    best: (u64, Vec<usize>),
}

impl Search<'_> {
    /// Decide `order[at..]`, the chosen regions weighing `total` and the
    /// undecided ones `left`.
    fn go(&mut self, at: usize, total: u64, left: u64) {
        if total > self.best.0 {
            self.best = (total, self.chosen.clone());
        }
        if at == self.order.len() || total + left <= self.best.0 {
            return;
        }
        let r = self.order[at];
        let bytes = self.regions[r].bytes;
        if self
            .chosen
            .iter()
            .all(|&c| !ordered(self.reach, self.regions[c].step, self.regions[r].step))
        {
            self.chosen.push(r);
            self.go(at + 1, total + bytes, left - bytes);
            self.chosen.pop();
        }
        self.go(at + 1, total, left - bytes);
    }
}
