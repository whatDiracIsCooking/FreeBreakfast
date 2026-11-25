//! The scratch packer: the transitive closure of the step DAG, first fit
//! largest-first, and the lower bound. Pure functions over indices and
//! bytes: no device, no ledger.
//!
//! Two regions may share bytes only if the DAG orders their steps, one
//! reaching the other. That is the packer's constraint and not its
//! objective: any placement that keeps unordered regions apart is safe, so
//! a poorer packing costs bytes and never a collision.

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
/// If `preds` has a cycle; `Dag::build` refuses one before this runs.
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
pub fn pack(reach: &[Reach], regions: &[Region]) -> Packed {
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by(|&a, &b| {
        (regions[b].bytes, regions[a].name).cmp(&(regions[a].bytes, regions[b].name))
    });
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

/// The heaviest antichain of `regions`: the largest total of regions no
/// two of which the DAG orders, and the regions that make it. A branch and
/// bound; at a few dozen regions it is instant.
pub fn lower_bound(reach: &[Reach], regions: &[Region]) -> (u64, Vec<usize>) {
    fn go(
        reach: &[Reach],
        regions: &[Region],
        order: &[usize],
        at: usize,
        chosen: &mut Vec<usize>,
        total: u64,
        left: u64,
        best: &mut (u64, Vec<usize>),
    ) {
        if total > best.0 {
            *best = (total, chosen.clone());
        }
        if at == order.len() || total + left <= best.0 {
            return;
        }
        let r = order[at];
        let bytes = regions[r].bytes;
        if chosen
            .iter()
            .all(|&c| !ordered(reach, regions[c].step, regions[r].step))
        {
            chosen.push(r);
            go(
                reach,
                regions,
                order,
                at + 1,
                chosen,
                total + bytes,
                left - bytes,
                best,
            );
            chosen.pop();
        }
        go(
            reach,
            regions,
            order,
            at + 1,
            chosen,
            total,
            left - bytes,
            best,
        );
    }
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by(|&a, &b| {
        (regions[b].bytes, regions[a].name).cmp(&(regions[a].bytes, regions[b].name))
    });
    let mut best = (0, Vec::new());
    let left = regions.iter().map(|r| r.bytes).sum();
    go(
        reach,
        regions,
        &order,
        0,
        &mut Vec::new(),
        0,
        left,
        &mut best,
    );
    best.1.sort_unstable();
    best
}
