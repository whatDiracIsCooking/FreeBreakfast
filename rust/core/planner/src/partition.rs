//! A partition of rows over workers: the axis a task's work is sliced
//! along (`docs/programming_model.md`, "A task declares").

use std::ops::Range;

use contracts::debug_requires;

/// `total` rows cut into consecutive ranges, one per part: part `w`, the
/// one GPU worker `w` holds, is rows [`range(w)`](Partition::range). What a
/// task's partition is when it splits its output's rows over the workers
/// (`docs/programming_model.md`, "Two kinds of distribution"), and what a
/// gather of the pieces needs to put each back in place
/// (`Dag::gather`).
///
/// ```
/// use fb_core_planner::Partition;
///
/// let even = Partition::even(10, 4);
/// assert_eq!((0..4).map(|w| even.range(w)).collect::<Vec<_>>(), [0..3, 3..6, 6..8, 8..10]);
/// let uneven = Partition::new(&[1, 0, 5]);
/// assert_eq!((uneven.parts(), uneven.total(), uneven.range(2)), (3, 6, 1..6));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    /// Each part's first row, and last the total: `parts + 1` of them.
    bounds: Vec<usize>,
}

impl Partition {
    /// `total` rows over `parts` parts as evenly as they go: the first
    /// `total mod parts` parts have one row more than the rest.
    #[debug_requires(parts > 0, "a partition has a part")]
    pub fn even(total: usize, parts: usize) -> Self {
        let (each, extra) = (total / parts, total % parts);
        let counts: Vec<usize> = (0..parts).map(|w| each + usize::from(w < extra)).collect();
        Self::new(&counts)
    }

    /// Part `w` holds `counts[w]` rows, in order; a part may hold none.
    #[debug_requires(!counts.is_empty(), "a partition has a part")]
    pub fn new(counts: &[usize]) -> Self {
        let mut bounds = Vec::with_capacity(counts.len() + 1);
        bounds.push(0);
        let mut at = 0;
        for &c in counts {
            at += c;
            bounds.push(at);
        }
        Partition { bounds }
    }

    /// How many parts.
    pub fn parts(&self) -> usize {
        self.bounds.len() - 1
    }

    /// The rows of every part together.
    pub fn total(&self) -> usize {
        self.bounds[self.bounds.len() - 1]
    }

    /// Part `part`'s rows.
    ///
    /// # Panics
    /// If `part` is not below [`parts`](Partition::parts), as an index out
    /// of a slice's bounds does.
    pub fn range(&self, part: usize) -> Range<usize> {
        self.bounds[part]..self.bounds[part + 1]
    }
}
