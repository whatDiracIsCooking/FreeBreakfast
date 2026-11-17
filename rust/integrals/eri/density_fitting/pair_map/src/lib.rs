//! The packed shell-pair map: which shell pairs of an orbital basis a
//! density-fitting factor B keeps, and where each one's block lies among
//! B's columns, built once at planning on the host from the basis's
//! `fb_integrals_mmd::record_tables` and, optionally, a bound per shell pair.
//!
//! **What is kept.** Every shell pair `i <= j` of the tables, once, in
//! their order: class by class, `(la, lb)` ascending, `la <= lb`, and
//! within a class in the order the tables list it. With a bound per pair,
//! the largest Cartesian `sqrt((mn|mn))` of its block, a pair whose bound
//! is below the threshold is dropped whole; with no bounds, or a threshold
//! of 0, every pair is kept. With the Coulomb metric
//! `Σ_Q B_{Q,mn}² = (mn|mn)_DF <= (mn|mn)`, so a dropped column of B has a
//! 2-norm below the threshold.
//!
//! **The packed layout.** Each kept pair ([`KeptPair`]) owns one block of
//! columns ([`PairBlock`]), its bra shell's functions by its ket shell's, side by side in
//! the kept pairs' order, so every class's blocks ([`KeptClass`]) are
//! contiguous and of one shape. Element `(a, b)` of a block, `a` the bra
//! shell's function and `b` the ket's, is column `column + a · n_b + b`.
//! A diagonal shell pair's block (`i == j`) is its full
//! square: `(m, n)` and `(n, m)` are two columns of it. Any other pair's
//! `(m, n)` is one column, which stands for `(n, m)` too. The map has
//! this layout twice, over the Cartesian functions and over the
//! spherical ones ([`Packing`]), with the same pairs in the same order.
//! With every pair kept it is the lower triangle at shell-pair
//! granularity, the diagonal blocks whole.
//!
//! **The heap layout of B.** This is the one definition every reader of B
//! uses. B is `naux` rows of [`Packing::n_columns`] columns, pair-fastest:
//! element `(column, Q)` is at
//!
//! ```text
//! column + n_columns · Q
//! ```
//!
//! so a batch of aux functions `[Q₀, Q₀ + b)` is the one contiguous range
//! `[n_columns · Q₀, n_columns · (Q₀ + b))`, the unit B is streamed in
//! (`docs/memory.md`, "The three kinds": B is a heap row). A range of
//! kept pairs `[p₀, p₁)` is the columns from `p₀`'s first to `p₁`'s, and a
//! tensor over that range alone is the same layout with that range's
//! columns as its width, each pair's column taken relative to `p₀`'s.
//!
//! **The index tables.** [`Packing::write_tables`] writes, per column,
//! its `(m, n)` in the `nao × nao` matrix and its weight in a packed
//! density, 1 on a diagonal shell pair's block and 2 on the others: the
//! `i32` and `f64` arrays the device pack, unpack and gather kernels read,
//! written by a CPU task straight into its pinned rows (`docs/memory.md`,
//! "The table"), since they are as long as a row of B. The map itself
//! holds a block per pair, never a table per column. [`Packing::pack`]
//! and [`Packing::unpack`] are the host's reading of the same tables, the
//! reference those kernels are held to bitwise.
//!
//! **Counts.** Every count here fits the kernels' `i32`: the columns are at
//! most `nao_cart²` (each pair `i < j` contributes `n_i n_j`, each diagonal
//! one `n_i²`, `Σ_{i<=j} ≤ (Σ n_i)²`), which [`RecordTables::new`] holds
//! within `i32` for every basis it accepts, so there is nothing left to
//! refuse at planning.
//!
//! There is no C++ counterpart: the C++ B is square, `[m][n][P]`, and
//! resident on the device.
//!
//! ```
//! use fb_basis::{BasisSet, Shell};
//! use fb_core_types::Double3;
//! use fb_integrals_eri_density_fitting_pair_map::PairMap;
//! use fb_integrals_mmd::record_tables;
//!
//! let at = |z| Double3 { x: 0.0, y: 0.0, z };
//! // An s shell and a d shell, on two centres.
//! let basis = BasisSet::new(vec![
//!     Shell::zipped(0, &[1.0], &[1.0], at(0.0)),
//!     Shell::zipped(2, &[0.5], &[1.0], at(1.5)),
//! ]);
//! let tables = record_tables(&basis);
//!
//! // Every pair: (s,s) 1×1, (s,d) 1×6 and (d,d) 6×6 Cartesian columns.
//! let map = PairMap::new(&tables, None, 0.0);
//! assert_eq!(map.pairs().len(), 3);
//! assert_eq!(map.cart().n_columns(), 1 + 6 + 36);
//! assert_eq!(map.sphe().n_columns(), 1 + 5 + 25);
//!
//! // The (s,d) pair's bound below the threshold: it is dropped whole.
//! let map = PairMap::new(&tables, Some(&[1.0, 1e-12, 1.0]), 1e-10);
//! assert_eq!(map.pairs().iter().map(|p| p.pair).collect::<Vec<_>>(), [0, 2]);
//! assert_eq!(map.sphe().n_columns(), 1 + 25);
//! assert_eq!(map.sphe().blocks()[1].column, 1);
//! ```
//!

#![forbid(unsafe_code)]

use std::ops::Range;

use contracts::debug_requires;
use fb_integrals_sys::RecordTables;

/// The packed map of one basis's shell pairs: the kept pairs, their
/// classes, and the Cartesian and spherical packings over them. Made by
/// [`PairMap::new`] alone, so its tables agree with one another.
#[derive(Clone, Debug, PartialEq)]
pub struct PairMap {
    pairs: Vec<KeptPair>,
    classes: Vec<KeptClass>,
    cart: Packing,
    sphe: Packing,
}

/// One kept shell pair: where it was among the tables' pairs, and its two
/// shells. Its blocks are its index's in each [`Packing::blocks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeptPair {
    /// The pair's index among every pair of the tables, in their order:
    /// its bound's index.
    pub pair: usize,
    /// Its bra and ket shells, indices into the tables' shell table; the
    /// same shell twice for a diagonal pair.
    pub shells: [u32; 2],
}

/// A kept pair's block in one packing: element `(a, b)`, the bra shell's
/// `a`-th function and the ket's `b`-th, is column `column + a · n[1] + b`
/// and stands at `(ao[0] + a, ao[1] + b)` of the `nao × nao` matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairBlock {
    /// The bra and ket shells' first functions.
    pub ao: [usize; 2],
    /// Their function counts: the block is `n[0] × n[1]`.
    pub n: [usize; 2],
    /// The block's first column.
    pub column: usize,
}

impl PairBlock {
    /// Whether the block is a shell's with itself, the full square: its
    /// two first functions are one, as two shells' never are.
    pub fn is_diagonal(&self) -> bool {
        self.ao[0] == self.ao[1]
    }

    /// The column after the block's last.
    pub fn end(&self) -> usize {
        self.column + self.n[0] * self.n[1]
    }
}

/// One class `(la, lb)` with a kept pair: its kept pairs, a contiguous
/// range of [`PairMap::pairs`], whose blocks are therefore contiguous and
/// of one shape in each packing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeptClass {
    /// The bra shells' momentum.
    pub la: i32,
    /// The ket shells', at least `la`.
    pub lb: i32,
    /// Its kept pairs, indices into [`PairMap::pairs`].
    pub pairs: Range<usize>,
}

/// The packed layout over one kind of function, Cartesian or spherical:
/// each kept pair's block, and the per-column tables they make.
#[derive(Clone, Debug, PartialEq)]
pub struct Packing {
    nao: usize,
    blocks: Vec<PairBlock>,
}

impl PairMap {
    /// The map of the basis `tables` were formed from
    /// (`fb_integrals_mmd::record_tables`),
    /// keeping every pair whose bound is not below `threshold`, or every
    /// pair when `bounds` is `None`. The bounds are indexed as the tables
    /// list their pairs, class by class ([`KeptPair::pair`]); a NaN bound
    /// keeps its pair.
    #[debug_requires(
        tables.n_pairs() == tables.shells().ao_offset.len() * (tables.shells().ao_offset.len() + 1) / 2,
        "the tables are one basis's, every shell pair i <= j once"
    )]
    #[debug_requires(
        bounds.is_none_or(|b| b.len() == tables.n_pairs()),
        "a bound per pair of the tables"
    )]
    #[debug_requires(threshold >= 0.0, "the threshold is not negative")]
    pub fn new(tables: &RecordTables, bounds: Option<&[f64]>, threshold: f64) -> PairMap {
        let n_shells = tables.shells().ao_offset.len();
        // Each shell's momentum, from the classes it is a bra or a ket of:
        // every shell is in its diagonal pair at least.
        let mut l = vec![0_i32; n_shells];
        for c in tables.classes() {
            for (&a, &b) in c.bra.iter().zip(&c.ket) {
                l[a as usize] = c.la;
                l[b as usize] = c.lb;
            }
        }
        let n_cart = |l: i32| ((l + 1) * (l + 2) / 2) as usize;
        let n_sphe = |l: i32| (2 * l + 1) as usize;
        let ao_cart: Vec<usize> = tables
            .shells()
            .ao_offset
            .iter()
            .map(|&o| o as usize)
            .collect();
        // The spherical functions follow the shells' order, as the
        // Cartesian ones do.
        let ao_sphe: Vec<usize> = l
            .iter()
            .scan(0, |next, &l| {
                let at = *next;
                *next += n_sphe(l);
                Some(at)
            })
            .collect();
        let nao_sphe = l.iter().map(|&l| n_sphe(l)).sum();

        let mut pairs = Vec::new();
        let mut classes = Vec::new();
        let mut cart = Vec::new();
        let mut sphe = Vec::new();
        let mut index = 0;
        for c in tables.classes() {
            let first = pairs.len();
            for (&a, &b) in c.bra.iter().zip(&c.ket) {
                let keep = bounds.is_none_or(|b| b[index].is_nan() || b[index] >= threshold);
                if keep {
                    cart.push(PairBlock {
                        ao: [ao_cart[a as usize], ao_cart[b as usize]],
                        n: [n_cart(c.la), n_cart(c.lb)],
                        column: cart.last().map_or(0, PairBlock::end),
                    });
                    sphe.push(PairBlock {
                        ao: [ao_sphe[a as usize], ao_sphe[b as usize]],
                        n: [n_sphe(c.la), n_sphe(c.lb)],
                        column: sphe.last().map_or(0, PairBlock::end),
                    });
                    pairs.push(KeptPair {
                        pair: index,
                        shells: [a, b],
                    });
                }
                index += 1;
            }
            if pairs.len() > first {
                classes.push(KeptClass {
                    la: c.la,
                    lb: c.lb,
                    pairs: first..pairs.len(),
                });
            }
        }
        let (cart, sphe) = (
            Packing {
                nao: tables.n_cart(),
                blocks: cart,
            },
            Packing {
                nao: nao_sphe,
                blocks: sphe,
            },
        );
        // INVARIANT: the Cartesian columns are at most nao_cart² (the crate
        // doc), which RecordTables::new holds within i32, and the spherical
        // ones at most the Cartesian.
        debug_assert!(
            i32::try_from(cart.n_columns()).is_ok() && sphe.n_columns() <= cart.n_columns()
        );
        PairMap {
            pairs,
            classes,
            cart,
            sphe,
        }
    }

    /// The kept pairs, in the tables' order.
    pub fn pairs(&self) -> &[KeptPair] {
        &self.pairs
    }

    /// The classes with a kept pair, ascending in `(la, lb)`.
    pub fn classes(&self) -> &[KeptClass] {
        &self.classes
    }

    /// The packing over the Cartesian functions: the three-centre
    /// kernels' output.
    pub fn cart(&self) -> &Packing {
        &self.cart
    }

    /// The packing over the spherical functions: B's.
    pub fn sphe(&self) -> &Packing {
        &self.sphe
    }
}

impl Packing {
    /// The functions: the matrix a packed vector stands for is
    /// `nao × nao`.
    pub fn nao(&self) -> usize {
        self.nao
    }

    /// The columns: a packed vector's length, and B's row width.
    pub fn n_columns(&self) -> usize {
        self.blocks.last().map_or(0, PairBlock::end)
    }

    /// Each kept pair's block, in [`PairMap::pairs`]' order, side by side.
    pub fn blocks(&self) -> &[PairBlock] {
        &self.blocks
    }

    /// Each column's `(m, n)` in the `nao × nao` matrix, its block's row
    /// and column, and whether its block is diagonal, in column order.
    fn columns(&self) -> impl Iterator<Item = (usize, usize, bool)> {
        self.blocks.iter().flat_map(|b| {
            (0..b.n[0]).flat_map(move |a| {
                (0..b.n[1]).map(move |k| (b.ao[0] + a, b.ao[1] + k, b.is_diagonal()))
            })
        })
    }

    /// The index tables, written in place (a CPU task's pinned rows): per
    /// column `c`, `rows[c]` its row `m` in the `nao × nao` matrix (its
    /// pair's bra shell's function), `cols[c]` its column `n` (the ket
    /// shell's), and `weights[c]` its weight in a packed density,
    /// `D_packed = w · pack(D)`: 1 on a diagonal shell pair's block, where
    /// `(m, n)` and `(n, m)` are columns of their own, and 2 on any other,
    /// whose one column stands for both; so `Σ_c w_c x_c y_c = Σ_{mn}
    /// X_{mn} Y_{mn}` for symmetric X and Y supported on the kept pairs.
    #[debug_requires(rows.len() == self.n_columns(), "rows is a column each")]
    #[debug_requires(cols.len() == self.n_columns(), "cols is a column each")]
    #[debug_requires(weights.len() == self.n_columns(), "weights is a column each")]
    pub fn write_tables(&self, rows: &mut [i32], cols: &mut [i32], weights: &mut [f64]) {
        let out = rows.iter_mut().zip(cols.iter_mut()).zip(weights.iter_mut());
        for (((r, c), w), (m, n, diagonal)) in out.zip(self.columns()) {
            // Every AO index is below nao, whose square RecordTables::new
            // holds within i32.
            (*r, *c) = (m as i32, n as i32);
            *w = if diagonal { 1.0 } else { 2.0 };
        }
    }

    /// The symmetric `nao × nao` matrix `matrix` packed into `packed`:
    /// column `c` at `(m, n)` is `matrix[m · nao + n]`, a copy. A screened
    /// pair's elements are read by no column.
    #[debug_requires(matrix.len() == self.nao * self.nao, "matrix is nao × nao")]
    #[debug_requires(packed.len() == self.n_columns(), "packed is a column each")]
    pub fn pack(&self, matrix: &[f64], packed: &mut [f64]) {
        for (p, (m, n, _)) in packed.iter_mut().zip(self.columns()) {
            *p = matrix[m * self.nao + n];
        }
    }

    /// The packed vector `packed` unpacked into the symmetric `nao × nao`
    /// matrix `matrix`, every element written: zero where no kept pair
    /// covers it, and else, for column `c` at `(m, n)`, both `(m, n)` and
    /// `(n, m)` set to `packed[c]` when its block is not diagonal (weight
    /// 2) or `m >= n`. So a diagonal block is read from its lower triangle
    /// and its upper columns are not read, and `(m, n)` and `(n, m)` are
    /// the same bits whatever `packed` holds. [`Packing::pack`] then this
    /// is the identity, bitwise, on a symmetric matrix supported on the
    /// kept pairs.
    #[debug_requires(packed.len() == self.n_columns(), "packed is a column each")]
    #[debug_requires(matrix.len() == self.nao * self.nao, "matrix is nao × nao")]
    pub fn unpack(&self, packed: &[f64], matrix: &mut [f64]) {
        matrix.fill(0.0);
        for (&v, (m, n, diagonal)) in packed.iter().zip(self.columns()) {
            if !diagonal || m >= n {
                matrix[m * self.nao + n] = v;
                matrix[n * self.nao + m] = v;
            }
        }
    }
}
