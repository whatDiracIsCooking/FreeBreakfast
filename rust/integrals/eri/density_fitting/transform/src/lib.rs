//! The orbital half of the Cartesian-to-spherical transform of a packed
//! three-centre batch, as a task of a [`Dag`]: for a range of a
//! [`PairMap`]'s kept shell pairs, their Cartesian columns of `(mn|Q)`,
//! over the Cartesian functions of the orbital basis and the spherical
//! ones of the aux basis, become their spherical columns,
//!
//! ```text
//! (αβ|Q) = Σ_ab U[a, α] U[b, β] (ab|Q)
//! ```
//!
//! block by block, each kept pair's block alone: U is block-diagonal by
//! shell (`fb_basis::spherical_transform_matrix`), so no pair's
//! spherical block reads another's Cartesian one, and a range of pairs is
//! transformed apart from the rest of the map. The aux index arrives
//! spherical: the packed three-centre kernels write it so
//! (`fb_integrals_sys::add_packed_three_center_tasks`).
//!
//! **The layouts.** Input and output are the map's packed columns
//! (`fb_integrals_eri_density_fitting_pair_map`'s crate doc) over the
//! batch alone, each pair's column taken relative to the batch's first,
//! with the aux index fastest: the input is `n_aux` × the batch's
//! Cartesian columns ([`Packing::blocks`] of [`PairMap::cart`]), element
//! `(Q, c)` at `Q + n_aux · c`, as the packed three-centre kernels
//! write it, and the output `n_aux` × its spherical columns
//! ([`PairMap::sphe`]), `(Q, c)` at `Q + n_aux · c`, column-major, ready
//! for the fit's left solve against L. Inside a block, element `(a, b)` is
//! column `column + a · n_b + b`, `a` the bra shell's function.
//!
//! **The task**, **`spherical transform`** on the worker's stream, reads
//! the input and the plan's device row of U
//! (`fb_recipes_matrix_ao::add_harmonics`'), and runs, on the worker's
//! BLAS handle, per class `(la, lb)` with a pair in the batch, whose
//! blocks the map keeps contiguous and of one shape
//! ([`PairMap::classes`]), two strided-batched GEMMs, U's block of the
//! class's momentum shared by every product (stride 0), read in place
//! from U's row at the first pair's shells, since every shell of one
//! momentum has the same block:
//!
//! - the ket index, one product per pair and bra function `a`: the
//!   `n_aux × n_b` slab of the input at `(a, ·)` times `U_lb`, into the
//!   scratch `transform ket`, `n_aux × n_b_sph` per slab;
//! - the bra index, one product per pair: the pair's
//!   `(n_aux · n_b_sph) × n_a` view of that scratch times `U_la`, into
//!   the output, which is then the pair's spherical block, `(α, β)` at
//!   column `column + α · n_b_sph + β`.
//!
//! Every element of the output is written by exactly one product, so
//! nothing is zeroed. The scratch row and the output are in the plan's
//! table at planning, so a batch that does not fit is refused by
//! admission with the rows named.
//!
//! An s or p class's U block is a permutation or the identity, and still
//! goes through its two GEMMs: the products are exact, and one path is
//! one thing to test.
//!
//! There is no C++ counterpart: the C++ (mn|P) is square, `[m][n][P]`, its
//! aux index Cartesian, and its transform the dense three GEMMs
//! `fb_integrals_sys`'s three-center task runs.
//!
//! ```no_run
//! use fb_basis::{BasisSet, Shell};
//! use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
//! use fb_core_error_handling::Status;
//! use fb_core_planner::{Dag, GpuTask};
//! use fb_core_thread_pool::DeviceStream;
//! use fb_integrals_eri_density_fitting_pair_map::PairMap;
//! use fb_integrals_eri_density_fitting_transform::add_spherical_transform;
//! use fb_integrals_mmd::record_tables;
//! use fb_recipes_matrix_ao::add_harmonics;
//!
//! /// The packed batch's producer, the three-centre launches.
//! struct ThreeCentre {
//!     len: usize,
//! }
//!
//! impl GpuTask for ThreeCentre {
//!     type Rows = RowId<Device, f64>;
//!
//!     fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
//!         device.output("(mn|Q) cartesian", self.len)
//!     }
//!
//!     fn run(
//!         &self,
//!         _: &Self::Rows,
//!         _: &Bufs<'_, Device>,
//!         _: &CopyOnly<'_, Pinned>,
//!         _: &DeviceStream,
//!     ) -> Status {
//!         Ok(())
//!     }
//! }
//!
//! let shell = |l, z| Shell::zipped(l, &[1.0], &[1.0], fb_core_types::Double3 { x: 0.0, y: 0.0, z });
//! let basis = BasisSet::new(vec![shell(0, 0.0), shell(2, 1.5)]);
//! let aux = BasisSet::new(vec![shell(0, 0.0), shell(3, 1.5)]);
//! let map = PairMap::new(&record_tables(&basis), None, 0.0);
//! let n_aux = aux.nao_sphe() as usize;
//!
//! let mut dag = Dag::new();
//! let cartesian = dag.add_gpu(
//!     "three-centre",
//!     0,
//!     ThreeCentre {
//!         len: n_aux * map.cart().n_columns(),
//!     },
//! );
//! let u = add_harmonics(&mut dag, 0, &basis, &mut None);
//! // Every kept pair as one batch: n_aux × map.sphe().n_columns() doubles.
//! let every = map.pairs().len();
//! let b = add_spherical_transform(&mut dag, 0, &map, 0..every, every, n_aux, cartesian, u)?;
//! # Ok::<(), fb_core_error_handling::Error>(())
//! ```
//!
//! [`Packing::blocks`]: fb_integrals_eri_density_fitting_pair_map::Packing::blocks

#![forbid(unsafe_code)]

use std::ops::Range;

use contracts::debug_requires;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::gemm_strided_batched;
use fb_core_wwr::device_slice::{split_at, split_at_mut};
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N;

/// Add the spherical transform of the packed Cartesian batch `cartesian`,
/// of the kept pairs `pairs` of `map` in batches of `batch`, to `dag` on
/// GPU worker `worker`, and return it: an output row of `n_aux · s`
/// doubles, `s` the first batch's spherical columns, element `(Q, c)` at
/// `Q + n_aux · c`, `c` relative to the batch's first spherical column.
///
/// Batch `k` is the `k`-th `batch` of `pairs`, the last what is left:
/// outside a loop the task transforms the first, and in a counted loop of
/// `pairs.len().div_ceil(batch)` trips batch `k` in trip `k`, from its
/// views' iteration. The rows and the GEMMs' shapes are the first
/// batch's, and every batch's blocks are a prefix of its, pair for pair,
/// so a short last batch is transformed by the same GEMMs over its own
/// pairs' counts, its columns the front of each row, the rest left.
///
/// `cartesian` is an earlier task's output of `n_aux · k` doubles, `k` the
/// first batch's Cartesian columns, `(Q, c)` at `Q + n_aux · c`, `n_aux`
/// the aux basis's spherical functions; and `u` the device row of the
/// orbital basis's U, `map.cart().nao()` × `map.sphe().nao()`,
/// column-major, written by an earlier task of `dag` on `worker`
/// (`fb_recipes_matrix_ao::add_harmonics`'). The task added is
/// `spherical transform` (stream), which reads the two, holds the
/// ket-transformed batch as scratch, and leaves `cartesian` as it was.
///
/// # Errors
/// An `Unsupported` error naming the count, if a GEMM's count or leading
/// dimension would not fit the BLAS's 32-bit index; nothing is added to
/// `dag` then.
///
/// # Panics
/// If a row is of another worker's or another plan's table
/// ([`Dag::add_gpu`]), or, at run time, if `cartesian` or U's row is
/// shorter than the extent above (`fb_core_wwr::blas`'s GEMMs).
#[debug_requires(pairs.start < pairs.end, "a batch of at least one kept pair")]
#[debug_requires(batch > 0, "a batch of at least one pair")]
#[debug_requires(pairs.end <= map.pairs().len(), "the batches are of map's kept pairs")]
#[debug_requires(
    prefixes(map, pairs.clone(), batch),
    "every batch's blocks are a prefix of the first's, pair for pair"
)]
#[debug_requires(n_aux > 0, "an aux basis of at least one function")]
#[allow(
    clippy::too_many_arguments,
    reason = "the map's range of pairs and its batch, the aux count, the input and U's row"
)]
#[track_caller]
pub fn add_spherical_transform(
    dag: &mut Dag<'_>,
    worker: usize,
    map: &PairMap,
    pairs: Range<usize>,
    batch: usize,
    n_aux: usize,
    cartesian: RowId<Device, f64>,
    u: RowId<Device, f64>,
) -> Result<RowId<Device, f64>> {
    let all = pairs.clone();
    let pairs = pairs.start..pairs.end.min(pairs.start + batch);
    let (cart, sphe) = (map.cart().blocks(), map.sphe().blocks());
    let first = [cart[pairs.start].column, sphe[pairs.start].column];
    let columns = [
        cart[pairs.end - 1].end() - first[0],
        sphe[pairs.end - 1].end() - first[1],
    ];
    let ld_u = map.cart().nao();
    let mut steps = Vec::new();
    let mut ket_columns = 0;
    // The counts the GEMMs take as i32, beside those every map holds within
    // it (its columns, its nao, a shell's functions).
    let mut largest = n_aux;
    for class in map.classes() {
        let r = class.pairs.start.max(pairs.start)..class.pairs.end.min(pairs.end);
        if r.is_empty() {
            continue;
        }
        let (c, s) = (cart[r.start], sphe[r.start]);
        largest = largest
            .max(n_aux.saturating_mul(s.n[1]))
            .max(r.len() * c.n[0]);
        steps.push(Step {
            place: r.start - pairs.start,
            pairs: r.len(),
            n_cart: c.n,
            n_sph: s.n,
            cart: c.column - first[0],
            ket: ket_columns,
            sphe: s.column - first[1],
            u: [c.ao[0] + ld_u * s.ao[0], c.ao[1] + ld_u * s.ao[1]],
        });
        ket_columns += r.len() * c.n[0] * s.n[1];
    }
    if i32::try_from(largest).is_err() {
        return Err(Error::unsupported(format!(
            "add_spherical_transform: a GEMM count of {largest}, for {n_aux} aux functions over \
             the batch's {} Cartesian columns, is past the BLAS's 32-bit index",
            columns[0]
        )));
    }
    Ok(dag
        .add_gpu(
            "spherical transform",
            worker,
            Transform {
                cartesian,
                u,
                n_aux,
                ld_u,
                columns,
                ket_columns,
                steps,
                all: all.len(),
                batch: pairs.len(),
            },
        )
        .out)
}

/// Whether every batch of `batch` of the kept pairs `pairs` of `map` has
/// blocks of the first's shapes, place for place, as far as it goes.
fn prefixes(map: &PairMap, pairs: Range<usize>, batch: usize) -> bool {
    let (cart, sphe) = (map.cart().blocks(), map.sphe().blocks());
    let shape = |p: usize| (cart[p].n, sphe[p].n);
    pairs
        .clone()
        .all(|p| shape(p) == shape(pairs.start + (p - pairs.start) % batch))
}

/// One class's share of a batch: its pairs' blocks, contiguous in each of
/// the three layouts, and where U's blocks of its momenta lie.
#[derive(Debug, Clone, Copy)]
struct Step {
    /// The class's first pair's place in the batch.
    place: usize,
    /// The class's pairs in the first batch.
    pairs: usize,
    /// The bra and ket shells' Cartesian functions.
    n_cart: [usize; 2],
    /// Their spherical functions.
    n_sph: [usize; 2],
    /// The first pair's first column in the input, in the ket-transformed
    /// scratch and in the output, relative to the batch.
    cart: usize,
    ket: usize,
    sphe: usize,
    /// The offsets in U's row of the bra and the ket shell's blocks,
    /// `n_cart(l)` × `n_sph(l)` at U's leading dimension.
    u: [usize; 2],
}

/// GPU: each class's ket and bra index.
struct Transform {
    cartesian: RowId<Device, f64>,
    /// U, the caller's device row.
    u: RowId<Device, f64>,
    /// The aux basis's spherical functions.
    n_aux: usize,
    /// U's rows, the orbital basis's Cartesian functions.
    ld_u: usize,
    /// The batch's Cartesian and spherical columns.
    columns: [usize; 2],
    /// The ket-transformed scratch's columns.
    ket_columns: usize,
    steps: Vec<Step>,
    /// Every batch's pairs, and the first's.
    all: usize,
    batch: usize,
}

#[derive(Clone, Copy)]
struct TransformRows {
    ket: RowId<Device, f64>,
    out: RowId<Device, f64>,
}

impl GpuTask for Transform {
    type Rows = TransformRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.cartesian);
        device.read(self.u);
        let n_aux = self.n_aux;
        TransformRows {
            ket: device.scratch("transform ket", n_aux * self.ket_columns),
            out: device.output("(mn|P) packed", n_aux * self.columns[1]),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("n_aux", self.n_aux),
            ("columns_cart", self.columns[0]),
            ("columns", self.columns[1]),
        ]
    }

    fn run(
        &self,
        rows: &TransformRows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let input = device.get(self.cartesian);
        let u = device.get(self.u);
        let mut ket = device.take_mut(rows.ket);
        let mut out = device.take_mut(rows.out);
        let handle = worker.blas();

        // This trip's pairs, and each class's of them: the front of the
        // first batch's, so of its steps, each cut to the trip's pairs.
        let trip = self.batch.min(self.all - device.iteration() * self.batch);
        let pairs_of = |s: &Step| s.pairs.min(trip.saturating_sub(s.place));

        // In range, as every cast below: add_spherical_transform refused a
        // count past i32 (n_aux, n_aux · n_b_sph, a class's pairs · n_a),
        // the trip's counts are at most the first batch's, and the map
        // holds its columns and nao within it.
        let n_aux = self.n_aux;
        let (na, ld_u) = (n_aux as i32, self.ld_u as i32);
        for s in &self.steps {
            let pairs = pairs_of(s);
            if pairs == 0 {
                break;
            }
            let [ca, cb] = s.n_cart.map(|n| n as i32);
            let [sa, sb] = s.n_sph.map(|n| n as i32);
            // The ket index: per pair and bra function, the n_aux × n_b
            // slab of the input times U_lb, into an n_aux × n_b_sph slab.
            let (_, w) = split_at(&input, n_aux * s.cart);
            let (_, u_b) = split_at(&u, s.u[1]);
            let (_, mut k) = split_at_mut(&mut ket, n_aux * s.ket);
            gemm_strided_batched(
                handle,
                WWR_BLAS_OP_N,
                WWR_BLAS_OP_N,
                na,
                sb,
                cb,
                1.0,
                &w,
                na,
                i64::from(na) * i64::from(cb),
                &u_b,
                ld_u,
                0,
                0.0,
                &mut k,
                na,
                i64::from(na) * i64::from(sb),
                (pairs * s.n_cart[0]) as i32,
            )?;
            // The bra index: per pair, its (n_aux n_b_sph) × n_a view of the
            // ket-transformed slabs times U_la, into its spherical block.
            let slab = na * sb;
            let (_, k) = split_at(&ket, n_aux * s.ket);
            let (_, u_a) = split_at(&u, s.u[0]);
            let (_, mut o) = split_at_mut(&mut out, n_aux * s.sphe);
            gemm_strided_batched(
                handle,
                WWR_BLAS_OP_N,
                WWR_BLAS_OP_N,
                slab,
                sa,
                ca,
                1.0,
                &k,
                slab,
                i64::from(slab) * i64::from(ca),
                &u_a,
                ld_u,
                0,
                0.0,
                &mut o,
                slab,
                i64::from(slab) * i64::from(sa),
                pairs as i32,
            )?;
        }
        Ok(())
    }
}
