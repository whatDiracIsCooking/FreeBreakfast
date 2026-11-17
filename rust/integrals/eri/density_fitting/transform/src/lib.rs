//! The Cartesian-to-spherical transform of a packed three-centre batch, as
//! a task of a [`Dag`]: for a range of a [`PairMap`]'s kept shell pairs,
//! their Cartesian columns of `(mn|P)`, over the Cartesian functions of
//! both the orbital and the aux basis, become their spherical columns,
//!
//! ```text
//! (αβ|Q) = Σ_P Σ_ab U_aux[P, Q] U[a, α] U[b, β] (ab|P)
//! ```
//!
//! block by block, each kept pair's block alone: U is block-diagonal by
//! shell (`fb_basis::spherical_transform_matrix`), so no pair's
//! spherical block reads another's Cartesian one, and a range of pairs is
//! transformed apart from the rest of the map.
//!
//! **The layouts.** Input and output are the map's packed columns
//! (`fb_integrals_eri_density_fitting_pair_map`'s crate doc) over the
//! batch alone, each pair's column taken relative to the batch's first,
//! with the aux index fastest: the input is `n_aux_cart` × the batch's
//! Cartesian columns ([`Packing::blocks`] of [`PairMap::cart`]), element
//! `(P, c)` at `P + n_aux_cart · c`, as the packed three-centre kernels
//! write it, and the output `n_aux` × its spherical columns
//! ([`PairMap::sphe`]), `(Q, c)` at `Q + n_aux · c`, column-major, ready
//! for the fit's left solve against L. Inside a block, element `(a, b)` is
//! column `column + a · n_b + b`, `a` the bra shell's function.
//!
//! **The task**, **`spherical transform`** on the worker's stream, reads
//! the input and the plan's device rows of U and U_aux
//! (`fb_recipes_matrix_ao::add_harmonics`'), and runs, on the worker's
//! BLAS handle:
//!
//! 1. the aux index, one GEMM over every column: `W = U_auxᵀ X`, into the
//!    scratch `transform aux`, `n_aux` × the Cartesian columns;
//! 2. per class `(la, lb)` with a pair in the batch, whose blocks the map
//!    keeps contiguous and of one shape ([`PairMap::classes`]), two
//!    strided-batched GEMMs, U's block of the class's momentum shared by
//!    every product (stride 0), read in place from U's row at the first
//!    pair's shells, since every shell of one momentum has the same block:
//!    - the ket index, one product per pair and bra function `a`: the
//!      `n_aux × n_b` slab of W at `(a, ·)` times `U_lb`, into the scratch
//!      `transform ket`, `n_aux × n_b_sph` per slab;
//!    - the bra index, one product per pair: the pair's
//!      `(n_aux · n_b_sph) × n_a` view of that scratch times `U_la`, into
//!      the output, which is then the pair's spherical block, `(α, β)` at
//!      column `column + α · n_b_sph + β`.
//!
//! Every element of the output is written by exactly one product, so
//! nothing is zeroed. The two scratch rows and the output are in the plan's
//! table at planning, so a batch that does not fit is refused by
//! admission with the rows named.
//!
//! An s or p class's U block is a permutation or the identity, and still
//! goes through its two GEMMs: the products are exact, and one path is
//! one thing to test.
//!
//! There is no C++ counterpart: the C++ (mn|P) is square, `[m][n][P]`, and
//! its transform the dense three GEMMs `fb_integrals_sys`'s three-center
//! task runs.
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
//! /// The packed Cartesian batch's producer, the three-centre launches.
//! struct ThreeCentre {
//!     len: usize,
//! }
//!
//! impl GpuTask for ThreeCentre {
//!     type Rows = RowId<Device, f64>;
//!
//!     fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
//!         device.output("(mn|P) cartesian", self.len)
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
//! let n_aux = [aux.nao_cart() as usize, aux.nao_sphe() as usize];
//!
//! let mut dag = Dag::new();
//! let cartesian = dag.add_gpu(
//!     "three-centre",
//!     0,
//!     ThreeCentre {
//!         len: n_aux[0] * map.cart().n_columns(),
//!     },
//! );
//! let u = add_harmonics(&mut dag, 0, &basis, &mut None);
//! let u_aux = add_harmonics(&mut dag, 0, &aux, &mut None);
//! // Every kept pair as one batch: n_aux × map.sphe().n_columns() doubles.
//! let b = add_spherical_transform(
//!     &mut dag,
//!     0,
//!     &map,
//!     0..map.pairs().len(),
//!     n_aux,
//!     cartesian,
//!     [u, u_aux],
//! )?;
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
use fb_core_wwr::blas::{gemm, gemm_strided_batched};
use fb_core_wwr::device_slice::{split_at, split_at_mut};
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};

/// Add the spherical transform of the packed Cartesian batch `cartesian`,
/// the kept pairs `pairs` of `map`, to `dag` on GPU worker `worker`, and
/// return it: an output row of `n_aux · s` doubles, `s` the batch's
/// spherical columns, element `(Q, c)` at `Q + n_aux · c`, `c` relative
/// to the batch's first spherical column.
///
/// `cartesian` is an earlier task's output of `n_aux_cart · k` doubles,
/// `k` the batch's Cartesian columns, `(P, c)` at `P + n_aux_cart · c`;
/// `n_aux` is `[n_aux_cart, n_aux]`, the aux basis's Cartesian and
/// spherical counts; and `[u, u_aux]` the device rows of the orbital and
/// the aux basis's U, `map.cart().nao()` × `map.sphe().nao()` and
/// `n_aux_cart` × `n_aux`, column-major, written by earlier tasks of `dag`
/// on `worker` (`fb_recipes_matrix_ao::add_harmonics`'). The task added is
/// `spherical transform` (stream), which reads the three, holds the
/// aux-transformed batch (`n_aux · k` doubles) and the ket-transformed one
/// as scratch, and leaves `cartesian` as it was.
///
/// # Errors
/// An `Unsupported` error naming the count, if a GEMM's count or leading
/// dimension would not fit the BLAS's 32-bit index; nothing is added to
/// `dag` then.
///
/// # Panics
/// If a row is of another worker's or another plan's table
/// ([`Dag::add_gpu`]), or, at run time, if `cartesian`, U's or U_aux's row
/// is shorter than the extent above (`fb_core_wwr::blas`'s GEMMs).
#[debug_requires(pairs.start < pairs.end, "a batch of at least one kept pair")]
#[debug_requires(pairs.end <= map.pairs().len(), "the batch is of map's kept pairs")]
#[debug_requires(
    1 <= n_aux[1] && n_aux[1] <= n_aux[0],
    "U_aux has 1 to its n_aux_cart rows of columns"
)]
#[track_caller]
pub fn add_spherical_transform(
    dag: &mut Dag<'_>,
    worker: usize,
    map: &PairMap,
    pairs: Range<usize>,
    n_aux: [usize; 2],
    cartesian: RowId<Device, f64>,
    u: [RowId<Device, f64>; 2],
) -> Result<RowId<Device, f64>> {
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
    let mut largest = n_aux[0];
    for class in map.classes() {
        let r = class.pairs.start.max(pairs.start)..class.pairs.end.min(pairs.end);
        if r.is_empty() {
            continue;
        }
        let (c, s) = (cart[r.start], sphe[r.start]);
        largest = largest
            .max(n_aux[1].saturating_mul(s.n[1]))
            .max(r.len() * c.n[0]);
        steps.push(Step {
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
            "add_spherical_transform: a GEMM count of {largest}, for {} aux functions over the \
             batch's {} Cartesian columns, is past the BLAS's 32-bit index",
            n_aux[1], columns[0]
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
            },
        )
        .out)
}

/// One class's share of a batch: its pairs' blocks, contiguous in each of
/// the three layouts, and where U's blocks of its momenta lie.
#[derive(Debug, Clone, Copy)]
struct Step {
    /// The class's pairs in the batch.
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

/// GPU: the aux index, then each class's ket and bra index.
struct Transform {
    cartesian: RowId<Device, f64>,
    /// U and U_aux, the caller's device rows.
    u: [RowId<Device, f64>; 2],
    n_aux: [usize; 2],
    /// U's rows, the orbital basis's Cartesian functions.
    ld_u: usize,
    /// The batch's Cartesian and spherical columns.
    columns: [usize; 2],
    /// The ket-transformed scratch's columns.
    ket_columns: usize,
    steps: Vec<Step>,
}

#[derive(Clone, Copy)]
struct TransformRows {
    aux: RowId<Device, f64>,
    ket: RowId<Device, f64>,
    out: RowId<Device, f64>,
}

impl GpuTask for Transform {
    type Rows = TransformRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.cartesian);
        device.read(self.u[0]);
        device.read(self.u[1]);
        let n_aux = self.n_aux[1];
        TransformRows {
            aux: device.scratch("transform aux", n_aux * self.columns[0]),
            ket: device.scratch("transform ket", n_aux * self.ket_columns),
            out: device.output("(mn|P) packed", n_aux * self.columns[1]),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("n_aux", self.n_aux[1]),
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
        let (u, u_aux) = (device.get(self.u[0]), device.get(self.u[1]));
        let mut aux = device.take_mut(rows.aux);
        let mut ket = device.take_mut(rows.ket);
        let mut out = device.take_mut(rows.out);
        let handle = worker.blas();

        // In range, as every cast below: add_spherical_transform refused a
        // count past i32 (n_aux_cart, n_aux · n_b_sph, a class's pairs ·
        // n_a), and the map holds its columns and nao within it.
        let [n_aux_cart, n_aux] = self.n_aux;
        let (nac, na, ld_u) = (n_aux_cart as i32, n_aux as i32, self.ld_u as i32);
        // The aux index: W = U_auxᵀ X, n_aux × the Cartesian columns.
        gemm(
            handle,
            WWR_BLAS_OP_T,
            WWR_BLAS_OP_N,
            na,
            self.columns[0] as i32,
            nac,
            1.0,
            &u_aux,
            nac,
            &input,
            nac,
            0.0,
            &mut aux,
            na,
        )?;
        for s in &self.steps {
            let [ca, cb] = s.n_cart.map(|n| n as i32);
            let [sa, sb] = s.n_sph.map(|n| n as i32);
            // The ket index: per pair and bra function, the n_aux × n_b
            // slab of W times U_lb, into an n_aux × n_b_sph slab.
            let (_, w) = split_at(&aux, n_aux * s.cart);
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
                (s.pairs * s.n_cart[0]) as i32,
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
                s.pairs as i32,
            )?;
        }
        Ok(())
    }
}
