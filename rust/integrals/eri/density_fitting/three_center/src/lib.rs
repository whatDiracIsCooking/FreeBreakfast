//! The three-center Coulomb integrals
//!
//! ```text
//! (mn|P) = ∫∫ m(r₁) n(r₁) P(r₂) / r₁₂ dr₁ dr₂
//! ```
//!
//! of an orbital [`BasisSet`] and an auxiliary (fitting) one, over their
//! real solid harmonics on all three indices, as tasks of the model
//! (`docs/programming_model.md`, "A task declares"), by McMurchie–Davidson
//! over the packed `peri3<la, lb, lP>` kernels FelizJueves emits
//! (`deps/FelizJueves/docs/implementation/three_center.md`; the packed
//! output is its `kernel_eri3.py`'s). The C++ `src/integrals/eri/density_fitting/three_center` is
//! the reference design; its README has the build, and why the weights
//! need no further normalization (the metric's argument). The C++ tensor
//! is square, `nao² naux`; here it is never built whole.
//!
//! [`add_packed_three_center`] is the integrals over a batch of a packed
//! shell-pair map's kept pairs ([`PairMap`]), the body a build of a packed
//! B wraps: after the kernel library's load check and the kernels' limits
//! at planning, `harmonics` and `upload U` for each basis whose U the
//! caller's slot does not hold yet ([`add_harmonics`]); the packed
//! launches of [`add_packed_three_center_tasks`] (`packed three-center
//! tables` on the CPU, the aux records and the cells formed at planning
//! from the basis's [`record_tables`]; `packed three-center` on the
//! stream, the orbital records' bra fields read in place, one launch per
//! class), each pair's `(m, n)` stored once and so no average; then the
//! per-class spherical transform of [`add_spherical_transform`]. Its
//! output, `naux` × the batch's spherical columns with the aux index
//! fastest, is what the fit's left solve against L takes
//! (`fb_integrals_eri_density_fitting_fit`'s `add_batch_fit`). B's heap
//! row is pair-fastest (the pair map's crate doc), so the batch's element
//! `(Q, c)` lands at `first + c + n_columns · Q` there, `first` the
//! batch's first spherical column: the copy into the heap row is a
//! transpose.
//!
//! [`add_packed_b`] is the build loop that makes it: B in a heap row on
//! the host, built by class in batches of an adaptive width, each a
//! counted loop's trip of those tasks, the fit against one L, the
//! transpose and the pitched download into the heap row, so no device row
//! holds B whole.
//!
//! | C++ | Here |
//! |---|---|
//! | `three_center(res, basis, classes, aux, b, max_bytes)`, the square tensor into the caller's `b` | [`add_packed_three_center`] over a batch of kept pairs, and [`add_packed_b`], the batches fitted into B on the host; no square tensor (#1721) |
//! | U and U_aux staged with the aux records and cells | the plan's device rows of U and U_aux ([`add_harmonics`]), each copied up once for every reader over its basis on the worker |
//! | `mmd`'s records copied back, regrouped by shell pair on the host, undoing their sort, and staged up again, the call synchronizing after the copy | nothing copied: each launch reads its class's bra fields in place from `fb_integrals_mmd`'s rows, which are unsorted, each pair's run its prefix in the tables |
//! | the staging, the cells, the Cartesian tensor and the work, the call's own buffers | pinned rows and the stream tasks' device rows, all in the plan's arena |
//! | `three_center_class` per class, `gemm` and `gemmStridedBatched` (`recipes/blas`, wwr) | [`add_packed_three_center_tasks`] (`fb_three_center_packed_class` per class), then [`add_spherical_transform`]'s per-class GEMMs |
//! | `three_center_symmetrize`, the (m, n) average | nothing: each pair's `(m, n)` is computed once |
//! | `Unsupported` for an orbital shell above f or an aux shell above i, or a tensor past `INT32_MAX` | the same, at planning, adding nothing, for a batch's Cartesian columns |
//! | `Unsupported` past `max_bytes` or the device's memory budget, naming the bytes | admission's `Allocation` over the plan's table, naming the rows at the peak; the batch narrowed to the room first |
//! | `InvalidConfig` for a class that is not the basis's | a contract that `classes` are the basis's records, checked always by [`add_packed_three_center_tasks`] against the basis's tables |
//! | success with nothing written for an empty basis | a contract: both bases have a shell |
//! | `three_center_footprint(basis, aux)` | nothing: the layout declares every row, and the arena's size is the plan's golden |

#![forbid(unsafe_code)]

mod build;

pub use build::add_packed_b;

use std::ops::Range;

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Device, RowId};
use fb_core_error_handling::{Error, Result};
use fb_core_planner::Dag;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_transform::add_spherical_transform;
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::{
    AuxTable, ClassRows, FB_THREE_CENTER_AUX_MAX_MOMENTUM, FB_THREE_CENTER_MAX_MOMENTUM,
    add_packed_three_center_tasks,
};
use fb_recipes_matrix_ao::add_harmonics;

/// Add the spherical three-center integrals of the kept shell pairs
/// `pairs` of `map`, in batches of `batch`, to `dag` on GPU worker
/// `worker`, and return them, packed: an output row of `naux · s` doubles,
/// `s` the first batch's spherical columns ([`PairMap::sphe`]), element
/// `(Q, c)` at `Q + naux · c`, the aux index fastest and `c` relative to
/// the batch's first spherical column. Each pair's block is its `(m, n)` alone, once,
/// with no average: a diagonal pair's full square, any other pair's `m`
/// on its bra shell.
///
/// `classes` are the records of `basis`, added on `worker` by
/// [`fb_integrals_mmd::add_records`] earlier in `dag`. `u` and `u_aux` are
/// the plan's slots for U over `basis` and over `aux` on `worker`, shared
/// with every other reader of each (the metric's transform takes
/// `u_aux`'s). `map` is the packed shell-pair map of `basis`, built from
/// its [`record_tables`]. The tasks are `harmonics` and
/// `upload U` for each slot still `None` ([`add_harmonics`]), then those
/// of [`add_packed_three_center_tasks`] (`packed three-center tables`,
/// CPU, and `packed three-center`, stream: the batch's Cartesian columns,
/// aux fastest), then [`add_spherical_transform`]'s (`spherical
/// transform`, stream), whose output this is. The Cartesian batch is the
/// transform's input row, and ends there.
///
/// Batch `k` is the `k`-th `batch` of `pairs`, the last what is left:
/// outside a loop the tasks compute the first, and in a counted loop of
/// `pairs.len().div_ceil(batch)` trips batch `k` in trip `k`, as
/// [`add_packed_b`] builds B; one batch of every pair is
/// `batch == pairs.len()`. Every batch's pairs must be a prefix of the
/// first's classes, pair for pair, so that its columns and its transform
/// are the front of the first's: a short last batch's columns are the
/// front of the output, the rest left. U's slots make U copied up once
/// for every build over them.
///
/// First, before anything is added, the kernel library's load check
/// ([`fb_integrals_sys::check_library`]), and then what the kernels can
/// take, both at planning.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if the loaded
/// `libfb_integrals_kernels.so` is not the one the bindings describe; an
/// `Unsupported` one naming the shell, if an orbital shell is above f
/// ([`FB_THREE_CENTER_MAX_MOMENTUM`]) or an aux shell above i
/// ([`FB_THREE_CENTER_AUX_MAX_MOMENTUM`]), or naming the count, if the
/// batch's Cartesian columns times the aux basis's Cartesian functions do
/// not fit the kernels' 32-bit index, which bounds every count
/// [`add_spherical_transform`] refuses too. Nothing is added to `dag` on
/// an error.
///
/// # Panics
/// If `classes`, or a row `u` or `u_aux` holds, are rows of another
/// worker's or another plan's table ([`Dag::add_gpu`]), or, in every
/// build, if `classes` are not the records of `basis`
/// ([`add_packed_three_center_tasks`]' contract against its tables).
#[debug_requires(basis.nshells() > 0, "an orbital basis of at least one shell")]
#[debug_requires(aux.nshells() > 0, "an aux basis of at least one shell")]
#[debug_requires(
    map.cart().nao() == basis.nao_cart() as usize,
    "map is a map of basis's pairs"
)]
#[debug_requires(pairs.start < pairs.end, "a batch of at least one kept pair")]
#[debug_requires(batch > 0, "a batch of at least one pair")]
#[debug_requires(pairs.end <= map.pairs().len(), "every batch is of map's kept pairs")]
#[allow(
    clippy::too_many_arguments,
    reason = "the bases, their records and U slots, the map, the range of its pairs and the \
              batch"
)]
#[track_caller]
pub fn add_packed_three_center(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    aux: &BasisSet,
    map: &PairMap,
    pairs: Range<usize>,
    batch: usize,
    u: &mut Option<RowId<Device, f64>>,
    u_aux: &mut Option<RowId<Device, f64>>,
) -> Result<RowId<Device, f64>> {
    let table = checked_aux_table("add_packed_three_center", basis, aux)?;
    let blocks = map.cart().blocks();
    let last = pairs.end.min(pairs.start + batch) - 1;
    let n_col = blocks[last].end() - blocks[pairs.start].column;
    let n_aux = aux.nao_cart() as usize;
    if i32::try_from(n_col * n_aux).is_err() {
        return Err(Error::unsupported(format!(
            "add_packed_three_center: the Cartesian batch, {n_col} columns x {n_aux} elements ({} \
             bytes), is past the kernels' 32-bit index",
            8 * n_col * n_aux
        )));
    }
    let n_aux = [n_aux, aux.nao_sphe() as usize];
    let tables = record_tables(basis);
    let kept: Vec<usize> = map.pairs()[pairs.clone()].iter().map(|k| k.pair).collect();
    let u = [
        add_harmonics(dag, worker, basis, u),
        add_harmonics(dag, worker, aux, u_aux),
    ];
    let cartesian =
        add_packed_three_center_tasks(dag, worker, classes, &tables, &table, &kept, batch);
    // Never an error past the refusal above, so nothing is added and then
    // refused: each of the transform's counts is at most `n_col · n_aux`
    // (`n_aux_cart`; `n_aux · n_b_sph` and a class's `pairs · n_a`, since
    // a block's `n_a · n_b` columns are at least `n_b` and `n_a`).
    add_spherical_transform(dag, worker, map, pairs, batch, n_aux, cartesian, u)
}

/// The kernel library's load check, then each shell of `basis` and `aux`
/// within the kernels' momenta, `who` naming the caller in the error; and
/// the aux table of `aux`.
#[track_caller]
fn checked_aux_table(who: &str, basis: &BasisSet, aux: &BasisSet) -> Result<AuxTable> {
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    for (which, set, max) in [
        ("orbital", basis, FB_THREE_CENTER_MAX_MOMENTUM),
        ("aux", aux, FB_THREE_CENTER_AUX_MAX_MOMENTUM),
    ] {
        if let Some((s, shell)) = (0..)
            .zip(set.shells())
            .find(|(_, shell)| shell.angular_momentum() > max)
        {
            return Err(Error::unsupported(format!(
                "{who}: {which} shell {s} has momentum {}, but the three-center kernels cover \
                 {which} shells through l = {max}",
                shell.angular_momentum()
            )));
        }
    }
    let mut table = AuxTable::new(aux.nao_cart() as usize);
    for (s, shell) in (0..).zip(aux.shells()) {
        let c = shell.center();
        table.push_shell(
            shell.angular_momentum(),
            [c.x, c.y, c.z],
            aux.ao_offset_cart(s) as usize,
            shell
                .primitives()
                .iter()
                .map(|p| (p.exponent, p.coefficient)),
        );
    }
    Ok(table)
}
