//! The three-center Coulomb integrals
//!
//! ```text
//! (mn|P) = ∫∫ m(r₁) n(r₁) P(r₂) / r₁₂ dr₁ dr₂
//! ```
//!
//! of an orbital [`BasisSet`] and an auxiliary (fitting) one, over their
//! real solid harmonics on all three indices, as tasks of the model
//! (`docs/programming_model.md`, "A task declares"), by McMurchie–Davidson
//! over the `eri3<la, lb, lP>` kernels FelizJueves emits
//! (`deps/FelizJueves/docs/implementation/three_center.md`). The C++
//! `src/integrals/eri/density_fitting/three_center` is the reference
//! design; its README has the build, and why the weights need no further
//! normalization (the metric's argument).
//!
//! [`add_three_center`] adds, after the kernel library's load check and the
//! kernels' limits at planning, the three tasks of
//! [`fb_integrals_sys::add_three_center_tasks`]:
//!
//! - **`three-center download`**, on the worker's stream: the orbital
//!   records' bra fields (`fb_integrals_mmd`'s, read as an edge) into
//!   pinned rows.
//! - **`three-center tables`**, on the CPU: those records regrouped by
//!   shell pair, the aux records (the metric's, one per aux primitive,
//!   `s = c (π/α)^{3/2}`), U and U_aux into one pinned row, and the cells,
//!   one per (shell pair, aux shell) in its class `(la, lb, lP)`, into
//!   another.
//! - **`three-center`**, on the worker's stream: both uploaded; one launch
//!   per class into the Cartesian tensor, `nao_cart² naux_cart` doubles,
//!   every element a plain store by the one cell it belongs to, so nothing
//!   is zeroed; the three spherical GEMMs (the aux index against `U_auxᵀ`,
//!   the n index against U per block of m, a strided-batched GEMM, the m
//!   index against U); and the (m, n) average out of place into the
//!   output, `(mn|P)` and `(nm|P)` the same bits.
//!
//! The Cartesian tensor and the transform's intermediate are the last
//! task's scratch, and the output its output row, so all three are in the
//! plan's table at planning: a pair of sets that does not fit is refused by
//! admission ([`fb_core_arena::admit`]), with the rows at the peak named.
//!
//! | C++ | Here |
//! |---|---|
//! | `three_center(res, basis, classes, aux, b, max_bytes)`, into the caller's `b` | [`add_three_center`]`(dag, worker, basis, classes, aux)`, returning (mn\|P)'s row |
//! | `mmd`'s records copied back and regrouped on the host, the call synchronizing after the copy | `three-center download` and `three-center tables`: a stream task, its pinned outputs read by a CPU task after the executor waits on its event |
//! | the staging, the cells, the Cartesian tensor and the work, the call's own buffers | pinned rows and the stream task's device scratch, all in the plan's arena |
//! | `three_center_class` per class, `gemm` and `gemmStridedBatched` (`recipes/blas`, wwr) | [`fb_integrals_sys::add_three_center_tasks`], over `fb_core_wwr::blas`'s `gemm` and `gemm_strided_batched` |
//! | `three_center_symmetrize`, out of place into `b` | the same, into the output row |
//! | `Unsupported` for an orbital shell above f or an aux shell above i, or a tensor past `INT32_MAX` | the same, at planning, adding nothing |
//! | `Unsupported` past `max_bytes` or the device's memory budget, naming the bytes | admission's `Allocation` over the plan's table, naming the rows at the peak |
//! | `InvalidConfig` for a class that is not the basis's | a contract that `classes` are the basis's records, and `InvalidConfig` at run time if the regroup finds another count of shell pairs |
//! | success with nothing written for an empty basis | a contract: both bases have a shell |
//! | `three_center_footprint(basis, aux)` | nothing: the layout declares every row, and the arena's size is the plan's golden |

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Device, RowId};
use fb_core_error_handling::{Error, Result};
use fb_core_planner::Dag;
use fb_integrals_sys::{
    AuxTable, ClassRows, FB_THREE_CENTER_AUX_MAX_MOMENTUM, FB_THREE_CENTER_MAX_MOMENTUM,
    add_three_center_tasks,
};

/// Add the spherical three-center integrals (mn|P) of the orbital basis
/// `basis` and the auxiliary basis `aux` to `dag` on GPU worker `worker`,
/// and return them: an output row of `naux · nao²` doubles, `nao` and
/// `naux` the two bases' spherical counts, element `(m, n, P)` at
/// `P + naux (n + nao m)`, P fastest (`[m][n][P]`), exactly symmetric in m
/// and n.
///
/// `classes` are the records of `basis`, added on `worker` by
/// [`fb_integrals_mmd::add_records`] earlier in `dag`, so one build serves
/// every reader of the plan. The tasks are `three-center download`
/// (stream), `three-center tables` (CPU) and `three-center` (stream), whose
/// scratch holds the Cartesian tensor (`naux_cart · nao_cart²` doubles) and
/// the transform's intermediate (`naux · nao_cart²`).
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
/// Cartesian tensor does not fit the kernels' 32-bit index. Nothing is
/// added to `dag` on an error.
///
/// # Panics
/// If `classes` are rows of another worker's or another plan's table
/// ([`Dag::add`]).
#[debug_requires(basis.nshells() > 0, "an orbital basis of at least one shell")]
#[debug_requires(aux.nshells() > 0, "an aux basis of at least one shell")]
#[debug_requires(are_records_of(classes, basis), "classes are the records of basis")]
#[track_caller]
pub fn add_three_center(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    aux: &BasisSet,
) -> Result<RowId<Device, f64>> {
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
                "add_three_center: {which} shell {s} has momentum {}, but the three-center \
                 kernels cover {which} shells through l = {max}",
                shell.angular_momentum()
            )));
        }
    }
    let (n, n_aux) = (basis.nao_cart() as usize, aux.nao_cart() as usize);
    if i32::try_from(n * n * n_aux).is_err() {
        return Err(Error::unsupported(format!(
            "add_three_center: the Cartesian tensor, {n} x {n} x {n_aux} elements ({} bytes), is \
             past the kernels' 32-bit index",
            8 * n * n * n_aux
        )));
    }

    let mut table = AuxTable::new(n_aux);
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
    let pairs = shell_pairs(basis);
    let n_pairs: Vec<usize> = classes
        .iter()
        .map(|c| pairs.get(&(c.la(), c.lb())).map_or(0, |&(n, _)| n))
        .collect();
    Ok(add_three_center_tasks(
        dag,
        worker,
        classes,
        &n_pairs,
        &table,
        fb_basis::spherical_transform_matrix(basis),
        fb_basis::spherical_transform_matrix(aux),
    ))
}

/// The shell pairs `i <= j` of `basis` in each class `(la, lb)`, `la <= lb`,
/// with their records, the pairs' primitive products summed.
fn shell_pairs(basis: &BasisSet) -> BTreeMap<(i32, i32), (usize, usize)> {
    let mut pairs = BTreeMap::new();
    let shells = basis.shells();
    for (i, a) in shells.iter().enumerate() {
        for b in &shells[i..] {
            let (la, lb) = (a.angular_momentum(), b.angular_momentum());
            let entry = pairs.entry((la.min(lb), la.max(lb))).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += a.nprimitives() as usize * b.nprimitives() as usize;
        }
    }
    pairs
}

/// Whether `classes` are the records of `basis`: one per class of its
/// shell pairs, each with that class's record count, over its Cartesian
/// AOs.
fn are_records_of(classes: &[ClassRows], basis: &BasisSet) -> bool {
    let pairs = shell_pairs(basis);
    classes.len() == pairs.len()
        && classes.iter().all(|c| {
            c.n_cart() == basis.nao_cart() as usize
                && pairs
                    .get(&(c.la(), c.lb()))
                    .is_some_and(|&(_, n_rec)| n_rec == c.n_rec() as usize)
        })
}
