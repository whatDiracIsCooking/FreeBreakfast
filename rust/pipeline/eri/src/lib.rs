//! `src/pipeline/eri` (module `free_breakfast.pipeline.eri`), its
//! density-fitted branch: the pipeline's two-electron integrals as the
//! fitted factor
//!
//! ```text
//! M = (P|Q) = L Lᵀ
//! B = L⁻¹ (P|mn)
//! ```
//!
//! of the orbital basis over an auxiliary one, built once as tasks of a
//! [`Dag`] for every reader of the plan: each SCF iteration's J and K
//! (`fb_integrals_eri_density_fitting_jk`) read B's row, and none rebuilds
//! it. The C++ `src/pipeline/eri` is the reference design; its README has
//! why the aux basis and L are kept beside B.
//!
//! [`add_df_eri`] does at planning what the C++ step does first: the aux
//! basis resolved by its compiled-in name
//! ([`fb_io_basis::resolve_aux_basis`]) and placed on the molecule as the
//! orbital basis is (`stamp.primitive_normalized().place(&molecule)`).
//! Then it adds, on one GPU worker:
//!
//! - the three-center integrals (mn|P)
//!   ([`add_three_center`]: `harmonics` for the aux basis, and for the
//!   orbital one unless the caller's slot holds its U already,
//!   `three-center download`, `three-center tables`, `three-center`), over
//!   the orbital records and U the caller built once;
//! - the metric M ([`add_metric`]: `metric tables`, `metric`, `spherical
//!   transform`, `metric average`), reading the aux basis's U from the
//!   `harmonics` row the three-center integrals added, so U_aux is staged
//!   once;
//! - the fit ([`add_density_fit`]: `cholesky`, `cholesky check`, `density
//!   fit`), B and L its output rows. A metric that is not positive definite
//!   fails the run as `Numerical` at the check, and nothing that reads B
//!   runs.
//!
//! (mn|P) comes first so that its scratch, about three B's, is the plan's
//! peak with nothing of the metric beside it; the metric's rows end at the
//! fit, beside (mn|P) and B.
//!
//! **Memory is admission's.** Every row the step needs, B (`naux · nao²`
//! doubles) and L (`naux²`) among them, is in the plan's table before the
//! first launch, so a pair of sets that does not fit the card is refused at
//! planning by [`fb_core_arena::admit`], naming the bytes over and the rows
//! live at the peak, not by an allocation that fails mid-run
//! (`docs/memory.md`, "Sizes the dimensions do not fix").
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `EriStep::run` with `mode = DensityFitting`, `DensityFitted` | [`add_df_eri`], returning [`DensityFitted`] |
//! | `EriMode::IntegralDirect`, J's and K's pair classes | not ported |
//! | `default_aux_basis`, `<name>-acCD-6-6` when `aux_basis` is empty | not ported: the aux basis is always named |
//! | `resolve_aux_basis(spec)` over a name or a path, an unknown name `IO` | a compiled-in name alone, an unknown one `InvalidConfig` at planning |
//! | `place(stamp.primitive_normalized(), molecule)` | the same, at planning |
//! | B's and L's bytes against `memory_budget()`, `Unsupported` past it | B and L rows of the plan's table, refused by admission as `Allocation` |
//! | `l`, the metric's spherical matrix copied into it, the pair freed, a sync | M an output row read by the fit; no sync |
//! | `three_center` into `b`, then `density_fit` in place over `l` | (mn\|P) its own row, B and L the fit's output rows, the tensor's bytes reused after its last read |
//! | `DensityFitted::naux` | [`DensityFitted::aux`]'s `nao_sphe()` |
//! | `EriStep::footprint` | nothing: the layout declares every row, and the arena's size is the plan's golden |

#![forbid(unsafe_code)]

use fb_basis::BasisSet;
use fb_core_arena::{Device, Pinned, RowId};
use fb_core_error_handling::{Error, Result};
use fb_core_planner::Dag;
use fb_integrals_eri_density_fitting_fit::add_density_fit;
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_three_center::add_three_center;
use fb_integrals_sys::ClassRows;
use fb_molecule::Molecule;

/// The density-fitted integrals of a plan, [`add_df_eri`]'s product.
pub struct DensityFitted {
    /// B = L⁻¹ (P|mn): an output row of `naux · nao²` doubles, column-major
    /// at leading dimension `naux`, element `(Q, n + nao m)` at
    /// `Q + naux (n + nao m)`, Q fastest: what J and K read.
    pub b: RowId<Device, f64>,
    /// L, with M = L Lᵀ: an output row of `naux²` doubles, column-major,
    /// L in its lower triangle and M's strict upper triangle above it.
    pub l: RowId<Device, f64>,
    /// The aux basis, placed over the molecule; `naux` is its
    /// `nao_sphe()`.
    pub aux: BasisSet,
}

/// Add the density-fitted integrals of `basis` over the fitting set named
/// `aux_basis` to `dag` on GPU worker `worker`, and return them: B, L and
/// the aux basis placed over `molecule`.
///
/// `aux_basis` is a compiled-in fitting set's name, in any spelling
/// [`fb_io_basis::is_builtin_aux_basis`] folds (`"cc-pvdz-jkfit"`); it is
/// resolved and placed over `molecule`, the atoms `basis` is placed on,
/// here at planning. `classes` are the records of `basis`, added on
/// `worker` by `fb_integrals_mmd::add_records` earlier in `dag`. `u` is the
/// plan's slot for the orbital basis's U, the one its one-electron
/// matrices fill (`fb_recipes_matrix_ao::add_harmonics`): (mn|P) reads the
/// row it holds, and when it is `None` `harmonics` is added here and its
/// row left in it. The tasks are the crate doc's: (mn|P), then M, then the
/// fit, the aux basis's U staged once for (mn|P) and M. The step declares no
/// BLAS workspace; the plan's caller does
/// ([`Dag::blas_workspace`]).
///
/// # Errors
/// At planning, each led by `add_df_eri: `: `InvalidConfig` for a name
/// that is not a compiled-in fitting set, or a set without an element of
/// `molecule` (naming its Z); and the refusals of [`add_three_center`],
/// [`add_metric`] and [`add_density_fit`]: a kernel library that is not
/// the one the bindings describe (`GpuRuntime`), an orbital shell above f
/// or an aux shell above i, or a basis past the kernels' 32-bit index
/// (`Unsupported`). `dag` may hold some of the step's tasks after an error,
/// and is dropped with it. At run time, the tasks' errors, a metric that
/// is not positive definite (`Numerical`) among them.
///
/// # Panics
/// If `classes` are rows of another worker's or another plan's table
/// ([`Dag::add`]).
#[track_caller]
pub fn add_df_eri(
    dag: &mut Dag<'_>,
    worker: usize,
    molecule: &Molecule,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Pinned, f64>>,
    aux_basis: &str,
) -> Result<DensityFitted> {
    if !fb_io_basis::is_builtin_aux_basis(aux_basis) {
        return Err(Error::config(format!(
            "add_df_eri: '{aux_basis}' is not a compiled-in fitting set"
        )));
    }
    let aux = fb_io_basis::resolve_aux_basis(aux_basis)
        .and_then(|stamp| stamp.primitive_normalized().place(molecule))
        .map_err(|mut e| {
            e.message = format!("add_df_eri: fitting set '{aux_basis}': {}", e.message);
            e
        })?;
    let mut u_aux = None;
    let tensor = add_three_center(dag, worker, basis, classes, &aux, u, &mut u_aux)?;
    let metric = add_metric(dag, worker, &aux, &mut u_aux)?;
    let (nao, naux) = (basis.nao_sphe() as usize, aux.nao_sphe() as usize);
    let (b, l) = add_density_fit(dag, worker, nao, naux, metric, tensor)?;
    Ok(DensityFitted { b, l, aux })
}
