//! `src/pipeline/eri` (module `free_breakfast.pipeline.eri`), its
//! density-fitted branch, and a conventional one the C++ tree does not
//! have: the pipeline's two-electron integrals, built once as tasks of a
//! [`Dag`] for every reader of the plan, in the mode `[eri] mode` names
//! ([`EriStep`], [`add_eri`], returning an [`Eri`]).
//!
//! **`"conventional"`** ([`add_conventional`]): the whole four-index
//! tensor of a small basis stored once, packed by AO pair, and its
//! exchange-ordered gather, each SCF iteration's J and K then one GEMV
//! over each (`fb_integrals_eri_conventional`'s crate doc has the packing
//! and the Cartesian-to-spherical handling). Exact, with no fitting set;
//! a basis of at most 150 Cartesian AOs and shells through d, refused at
//! planning otherwise. Rust only: it is named for what it is in the
//! literature, the stored-integral SCF that integral-direct is the
//! alternative to, beside the C++ `IntegralDirect` and `DensityFitting`
//! that each name how J and K are formed.
//!
//! **`"density_fitting"`** ([`add_df_eri`]): the fitted factor
//!
//! ```text
//! M = (P|Q) = L Lᵀ
//! B = L⁻¹ (P|mn)
//! ```
//!
//! of the orbital basis over an auxiliary one: each SCF iteration's J and K
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
//!   ([`add_three_center`]: `harmonics` and `upload U` for the aux
//!   basis, and for the
//!   orbital one unless the caller's slot holds its U already,
//!   `three-center tables`, `three-center`), over
//!   the orbital records and U the caller built once;
//! - the metric M ([`add_metric`]: `metric tables`, `metric`, `spherical
//!   transform`, `metric average`), reading the aux basis's U from the
//!   device row the three-center integrals added, so U_aux is staged
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
//! | `EriStep::run`, returning the mode's product | [`add_eri`], returning an [`Eri`] |
//! | `EriStep::run` with `mode = DensityFitting`, `DensityFitted` | [`add_df_eri`], returning [`DensityFitted`] |
//! | nothing: the C++ tree stores no four-index tensor | `mode = "conventional"`, [`EriMode::Conventional`]: [`add_conventional`], [`Eri::Conventional`] (Rust only) |
//! | `[eri] mode`, `aux_basis`, `EriStep`'s options read by `src/pipeline/config` | [`EriStep`], [`EriMode`]: the config's `[eri]` table, deserialized by `fb-pipeline-config`, its keys checked against the mode by [`EriStep::check`] |
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
use fb_core_arena::{Device, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::Dag;
use fb_integrals_eri_conventional::{Conventional, add_conventional};
use fb_integrals_eri_density_fitting_fit::add_density_fit;
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_three_center::add_three_center;
use fb_integrals_sys::ClassRows;
use fb_molecule::Molecule;
use serde::Deserialize;

/// `[eri]`: the config's table of the two-electron integrals, built once
/// for every reader of the plan ([`add_eri`]). Each field is a key, and
/// any other key is refused by name.
///
/// ```text
/// [eri]
/// mode = "density_fitting"
/// aux_basis = "cc-pvdz-jkfit"
/// ```
///
/// or, with no fitting set,
///
/// ```text
/// [eri]
/// mode = "conventional"
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EriStep {
    /// `mode`: how the integrals are built. Required: the C++ default,
    /// `"integral_direct"`, is not ported, so no default stands in for it.
    pub mode: EriMode,
    /// `aux_basis`: the fitting set, a compiled-in name in any spelling
    /// [`fb_io_basis::is_builtin_aux_basis`] folds (`"cc-pvdz-jkfit"`),
    /// placed over the molecule when the run is planned. Never a path.
    /// Required by `"density_fitting"`, with no default (the C++
    /// `<basis>-acCD-6-6` is not ported), and refused by
    /// `"conventional"`, which fits nothing ([`EriStep::check`]).
    pub aux_basis: Option<String>,
}

impl EriStep {
    /// The density-fitted step over the fitting set `aux_basis`: the table
    /// `mode = "density_fitting"` and `aux_basis` read into.
    pub fn density_fitting(aux_basis: &str) -> EriStep {
        EriStep {
            mode: EriMode::DensityFitting,
            aux_basis: Some(aux_basis.to_owned()),
        }
    }

    /// Check the table's values against its mode, so a reader of a config
    /// can refuse them before anything is planned; [`add_df_eri`] checks
    /// the name too.
    ///
    /// # Errors
    /// `InvalidConfig`, naming the key: under `"density_fitting"`, no
    /// `aux_basis` or one that is not a compiled-in fitting set; under
    /// `"conventional"`, an `aux_basis` at all.
    pub fn check(&self) -> Status {
        match (self.mode, &self.aux_basis) {
            (EriMode::DensityFitting, None) => Err(Error::config(
                "[eri] aux_basis: mode = \"density_fitting\" needs a fitting set",
            )),
            (EriMode::DensityFitting, Some(aux)) if !fb_io_basis::is_builtin_aux_basis(aux) => {
                Err(Error::config(format!(
                    "[eri] aux_basis: '{aux}' is not a compiled-in fitting set"
                )))
            }
            (EriMode::Conventional, Some(aux)) => Err(Error::config(format!(
                "[eri] aux_basis: '{aux}' given, but mode = \"conventional\" fits nothing"
            ))),
            _ => Ok(()),
        }
    }
}

/// `[eri] mode`: how the two-electron integrals are built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EriMode {
    /// `"density_fitting"`: the fitted factor B = L⁻¹ (P|mn) over
    /// [`EriStep::aux_basis`], with M = (P|Q) = L Lᵀ ([`add_df_eri`]).
    DensityFitting,
    /// `"conventional"`: the packed four-index tensor and its
    /// exchange-ordered gather, stored once ([`add_conventional`]); exact,
    /// for a basis of at most 150 Cartesian AOs with shells through d.
    /// Rust only: the C++ tree has no such mode.
    Conventional,
}

/// The two-electron integrals of a plan, [`add_eri`]'s product: what the
/// SCF's J and K read, by mode.
pub enum Eri {
    /// `"density_fitting"`'s B and L ([`add_df_eri`]).
    DensityFitted(DensityFitted),
    /// `"conventional"`'s packed tensors ([`add_conventional`]).
    Conventional(Conventional),
}

/// Add the two-electron integrals `step` asks for, of `basis` placed over
/// `molecule`, to `dag` on GPU worker `worker`, and return them: by
/// `step.mode`, [`add_df_eri`] over `step.aux_basis` or
/// [`add_conventional`].
///
/// `classes` and `u` are as [`add_df_eri`] takes them, the records of
/// `basis` and the plan's slot for its U.
///
/// # Errors
/// [`EriStep::check`]'s refusals, at planning, before anything is added;
/// then the mode's: [`add_df_eri`]'s, or [`add_conventional`]'s, led by
/// `add_eri: ` (a kernel library that is not the one the bindings describe,
/// `GpuRuntime`; a shell above d or a basis past 150 Cartesian AOs,
/// `Unsupported`).
///
/// # Panics
/// As [`add_df_eri`].
#[track_caller]
pub fn add_eri(
    dag: &mut Dag<'_>,
    worker: usize,
    molecule: &Molecule,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Device, f64>>,
    step: &EriStep,
) -> Result<Eri> {
    step.check()?;
    match (step.mode, &step.aux_basis) {
        (EriMode::DensityFitting, Some(aux)) => {
            add_df_eri(dag, worker, molecule, basis, classes, u, aux).map(Eri::DensityFitted)
        }
        _ => add_conventional(dag, worker, basis, classes, u)
            .map(Eri::Conventional)
            .map_err(|mut e| {
                e.message = format!("add_eri: {}", e.message);
                e
            }),
    }
}

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
/// row it holds, and when it is `None` `harmonics` and `upload U`
/// are added here and U's device row left in it. The tasks are the crate doc's: (mn|P), then M, then the
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
/// ([`Dag::add_gpu`]).
#[track_caller]
pub fn add_df_eri(
    dag: &mut Dag<'_>,
    worker: usize,
    molecule: &Molecule,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Device, f64>>,
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
