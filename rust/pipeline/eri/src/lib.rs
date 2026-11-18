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
//! With **`storage = "dense"`** ([`add_dense`]), `"conventional"` stores
//! the tensor unpacked instead, as the two dense matrices
//! `M_J[(m,n), (l,s)] = (mn|ls)` and `M_K[(m,n), (l,s)] = (ml|ns)`, cut
//! by rows over the run's GPUs (`num_gpus`), each card its rows, so the
//! cap is the cards' memory, which admission holds each to: each build's
//! J and K then a broadcast of D, a product per card and a gather.
//!
//! **`"density_fitting"`** ([`add_df_eri`]): the fitted factor
//!
//! ```text
//! M = (P|Q) = L Lᵀ
//! B = L⁻¹ (P|mn)
//! ```
//!
//! of the orbital basis over an auxiliary one, packed by kept shell pair
//! and held in a heap row on the host: each SCF iteration's J and K are
//! one streamed pass over it in aux batches
//! (`fb_integrals_eri_density_fitting_jk::add_streamed_jk`), and none
//! rebuilds it. The C++ `src/pipeline/eri` is the reference design; its
//! README has why the aux basis and L are kept beside B.
//!
//! [`add_df_eri`] does at planning what the C++ step does first: the aux
//! basis resolved by its compiled-in name
//! ([`fb_io_basis::resolve_aux_basis`]) and placed on the molecule as the
//! orbital basis is (`stamp.primitive_normalized().place(&molecule)`).
//! Then, still at planning, the screen: the shell pairs of
//! [`record_tables`], each bounded by the largest element of its host
//! Schwarz diagonal ([`schwarz_diagonal`], the Cartesian `sqrt((mn|mn))`),
//! and the packed map of those not below `[eri] schwarz_threshold`
//! ([`PairMap::new`]). With the Coulomb metric `Σ_Q B_Q,mn² = (mn|mn)_DF
//! <= (mn|mn)`, so a dropped pair's columns of B have 2-norm below the
//! threshold. At threshold 0 no diagonal is formed and every pair `i <= j`
//! is kept: the lower triangle at shell-pair granularity. Then it adds, on
//! one GPU worker:
//!
//! - the metric M ([`add_metric`]: `harmonics` and `upload U` for the aux
//!   basis, `metric tables`, `metric`, `spherical transform`, `metric
//!   average`) and its factor L ([`add_cholesky`]: `cholesky`, `cholesky
//!   check`). A metric that is not positive definite fails the run as
//!   `Numerical` at the check, and nothing that reads B runs;
//! - the build loop ([`add_packed_b`]): B's heap row, then per class of
//!   the map its pairs in batches of the adaptive width `packed B batch`,
//!   each the packed three-center integrals, the fit against L, a
//!   transpose and the download into the heap row, the aux basis's U
//!   staged once for M and every batch;
//! - the map's index tables on the device ([`add_packed_tables`], `packed
//!   tables` and its uploads), once, for every streamed pass over B.
//!
//! **Memory is admission's.** No device row holds B whole: the device holds
//! L (`naux²` doubles), the records and one batch's ring, B
//! (`naux · n_columns` doubles) is a heap row, and every row is in the
//! plan's table before the first launch. The batch's width is the widest
//! that fits the worker's room ([`Dag::plan_within`]), and a room short of
//! one pair's batch is refused at planning by [`fb_core_arena::admit`],
//! naming the bytes over and the rows live at the peak
//! (`docs/memory.md`, "Sizes the dimensions do not fix").
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `EriStep::run`, returning the mode's product | [`add_eri`], returning an [`Eri`] |
//! | `EriStep::run` with `mode = DensityFitting`, `DensityFitted` | [`add_df_eri`], returning [`DensityFitted`] |
//! | nothing: the C++ tree stores no four-index tensor | `mode = "conventional"`, [`EriMode::Conventional`]: [`add_conventional`], [`Eri::Conventional`]; with `storage = "dense"` ([`EriStorage::Dense`]), [`add_dense`], [`Eri::Dense`], over the run's GPUs (Rust only) |
//! | `[eri] mode`, `aux_basis`, `EriStep`'s options read by `src/pipeline/config` | [`EriStep`], [`EriMode`]: the config's `[eri]` table, deserialized by `fb-pipeline-config`, its keys checked against the mode by [`EriStep::check`] |
//! | `EriMode::IntegralDirect`, J's and K's pair classes | not ported |
//! | `default_aux_basis`, `<name>-acCD-6-6` when `aux_basis` is empty | not ported: the aux basis is always named |
//! | `resolve_aux_basis(spec)` over a name or a path, an unknown name `IO` | a compiled-in name alone, an unknown one `InvalidConfig` at planning |
//! | `place(stamp.primitive_normalized(), molecule)` | the same, at planning |
//! | B's and L's bytes against `memory_budget()`, `Unsupported` past it | L and the batches' ring rows of the plan's table, B a heap row, refused by admission as `Allocation` |
//! | `l`, the metric's spherical matrix copied into it, the pair freed, a sync | M an output row read by the Cholesky; no sync |
//! | `three_center` into a square `b` on the device, then `density_fit` in place over `l` | B packed by kept shell pair in a heap row, built in batches ([`add_packed_b`]), screened at planning by `[eri] schwarz_threshold` (Rust only) |
//! | `DensityFitted::naux` | [`DensityFitted::aux`]'s `nao_sphe()` |
//! | `EriStep::footprint` | nothing: the layout declares every row, and the arena's size is the plan's golden |

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Device, Heap, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::Dag;
use fb_integrals_eri_conventional::{Conventional, Dense, add_conventional, add_dense};
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_eri_density_fitting_three_center::add_packed_b;
use fb_integrals_mmd::record_tables;
use fb_integrals_schwarz::schwarz_diagonal;
use fb_integrals_sys::{ClassRows, PackedTables, add_packed_tables};
use fb_molecule::Molecule;
use fb_recipes_cholesky::add_cholesky;
use serde::Deserialize;

/// `[eri]`: the config's table of the two-electron integrals, built once
/// for every reader of the plan ([`add_eri`]). Each field is a key, and
/// any other key is refused by name.
///
/// ```text
/// [eri]
/// mode = "density_fitting"
/// aux_basis = "cc-pvdz-jkfit"
/// schwarz_threshold = 1e-12   # the default; 0 screens nothing
/// ```
///
/// or, with no fitting set,
///
/// ```text
/// [eri]
/// mode = "conventional"
/// storage = "dense"   # or "packed", the default
/// ```
#[derive(Clone, Debug, PartialEq, Deserialize)]
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
    /// `schwarz_threshold`: the density-fitting screen, in the atomic
    /// units of `sqrt((mn|mn))`. A shell pair whose every Cartesian
    /// `sqrt((mn|mn))` is below it is dropped from B whole
    /// ([`add_df_eri`]); 0 drops none. Optional under
    /// `"density_fitting"`, **1e-12** when absent, a non-negative number;
    /// refused by `"conventional"`, which screens nothing
    /// ([`EriStep::check`]).
    ///
    /// What the bound means: a dropped column of B has 2-norm below the
    /// threshold. The fit is the projection of `|mn)` onto the fitting
    /// space in the Coulomb metric, so `Σ_Q B_Q,mn² = (mn|mn)_DF <=
    /// (mn|mn)` (`fb_integrals_schwarz`'s crate doc), and a dropped
    /// column's share of J or K is at most the threshold times the norm
    /// of the vector it is dotted with. The default is pinned by the
    /// largest density-fitted case of the tree, benzene in cc-pVDZ /
    /// cc-pVDZ-JKFIT: its DF-RHF energy at the default and at 0 agree
    /// within 1e-10 hartree, the bar of the PySCF comparison
    /// (`fb-pipeline-config`'s `tests/schwarz_default.rs`).
    #[serde(default)]
    pub schwarz_threshold: Option<f64>,
    /// `storage`: how `"conventional"` stores the tensor
    /// ([`EriStorage`]): `"packed"` by AO pair on GPU worker 0, the
    /// default, or `"dense"`, unpacked and unsymmetrized, its rows spread
    /// over the run's GPUs (`num_gpus`). Refused by `"density_fitting"`,
    /// which stores no four-index tensor ([`EriStep::check`]).
    #[serde(default)]
    pub storage: Option<EriStorage>,
}

/// `[eri] storage`: how `mode = "conventional"` stores the four-index
/// tensor. Both are exact and give the same J and K to rounding; they
/// differ in what is stored where.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EriStorage {
    /// `"packed"`: `ERI_pp` and its exchange-ordered gather, `n_pp²`
    /// doubles each, `n_pp = n_cart (n_cart + 1) / 2`, on GPU worker 0, at
    /// most 150 Cartesian AOs ([`add_conventional`]). The default.
    Packed,
    /// `"dense"`: `M_J` and `M_K`, `n_cart⁴` doubles each, with no
    /// symmetry, cut by rows over GPU workers `0..num_gpus` at shell
    /// boundaries, each worker its rows, admission sizing each card
    /// ([`add_dense`]); J and K by a broadcast of D, a product per worker
    /// and a gather of the pieces. No AO cap beyond the cards' memory.
    Dense,
}

impl EriStep {
    /// The density-fitted step over the fitting set `aux_basis`, screened
    /// at `schwarz_threshold`: the table `mode = "density_fitting"`,
    /// `aux_basis` and `schwarz_threshold` read into.
    pub fn density_fitting(aux_basis: &str, schwarz_threshold: f64) -> EriStep {
        EriStep {
            mode: EriMode::DensityFitting,
            aux_basis: Some(aux_basis.to_owned()),
            schwarz_threshold: Some(schwarz_threshold),
            storage: None,
        }
    }

    /// Check the table's values against its mode, so a reader of a config
    /// can refuse them before anything is planned; [`add_df_eri`] checks
    /// the name too.
    ///
    /// # Errors
    /// `InvalidConfig`, naming the key: under `"density_fitting"`, no
    /// `aux_basis` or one that is not a compiled-in fitting set, a
    /// `schwarz_threshold` that is negative or not a number, or a
    /// `storage` at all; under `"conventional"`, an `aux_basis` or a
    /// `schwarz_threshold` at all.
    pub fn check(&self) -> Status {
        if let (EriMode::DensityFitting, Some(storage)) = (self.mode, self.storage) {
            return Err(Error::config(format!(
                "[eri] storage: {storage:?} given, but mode = \"density_fitting\" stores no \
                 four-index tensor"
            )));
        }
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
        }?;
        match (self.mode, self.schwarz_threshold) {
            (EriMode::DensityFitting, Some(t)) if t.is_nan() || t < 0.0 => Err(Error::config(
                format!("[eri] schwarz_threshold: {t} is not a non-negative number"),
            )),
            (EriMode::Conventional, Some(t)) => Err(Error::config(format!(
                "[eri] schwarz_threshold: {t} given, but mode = \"conventional\" screens \
                     nothing"
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
    /// [`EriStep::aux_basis`], with M = (P|Q) = L Lᵀ, packed by the shell
    /// pairs [`EriStep::schwarz_threshold`] keeps, on the host
    /// ([`add_df_eri`]).
    DensityFitting,
    /// `"conventional"`: the packed four-index tensor and its
    /// exchange-ordered gather, stored once ([`add_conventional`]); exact,
    /// for a basis of at most 150 Cartesian AOs with shells through d.
    /// Rust only: the C++ tree has no such mode.
    Conventional,
}

/// The two-electron integrals of a plan, [`add_eri`]'s product: what the
/// SCF's J and K read, by mode.
#[allow(
    clippy::large_enum_variant,
    reason = "one per plan, built at planning and read by reference"
)]
pub enum Eri {
    /// `"density_fitting"`'s packed B, L and map ([`add_df_eri`]).
    DensityFitted(DensityFitted),
    /// `"conventional"`'s packed tensors ([`add_conventional`]).
    Conventional(Conventional),
    /// `"conventional"`'s dense row blocks, `storage = "dense"`
    /// ([`add_dense`]).
    Dense(Dense),
}

/// Add the two-electron integrals `step` asks for, of `basis` placed over
/// `molecule`, to `dag`, and return them: by `step.mode`, [`add_df_eri`]
/// over `step.aux_basis` at `step.schwarz_threshold` (1e-12 when absent)
/// or [`add_conventional`] on GPU worker `worker`,
/// or, for `storage = "dense"`, [`add_dense`] over GPU workers
/// `0..gpus`, U on worker 0, which `worker` must then be.
///
/// `classes` and `u` are as [`add_df_eri`] takes them, the records of
/// `basis` and the plan's slot for its U; the dense storage forms records
/// of its own on each worker.
///
/// # Errors
/// [`EriStep::check`]'s refusals, at planning, before anything is added;
/// then the mode's: [`add_df_eri`]'s, or [`add_conventional`]'s or
/// [`add_dense`]'s, led by `add_eri: ` (a kernel library that is not the
/// one the bindings describe, `GpuRuntime`; a shell above d, a packed
/// basis past 150 Cartesian AOs, or a dense one whose `n_cart²` passes
/// the kernels' index, `Unsupported`).
///
/// # Panics
/// As [`add_df_eri`].
#[allow(
    clippy::too_many_arguments,
    reason = "the step's inputs, as the C++ step's"
)]
#[debug_requires(
    step.storage != Some(EriStorage::Dense) || worker == 0,
    "the dense storage's J and K are worker 0's"
)]
#[debug_requires(gpus > 0, "at least one GPU worker")]
#[track_caller]
pub fn add_eri(
    dag: &mut Dag<'_>,
    worker: usize,
    gpus: usize,
    molecule: &Molecule,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Device, f64>>,
    step: &EriStep,
) -> Result<Eri> {
    step.check()?;
    let led = |mut e: fb_core_error_handling::Error| {
        e.message = format!("add_eri: {}", e.message);
        e
    };
    match (step.mode, &step.aux_basis, step.storage) {
        (EriMode::DensityFitting, Some(aux), _) => {
            // The key's documented default.
            let threshold = step.schwarz_threshold.unwrap_or(1e-12);
            add_df_eri(dag, worker, molecule, basis, classes, u, aux, threshold)
                .map(Eri::DensityFitted)
        }
        (_, _, Some(EriStorage::Dense)) => {
            add_dense(dag, gpus, basis, u).map(Eri::Dense).map_err(led)
        }
        _ => add_conventional(dag, worker, basis, classes, u)
            .map(Eri::Conventional)
            .map_err(led),
    }
}

/// The density-fitted integrals of a plan, [`add_df_eri`]'s product.
pub struct DensityFitted {
    /// B = L⁻¹ (P|mn), packed by [`DensityFitted::map`]'s kept pairs: a
    /// heap row of `naux · n_columns` doubles, `n_columns` the map's
    /// spherical columns, element `(column, Q)` at `column + n_columns Q`,
    /// pair-fastest (the pair map's crate doc): what the streamed J and K
    /// read, an aux batch at a time.
    pub b: RowId<Heap, f64>,
    /// L, with M = L Lᵀ: an output row of `naux²` doubles, column-major,
    /// L in its lower triangle.
    pub l: RowId<Device, f64>,
    /// The aux basis, placed over the molecule; `naux` is its
    /// `nao_sphe()`.
    pub aux: BasisSet,
    /// The packed shell-pair map B was built over: the pairs the screen
    /// kept, and their columns.
    pub map: PairMap,
    /// The map's spherical index tables on the worker, uploaded once for
    /// every streamed pass over B ([`add_packed_tables`]).
    pub tables: PackedTables,
}

/// Add the density-fitted integrals of `basis` over the fitting set named
/// `aux_basis`, screened at `schwarz_threshold`, to `dag` on GPU worker
/// `worker`, and return them: the packed B on the host, L, the aux basis
/// placed over `molecule`, the map and its tables.
///
/// `aux_basis` is a compiled-in fitting set's name, in any spelling
/// [`fb_io_basis::is_builtin_aux_basis`] folds (`"cc-pvdz-jkfit"`); it is
/// resolved and placed over `molecule`, the atoms `basis` is placed on,
/// here at planning. So is the screen: each shell pair of
/// [`record_tables`] bounded by its largest Cartesian `sqrt((mn|mn))`
/// ([`schwarz_diagonal`]) and kept when that is not below
/// `schwarz_threshold` ([`PairMap::new`]); at 0 no diagonal is formed and
/// every pair kept. `classes` are the records of `basis`, added on
/// `worker` by `fb_integrals_mmd::add_records` earlier in `dag`. `u` is
/// the plan's slot for the orbital basis's U, the one its one-electron
/// matrices fill (`fb_recipes_matrix_ao::add_harmonics`): the build reads
/// the row it holds, and when it is `None` `harmonics` and `upload U` are
/// added here and U's device row left in it. The tasks are the crate
/// doc's: M, L, the build loop, then the tables, the aux basis's U staged
/// once. The step declares no BLAS workspace; the plan's caller does
/// ([`Dag::blas_workspace`]).
///
/// # Errors
/// At planning, each led by `add_df_eri: `: `InvalidConfig` for a name
/// that is not a compiled-in fitting set, or a set without an element of
/// `molecule` (naming its Z); `Unsupported` when the screen keeps no
/// pair; and the refusals of [`add_metric`], [`add_cholesky`] and
/// [`add_packed_b`]: a kernel library that is not the one the bindings
/// describe (`GpuRuntime`), an orbital shell above f or an aux shell above
/// i, or a pair's block past the kernels' 32-bit index (`Unsupported`).
/// `dag` may hold some of the step's tasks after an error, and is dropped
/// with it. At run time, the tasks' errors, a metric that is not positive
/// definite (`Numerical`) among them.
///
/// # Panics
/// If `classes` are rows of another worker's or another plan's table
/// ([`Dag::add_gpu`]).
#[allow(
    clippy::too_many_arguments,
    reason = "the step's inputs, as the C++ step's, with the screen"
)]
#[debug_requires(schwarz_threshold >= 0.0, "the threshold is not negative")]
#[track_caller]
pub fn add_df_eri(
    dag: &mut Dag<'_>,
    worker: usize,
    molecule: &Molecule,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Device, f64>>,
    aux_basis: &str,
    schwarz_threshold: f64,
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
    let tables = record_tables(basis);
    let bounds = (schwarz_threshold > 0.0).then(|| {
        let pairs: Vec<(u32, u32)> = tables
            .classes()
            .iter()
            .flat_map(|c| c.bra.iter().copied().zip(c.ket.iter().copied()))
            .collect();
        let diagonal = schwarz_diagonal(basis, &pairs);
        let shells = basis.shells();
        let mut at = 0;
        pairs
            .iter()
            .map(|&(a, b)| {
                let n = (shells[a as usize].function_count_cart()
                    * shells[b as usize].function_count_cart()) as usize;
                let block = &diagonal[at..at + n];
                at += n;
                block.iter().copied().fold(0.0, f64::max)
            })
            .collect::<Vec<f64>>()
    });
    let map = PairMap::new(&tables, bounds.as_deref(), schwarz_threshold);
    if map.pairs().is_empty() {
        return Err(Error::unsupported(format!(
            "add_df_eri: schwarz_threshold {schwarz_threshold} drops every shell pair"
        )));
    }
    let mut u_aux = None;
    let metric = add_metric(dag, worker, &aux, &mut u_aux)?;
    let naux = aux.nao_sphe() as usize;
    let factor = add_cholesky(dag, worker, "L", naux, metric)?;
    let b = add_packed_b(
        dag, worker, basis, classes, &aux, &map, factor, u, &mut u_aux,
    )?;
    let packing = map.sphe().clone();
    let packed = add_packed_tables(
        dag,
        worker,
        packing.nao(),
        packing.n_columns(),
        move |r, c, w| packing.write_tables(r, c, w),
    );
    Ok(DensityFitted {
        b,
        l: factor.0,
        aux,
        map,
        tables: packed,
    })
}
