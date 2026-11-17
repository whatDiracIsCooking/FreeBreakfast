//! Planning: a [`Config`]'s calculation as its tasks and write nodes.

use fb_core_arena::{Device, Heap, Pinned, RowId};
use fb_core_error_handling::{Error, ErrorKind, Result};
use fb_core_planner::{Dag, Plan};
use fb_integrals_mmd::add_records;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_localization_pipek_mezey::PmOptions;
use fb_molecule::Molecule;
use fb_molecule_nuclei::add_nuclei;
use fb_pipeline_eri::add_eri;
use fb_pipeline_guess::{add_guess_upload, add_loaded_guess, add_sad_guess};
use fb_pipeline_scf::{add_scf, occupied_orbitals};
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};

use crate::localize::add_localization;
use crate::{Calculation, Config, add_write_host_npy, add_write_npy, read, write};

/// Plan the run `config` asks for, on GPU worker 0: the geometry read, the
/// basis resolved, normalized and placed on it, then the calculation's
/// tasks and its write nodes, each product written into
/// [`Config::output`]. `lwork` maps the number of spherical AOs `n` to
/// syevd's work at `n` on the device the plan will run on
/// ([`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) on a solver handle
/// there); it is asked once, by [`Calculation::Rhf`] and
/// [`Calculation::Localization`] alone, after every
/// refusal planning can make of the config. Touches no device unless
/// `lwork` does; the output directory need not exist until the plan runs.
///
/// For [`Calculation::Overlap`]: the records ([`add_records`]) and
/// [`add_overlap`] over them, then [`add_write_npy`] of S to
/// `overlap.npy`. For [`Calculation::Guess`]: [`add_sad_guess`], or
/// [`add_loaded_guess`] when [`GuessStep::from`] is set, then
/// [`add_write_host_npy`] of D to `d.npy`. For [`Calculation::Rhf`]: the
/// nuclei ([`add_nuclei`]), the records, S, H ([`add_core_hamiltonian`])
/// and the two-electron integrals ([`add_eri`]: fitted, or the
/// conventional tensors, by `[eri] mode`, the dense storage's row blocks
/// over GPU workers `0..num_gpus`) over one staging of the
/// orbital basis's spherical transform, that guess uploaded ([`add_guess_upload`]),
/// `X = S^(-1/2)` ([`add_orthogonalize`], at the default floor) and the
/// loop ([`add_scf`]); then [`add_write_npy`] of C to `c.npy` and of D to
/// `d.npy`, and `download epsilon` (stream: ε into a pinned row) and
/// `write scf.json` (CPU: ε beside the last build's energy, residual and
/// count). For [`Calculation::Localization`]: all of that, then
/// [`add_pm_localize`] over S of each occupied space of C, the `n_beta`
/// doubly occupied columns and the `n_alpha - n_beta` singly occupied,
/// with [`Config::localization`]'s options, each space sorted by `<i|F|i>`
/// (the C++ step's; `localize.rs`'s `add_localization` has the tasks),
/// [`add_write_npy`] of `C_local` to `c_local.npy`, `n × n_alpha`, and
/// `write localization.json` (CPU: the localization's heap row
/// `[L before, L after, sweeps]`, summed over the spaces).
///
/// # Errors
/// The readers' refusals, the message led by the key that named the file
/// (`[molecule] xyz: `, `[basis_set] basis: `, `[guess] from: `,
/// `[eri] aux_basis: `): an XYZ file that cannot be read or parsed (`Io`),
/// a basis that is neither a compiled-in name nor a readable document
/// (`Io`), an element the basis or the fitting set lacks; a loaded guess's
/// file that cannot be opened or whose header cannot be parsed (`Io`), or
/// whose shape is not the basis's (`InvalidConfig`), checked from its
/// header alone ([`add_loaded_guess`]); for [`Calculation::Rhf`] and
/// [`Calculation::Localization`], no
/// `[eri]`, an electron count of the wrong parity for the multiplicity
/// ([`occupied_orbitals`]) and an option out of range
/// ([`ScfStep::check`]), each `InvalidConfig`; for
/// [`Calculation::Localization`], [`LocalizationStep::check`]'s refusals;
/// `lwork`'s error;
/// [`EriStep::check`]'s refusals, an open shell or a basis past the
/// kernels over the conventional integrals (`Unsupported`, [`add_eri`],
/// [`add_scf`]),
/// and the kernel libraries' load checks ([`add_records`], [`add_eri`],
/// [`add_scf`], [`add_pm_localize`]). The guess's data is read when the
/// plan runs.
///
/// [`EriStep::check`]: fb_pipeline_eri::EriStep::check
/// [`GuessStep::from`]: fb_pipeline_guess::GuessStep::from
/// [`ScfStep::check`]: fb_pipeline_scf::ScfStep::check
/// [`LocalizationStep::check`]: fb_localization_pipek_mezey::LocalizationStep::check
/// [`add_pm_localize`]: fb_localization_pipek_mezey::add_pm_localize
pub fn plan(config: &Config, lwork: impl FnOnce(usize) -> Result<usize>) -> Result<Plan<'static>> {
    let step = &config.molecule;
    let molecule = fb_io_xyz::read_xyz(&step.xyz, step.charge, step.multiplicity)
        .map_err(|e| keyed(e, "[molecule] xyz"))?;
    let basis = fb_io_basis::resolve_basis(&config.basis_set.basis)
        .and_then(|stamp| stamp.primitive_normalized().place(&molecule))
        .map_err(|e| keyed(e, "[basis_set] basis"))?;
    let n = basis.nao_sphe() as usize;
    let mut dag = Dag::new();
    match config.calculation {
        Calculation::Overlap => {
            dag.blas_workspace(0);
            let classes = add_records(&mut dag, 0, &basis)?;
            let s = add_overlap(&mut dag, 0, &basis, &classes, &mut None);
            add_write_npy(&mut dag, 0, s, [n, n], config.output.join("overlap.npy"));
        }
        Calculation::Guess => {
            let d = add_guess(&mut dag, config, &basis, &molecule)?;
            add_write_host_npy(&mut dag, 0, d, [n, n], config.output.join("d.npy"));
        }
        Calculation::Rhf | Calculation::Localization => {
            let localization = (config.calculation == Calculation::Localization)
                .then(|| config.localization.unwrap_or_default());
            // Refused before lwork can touch a device, as the RHF's options.
            if let Some(step) = &localization {
                step.check()?;
            }
            let rhf = plan_rhf(&mut dag, config, &molecule, &basis, lwork)?;
            add_write_npy(&mut dag, 0, rhf.c, [n, n], config.output.join("c.npy"));
            add_write_npy(&mut dag, 0, rhf.d, [n, n], config.output.join("d.npy"));
            write::add_write_scf_json(
                &mut dag,
                0,
                rhf.scalars,
                rhf.eps,
                n,
                rhf.occupied,
                config.output.join("scf.json"),
            );
            if let Some(step) = localization {
                let options = PmOptions {
                    max_sweeps: step.max_sweeps,
                    functional_tolerance: step.functional_tolerance,
                    rotation_gain_threshold: step.rotation_gain_threshold,
                    exponent: step.exponent,
                };
                let n_occ = rhf.occupied[0];
                let (info, c_local) = add_localization(
                    &mut dag,
                    0,
                    &options,
                    &basis,
                    rhf.occupied,
                    [rhf.s, rhf.c, rhf.eps],
                )?;
                add_write_npy(
                    &mut dag,
                    0,
                    c_local,
                    [n, n_occ],
                    config.output.join("c_local.npy"),
                );
                write::add_write_localization_json(
                    &mut dag,
                    0,
                    info,
                    config.output.join("localization.json"),
                );
            }
        }
    }
    Ok(dag.plan())
}

/// What [`plan_rhf`] leaves for the calculations that read an RHF: device
/// rows of the plan's GPU worker 0.
struct Rhf {
    /// S, `n²`.
    s: RowId<Device, f64>,
    /// The orbitals C, `n²`, column `j` that of `ε_j`, ascending.
    c: RowId<Device, f64>,
    /// The spin-summed density, `D = 2 C_occ C_occᵀ` or `Da + Db`, `n²`.
    d: RowId<Device, f64>,
    /// ε, `n`.
    eps: RowId<Device, f64>,
    /// [`add_scf`]'s heap row `[E, ‖e‖, builds]`.
    scalars: RowId<Heap, f64>,
    /// `[n_alpha, n_beta]` ([`occupied_orbitals`]).
    occupied: [usize; 2],
}

/// Add the RHF `config` asks for to `dag` on GPU worker 0, its writes
/// left to the caller: the nuclei, the records, S, H and the two-electron
/// integrals over one staging of the orbital basis's spherical transform,
/// the guess uploaded, `X = S^(-1/2)` and the loop, as [`plan`]'s doc has
/// them; `lwork` asked once, after every refusal.
///
/// # Errors
/// As [`plan`]'s, for [`Calculation::Rhf`].
fn plan_rhf(
    dag: &mut Dag<'static>,
    config: &Config,
    molecule: &Molecule,
    basis: &fb_basis::BasisSet,
    lwork: impl FnOnce(usize) -> Result<usize>,
) -> Result<Rhf> {
    // read_config refuses this config by the same table; a Config built in
    // code reaches it here.
    let Some(eri) = &config.eri else {
        return Err(read::needs(config.calculation, "eri"));
    };
    let n = basis.nao_sphe() as usize;
    let step = config.scf.clone().unwrap_or_default();
    // Refused here, before lwork can touch a device; add_scf checks both
    // again.
    step.check()?;
    let (n_alpha, n_beta) = occupied_orbitals(molecule)?;
    dag.blas_workspace(0);
    let nuclei = add_nuclei(dag, 0, molecule);
    let mut u = None;
    let classes = add_records(dag, 0, basis)?;
    let s = add_overlap(dag, 0, basis, &classes, &mut u);
    let h = add_core_hamiltonian(dag, 0, basis, &classes, nuclei, &mut u)?;
    let gpus = config.num_gpus.get();
    let eri = add_eri(dag, 0, gpus, molecule, basis, &classes, &mut u, eri).map_err(|e| {
        match e.kind {
            // check's own carry their key; add_df_eri's name the set.
            ErrorKind::InvalidConfig if !e.message.starts_with("[eri]") => {
                keyed(e, "[eri] aux_basis")
            }
            _ => e,
        }
    })?;
    let guess = add_guess(dag, config, basis, molecule)?;
    let guess = add_guess_upload(dag, 0, basis, guess);
    let lwork = lwork(n)?;
    let x = add_orthogonalize(dag, 0, "X", n, lwork, s, ORTHOGONALIZE_EIGENVALUE_FLOOR)?;
    let (scalars, [eps, c, d]) =
        add_scf(dag, 0, &step, molecule, n, lwork, &eri, [h, s, x], guess)?;
    Ok(Rhf {
        s,
        c,
        d,
        eps,
        scalars,
        occupied: [n_alpha, n_beta],
    })
}

/// The guess [`Config::guess`] asks for, on GPU worker 0's host table:
/// [`add_loaded_guess`] when it sets `from`, else [`add_sad_guess`].
fn add_guess(
    dag: &mut Dag<'static>,
    config: &Config,
    basis: &fb_basis::BasisSet,
    molecule: &Molecule,
) -> Result<RowId<Pinned, f64>> {
    match config.guess.as_ref().and_then(|g| g.from.clone()) {
        Some(from) => add_loaded_guess(dag, 0, basis, from).map_err(|e| keyed(e, "[guess] from")),
        None => add_sad_guess(dag, 0, basis, molecule.atoms()),
    }
}

/// `error` with its message led by `key`, its origin kept.
fn keyed(mut error: Error, key: &str) -> Error {
    error.message = format!("{key}: {}", error.message);
    error
}
