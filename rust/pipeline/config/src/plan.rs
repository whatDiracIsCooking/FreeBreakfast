//! Planning: a [`Config`]'s calculation as its tasks and write nodes.

use fb_core_arena::{Pinned, RowId};
use fb_core_error_handling::{Error, ErrorKind, Result};
use fb_core_planner::{Dag, Planned};
use fb_integrals_mmd::add_records;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_molecule::Molecule;
use fb_molecule_nuclei::add_nuclei;
use fb_pipeline_eri::add_df_eri;
use fb_pipeline_guess::{add_guess_upload, add_loaded_guess, add_sad_guess};
use fb_pipeline_scf::{add_scf, occupied_orbitals};
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};

use crate::{Calculation, Config, add_write_host_npy, add_write_npy, read, write};

/// Plan the run `config` asks for, on GPU worker 0: the geometry read, the
/// basis resolved, normalized and placed on it, then the calculation's
/// tasks and its write nodes, each product written into
/// [`Config::output`]. `lwork` maps the number of spherical AOs `n` to
/// syevd's work at `n` on the device the plan will run on
/// ([`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) on a solver handle
/// there); it is asked once, by [`Calculation::Rhf`] alone, after every
/// refusal planning can make of the config. Touches no device unless
/// `lwork` does; the output directory need not exist until the plan runs.
///
/// For [`Calculation::Overlap`]: the records ([`add_records`]) and
/// [`add_overlap`] over them, then [`add_write_npy`] of S to
/// `overlap.npy`. For [`Calculation::Guess`]: [`add_sad_guess`], or
/// [`add_loaded_guess`] when [`GuessStep::from`] is set, then
/// [`add_write_host_npy`] of D to `d.npy`. For [`Calculation::Rhf`]: the
/// nuclei ([`add_nuclei`]), the records, S, H ([`add_core_hamiltonian`])
/// and the fitted integrals ([`add_df_eri`]) over one staging of the
/// orbital basis's spherical transform, that guess uploaded ([`add_guess_upload`]),
/// `X = S^(-1/2)` ([`add_orthogonalize`], at the default floor) and the
/// loop ([`add_scf`]); then [`add_write_npy`] of C to `c.npy` and of D to
/// `d.npy`, and `download epsilon` (stream: ε into a pinned row) and
/// `write scf.json` (CPU: ε beside the last build's energy, residual and
/// count).
///
/// # Errors
/// The readers' refusals, the message led by the key that named the file
/// (`[molecule] xyz: `, `[basis_set] basis: `, `[guess] from: `,
/// `[eri] aux_basis: `): an XYZ file that cannot be read or parsed (`Io`),
/// a basis that is neither a compiled-in name nor a readable document
/// (`Io`), an element the basis or the fitting set lacks; a loaded guess's
/// file that cannot be opened or whose header cannot be parsed (`Io`), or
/// whose shape is not the basis's (`InvalidConfig`), checked from its
/// header alone ([`add_loaded_guess`]); for [`Calculation::Rhf`], no
/// `[eri]`, an odd electron count ([`occupied_orbitals`]) and an option out
/// of range ([`ScfStep::check`]), each `InvalidConfig`; `lwork`'s error;
/// and the kernel libraries' load checks ([`add_records`], [`add_df_eri`],
/// [`add_scf`]). The guess's data is read when the plan runs.
///
/// [`GuessStep::from`]: fb_pipeline_guess::GuessStep::from
/// [`ScfStep::check`]: fb_pipeline_scf::ScfStep::check
pub fn plan(
    config: &Config,
    lwork: impl FnOnce(usize) -> Result<usize>,
) -> Result<Planned<'static>> {
    // Charge and multiplicity: [molecule] has no key for either, so every
    // molecule is the neutral singlet.
    let molecule =
        fb_io_xyz::read_xyz(&config.molecule.xyz, 0, 1).map_err(|e| keyed(e, "[molecule] xyz"))?;
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
            add_write_npy(&mut dag, 0, s, n, config.output.join("overlap.npy"));
        }
        Calculation::Guess => {
            let d = add_guess(&mut dag, config, &basis, &molecule)?;
            add_write_host_npy(&mut dag, 0, d, n, config.output.join("d.npy"));
        }
        Calculation::Rhf => {
            // read_config refuses this config by the same table; a Config
            // built in code reaches it here.
            let Some(eri) = &config.eri else {
                return Err(read::needs(Calculation::Rhf, "eri"));
            };
            let step = config.scf.clone().unwrap_or_default();
            // Refused here, before lwork can touch a device; add_scf checks
            // both again.
            step.check()?;
            occupied_orbitals(&molecule)?;
            dag.blas_workspace(0);
            let nuclei = add_nuclei(&mut dag, 0, &molecule);
            let mut u = None;
            let classes = add_records(&mut dag, 0, &basis)?;
            let s = add_overlap(&mut dag, 0, &basis, &classes, &mut u);
            let h = add_core_hamiltonian(&mut dag, 0, &basis, &classes, nuclei, &mut u)?;
            let eri = add_df_eri(
                &mut dag,
                0,
                &molecule,
                &basis,
                &classes,
                &mut u,
                &eri.aux_basis,
            )
            .map_err(|e| match e.kind {
                ErrorKind::InvalidConfig => keyed(e, "[eri] aux_basis"),
                _ => e,
            })?;
            let guess = add_guess(&mut dag, config, &basis, &molecule)?;
            let guess = add_guess_upload(&mut dag, 0, &basis, guess);
            let lwork = lwork(n)?;
            let x = add_orthogonalize(
                &mut dag,
                0,
                "X",
                n,
                lwork,
                s,
                ORTHOGONALIZE_EIGENVALUE_FLOOR,
            )?;
            let (scalars, [eps, c, d]) = add_scf(
                &mut dag,
                0,
                &step,
                &molecule,
                n,
                lwork,
                &eri,
                [h, s, x],
                guess,
            )?;
            add_write_npy(&mut dag, 0, c, n, config.output.join("c.npy"));
            add_write_npy(&mut dag, 0, d, n, config.output.join("d.npy"));
            write::add_write_scf_json(&mut dag, 0, scalars, eps, n, config.output.join("scf.json"));
        }
    }
    Ok(dag.plan())
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
