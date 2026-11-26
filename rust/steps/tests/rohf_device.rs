//! ROHF in the SCF step on a card (tier 4): the water cation's doublet and
//! triplet CH₂, cc-pVDZ over cc-pVDZ-JKFIT, from the SAD guess through
//! [`fb_steps::run`], against PySCF's DF-ROHF over the same documents and
//! geometries: `tests/data/df_rohf.json`, the deleted old path's reference
//! (`pipeline/scf/tests/rohf.rs`, #1873) and the C++ tree's before it, copied, its
//! provenance inside, written by `tools/emit/df_reference.py` at
//! `conv_tol = 1e-12`, `conv_tol_grad = 1e-9`. Its geometries are in bohr
//! and written out here as an XYZ file in Angstrom, at CODATA 2018's bohr,
//! the reader's, so the run's coordinates are the reference's to a few
//! ulp.
//!
//! **The bars, by analysis, one run, the old path's.** The energy:
//! `|E - E_ref| <= 1e-9` hartree. The ROHF energy is stationary in the
//! rotations between occupation classes, which the residual measures, so
//! the convergence error is second order in `‖e‖ < 1e-7`; what is left is
//! the fitted integrals' agreement with PySCF's, about 1e-10 hartree here.
//! Every orbital energy: `|ε - ε_ref| <= 1e-6` hartree. An eigenvalue
//! moves at first order in F, whose error is first order in the orbitals':
//! a rotation of about `‖e‖ / 2Δ` between classes, `Δ` the smallest gap
//! across them (above 0.3 hartree in both cases), through a response of F
//! of a few hartree, so a few 1e-7. The published energies are PySCF's
//! class by class (`ScfOutput`'s module doc), each class ascending.
//! Beside them, on the same run: the residual under the tolerance, one
//! J/K pass per iteration (both spins' factors in one), D the spin
//! densities of the published C to `1e-12 max|D|` (two gemms of at most 24
//! terms), and the effective Fock matrix diagonal in the published
//! orbitals to the orbital energies' bar: `diag(Cᵀ F C) = ε`.
//!
//! Every run is NaN-filled, and under the allocation record every test
//! binary of the steps installs (`tests/common`), so an open-shell `run`
//! that reads scratch before writing it, or allocates on the plain heap,
//! fails here. One run each, well under a second on an RTX 3080.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
#![allow(
    clippy::disallowed_macros,
    reason = "a test's reference and host checks, outside any step's run"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use std::path::Path;

use fb_core_budget::{Backing, Caps, Ledger};
use fb_core_device::Stream;
use fb_core_flow::Bus;
use fb_steps::{Config, EriStep, ScfStep, ScfTable, run};

/// The energy bar, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// The orbital energy bar, hartree.
const EPSILON_BAR: f64 = 1e-6;

/// The density's bar, relative to its largest element.
const DENSITY_BAR: f64 = 1e-12;

/// Angstrom per bohr, CODATA 2018: the XYZ reader's.
const ANGSTROM_PER_BOHR: f64 = 0.529_177_210_903;

/// One case of `df_rohf.json`.
struct Case {
    charge: i32,
    multiplicity: i32,
    n_alpha: usize,
    n_beta: usize,
    energy: f64,
    mo_energies: Vec<f64>,
    /// Each atom's symbol and position, bohr.
    atoms: Vec<(String, [f64; 3])>,
}

/// The case `name` of `tests/data/df_rohf.json`.
fn case(name: &str) -> Case {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/df_rohf.json");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let case = json["df_rohf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("{name} is in the reference"));
    let numbers = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let int = |key: &str| case[key].as_i64().unwrap();
    Case {
        charge: int("charge") as i32,
        multiplicity: int("multiplicity") as i32,
        n_alpha: int("n_alpha") as usize,
        n_beta: int("n_beta") as usize,
        energy: case["energy"].as_f64().unwrap(),
        mo_energies: numbers(&case["mo_energies"]),
        atoms: case["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                let r = numbers(&a["coord"]);
                (a["symbol"].as_str().unwrap().to_owned(), [r[0], r[1], r[2]])
            })
            .collect(),
    }
}

/// The chain's config of `case`: its geometry written to `name.xyz` in
/// the target's scratch directory, cc-pVDZ over cc-pVDZ-JKFIT, `[scf]` at
/// its defaults, NaN-filled.
fn config(name: &str, case: &Case) -> Config {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let mut xyz = format!("{}\n{name}, Angstrom\n", case.atoms.len());
    for (symbol, r) in &case.atoms {
        let [x, y, z] = r.map(|v| v * ANGSTROM_PER_BOHR);
        xyz += &format!("{symbol} {x:.17e} {y:.17e} {z:.17e}\n");
    }
    std::fs::write(dir.join(format!("{name}.xyz")), xyz).unwrap();
    let text = format!(
        "[molecule]\nxyz = \"{name}.xyz\"\ncharge = {}\nmultiplicity = {}\n\
         [basis]\nname = \"cc-pvdz\"\n[eri]\naux_basis = \"cc-pvdz-jkfit\"\n",
        case.charge, case.multiplicity
    );
    let mut config = Config::parse(&text, dir).unwrap();
    config.scf = Some(ScfTable::default());
    config.resources.nan_fill = true;
    config
}

/// The case `name` run on the card and held to the bars.
fn converges_to_pyscf(name: &str) {
    let case = case(name);
    let run = run(config(name, &case), Backing::Real).expect("the chain runs");
    let dims = &run.plan.bus().shared().dims;
    assert_eq!(
        (dims.n_alpha, dims.n_beta),
        (case.n_alpha, case.n_beta),
        "{name}: the occupations"
    );
    let (n, na, nb) = (dims.n_ao, case.n_alpha, case.n_beta);
    let scf = run.plan.output(ScfStep);
    assert!(scf.converged(), "{name}: converged");
    let miss = (scf.energy() - case.energy).abs();
    let eps_miss = scf
        .epsilon()
        .iter()
        .zip(&case.mo_energies)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!(
        "{name}: E = {:.12} against {:.12}, |ΔE| = {miss:.1e}, max |Δε| = {eps_miss:.1e}, \
         {} iterations, ‖e‖ = {:.1e}",
        scf.energy(),
        case.energy,
        scf.iterations(),
        scf.residual()
    );
    assert!(miss <= ENERGY_BAR, "{name}: |ΔE| = {miss:e}");
    assert_eq!(scf.epsilon().len(), case.mo_energies.len(), "{name}: n_ao");
    assert!(
        eps_miss <= EPSILON_BAR,
        "{name}: max |Δε| = {eps_miss:e}: {:?} against {:?}",
        scf.epsilon(),
        case.mo_energies
    );
    assert!(scf.residual() < ScfTable::default().commutator_tolerance);
    assert_eq!(
        run.plan.output(EriStep).passes(),
        scf.iterations(),
        "{name}: one J/K pass per iteration, both spins in one"
    );

    // D from the published C's columns, on the host.
    let (c, d) = (scf.c(), scf.d());
    let scale = d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for nu in 0..n {
        for mu in 0..n {
            let product =
                |cols: usize| -> f64 { (0..cols).map(|i| c[i * n + mu] * c[i * n + nu]).sum() };
            worst = worst.max((d[nu * n + mu] - product(na) - product(nb)).abs());
        }
    }
    println!("{name}: max |D - Da - Db| = {worst:.1e} of max |D| = {scale:.2}");
    assert!(
        worst <= DENSITY_BAR * scale,
        "{name}: max |D - Da - Db| = {worst:e}"
    );

    // The effective Fock matrix, brought back, in the published orbitals.
    let fock = scf.fock().expect("an open shell's effective Fock matrix");
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    let ledger = Ledger::new(caps, Backing::Real, vec![0], false);
    let host = ledger.budget("test").pinned::<f64>(0, n * n);
    let f = Stream::new(0).download(host, fock).unwrap().sync().unwrap();
    let diagonal_miss = (0..n)
        .map(|i| {
            let col = &c[i * n..(i + 1) * n];
            let fc: f64 = (0..n)
                .map(|mu| col[mu] * (0..n).map(|nu| f[nu * n + mu] * col[nu]).sum::<f64>())
                .sum();
            (fc - scf.epsilon()[i]).abs()
        })
        .fold(0.0_f64, f64::max);
    println!("{name}: max |(Cᵀ F C)_ii - ε_i| = {diagonal_miss:.1e}");
    assert!(
        diagonal_miss <= EPSILON_BAR,
        "{name}: the effective Fock matrix's diagonal: {diagonal_miss:e}"
    );
}

/// H₂O⁺, a doublet: one open orbital, so its rotation is a sign; charged,
/// so the SAD guess carries a tenth electron the first build does not
/// rescale away, and the loop converges from it all the same. The
/// representative of the open shell.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_water_cations_doublet_matches_pyscfs() {
    converges_to_pyscf("h2o_cation_ccpvdz");
}

/// CH₂, a triplet, off every axis: two open orbitals, so the open block is
/// 2 × 2 and its eigensolve a real rotation, and `n_alpha - n_beta = 2` in
/// every factor and density.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_triplet_methylenes_energy_matches_pyscfs() {
    converges_to_pyscf("ch2_ccpvdz");
}

/// The water cation on two logical GPUs of one card (both spins' factors
/// downloaded from gpu0 and uploaded to gpu1, each density's fragments
/// reduced onto gpu0), then replayed: each at the energy bar against the
/// one-GPU run, the replay from the carried orbitals in fewer iterations,
/// one J/K pass per iteration, and allocating nothing.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_water_cation_on_two_gpus_and_replayed_holds_its_energy() {
    let name = "h2o_cation_ccpvdz";
    let case = case(name);
    let mut config = config(name, &case);
    config.resources.devices = vec![0, 0];
    let mut run = run(config, Backing::Real).expect("the chain runs");
    let (first, n_first) = {
        let scf = run.plan.output(ScfStep);
        (scf.energy(), scf.iterations())
    };
    let allocations = run.plan.bus().ledger().allocations();
    run.replay().expect("the replay runs");
    let scf = run.plan.output(ScfStep);
    let (again, n_again) = (scf.energy(), scf.iterations());
    println!(
        "{name} on two GPUs: E = {first:.12} in {n_first} iterations, |ΔE| = {:.1e}; replay \
         {n_again}, |E - first| = {:.1e}",
        (first - case.energy).abs(),
        (again - first).abs()
    );
    assert!(scf.converged());
    assert!((first - case.energy).abs() <= ENERGY_BAR);
    assert!((again - first).abs() <= ENERGY_BAR);
    assert!(
        n_again < n_first,
        "the replay starts from the converged orbitals"
    );
    assert_eq!(run.plan.output(EriStep).passes(), n_again);
    assert_eq!(run.plan.bus().ledger().allocations(), allocations);
}
