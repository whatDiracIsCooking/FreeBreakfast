//! A restart on a card (tier 4): water's DF-RHF in cc-pVDZ over
//! cc-pVDZ-JKFIT run and written, then run again with `[guess] from` at
//! the `d.npy` it wrote.
//!
//! The loaded D is a converged D, so the restart's first build is the
//! converged energy to the solver's own tolerance: the reference's energy
//! is that of the density its last F was built from, at a commutator under
//! 1e-7, whose square bounds its distance from the stationary point (a few
//! 1e-11, `chain_device.rs`), and `d.npy` is the next Roothaan step's
//! density, nearer still. So the first build's energy is within the
//! config's `energy_tolerance`, 1e-10, of the reference's; the guess's
//! factor keeps exactly the five occupied directions of a rank-five D; and
//! the restart, run to convergence, lands within the chain's 1e-9 of it.
//! Each run under the NaN fill and the allocation record (`common`).
//!
//! Three runs of water's whole chain; this is its family's
//! representative, the one case of a loaded guess, so it is no slow case
//! (`docs/testing.md`, "Slow cases").

mod common;

use std::path::Path;

use fb_core_budget::Backing;
use fb_core_error_handling::ErrorKind;
use fb_core_flow::{Bus, Executor};
use fb_steps::{Config, MainBus, ScfStep, plan, run};

use common::ENERGY_BAR;

/// Water's chain under the NaN fill, with `extra` appended to the config.
fn config(extra: &str) -> Config {
    let text = format!(
        "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
         [eri]\naux_basis = \"cc-pvdz-jkfit\"\n[resources]\nnan_fill = true\n{extra}"
    );
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    Config::parse(&text, &dir).unwrap()
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_restart_from_a_written_density_starts_converged() {
    let reference = run(config("[scf]\n"), Backing::Real).expect("the chain runs");
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("guess_device");
    reference.write(&dir).unwrap();
    let (e_ref, tolerance) = {
        let scf = reference.plan.output(ScfStep);
        assert!(scf.converged());
        let tolerance = reference.plan.bus().shared().config.scf.as_ref();
        (scf.energy(), tolerance.unwrap().energy_tolerance)
    };
    let from = format!("[guess]\nfrom = \"{}\"\n", dir.join("d.npy").display());

    // One build from the loaded D: its energy, then the loop's refusal.
    let first = config(&format!("{from}[scf]\nmax_iterations = 1\n"));
    let threads = first.resources.threads;
    let one = plan(MainBus::setup(first, Backing::Real).unwrap()).unwrap();
    let executor = Executor::new(threads);
    one.budget(&executor).unwrap();
    let error = one.execute(&executor).unwrap_err();
    assert_eq!(
        error.kind,
        ErrorKind::MaxIterationsExceeded,
        "{}",
        error.message
    );
    let scf = one.output(ScfStep);
    let n_occ = one.bus().shared().dims.n_beta;
    assert_eq!(scf.iterations(), 1);
    assert_eq!(scf.guess_rank(), n_occ, "the loaded D's factor");
    let e_first = scf.energy();
    println!(
        "water: E = {e_ref:.12}, the restart's first build {e_first:.12} \
         ({:+.1e} against {tolerance:.0e})",
        e_first - e_ref
    );
    assert!((e_first - e_ref).abs() < tolerance);

    // And to convergence.
    let restart = run(config(&format!("{from}[scf]\n")), Backing::Real).expect("the restart");
    let scf = restart.plan.output(ScfStep);
    assert!(scf.converged());
    println!(
        "water: the restart converged in {} builds to {:.12}",
        scf.iterations(),
        scf.energy()
    );
    assert!((scf.energy() - e_ref).abs() < ENERGY_BAR);
}
