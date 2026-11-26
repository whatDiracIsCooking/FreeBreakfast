//! The default `[eri] schwarz_threshold`, 1e-12, pinned on a card (tier 4,
//! device 0, one run per threshold): benzene's DF-RHF energy in cc-pVDZ /
//! cc-pVDZ-JKFIT at the default against its energy at 0, each run to
//! convergence through the chain.
//!
//! **The bar, one run: `|E(default) - E(0)| <= 1e-10` Eh**, the old
//! path's (the deleted `fb-pipeline-config`'s `tests/schwarz_default.rs`). A dropped
//! pair's columns of B have 2-norm under the threshold (the fit a
//! projection, the pair's bound at least its largest `sqrt((mn|mn))`), so
//! the energy moves by about the threshold times the density's elements,
//! well under the bar at 1e-12; the two runs take the same path, and each
//! one's convergence error is second order in its commutator, under 1e-7.
//! At 0 the energy is also the chain's recorded one within the chain's
//! bar, 1e-9 (`tests/chain_device.rs`).
//!
//! A slow case: two full SCF runs of benzene, which return at once under
//! `FREE_BREAKFAST_FAST_TESTS` (`docs/testing.md`, "Slow cases");
//! `eri_device.rs`'s `the_screen_zeroes_only_columns_under_the_threshold_on_benzene`
//! is its family's representative at the gates.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use fb_core_budget::Backing;
use fb_steps::{EriStep, ScfStep, ScfTable, run};

/// The energy bar between the two thresholds, hartree.
const BAR: f64 = 1e-10;

/// The chain's bar on an energy against its recorded one, hartree.
const CHAIN_BAR: f64 = 1e-9;

/// Benzene's recorded DF-RHF energy (`tests/chain_device.rs`).
const RECORDED: f64 = -230.721_729_270_921;

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0`, as `FBF_SKIP_IF_FAST_TESTS` reads it.
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// Benzene's converged energy at `threshold`, the default when `None`, and
/// the pairs the screen dropped.
fn energy(threshold: Option<f64>) -> (f64, usize) {
    let mut config = common::config("benzene.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    if let Some(t) = threshold {
        config.eri.as_mut().unwrap().schwarz_threshold = t;
    }
    let run = run(config, Backing::Real).expect("the chain runs");
    let scf = run.plan.output(ScfStep);
    assert!(scf.converged());
    (scf.energy(), run.plan.output(EriStep).dropped_pairs())
}

/// Benzene's energy at the default threshold against 0, within [`BAR`],
/// and at 0 against the recorded one, within [`CHAIN_BAR`].
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_default_threshold_keeps_benzenes_energy() {
    if fast_tests() {
        eprintln!("skipped: slow case (FREE_BREAKFAST_FAST_TESTS is set)");
        return;
    }
    let (e_all, none) = energy(Some(0.0));
    let (e_default, dropped) = energy(None);
    let miss = (e_default - e_all).abs();
    println!(
        "benzene: the default drops {dropped} shell pairs; E(default) = {e_default:.12}, \
         E(0) = {e_all:.12}, |ΔE| = {miss:e}, |E(0) - recorded| = {:e}",
        (e_all - RECORDED).abs()
    );
    assert_eq!(none, 0, "threshold 0 drops nothing");
    assert!(miss <= BAR, "|ΔE| = {miss:e} over {BAR:e}");
    assert!(
        (e_all - RECORDED).abs() < CHAIN_BAR,
        "E(0) = {e_all} against {RECORDED}"
    );
}
