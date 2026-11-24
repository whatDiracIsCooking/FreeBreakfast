//! The default `[eri] schwarz_threshold`, 1e-12, pinned (tier 4, device
//! 0, one run per threshold) on two molecules in cc-pVDZ /
//! cc-pVDZ-JKFIT, each planned from an `[eri]` table without the key and
//! from one at 0, each run to convergence from the SAD guess:
//!
//! - benzene (114 AOs, 558 fitting functions, `fb-pipeline-scf`'s
//!   `benzene_converges_to_pyscf`), compact, where the default drops 3 of
//!   1485 shell pairs;
//! - five glycines (`fb-pipeline-scf`'s `tests/data/gly5.xyz`, the dense
//!   conventional path's chain), spread out, where the default drops
//!   5318 of 15753 shell pairs (34%) and B's heap row shrinks from
//!   134,730,817 doubles to 83,635,201: the case that tells 1e-12 from a
//!   smaller threshold.
//!
//! **The bar, one run: `|E(default) - E(0)| <= 1e-10` hartree**, the bar
//! of the integrals' PySCF comparison (`fb-pipeline-eri`'s `tests/jk.rs`),
//! so the default cannot move a result across it. Each run stops at `‖e‖
//! < 1e-7` (`commutator_tolerance`'s default), and the energy is
//! stationary in the orbitals, so each one's convergence error is second
//! order in its residual, about 1e-14 hartree: what is left of the
//! difference is the screen's.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fb_core_planner::Dag;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_integrals_mmd::add_records;
use fb_molecule::Molecule;
use fb_pipeline_eri::{Eri, EriStep, add_eri};
use fb_pipeline_scf::{Run, ScfStep};
use fb_recipes_diagonalize::syevd_lwork;

/// The energy bar, hartree.
const BAR: f64 = 1e-10;

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0`, as `FBF_SKIP_IF_FAST_TESTS` reads it.
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// A geometry of `fb-pipeline-scf`'s test data.
fn data(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../scf/tests/data")
        .join(file)
}

/// The number of shell pairs `eri` keeps for `molecule` in cc-pVDZ, and
/// the doubles of B's heap row (`naux · n_columns`), at planning alone.
fn kept_pairs(molecule: &Molecule, eri: &EriStep) -> (usize, usize) {
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(molecule)
        .unwrap();
    let mut dag = Dag::new();
    let classes = add_records(&mut dag, 0, &basis).unwrap();
    let mut u = None;
    match add_eri(&mut dag, 0, 1, molecule, &basis, &classes, &mut u, eri).unwrap() {
        Eri::DensityFitted(df) => (
            df.map.pairs().len(),
            df.aux.nao_sphe() as usize * df.map.sphe().n_columns(),
        ),
        _ => unreachable!("a density-fitted step"),
    }
}

/// The converged DF-RHF energy of `molecule` in cc-pVDZ over `eri`.
fn energy(molecule: &Molecule, eri: &EriStep) -> f64 {
    let run = Run::new(molecule, "cc-pvdz", eri, &ScfStep::default(), 1, |n| {
        syevd_lwork(&SolverHandle::new(Arc::new(Stream::new(0))), n)
    })
    .expect("the case plans");
    let (record, out) = run.run(&[0]);
    record.result.unwrap();
    out.expect("a converged run brings its products back")
        .energy
}

/// `stem`'s DF-RHF energy at the default threshold, the key absent,
/// against threshold 0, within [`BAR`]; prints the pairs the default
/// drops and B's heap row at each.
fn pins_the_default(stem: &str, xyz: &Path) {
    let molecule = fb_io_xyz::read_xyz(xyz, 0, 1).unwrap();
    let all = EriStep::density_fitting("cc-pvdz-jkfit", 0.0);
    let default = EriStep {
        schwarz_threshold: None,
        ..all.clone()
    };
    let ((n_all, b_all), (n_default, b_default)) =
        (kept_pairs(&molecule, &all), kept_pairs(&molecule, &default));
    let (e_all, e_default) = (energy(&molecule, &all), energy(&molecule, &default));
    let miss = (e_default - e_all).abs();
    let dropped = n_all - n_default;
    println!(
        "{stem}: the default keeps {n_default} of {n_all} shell pairs, drops {dropped} ({:.1}%); \
         B's heap row {b_default} doubles at the default, {b_all} at 0; E(default) = \
         {e_default:.12}, E(0) = {e_all:.12}, |ΔE| = {miss:e}",
        100.0 * dropped as f64 / n_all as f64
    );
    assert!(miss <= BAR, "{stem}: |ΔE| = {miss:e} over {BAR:e}");
}

/// Benzene's DF-RHF energy at the default threshold against threshold 0
/// ([`pins_the_default`]). A slow case, which returns at once under
/// `FREE_BREAKFAST_FAST_TESTS` (`docs/testing.md`, "Slow cases"). Needs a
/// GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_default_threshold_keeps_benzenes_energy() {
    if fast_tests() {
        eprintln!("skipped: slow case (FREE_BREAKFAST_FAST_TESTS is set)");
        return;
    }
    pins_the_default("benzene", &data("benzene.xyz"));
}

/// Five glycines' DF-RHF energy at the default threshold against
/// threshold 0 ([`pins_the_default`]), the case where the default drops a
/// sizeable share of the pairs. A slow case, which returns at once under
/// `FREE_BREAKFAST_FAST_TESTS` (`docs/testing.md`, "Slow cases"). Needs a
/// GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_default_threshold_keeps_gly5s_energy() {
    if fast_tests() {
        eprintln!("skipped: slow case (FREE_BREAKFAST_FAST_TESTS is set)");
        return;
    }
    pins_the_default("gly5", &data("gly5.xyz"));
}
