//! The RHF over the dense conventional ERI (`[eri] mode = "conventional"`,
//! `storage = "dense"`) on one, two and four cards, GPU worker `w` on
//! device `w`, from the SAD guess, against PySCF's exact RHF over the same
//! orbital document and geometry, converged to `conv_tol = 1e-12`
//! (`tests/data/scf_<stem>_<basis>_conventional.npy`, written by
//! `tests/data/conventional_reference.py`, whose doc says how).
//!
//! **The bar, by analysis, one run: `|E - E_ref| <= 1e-11 |E_ref|`.** The
//! packed path's water case holds 1e-9 hartree (`tests/scf.rs` has the
//! analysis): the convergence error is second order in the residual, and
//! what is left is J's and K's agreement with PySCF's, a few times
//! `1e-15 max |J|` here (`fb-integrals-eri-conventional`'s
//! `tests/dense.rs`), carried into the energy's trace over `n²` terms. That
//! rounding grows with the energy's own scale, so the bar is relative,
//! water's 1e-9 over its 76 hartree, 1.3e-11 of it, rounded down: 7.6e-10
//! for water, 1.1e-8 for five glycines and 2.1e-8 for ten.
//!
//! Ten glycines on one card are refused by admission, at planning
//! (`fb-pipeline-config`'s `tests/admission.rs`, where the binary's
//! admission is).
//!
//! **Across cards.** J and K over the dense blocks are the same bits on
//! any number of workers (`fb-integrals-eri-conventional`'s
//! `tests/dense.rs` asserts it), but the rest of the loop is not bitwise
//! from one run to the next even on one card (two one-card runs of water
//! differ in the last bit of E), so each card count is held to the bar on
//! its own, not to another's bits.
//!
//! The glycine chains (`tests/data/gly5.xyz`, `gly10.xyz`) are the slow
//! cases (`docs/testing.md`, "Slow cases"): each returns at once under
//! `FREE_BREAKFAST_FAST_TESTS`, and water is the family's representative.
//! Each prints the run's times read off its record: the dense ERI's build
//! (`dense eri`, the slowest worker's device time), J and K per iteration
//! (worker 0's stream time over `dense density`, the broadcast, `dense jk`,
//! the two gathers and `dense spherical`), the products alone (`dense
//! jk`), the run's wall time and the iteration count.

use std::path::{Path, PathBuf};
use std::rc::Rc;

use fb_core_thread_pool::Record;
use fb_core_wwr::runtime::{Stream, device_count};
use fb_core_wwr::solver::SolverHandle;
use fb_pipeline_eri::{EriMode, EriStep, EriStorage};
use fb_pipeline_scf::{Run, ScfStep};
use fb_recipes_diagonalize::syevd_lwork;

/// The bar's fraction of `|E_ref|`.
const REL_BAR: f64 = 1e-11;

/// The slow-case switch (`docs/testing.md`, "Slow cases").
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

fn data(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// The geometry of `stem`: water's is the overlap crate's.
fn xyz(stem: &str) -> PathBuf {
    if stem == "h2o" {
        return Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../integrals/one_electron/overlap/tests/data/h2o.xyz");
    }
    data(&format!("{stem}.xyz"))
}

/// The dense conventional integrals.
fn dense() -> EriStep {
    EriStep {
        mode: EriMode::Conventional,
        aux_basis: None,
        schwarz_threshold: None,
        storage: Some(EriStorage::Dense),
    }
}

/// The plan of `stem` in `basis` over the dense integrals on `gpus`
/// workers, syevd's work asked of a solver handle on device 0.
fn plan(stem: &str, basis: &str, gpus: usize) -> Run {
    let molecule = fb_io_xyz::read_xyz(&xyz(stem), 0, 1).unwrap();
    Run::new(&molecule, basis, &dense(), &ScfStep::default(), gpus, |n| {
        syevd_lwork(&SolverHandle::new(Rc::new(Stream::new(0))), n)
    })
    .expect("the case plans")
}

/// The slowest worker's device time, in seconds, of the entries of
/// `record` named `name`, in iteration `iteration` (`None`: outside the
/// loop), summed over a worker's entries.
fn slowest(record: &Record, name: &str, iteration: Option<usize>) -> f64 {
    let mut by_worker = std::collections::BTreeMap::<String, f64>::new();
    for t in &record.tasks {
        if t.name == name && t.iteration == iteration {
            let worker = t.worker.split('.').next().unwrap_or("").to_string();
            *by_worker.entry(worker).or_default() += t.device.map_or(0.0, |d| d.as_secs_f64());
        }
    }
    by_worker.values().copied().fold(0.0, f64::max)
}

/// J and K's stages, in the order each worker's stream runs its own.
const STAGES: [&str; 6] = [
    "dense density",
    "broadcast D cartesian",
    "dense jk",
    "gather J cartesian",
    "gather K cartesian",
    "dense spherical",
];

/// The run's times, in seconds: the ERI build; J and K per iteration,
/// worker 0's stream time over its stages, from D in to J and K out, the
/// gathers' waits for the other workers' products among it, the mean over
/// the iterations; the products alone per iteration (`dense jk`, the
/// slowest worker's); and the wall. Worker 0's, not the slowest
/// worker's: another worker's broadcast part starts its clock when its
/// thread issues it, and waits there for worker 0's whole iteration.
fn times(record: &Record, iterations: usize) -> (f64, f64, f64, f64) {
    let build = slowest(record, "dense eri", None);
    let jk: f64 = record
        .tasks
        .iter()
        .filter(|t| {
            t.iteration.is_some()
                && STAGES.contains(&t.name.as_str())
                && t.worker.split('.').next() == Some("gpu0")
        })
        .map(|t| t.device.map_or(0.0, |d| d.as_secs_f64()))
        .sum();
    let product: f64 = (0..iterations)
        .map(|i| slowest(record, "dense jk", Some(i)))
        .sum();
    let n = iterations.max(1) as f64;
    (build, jk / n, product / n, record.wall.as_secs_f64())
}

/// `stem` in `basis` on `gpus` cards, converged and held to the bar
/// against its reference; its times printed.
fn converges(stem: &str, basis: &str, gpus: usize) {
    let run = plan(stem, basis, gpus);
    assert_eq!(
        run.planned().n_workers(),
        gpus,
        "{stem}: the plan's workers"
    );
    let devices: Vec<i32> = (0..gpus as i32).collect();
    let (record, out) = run.run(&devices);
    record.result.clone().unwrap();
    let out = out.expect("a converged run brings its products back");
    let e_ref = fb_io_npy::read_npy(&data(&format!("scf_{stem}_{basis}_conventional.npy")))
        .unwrap()
        .data[0];
    let miss = (out.energy - e_ref).abs();
    let (build, jk, product, wall) = times(&record, out.iterations);
    println!(
        "{stem}/{basis} on {gpus} GPU(s): E = {:.12} against {e_ref:.12}, |ΔE| = {miss:e}, \
         {} iterations, ‖e‖ = {:e}; ERI build {build:.3} s, J+K {:.2} ms per iteration \
         (the products {:.2} ms), wall {wall:.3} s ({} profile)",
        out.energy,
        out.iterations,
        out.residual,
        jk * 1e3,
        product * 1e3,
        record.profile
    );
    assert!(
        miss <= REL_BAR * e_ref.abs(),
        "{stem}/{basis} on {gpus}: |ΔE| = {miss:e} over {:e}",
        REL_BAR * e_ref.abs()
    );
}

/// The visible devices, at least two.
fn visible() -> usize {
    let n = device_count().unwrap();
    assert!(n >= 2, "{n} devices");
    n
}

/// Water in cc-pVDZ over the dense integrals on one card: the family's
/// representative, every class through (dd|dd). Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_dense_converges_to_pyscf() {
    converges("h2o", "cc-pvdz", 1);
}

/// Water on two cards, and on four where four are visible, each to the
/// bar: the tier-5 representative. Needs two GPUs.
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_water_dense_across_cards() {
    let visible = visible();
    let counts: &[usize] = if visible >= 4 { &[2, 4] } else { &[2] };
    for &gpus in counts {
        converges("h2o", "cc-pvdz", gpus);
    }
}

/// Five glycines in STO-3G, 122 AOs, `M_J` and `M_K` 3.5 GB together, on
/// one card. Slow. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn gly5_sto3g_on_one_card() {
    if fast_tests() {
        return;
    }
    converges("gly5", "sto-3g", 1);
}

/// Five glycines in 6-31G, 223 AOs, 40 GB together, on two cards. Slow.
/// Needs two GPUs.
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_gly5_631g_on_two_cards() {
    if fast_tests() {
        return;
    }
    visible();
    converges("gly5", "6-31g", 2);
}

/// Five glycines in STO-3G on two cards and on four where four are
/// visible, each to the bar. Slow. Needs two GPUs.
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_gly5_sto3g_across_cards() {
    if fast_tests() {
        return;
    }
    let visible = visible();
    let counts: &[usize] = if visible >= 4 { &[2, 4] } else { &[2] };
    for &gpus in counts {
        converges("gly5", "sto-3g", gpus);
    }
}

/// Ten glycines in STO-3G, 237 AOs, `M_J` and `M_K` 50 GB together: on
/// two cards, and on four where four are visible. Slow. Needs two GPUs.
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_gly10_sto3g_across_cards() {
    if fast_tests() {
        return;
    }
    let visible = visible();
    converges("gly10", "sto-3g", 2);
    if visible >= 4 {
        converges("gly10", "sto-3g", 4);
    }
}
