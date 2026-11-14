//! The DF-RHF loop run on a card (tier 4, device 0, one run per case), from
//! the SAD guess, against PySCF's DF-RHF energy over the same orbital and
//! fitting documents and geometry, converged to `conv_tol = 1e-12`
//! (`tests/data/scf_<stem>.npy`, written by `tests/data/scf_reference.py`,
//! which says how).
//!
//! **The bar, by analysis, one run: `|E - E_ref| <= 1e-9` hartree.** The
//! energy is stationary in the orbitals, so each run's convergence error
//! is second order in its residual: ours stops at `‖e‖ < 1e-7`
//! (`commutator_tolerance`'s default), so at about 1e-14 of the energy's
//! curvature, and PySCF's at `conv_tol = 1e-12`. What is left is the
//! integrals' agreement: B's `BᵀB`, J and K agree with PySCF's to 1e-10 of
//! their largest (`fb-pipeline-eri`'s `tests/jk.rs`), and the energy is a
//! trace of D with them, a few times `1e-10 max|J|` over water's 24 AOs,
//! about 1e-10 hartree: one decade of margin.
//!
//! Beside it, on the same run: the final residual under
//! `commutator_tolerance`, and replay allocates nothing (`docs/memory.md`,
//! "What is checked"): the card's free memory, read at each iteration's
//! log line from the loop's decision, after the body's GPU work has
//! completed, never falls from one iteration to the next, from the second
//! to the last. Not from the first: its DIIS push holds one pair, so it
//! launches neither the solve nor the extrapolating gemv, and the second
//! iteration's first launch of them loads code onto the card once per
//! process (2 MiB on the gfx1200, measured: a second SCF in the same
//! process moves nothing from its first iteration on). That is the
//! reserve's growth as modules load (`docs/memory.md`, "Open questions"),
//! not an allocation of the replay.
//!
//! **Why a fall, not any change.** The reading is the device's
//! (`wwr_mem_get_info`: `cudaMemGetInfo`, `hipMemGetInfo`), so it counts
//! every tenant of the card, and the gates' card lock holds out only the
//! gates' own: #1553 saw 8.25 MiB come back between two iterations, freed
//! by a process outside the container. Nothing reachable counts this
//! process alone: wwr's memory-pool attributes count the stream-ordered
//! pool, which a library's plain allocation bypasses, and the per-process
//! counts are NVML's and amd-smi's, which wwr does not wrap and which do
//! not see across a container's PID namespace. So the check reads the one
//! direction an allocation of this process moves the reading: an
//! allocation behind the arena lowers free memory, and another tenant's
//! free cannot. Against equality it gives up two cases, neither the
//! allocation the rule forbids: a library freeing mid-run what it took
//! before the first sample (the allocation was already outside the
//! window; its free was all equality saw), and an allocation in the same
//! interval as a larger foreign free, which a run on a quiet card fails.
//! Neither check sees an allocation freed again within one iteration,
//! between two samples. Another tenant allocating mid-run can still fail
//! it; the samples are printed so the failure can be read against the
//! card's other tenants.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};

use fb_core_error_handling::ErrorKind;
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_pipeline_scf::{Run, ScfOutput, ScfStep};
use fb_recipes_diagonalize::syevd_lwork;
use wwr_sys::wwr_status;

/// The energy bar, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// Held by each case for its whole length: the free-memory samples are
/// one process's, so the cases do not overlap.
static TURN: Mutex<()> = Mutex::new(());

/// The card's free memory at each `scf: iteration` line, in order.
static FREE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Device 0's free memory, with it made current.
#[allow(
    clippy::disallowed_methods,
    reason = "a test measuring that a loop's replay allocates nothing reads free memory; no width is sized from it"
)]
fn free_bytes() -> usize {
    let _device = ScopedDeviceIndex::new(0);
    let (mut free, mut total) = (0, 0);
    // SAFETY: a plain runtime call writing two locals.
    let status = unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) };
    assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
    free
}

/// A logger that prints every line and, at each iteration line, samples
/// the free memory: the decision writes the line after the body's GPU
/// work has completed.
struct Sampler;

impl log::Log for Sampler {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record<'_>) {
        let line = record.args().to_string();
        if line.starts_with("scf: iteration") {
            FREE.lock().unwrap().push(free_bytes());
        }
        println!("{line}");
    }

    fn flush(&self) {}
}

/// Install [`Sampler`] once, empty the samples, and return the case's
/// turn.
fn turn() -> std::sync::MutexGuard<'static, ()> {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    let guard = TURN.lock().unwrap_or_else(|e| e.into_inner());
    INSTALLED.get_or_init(|| {
        log::set_logger(&Sampler).expect("no other logger in this binary");
        log::set_max_level(log::LevelFilter::Info);
    });
    FREE.lock().unwrap().clear();
    guard
}

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0`, as `FBF_SKIP_IF_FAST_TESTS` reads it.
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

fn data(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

fn water() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../integrals/one_electron/overlap/tests/data/h2o.xyz")
}

/// The run of `xyz` in cc-pVDZ over cc-pVDZ-JKFIT with `step`, syevd's
/// work asked of a solver handle on device 0, run once there.
fn run(xyz: &Path, step: &ScfStep) -> (fb_core_thread_pool::Record, Option<ScfOutput>) {
    let run = Run::new(xyz, "cc-pvdz", "cc-pvdz-jkfit", step, |n| {
        syevd_lwork(&SolverHandle::new(Rc::new(Stream::new(0))), n)
    })
    .expect("the case plans");
    run.run(0)
}

/// The case `stem` converged and held to the energy bar, the residual
/// under the tolerance, and the free memory never falling from the second
/// iteration to the last.
fn converges_to_pyscf(stem: &str, xyz: &Path) {
    let _turn = turn();
    let step = ScfStep::default();
    let (record, out) = run(xyz, &step);
    record.result.unwrap();
    let out = out.expect("a converged run brings its products back");
    let e_ref = fb_io_npy::read_npy(&data(&format!("scf_{stem}_cc-pvdz_cc-pvdz-jkfit.npy")))
        .unwrap()
        .data[0];
    let miss = (out.energy - e_ref).abs();
    println!(
        "{stem}: E = {:.12} against {e_ref:.12}, |ΔE| = {miss:e}, {} iterations, ‖e‖ = {:e}, \
         wall {:?}",
        out.energy, out.iterations, out.residual, record.wall
    );
    assert!(
        miss <= ENERGY_BAR,
        "{stem}: |ΔE| = {miss:e} over {ENERGY_BAR:e}"
    );
    assert!(
        out.residual < step.commutator_tolerance,
        "{stem}: ‖e‖ = {:e}",
        out.residual
    );
    let n = (out.epsilon.len(), out.c.len(), out.d.len());
    assert_eq!((n.0 * n.0, n.0 * n.0), (n.1, n.2), "{stem}: ε, C and D");
    let free = FREE.lock().unwrap();
    assert_eq!(free.len(), out.iterations, "{stem}: one line per iteration");
    println!("{stem}: free memory at each iteration: {free:?}");
    // From the second iteration on: the first launches no DIIS solve. A
    // fall only: another tenant's free raises the device's reading.
    assert!(free.len() >= 2, "{stem}: {free:?}");
    let fell = free[1..].windows(2).position(|w| w[1] < w[0]);
    assert!(
        fell.is_none(),
        "{stem}: free memory fell after iteration {}: {free:?}",
        fell.map_or(0, |i| i + 2)
    );
}

/// Water in cc-pVDZ over cc-pVDZ-JKFIT: the gate's representative. Needs a
/// GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_converges_to_pyscf() {
    converges_to_pyscf("h2o", &water());
}

/// Benzene in cc-pVDZ over cc-pVDZ-JKFIT (114 AOs, 558 fitting functions),
/// the same bars: a slow case, which returns at once under
/// `FREE_BREAKFAST_FAST_TESTS` (`docs/testing.md`, "Slow cases"). Needs a
/// GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn benzene_converges_to_pyscf() {
    if fast_tests() {
        eprintln!("skipped: slow case (FREE_BREAKFAST_FAST_TESTS is set)");
        return;
    }
    converges_to_pyscf("benzene", &data("benzene.xyz"));
}

/// Water's plan at `max_iterations = 2` fails the run with
/// `MaxIterationsExceeded`, and brings nothing back: two Fock builds are
/// not enough from the SAD guess. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn too_few_iterations_is_max_iterations_exceeded() {
    let _turn = turn();
    let step = ScfStep {
        max_iterations: 2,
        ..ScfStep::default()
    };
    let (record, out) = run(&water(), &step);
    let err = record.result.expect_err("not converged in two builds");
    assert_eq!(err.kind, ErrorKind::MaxIterationsExceeded, "{err:?}");
    assert!(out.is_none());
    assert_eq!(FREE.lock().unwrap().len(), 2, "two iterations logged");
}
