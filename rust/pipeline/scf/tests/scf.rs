//! The DF-RHF loop run on a card (tier 4, device 0, one run per case), from
//! the SAD guess, against PySCF's DF-RHF energy over the same orbital and
//! fitting documents and geometry, converged to `conv_tol = 1e-12`
//! (`tests/data/scf_<stem>.npy`, written by `tests/data/scf_reference.py`,
//! which says how; ethylene's by PySCF's second-order solver, whose case
//! says why).
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
//! Beside it, on the same run: no warning (the SAD guess is positive
//! semidefinite; a loaded guess that is not is its own case, warned of
//! once), the final residual under
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
use fb_core_planner::Dag;
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_integrals_mmd::add_records;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_molecule_nuclei::add_nuclei;
use fb_pipeline_eri::add_df_eri;
use fb_pipeline_guess::{add_guess_upload, add_loaded_guess};
use fb_pipeline_scf::{Run, ScfOutput, ScfStep, add_scf};
use fb_recipes_diagonalize::syevd_lwork;
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};
use wwr_sys::wwr_status;

/// The energy bar, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// Held by each case for its whole length: the free-memory samples are
/// one process's, so the cases do not overlap.
static TURN: Mutex<()> = Mutex::new(());

/// The card's free memory at each `scf: iteration` line, in order.
static FREE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Every line logged at `warn`, in order.
static WARNED: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Every line the loop's decision logs as it restarts a singular DIIS
/// subspace, in order.
static RESTARTED: Mutex<Vec<String>> = Mutex::new(Vec::new());

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

/// A logger that prints every line, keeps each warning and each DIIS
/// restart, and, at each iteration line, samples the free memory: the
/// decision writes the line after the body's GPU work has completed.
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
        if record.level() == log::Level::Warn {
            WARNED.lock().unwrap().push(line.clone());
        }
        if line.starts_with("scf: DIIS subspace singular") {
            RESTARTED.lock().unwrap().push(line.clone());
        }
        println!("{line}");
    }

    fn flush(&self) {}
}

/// Install [`Sampler`] once, empty the samples, the warnings and the
/// restarts, and return the case's turn.
fn turn() -> std::sync::MutexGuard<'static, ()> {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    let guard = TURN.lock().unwrap_or_else(|e| e.into_inner());
    INSTALLED.get_or_init(|| {
        log::set_logger(&Sampler).expect("no other logger in this binary");
        log::set_max_level(log::LevelFilter::Info);
    });
    FREE.lock().unwrap().clear();
    WARNED.lock().unwrap().clear();
    RESTARTED.lock().unwrap().clear();
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

/// The case `stem` run with `step`, converged and held to the energy bar,
/// the residual under the tolerance, the free memory never falling from the
/// second iteration to the last, and no warning: the SAD guess is positive
/// semidefinite, so `guess check` has nothing to say. Returns the lines the
/// decision logged as it restarted a singular DIIS subspace.
fn converges_to_pyscf(stem: &str, xyz: &Path, step: &ScfStep) -> Vec<String> {
    let _turn = turn();
    let (record, out) = run(xyz, step);
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
    let warned = WARNED.lock().unwrap();
    assert!(warned.is_empty(), "{stem}: warned {warned:?}");
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
    std::mem::take(&mut *RESTARTED.lock().unwrap())
}

/// Water in cc-pVDZ over cc-pVDZ-JKFIT: the gate's representative. Needs a
/// GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_converges_to_pyscf() {
    converges_to_pyscf("h2o", &water(), &ScfStep::default());
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
    converges_to_pyscf("benzene", &data("benzene.xyz"), &ScfStep::default());
}

/// Ethylene at #856's DF-SA3-CASSCF(2,2) S1/S0 MECI in cc-pVDZ over
/// cc-pVDZ-JKFIT (48 AOs, 232 fitting functions), the C++ tree's issue #865
/// case (`test/gtest/pipeline_df_scf_gpu.cpp`, `meci856_ccpvdz`): a run
/// that takes the singular-DIIS restart in the loop, and converges past it
/// to PySCF's energy at the same bar. Needs a GPU.
///
/// **Why it restarts.** The RHF orbital Hessian has one eigenvalue near
/// zero there (PySCF: 6.4e-6 Eh), along which the plain Roothaan step
/// barely moves, so the loop sits at `‖e‖` about 1.3e-6, where DIIS's
/// residuals differ by too little for its absolute singular pivot
/// (`fb_recipes_diis`). The C++ loop stalled there for 400 iterations
/// before its restart; this one logs one restart at that `‖e‖` and
/// converges. The case asserts the restart's log line, so a loop that
/// never took the path fails it.
///
/// **The reference** is PySCF's second-order solver (`.newton()`;
/// `tests/data/scf_reference.py`, its `newton` argument): PySCF's own DIIS
/// stalls on the same mode, at an orbital gradient of 1e-6 and 2.1e-7 Eh
/// above the minimum, and reports itself converged at `conv_tol = 1e-12`.
/// The C++ case's energy, also `.newton()`'s, agrees with it to 1e-12.
///
/// **The bar** stays [`ENERGY_BAR`], and the run is at
/// `commutator_tolerance = 1e-8`, a decade under the default, to keep it:
/// the energy's convergence error is `g² / 2λ` along the mode, `g` the
/// orbital gradient and `λ` the Hessian's eigenvalue. `‖e‖ = 2√2 ‖F_vo‖`
/// in the orthonormal basis, and PySCF's `g` is `2 F_vo` with its `λ` in
/// the same convention, so the error is at most `‖e‖² / 4λ`, and `‖e‖² /
/// λ` leaves a factor of 4 for the convention: 1.6e-9 at the default
/// `‖e‖ < 1e-7` if all of it lay along the soft mode, past the bar, and
/// 1.6e-11 at 1e-8, under it beside the integrals' 1e-10. The restart is
/// at `‖e‖ = 1.3e-6`, well above either tolerance.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn ethylene_takes_the_diis_restart_and_converges_to_pyscf() {
    let step = ScfStep {
        commutator_tolerance: 1e-8,
        ..ScfStep::default()
    };
    let restarted = converges_to_pyscf("ethylene_meci856", &data("ethylene_meci856.xyz"), &step);
    assert!(
        !restarted.is_empty(),
        "ethylene_meci856: no singular-DIIS restart was logged"
    );
    println!("ethylene_meci856: {restarted:?}");
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

/// Needs a GPU. Water planned from a loaded guess that is not positive
/// semidefinite, `D = diag(1/2, …, 1/2, -1/4)` written to a `.npy` file
/// ([`add_loaded_guess`]), at `max_iterations = 1`: the run fails with
/// `MaxIterationsExceeded` after its one build, and before the loop
/// `guess check` has warned once, of the eigenvalue `-2.500e-1` below
/// `-1e-12` of the largest, as the C++ first build does. The SAD runs
/// above warn of nothing.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_guess_with_a_negative_eigenvalue_is_warned_of() {
    let _turn = turn();
    let molecule = fb_io_xyz::read_xyz(&water(), 0, 1).unwrap();
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let n = basis.nao_sphe() as usize;
    let mut d = vec![0.0; n * n];
    for i in 0..n {
        d[i * (n + 1)] = 0.5;
    }
    d[n * n - 1] = -0.25;
    let path = std::env::temp_dir().join(format!(
        "fb-pipeline-scf-negative-guess-{}.npy",
        std::process::id()
    ));
    fb_io_npy::write_npy(
        &path,
        &fb_io_npy::NpyArray {
            shape: vec![n, n],
            data: d,
        },
    )
    .unwrap();

    let lwork = syevd_lwork(&SolverHandle::new(Rc::new(Stream::new(0))), n).unwrap();
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let nuclei = add_nuclei(&mut dag, 0, &molecule);
    let mut u = None;
    let classes = add_records(&mut dag, 0, &basis).unwrap();
    let s = add_overlap(&mut dag, 0, &basis, &classes, &mut u);
    let h = add_core_hamiltonian(&mut dag, 0, &basis, &classes, nuclei, &mut u).unwrap();
    let x = add_orthogonalize(
        &mut dag,
        0,
        "X",
        n,
        lwork,
        s,
        ORTHOGONALIZE_EIGENVALUE_FLOOR,
    )
    .unwrap();
    let eri = add_df_eri(
        &mut dag,
        0,
        &molecule,
        &basis,
        &classes,
        &mut u,
        "cc-pvdz-jkfit",
    )
    .unwrap();
    let guess = add_loaded_guess(&mut dag, 0, &basis, path.clone()).unwrap();
    let guess = add_guess_upload(&mut dag, 0, &basis, guess);
    let step = ScfStep {
        max_iterations: 1,
        ..ScfStep::default()
    };
    add_scf(
        &mut dag,
        0,
        &step,
        &molecule,
        n,
        lwork,
        &eri,
        [h, s, x],
        guess,
    )
    .unwrap();
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    let _ = std::fs::remove_file(&path);

    let err = record.result.expect_err("not converged in one build");
    assert_eq!(err.kind, ErrorKind::MaxIterationsExceeded, "{err:?}");
    let warned = WARNED.lock().unwrap();
    assert_eq!(
        *warned,
        [
            "scf: the first density has an eigenvalue -2.500e-1, below -1.0e-12 of its largest; \
          its exchange is of the positive part"
        ],
    );
}
