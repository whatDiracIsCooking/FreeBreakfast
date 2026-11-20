//! DF-RHF in cc-pVDZ over cc-pVDZ-JKFIT on device 0, through the new
//! layer, held against the old path's energy in the same process and
//! against PySCF's (`rust/pipeline/scf/tests/data`). Water is the case the
//! prototype was built for; benzene and the ethylene geometry whose DIIS
//! subspace goes singular are run beside it.
//!
//! **The bar: `|ΔE| <= 1e-9` hartree**, the old test's, by its analysis:
//! the energy is stationary in the orbitals, both loops stop at
//! `‖e‖ < 1e-7`, and the integrals agree with PySCF's to about 1e-10 of
//! their largest. The new path's integrals are the old kernels'; what
//! differs is the order of the linear algebra around them.
//!
//! Needs a GPU:
//! `cargo test --test water -- --include-ignored --nocapture --test-threads=1`.

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use fb_core_error_handling::ErrorKind;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_pipeline_eri::EriStep as OldEriStep;
use fb_pipeline_scf::{Run as OldRun, ScfStep as OldScfStep};
use fb_recipes_diagonalize::syevd_lwork;
use fb_v02::budget::{Backing, Kind};
use fb_v02::bus::{Config, MainBus};
use fb_v02::flow::{Bus, StepTime};
use fb_v02::steps::scf::ScfStep;
use rustflow::Executor;

const ENERGY_BAR: f64 = 1e-9;

/// One case at a time: all share device 0.
static TURN: Mutex<()> = Mutex::new(());

fn rust_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../rust")
}

fn data(file: &str) -> PathBuf {
    rust_dir().join("pipeline/scf/tests/data").join(file)
}

fn water() -> PathBuf {
    rust_dir().join("integrals/one_electron/overlap/tests/data/h2o.xyz")
}

fn config(xyz: &Path, resources: &str) -> Config {
    let text = format!(
        r#"
[molecule]
xyz = "{}"

[basis]
name = "cc-pvdz"

[eri]
aux_basis = "cc-pvdz-jkfit"

[guess]

[scf]

[resources]
{resources}
"#,
        xyz.display()
    );
    Config::parse(&text, Path::new("/")).expect("the config parses")
}

struct Printer;
impl log::Log for Printer {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        println!("{}", record.args());
    }
    fn flush(&self) {}
}

fn logging(level: log::LevelFilter) {
    let _ = log::set_logger(&Printer);
    log::set_max_level(level);
}

/// The old path's converged energy, iteration count and wall time.
fn old_path(xyz: &Path) -> (f64, usize, Duration) {
    let molecule = fb_io_xyz::read_xyz(xyz, 0, 1).unwrap();
    let eri = OldEriStep::density_fitting("cc-pvdz-jkfit", 0.0);
    let run = OldRun::new(&molecule, "cc-pvdz", &eri, &OldScfStep::default(), 1, |n| {
        syevd_lwork(&SolverHandle::new(Rc::new(Stream::new(0))), n)
    })
    .expect("the old path plans");
    let (record, out) = run.run(&[0]);
    record.result.as_ref().expect("the old path runs");
    let out = out.expect("the old path converges");
    (out.energy, out.iterations, record.wall)
}

fn print_steps(steps: &[StepTime]) {
    for step in steps {
        println!(
            "  {:<18} trips {:>2}  began {:>9.3?}  ended {:>9.3?}  run {:>9.3?}  wrapper sync {:>9.3?}",
            step.name, step.trips, step.began, step.ended, step.run, step.sync
        );
    }
}

/// `stem` through the new layer, against PySCF and the old path at the bar;
/// then a replay in the same buffers, timed warm beside a second run of the
/// old path.
fn matches(stem: &str, xyz: &Path) {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    logging(log::LevelFilter::Info);
    let run = fb_v02::run(config(xyz, "threads = 4")).expect("the new path runs");
    let dims = run.plan.bus().shared().dims;
    println!("{stem}: {dims:?}");
    println!("{}", run.table);
    println!(
        "{stem}: setup {:?}, budget mode {:?}, execute mode {:?} (cold: the kernels' first launches)",
        run.setup, run.budget, run.execute
    );
    print_steps(&run.steps);
    let (energy, iterations, residual) = {
        let scf = run.plan.output(ScfStep);
        assert!(scf.converged());
        let n = scf.epsilon().len();
        assert_eq!((scf.c().len(), scf.d().len()), (n * n, n * n));
        assert!(scf.epsilon().windows(2).all(|w| w[0] <= w[1]), "ε ascends");
        (scf.energy(), scf.iterations(), scf.residual())
    };
    assert!(residual < 1e-7, "‖e‖ = {residual:e}");

    let e_ref = fb_io_npy::read_npy(&data(&format!("scf_{stem}_cc-pvdz_cc-pvdz-jkfit.npy")))
        .unwrap()
        .data[0];
    let (e_old, old_iterations, _) = old_path(xyz);
    println!(
        "{stem}: new  E = {energy:.12}  {iterations} iterations\n\
         {stem}: old  E = {e_old:.12}  {old_iterations} iterations\n\
         {stem}: ref  E = {e_ref:.12}\n\
         {stem}: |new - old| = {:.3e}, |new - ref| = {:.3e}, |old - ref| = {:.3e}",
        (energy - e_old).abs(),
        (energy - e_ref).abs(),
        (e_old - e_ref).abs()
    );
    assert!(
        (energy - e_ref).abs() <= ENERGY_BAR,
        "{stem}: against PySCF"
    );
    assert!(
        (energy - e_old).abs() <= ENERGY_BAR,
        "{stem}: against the old path"
    );

    // Execute allocated nothing, and a replay runs in the same buffers to
    // the same energy. Warm, so its time is the steps' and not the kernels'
    // first launches; the old path's second run is timed the same way.
    logging(log::LevelFilter::Warn);
    let ledger = run.plan.bus().ledger();
    let allocated = ledger.allocations();
    let executor = Executor::new(4);
    let start = Instant::now();
    let replay = run.plan.execute(&executor).expect("the replay runs");
    let warm = start.elapsed();
    assert_eq!(
        ledger.allocations(),
        allocated,
        "a replay allocates nothing"
    );
    let again = run.plan.output(ScfStep).energy();
    let trips = replay.iter().find(|t| t.name == "scf").unwrap().trips as usize;
    let (_, _, old_warm) = old_path(xyz);
    println!(
        "{stem}: replay E = {again:.12}, {trips} trips, |Δ| = {:.3e}; warm execute {warm:?}, \
         old path's warm run {old_warm:?}",
        (again - energy).abs()
    );
    print_steps(&replay);
    assert!(
        (again - energy).abs() <= 1e-10,
        "{stem}: the replay's energy"
    );
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_matches_the_old_path_and_pyscf() {
    matches("h2o", &water());
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn benzene_matches_the_old_path_and_pyscf() {
    matches("benzene", &data("benzene.xyz"));
}

/// The old tree's singular-DIIS case (#865): the crude restart here is not
/// the old loop's, so the iteration counts may differ; the energy may not.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn ethylene_past_a_singular_diis_matches_the_old_path_and_pyscf() {
    matches("ethylene_meci856", &data("ethylene_meci856.xyz"));
}

/// Work stealing against the device: twenty whole runs at eight workers,
/// any worker running any step, each run's energy the first's.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn twenty_runs_at_eight_workers_agree() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    logging(log::LevelFilter::Warn);
    let mut energies = Vec::new();
    let mut overlapped = 0;
    for _ in 0..20 {
        let run = fb_v02::run(config(&water(), "threads = 8")).expect("runs");
        energies.push(run.plan.output(ScfStep).energy());
        // Did two independent steps hold the card at once?
        let time = |name: &str| run.steps.iter().find(|t| t.name == name).unwrap().clone();
        let (eri, core) = (time("eri"), time("core_hamiltonian"));
        if eri.began < core.ended && core.began < eri.ended {
            overlapped += 1;
        }
    }
    let spread = energies
        .iter()
        .map(|e| (e - energies[0]).abs())
        .fold(0.0, f64::max);
    println!(
        "20 runs: E = {:.12}, spread {spread:.3e}; eri and core_hamiltonian overlapped in \
         {overlapped}",
        energies[0]
    );
    assert!(spread <= 1e-10, "the runs' energies: {energies:?}");
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_device_cap_too_small_refuses_the_plan_before_anything_runs() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    logging(log::LevelFilter::Warn);
    let error = fb_v02::run(config(&water(), "device_cap_mib = 4"))
        .err()
        .expect("refused");
    println!("{error}");
    assert_eq!(error.kind, ErrorKind::Allocation);
    for step in ["eri", "scf", "records", "overlap", "workspace (max)"] {
        assert!(error.message.contains(step), "the table names {step}");
    }
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn scf_without_eri_is_refused_when_the_dag_is_built() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let mut config = config(&water(), "");
    config.eri = None;
    let error = fb_v02::run(config).err().expect("refused");
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert_eq!(
        error.message,
        "step 'scf' reads step 'eri', which the config does not ask for"
    );
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_budget_is_the_same_dry_and_real_but_for_the_workspace() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    logging(log::LevelFilter::Warn);
    let executor = Executor::new(4);
    let totals = |backing| {
        let bus = MainBus::setup(config(&water(), ""), backing).expect("setup");
        let plan = fb_v02::plan(bus).expect("the plan");
        plan.budget(&executor).expect("accepted");
        let ledger = plan.bus().ledger();
        let workspace = ledger.total(Kind::Workspace(0));
        (
            ledger.total(Kind::Device(0)) - workspace,
            ledger.total(Kind::Pinned(0)),
            workspace,
        )
    };
    let (dry, real) = (totals(Backing::Dry), totals(Backing::Real));
    println!("dry {dry:?}, real {real:?}");
    assert_eq!(
        (dry.0, dry.1),
        (real.0, real.1),
        "sizes are the dimensions' alone"
    );
    // The workspace query needs a live handle: a dry budget has none.
    assert_eq!(dry.2, 0);
    assert!(real.2 > 0);
}
