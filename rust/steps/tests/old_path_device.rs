//! The steps against the old Rust path in the same process (tier 4,
//! device 0): `rust/`'s planner-and-arena layer as merged, a
//! dev-dependency of this binary alone.
//!
//! - The DF-RHF energy of water, benzene and ethylene in cc-pVDZ over
//!   cc-pVDZ-JKFIT, `fb_steps::run` against `fb_pipeline_scf::Run` with the
//!   old path's Schwarz screen off (`schwarz_threshold = 0`, as the steps
//!   do not screen) and both at their default tolerances. The bar is 1e-9
//!   Eh, `chain_device.rs`'s: `energy_tolerance` 1e-10 between two builds
//!   and the commutator 1e-7, whose square bounds a converged energy's
//!   distance from the stationary point, so two loops that converge along
//!   different paths (one streamed pass over B against a resident one,
//!   different batches) agree within a few 1e-11.
//! - S, H and the metric of water from the host steps (rayon's integrals,
//!   uploaded) against the old path's from its GPU tasks, each within 1e-12
//!   of its largest element: two builds of the same integrals in double
//!   precision, whose sums are reordered between the host's triangle and
//!   the device's classes, each element's rounding a few ulps of the
//!   largest, the bar the integral crates hold each build to against
//!   PySCF's.
//!
//! The water case is the family's representative; benzene and ethylene
//! are slow cases (`docs/testing.md`, "Slow cases").

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
#![allow(
    clippy::disallowed_macros,
    reason = "a test's comparisons, outside any step's run"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use std::sync::{Arc, Mutex, PoisonError};

use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_budget::{Backing, Caps, DeviceBuf, Ledger};
use fb_core_error_handling::Status;
use fb_core_flow::Bus;
use fb_core_planner::{CpuTask, Dag};
use fb_steps::{CoreHamiltonianStep, MetricStep, OverlapStep, ScfStep, ScfTable, run};

/// The bar on an energy, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// The bar on S, H and the metric, relative to the largest element.
const MATRIX_BAR: f64 = 1e-12;

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0` (`docs/testing.md`, "Slow cases").
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// A file of `tests/data`.
fn data(file: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(file)
}

/// The old path's DF-RHF energy of `xyz`, syevd's work asked of a solver
/// handle on device 0, run once there.
fn old_energy(xyz: &str) -> (f64, usize) {
    let molecule = fb_io_xyz::read_xyz(&data(xyz), 0, 1).unwrap();
    let eri = fb_pipeline_eri::EriStep::density_fitting("cc-pvdz-jkfit", 0.0);
    let step = fb_pipeline_scf::ScfStep::default();
    let old = fb_pipeline_scf::Run::new(&molecule, "cc-pvdz", &eri, &step, 1, |n| {
        fb_recipes_diagonalize::syevd_lwork(
            &fb_core_wwr::solver::SolverHandle::new(Arc::new(fb_core_wwr::runtime::Stream::new(0))),
            n,
        )
    })
    .expect("the old path plans");
    let (record, out) = old.run(&[0]);
    record.result.expect("the old path runs");
    let out = out.expect("the old path converges");
    (out.energy, out.iterations)
}

/// The steps' DF-RHF energy of `xyz`, converged.
fn new_energy(xyz: &str) -> (f64, usize) {
    let mut config = common::config(xyz, "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    let run = run(config, Backing::Real).expect("the chain runs");
    let scf = run.plan.output(ScfStep);
    assert!(scf.converged());
    (scf.energy(), scf.iterations())
}

/// `xyz`'s two energies, within the bar.
fn energies_agree(xyz: &str) {
    let ((new, n_new), (old, n_old)) = (new_energy(xyz), old_energy(xyz));
    println!(
        "{xyz}: steps E = {new:.12} in {n_new} iterations, old path {old:.12} in {n_old}, \
         |ΔE| = {:.1e}",
        (new - old).abs()
    );
    assert!((new - old).abs() < ENERGY_BAR, "{xyz}: {new} against {old}");
}

/// Water's energy: the family's representative.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_energy_matches_the_old_paths() {
    energies_agree("h2o.xyz");
}

/// Benzene's and ethylene's, whose DIIS subspace goes singular and
/// restarts: a slow case, about 1.5 s on an RTX 3080, ethylene's 50-odd
/// iterations on each path most of it.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn benzenes_and_ethylenes_energies_match_the_old_paths() {
    if fast_tests() {
        return;
    }
    energies_agree("benzene.xyz");
    energies_agree("ethylene_meci856.xyz");
}

/// CPU: the old plan's pinned copies read out, after the executor waited
/// for their downloads.
struct Collect {
    pinned: [RowId<Pinned, f64>; 3],
    out: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        for &row in &self.pinned {
            host.read(row);
        }
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap_or_else(PoisonError::into_inner) =
            self.pinned.iter().map(|&p| host.get(p).to_vec()).collect();
        Ok(())
    }
}

/// The old path's S, H and metric of water on device 0: its records, the
/// overlap's and the core Hamiltonian's tasks over them, the metric's, and
/// a download of each.
fn old_matrices(n: usize, q: usize) -> Vec<Vec<f64>> {
    let molecule = fb_io_xyz::read_xyz(&data("h2o.xyz"), 0, 1).unwrap();
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let aux = fb_io_basis::resolve_aux_basis("cc-pvdz-jkfit")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let mut dag = Dag::new();
    let nuclei = fb_molecule_nuclei::add_nuclei(&mut dag, 0, &molecule);
    let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
    let mut u = None;
    let s = fb_integrals_one_electron_overlap::add_overlap(&mut dag, 0, &basis, &classes, &mut u);
    let h = fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian(
        &mut dag, 0, &basis, &classes, nuclei, &mut u,
    )
    .unwrap();
    let m =
        fb_integrals_eri_density_fitting_metric::add_metric(&mut dag, 0, &aux, &mut None).unwrap();
    let pinned = [
        dag.download(0, s, "S", n * n),
        dag.download(0, h, "H", n * n),
        dag.download(0, m, "M", q * q),
    ];
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu(
        "collect",
        0,
        Collect {
            pinned,
            out: Arc::clone(&out),
        },
    );
    let planned = dag.plan();
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    std::mem::take(&mut *out.lock().unwrap_or_else(PoisonError::into_inner))
}

/// `buf`'s first `len` doubles, brought back on a stream of the test's own
/// into a pinned buffer of a ledger of its own.
fn download(buf: &DeviceBuf<f64>, len: usize) -> Vec<f64> {
    let caps = Caps::read(&[0], &[None], None, Backing::Real).unwrap();
    let ledger = Ledger::new(caps, Backing::Real, vec![0], false);
    let host = ledger.budget("test").pinned::<f64>(0, len);
    let host = fb_core_device::Stream::new(0)
        .download(host, buf)
        .unwrap()
        .sync()
        .unwrap();
    host.to_vec()
}

/// Water's S, H and metric from the host steps against the old path's
/// from the GPU, each symmetric and full, element for element.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_s_h_and_metric_match_the_old_paths() {
    let config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    let run = run(config, Backing::Real).expect("the steps run");
    let dims = &run.plan.bus().shared().dims;
    let (n, q) = (dims.n_ao, dims.n_aux);
    let new = [
        download(run.plan.output(OverlapStep).s(), n * n),
        download(run.plan.output(CoreHamiltonianStep).h(), n * n),
        download(&run.plan.output(MetricStep).gpus()[0], q * q),
    ];
    let old = old_matrices(n, q);
    for ((what, new), old) in ["S", "H", "the metric"].iter().zip(&new).zip(&old) {
        assert_eq!(new.len(), old.len(), "{what}: the lengths");
        let largest = old.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = new
            .iter()
            .zip(old)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "{what}: {} elements, max |old| = {largest:.6}, max |new - old| = {worst:.1e} \
             (bar {:.1e})",
            old.len(),
            MATRIX_BAR * largest
        );
        assert!(
            largest > 0.1 && !worst.is_nan() && worst <= MATRIX_BAR * largest,
            "{what}: {worst:e} against {:e}",
            MATRIX_BAR * largest
        );
    }
}
