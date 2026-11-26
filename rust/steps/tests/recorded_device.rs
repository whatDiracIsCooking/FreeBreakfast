//! The steps against the old Rust path's recorded results (tier 4, device
//! 0): `rust/`'s planner-and-arena layer, run once and its output
//! committed under `tests/data/` with its provenance, so this binary needs
//! none of its crates. The writer was the old `pipeline/scf`'s
//! `tests/steps_reference.rs`, deleted with that crate (#1873);
//! `tests/data/old_path.json` names it, the card and the revision it ran
//! on, where it can still be read and rerun.
//!
//! - The DF-RHF energy of water, benzene and ethylene in cc-pVDZ over
//!   cc-pVDZ-JKFIT, `fb_steps::run` against `old_path.json`'s, both with
//!   the Schwarz screen off (`schwarz_threshold = 0`) and at their default
//!   tolerances. The bar is 1e-9 Eh, `chain_device.rs`'s:
//!   `energy_tolerance` 1e-10 between two builds and the commutator 1e-7,
//!   whose square bounds a converged energy's distance from the stationary
//!   point, so two loops that converge along different paths (one streamed
//!   pass over B against a resident one, different batches) agree within a
//!   few 1e-11.
//! - S, H and the metric of water from the host steps (rayon's integrals,
//!   uploaded) against the old path's from its GPU tasks
//!   (`old_path_h2o_{s,h,metric}.npy`, each buffer flat in row-major
//!   order), each within 1e-12 of its largest element: two builds of the
//!   same integrals in double precision, whose sums are reordered between
//!   the host's triangle and the device's classes, each element's rounding
//!   a few ulps of the largest, the bar the integral crates hold each build
//!   to against PySCF's.
//!
//! The water cases are the family's representatives; benzene and ethylene
//! are a slow case (`docs/testing.md`, "Slow cases").

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

use fb_core_budget::{Backing, Caps, DeviceBuf, Ledger};
use fb_core_flow::Bus;
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

/// The old path's recorded DF-RHF energy of `xyz` and its iterations.
fn recorded_energy(xyz: &str) -> (f64, u64) {
    let text = std::fs::read_to_string(data("old_path.json")).unwrap();
    let record: serde_json::Value = serde_json::from_str(&text).unwrap();
    let entry = record["energies"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["xyz"] == xyz)
        .unwrap_or_else(|| panic!("old_path.json records {xyz}"));
    (
        entry["energy"].as_f64().unwrap(),
        entry["iterations"].as_u64().unwrap(),
    )
}

/// The steps' DF-RHF energy of `xyz`, converged, the screen off.
fn new_energy(xyz: &str) -> (f64, usize) {
    let mut config = common::config(xyz, "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.eri.as_mut().unwrap().schwarz_threshold = 0.0;
    config.scf = Some(ScfTable::default());
    let run = run(config, Backing::Real).expect("the chain runs");
    let scf = run.plan.output(ScfStep);
    assert!(scf.converged());
    (scf.energy(), scf.iterations())
}

/// `xyz`'s energy against the recording, within the bar.
fn energy_agrees(xyz: &str) {
    let ((new, n_new), (old, n_old)) = (new_energy(xyz), recorded_energy(xyz));
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
    energy_agrees("h2o.xyz");
}

/// Benzene's and ethylene's, whose DIIS subspace goes singular and
/// restarts: a slow case, ethylene's 50-odd iterations most of it.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn benzenes_and_ethylenes_energies_match_the_old_paths() {
    if fast_tests() {
        return;
    }
    energy_agrees("benzene.xyz");
    energy_agrees("ethylene_meci856.xyz");
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
/// recorded from the GPU, each symmetric and full, element for element.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_s_h_and_metric_match_the_old_paths() {
    let config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    let run = run(config, Backing::Real).expect("the steps run");
    let dims = &run.plan.bus().shared().dims;
    let (n, q) = (dims.n_ao, dims.n_aux);
    let new = [
        (n, download(run.plan.output(OverlapStep).s(), n * n)),
        (n, download(run.plan.output(CoreHamiltonianStep).h(), n * n)),
        (q, download(&run.plan.output(MetricStep).gpus()[0], q * q)),
    ];
    let files = [
        "old_path_h2o_s.npy",
        "old_path_h2o_h.npy",
        "old_path_h2o_metric.npy",
    ];
    for ((what, (dim, new)), file) in ["S", "H", "the metric"].iter().zip(&new).zip(files) {
        let old = fb_io_npy::read_npy(&data(file)).unwrap();
        assert_eq!(old.shape, [*dim, *dim], "{what}: the shape");
        let old = old.data;
        let largest = old.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let worst = new
            .iter()
            .zip(&old)
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
