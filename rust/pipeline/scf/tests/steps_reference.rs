//! The writer of `fb-steps`' recorded old-path references (tier 4, device
//! 0): what `steps/tests/recorded_device.rs` holds the v0.0.4 layer to,
//! recorded once from this layer so the steps need none of its crates.
//!
//! - `steps/tests/data/old_path.json`: the DF-RHF energy and iteration
//!   count of water, benzene and ethylene in cc-pVDZ over cc-pVDZ-JKFIT
//!   ([`fb_pipeline_scf::Run`] with the Schwarz screen off,
//!   `schwarz_threshold = 0`, at its default tolerances), with the
//!   recording's provenance: this writer, the card and the revision.
//! - `steps/tests/data/old_path_h2o_{s,h,metric}.npy`: water's S, H and
//!   spherical metric from this layer's GPU tasks, each buffer flat in its
//!   row-major order (the matrices are symmetric), written as a 2-D array.
//!
//! The geometries are `steps/tests/data/`'s, the ones the steps read.
//! Under `FB_BLESS=1`, with `FB_RECORD_CARD` (the card's name) and
//! `FB_RECORD_REV` (the commit) set, the case writes the files:
//!
//! ```text
//! cd rust && FB_BLESS=1 FB_RECORD_CARD="NVIDIA GeForce RTX 3080" \
//!   FB_RECORD_REV=$(git rev-parse --short HEAD) \
//!   cargo test -p fb-pipeline-scf --test steps_reference -- --include-ignored
//! ```
//!
//! Without it, the case holds this layer to the recording at the steps'
//! bars, so a change here that would move the reference fails: 1e-9 Eh on
//! an energy, 1e-12 of the largest element on a matrix (the reasons are
//! `recorded_device.rs`'s), water's energy alone: benzene's and ethylene's,
//! about a second on an RTX 3080, are left to the blessing run.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
#![allow(
    clippy::disallowed_macros,
    reason = "a test's comparisons and the recording's writes, outside any task"
)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag};

/// The bar on an energy, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// The bar on S, H and the metric, relative to the largest element.
const MATRIX_BAR: f64 = 1e-12;

/// The molecules whose energies are recorded; the first is the gates'.
const MOLECULES: [&str; 3] = ["h2o.xyz", "benzene.xyz", "ethylene_meci856.xyz"];

/// A file of `steps/tests/data`.
fn data(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../steps/tests/data")
        .join(file)
}

/// This layer's DF-RHF energy of `xyz` and its iterations, syevd's work
/// asked of a solver handle on device 0, run once there.
fn energy(xyz: &str) -> (f64, usize) {
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

/// CPU: the plan's pinned copies read out, after the executor waited for
/// their downloads.
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

/// Water's S, H and metric on device 0, each with its dimension: the
/// records, the overlap's and the core Hamiltonian's tasks over them, the
/// metric's, and a download of each.
fn matrices() -> [(usize, Vec<f64>); 3] {
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
    let (n, q) = (basis.nao_sphe() as usize, aux.nao_sphe() as usize);
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
    let mut out = std::mem::take(&mut *out.lock().unwrap_or_else(PoisonError::into_inner));
    let metric = out.pop().unwrap();
    let h = out.pop().unwrap();
    let s = out.pop().unwrap();
    [(n, s), (n, h), (q, metric)]
}

/// The matrices' files, in [`matrices`]' order.
const MATRIX_FILES: [&str; 3] = [
    "old_path_h2o_s.npy",
    "old_path_h2o_h.npy",
    "old_path_h2o_metric.npy",
];

/// Record the references under `FB_BLESS`, or hold this layer to them.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_recorded_references_are_this_layers() {
    let bless = std::env::var_os("FB_BLESS").is_some();
    let matrices = matrices();
    if bless {
        for (file, (dim, m)) in MATRIX_FILES.iter().zip(&matrices) {
            fb_io_npy::write_npy(&data(file), &[*dim, *dim], m).unwrap();
        }
        let energies: Vec<_> = MOLECULES
            .iter()
            .map(|xyz| {
                let (e, iterations) = energy(xyz);
                println!("{xyz}: E = {e:.12} in {iterations} iterations");
                serde_json::json!({
                    "xyz": xyz,
                    "basis": "cc-pvdz",
                    "aux_basis": "cc-pvdz-jkfit",
                    "energy": e,
                    "iterations": iterations,
                })
            })
            .collect();
        let record = serde_json::json!({
            "description": "The old Rust path's (planner and arena) DF-RHF energies, hartree, \
                from the SAD guess with schwarz_threshold = 0 at its default tolerances, \
                and its iteration counts; water's S, H and metric beside this file \
                as old_path_h2o_*.npy",
            "provenance": {
                "writer": "rust/pipeline/scf/tests/steps_reference.rs",
                "card": std::env::var("FB_RECORD_CARD").expect("FB_RECORD_CARD names the card"),
                "rev": std::env::var("FB_RECORD_REV").expect("FB_RECORD_REV names the commit"),
            },
            "energies": energies,
        });
        let text = serde_json::to_string_pretty(&record).unwrap() + "\n";
        std::fs::write(data("old_path.json"), text).unwrap();
        return;
    }
    for (file, (dim, m)) in MATRIX_FILES.iter().zip(&matrices) {
        let recorded = fb_io_npy::read_npy(&data(file)).unwrap();
        assert_eq!(recorded.shape, [*dim, *dim], "{file}: the shape");
        let largest = recorded.data.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        let worst = m
            .iter()
            .zip(&recorded.data)
            .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        println!("{file}: max |old - recorded| = {worst:.1e}");
        assert!(worst <= MATRIX_BAR * largest, "{file}: {worst:e}");
    }
    let text = std::fs::read_to_string(data("old_path.json")).unwrap();
    let record: serde_json::Value = serde_json::from_str(&text).unwrap();
    for entry in &record["energies"].as_array().unwrap()[..1] {
        let xyz = entry["xyz"].as_str().unwrap();
        let recorded = entry["energy"].as_f64().unwrap();
        let (e, _) = energy(xyz);
        println!("{xyz}: |E - recorded| = {:.1e}", (e - recorded).abs());
        assert!(
            (e - recorded).abs() < ENERGY_BAR,
            "{xyz}: {e} against {recorded}"
        );
    }
}
