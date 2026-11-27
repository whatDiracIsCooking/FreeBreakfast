//! The run's products on a card (tier 4): water's DF-RHF in cc-pVDZ over
//! cc-pVDZ-JKFIT run and written, then read back.
//!
//! No arithmetic stands between the `ScfOutput`'s host copies and the
//! files, so the bar is bitwise: `c.npy` and `d.npy` hold C's and D's bits
//! (and so their `n_ao × n_ao` shape), and every double of `scf.json`
//! parses back to the same `f64`, serde_json writing its shortest
//! round-trip form and reading it, under `float_roundtrip`, as a correctly
//! rounding reader does. `budget.txt` is the run's table, and
//! `timing.json` names the steps the run timed, in its order. One run, 5.1 s
//! in a debug build on the gfx1200, water's whole chain; it is its family's
//! representative, the one case of the writer, so it is no slow case
//! (`docs/testing.md`, "Slow cases").

mod common;

use fb_core_budget::Backing;
use fb_core_flow::Bus;
use fb_steps::{ScfStep, run};
use serde_json::Value;

/// Whether `a` and `b` hold the same bits.
fn bitwise(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn waters_products_read_back_bitwise() {
    let config = common::chain("h2o.xyz");
    let run = run(config, Backing::Real).expect("the chain runs");
    let dir = std::env::temp_dir().join(format!("fb-steps-write-device-{}", std::process::id()));
    run.write(&dir).unwrap();

    let scf = run.plan.output(ScfStep);
    let dims = &run.plan.bus().shared().dims;
    let n = dims.n_ao;
    for (file, host) in [("c.npy", scf.c()), ("d.npy", scf.d())] {
        let array = fb_io_npy::read_npy(&dir.join(file)).unwrap();
        assert_eq!(array.shape, [n, n], "{file}");
        assert!(bitwise(&array.data, host), "{file} is not the host copy");
    }

    let json: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("scf.json")).unwrap()).unwrap();
    let f = |key: &str| json[key].as_f64().unwrap();
    assert_eq!(f("energy").to_bits(), scf.energy().to_bits());
    assert_eq!(f("residual").to_bits(), scf.residual().to_bits());
    assert_eq!(json["iterations"], scf.iterations());
    assert_eq!(json["converged"], scf.converged());
    assert_eq!(json["n_alpha"], dims.n_alpha);
    assert_eq!(json["n_beta"], dims.n_beta);
    let epsilon: Vec<f64> = json["epsilon"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    assert!(bitwise(&epsilon, scf.epsilon()), "epsilon");
    println!("water: E = {} written as {}", scf.energy(), json["energy"]);

    assert_eq!(
        std::fs::read_to_string(dir.join("budget.txt")).unwrap(),
        run.table
    );
    let timing: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("timing.json")).unwrap()).unwrap();
    let names: Vec<&str> = timing["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    let timed: Vec<&str> = run.steps.iter().map(|s| s.name).collect();
    assert_eq!(names, timed);
    std::fs::remove_dir_all(&dir).unwrap();
}
