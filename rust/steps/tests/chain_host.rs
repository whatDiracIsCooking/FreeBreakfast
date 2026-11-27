//! The chain with no device: the whole plan budgeted dry through
//! [`fb_steps::run`], with the SCF's row, locks and edges; the SCF refused
//! without the ERI it reads, and its options out of range; and the run as
//! the layer's one maker of an executor.

#![allow(
    clippy::disallowed_macros,
    reason = "a test's file lists, outside any step's run"
)]

mod common;

use std::path::Path;

use fb_core_budget::Backing;
use fb_core_error_handling::ErrorKind;
use fb_steps::{MainBus, ScfTable, plan, run};

/// The dry run on two GPUs is accepted, its table has every step's row and
/// the SCF's locks, and the SCF reads exactly the five steps it declares.
#[test]
fn a_dry_run_budgets_the_whole_chain() {
    let mut config = common::chain("h2o.xyz");
    config.resources.devices = [0, 1].into();
    let run = run(config, Backing::Dry).expect("the plan is accepted");
    for row in [
        "overlap",
        "core_hamiltonian",
        "metric",
        "guess",
        "orthogonalizer",
        "eri",
        "scf",
        "scf: Manager(Eri)",
        "scf: Gpu(1)",
    ] {
        assert!(run.table.contains(row), "no '{row}':\n{}", run.table);
    }
    let edges = run.plan.edges();
    let mut scf = edges.iter().find(|(s, _)| *s == "scf").unwrap().1.clone();
    scf.sort_unstable();
    assert_eq!(
        scf,
        [
            "core_hamiltonian",
            "eri",
            "guess",
            "orthogonalizer",
            "overlap"
        ]
    );
    assert!(run.steps.is_empty(), "a dry run executes nothing");
}

/// `[scf]` without `[eri]` is refused at the plan, naming both steps.
#[test]
fn the_scf_without_the_eri_is_refused() {
    let mut config = common::config("h2o.xyz", "cc-pvdz", None);
    config.scf = Some(ScfTable::default());
    let refused = plan(MainBus::setup(config, Backing::Dry).unwrap())
        .err()
        .expect("scf reads eri")
        .to_string();
    assert!(
        refused.contains("'scf'") && refused.contains("'eri'"),
        "{refused}"
    );
}

/// Each `[scf]` key out of range is refused in budget mode, naming it.
#[test]
fn scf_options_out_of_range_are_refused() {
    type Set = fn(&mut ScfTable);
    let cases: [(Set, &str); 4] = [
        (|t| t.max_iterations = 0, "max_iterations"),
        (|t| t.energy_tolerance = 0.0, "energy_tolerance"),
        (|t| t.diis_history = 0, "diis_history"),
        (|t| t.diis_history = 65, "diis_history is 65"),
    ];
    for (set, key) in cases {
        let mut config = common::chain("h2o.xyz");
        set(config.scf.as_mut().unwrap());
        let error = run(config, Backing::Dry).err().expect("refused");
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{error}");
        assert!(error.to_string().contains(key), "{key}: {error}");
    }
}

/// The layer's executor is the run's: in `fb-steps`' sources and its
/// example, `Executor::new` is called once, in `run`, from `[resources]
/// threads` (`docs/programming_model.md`, "CPU workers"), outside a
/// `#[cfg(test)]` module. A second maker, or one sized otherwise, fails
/// here.
#[test]
fn the_run_makes_the_layers_one_executor() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    let mut dirs = vec![root.join("src"), root.join("examples")];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    let makers: Vec<(String, String)> = files
        .iter()
        .flat_map(|file| {
            let text = std::fs::read_to_string(file).unwrap();
            // A unit test's executor is the harness's, not the layer's.
            let text = text
                .split("#[cfg(test)]")
                .next()
                .unwrap_or_default()
                .to_owned();
            let name = file.strip_prefix(root).unwrap().display().to_string();
            text.lines()
                .filter(|line| line.contains("Executor::new(") && !line.trim().starts_with("//"))
                .map(|line| (name.clone(), line.trim().to_owned()))
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(
        makers,
        [(
            "src/lib.rs".to_owned(),
            "let executor = Executor::new(config.resources.threads);".to_owned()
        )],
        "the executor is made once, in run"
    );
}
