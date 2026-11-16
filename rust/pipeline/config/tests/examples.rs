//! The end-to-end level (`docs/testing.md`, "Levels"): `rust/examples/`
//! run through the `free-breakfast` binary, which is then read back; and
//! the binary's refusals, each its kind's pinned exit code and a message
//! naming the key or the path, with no panic.
//!
//! Each run's working directory is a directory of its own under cargo's
//! `CARGO_TARGET_TMPDIR`, so an example's relative `output` lands there.
//!
//! The overlap's bar is its crate's, fixed by analysis, one run: `max |S -
//! S_ref| <= 1e-12 * max |S_ref|` against PySCF's S (`tests/data/`). The
//! guess's is bitwise: its `d.npy` against `fb-sad`'s serial density, the
//! task form placing the same stamps in the same order, and a run loading
//! that file writing the same bytes, since a `.npy` read and written back
//! is its bytes. The DF-RHF's is `fb-pipeline-scf`'s, fixed by analysis
//! there (`tests/scf.rs`), one run: `|E - E_ref| <= 1e-9` hartree against
//! PySCF's DF-RHF energy of the same water, basis and fitting set, its
//! committed reference; `scf.json` writes E at full precision, so the
//! file adds nothing to the miss.
//!
//! The localization's is fixed by analysis in
//! `tests/data/localization_reference.py`, one run: `|L - L_ref| <= sens
//! τ_c + τ_L + τ_ref` against PySCF's Pipek-Mezey maximum `L_ref` of its
//! DF-RHF of the same water, basis and fitting set, at exponent 2. The
//! tree's SCF stops at a residual under `τ_c`, the default
//! `commutator_tolerance` 1e-7, and the maximum moves with the occupied
//! space it publishes at first order, by at most `sens` per unit of
//! residual (the reference's sensitivity, about 0.13, so 1.3e-8). Each
//! localization stops within its last sweep's change of the maximum, both
//! converging faster than halving it there: `τ_L`, the example's
//! `functional_tolerance` 1e-10, and `τ_ref`, PySCF's `conv_tol` 1e-12.
//! PySCF's own SCF, at `conv_tol_grad` 1e-10, and the fitted integrals the
//! two codes share add under 1e-11.

mod common;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fb_core_error_handling::ErrorKind;
use fb_core_thread_pool::Record;

/// The bar, relative to `max |S_ref|`.
const BAR: f64 = 1e-12;

/// The DF-RHF energy bar, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// The SCF's default `commutator_tolerance`, the residual its orbitals are
/// published under.
const COMMUTATOR_TOLERANCE: f64 = 1e-7;

/// The localization example's `functional_tolerance` and PySCF's
/// `conv_tol` in the reference, each a bound on its code's distance from
/// the maximum.
const LOCALIZATION_TOLERANCES: f64 = 1e-10 + 1e-12;

/// `rust/examples/<file>`.
fn example(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .join(file)
}

/// A fresh, empty working directory named `name`.
fn workdir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `free-breakfast <config>` run in `cwd`, its output printed.
fn free_breakfast(cwd: &Path, config: &Path) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_free-breakfast"))
        .arg(config)
        .current_dir(cwd)
        .output()
        .expect("free-breakfast starts");
    println!("{}", String::from_utf8_lossy(&out.stdout));
    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    out
}

/// The example overlap through the binary: exit 0, its S against PySCF's,
/// and its `timing.json` one entry per task of the plan, the write nodes
/// among them. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn overlap_example() {
    let cwd = workdir("example_overlap");
    let config = example("overlap.toml");
    let out = free_breakfast(&cwd, &config);
    assert_eq!(out.status.code(), Some(0), "the example exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("completed: 0 errors"), "{stdout}");

    let s = fb_io_npy::read_npy(&cwd.join("overlap/overlap.npy")).unwrap();
    let want = fb_io_npy::read_npy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/h2o_cc-pvdz.npy"),
    )
    .unwrap();
    assert_eq!(s.shape, want.shape, "the AO count");
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = s
        .data
        .iter()
        .zip(&want.data)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!(
        "n_ao {}, max |S - S_ref| = {worst:e}, bar {:e}",
        s.shape[0],
        BAR * scale
    );
    assert!(worst <= BAR * scale, "max |S - S_ref| = {worst:e}");

    check_timing(
        &config,
        &cwd.join("overlap/timing.json"),
        &["download overlap.npy", "write overlap.npy"],
    );
}

/// The example under raised `[memory]` caps through the binary: exit 0,
/// its S against PySCF's at the overlap's bar, and its log naming worker
/// 0's device and pinned blocks admitted, the pinned one against the
/// example's 4 GiB, and the plan's heap block against its 32 GiB beside
/// the node's host blocks: the binary admits every table before it runs.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn memory_example() {
    let cwd = workdir("example_memory");
    let config = example("memory.toml");
    let out = free_breakfast(&cwd, &config);
    assert_eq!(out.status.code(), Some(0), "the example exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("completed: 0 errors"), "{stdout}");
    assert!(
        stdout.contains("gpu0 admitted: a device block of"),
        "{stdout}"
    );
    assert!(
        stdout.contains("a pinned block of") && stdout.contains("against a cap of 4294967296"),
        "{stdout}"
    );
    assert!(
        stdout.contains("plan admitted: a heap block of 0 bytes against a cap of 34359738368"),
        "{stdout}"
    );

    let s = fb_io_npy::read_npy(&cwd.join("memory/overlap.npy")).unwrap();
    let want = fb_io_npy::read_npy(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/h2o_cc-pvdz.npy"),
    )
    .unwrap();
    assert_eq!(s.shape, want.shape, "the AO count");
    let scale = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = s
        .data
        .iter()
        .zip(&want.data)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!("max |S - S_ref| = {worst:e}, bar {:e}", BAR * scale);
    assert!(worst <= BAR * scale, "max |S - S_ref| = {worst:e}");
    check_timing(
        &config,
        &cwd.join("memory/timing.json"),
        &["download overlap.npy", "write overlap.npy"],
    );
}

/// The timing record `timing` read back, parsed as the timing record's
/// type: every task of `config`'s plan timed, in rank order and then
/// iteration; a loop's body node and decision once per iteration, 0 to its
/// loop's count; every other node once; each of `named` among them. Returns
/// the record. The plan is laid out at a stand-in syevd work: the tasks do
/// not depend on it.
fn check_timing(config: &Path, timing: &Path, named: &[&str]) -> Record {
    let planned =
        fb_pipeline_config::plan(&fb_pipeline_config::read_config(config).unwrap(), |_| Ok(1))
            .unwrap();
    let text = std::fs::read_to_string(timing).unwrap();
    let record: Record = serde_json::from_str(&text).expect("timing.json parses");
    record.result.as_ref().unwrap();
    let mut entries = std::collections::BTreeMap::<usize, Vec<Option<usize>>>::new();
    let mut previous = None;
    for task in &record.tasks {
        let at = (task.node.index(), task.iteration);
        assert!(
            previous < Some(at),
            "{}: in rank and iteration order",
            task.name
        );
        previous = Some(at);
        entries.entry(at.0).or_default().push(task.iteration);
    }
    assert_eq!(entries.len(), planned.len(), "every task timed");
    for (node, iterations) in &entries {
        if iterations[0].is_some() {
            // A loop is a contiguous run of nodes ending at its decision, so
            // a body node's loop is the first decision at or after it.
            let count = record
                .iterations
                .iter()
                .find(|(decision, _)| decision.index() >= *node)
                .map(|(_, &count)| count)
                .expect("an iterated node is in a loop");
            let want: Vec<_> = (0..count).map(Some).collect();
            assert_eq!(iterations, &want, "node {node}: one entry per iteration");
        } else {
            assert_eq!(iterations, &[None], "node {node}: one entry");
        }
    }
    let names: Vec<&str> = record.tasks.iter().map(|t| t.name.as_str()).collect();
    for n in named {
        assert!(names.contains(n), "{n} timed: {names:?}");
    }
    record
}

/// The example guess through the binary: exit 0, its `d.npy` bitwise
/// `fb-sad`'s serial SAD density of water in cc-pVDZ (the compiled-in
/// stamps), its `timing.json` one entry per task, the stamp nodes and the
/// write node among them. Then a second run, `[guess] from` that file,
/// writes the same bytes. The binary opens device 0 for any plan, so it
/// needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn guess_example() {
    let cwd = workdir("example_guess");
    let config = example("guess.toml");
    let out = free_breakfast(&cwd, &config);
    assert_eq!(out.status.code(), Some(0), "the example exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("completed: 0 errors"), "{stdout}");

    let written = cwd.join("guess/d.npy");
    let d = fb_io_npy::read_npy(&written).unwrap();
    let molecule = fb_io_xyz::read_xyz(&example("water.xyz"), 0, 1).unwrap();
    let basis = fb_io_basis::resolve_basis("cc-pvdz")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    let want = fb_sad::sad_density(
        &basis,
        molecule.atoms(),
        Some(fb_io_sad::builtin_sad_density),
    )
    .unwrap();
    let n = basis.nao_sphe() as usize;
    assert_eq!(d.shape, [n, n], "nao_sphe square");
    let differ = d
        .data
        .iter()
        .zip(&want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    println!("n_ao {n}, elements differing from the serial density: {differ}");
    assert_eq!(differ, 0, "d.npy is the serial SAD density bitwise");
    check_timing(
        &config,
        &cwd.join("guess/timing.json"),
        &["baked stamp O", "baked stamp H", "sad guess", "write d.npy"],
    );

    // The second run loads the first's d.npy, from a config of its own.
    let loaded = cwd.join("load.toml");
    std::fs::copy(example("water.xyz"), cwd.join("water.xyz")).unwrap();
    std::fs::write(
        &loaded,
        "calculation = \"guess\"\noutput = \"loaded\"\n\n[molecule]\nxyz = \"water.xyz\"\n\n\
         [basis_set]\nbasis = \"cc-pvdz\"\n\n[guess]\nfrom = \"guess/d.npy\"\n",
    )
    .unwrap();
    let out = free_breakfast(&cwd, &loaded);
    assert_eq!(out.status.code(), Some(0), "the loading run exits 0");
    let again = std::fs::read(cwd.join("loaded/d.npy")).unwrap();
    assert!(
        again == std::fs::read(&written).unwrap(),
        "the loaded run writes the same bytes"
    );
    check_timing(
        &loaded,
        &cwd.join("loaded/timing.json"),
        &["load guess", "write d.npy"],
    );
}

/// What a DF-RHF run wrote into `dir`: `scf.json` parsed, its residual
/// under the default tolerance and its orbital energies ascending, one per
/// AO; returns its energy and iteration count.
fn scf_json(dir: &Path) -> (f64, usize) {
    let text = std::fs::read_to_string(dir.join("scf.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).expect("scf.json parses");
    let energy = json["energy"].as_f64().expect("energy");
    let iterations = json["iterations"].as_u64().expect("iterations") as usize;
    assert!(
        json["residual"].as_f64().expect("residual") < 1e-7,
        "{text}"
    );
    let epsilon: Vec<f64> = json["epsilon"]
        .as_array()
        .expect("epsilon")
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    let n = fb_io_npy::read_npy(&dir.join("c.npy")).unwrap().shape[0];
    assert_eq!(epsilon.len(), n, "one orbital energy per AO");
    assert!(epsilon.is_sorted(), "ascending: {epsilon:?}");
    (energy, iterations)
}

/// The example DF-RHF through the binary: exit 0, its `scf.json` energy
/// within the bar of PySCF's, `c.npy` and `d.npy` square, its
/// `timing.json` the timing record with the loop's entries per iteration,
/// the write nodes among them. Then a second run, `[guess] from` the
/// first's `d.npy`, converges at the decision's first chance, its second
/// iteration: from the converged density the first build's residual is
/// already under the tolerance, and the second's energy moves by the
/// round-off of one more Roothaan step. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn rhf_example() {
    let cwd = workdir("example_rhf");
    let config = example("rhf.toml");
    let out = free_breakfast(&cwd, &config);
    assert_eq!(out.status.code(), Some(0), "the example exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("completed: 0 errors"), "{stdout}");

    let dir = cwd.join("rhf");
    let (energy, iterations) = scf_json(&dir);
    let e_ref = fb_io_npy::read_npy(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../scf/tests/data/scf_h2o_cc-pvdz_cc-pvdz-jkfit.npy"),
    )
    .unwrap()
    .data[0];
    let miss = (energy - e_ref).abs();
    println!("E = {energy:.12} against {e_ref:.12}, |dE| = {miss:e}, {iterations} iterations");
    assert!(miss <= ENERGY_BAR, "|dE| = {miss:e} over {ENERGY_BAR:e}");
    for m in ["c.npy", "d.npy"] {
        let a = fb_io_npy::read_npy(&dir.join(m)).unwrap();
        assert_eq!(a.shape, [24, 24], "{m}: water's cc-pVDZ AOs");
    }
    let record = check_timing(
        &config,
        &dir.join("timing.json"),
        &[
            "converged",
            "write c.npy",
            "write d.npy",
            "download epsilon",
            "write scf.json",
        ],
    );
    assert_eq!(
        record.iterations.values().copied().sum::<usize>(),
        iterations,
        "scf.json's count is the loop's"
    );

    // The second run loads the first's d.npy.
    let loaded = cwd.join("load.toml");
    std::fs::copy(example("water.xyz"), cwd.join("water.xyz")).unwrap();
    std::fs::write(
        &loaded,
        std::fs::read_to_string(&config)
            .unwrap()
            .replace("output = \"rhf\"", "output = \"loaded\"")
            .replace("type = \"sad\"", "from = \"rhf/d.npy\""),
    )
    .unwrap();
    let out = free_breakfast(&cwd, &loaded);
    assert_eq!(out.status.code(), Some(0), "the loading run exits 0");
    let (again, iterations) = scf_json(&cwd.join("loaded"));
    println!("from d.npy: E = {again:.12}, {iterations} iterations");
    assert_eq!(iterations, 2, "converged at the decision's first chance");
    assert!((again - e_ref).abs() <= ENERGY_BAR, "E = {again:.12}");
    check_timing(
        &loaded,
        &cwd.join("loaded/timing.json"),
        &["load guess", "write scf.json"],
    );
}

/// The example DF-ROHF through the binary, the water cation's doublet:
/// exit 0, its `scf.json` energy within the DF-RHF's bar of PySCF's DF-ROHF
/// energy and every orbital energy within `fb-pipeline-scf`'s 1e-6 hartree
/// (its `tests/rohf.rs` has both bars' analysis), against its copy of the
/// C++ tree's reference (`pipeline/scf/tests/data/df_rohf.json`, the case
/// `h2o_cation_ccpvdz`, whose geometry `water_cation.xyz` carries in
/// Angstrom); `n_alpha` 5 and `n_beta` 4; `c.npy` and `d.npy` square; and
/// its `timing.json` the timing record, the open shell's nodes among it.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn rohf_example() {
    let cwd = workdir("example_rohf");
    let config = example("rohf.toml");
    let out = free_breakfast(&cwd, &config);
    assert_eq!(out.status.code(), Some(0), "the example exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("completed: 0 errors"), "{stdout}");

    let dir = cwd.join("rohf");
    let (energy, iterations) = scf_json(&dir);
    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../scf/tests/data/df_rohf.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let case = reference["df_rohf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "h2o_cation_ccpvdz")
        .expect("the water cation is in the reference");
    let e_ref = case["energy"].as_f64().unwrap();
    let miss = (energy - e_ref).abs();
    println!("E = {energy:.12} against {e_ref:.12}, |dE| = {miss:e}, {iterations} iterations");
    assert!(miss <= ENERGY_BAR, "|dE| = {miss:e} over {ENERGY_BAR:e}");
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("scf.json")).unwrap()).unwrap();
    assert_eq!(
        (json["n_alpha"].as_u64(), json["n_beta"].as_u64()),
        (Some(5), Some(4)),
        "the doublet's occupations"
    );
    let worst = json["epsilon"]
        .as_array()
        .unwrap()
        .iter()
        .zip(case["mo_energies"].as_array().unwrap())
        .fold(0.0_f64, |m, (g, w)| {
            m.max((g.as_f64().unwrap() - w.as_f64().unwrap()).abs())
        });
    println!("max |d epsilon| = {worst:e}");
    assert!(worst <= 1e-6, "max |d epsilon| = {worst:e}");
    for m in ["c.npy", "d.npy"] {
        let a = fb_io_npy::read_npy(&dir.join(m)).unwrap();
        assert_eq!(a.shape, [24, 24], "{m}: water's cc-pVDZ AOs");
    }
    check_timing(
        &config,
        &dir.join("timing.json"),
        &[
            "guess rescale",
            "spin densities",
            "roothaan fock",
            "open density",
            "publish open",
            "write scf.json",
        ],
    );
}

/// The example localization through the binary: exit 0, its `scf.json`
/// energy at the DF-RHF's bar, its `localization.json` L after within the
/// bar of PySCF's maximum (the module's doc), L before under it and a
/// sweep count within the bound, `c_local.npy` water's 24 AOs by its five
/// occupied orbitals, in ascending `<i|F|i>` as the C++ step sorts them
/// (S from the overlap example, run beside it), and its `timing.json` the
/// timing record with the loops' entries, the write nodes among them.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn localization_example() {
    let cwd = workdir("example_localization");
    let config = example("localization.toml");
    let out = free_breakfast(&cwd, &config);
    assert_eq!(out.status.code(), Some(0), "the example exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("completed: 0 errors"), "{stdout}");

    let dir = cwd.join("localization");
    let (energy, _) = scf_json(&dir);
    let e_ref = fb_io_npy::read_npy(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../scf/tests/data/scf_h2o_cc-pvdz_cc-pvdz-jkfit.npy"),
    )
    .unwrap()
    .data[0];
    assert!((energy - e_ref).abs() <= ENERGY_BAR, "E = {energy:.12}");

    let text = std::fs::read_to_string(dir.join("localization.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).expect("localization.json parses");
    let before = json["functional_before"]
        .as_f64()
        .expect("functional_before");
    let after = json["functional_after"].as_f64().expect("functional_after");
    let sweeps = json["sweeps"].as_u64().expect("sweeps");
    let [l_ref, sens] = fb_io_npy::read_npy(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/localization_h2o_cc-pvdz_cc-pvdz-jkfit.npy"),
    )
    .unwrap()
    .data[..] else {
        panic!("the reference is [L, sens]")
    };
    let bar = sens * COMMUTATOR_TOLERANCE + LOCALIZATION_TOLERANCES;
    let miss = (after - l_ref).abs();
    println!(
        "L {before:.12} -> {after:.12} in {sweeps} sweeps, against {l_ref:.12}: |dL| = \
         {miss:e}, bar {bar:e}"
    );
    assert!(miss <= bar, "|dL| = {miss:e} over {bar:e}");
    assert!(before < after, "{text}");
    assert!((1..=64).contains(&sweeps), "{text}");

    let c_local = fb_io_npy::read_npy(&dir.join("c_local.npy")).unwrap();
    assert_eq!(
        c_local.shape,
        [24, 5],
        "water's cc-pVDZ AOs by its occupied"
    );
    // Its orbitals ascend in <i|F|i>, over the same water's S.
    let overlap = free_breakfast(&cwd, &example("overlap.toml"));
    assert_eq!(
        overlap.status.code(),
        Some(0),
        "the overlap example exits 0"
    );
    let s = fb_io_npy::read_npy(&cwd.join("overlap").join("overlap.npy")).unwrap();
    let c = fb_io_npy::read_npy(&dir.join("c.npy")).unwrap();
    let text = std::fs::read_to_string(dir.join("scf.json")).unwrap();
    let scf: serde_json::Value = serde_json::from_str(&text).expect("scf.json parses");
    let eps: Vec<f64> = scf["epsilon"]
        .as_array()
        .expect("epsilon")
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    let energies = common::fock_expectations(&c.data, &s.data, &c_local.data, &eps);
    println!("<i|F|i> = {energies:?}");
    assert!(
        common::ascending(&energies),
        "<i|F|i> {energies:?} does not ascend"
    );
    check_timing(
        &config,
        &dir.join("timing.json"),
        &[
            "write scf.json",
            "pm converged",
            "pm rotate",
            "write c_local.npy",
            "write localization.json",
        ],
    );
}

/// A config the binary runs and the run fails: exit code `kind`'s, no
/// panic, the log naming every one of `named`, and `timing.json` written
/// with the error.
fn failed(name: &str, config: &str, xyz: &str, kind: ErrorKind, named: &[&str]) {
    let cwd = workdir(name);
    std::fs::write(cwd.join("m.xyz"), xyz).unwrap();
    let path = cwd.join("config.toml");
    std::fs::write(&path, config).unwrap();
    let out = free_breakfast(&cwd, &path);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(!stderr.contains("panicked"), "{name}: a panic: {stderr}");
    assert_eq!(
        out.status.code(),
        Some(kind.exit_code()),
        "{name}: {kind:?}'s code"
    );
    assert!(stdout.contains("completed: 1 errors"), "{name}: {stdout}");
    for n in named {
        assert!(stdout.contains(n), "{name}: names '{n}': {stdout}");
    }
    let text = std::fs::read_to_string(cwd.join("fbf_output/timing.json")).unwrap();
    let record: Record = serde_json::from_str(&text).expect("timing.json parses");
    assert_eq!(record.result.expect_err(name).kind, kind, "{name}");
}

/// A DF-RHF config of water in STO-3G over cc-pVDZ-JKFIT, `m.xyz` its
/// geometry, `extra` appended.
fn rhf_config(extra: &str) -> String {
    format!(
        "calculation = \"rhf\"\n[molecule]\nxyz = \"m.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n\
         [eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n{extra}"
    )
}

/// Two Fock builds are not enough from the SAD guess: the loop's bound
/// fails the run, `MaxIterationsExceeded`, its pinned code 6. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn too_few_iterations_exits_max_iterations_exceeded() {
    failed(
        "failed_max_iterations",
        &rhf_config("[scf]\nmax_iterations = 2\n"),
        &std::fs::read_to_string(example("water.xyz")).unwrap(),
        ErrorKind::MaxIterationsExceeded,
        &["did not stop in 2 iterations"],
    );
}

/// Two nuclei at one point: every fitting function is there twice, so the
/// metric `(P|Q)` is singular in exact arithmetic. The duplicate's pivot in
/// the Cholesky factor is the round-off of `M - (M/√M)²`, of either sign:
/// a pivot that is not positive fails the factor at once, and a positive
/// one, of order `ε M`, divides the rows below it by its root, so a later
/// duplicate's pivot is not positive. The run fails, `Numerical`, its
/// pinned code 8, before the loop. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_singular_metric_exits_numerical() {
    failed(
        "failed_numerical",
        &rhf_config(""),
        "2\nH2 at one point\nH 0 0 0\nH 0 0 0\n",
        ErrorKind::Numerical,
        &["not positive"],
    );
}

/// A config the binary refuses, written into a working directory of its
/// own beside an XYZ file, run, and checked: exit code `kind`'s, no panic,
/// and the log naming every one of `named`.
fn refused(name: &str, config: &str, kind: ErrorKind, named: &[&str]) {
    let cwd = workdir(name);
    std::fs::copy(example("water.xyz"), cwd.join("water.xyz")).unwrap();
    let path = cwd.join("config.toml");
    std::fs::write(&path, config).unwrap();
    let out = free_breakfast(&cwd, &path);
    let (stdout, stderr) = (
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    assert!(!stderr.contains("panicked"), "{name}: a panic: {stderr}");
    assert_eq!(
        out.status.code(),
        Some(kind.exit_code()),
        "{name}: {kind:?}'s code"
    );
    assert!(stdout.contains("completed: 1 errors"), "{name}: {stdout}");
    for n in named {
        assert!(stdout.contains(n), "{name}: names '{n}': {stdout}");
    }
    assert!(
        !cwd.join("fbf_output").exists(),
        "{name}: refused before the output directory is made"
    );
}

/// An unknown key: `InvalidConfig`, naming the key, the file and its line.
#[test]
fn an_unknown_key_is_refused() {
    let config = "calculation = \"overlap\"\n\n[molecule]\nxyz = \"water.xyz\"\ncolour = \"blue\"\n\n[basis_set]\nbasis = \"sto-3g\"\n";
    refused(
        "refused_unknown_key",
        config,
        ErrorKind::InvalidConfig,
        &["unknown field `colour`", "config.toml:5:"],
    );
}

/// A geometry file that is not there: `Io`, naming the key and the path.
#[test]
fn a_missing_geometry_is_refused() {
    let config = "calculation = \"overlap\"\n\n[molecule]\nxyz = \"missing.xyz\"\n\n[basis_set]\nbasis = \"sto-3g\"\n";
    refused(
        "refused_missing_geometry",
        config,
        ErrorKind::Io,
        &["[molecule] xyz", "missing.xyz"],
    );
}

/// A basis that is neither a compiled-in name nor a file: `Io`, naming the
/// key and the path it was taken for.
#[test]
fn an_unknown_basis_is_refused() {
    let config = "calculation = \"overlap\"\n\n[molecule]\nxyz = \"water.xyz\"\n\n[basis_set]\nbasis = \"no-such-basis\"\n";
    refused(
        "refused_unknown_basis",
        config,
        ErrorKind::Io,
        &["[basis_set] basis", "no-such-basis"],
    );
}

/// A loaded guess whose file is not there: `Io`, naming the key and the
/// path, at planning.
#[test]
fn a_missing_guess_file_is_refused() {
    let config = "calculation = \"guess\"\n\n[molecule]\nxyz = \"water.xyz\"\n\n[basis_set]\nbasis = \"sto-3g\"\n\n[guess]\nfrom = \"missing.npy\"\n";
    refused(
        "refused_missing_guess",
        config,
        ErrorKind::Io,
        &["[guess] from", "missing.npy"],
    );
}

/// A loaded guess of the wrong shape, water's STO-3G count less one:
/// `InvalidConfig`, naming the key and the path, at planning.
#[test]
fn a_guess_file_of_the_wrong_shape_is_refused() {
    let dir = workdir("refused_wrong_guess_file");
    let wrong = dir.join("wrong.npy");
    fb_io_npy::write_npy(&wrong, &[6, 6], &[0.0; 36]).unwrap();
    let config = format!(
        "calculation = \"guess\"\n\n[molecule]\nxyz = \"water.xyz\"\n\n[basis_set]\nbasis = \"sto-3g\"\n\n[guess]\nfrom = \"{}\"\n",
        wrong.display()
    );
    refused(
        "refused_wrong_guess",
        &config,
        ErrorKind::InvalidConfig,
        &["[guess] from", &wrong.display().to_string(), "is not 7 x 7"],
    );
}

/// Water's DF-RHF at a charge and multiplicity of the wrong parity, +1 as
/// a singlet: `InvalidConfig`, naming the count and the multiplicity, at
/// planning.
#[test]
fn a_multiplicity_of_the_wrong_parity_is_refused() {
    refused(
        "refused_wrong_parity",
        "calculation = \"rhf\"\n[molecule]\nxyz = \"water.xyz\"\ncharge = 1\n\
         [basis_set]\nbasis = \"sto-3g\"\n[eri]\nmode = \"density_fitting\"\n\
         aux_basis = \"cc-pvdz-jkfit\"\n",
        ErrorKind::InvalidConfig,
        &["9 electrons", "multiplicity 1"],
    );
}

/// The overlap of water in STO-3G, `memory` its `[memory]` table.
fn capped(memory: &str) -> String {
    format!(
        "calculation = \"overlap\"\n\n[molecule]\nxyz = \"water.xyz\"\n\n[basis_set]\nbasis = \
         \"sto-3g\"\n\n[memory]\n{memory}\n"
    )
}

/// A device cap under its floor: `InvalidConfig`, naming the key and the
/// value, before the output directory is made.
#[test]
fn a_cap_under_its_floor_is_refused() {
    refused(
        "refused_cap_under_floor",
        &capped("max_device_mem_per_gpu = \"4GiB\""),
        ErrorKind::InvalidConfig,
        &[
            "[memory] max_device_mem_per_gpu = 4294967296 bytes",
            "under its floor",
        ],
    );
}

/// A pinned cap past the node's `MemAvailable`: the binary's admission
/// refuses it, `InvalidConfig`, naming the key and `MemAvailable`, after
/// planning and before the output directory is made or a block allocated.
#[test]
fn a_pinned_cap_past_the_node_is_refused() {
    refused(
        "refused_pinned_past_node",
        &capped("max_pinned_mem_per_gpu = \"8000TiB\""),
        ErrorKind::InvalidConfig,
        &["max_pinned_mem_per_gpu", "MemAvailable"],
    );
}
