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

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fb_core_error_handling::ErrorKind;
use fb_core_thread_pool::Record;

/// The bar, relative to `max |S_ref|`.
const BAR: f64 = 1e-12;

/// The DF-RHF energy bar, hartree.
const ENERGY_BAR: f64 = 1e-9;

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

/// The timing record `timing` read back, parsed as the timing record's
/// type: every task of `config`'s plan timed, in rank order and then
/// iteration; a loop's body node and decision once per iteration, 0 to the
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
    // One loop at most in these plans.
    let count = record.iterations.values().copied().sum::<usize>();
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
            "download scf.json",
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
    fb_io_npy::write_npy(
        &wrong,
        &fb_io_npy::NpyArray {
            shape: vec![6, 6],
            data: vec![0.0; 36],
        },
    )
    .unwrap();
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
