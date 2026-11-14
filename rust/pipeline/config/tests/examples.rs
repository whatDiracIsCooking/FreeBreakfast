//! The end-to-end level (`docs/testing.md`, "Levels"): `rust/examples/`
//! run through the `free-breakfast` binary, which is then read back; and
//! the binary's refusals, each its kind's pinned exit code and a message
//! naming the key or the path, with no panic.
//!
//! Each run's working directory is a directory of its own under cargo's
//! `CARGO_TARGET_TMPDIR`, so an example's relative `output` lands there.
//!
//! The overlap's bar is its crate's, fixed by analysis, one run: `max |S -
//! S_ref| <= 1e-12 * max |S_ref|` against PySCF's S (`tests/data/`).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fb_core_error_handling::ErrorKind;
use fb_core_thread_pool::Record;

/// The bar, relative to `max |S_ref|`.
const BAR: f64 = 1e-12;

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

    let planned =
        fb_pipeline_config::plan(&fb_pipeline_config::read_config(&config).unwrap()).unwrap();
    let text = std::fs::read_to_string(cwd.join("overlap/timing.json")).unwrap();
    let record: Record = serde_json::from_str(&text).expect("timing.json parses");
    record.result.unwrap();
    assert_eq!(record.tasks.len(), planned.len(), "one entry per task");
    for (rank, task) in record.tasks.iter().enumerate() {
        assert_eq!(task.node.index(), rank, "{}: in rank order", task.name);
    }
    let names: Vec<&str> = record.tasks.iter().map(|t| t.name.as_str()).collect();
    for write in ["download overlap.npy", "write overlap.npy"] {
        assert!(names.contains(&write), "{write} timed: {names:?}");
    }
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
