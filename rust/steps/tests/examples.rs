//! The end-to-end level (`docs/testing.md`, "Levels"): every config of
//! `rust/examples/` run through the `free-breakfast` binary, its products
//! read back; and the binary's refusals and a failed run, each its kind's
//! exit code and `completed: 1 errors`, with no panic.
//!
//! Each run's working directory is a directory of its own under cargo's
//! `CARGO_TARGET_TMPDIR`, its products in `--output out` there.
//!
//! **The bars, by analysis, one run, each the steps' own test's.** S:
//! `max |S - S_ref| <= 1e-12 max |S_ref|` against the old path's recorded
//! S of the same water (`tests/data/old_path_h2o_s.npy`, `recorded_device.rs`'s
//! bar: the same integrals, a few ulps of the largest). The guess:
//! bitwise, its `d.npy` against `fb-sad`'s serial density (the same stamps
//! placed in the same order), and a run loading that file writing the same
//! bytes, since a `.npy` read and written back is its bytes. The DF-RHF:
//! `|E - E_ref| <= 1e-9` hartree against `chain_device.rs`'s recorded water
//! energy, its bar and its analysis; `scf.json` writes E in its shortest
//! round-trip form, so the file adds nothing to the miss. The DF-ROHF: the
//! same bar against the C++ tree's PySCF reference of the cation
//! (`tests/data/df_rohf.json`, `rohf_device.rs`'s), and its orbital
//! energies at `rohf_device.rs`'s 1e-6 hartree. The localization's
//! functional is held to PySCF's by `localization_device.rs`; here its
//! products are read back and its SCF energy held to the DF-RHF's bar.

#![allow(
    clippy::disallowed_macros,
    reason = "a test's case lists and its readings back, outside any step's run"
)]

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use fb_core_error_handling::ErrorKind;

use common::{ENERGY_BAR, data};

/// The bar on S, relative to its largest element (`recorded_device.rs`).
const MATRIX_BAR: f64 = 1e-12;

/// The bar on an ROHF orbital energy, hartree (`rohf_device.rs`).
const EPSILON_BAR: f64 = 1e-6;

/// Water's DF-RHF energy in cc-pVDZ over cc-pVDZ-JKFIT, `chain_device.rs`'s
/// recorded one: the geometry of `rust/examples/water.xyz` is
/// `tests/data/h2o.xyz`'s.
const WATER_ENERGY: f64 = -76.026_751_140_543;

/// Every config of `rust/examples/`, by file name: [`the_examples_are_listed`]
/// fails on one missing here.
const EXAMPLES: [&str; 7] = [
    "overlap.toml",
    "guess.toml",
    "rhf.toml",
    "rohf.toml",
    "localization.toml",
    "resources.toml",
    "two_gpus.toml",
];

/// `rust/examples/<file>`.
fn example(file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../examples")
        .join(file)
}

/// A fresh, empty working directory named `name`.
fn workdir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `free-breakfast <config> --output out` run in `cwd`, its output printed.
fn free_breakfast(cwd: &Path, config: &Path) -> Output {
    let out = Command::new(env!("CARGO_BIN_EXE_free-breakfast"))
        .arg(config)
        .args(["--output", "out"])
        .current_dir(cwd)
        .output()
        .expect("free-breakfast starts");
    println!("{}", String::from_utf8_lossy(&out.stdout));
    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    out
}

/// `config` through the binary in a fresh `name` directory: exit 0, no
/// error logged, the record written. Returns the output directory.
fn ran(name: &str, config: &Path) -> PathBuf {
    let cwd = workdir(name);
    let out = free_breakfast(&cwd, config);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{name} exits 0");
    assert!(stdout.contains("completed: 0 errors"), "{name}: {stdout}");
    let dir = cwd.join("out");
    for file in ["budget.txt", "timing.json"] {
        assert!(dir.join(file).exists(), "{name}: {file} is written");
    }
    dir
}

/// The overlap example: S against the recorded S, at the bar, and nothing
/// of the SCF written.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn overlap_example() {
    let dir = ran("example_overlap", &example("overlap.toml"));
    let s = fb_io_npy::read_npy(&dir.join("overlap.npy")).unwrap();
    let want = fb_io_npy::read_npy(&data("old_path_h2o_s.npy")).unwrap();
    assert_eq!(s.shape, want.shape, "the AO count");
    let largest = want.data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = s
        .data
        .iter()
        .zip(&want.data)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!(
        "max |S - S_ref| = {worst:e}, bar {:e}",
        MATRIX_BAR * largest
    );
    assert!(largest > 0.1 && worst <= MATRIX_BAR * largest, "{worst:e}");
    assert!(!dir.join("scf.json").exists() && !dir.join("d.npy").exists());
}

/// The guess example: `d.npy` the serial SAD density bitwise; then a run
/// loading it with `[guess] from` writes the same bytes.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn guess_example() {
    let dir = ran("example_guess", &example("guess.toml"));
    let written = dir.join("d.npy");
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
    assert!(!dir.join("overlap.npy").exists());

    let cwd = workdir("example_guess_input");
    std::fs::copy(example("water.xyz"), cwd.join("water.xyz")).unwrap();
    let loaded = cwd.join("load.toml");
    std::fs::write(
        &loaded,
        format!(
            "calculation = \"guess\"\n[molecule]\nxyz = \"water.xyz\"\n[basis]\n\
             name = \"cc-pvdz\"\n[guess]\nfrom = \"{}\"\n",
            written.display()
        ),
    )
    .unwrap();
    let again = ran("example_guess_loaded", &loaded);
    assert!(
        std::fs::read(again.join("d.npy")).unwrap() == std::fs::read(&written).unwrap(),
        "the loaded run writes the same bytes"
    );
}

/// What a DF-RHF run wrote into `dir`: `scf.json` parsed, its residual
/// under the default tolerance and its orbital energies ascending, one per
/// AO, and `c.npy` and `d.npy` water's 24 AOs square.
fn scf_json(dir: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(dir.join("scf.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).expect("scf.json parses");
    assert!(json["residual"].as_f64().unwrap() < 1e-7, "{text}");
    let epsilon: Vec<f64> = json["epsilon"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    assert_eq!(epsilon.len(), 24, "one orbital energy per AO");
    assert!(epsilon.is_sorted(), "ascending: {epsilon:?}");
    for m in ["c.npy", "d.npy"] {
        let a = fb_io_npy::read_npy(&dir.join(m)).unwrap();
        assert_eq!(a.shape, [24, 24], "{m}: water's cc-pVDZ AOs");
    }
    json
}

/// `scf.json`'s energy in `dir` within the bar of `want`.
fn energy_at_bar(dir: &Path, want: f64) -> serde_json::Value {
    let json = scf_json(dir);
    let energy = json["energy"].as_f64().unwrap();
    let miss = (energy - want).abs();
    println!(
        "E = {energy:.12} against {want:.12}, |dE| = {miss:e}, {} iterations",
        json["iterations"]
    );
    assert!(miss <= ENERGY_BAR, "|dE| = {miss:e} over {ENERGY_BAR:e}");
    json
}

/// The DF-RHF example: its energy at the bar of the recorded one.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn rhf_example() {
    let dir = ran("example_rhf", &example("rhf.toml"));
    energy_at_bar(&dir, WATER_ENERGY);
}

/// The resources example, the DF-RHF under written caps and a 256 MiB ERI
/// buffer: the same energy at the same bar.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn resources_example() {
    let dir = ran("example_resources", &example("resources.toml"));
    energy_at_bar(&dir, WATER_ENERGY);
}

/// The DF-ROHF example: the cation's energy and orbital energies at their
/// bars against the reference, its occupations 5 and 4.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn rohf_example() {
    let dir = ran("example_rohf", &example("rohf.toml"));
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(data("df_rohf.json")).unwrap()).unwrap();
    let case = reference["df_rohf"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "h2o_cation_ccpvdz")
        .expect("the water cation is in the reference");
    let json = energy_at_bar(&dir, case["energy"].as_f64().unwrap());
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
    assert!(worst <= EPSILON_BAR, "max |d epsilon| = {worst:e}");
}

/// The localization example: the SCF's products at the DF-RHF's bar, then
/// `localization.json` (L rising, the sweeps within the bound) and
/// `c_local.npy`, water's 24 AOs by its five occupied orbitals.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn localization_example() {
    let dir = ran("example_localization", &example("localization.toml"));
    energy_at_bar(&dir, WATER_ENERGY);
    let text = std::fs::read_to_string(dir.join("localization.json")).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).expect("localization.json parses");
    let before = json["functional_before"].as_f64().unwrap();
    let after = json["functional_after"].as_f64().unwrap();
    let sweeps = json["sweeps"].as_u64().unwrap();
    assert!(before < after, "{text}");
    assert!((1..=64).contains(&sweeps), "{text}");
    let c_local = fb_io_npy::read_npy(&dir.join("c_local.npy")).unwrap();
    assert_eq!(c_local.shape, [24, 5], "water's AOs by its occupied");
}

/// The two-GPU example: the DF-RHF over devices 0 and 1 at the one-GPU
/// energy's bar. Tier 5: it needs a second device (`docs/testing.md`,
/// "Selecting GPU tiers in cargo").
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_two_gpus_example() {
    let dir = ran("example_two_gpus", &example("two_gpus.toml"));
    energy_at_bar(&dir, WATER_ENERGY);
}

/// A config the binary refuses or fails, written into a working directory
/// of its own beside the water: exit code `kind`'s, no panic, `completed:
/// 1 errors`, the log naming every one of `named`. Returns the directory.
fn refused(name: &str, config: &str, kind: ErrorKind, named: &[&str]) -> PathBuf {
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
    let log = format!("{stdout}{stderr}");
    for n in named {
        assert!(log.contains(n), "{name}: names '{n}': {log}");
    }
    cwd
}

/// A config refused as it is read: [`refused`], `InvalidConfig`, and no
/// output directory made.
fn refused_on_reading(name: &str, config: &str, named: &[&str]) {
    let cwd = refused(name, config, ErrorKind::InvalidConfig, named);
    assert!(
        !cwd.join("out").exists(),
        "{name}: refused before the output"
    );
}

/// The old binary's keys are gone, not aliased: `[basis_set]`,
/// `num_gpus` and `[memory]` are each an unknown key, refused naming it,
/// the file and its line.
#[test]
fn the_old_keys_are_unknown() {
    let molecule = "[molecule]\nxyz = \"water.xyz\"\n";
    for (name, config, key, line) in [
        (
            "refused_basis_set",
            format!("calculation = \"overlap\"\n{molecule}[basis_set]\nbasis = \"sto-3g\"\n"),
            "basis_set",
            4,
        ),
        (
            "refused_num_gpus",
            format!(
                "calculation = \"overlap\"\nnum_gpus = 2\n{molecule}[basis]\nname = \"sto-3g\"\n"
            ),
            "num_gpus",
            2,
        ),
        (
            "refused_memory",
            format!(
                "calculation = \"overlap\"\n{molecule}[basis]\nname = \"sto-3g\"\n[memory]\n\
                 max_host_mem = \"32GiB\"\n"
            ),
            "memory",
            6,
        ),
    ] {
        refused_on_reading(
            name,
            &config,
            &[
                &format!("unknown field `{key}`"),
                &format!("config.toml:{line}:"),
            ],
        );
    }
}

/// A table beside a calculation that does not read it is refused at the
/// table's line, and `"rhf"` with no `[eri]` at `calculation`'s.
#[test]
fn a_table_the_calculation_does_not_read_is_refused() {
    let base = "[molecule]\nxyz = \"water.xyz\"\n[basis]\nname = \"sto-3g\"\n";
    refused_on_reading(
        "refused_stray_table",
        &format!("calculation = \"overlap\"\n{base}[scf]\nmax_iterations = 3\n"),
        &["config.toml:6: [scf] is set, but calculation = \"overlap\" does not read it"],
    );
    refused_on_reading(
        "refused_stray_eri",
        &format!("calculation = \"guess\"\n{base}[eri]\naux_basis = \"cc-pvdz-jkfit\"\n"),
        &["config.toml:6: [eri] is set, but calculation = \"guess\""],
    );
    refused_on_reading(
        "refused_no_eri",
        &format!("\ncalculation = \"rhf\"\n{base}"),
        &["config.toml:2: calculation = \"rhf\" needs [eri]"],
    );
}

/// Two Fock builds are not enough from the SAD guess: the run fails,
/// `MaxIterationsExceeded`, the boundary logging it once, and the record
/// still written.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn too_few_iterations_exits_max_iterations_exceeded() {
    let cwd = refused(
        "failed_max_iterations",
        "calculation = \"rhf\"\n[molecule]\nxyz = \"water.xyz\"\n[basis]\nname = \"sto-3g\"\n\
         [eri]\naux_basis = \"cc-pvdz-jkfit\"\n[scf]\nmax_iterations = 2\n",
        ErrorKind::MaxIterationsExceeded,
        &["2 iterations"],
    );
    assert!(
        !cwd.join("out/scf.json").exists(),
        "no product of a failed run"
    );
}

/// `rust/examples/` holds exactly [`EXAMPLES`]' configs, so a new example
/// is run here or this fails.
#[test]
fn the_examples_are_listed() {
    let on_disk: BTreeSet<String> = std::fs::read_dir(example(""))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|f| f.ends_with(".toml"))
        .collect();
    let listed: BTreeSet<String> = EXAMPLES.iter().map(|f| f.to_string()).collect();
    assert_eq!(on_disk, listed);
}

/// The keys `fb_steps::Config` accepts, as dotted paths
/// (`scf.diis_history`), read from the type's own refusal of an unknown
/// key: serde's `deny_unknown_fields` lists every field it expected. A
/// field that refuses a table of its own is a key; one that takes one is a
/// table, whose keys are read the same way.
fn accepted_keys() -> BTreeSet<String> {
    /// The fields the refusal of `text` lists, `None` when it refuses
    /// something other than the unknown key.
    fn expected(text: &str) -> Option<Vec<String>> {
        let error = toml::from_str::<fb_steps::Config>(text).expect_err(text);
        let message = error.message();
        let list = message.strip_prefix("unknown field `zz_unknown`, expected ")?;
        Some(
            list.split('`')
                .skip(1)
                .step_by(2)
                .map(str::to_string)
                .collect(),
        )
    }
    let mut keys = BTreeSet::new();
    for field in expected("zz_unknown = 1\n").expect("the root refuses an unknown key") {
        match expected(&format!("[{field}]\nzz_unknown = 1\n")) {
            Some(inner) => keys.extend(inner.into_iter().map(|k| format!("{field}.{k}"))),
            None => {
                keys.insert(field);
            }
        }
    }
    keys
}

/// The keys `table` sets, as dotted paths.
fn set_keys(prefix: &str, table: &toml::Table, keys: &mut BTreeSet<String>) {
    for (key, value) in table {
        match value {
            toml::Value::Table(inner) => set_keys(&format!("{prefix}{key}."), inner, keys),
            _ => {
                keys.insert(format!("{prefix}{key}"));
            }
        }
    }
}

/// The keys no example can set beside the key it shows in a comment
/// instead: `[guess] from` excludes `type`, which the examples set, and
/// needs a `.npy` file beside the example.
const SHOWN_IN_A_COMMENT: [&str; 1] = ["guess.from"];

/// Every key the config accepts is set by an example
/// (`docs/documentation.md`, "Configuration: the reference is emitted
/// from the type"), the keys read from the type itself, so a key added to
/// `Config` or to one of its tables fails here until an example sets it. A
/// key that excludes one an example sets is listed in
/// [`SHOWN_IN_A_COMMENT`], and must then appear as `key =` in an example's
/// comment.
#[test]
fn every_key_is_set_by_an_example() {
    let accepted = accepted_keys();
    assert!(
        accepted.contains("calculation")
            && accepted.contains("scf.diis_history")
            && accepted.contains("resources.devices"),
        "{accepted:?}"
    );
    println!("{} keys: {accepted:?}", accepted.len());
    let mut set = BTreeSet::new();
    let mut comments = String::new();
    for file in EXAMPLES {
        let text = std::fs::read_to_string(example(file)).unwrap();
        set_keys("", &toml::from_str(&text).unwrap(), &mut set);
        for line in text.lines() {
            if let Some((_, comment)) = line.split_once('#') {
                comments.push_str(comment);
                comments.push('\n');
            }
        }
    }
    for key in &accepted {
        let leaf = key.rsplit('.').next().unwrap();
        if SHOWN_IN_A_COMMENT.contains(&key.as_str()) {
            assert!(!set.contains(key), "{key} is set: drop it from the list");
            assert!(
                comments.contains(&format!("{leaf} =")),
                "{key} is shown in no example's comment"
            );
        } else {
            assert!(set.contains(key), "{key} is set by no example");
        }
    }
    let unknown: Vec<_> = set.difference(&accepted).collect();
    assert!(unknown.is_empty(), "set but not accepted: {unknown:?}");
}
