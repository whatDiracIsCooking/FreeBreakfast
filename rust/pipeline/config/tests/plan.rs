//! The examples' plans (tier 3): each one's text, the write nodes and their
//! edges among it with each worker's tables, is a committed golden,
//! `tests/golden/plan_overlap.txt` and `tests/golden/plan_guess.txt`,
//! `FB_BLESS=1` rewriting them (`docs/testing.md`, "Performance"). Planning
//! touches no device, but the overlap's records ask the kernel library for
//! the sort's workspace, so the library must be built. And the config's
//! paths, and its `[guess]` table read and refused.

use std::path::{Path, PathBuf};

use fb_core_error_handling::ErrorKind;
use fb_pipeline_config::{Calculation, GuessKind, GuessStep, plan, read_config};
use fb_test_support::golden;

/// `rust/examples/<file>`'s plan text, checked against the golden
/// `golden`.
fn plan_text(file: &str, calculation: Calculation, golden_name: &str) {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let config = read_config(&examples.join(file)).unwrap();
    assert_eq!(config.calculation, calculation);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = plan(&config)
        .unwrap()
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), golden_name, &text);
}

#[test]
fn plan_golden() {
    plan_text("overlap.toml", Calculation::Overlap, "plan_overlap.txt");
}

/// The guess example: a baked stamp per element of water in cc-pVDZ, `sad
/// guess`, and `write d.npy`, all on the CPU.
#[test]
fn plan_golden_guess() {
    plan_text("guess.toml", Calculation::Guess, "plan_guess.txt");
}

/// The geometry and a basis path are joined to the config's directory; a
/// compiled-in name is not, and `output` defaults to `fbf_output`.
#[test]
fn paths_are_the_configs() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("config_paths");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    let read = |basis: &str| {
        std::fs::write(
            &path,
            format!(
                "calculation = \"overlap\"\n[molecule]\nxyz = \"m.xyz\"\n[basis_set]\nbasis = \"{basis}\"\n"
            ),
        )
        .unwrap();
        read_config(&path).unwrap()
    };
    let named = read("STO-3G");
    assert_eq!(named.molecule.xyz, dir.join("m.xyz"));
    assert_eq!(named.basis_set.basis, "STO-3G");
    assert_eq!(named.output, Path::new("fbf_output"));
    let by_path = read("my.json");
    assert_eq!(
        by_path.basis_set.basis,
        dir.join("my.json").display().to_string()
    );
}

/// `text` written as `config.toml` in a directory of its own, `name`, and
/// read.
fn read_text(
    name: &str,
    text: &str,
) -> (
    PathBuf,
    fb_core_error_handling::Result<fb_pipeline_config::Config>,
) {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.toml");
    std::fs::write(&path, text).unwrap();
    let config = read_config(&path);
    (dir, config)
}

const GUESS: &str =
    "calculation = \"guess\"\n[molecule]\nxyz = \"m.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n";

/// `[guess]` is optional, a SAD guess; `type = "sad"` names it; `from` is
/// joined to the config's directory.
#[test]
fn guess_is_read() {
    let (_, absent) = read_text("guess_absent", GUESS);
    assert_eq!(absent.unwrap().guess, None);
    let (_, sad) = read_text("guess_sad", &format!("{GUESS}[guess]\ntype = \"sad\"\n"));
    assert_eq!(
        sad.unwrap().guess,
        Some(GuessStep {
            kind: Some(GuessKind::Sad),
            from: None
        })
    );
    let (dir, load) = read_text("guess_from", &format!("{GUESS}[guess]\nfrom = \"d.npy\"\n"));
    assert_eq!(load.unwrap().guess.unwrap().from, Some(dir.join("d.npy")));
}

/// What `[guess]` refuses, each `InvalidConfig` naming the key or table:
/// a kind that is not one, `type` beside `from`, an unknown key, and the
/// table beside another calculation.
#[test]
fn guess_refusals() {
    let cases = [
        (
            "guess_bad_type",
            format!("{GUESS}[guess]\ntype = \"huckel\"\n"),
            "unknown variant `huckel`",
        ),
        (
            "guess_type_and_from",
            format!("{GUESS}[guess]\ntype = \"sad\"\nfrom = \"d.npy\"\n"),
            "[guess] loads 'from' a file, so takes no option 'type'",
        ),
        (
            "guess_unknown_key",
            format!("{GUESS}[guess]\nscale = 1.0\n"),
            "unknown field `scale`",
        ),
        (
            "guess_beside_overlap",
            format!("{}[guess]\n", GUESS.replace("\"guess\"", "\"overlap\"")),
            "[guess] is read by calculation = \"guess\" alone",
        ),
    ];
    for (name, text, named) in cases {
        let (_, config) = read_text(name, &text);
        let error = config.expect_err(name);
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{name}: {error}");
        assert!(error.message.contains(named), "{name}: {error}");
        assert!(error.message.contains("config.toml"), "{name}: {error}");
    }
}
