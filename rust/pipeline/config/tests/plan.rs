//! The examples' plans (tier 3): each one's text, the write nodes and their
//! edges among it with each worker's tables, is a committed golden,
//! `tests/golden/plan_<example>.txt`, `FB_BLESS=1` rewriting them
//! (`docs/testing.md`, "Performance"), the DF-RHF's at a fixed syevd work
//! of 1234 doubles, since the vendor's number needs a device. Planning
//! touches no device, but the records ask the kernel library for the
//! sort's workspace, so the library must be built. And the config's paths,
//! its `[guess]`, `[eri]` and `[scf]` tables read and refused, and every
//! key of the config set by an example.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use fb_core_error_handling::ErrorKind;
use fb_pipeline_config::{Calculation, GuessKind, GuessStep, plan, read_config};
use fb_pipeline_scf::ScfStep;
use fb_test_support::golden;

/// The syevd work the DF-RHF's plan is laid out with.
const LWORK: usize = 1234;

/// The examples, by file name.
const EXAMPLES: [&str; 3] = ["overlap.toml", "guess.toml", "rhf.toml"];

/// `rust/examples/`.
fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

/// `rust/examples/<file>`'s plan text, checked against the golden
/// `golden`.
fn plan_text(file: &str, calculation: Calculation, golden_name: &str) {
    let config = read_config(&examples().join(file)).unwrap();
    assert_eq!(config.calculation, calculation);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = plan(&config, |_| Ok(LWORK))
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

/// The DF-RHF example: the setup, B, the SAD guess, the loop, and the write
/// nodes of `c.npy`, `d.npy` and `scf.json` after it, each a stream
/// download and a CPU write.
#[test]
fn plan_golden_rhf() {
    plan_text("rhf.toml", Calculation::Rhf, "plan_rhf.txt");
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
            "[guess] is read by calculation = \"guess\" and \"rhf\" alone",
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

/// What `[eri]` refuses, each `InvalidConfig` naming the key or table: the
/// C++ default mode, which is not ported; a missing `aux_basis`, which has
/// no default; an unknown key; a well-formed table beside a calculation
/// other than `rhf`; its absence beside `rhf`; and a fitting set that is
/// not compiled in.
#[test]
fn eri_refusals() {
    let cases = [
        (
            "eri_integral_direct",
            format!("{GUESS}[eri]\nmode = \"integral_direct\"\naux_basis = \"cc-pvdz-jkfit\"\n"),
            "unknown variant `integral_direct`",
        ),
        (
            "eri_no_aux_basis",
            format!("{GUESS}[eri]\nmode = \"density_fitting\"\n"),
            "missing field `aux_basis`",
        ),
        (
            "eri_unknown_key",
            format!(
                "{GUESS}[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n\
                 threshold = 1e-8\n"
            ),
            "unknown field `threshold`",
        ),
        (
            "eri_beside_guess",
            format!("{GUESS}[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"),
            "[eri] is read by calculation = \"rhf\" alone, and calculation = \"guess\"",
        ),
        (
            "eri_absent_beside_rhf",
            RHF.replace(ERI, ""),
            "calculation = \"rhf\" needs an [eri] table",
        ),
        (
            "eri_aux_not_carried",
            RHF.replace("cc-pvdz-jkfit", "no-such-jkfit"),
            "[eri] aux_basis: 'no-such-jkfit' is not a compiled-in fitting set",
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

/// The `[eri]` table of [`RHF`].
const ERI: &str = "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n";

/// A DF-RHF config with no `[scf]`.
const RHF: &str = "calculation = \"rhf\"\n[molecule]\nxyz = \"m.xyz\"\n[basis_set]\nbasis = \
                   \"sto-3g\"\n[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n";

/// `[scf]` is optional, every key at its default; a key set replaces its
/// default alone.
#[test]
fn scf_is_read() {
    let (_, absent) = read_text("scf_absent", RHF);
    assert_eq!(absent.unwrap().scf, None);
    let (_, set) = read_text("scf_set", &format!("{RHF}[scf]\nmax_iterations = 7\n"));
    assert_eq!(
        set.unwrap().scf,
        Some(ScfStep {
            max_iterations: 7,
            ..ScfStep::default()
        })
    );
}

/// What `[scf]` refuses, each `InvalidConfig` naming the key or table and
/// the file: an unknown key, an option out of range, a value of the wrong
/// type, and the table beside another calculation.
#[test]
fn scf_refusals() {
    let cases = [
        (
            "scf_unknown_key",
            format!("{RHF}[scf]\nlevel_shift = 0.5\n"),
            "unknown field `level_shift`",
        ),
        (
            "scf_out_of_range",
            format!("{RHF}[scf]\ndiis_history = 0\n"),
            "[scf] diis_history is 0",
        ),
        (
            "scf_wrong_type",
            format!("{RHF}[scf]\nmax_iterations = \"many\"\n"),
            "invalid type",
        ),
        (
            "scf_beside_guess",
            format!("{GUESS}[scf]\n"),
            "[scf] is read by calculation = \"rhf\" alone",
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

/// An odd electron count is refused at planning, `InvalidConfig`, before
/// `lwork` is asked: the plan touches no device for a config it refuses.
#[test]
fn an_open_shell_is_refused_before_lwork() {
    let (dir, config) = read_text("rhf_open_shell", RHF);
    std::fs::write(dir.join("m.xyz"), "1\nH atom\nH 0 0 0\n").unwrap();
    let Err(error) = plan(&config.unwrap(), |_| {
        panic!("lwork asked of a refused config")
    }) else {
        panic!("one electron is planned");
    };
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{error}");
}

/// The keys `Config` accepts, as dotted paths (`scf.diis_history`), read
/// from the type's own refusal of an unknown key: serde's
/// `deny_unknown_fields` lists every field it expected. A field that
/// refuses a table of its own is a key; one that takes one is a table,
/// whose keys are read the same way.
fn accepted_keys() -> BTreeSet<String> {
    /// The fields the refusal of `text` lists, `None` when it refuses
    /// something other than the unknown key.
    fn expected(text: &str) -> Option<Vec<String>> {
        let error = toml::from_str::<fb_pipeline_config::Config>(text).expect_err(text);
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
/// `Config` or to `ScfStep` fails here until an example sets it. A key that
/// excludes one an example sets is listed in [`SHOWN_IN_A_COMMENT`], and
/// must then appear as `key =` in an example's comment.
#[test]
fn every_key_is_set_by_an_example() {
    let accepted = accepted_keys();
    assert!(
        accepted.contains("scf.diis_history") && accepted.contains("guess.from"),
        "{accepted:?}"
    );
    println!("{} keys: {accepted:?}", accepted.len());
    let mut set = BTreeSet::new();
    let mut comments = String::new();
    for file in EXAMPLES {
        let text = std::fs::read_to_string(examples().join(file)).unwrap();
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
