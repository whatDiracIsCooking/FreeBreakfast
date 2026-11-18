//! The examples' plans (tier 3): each one's text, the write nodes and their
//! edges among it with each worker's tables, is a committed golden,
//! `tests/golden/plan_<example>.txt`, `FB_BLESS=1` rewriting them
//! (`docs/testing.md`, "Performance"), the DF-RHF's at a fixed syevd work
//! of 1234 doubles, since the vendor's number needs a device. Planning
//! touches no device, but the records ask the kernel library for the
//! sort's workspace, so the library must be built. And the config's paths,
//! its `[guess]`, `[eri]`, `[scf]`, `[localization]` and `[memory]` tables
//! read and refused,
//! every key of the config set by an example, and admission against the
//! caps, each card's free memory a number the case gives.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use fb_core_error_handling::ErrorKind;
use std::cell::Cell;

use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, Plan};
use fb_localization_pipek_mezey::{LocalizationMode, LocalizationStep, PmExponent};
use fb_pipeline_config::{
    Calculation, Card, DEVICE_RESERVE, MemoryCaps, admit_plan, plan, read_config,
};
use fb_pipeline_guess::{GuessKind, GuessStep};
use fb_pipeline_scf::ScfStep;
use fb_test_support::golden;

/// The syevd work the DF-RHF's plan is laid out with.
const LWORK: usize = 1234;

/// The card reader of a plan with no adaptive part (every plan but the
/// density-fitted ones), or one refused before any is added: it reads no
/// card.
fn no_card(_: usize) -> fb_core_error_handling::Result<Card> {
    panic!("a card read by a plan with no adaptive part")
}

/// The examples, by file name.
const EXAMPLES: [&str; 9] = [
    "overlap.toml",
    "guess.toml",
    "rhf.toml",
    "rohf.toml",
    "memory.toml",
    "localization.toml",
    "two_gpus.toml",
    "conventional.toml",
    "conventional_dense.toml",
];

/// `rust/examples/`.
fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples")
}

/// `rust/examples/<file>`'s plan text, checked against the golden
/// `golden`, planned on [`tib`]'s card, so a density-fitted plan's widths
/// are those of the default `[memory]` caps, whatever card runs the test.
fn plan_text(file: &str, calculation: Calculation, golden_name: &str) {
    let config = read_config(&examples().join(file)).unwrap();
    assert_eq!(config.calculation, calculation);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = plan(&config, |_| Ok(LWORK), tib)
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

/// The two-device example plans the overlap example's plan, to the byte:
/// `num_gpus` places no worker (`docs/programming_model.md`, "The run's
/// devices").
#[test]
fn plan_golden_two_gpus() {
    plan_text("two_gpus.toml", Calculation::Overlap, "plan_overlap.txt");
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

/// The DF-ROHF example, the water cation's doublet: the DF-RHF's plan with
/// the guess rescaled, the open shell's nodes before, in and after the
/// loop, and the same write nodes.
#[test]
fn plan_golden_rohf() {
    plan_text("rohf.toml", Calculation::Rhf, "plan_rohf.txt");
}

/// The conventional example: the RHF over the packed tensors, their three
/// tasks in place of B's and the body's `conventional jk`, and the RHF's
/// write nodes.
#[test]
fn plan_golden_conventional() {
    plan_text(
        "conventional.toml",
        Calculation::Rhf,
        "plan_conventional.txt",
    );
}

/// The dense conventional example on two GPU workers: the setup on worker
/// 0, each worker's records and row blocks, the body's J and K spanning
/// both (`dense density`, the broadcast, a `dense jk` per worker, the two
/// gathers, `dense spherical`), and the RHF's write nodes.
#[test]
fn plan_golden_conventional_dense() {
    plan_text(
        "conventional_dense.toml",
        Calculation::Rhf,
        "plan_conventional_dense.txt",
    );
}

/// The localization example: the DF-RHF's plan and its writes, then the
/// Pipek-Mezey tables, the loop of sweeps and its decision, `pm rotate`,
/// and the write nodes of `c_local.npy` (a stream download and a CPU write)
/// and `localization.json` (a CPU write of the heap row).
#[test]
fn plan_golden_localization() {
    plan_text(
        "localization.toml",
        Calculation::Localization,
        "plan_localization.txt",
    );
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

/// `num_gpus` defaults to 1 and is read as written; 0 is refused as the
/// file is read, `InvalidConfig` naming the file and the line; and the
/// run's devices are `0..num_gpus`, a count above the devices the runtime
/// sees refused, `InvalidConfig` naming both numbers, and the runtime not
/// asked at 1.
#[test]
fn num_gpus_is_read() {
    let (_, absent) = read_text("num_gpus_absent", &overlap_config(""));
    let absent = absent.unwrap();
    assert_eq!(absent.num_gpus.get(), 1);
    let unasked = || -> fb_core_error_handling::Result<usize> {
        panic!("one device asks the runtime nothing")
    };
    assert_eq!(absent.devices(unasked).unwrap(), [0]);
    let example = read_config(&examples().join("two_gpus.toml")).unwrap();
    assert_eq!(example.num_gpus.get(), 2);
    assert_eq!(example.devices(|| Ok(4)).unwrap(), [0, 1]);
    assert_eq!(example.devices(|| Ok(2)).unwrap(), [0, 1]);
    let error = example.devices(|| Ok(1)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{error}");
    assert!(
        error.message.contains("num_gpus = 2") && error.message.contains("the 1 the runtime sees"),
        "{error}"
    );
    let (_, zero) = read_text(
        "num_gpus_zero",
        "num_gpus = 0\ncalculation = \"overlap\"\n[molecule]\nxyz = \"m.xyz\"\n\
         [basis_set]\nbasis = \"sto-3g\"\n",
    );
    let error = zero.unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{error}");
    assert!(
        error.message.contains("config.toml:1:") && error.message.contains("nonzero"),
        "{error}"
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
            "[guess] is read by calculation = \"guess\", \"rhf\" and \"localization\" alone",
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
/// C++ default mode, which is not ported; a missing `aux_basis` or
/// `schwarz_threshold` under `density_fitting`, which have no default, or
/// a negative threshold; an `aux_basis` or a `schwarz_threshold` under
/// `conventional`, which fits and screens nothing; an unknown key; a well-formed table
/// beside a calculation other than `rhf`; its absence beside `rhf`; and a
/// fitting set that is not compiled in.
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
            RHF.replace("aux_basis = \"cc-pvdz-jkfit\"\n", ""),
            "[eri] aux_basis: mode = \"density_fitting\" needs a fitting set",
        ),
        (
            "eri_no_schwarz_threshold",
            RHF.replace("schwarz_threshold = 0.0\n", ""),
            "[eri] schwarz_threshold: mode = \"density_fitting\" needs a screen",
        ),
        (
            "eri_negative_schwarz_threshold",
            RHF.replace("schwarz_threshold = 0.0", "schwarz_threshold = -1e-12"),
            "[eri] schwarz_threshold: -0.000000000001 is not a non-negative number",
        ),
        (
            "eri_conventional_with_schwarz_threshold",
            RHF.replace("density_fitting", "conventional")
                .replace("aux_basis = \"cc-pvdz-jkfit\"\n", ""),
            "[eri] schwarz_threshold: 0 given, but mode = \"conventional\" screens nothing",
        ),
        (
            "eri_conventional_with_aux_basis",
            RHF.replace("density_fitting", "conventional"),
            "[eri] aux_basis: 'cc-pvdz-jkfit' given, but mode = \"conventional\" fits nothing",
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
            format!("{GUESS}{ERI}"),
            "[eri] is read by calculation = \"rhf\" and \"localization\" alone",
        ),
        (
            "eri_absent_beside_rhf",
            RHF.replace(ERI, ""),
            "calculation = \"rhf\" needs the [eri] table",
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
const ERI: &str =
    "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\nschwarz_threshold = 0.0\n";

/// A DF-RHF config with no `[scf]`.
const RHF: &str = "calculation = \"rhf\"\n[molecule]\nxyz = \"m.xyz\"\n[basis_set]\nbasis = \
                   \"sto-3g\"\n[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n\
                   schwarz_threshold = 0.0\n";

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
            "[scf] is read by calculation = \"rhf\" and \"localization\" alone",
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
    let Err(error) = plan(
        &config.unwrap(),
        |_| panic!("lwork asked of a refused config"),
        no_card,
    ) else {
        panic!("one electron is planned");
    };
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{error}");
}

/// A DF-RHF config's tables under `calculation = "localization"`.
fn localization(extra: &str) -> String {
    format!("{}{extra}", RHF.replace("\"rhf\"", "\"localization\""))
}

/// `[localization]` is optional, every key at its default; a key set
/// replaces its default alone.
#[test]
fn localization_is_read() {
    let (_, absent) = read_text("localization_absent", &localization(""));
    assert_eq!(absent.unwrap().localization, None);
    let (_, set) = read_text(
        "localization_set",
        &localization("[localization]\nexponent = 4\n"),
    );
    assert_eq!(
        set.unwrap().localization,
        Some(LocalizationStep {
            exponent: PmExponent::Four,
            ..LocalizationStep::default()
        })
    );
}

/// What `[localization]` refuses, naming the key or table and the file:
/// `mode = "ibo"`, read and not ported, `Unsupported`; and, each
/// `InvalidConfig`, a mode that is not one, an exponent other than 2 or
/// 4, `max_sweeps = 0`, a negative tolerance, the C++ `livvo`, which is
/// not ported, the table beside a calculation that does not read it, and
/// the calculation without `[eri]`.
#[test]
fn localization_refusals() {
    let cases = [
        (
            "localization_ibo",
            localization("[localization]\nmode = \"ibo\"\n"),
            ErrorKind::Unsupported,
            "does not port IBO yet",
        ),
        (
            "localization_bad_mode",
            localization("[localization]\nmode = \"boys\"\n"),
            ErrorKind::InvalidConfig,
            "unknown variant `boys`",
        ),
        (
            "localization_bad_exponent",
            localization("[localization]\nexponent = 3\n"),
            ErrorKind::InvalidConfig,
            "exponent is 3, not 2 or 4",
        ),
        (
            "localization_no_sweeps",
            localization("[localization]\nmax_sweeps = 0\n"),
            ErrorKind::InvalidConfig,
            "[localization] max_sweeps is 0",
        ),
        (
            "localization_negative_tolerance",
            localization("[localization]\nfunctional_tolerance = -1.0\n"),
            ErrorKind::InvalidConfig,
            "must be non-negative",
        ),
        (
            "localization_livvo",
            localization("[localization]\nlivvo = true\n"),
            ErrorKind::InvalidConfig,
            "unknown field `livvo`",
        ),
        (
            "localization_beside_rhf",
            format!("{RHF}[localization]\n"),
            ErrorKind::InvalidConfig,
            "[localization] is read by calculation = \"localization\" alone",
        ),
        (
            "localization_without_eri",
            localization("").replace(ERI, ""),
            ErrorKind::InvalidConfig,
            "calculation = \"localization\" needs the [eri] table",
        ),
    ];
    for (name, text, kind, named) in cases {
        let (_, config) = read_text(name, &text);
        let error = config.expect_err(name);
        assert_eq!(error.kind, kind, "{name}: {error}");
        assert!(error.message.contains(named), "{name}: {error}");
        assert!(error.message.contains("config.toml"), "{name}: {error}");
    }
}

/// `mode = "ibo"` in a `Config` built in code, past [`read_config`], is
/// refused at planning, `Unsupported`, before `lwork` is asked.
#[test]
fn ibo_is_refused_at_planning_before_lwork() {
    let (dir, config) = read_text("localization_ibo_planned", &localization(""));
    std::fs::copy(examples().join("water.xyz"), dir.join("m.xyz")).unwrap();
    let mut config = config.unwrap();
    config.localization = Some(LocalizationStep {
        mode: LocalizationMode::Ibo,
        ..LocalizationStep::default()
    });
    let Err(error) = plan(
        &config,
        |_| panic!("lwork asked of a refused config"),
        no_card,
    ) else {
        panic!("IBO is planned");
    };
    assert_eq!(error.kind, ErrorKind::Unsupported, "{error}");
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
/// `Config` or to a step's table (`GuessStep`, `EriStep`, `ScfStep`,
/// `LocalizationStep`) fails here until an example sets it. A key that
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

const GIB: usize = 1 << 30;

/// The overlap of water in STO-3G, `extra` appended.
fn overlap_config(extra: &str) -> String {
    format!(
        "calculation = \"overlap\"\n[molecule]\nxyz = \"m.xyz\"\n[basis_set]\nbasis = \
         \"sto-3g\"\n{extra}"
    )
}

/// `[memory]` is optional, each key at its default, which is its floor;
/// a key set replaces its default alone, read as the C++ reads a byte
/// size: an integer of bytes, or a number and a unit, rounded down.
#[test]
fn memory_is_read() {
    let caps = |name: &str, table: &str| {
        let (_, config) = read_text(name, &overlap_config(table));
        config.unwrap().memory
    };
    assert_eq!(
        caps("memory_absent", ""),
        MemoryCaps {
            max_device_mem_per_gpu: 8 * GIB,
            max_pinned_mem_per_gpu: 2 * GIB,
            max_host_mem: 16 * GIB,
        }
    );
    assert_eq!(caps("memory_empty", "[memory]\n"), MemoryCaps::default());
    let example = read_config(&examples().join("memory.toml")).unwrap();
    assert_eq!(
        example.memory,
        MemoryCaps {
            max_device_mem_per_gpu: 10 * GIB,
            max_pinned_mem_per_gpu: 4 * GIB,
            max_host_mem: 32 * GIB,
        }
    );
    for (value, bytes) in [
        ("8589934592", 8 * GIB),
        ("\"8GiB\"", 8 * GIB),
        ("\"8192 MiB\"", 8 * GIB),
        ("\"9GB\"", 9_000_000_000),
        ("\"8.5GiB\"", 8 * GIB + GIB / 2),
        ("\"0.01TiB\"", (0.01 * 1024.0 * GIB as f64) as usize),
        ("\"10000000000B\"", 10_000_000_000),
    ] {
        let device = caps(
            "memory_value",
            &format!("[memory]\nmax_device_mem_per_gpu = {value}\n"),
        );
        assert_eq!(
            device,
            MemoryCaps {
                max_device_mem_per_gpu: bytes,
                ..MemoryCaps::default()
            },
            "{value}"
        );
    }
}

/// Each key's default is its floor: a value under it is refused,
/// `InvalidConfig`, naming the key and the value, and the floor itself is
/// read.
#[test]
fn memory_floors() {
    let cases = [
        (
            "max_device_mem_per_gpu = \"7GiB\"",
            "[memory] max_device_mem_per_gpu = 7516192768 bytes (7.000 GiB) is under its floor \
             of 8589934592 bytes (8 GiB)",
        ),
        (
            "max_device_mem_per_gpu = 8589934591",
            "[memory] max_device_mem_per_gpu = 8589934591 bytes",
        ),
        (
            "max_pinned_mem_per_gpu = \"1GiB\"",
            "[memory] max_pinned_mem_per_gpu = 1073741824 bytes (1.000 GiB) is under its floor \
             of 2147483648 bytes (2 GiB)",
        ),
        (
            "max_host_mem = \"15GiB\"",
            "[memory] max_host_mem = 16106127360 bytes (15.000 GiB) is under its floor of \
             17179869184 bytes (16 GiB)",
        ),
    ];
    for (key, named) in cases {
        let (_, config) = read_text(
            "memory_floor",
            &overlap_config(&format!("[memory]\n{key}\n")),
        );
        let error = config.expect_err(key);
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{key}: {error}");
        assert!(error.message.contains(named), "{key}: {error}");
        assert!(error.message.contains("config.toml"), "{key}: {error}");
    }
    let (_, at) = read_text(
        "memory_at_floor",
        &overlap_config(
            "[memory]\nmax_device_mem_per_gpu = 8589934592\nmax_pinned_mem_per_gpu = \"2GiB\"\n\
             max_host_mem = \"16GiB\"\n",
        ),
    );
    assert_eq!(at.unwrap().memory, MemoryCaps::default());
}

/// What `[memory]` refuses as a byte size, each `InvalidConfig` naming the
/// file and the line: no unit, an unknown one, an exponent, a float, below
/// 1 byte, past 2^63 bytes, no number; and an unknown key.
#[test]
fn memory_refusals() {
    let cases = [
        ("max_device_mem_per_gpu = \"10\"", "has no unit"),
        ("max_device_mem_per_gpu = \"10 GiBs\"", "has no unit 'GiBs'"),
        ("max_device_mem_per_gpu = \"1e10B\"", "has no unit 'e10B'"),
        ("max_device_mem_per_gpu = 10.5", "expected a byte size"),
        ("max_device_mem_per_gpu = true", "expected a byte size"),
        ("max_device_mem_per_gpu = 0", "0 is below 1 byte"),
        ("max_pinned_mem_per_gpu = -1", "-1 is below 1 byte"),
        ("max_pinned_mem_per_gpu = \"-5GiB\"", "is below 1 byte"),
        ("max_pinned_mem_per_gpu = \"0.5B\"", "is below 1 byte"),
        ("max_pinned_mem_per_gpu = \"9000000TiB\"", "past 2^63 bytes"),
        (
            "max_pinned_mem_per_gpu = \"GiB\"",
            "not a number and a unit",
        ),
        ("max_disk_mem = \"16GiB\"", "unknown field `max_disk_mem`"),
    ];
    for (key, named) in cases {
        let (_, config) = read_text(
            "memory_refused",
            &overlap_config(&format!("[memory]\n{key}\n")),
        );
        let error = config.expect_err(key);
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{key}: {error}");
        assert!(error.message.contains(named), "{key}: {error}");
        assert!(error.message.contains("config.toml:7:"), "{key}: {error}");
    }
}

/// A plan of one worker's two worker rows, `device` and `pinned` bytes.
fn rows(device: usize, pinned: usize) -> Plan<'static> {
    let mut dag = Dag::new();
    let _ = dag.device_worker_row::<u8>(0, "device row", device);
    let _ = dag.host_worker_row::<u8>(0, "pinned row", pinned);
    dag.plan()
}

/// The node's `MemAvailable` the admission cases give: past any pinned cap
/// they set.
const NODE: usize = 1 << 50;

/// A card of a TiB with all of it free and no kernel stack left to take:
/// its measured reserve is 0, so the reserve is the floor,
/// [`DEVICE_RESERVE`].
fn tib(_: usize) -> fb_core_error_handling::Result<Card> {
    Ok(Card {
        free: 1 << 40,
        total: 1 << 40,
        stack: 0,
    })
}

/// A plan with no key is admitted against the default caps: a device block
/// of 8 GiB less the reserve, and a pinned one of 2 GiB, fit on a card with
/// a TiB free; one byte more of either is refused, `Allocation`, naming the
/// worker, the row over the cap alone and the key.
#[test]
fn no_key_is_the_default_cap() {
    let caps = MemoryCaps::default();
    let device = 8 * GIB - DEVICE_RESERVE;
    admit_plan(&rows(device, 2 * GIB), &caps, tib, NODE).unwrap();
    let error = admit_plan(&rows(device + 1, 1), &caps, tib, NODE).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    for named in [
        "gpu0: the device block",
        "over the cap alone: device row",
        "[memory] max_device_mem_per_gpu = 8589934592 bytes",
        "less the reserve",
    ] {
        assert!(error.message.contains(named), "{named}: {error}");
    }
    let error = admit_plan(&rows(1, 2 * GIB + 1), &caps, tib, NODE).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    for named in [
        "gpu0: the pinned block",
        "over the cap alone: pinned row",
        "[memory] max_pinned_mem_per_gpu = 2147483648 bytes",
    ] {
        assert!(error.message.contains(named), "{named}: {error}");
    }
}

/// Any cap is bounded by the card: on a card of the cap's size, or of a
/// TiB under a cap past every card, the reserve is what the card does not
/// have free, so the example's device block is admitted with the block and
/// the stack free, and refused with one byte less, naming the free memory.
#[test]
fn admission_is_bounded_by_the_cards_free_memory() {
    let config = read_config(&examples().join("overlap.toml")).unwrap();
    let planned = plan(&config, |_| Ok(LWORK), no_card).unwrap();
    let block = planned.device_table(0).total();
    assert!(block > 0, "the overlap has device rows");
    let stack: usize = 64 << 20;
    for caps in [
        MemoryCaps::default(),
        MemoryCaps {
            max_device_mem_per_gpu: 1 << 62,
            ..MemoryCaps::default()
        },
    ] {
        let total = caps.max_device_mem_per_gpu.min(1 << 40);
        let card = |free: usize| move |_| Ok(Card { free, total, stack });
        admit_plan(&planned, &caps, card(block + stack), NODE).unwrap();
        let free = block + stack - 1;
        let error = admit_plan(&planned, &caps, card(free), NODE).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
        let named = format!("its {free} bytes free at planning");
        assert!(error.message.contains(&named), "{error}");
        assert!(error.message.contains("1 over its cap"), "{error}");
    }
}

/// `block + reserve + stack <= cap`, each term to the byte: under the
/// default cap, on a 16 GiB card with 1 GiB held (the measured reserve) and
/// 512 MiB of stack to take, a block of 6.5 GiB is admitted, and refused
/// when the block, the reserve or the stack is one byte larger, the refusal
/// naming the reserve, its reading and the stack. A reserve measured under
/// the floor is taken as [`DEVICE_RESERVE`].
#[test]
fn the_reserve_and_the_stack_enter_the_sum() {
    let caps = MemoryCaps::default();
    let total = 16 * GIB;
    let (reserve, stack) = (GIB, 512 << 20);
    let block = 8 * GIB - reserve - stack;
    let card = |held: usize, stack: usize| {
        move |_| {
            Ok(Card {
                free: total - held,
                total,
                stack,
            })
        }
    };
    admit_plan(&rows(block, 1), &caps, card(reserve, stack), NODE).unwrap();
    for (planned, held, stack) in [
        (rows(block + 1, 1), reserve, stack),
        (rows(block, 1), reserve + 1, stack),
        (rows(block, 1), reserve, stack + 1),
    ] {
        let error = admit_plan(&planned, &caps, card(held, stack), NODE).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
        for named in [
            "1 over its cap".to_owned(),
            format!("the card's {total} bytes"),
            format!("the reserve of {held} bytes"),
            format!("its {} bytes free at planning", total - held),
            format!("at least {DEVICE_RESERVE}"),
            format!("the kernels' stack of {stack} bytes"),
        ] {
            assert!(error.message.contains(&named), "{named}: {error}");
        }
    }
    // A reserve measured at 1 MiB is the floor's.
    let block = 8 * GIB - DEVICE_RESERVE - stack;
    admit_plan(&rows(block, 1), &caps, card(1 << 20, stack), NODE).unwrap();
    let error = admit_plan(&rows(block + 1, 1), &caps, card(1 << 20, stack), NODE).unwrap_err();
    let named = format!("the reserve of {DEVICE_RESERVE} bytes");
    assert!(error.message.contains(&named), "{named}: {error}");
}

/// Every table is admitted, and the card is read once per worker with a
/// device block, never for a plan without one: the guess computes on the
/// CPU, and its admission touches no device.
#[test]
fn admission_reads_the_card_only_for_a_device_block() {
    let caps = MemoryCaps::default();
    let config = read_config(&examples().join("guess.toml")).unwrap();
    let guess = plan(&config, |_| Ok(LWORK), no_card).unwrap();
    assert_eq!(
        guess.device_table(0).total(),
        0,
        "the guess has no device rows"
    );
    assert!(guess.host_table(0).total() > 0, "the guess has pinned rows");
    admit_plan(&guess, &caps, |_| panic!("a card read"), NODE).unwrap();
    let config = read_config(&examples().join("overlap.toml")).unwrap();
    let overlap = plan(&config, |_| Ok(LWORK), no_card).unwrap();
    let asked = Cell::new(0);
    let card = |w| {
        assert_eq!((w, asked.get()), (0, 0), "the one worker, read once");
        asked.set(1);
        tib(w)
    };
    admit_plan(&overlap, &caps, card, NODE).unwrap();
    assert_eq!(asked.get(), 1, "read once");
}

/// The workers' pinned caps together must fit the node's `MemAvailable`
/// within its cgroup's limit, as the C++ `check_pinned_node`: one worker's
/// 2 GiB default fits 2 GiB, and is refused by a byte less,
/// `InvalidConfig`, naming the key and `MemAvailable`, before any card is
/// read.
#[test]
fn pinned_caps_fit_the_node() {
    let caps = MemoryCaps::default();
    let planned = rows(1, 1);
    admit_plan(&planned, &caps, tib, 2 * GIB).unwrap();
    let error = admit_plan(&planned, &caps, |_| panic!("a card read"), 2 * GIB - 1).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig, "{error}");
    for named in [
        "1 GPU workers, each allowed [memory] max_pinned_mem_per_gpu = 2147483648 bytes",
        "MemAvailable within its cgroup's limit, 2147483647 bytes",
    ] {
        assert!(error.message.contains(named), "{named}: {error}");
    }
}

/// CPU: one heap output of `bytes` bytes, for a plan with a heap block.
struct HeapRow(usize);

impl CpuTask for HeapRow {
    type Rows = RowId<Heap, u8>;

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        heap.output("heap row", self.0)
    }

    fn run(&self, _: &Self::Rows, _: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        Ok(())
    }
}

/// [`rows`] with a heap row of `heap` bytes beside them.
fn rows_and_heap(device: usize, pinned: usize, heap: usize) -> Plan<'static> {
    let mut dag = Dag::new();
    let _ = dag.device_worker_row::<u8>(0, "device row", device);
    let _ = dag.host_worker_row::<u8>(0, "pinned row", pinned);
    dag.add_cpu("heap", 0, HeapRow(heap));
    dag.plan()
}

/// The heap block is admitted against `[memory] max_host_mem`: a block at
/// the cap fits, and one past it is refused, `Allocation`, naming the
/// plan, the heap block, its row and the key. The DF-RHF's plan holds the
/// packed B and the SCF's carried `last build` on the heap, and fits the
/// default.
#[test]
fn heap_is_admitted_against_max_host_mem() {
    let caps = MemoryCaps {
        max_host_mem: 4096,
        ..MemoryCaps::default()
    };
    admit_plan(&rows_and_heap(1, 1, 4096), &caps, tib, NODE).unwrap();
    let error = admit_plan(&rows_and_heap(1, 1, 4097), &caps, tib, NODE).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    for named in [
        "plan: the heap block needs 4097 bytes, 1 over its cap of 4096",
        "over the cap alone: heap row (4097 B)",
        "[memory] max_host_mem = 4096 bytes",
    ] {
        assert!(error.message.contains(named), "{named}: {error}");
    }
    let config = read_config(&examples().join("rhf.toml")).unwrap();
    let rhf = plan(&config, |_| Ok(LWORK), tib).unwrap();
    assert_eq!(rhf.heap_table().n_rows(), 2, "B and the SCF's last build");
    admit_plan(&rhf, &MemoryCaps::default(), tib, NODE).unwrap();
}

/// The node's pinned blocks and the heap block are admitted together
/// against the host's available memory: at their sum exactly, and refused
/// one byte short of it, `Allocation`, naming each kind's bytes; the
/// pinned caps alone fit that memory either way.
#[test]
fn admission_counts_pinned_and_heap_together() {
    let caps = MemoryCaps::default();
    let planned = rows_and_heap(1, GIB, GIB + 256);
    let sum = planned.host_table(0).total() + planned.heap_table().total();
    assert_eq!(sum, 2 * GIB + 256);
    admit_plan(&planned, &caps, tib, sum).unwrap();
    let error = admit_plan(&planned, &caps, tib, sum - 1).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Allocation, "{error}");
    for named in [
        "node: the host blocks need 2147483904 bytes, 1 over the 2147483903 bytes available",
        "the pinned blocks 1073741824 bytes",
        "the heap block 1073742080 bytes",
    ] {
        assert!(error.message.contains(named), "{named}: {error}");
    }
}
