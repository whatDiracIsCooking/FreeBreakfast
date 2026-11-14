//! The example's plan (tier 3): its text, the write nodes and their edges
//! among it with each worker's tables, is the committed golden
//! `tests/golden/plan_overlap.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device, but the
//! overlap's records ask the kernel library for the sort's workspace, so
//! the library must be built.

use std::path::Path;

use fb_pipeline_config::{Calculation, plan, read_config};
use fb_test_support::golden;

#[test]
fn plan_golden() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples");
    let config = read_config(&examples.join("overlap.toml")).unwrap();
    assert_eq!(config.calculation, Calculation::Overlap);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = plan(&config)
        .unwrap()
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_overlap.txt", &text);
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
