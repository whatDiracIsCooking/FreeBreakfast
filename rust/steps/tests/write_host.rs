//! The run's record on disk with no device (tier 3): a dry run of water
//! with `[eri]` and `[scf]` writes `budget.txt`, the very table
//! `--budget-only` prints (`Run::table`), and `timing.json` with no step,
//! and no product, since no step ran.

#![allow(
    clippy::disallowed_macros,
    reason = "a test's paths, outside any step's run"
)]

mod common;

use fb_core_budget::Backing;
use fb_steps::run;

#[test]
fn a_dry_run_writes_the_budget_only_table_and_no_product() {
    let config = common::chain("h2o.xyz");
    let run = run(config, Backing::Dry).unwrap();
    let dir = std::env::temp_dir().join(format!("fb-steps-write-host-{}", std::process::id()));
    run.write(&dir).unwrap();

    assert_eq!(
        std::fs::read_to_string(dir.join("budget.txt")).unwrap(),
        run.table
    );
    let timing: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("timing.json")).unwrap()).unwrap();
    assert_eq!(timing["steps"], serde_json::json!([]));
    assert_eq!(timing["execute"], 0);
    for product in ["scf.json", "c.npy", "d.npy"] {
        assert!(!dir.join(product).exists(), "{product} after a dry run");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
