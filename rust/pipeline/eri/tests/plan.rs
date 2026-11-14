//! Water's plan (tier 3): the plan text, the device and pinned tables
//! among it with the arena's size, is a committed golden, `FB_BLESS=1`
//! rewriting it (`docs/testing.md`, "Performance"); admission of its device
//! table; and the refusal of an unknown fitting set. Planning touches no
//! device, but the load checks read both kernel libraries, so they must be
//! built.

mod common;

use fb_core_error_handling::ErrorKind;
use fb_core_planner::Dag;
use fb_test_support::golden;

#[test]
fn plan_golden() {
    let (molecule, basis) = common::water();
    let mut dag = Dag::new();
    common::add_step(&mut dag, &molecule, &basis, common::AUX).expect("the step plans");
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = dag
        .plan()
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "plan_h2o_cc-pvdz_cc-pvdz-jkfit.txt",
        &text,
    );
}

/// B and L are rows of the device table at planning, `naux · nao²` and
/// `naux²` doubles, so the block admission checks counts them: a budget
/// one byte short of B, or of the block, is refused as `Allocation` before
/// anything runs, naming the bytes and the rows live at the peak, and the
/// block itself is admitted. The peak is the three-center task's, where B
/// is not yet live, so the budget short of B names it among the rows over
/// the cap alone. Catches B or L built outside the table, which admission
/// would never see.
#[test]
fn admission_counts_b_and_l() {
    let (molecule, basis) = common::water();
    let mut dag = Dag::new();
    let fitted = common::add_step(&mut dag, &molecule, &basis, common::AUX).unwrap();
    let (nao, naux) = (basis.nao_sphe() as usize, fitted.aux.nao_sphe() as usize);
    let planned = dag.plan();
    let table = planned.device_table(0);
    assert_eq!(table.len(fitted.b), naux * nao * nao, "B's length");
    assert_eq!(table.len(fitted.l), naux * naux, "L's length");
    let b_bytes = 8 * naux * nao * nao;
    for cap in [b_bytes - 1, table.total() - 1] {
        let err = fb_core_arena::admit(table, cap, "gpu0").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Allocation, "{err}");
        let named = format!(
            "needs {} bytes, {} over its cap",
            table.total(),
            table.total() - cap
        );
        assert!(err.message.contains(&named), "{named} not in: {err}");
    }
    let err = fb_core_arena::admit(table, b_bytes - 1, "gpu0").unwrap_err();
    let alone = err
        .message
        .split_once("over the cap alone: ")
        .and_then(|(_, rest)| rest.split_once("; its peak"))
        .map(|(rows, _)| rows)
        .unwrap_or_else(|| panic!("no rows over the cap alone in: {err}"));
    let b = format!("B ({b_bytes} B)");
    assert!(
        alone.split(", ").any(|row| row == b),
        "{b} not over the cap alone in: {err}"
    );
    fb_core_arena::admit(table, table.total(), "gpu0").unwrap();
}

/// A fitting set the tree does not carry is `InvalidConfig` at planning,
/// naming it, and is never read as a path: no file of that name is looked
/// for.
#[test]
fn an_unknown_fitting_set_is_invalid_config() {
    let (molecule, basis) = common::water();
    let mut dag = Dag::new();
    let err = common::add_step(&mut dag, &molecule, &basis, "cc-pvdz-jkfitt")
        .err()
        .expect("an unknown set is refused");
    assert_eq!(err.kind, ErrorKind::InvalidConfig, "{err}");
    assert!(err.message.contains("'cc-pvdz-jkfitt'"), "{err}");
}
