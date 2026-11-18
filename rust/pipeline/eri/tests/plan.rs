//! Water's plan at `schwarz_threshold = 0` (tier 3): the plan text, the
//! device and pinned tables among it with the arena's size, is a committed
//! golden, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance");
//! B on the heap and admission of the device table; and the refusal of an
//! unknown fitting set. Planning touches no device, but the load checks
//! read both kernel libraries, so they must be built.

mod common;

use fb_core_error_handling::ErrorKind;
use fb_core_planner::Dag;
use fb_test_support::golden;

#[test]
fn plan_golden() {
    let (molecule, basis) = common::water();
    let mut dag = Dag::new();
    common::add_step(&mut dag, &molecule, &basis, common::AUX, 0.0).expect("the step plans");
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

/// B is a heap row of `naux · n_columns` doubles, `n_columns` the
/// triangle's spherical columns at threshold 0, and no device row but the
/// BLAS workspace holds `naux · n_columns` doubles or more (a resident B,
/// square or packed);
/// L is a device row of `naux²`, so admission counts it: a budget one
/// byte short of the device table, or of L, is refused as `Allocation`
/// before anything runs, naming the bytes, and the table itself is
/// admitted. Catches B built on the device, or L outside the table.
#[test]
fn b_is_on_the_heap_and_admission_counts_l() {
    let (molecule, basis) = common::water();
    let mut dag = Dag::new();
    let fitted = common::add_step(&mut dag, &molecule, &basis, common::AUX, 0.0).unwrap();
    let naux = fitted.aux.nao_sphe() as usize;
    let nao = basis.nao_sphe() as usize;
    let n_columns = fitted.map.sphe().n_columns();
    assert!(n_columns < nao * nao, "the triangle is packed");
    let planned = dag.plan();
    assert_eq!(
        planned.heap_table().len(fitted.b),
        naux * n_columns,
        "B's length"
    );
    let table = planned.device_table(0);
    assert_eq!(table.len(fitted.l), naux * naux, "L's length");
    for row in table.rows().iter().filter(|r| r.name() != "blas workspace") {
        assert!(
            row.bytes() < 8 * naux * n_columns,
            "{} holds {} bytes, a whole B's {}",
            row.name(),
            row.bytes(),
            8 * naux * n_columns
        );
    }
    for cap in [8 * naux * naux - 1, table.total() - 1] {
        let err = fb_core_arena::admit(table, cap, "gpu0").unwrap_err();
        assert_eq!(err.kind, ErrorKind::Allocation, "{err}");
        let named = format!(
            "needs {} bytes, {} over its cap",
            table.total(),
            table.total() - cap
        );
        assert!(err.message.contains(&named), "{named} not in: {err}");
    }
    fb_core_arena::admit(table, table.total(), "gpu0").unwrap();
}

/// A fitting set the tree does not carry is `InvalidConfig` at planning,
/// naming it, and is never read as a path: no file of that name is looked
/// for.
#[test]
fn an_unknown_fitting_set_is_invalid_config() {
    let (molecule, basis) = common::water();
    let mut dag = Dag::new();
    let err = common::add_step(&mut dag, &molecule, &basis, "cc-pvdz-jkfitt", 0.0)
        .err()
        .expect("an unknown set is refused");
    assert_eq!(err.kind, ErrorKind::InvalidConfig, "{err}");
    assert!(err.message.contains("'cc-pvdz-jkfitt'"), "{err}");
}
