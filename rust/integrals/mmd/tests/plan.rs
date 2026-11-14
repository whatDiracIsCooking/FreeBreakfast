//! The records' plan for both molecules (tier 3): the plan text, the
//! device table's size (the arena's) among it, is a committed golden per
//! molecule, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance").
//! Planning touches no device, but the layout asks the kernel library for
//! the sort's workspace, so the library must be built.

mod common;

use fb_core_planner::Dag;
use fb_integrals_mmd::add_records;
use fb_test_support::golden;

#[test]
fn plan_goldens() {
    for (stem, basis) in common::cases() {
        let mut dag = Dag::new();
        let classes = add_records(&mut dag, 0, &basis).unwrap();
        let planned = dag.plan();
        // Every class is non-empty, ascending, la <= lb.
        assert!(classes.iter().all(|c| c.n_rec() > 0 && c.la() <= c.lb()));
        assert!(classes.is_sorted_by_key(|c| (c.la(), c.lb())));
        // The table's text ends an empty handoffs line in a space, which the
        // trailing-whitespace hook strips from the committed file.
        let text: String = planned
            .to_string()
            .lines()
            .map(|l| format!("{}\n", l.trim_end()))
            .collect();
        golden(
            env!("CARGO_MANIFEST_DIR"),
            &format!("plan_{stem}.txt"),
            &text,
        );
    }
}

/// A basis with no shells adds no task.
#[test]
fn an_empty_basis_adds_nothing() {
    let mut dag = Dag::new();
    assert!(
        add_records(&mut dag, 0, &fb_basis::BasisSet::default())
            .unwrap()
            .is_empty()
    );
    assert!(dag.plan().is_empty());
}
