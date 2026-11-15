//! The records' plan for both molecules (tier 3): the plan text, the
//! device table's size (the arena's) among it, is a committed golden per
//! molecule, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance").
//! Planning touches no device, but `add_records` runs the kernel library's
//! load check, so the library must be built.

mod common;

use fb_core_planner::Dag;
use fb_integrals_mmd::{add_mixed_records, add_records};
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

/// Two bases' records (water in cc-pVDZ against water in STO-3G at another
/// geometry): the classes and each `n_rec` are the ordered shell pairs'
/// primitive products counted from the bases directly, and the plan text
/// is the committed golden `plan_mixed_h2o_ccpvdz_sto3g.txt`.
#[test]
fn mixed_plan_golden() {
    let (rows, cols) = common::two_bases();
    let mut dag = Dag::new();
    let classes = add_mixed_records(&mut dag, 0, &rows, &cols).unwrap();
    let mut want = std::collections::BTreeMap::<(i32, i32), i32>::new();
    for a in rows.shells() {
        for b in cols.shells() {
            *want
                .entry((a.angular_momentum(), b.angular_momentum()))
                .or_insert(0) += (a.nprimitives() * b.nprimitives()) as i32;
        }
    }
    let got: Vec<_> = classes
        .iter()
        .map(|c| ((c.la(), c.lb()), c.n_rec()))
        .collect();
    assert_eq!(got, want.into_iter().collect::<Vec<_>>());
    assert!(
        got.iter().any(|&((la, lb), _)| la > lb),
        "a class with la > lb"
    );
    assert!(
        classes
            .iter()
            .all(|c| c.n_cart() == [rows.nao_cart() as usize, cols.nao_cart() as usize])
    );
    let text: String = dag
        .plan()
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "plan_mixed_h2o_ccpvdz_sto3g.txt",
        &text,
    );
}

/// Either basis with no shells adds no task.
#[test]
fn an_empty_basis_adds_no_mixed_records() {
    let (rows, _) = common::two_bases();
    let empty = fb_basis::BasisSet::default();
    let mut dag = Dag::new();
    assert!(
        add_mixed_records(&mut dag, 0, &rows, &empty)
            .unwrap()
            .is_empty()
    );
    assert!(
        add_mixed_records(&mut dag, 0, &empty, &rows)
            .unwrap()
            .is_empty()
    );
    assert!(dag.plan().is_empty());
}
