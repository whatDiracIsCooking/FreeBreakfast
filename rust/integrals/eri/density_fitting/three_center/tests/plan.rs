//! Water's plan (tier 3): the plan text, each worker's device and pinned
//! table among it with the arena's size, is a committed golden,
//! `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance"); admission
//! of its device table; and the refusals planning makes of a basis.
//! Planning touches no device, but the load check reads the kernel library,
//! so it must be built.

mod common;

use fb_basis::{BasisSet, Shell};
use fb_core_error_handling::ErrorKind;
use fb_core_planner::Dag;
use fb_integrals_eri_density_fitting_three_center::add_three_center;
use fb_test_support::golden;

#[test]
fn plan_golden() {
    let (basis, aux) = common::water();
    let fixture = common::plan(&basis, &aux);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = fixture
        .planned
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

/// Both tensors are rows of the plan's device table at planning: the
/// Cartesian one (`nao_cart² naux_cart` doubles) and (mn|P)
/// (`nao² naux`), so a budget one byte short of the two is refused by
/// admission as `Allocation`, naming each with its bytes, and the table's
/// own total is admitted.
#[test]
fn admission_refuses_a_budget_short_of_the_two_tensors() {
    let (basis, aux) = common::water();
    let fixture = common::plan(&basis, &aux);
    let table = fixture.planned.device_table(0);
    let (n_cart, n_aux_cart) = (basis.nao_cart() as usize, aux.nao_cart() as usize);
    let cartesian = 8 * n_cart * n_cart * n_aux_cart;
    let spherical = 8 * fixture.n_ao * fixture.n_ao * fixture.n_aux;
    let err = fb_core_arena::admit(table, cartesian + spherical - 1, "gpu0").unwrap_err();
    assert_eq!(err.kind, ErrorKind::Allocation, "{err}");
    for named in [
        format!("(mn|P) cartesian ({cartesian} B)"),
        format!("(mn|P) ({spherical} B)"),
    ] {
        assert!(err.message.contains(&named), "{named} not in: {err}");
    }
    fb_core_arena::admit(table, table.total(), "gpu0").unwrap();
}

/// An orbital shell past f and an aux shell past i are refused at planning
/// as `Unsupported`, naming the shell, before anything is added: the
/// launcher would refuse their classes only at run time.
#[test]
fn a_shell_past_the_kernels_is_unsupported_at_planning() {
    let shell = |l| Shell::zipped(l, &[1.0], &[1.0], Default::default());
    let fine = BasisSet::new(vec![shell(0), shell(1)]);
    for (basis, aux, named) in [
        (
            BasisSet::new(vec![shell(0), shell(4)]),
            fine.clone(),
            "orbital shell 1 has momentum 4",
        ),
        (
            fine.clone(),
            BasisSet::new(vec![shell(7)]),
            "aux shell 0 has momentum 7",
        ),
    ] {
        let mut dag = Dag::new();
        let classes = fb_integrals_mmd::add_records(&mut dag, 0, &basis).unwrap();
        let err = add_three_center(&mut dag, 0, &basis, &classes, &aux).unwrap_err();
        assert_eq!(err.kind, ErrorKind::Unsupported, "{err}");
        assert!(err.message.contains(named), "{err}");
        let mut records = Dag::new();
        fb_integrals_mmd::add_records(&mut records, 0, &basis).unwrap();
        assert_eq!(
            dag.plan().to_string(),
            records.plan().to_string(),
            "nothing added"
        );
    }
}
