//! Each fixture's plan (tier 3): the plan text, each worker's device and
//! pinned table among it with the arena's size, is a committed golden per
//! fixture, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance"):
//! one build task, `core hamiltonian`, reading the records and the nuclei's
//! rows as edges, and one Cartesian row, `H cartesian`, with no row of T or
//! V. And the planning's refusal of nuclei with no nucleus. Planning touches
//! no device, but the records' layout asks the kernel library for the sort's
//! workspace, so the library must be built.

mod common;

use fb_core_error_handling::ErrorKind;
use fb_core_planner::Dag;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_molecule_nuclei::add_nuclei;
use fb_test_support::golden;

#[test]
fn plan_goldens() {
    for (stem, xyz, basis) in common::FIXTURES {
        let fixture = common::plan(xyz, basis);
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
            &format!("plan_{stem}.txt"),
            &text,
        );
    }
}

/// Nuclei with no nucleus, outside the nuclear-attraction wrapper's
/// `n_nuc >= 1`, are refused at planning with an `InvalidConfig` error, and
/// nothing is added: the plan holds the nuclei's two tasks alone, so neither
/// T nor V would launch.
#[test]
fn no_nucleus_is_refused_at_planning() {
    let molecule = fb_io_xyz::read_xyz(&common::data("h2o.xyz"), 0, 1).unwrap();
    let basis = fb_io_basis::resolve_basis("sto-3g")
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
    for n_nuc in [0, -1] {
        let mut dag = Dag::new();
        let mut nuclei = add_nuclei(&mut dag, 0, &molecule);
        nuclei.n_nuc = n_nuc;
        let mut u = None;
        let err = add_core_hamiltonian(&mut dag, 0, &basis, nuclei, &mut u)
            .expect_err("no nucleus is refused");
        assert_eq!(err.kind, ErrorKind::InvalidConfig, "{err}");
        assert!(err.to_string().contains("nuclei"), "{err}");
        assert!(u.is_none(), "U was not staged");
        assert_eq!(dag.plan().len(), 2, "only the nuclei's tasks");
    }
}
