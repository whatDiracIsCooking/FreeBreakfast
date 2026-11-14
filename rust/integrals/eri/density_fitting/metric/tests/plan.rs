//! Each fixture's plan (tier 3): the plan text, each worker's device and
//! pinned table among it with the arena's size, is a committed golden per
//! fixture, `FB_BLESS=1` rewriting it (`docs/testing.md`, "Performance").
//! Planning touches no device, but the load check reads the kernel library,
//! so it must be built. And the one refusal planning makes of a basis.

mod common;

use fb_basis::{BasisSet, Shell};
use fb_core_error_handling::ErrorKind;
use fb_core_planner::Dag;
use fb_integrals_eri_density_fitting_metric::add_metric;
use fb_test_support::golden;

#[test]
fn plan_goldens() {
    for (stem, xyz, aux, _) in common::FIXTURES {
        let fixture = common::plan(xyz, aux);
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

/// An aux shell past the kernels' i is refused at planning as
/// `Unsupported`, naming the shell, before anything is added: the launcher
/// would refuse its class only at run time.
#[test]
fn a_shell_past_i_is_unsupported_at_planning() {
    let shell = |l| Shell::zipped(l, &[1.0], &[1.0], Default::default());
    let aux = BasisSet::new(vec![shell(0), shell(7)]);
    let mut dag = Dag::new();
    let err = add_metric(&mut dag, 0, &aux, &mut None).unwrap_err();
    assert_eq!(err.kind, ErrorKind::Unsupported, "{err}");
    assert!(err.message.contains("aux shell 1 has momentum 7"), "{err}");
    let empty: Dag<'_> = Dag::new();
    assert_eq!(dag.plan().to_string(), empty.plan().to_string());
}
