//! The orthogonalizer's plan (tier 3): the eigensolver's two tasks after a
//! matrix's producer, then `inverse root` after `syevd`, reading U and w,
//! its Λ and M scratch and X its output. The plan text, each worker's device
//! and pinned table among it, is the committed golden
//! `tests/golden/plan_n24.txt`, `FB_BLESS=1` rewriting it (`docs/testing.md`,
//! "Performance"). Planning touches no device: `lwork` is a number here,
//! where a run asks the device's handle for it. It does load
//! `libfb_recipes_kernels.so`, whose load check planning runs.

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};
use fb_test_support::golden;

/// The order: the H2O/cc-pVDZ overlap's.
const N: usize = 24;
/// A work size handed to the layout, not the vendor's.
const LWORK: usize = 1234;

/// GPU: the symmetric matrix, declared and never computed.
struct Matrix;

impl Task for Matrix {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output("S", N * N)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

#[test]
fn plan_golden() {
    let mut dag = Dag::new();
    let s = dag.add("S", 0, Matrix);
    let x = add_orthogonalize(
        &mut dag,
        0,
        "X",
        N,
        LWORK,
        s,
        ORTHOGONALIZE_EIGENVALUE_FLOOR,
    )
    .unwrap();
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(x), N * N);
    // syevd after S; the check after syevd alone; the inverse root after
    // syevd, not after the check (and after S too, whose bytes it may
    // reuse).
    assert_eq!(planned.predecessors(1), [0]);
    assert_eq!(planned.predecessors(2), [1]);
    assert!(planned.predecessors(3).contains(&1));
    assert!(!planned.predecessors(3).contains(&2));
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_n24.txt", &text);
}

/// An empty matrix is refused at planning, by the eigensolver.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "a matrix of at least one row")]
fn an_empty_matrix_is_refused() {
    let mut dag = Dag::new();
    let s = dag.add("S", 0, Matrix);
    let _ = add_orthogonalize(&mut dag, 0, "X", 0, 0, s, ORTHOGONALIZE_EIGENVALUE_FLOOR);
}
