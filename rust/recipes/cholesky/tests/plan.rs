//! The factorization's plan at n = 200 (tier 3): `cholesky` after the
//! matrix's producer, `cholesky check` after it alone, L an output of n²
//! doubles and the packed tile a scratch of `CHOLESKY_BLOCK`². The plan text, each
//! worker's device and pinned table among it, is the committed golden
//! `tests/golden/plan_n200.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device; it does
//! load `libfb_recipes_kernels.so`, whose load check planning runs.

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_recipes_cholesky::add_cholesky;
use fb_test_support::golden;

/// The order: three full tiles and a partial one.
const N: usize = 200;

/// GPU: the symmetric matrix, declared and never computed.
struct Matrix;

impl Task for Matrix {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output("A", N * N)
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
    let a = dag.add("A", 0, Matrix);
    let (l, checked) = add_cholesky(&mut dag, 0, "L", N, a).unwrap();
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(l), N * N);
    assert_eq!(planned.host_table(0).len(checked), 1);
    // cholesky after A; the check after cholesky alone.
    assert_eq!(planned.predecessors(1), [0]);
    assert_eq!(planned.predecessors(2), [1]);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_n200.txt", &text);
}

/// An empty matrix is refused at planning.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "a matrix of at least one row")]
fn an_empty_matrix_is_refused() {
    let mut dag = Dag::new();
    let a = dag.add("A", 0, Matrix);
    let _ = add_cholesky(&mut dag, 0, "L", 0, a);
}
