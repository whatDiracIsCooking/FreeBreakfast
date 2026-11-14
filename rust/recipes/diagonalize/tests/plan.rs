//! The eigensolver's plan (tier 3): `syevd` and `syevd check` after a
//! matrix's producer, the work row at the `lwork` it was handed. The plan
//! text, each worker's device and pinned table among it, is the committed
//! golden `tests/golden/plan_n24.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device: `lwork`
//! is a number here, where a run asks the device's handle for it.

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_recipes_diagonalize::add_diagonalize;
use fb_test_support::golden;

/// The order: the H2O/cc-pVDZ overlap's.
const N: usize = 24;
/// A work size handed to the layout, not the vendor's: the golden shows the
/// row at exactly this many doubles.
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
    let (u, w) = add_diagonalize(&mut dag, 0, "S", N, LWORK, s);
    let planned = dag.plan();
    let table = planned.device_table(0);
    assert_eq!(table.len(u), N * N);
    assert_eq!(table.len(w), N);
    // syevd after S; the check after syevd alone.
    assert_eq!(planned.predecessors(1), [0]);
    assert_eq!(planned.predecessors(2), [1]);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    let bytes = (LWORK * size_of::<f64>()).to_string();
    assert!(
        text.lines().any(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            f.len() > 1 && f[1] == bytes && l.contains(" syevd work: ")
        }),
        "the work row is lwork doubles:\n{text}"
    );
    golden(env!("CARGO_MANIFEST_DIR"), "plan_n24.txt", &text);
}

/// An empty matrix is refused at planning.
#[test]
#[should_panic(expected = "add_diagonalize: an empty matrix")]
fn an_empty_matrix_is_refused() {
    let mut dag = Dag::new();
    let s = dag.add("S", 0, Matrix);
    add_diagonalize(&mut dag, 0, "S", 0, 0, s);
}
