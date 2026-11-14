//! The fit's plan at the fixture's shape, naux = 41 and nao = 5 (tier 3):
//! `cholesky` after the metric's producer, `cholesky check` after it, and
//! `density fit` after the tensor's producer, `cholesky` and the check, L
//! and B outputs of `naux²` and `naux · nao²` doubles. The plan text, each
//! worker's device and pinned table among it, is the committed golden
//! `tests/golden/plan_naux41_nao5.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device; it does
//! load `libfb_recipes_kernels.so`, whose load check planning runs.

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_integrals_eri_density_fitting_fit::add_density_fit;
use fb_test_support::golden;

/// The fitting set's size.
const NAUX: usize = 41;
/// The orbital basis's size.
const NAO: usize = 5;

/// GPU: an input, declared and never computed.
struct Input {
    name: &'static str,
    len: usize,
}

impl Task for Input {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.name, self.len)
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

/// The metric and the tensor's producers, on worker 0 of a new plan.
fn inputs(dag: &mut Dag<'_>) -> (RowId<Device, f64>, RowId<Device, f64>) {
    let metric = dag.add(
        "metric",
        0,
        Input {
            name: "M",
            len: NAUX * NAUX,
        },
    );
    let tensor = dag.add(
        "three center",
        0,
        Input {
            name: "(P|mn)",
            len: NAUX * NAO * NAO,
        },
    );
    (metric, tensor)
}

#[test]
fn plan_golden() {
    let mut dag = Dag::new();
    let (metric, tensor) = inputs(&mut dag);
    let (b, l) = add_density_fit(&mut dag, 0, NAO, NAUX, metric, tensor).unwrap();
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(b), NAUX * NAO * NAO);
    assert_eq!(planned.device_table(0).len(l), NAUX * NAUX);
    // cholesky after the metric; the check after cholesky; the fit after
    // the tensor, cholesky (L) and the check (checked), and after the
    // metric's producer too: B takes M's bytes once cholesky has read them,
    // a handoff (`docs/memory.md`, "Lifetimes").
    assert_eq!(planned.predecessors(2), [0]);
    assert_eq!(planned.predecessors(3), [2]);
    assert_eq!(planned.predecessors(4), [0, 1, 2, 3]);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_naux41_nao5.txt", &text);
}

/// An empty orbital basis is refused at planning.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "an orbital basis of at least one function")]
fn an_empty_orbital_basis_is_refused() {
    let mut dag = Dag::new();
    let (metric, tensor) = inputs(&mut dag);
    let _ = add_density_fit(&mut dag, 0, 0, NAUX, metric, tensor);
}
