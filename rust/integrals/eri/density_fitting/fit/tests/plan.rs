//! The fit's plan at the fixture's shape, naux = 41 and 25 columns (tier
//! 3): [`add_cholesky`] then [`add_batch_fit`], `cholesky` after the
//! metric's producer, `cholesky check` after it, and `density fit` after
//! the tensor's producer, `cholesky` and the check, L and B outputs of
//! `naux²` and `naux · 25` doubles. The plan text, each
//! worker's device and pinned table among it, is the committed golden
//! `tests/golden/plan_naux41_nao5.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device; it does
//! load `libfb_recipes_kernels.so`, whose load check planning runs. And
//! [`add_batch_fit`]'s plan over two batches against one factor.

use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_integrals_eri_density_fitting_fit::add_batch_fit;
use fb_recipes_cholesky::add_cholesky;
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

impl GpuTask for Input {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.output(self.name, self.len)
    }

    fn run(
        &self,
        _: &Self::Rows,
        _: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        _: &DeviceStream,
    ) -> Status {
        unreachable!("the plan is never run")
    }
}

/// The metric and the tensor's producers, on worker 0 of a new plan.
fn inputs(dag: &mut Dag<'_>) -> (RowId<Device, f64>, RowId<Device, f64>) {
    let metric = dag.add_gpu(
        "metric",
        0,
        Input {
            name: "M",
            len: NAUX * NAUX,
        },
    );
    let tensor = dag.add_gpu(
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
    let factor = add_cholesky(&mut dag, 0, "L", NAUX, metric).unwrap();
    let (b, l) = (
        add_batch_fit(&mut dag, 0, NAUX, NAO * NAO, NAO * NAO, factor, tensor),
        factor.0,
    );
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

/// An empty batch is refused at planning.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "a batch of at least one column")]
fn an_empty_batch_is_refused() {
    let mut dag = Dag::new();
    let (metric, tensor) = inputs(&mut dag);
    let factor = add_cholesky(&mut dag, 0, "L", NAUX, metric).unwrap();
    let _ = add_batch_fit(&mut dag, 0, NAUX, 0, 0, factor, tensor);
}

/// Two batches fitted against one factor: one `cholesky` and one check,
/// then a `density fit` per batch, each after its own tensor's producer,
/// `cholesky` and the check, its B `naux` × the batch's columns. Catches a
/// batch fit that refactors the metric or runs before the check.
#[test]
fn batches_share_one_factor() {
    let mut dag = Dag::new();
    let metric = dag.add_gpu(
        "metric",
        0,
        Input {
            name: "M",
            len: NAUX * NAUX,
        },
    );
    let factor = add_cholesky(&mut dag, 0, "L", NAUX, metric).unwrap();
    let mut bs = Vec::new();
    for (name, columns) in [("first batch", 7), ("second batch", 3)] {
        let tensor = dag.add_gpu(
            "three center",
            0,
            Input {
                name,
                len: NAUX * columns,
            },
        );
        bs.push((
            add_batch_fit(&mut dag, 0, NAUX, columns, columns, factor, tensor),
            columns,
        ));
    }
    let planned = dag.plan();
    assert_eq!(planned.len(), 7, "{planned}");
    // metric 0, cholesky 1, check 2, then a tensor and its fit per batch.
    for (fit, tensor) in [(4, 3), (6, 5)] {
        let after = planned.predecessors(fit);
        for t in [1, 2, tensor] {
            assert!(after.contains(&t), "fit {fit} after {t}: {planned}");
        }
    }
    for (b, columns) in bs {
        assert_eq!(planned.device_table(0).len(b), NAUX * columns);
    }
}
