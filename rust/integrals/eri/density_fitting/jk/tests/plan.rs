//! J's and K's plans at the fixtures' shape, naux = 23, nao = 7, nocc = 3
//! (tier 3): `df coulomb` after B's and D's producers, γ its scratch of
//! `naux` doubles and J an output of `nao²`; `df exchange` after B's and
//! C's, X its scratch of `naux · nocc · nao` doubles and K an output of
//! `nao²`. Each plan text, the worker's device table among it, is a
//! committed golden under `tests/golden/`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device; K's
//! does load `libfb_recipes_kernels.so`, whose load check it runs.

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Planned, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_integrals_eri_density_fitting_jk::{add_df_coulomb, add_df_exchange};
use fb_test_support::golden;

/// The fitting set's size.
const NAUX: usize = 23;
/// The orbital basis's size.
const NAO: usize = 7;
/// C's columns.
const NOCC: usize = 3;

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

/// B's producer and one density's (`name`, `len` doubles), on worker 0.
fn inputs(dag: &mut Dag<'_>, name: &'static str, len: usize) -> [RowId<Device, f64>; 2] {
    [("B", NAUX * NAO * NAO), (name, len)].map(|(name, len)| dag.add(name, 0, Input { name, len }))
}

/// `planned`'s text, each line's trailing whitespace stripped (an empty
/// handoffs line ends in a space, which the trailing-whitespace hook strips
/// from the committed file), compared with the golden `file`.
fn plan_golden(planned: &Planned<'_>, file: &str) {
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), file, &text);
}

#[test]
fn coulomb_plan_golden() {
    let mut dag = Dag::new();
    let [b, d] = inputs(&mut dag, "D", NAO * NAO);
    let j = add_df_coulomb(&mut dag, 0, NAO, NAUX, b, d);
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(j), NAO * NAO);
    assert_eq!(planned.predecessors(2), [0, 1]);
    plan_golden(&planned, "plan_coulomb_naux23_nao7.txt");
}

#[test]
fn exchange_plan_golden() {
    let mut dag = Dag::new();
    let [b, c] = inputs(&mut dag, "C", NAO * NOCC);
    let k = add_df_exchange(&mut dag, 0, NAO, NAUX, NOCC, b, c).unwrap();
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(k), NAO * NAO);
    assert_eq!(planned.predecessors(2), [0, 1]);
    plan_golden(&planned, "plan_exchange_naux23_nao7_nocc3.txt");
}

/// A factor of no columns is refused at planning.
#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "a factor of at least one column")]
fn an_empty_factor_is_refused() {
    let mut dag = Dag::new();
    let [b, c] = inputs(&mut dag, "C", NAO);
    let _ = add_df_exchange(&mut dag, 0, NAO, NAUX, 0, b, c);
}
