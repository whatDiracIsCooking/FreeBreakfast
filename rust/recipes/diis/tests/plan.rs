//! The plan of one DIIS history and its push in a loop's body, at
//! `len = 100` and a history of 8 (tier 3): `diis history` before the
//! loop, producing the two rings of `len × 8` doubles and the 8 × 8 Gram;
//! the pair's producer and `diis push` in the body, the push after both,
//! writing the history, which is therefore live across the whole loop and
//! outside its region; the decision reading the pinned flag. The plan
//! text, each worker's device and pinned table among it, is the committed
//! golden `tests/golden/plan_len100_cap8.txt`, `FB_BLESS=1` rewriting it
//! (`docs/testing.md`, "Performance"). Planning touches no device; it does
//! load `libfb_recipes_kernels.so`, whose load check planning runs.

use std::ops::ControlFlow;

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Decision, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_recipes_diis::add_diis;
use fb_test_support::golden;

/// The vector length and the history.
const LEN: usize = 100;
const CAP: usize = 8;

/// GPU: an iterate and its residual, declared and never computed.
struct Pair;

impl Task for Pair {
    type Rows = (RowId<Device, f64>, RowId<Device, f64>);
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        (device.output("F", LEN), device.output("e", LEN))
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

/// The decision, reading the flag: declared and never run.
struct Reads(RowId<Pinned, i32>);

impl Decision for Reads {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>) {
        host.read(self.0);
    }

    fn decide(&self, _: &(), _: &Bufs<'_, Pinned>, _: usize) -> Result<ControlFlow<()>> {
        unreachable!("the plan is never run")
    }
}

#[test]
fn plan_golden() {
    let mut dag = Dag::new();
    let diis = add_diis(&mut dag, 0, "F*", LEN, CAP).unwrap();
    let body = dag.len();
    let (f, e) = dag.add("pair", 0, Pair);
    let (out, singular) = diis.add_push(&mut dag, f, e);
    dag.add_loop("decide", 0, body, 50, Reads(singular));
    let planned = dag.plan();
    assert_eq!(planned.device_table(0).len(out), LEN);
    assert_eq!(planned.host_table(0).len(singular), 1);
    // The push after the history and the pair.
    assert_eq!(planned.predecessors(2), [0, 1]);
    // The table's text ends an empty handoffs line in a space, which the
    // trailing-whitespace hook strips from the committed file.
    let text: String = planned
        .to_string()
        .lines()
        .map(|l| format!("{}\n", l.trim_end()))
        .collect();
    golden(env!("CARGO_MANIFEST_DIR"), "plan_len100_cap8.txt", &text);
}

/// An empty vector and a history past the kernel's limit are refused at
/// planning.
#[test]
#[cfg(debug_assertions)] // the contracts are compiled out of release
fn an_empty_vector_and_a_history_past_the_limit_are_refused() {
    let message = fb_test_support::panic_message(|| {
        let _ = add_diis(&mut Dag::new(), 0, "F*", 0, CAP);
    });
    assert!(
        message.contains("a vector of at least one element"),
        "{message}"
    );
    let message = fb_test_support::panic_message(|| {
        let _ = add_diis(&mut Dag::new(), 0, "F*", LEN, 65);
    });
    assert!(
        message.contains("a history of 1 to DIIS_MAX_HISTORY_LIMIT pairs"),
        "{message}"
    );
}
