//! The step-end sync's negative control (tier 4, device 0): the wrapper
//! made to release a step's locks with its work still in flight, so that
//! a finished node is no longer finished device work
//! (`fb_core_flow::controls`, fb-core-flow's `test-support` feature,
//! which only this crate's dev-dependency turns on).
//!
//! The case: water's DF-RHF, NaN-filled. Every host step (the overlap, the
//! core Hamiltonian, the guess, the metric) uploads its pinned staging on
//! its own stream and drops the copies' token, since the step-end sync
//! covers the copies. The control's delay puts some 75 ms of device copies
//! ahead of those uploads on that stream, so the order is fixed by
//! construction, not by a race:
//!
//! - With the sync skipped, the wrapper refills the step's host scratch
//!   with NaN as it releases, as the next holder of those bytes would, and
//!   the delayed copies then carry NaN. A reader finds each host step's
//!   output either NaN, once the copy has landed, or as it was allocated,
//!   which in this binary's first run (its only case, and the first device
//!   work of its process) has never held that output. The run fails.
//! - With the sync kept, the copies have run before the step ends, so the
//!   delay changes nothing: the run gives water's energy to 1e-9 Eh
//!   (`chain_device.rs`'s bar).
//!
//! The control asserts the failure.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
#![allow(
    clippy::disallowed_macros,
    reason = "a test's case lists, outside any step's run"
)]

mod common;

use fb_core_budget::Backing;
use fb_core_flow::controls;
use fb_steps::{ScfStep, run};

use common::ENERGY_BAR;

/// Water's recorded DF-RHF energy (`chain_device.rs`).
const WATER: f64 = -76.026_751_140_543;

/// Water in cc-pVDZ over cc-pVDZ-JKFIT, `[scf]` at its defaults,
/// NaN-filled.
fn config() -> fb_steps::Config {
    let mut config = common::chain("h2o.xyz");
    config.resources.nan_fill = true;
    config
}

/// Without the step-end sync, a failed run; with it, under the same
/// delay, water's energy.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn skipping_the_step_end_sync_fails_under_the_nan_fill() {
    controls::delay_host_steps(&[0]);
    controls::skip_step_end_sync(true);
    let skipped = run(config(), Backing::Real);
    controls::skip_step_end_sync(false);
    let skipped = skipped.map(|run| {
        let scf = run.plan.output(ScfStep);
        (scf.converged(), scf.energy())
    });
    println!("without the step-end sync: {skipped:?}");
    let restored = run(config(), Backing::Real);
    controls::delay_host_steps(&[]);
    assert!(
        !matches!(skipped, Ok((true, e)) if (e - WATER).abs() < ENERGY_BAR),
        "the run is right without the step-end sync"
    );
    let restored = restored.expect("the chain runs with the sync");
    let scf = restored.plan.output(ScfStep);
    println!(
        "with it: E = {:.12} in {} iterations, |E - recorded| = {:.1e}",
        scf.energy(),
        scf.iterations(),
        (scf.energy() - WATER).abs()
    );
    assert!(scf.converged() && (scf.energy() - WATER).abs() < ENERGY_BAR);
}
