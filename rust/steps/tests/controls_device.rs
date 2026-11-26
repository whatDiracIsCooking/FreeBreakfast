//! The step-end sync's negative control (tier 4, device 0): the wrapper
//! made to release a step's locks with its work still in flight
//! (`fb_core_flow::controls::skip_step_end_sync`, fb-core-flow's
//! `test-support` feature, which only this crate's dev-dependency turns
//! on), so that a finished node is no longer finished device work.
//!
//! The case: benzene's DF-RHF on two GPUs of the one card, at forced
//! batches, on two workers, NaN-filled. The metric is a host step that
//! uploads one staging to both GPUs and drops the copies' token, as every
//! host step does, since the step-end sync covers the copies; `eri`, which
//! reads the metric, has its pinned scratch laid over that staging (the
//! DAG orders the two, so the packing may), and the wrapper fills it with
//! NaN before `eri` runs. With the sync skipped the fill lands while the
//! copy to gpu1 is still reading the staging, or `eri` factors gpu1's
//! metric before the copy has run, and the run fails; with the sync
//! restored it gives benzene's energy on one GPU to 1e-9 Eh
//! (`chain_device.rs`'s bar). The control asserts the failure.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]
#![allow(
    clippy::disallowed_macros,
    reason = "a test's case lists, outside any step's run"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use fb_core_budget::{Backing, Kind};
use fb_core_flow::{Bus, controls};
use fb_steps::{EriStep, MainBus, ScfStep, ScfTable, run};

/// The bar on an energy, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// Benzene's recorded DF-RHF energy (`chain_device.rs`).
const BENZENE: f64 = -230.721_729_270_921;

/// Benzene in cc-pVDZ over cc-pVDZ-JKFIT on two GPUs of device 0, at
/// forced batches, on two workers, NaN-filled.
fn config() -> fb_steps::Config {
    let mut config = common::config("benzene.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    let dry = MainBus::setup(config.clone(), Backing::Dry).unwrap();
    let forced = EriStep::least_mem_per_gpu(&dry.shared().dims, 2) * 3 / 2;
    config.eri.as_mut().unwrap().mem_per_gpu = Some(forced);
    config.resources.devices = vec![0, 0];
    (config.resources.threads, config.resources.nan_fill) = (2, true);
    config
}

/// Whether `eri`'s pinned scratch meets the metric's on some GPU in
/// `config`'s budget table: what the fill needs to reach the staging.
fn eris_pinned_scratch_meets_the_metrics(config: fb_steps::Config) -> bool {
    let run = run(config, Backing::Dry).unwrap();
    let regions = run.plan.bus().ledger().regions();
    let on = |gpu: usize, step: &str| {
        regions
            .iter()
            .find(|r| r.gpu == gpu && r.kind == Kind::Pinned && r.step == step)
    };
    (0..2).any(|gpu| match (on(gpu, "eri"), on(gpu, "metric")) {
        (Some(e), Some(m)) => e.offset < m.offset + m.bytes && m.offset < e.offset + e.bytes,
        _ => false,
    })
}

/// With the step-end sync, benzene's energy; then, in the same process
/// (its kernels loaded, so `eri` reaches the metric sooner), without it,
/// a failed run.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn skipping_the_step_end_sync_fails_under_the_nan_fill() {
    assert!(
        eris_pinned_scratch_meets_the_metrics(config()),
        "the control's precondition: eri's pinned scratch is laid over the metric's staging"
    );
    let restored = run(config(), Backing::Real).expect("the chain runs with the sync");
    let scf = restored.plan.output(ScfStep);
    println!(
        "with it: E = {:.12} in {} iterations, |E - recorded| = {:.1e}",
        scf.energy(),
        scf.iterations(),
        (scf.energy() - BENZENE).abs()
    );
    assert!(scf.converged() && (scf.energy() - BENZENE).abs() < ENERGY_BAR);

    controls::skip_step_end_sync(true);
    let skipped = run(config(), Backing::Real);
    controls::skip_step_end_sync(false);
    let skipped = skipped.map(|run| {
        let scf = run.plan.output(ScfStep);
        (scf.converged(), scf.energy())
    });
    println!("without the step-end sync: {skipped:?}");
    assert!(
        !matches!(skipped, Ok((true, e)) if (e - BENZENE).abs() < ENERGY_BAR),
        "the run is right without the step-end sync"
    );
}
