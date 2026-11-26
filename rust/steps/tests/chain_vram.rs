//! No library allocates behind the budget on a live handle set (tier 4,
//! `docs/memory.md`, "What is checked"; `docs/gpu_layer.md`, "The v0.0.4
//! runtime"): this process's VRAM, read from amdgpu's fdinfo
//! ([`amdgpu_process_vram`]), once budget mode has allocated the slabs and
//! the handle workspace, after a plan's first execute and after a second.
//!
//! - **Water's library steps hold no more than the budget allocated.** The
//!   four host steps and the orthogonalizer, whose syevd and BLAS calls
//!   are the plan's library calls: binding the workspace gives back
//!   rocBLAS's own pool, and the solver's first call with a caller's work
//!   buffer gives back its pool, each more than the libraries' first-call
//!   code objects, and no call grows, since every BLAS call's need is
//!   inside the bound workspace (no library `trsm`, the one call whose
//!   need grows, `docs/gpu_layer.md`, "Layering") and every solver call is
//!   handed its queried work. So its VRAM after the first execute is at
//!   most what it was once the budget had allocated.
//! - **A second execute of the whole chain moves nothing**: every library
//!   call and every launch is made again, and none takes memory. Its first
//!   execute is not bounded so: the ERI's launches load their code objects
//!   and take the runtime's scratch for their stack, which HIP sizes per
//!   dispatch, outside any budget (`docs/memory.md`, "Caps and margins": the margin covers the
//!   kernels' stack); it is printed.
//!
//! On CUDA nothing counts a library's allocations per process in a
//! container (`docs/rules.md`, the row "No library allocates behind the
//! arena"), so the readings are `None` and nothing is asserted.
//!
//! The case is alone in its binary, its two plans one after the other:
//! the reading is the whole process's, and a case beside it would move it.

#![allow(
    clippy::disallowed_macros,
    reason = "a test's report of its readings, outside any step's run"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use fb_core_budget::{Backing, bytes};
use fb_core_flow::{Bus, Executor};
use fb_steps::{Config, MainBus, ScfStep, ScfTable, plan};
use fb_test_support::amdgpu_process_vram;

/// `config`'s VRAM readings once budgeted, after the first execute and
/// after the second, with the device bytes its budget allocated;
/// `converged` asks the SCF's output to have converged.
fn readings(config: Config, converged: bool) -> ([Option<u64>; 3], u64) {
    let executor = Executor::new(config.resources.threads);
    let plan = plan(MainBus::setup(config, Backing::Real).unwrap()).unwrap();
    plan.budget(&executor).unwrap();
    let budgeted = amdgpu_process_vram();
    plan.execute(&executor).unwrap();
    let first = amdgpu_process_vram();
    plan.execute(&executor).unwrap();
    let second = amdgpu_process_vram();
    if converged {
        assert!(plan.output(ScfStep).converged());
    }
    (
        [budgeted, first, second],
        plan.bus().ledger().device_total(0),
    )
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_library_steps_hold_no_more_than_their_budget_and_a_replay_takes_nothing() {
    let library = readings(common::config("h2o.xyz", "cc-pvdz", None), false);
    let mut config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    let chain = readings(config, true);
    for (what, ([budgeted, first, second], allocated)) in
        [("library steps", library), ("chain", chain)]
    {
        println!(
            "{what}: this process's VRAM (amdgpu fdinfo) {budgeted:?} once {} is allocated, \
             {first:?} after the first execute, {second:?} after the second",
            bytes(allocated)
        );
    }
    let ([Some(budgeted), Some(first), Some(_)], _) = library else {
        return;
    };
    assert!(
        first <= budgeted,
        "the libraries took {} beyond the budget's allocation in the first execute",
        bytes(first - budgeted)
    );
    let ([Some(budgeted), Some(first), Some(second)], _) = chain else {
        return;
    };
    println!(
        "the chain's first execute took {} beyond its budget (the ERI kernels' code objects and stack scratch)",
        bytes(first.saturating_sub(budgeted))
    );
    assert_eq!(
        second, first,
        "a second execute of the chain moved this process's VRAM"
    );
}
