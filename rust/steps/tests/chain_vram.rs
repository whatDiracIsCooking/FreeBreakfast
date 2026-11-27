//! No library allocates behind the budget on a live handle set (tier 4,
//! `docs/memory.md`, "The budget"; `docs/gpu_layer.md`, "The v0.0.4
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
//! - **The whole chain's first execute and its replay take at most the
//!   kernel stack's bound beyond the budget.** ROCm sizes a launch's stack
//!   for every lane the card can keep resident and keeps it, per hardware
//!   queue, and a stream takes whichever queue is least loaded each time
//!   it goes busy (`docs/memory.md`, "Caps and margins"). So which queues
//!   hold which classes' stacks differs run to run, a replay can put the
//!   largest class on a queue that had not run it, and the one bound that
//!   holds is every queue's whole ascending staircase: [`queues`] times
//!   [`staircase`], and [`ALLOWANCE`]. It is asserted of the larger of the
//!   two readings on the gfx1200 the figures were measured on, and printed
//!   on any AMD card.
//! - **Off the gfx1200, a second execute of the whole chain moves
//!   nothing**: every library call and every launch is made again, and
//!   none takes memory.
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

mod common;

use fb_core_budget::{Backing, bytes};
use fb_core_flow::{Bus, Executor};
use fb_steps::{Config, MainBus, ScfStep, plan};
use fb_test_support::amdgpu_process_vram;

/// The lanes the gfx1200 keeps resident (`rocminfo`: 32 compute units of
/// 32 waves of 32). ROCm sizes a launch's stack for all of them. Measured
/// once: a 32-cell `ssperi3<2, 2, 3>` launch (4,000 bytes per lane) on a
/// fresh queue took 131,182,592 bytes, 4,000 times this and 108 KiB of
/// code.
const LANES: u64 = 32_768;

/// `ssperi3<la, lb, lp>`'s kernel stack per lane on the gfx1200, bytes,
/// for the classes of water in cc-pVDZ against cc-pVDZ-JKFIT: orbital
/// shells to d (rows `(0, 0)`, `(0, 1)`, `(0, 2)`, `(1, 1)`, `(1, 2)`,
/// `(2, 2)`) and aux shells to f (columns `lp` 0 to 3). Copied from the
/// table of all 70 classes in `steps/src/steps/eri.rs`'s tests (`STACK`,
/// the code objects' `.private_segment_fixed_size`), which a test binary
/// cannot reach.
const WATER_STACK: [[u64; 4]; 6] = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 56, 244],
    [0, 0, 288, 552],
    [304, 816, 1632, 2192],
    [1680, 2128, 3536, 4000],
];

/// ROCm's single stack allocation ceiling, bytes: 140 MiB, ROCr's
/// default `HSA_SCRATCH_SINGLE_LIMIT`. A launch whose whole-card stack
/// is past it is sized to its dispatch instead; measured on the gfx1200,
/// `ssperi3<1, 2, 6>` (134.6 MiB) took the whole card's and `ssperi3<2, 2,
/// 4>` (141.5 MiB) its dispatch's. Every size of [`WATER_STACK`] is under
/// it (checked in [`staircase`]).
const CEILING: u64 = 140 << 20;

/// What the first execute takes beside the ERI's stack, bytes, measured
/// once on the gfx1200 and rounded up: the kernels' code objects (2.1 MB
/// at the first `ssperi3` launch), rocBLAS's first-call stack and code at
/// the metric's factor (7.0 MiB) and the SCF's (4 MiB). The libraries'
/// pools the handle set gives back (64 MiB) are not subtracted, so the
/// bound is one-sided; the reading is in KiB, its drift a page.
const ALLOWANCE: u64 = 16 << 20;

/// The most one hardware queue can hold of water's `ssperi3` stacks,
/// bytes: a queue keeps every allocation it has made and takes a new one
/// when a launch needs more than it holds, so the worst is meeting every
/// distinct size of [`WATER_STACK`] in ascending order, their sum times
/// [`LANES`]. 17,428 bytes per lane, 544.6 MiB.
fn staircase() -> u64 {
    let mut sizes: Vec<u64> = WATER_STACK.into_iter().flatten().collect();
    sizes.sort_unstable();
    sizes.dedup();
    assert!(sizes.iter().all(|size| size * LANES <= CEILING));
    sizes.iter().sum::<u64>() * LANES
}

/// The hardware queues HIP spreads the streams over: `GPU_MAX_HW_QUEUES`
/// in the environment, 4 when it is unset, the runtime's default.
fn queues() -> u64 {
    std::env::var("GPU_MAX_HW_QUEUES")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(4)
}

/// Whether the card is the gfx1200 [`WATER_STACK`] was measured on: a GPU node
/// of the KFD topology whose `gfx_target_version` is 120000.
fn on_the_gfx1200() -> bool {
    let Ok(nodes) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else {
        return false;
    };
    nodes.flatten().any(|node| {
        std::fs::read_to_string(node.path().join("properties"))
            .is_ok_and(|text| text.lines().any(|l| l == "gfx_target_version 120000"))
    })
}

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
fn the_libraries_hold_their_budget_and_the_chain_its_kernel_stack_bound() {
    let library = readings(common::config("h2o.xyz", "cc-pvdz", None), false);
    let config = common::chain("h2o.xyz");
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
    let bound = queues() * staircase() + ALLOWANCE;
    let excess = first.max(second).saturating_sub(budgeted);
    println!(
        "the chain's first execute took {} beyond its budget and its replay {} more; on the \
         gfx1200 together at most {} ({} queues × {} + {})",
        bytes(first.saturating_sub(budgeted)),
        bytes(second.saturating_sub(first)),
        bytes(bound),
        queues(),
        bytes(staircase()),
        bytes(ALLOWANCE)
    );
    if on_the_gfx1200() {
        assert!(
            excess <= bound,
            "the chain's first execute and its replay took {} beyond its budget, past {}",
            bytes(excess),
            bytes(bound)
        );
    } else {
        assert_eq!(
            second, first,
            "a second execute of the chain moved this process's VRAM"
        );
    }
}
