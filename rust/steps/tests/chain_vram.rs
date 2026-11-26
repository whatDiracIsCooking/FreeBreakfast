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
//! - **The whole chain's first execute takes at most its kernels' scratch
//!   beyond the budget.** ROCm sizes a launch's scratch for every lane the
//!   card can keep resident, per queue, and keeps it
//!   (`docs/memory.md`, "Caps and margins"). The eri step launches its
//!   largest class first on each ring stream, so each stream takes one
//!   allocation, its largest class's, and water's largest, `ssperi3<2, 2,
//!   3>`, is under ROCm's single-allocation ceiling ([`CEILING`], checked):
//!   the excess is at most `ring × LARGEST` and [`ALLOWANCE`], asserted on
//!   the gfx1200 the figures were measured on and printed on any AMD card.
//! - **A second execute of the whole chain moves nothing**: every library
//!   call and every launch is made again, and none takes memory.
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

/// `ssperi3<2, 2, 3>`'s scratch on the gfx1200, bytes: 4,000 per lane,
/// its `.private_segment_fixed_size` in the code object (`llvm-readelf
/// --notes`), times the 32,768 lanes the card keeps resident (`rocminfo`:
/// 32 compute units of 32 waves of 32). Measured once: a 32-cell launch on
/// a fresh queue took 131,182,592 bytes, this and 108 KiB of code. It is
/// the largest of water's classes in cc-pVDZ against cc-pVDZ-JKFIT, whose
/// aux shells stop at f, and under [`CEILING`].
const LARGEST: u64 = 4000 * 32_768;

/// ROCm's single scratch allocation ceiling, bytes: 140 MiB, ROCr's
/// default `HSA_SCRATCH_SINGLE_LIMIT`. A launch whose whole-card scratch
/// is past it is sized to its dispatch instead; measured on the gfx1200,
/// `ssperi3<1, 2, 6>` (134.6 MiB) took the whole card's and `ssperi3<2, 2,
/// 4>` (141.5 MiB) its dispatch's.
const CEILING: u64 = 140 << 20;

/// What the first execute takes beside the ERI's scratch, bytes, measured
/// once on the gfx1200 and rounded up: the kernels' code objects (2.1 MB
/// at the first `ssperi3` launch), rocBLAS's first-call scratch and code at
/// the metric's factor (7.0 MiB) and the SCF's (4 MiB). The libraries'
/// pools the handle set gives back (64 MiB) are not subtracted, so the
/// bound is one-sided; the reading is in KiB, its drift a page.
const ALLOWANCE: u64 = 16 << 20;

/// Whether the card is the gfx1200 [`LARGEST`] was measured on: a GPU node
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
fn the_library_steps_hold_no_more_than_their_budget_and_a_replay_takes_nothing() {
    let library = readings(common::config("h2o.xyz", "cc-pvdz", None), false);
    let mut config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.scf = Some(ScfTable::default());
    let ring = config.eri.as_ref().map_or(1, |eri| eri.ring) as u64;
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
    const { assert!(LARGEST <= CEILING) };
    let bound = ring * LARGEST + ALLOWANCE;
    let excess = first.saturating_sub(budgeted);
    println!(
        "the chain's first execute took {} beyond its budget; on the gfx1200 at most {} \
         (ring {ring} × {} + {})",
        bytes(excess),
        bytes(bound),
        bytes(LARGEST),
        bytes(ALLOWANCE)
    );
    assert!(
        !on_the_gfx1200() || excess <= bound,
        "the chain's first execute took {} beyond its budget, past {}",
        bytes(excess),
        bytes(bound)
    );
    assert_eq!(
        second, first,
        "a second execute of the chain moved this process's VRAM"
    );
}
