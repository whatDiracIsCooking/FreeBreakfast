//! The ERI step on a card (tier 4, device 0): B from the build's ring of
//! two equals B from the ring of one, on water in cc-pVDZ / cc-pVDZ-JKFIT
//! at forced batches (`[eri] mem_per_gpu` twice the least that holds a
//! batch, so several pair batches go through each ring).
//!
//! The bar is 1e-10 of B's largest element (`docs/testing.md`, "The
//! steps' tests"): the same
//! kernels and fits in a different order and at different batch widths,
//! the blocked triangular solve's blocking changing with the width, its
//! forward error a modest multiple of `ε κ(L)` on a metric whose
//! condition water's JKFIT keeps far under 1e5.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use fb_core_budget::Backing;
use fb_core_flow::{Bus, Executor};
use fb_steps::{EriStep, MainBus, plan};

/// Water's B, through a ring of `ring`, at `mem_per_gpu`, and its batches.
fn b_of(ring: usize, mem_per_gpu: u64) -> (Vec<f64>, usize) {
    let mut config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    let eri = config.eri.as_mut().unwrap();
    (eri.mem_per_gpu, eri.ring) = (Some(mem_per_gpu), ring);
    let plan = plan(MainBus::setup(config, Backing::Real).unwrap()).unwrap();
    let executor = Executor::new(4);
    plan.budget(&executor).unwrap();
    plan.execute(&executor).unwrap();
    let eri = plan.output(EriStep);
    (eri.b().to_vec(), eri.batches())
}

#[test]
#[ignore = "tier 4: needs a GPU"]
fn b_from_the_ring_of_two_equals_the_ring_of_one_on_water() {
    let config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    let dims = MainBus::setup(config, Backing::Dry)
        .unwrap()
        .shared()
        .dims
        .clone();
    let mem = 2 * EriStep::least_mem_per_gpu(&dims, 2).max(EriStep::least_mem_per_gpu(&dims, 1));
    let (two, batches_two) = b_of(2, mem);
    let (one, batches_one) = b_of(1, mem);
    println!("water: {batches_two} pair batches through two, {batches_one} through one");
    assert!(batches_two > 2 && batches_one > 1, "several batches each");
    let largest = two.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = two
        .iter()
        .zip(&one)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    println!("max |B₂ - B₁| = {worst:e}, max |B| = {largest:e}");
    assert!(largest > 0.1, "B is not empty");
    assert!(
        !worst.is_nan() && worst <= 1e-10 * largest,
        "{worst:e} > 1e-10 × {largest:e}"
    );
}
