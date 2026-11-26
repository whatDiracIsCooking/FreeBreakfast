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
//!
//! The screen, on benzene at 1e-4, which drops 42 of its 1485 shell pairs
//! (water drops none short of 1e-1): a dropped
//! pair's columns of B are zero, and the unscreened B's same columns have
//! a 2-norm under the threshold (`Σ_Q B_Q,mn² = (mn|mn)_DF <= (mn|mn)`,
//! the fit a projection, and a pair's bound at least its largest
//! `sqrt((mn|mn))`); every other column is the unscreened one within the
//! bar above. At 0 nothing is dropped.

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
    let (b, batches, _, _) = screened("h2o.xyz", ring, Some(mem_per_gpu), 0.0);
    (b, batches)
}

/// `xyz`'s B at `[eri] schwarz_threshold = threshold`, through a ring of
/// `ring` at `mem_per_gpu` (the default when `None`), its batches, the
/// pairs the screen dropped, and B's row length.
fn screened(
    xyz: &str,
    ring: usize,
    mem_per_gpu: Option<u64>,
    threshold: f64,
) -> (Vec<f64>, usize, usize, usize) {
    let mut config = common::config(xyz, "cc-pvdz", Some("cc-pvdz-jkfit"));
    let eri = config.eri.as_mut().unwrap();
    (eri.mem_per_gpu, eri.ring, eri.schwarz_threshold) = (mem_per_gpu, ring, threshold);
    let plan = plan(MainBus::setup(config, Backing::Real).unwrap()).unwrap();
    let executor = Executor::new(4);
    plan.budget(&executor).unwrap();
    plan.execute(&executor).unwrap();
    let eri = plan.output(EriStep);
    (
        eri.b().to_vec(),
        eri.batches(),
        eri.dropped_pairs(),
        eri.n_col(),
    )
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

/// The screen on benzene ([the module doc](self)): at `THRESHOLD` it drops
/// pairs, their columns zero and under the threshold unscreened, and
/// leaves every other column; at 0 it drops none.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_screen_zeroes_only_columns_under_the_threshold_on_benzene() {
    const THRESHOLD: f64 = 1e-4;
    const XYZ: &str = "benzene.xyz";
    let (all, _, none, n_col) = screened(XYZ, 2, None, 0.0);
    assert_eq!(none, 0, "threshold 0 drops nothing");
    let (some, _, dropped, _) = screened(XYZ, 2, None, THRESHOLD);
    let largest = all.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let column = |b: &[f64], c: usize| b.iter().skip(c).step_by(n_col).copied().collect::<Vec<_>>();
    let (mut zero, mut worst_dropped, mut worst_kept) = (0, 0.0f64, 0.0f64);
    for c in 0..n_col {
        let (a, s) = (column(&all, c), column(&some, c));
        if s.iter().all(|&v| v == 0.0) {
            zero += 1;
            worst_dropped = worst_dropped.max(a.iter().map(|v| v * v).sum::<f64>().sqrt());
        } else {
            let diff = a
                .iter()
                .zip(&s)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
            worst_kept = worst_kept.max(diff);
        }
    }
    println!(
        "{XYZ} at {THRESHOLD:e}: {dropped} pairs dropped, {zero} of {n_col} columns zero; \
         largest dropped column norm unscreened {worst_dropped:e}, max |ΔB| kept {worst_kept:e}"
    );
    assert!(dropped > 0 && zero > 0, "the threshold drops pairs");
    assert!(
        worst_dropped < THRESHOLD,
        "{worst_dropped:e} >= {THRESHOLD:e}"
    );
    assert!(
        !worst_kept.is_nan() && worst_kept <= 1e-10 * largest,
        "{worst_kept:e} > 1e-10 × {largest:e}"
    );
}
