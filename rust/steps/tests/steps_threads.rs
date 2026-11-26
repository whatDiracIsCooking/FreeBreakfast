//! The steps use exactly the threads their config says (tier 3,
//! `docs/programming_model.md`, "CPU workers"): water in cc-pVDZ /
//! cc-pVDZ-JKFIT, its four host steps executed once on a host-backed plan,
//! leave the main thread, the executor's `threads` and the host integrals'
//! `host_threads`, and no more, so no host integral ran on rayon's global
//! pool; and once the plan and the executor drop, the main thread alone.
//!
//! No pool a linked C library opened is left either. A HIP build links
//! ROCm's host OpenBLAS (through hipSOLVER), which starts one thread per
//! core before `main`; setup sets it to one thread and joins that pool
//! (`fb_core_device::single_thread_host_blas`), so the count here is
//! against the main thread alone, and a pool that slips back fails it.
//!
//! The count is the `Threads:` line of `/proc/self/status`. `harness =
//! false`: libtest runs cases on threads of its own, which would be counted.

#[allow(
    dead_code,
    reason = "the shared module's orthogonalizer check is the other binaries'"
)]
mod common;

use std::time::{Duration, Instant};

use fb_core_budget::Backing;
use fb_core_flow::{Bus, Dag, Executor};
use fb_steps::{CoreHamiltonianStep, GuessStep, MainBus, MetricStep, OverlapStep};

fn threads() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    status
        .lines()
        .find_map(|l| l.strip_prefix("Threads:"))
        .expect("a Threads: line")
        .trim()
        .parse()
        .expect("a thread count")
}

/// The thread count once the kernel has caught up, waiting up to five
/// seconds for it to reach `want`; the last count seen if it never does. A
/// join returns before the exiting thread leaves `Threads:`
/// (`core/thread_pool/tests/thread_pool_threads.rs`'s `settled_threads`).
fn settled_threads(want: usize) -> usize {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut now = threads();
    while now != want && Instant::now() < deadline {
        std::thread::yield_now();
        now = threads();
    }
    now
}

fn main() {
    const THREADS: usize = 2;
    const HOST_THREADS: usize = 3;
    let before = threads();

    let mut config = common::config("h2o.xyz", "cc-pvdz", Some("cc-pvdz-jkfit"));
    config.resources.threads = THREADS;
    config.resources.host_threads = Some(HOST_THREADS);
    let bus = MainBus::setup(config, Backing::Host).unwrap();
    assert_eq!(bus.shared().host_pool().threads(), HOST_THREADS);
    let mut dag = Dag::new();
    dag.add(OverlapStep)
        .add(CoreHamiltonianStep)
        .add(GuessStep)
        .add(MetricStep);
    let plan = dag.build(bus).unwrap();
    let executor = Executor::new(THREADS);
    plan.budget(&executor).unwrap();
    plan.execute(&executor).unwrap();
    let want = 1 + THREADS + HOST_THREADS;
    assert_eq!(
        settled_threads(want),
        want,
        "after the run ({before} before setup): the main thread, the executor's {THREADS} and \
         the host pool's {HOST_THREADS}"
    );

    drop(plan);
    drop(executor);
    assert_eq!(settled_threads(1), 1, "only the main thread after the plan");
    println!(
        "steps_threads: 1 + {THREADS} + {HOST_THREADS} threads in the run, 1 after it, {before} before setup"
    );
}
