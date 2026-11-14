//! A run uses exactly the threads its plan says: `n_cpu` plus one per stream
//! worker, beside the main thread, and none survive it
//! (`core::thread_pool`'s fourth promise). The count is the `Threads:` line
//! of `/proc/self/status`, which also sees a thread a linked library opened
//! on its own, which no lint can.
//!
//! `harness = false`: libtest runs cases on threads of its own, which would
//! be counted. So this is a plain `main`, and the process holds only what it
//! starts.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use fb_core_thread_pool::{HostOnly, Placement, Plan, run};

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

/// The thread count once the kernel has caught up with the joins, waiting
/// up to [`SETTLE`] for it to fall to `want`; the last count seen if it
/// never does.
///
/// `run` has joined every worker before it returns, but a join returns at
/// the exiting thread's `CLONE_CHILD_CLEARTID` futex wake, which the kernel
/// does as it releases the thread's memory, before `release_task` takes it
/// out of the thread group. Until then it is still in `Threads:`, and in
/// `/proc/self/task` too, so neither is synchronous with the join. A busy
/// core can preempt the exiting thread in between: with three CPU hogs on
/// the one core, a bare `spawn` and `join` read one thread too many in 135
/// of 200 tries, for up to 10 ms (#1491). A worker that was never joined
/// keeps the count up for the whole wait and still fails.
fn settled_threads(want: usize) -> usize {
    let deadline = Instant::now() + SETTLE;
    let mut now = threads();
    while now != want && Instant::now() < deadline {
        std::thread::yield_now();
        now = threads();
    }
    now
}

/// How long [`settled_threads`] waits: the lag is the rest of one thread's
/// exit, a scheduler slice or a few on a loaded core (10 ms measured), so
/// five seconds is hundreds of times that, and short enough that a worker
/// that really survives fails the test promptly.
const SETTLE: Duration = Duration::from_secs(5);

fn main() {
    assert_eq!(threads(), 1, "the process starts with one thread");
    for (n_cpu, n_streams) in [(1, 0), (3, 2), (8, 4)] {
        let seen = Mutex::new(Vec::new());
        let mut plan = Plan::new();
        let mut last = Vec::new();
        for lane in 0..n_streams {
            last.push(
                plan.add(format!("gpu{lane}"), Placement::Stream(lane), &[], |_| {
                    seen.lock().unwrap().push(threads());
                    Ok(())
                }),
            );
        }
        let expected = 1 + n_cpu + n_streams;
        // Workers are spawned one after another, so an early node can run
        // before the last exists: wait for the full count, never past it.
        plan.add("cpu", Placement::Cpu, &last, |_| {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut now = threads();
            while now < expected && Instant::now() < deadline {
                std::thread::yield_now();
                now = threads();
            }
            seen.lock().unwrap().push(now);
            Ok(())
        });
        run(&mut plan, &mut HostOnly::new(n_streams), n_cpu)
            .result
            .unwrap();
        drop(plan);
        let seen = seen.into_inner().unwrap();
        assert!(
            seen.iter().all(|&t| t <= expected) && seen.last() == Some(&expected),
            "n_cpu {n_cpu}, n_streams {n_streams}: saw {seen:?}, expected {expected}"
        );
        assert_eq!(
            settled_threads(1),
            1,
            "n_cpu {n_cpu}, n_streams {n_streams}: a worker outlived its run \
             (still counted {SETTLE:?} after the join)"
        );
    }
    println!("thread_pool_threads: ok");
}
