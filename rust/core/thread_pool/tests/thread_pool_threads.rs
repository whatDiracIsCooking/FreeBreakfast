//! A run uses exactly the threads its plan says: `n_cpu` plus one per stream
//! worker, beside the main thread, and none survive it
//! (`core::thread_pool`'s fourth promise). The count is the `Threads:` line
//! of `/proc/self/status`, which also sees a thread a linked library opened
//! on its own, which no lint can. A loop's replay is on the same threads:
//! the count before its first iteration is the count after its last.
//!
//! `harness = false`: libtest runs cases on threads of its own, which would
//! be counted. So this is a plain `main`, and the process holds only what it
//! starts.

use std::ops::ControlFlow;
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
    replayed_on_the_runs_threads();
    println!("thread_pool_threads: ok");
}

/// Wait until the run's workers all exist, up to five seconds, and return
/// the count seen: workers are spawned one after another, so an early node
/// can run before the last exists.
fn all_workers(expected: usize) -> usize {
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut now = threads();
    while now < expected && Instant::now() < deadline {
        std::thread::yield_now();
        now = threads();
    }
    now
}

/// A loop of four iterations, its body a stream node and a CPU node: the
/// count read by the body's first node in the first iteration (once every
/// worker exists) equals the count its decision reads in the last, and
/// every body node of every iteration sees the same.
fn replayed_on_the_runs_threads() {
    const ITERATIONS: usize = 4;
    let (n_cpu, n_streams) = (2, 1);
    let expected = 1 + n_cpu + n_streams;
    let first = Mutex::new(0);
    let body_seen = Mutex::new(Vec::new());
    let last = Mutex::new(0);
    let mut plan = Plan::new();
    let gpu = plan.add("gpu", Placement::Stream(0), &[], |_| {
        let mut first = first.lock().unwrap();
        if *first == 0 {
            *first = all_workers(expected);
        }
        body_seen.lock().unwrap().push(threads());
        Ok(())
    });
    plan.add("cpu", Placement::Cpu, &[gpu], |_| {
        body_seen.lock().unwrap().push(threads());
        Ok(())
    });
    plan.add_loop("decide", gpu, ITERATIONS, |i| {
        if i + 1 < ITERATIONS {
            return Ok(ControlFlow::Continue(()));
        }
        *last.lock().unwrap() = threads();
        Ok(ControlFlow::Break(()))
    });
    let record = run(&mut plan, &mut HostOnly::new(n_streams), n_cpu);
    record.result.unwrap();
    drop(plan);
    let (first, last) = (first.into_inner().unwrap(), last.into_inner().unwrap());
    let body_seen = body_seen.into_inner().unwrap();
    assert_eq!(
        record.iterations.values().copied().collect::<Vec<_>>(),
        [ITERATIONS]
    );
    assert_eq!(
        (first, last),
        (expected, expected),
        "a loop: before its first iteration and after its last"
    );
    assert!(
        body_seen.len() == 2 * ITERATIONS && body_seen.iter().all(|&t| t == expected),
        "a loop's body saw {body_seen:?}, expected {expected} each time"
    );
    assert_eq!(settled_threads(1), 1, "a loop's worker outlived its run");
}
