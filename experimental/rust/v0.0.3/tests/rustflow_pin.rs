//! What the design leans on from rustflow at its pinned revision, checked
//! rather than read ("The step DAG on rustflow" in `docs/rust-redesign.html`):
//! a condition loop built inside a subflow runs, and joins before the
//! subflow's node finishes; and a multi-condition sends a loop back to a body
//! of several sources at once, so the body needs no empty entry node.
//!
//! Device-free; each case runs on one worker (the serial mode) and on four.

use rustflow::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The trips each loop makes.
const TRIPS: usize = 5;

type Log = Arc<Mutex<Vec<String>>>;

fn push(log: &Log, event: impl Into<String>) {
    log.lock().unwrap().push(event.into());
}

/// An SCF-shaped loop as a step's inner work: `init -> body -> cond`, the
/// condition's index 0 back to `body` and index 1 to `exit`, all built in the
/// subflow of one outer node. The outer node after it sees the loop done.
#[test]
fn a_condition_loop_runs_inside_a_subflow() {
    for workers in [1, 4] {
        let log: Log = Arc::default();
        let mut taskflow = Taskflow::new();
        let before = taskflow.emplace({
            let log = Arc::clone(&log);
            move || push(&log, "before")
        });
        let step = taskflow.emplace_subflow({
            let log = Arc::clone(&log);
            move |subflow: &mut Subflow<'_>| {
                let trips = Arc::new(AtomicUsize::new(0));
                let init = subflow.emplace({
                    let log = Arc::clone(&log);
                    move || push(&log, "init")
                });
                let body = subflow.emplace({
                    let (log, trips) = (Arc::clone(&log), Arc::clone(&trips));
                    move || {
                        let trip = trips.fetch_add(1, Ordering::SeqCst);
                        push(&log, format!("body{trip}"));
                    }
                });
                let cond = subflow.emplace_condition({
                    let trips = Arc::clone(&trips);
                    move || i32::from(trips.load(Ordering::SeqCst) >= TRIPS)
                });
                let exit = subflow.emplace({
                    let log = Arc::clone(&log);
                    move || push(&log, "exit")
                });
                subflow.linearize([init, body, cond]);
                subflow.task(cond).precede([body, exit]);
            }
        });
        let after = taskflow.emplace({
            let log = Arc::clone(&log);
            move || push(&log, "after")
        });
        taskflow.linearize([before, step, after]);

        Executor::new(workers)
            .run(&taskflow)
            .wait()
            .unwrap_or_else(|e| panic!("{workers} worker(s): {e}"));

        let mut want = vec!["before".to_string(), "init".to_string()];
        want.extend((0..TRIPS).map(|t| format!("body{t}")));
        want.extend(["exit".to_string(), "after".to_string()]);
        assert_eq!(*log.lock().unwrap(), want, "{workers} worker(s)");
    }
}

/// A loop body of two sources, `a` and `b`, joined by `j`, with no entry
/// node: `init` precedes both, and a multi-condition after `j` names both
/// again (indices 0 and 1) to loop, or `exit` (index 2) to stop. Each trip
/// runs `a` and `b` once each and then `j`, inside a subflow, many runs over.
#[test]
fn a_multi_condition_loops_a_body_of_two_sources_with_no_entry_node() {
    for workers in [1, 4] {
        for run in 0..50 {
            let log: Log = Arc::default();
            let mut taskflow = Taskflow::new();
            taskflow.emplace_subflow({
                let log = Arc::clone(&log);
                move |subflow: &mut Subflow<'_>| {
                    let trips = Arc::new(AtomicUsize::new(0));
                    let init = subflow.emplace(|| {});
                    let a = subflow.emplace({
                        let log = Arc::clone(&log);
                        move || push(&log, "a")
                    });
                    let b = subflow.emplace({
                        let log = Arc::clone(&log);
                        move || push(&log, "b")
                    });
                    let j = subflow.emplace({
                        let (log, trips) = (Arc::clone(&log), Arc::clone(&trips));
                        move || {
                            trips.fetch_add(1, Ordering::SeqCst);
                            push(&log, "j");
                        }
                    });
                    let cond = subflow.emplace_multi_condition({
                        let trips = Arc::clone(&trips);
                        move || {
                            if trips.load(Ordering::SeqCst) < TRIPS {
                                vec![0, 1]
                            } else {
                                vec![2]
                            }
                        }
                    });
                    let exit = subflow.emplace({
                        let log = Arc::clone(&log);
                        move || push(&log, "exit")
                    });
                    subflow.task(init).precede([a, b]);
                    subflow.task(j).succeed([a, b]).precede([cond]);
                    subflow.task(cond).precede([a, b, exit]);
                }
            });

            Executor::new(workers)
                .run(&taskflow)
                .wait()
                .unwrap_or_else(|e| panic!("{workers} worker(s), run {run}: {e}"));

            let log = log.lock().unwrap();
            let context = format!("{workers} worker(s), run {run}: {log:?}");
            assert_eq!(log.len(), 3 * TRIPS + 1, "{context}");
            for trip in log[..3 * TRIPS].chunks(3) {
                let mut sources = [trip[0].as_str(), trip[1].as_str()];
                sources.sort_unstable();
                assert_eq!(sources, ["a", "b"], "{context}");
                assert_eq!(trip[2], "j", "{context}");
            }
            assert_eq!(log[3 * TRIPS], "exit", "{context}");
        }
    }
}
