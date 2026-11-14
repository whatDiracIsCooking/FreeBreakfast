//! The executor's promises over random DAGs, with [`HostOnly`] workers (tier
//! 3), a loop's replay over a fixed contraction (tier 2), and the device
//! edges and a loop's stream scopes once on device 0 (tier 4).

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use fb_core_error_handling::{Error, ErrorKind, Status};
use fb_core_log as log;

/// A small seeded generator: the case's seed is printed on failure.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }
}

/// A random DAG's shape: each node's placement and predecessors.
struct Shape {
    placements: Vec<Placement>,
    preds: Vec<Vec<usize>>,
    n_streams: usize,
}

fn random_shape(rng: &mut Rng) -> Shape {
    let n = 1 + rng.below(40);
    let n_streams = rng.below(4);
    let mut placements = Vec::new();
    let mut preds = Vec::new();
    for id in 0..n {
        placements.push(if n_streams > 0 && rng.chance(50) {
            Placement::Stream(rng.below(n_streams))
        } else {
            Placement::Cpu
        });
        preds.push((0..id).filter(|_| rng.chance(15)).collect());
    }
    Shape {
        placements,
        preds,
        n_streams,
    }
}

/// What a run of a [`Shape`] observed.
#[derive(Default)]
struct Observed {
    /// Per node, each (start, end) tick it ran at.
    runs: Mutex<Vec<Vec<(usize, usize)>>>,
    clock: AtomicUsize,
    cpu_running: AtomicUsize,
    cpu_peak: AtomicUsize,
    lane_running: [AtomicUsize; 4],
    lane_peak: [AtomicUsize; 4],
}

fn track(running: &AtomicUsize, peak: &AtomicUsize) {
    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
    peak.fetch_max(now, Ordering::SeqCst);
}

/// Run `shape` with `n_cpu` CPU workers, failing the nodes in `fail`.
fn run_shape(shape: &Shape, n_cpu: usize, fail: &[usize]) -> (Observed, Status) {
    let obs = Observed::default();
    *obs.runs.lock().unwrap() = vec![Vec::new(); shape.placements.len()];
    let result = {
        let obs = &obs;
        let mut plan = Plan::new();
        let ids: Vec<NodeId> = Vec::new();
        let ids = Mutex::new(ids);
        for (id, (&placement, preds)) in shape.placements.iter().zip(&shape.preds).enumerate() {
            let deps: Vec<NodeId> = preds.iter().map(|&p| ids.lock().unwrap()[p]).collect();
            let fails = fail.contains(&id);
            let node = plan.add(
                format!("n{id}"),
                placement,
                &deps,
                move |lane: Lane<'_, ()>| {
                    let (running, peak) = match lane {
                        Lane::Cpu => (&obs.cpu_running, &obs.cpu_peak),
                        Lane::Stream { index, .. } => {
                            (&obs.lane_running[index], &obs.lane_peak[index])
                        }
                    };
                    track(running, peak);
                    let start = obs.clock.fetch_add(1, Ordering::SeqCst);
                    std::thread::yield_now();
                    let end = obs.clock.fetch_add(1, Ordering::SeqCst);
                    running.fetch_sub(1, Ordering::SeqCst);
                    obs.runs.lock().unwrap()[id].push((start, end));
                    if fails {
                        Err(Error::numerical(format!("n{id} failed")))
                    } else {
                        Ok(())
                    }
                },
            );
            ids.lock().unwrap().push(node);
        }
        run(&mut plan, &mut HostOnly::new(shape.n_streams), n_cpu).result
    };
    (obs, result)
}

/// The nodes downstream of any node in `from`, not counting `from` itself
/// unless it is downstream of another.
fn descendants(shape: &Shape, from: &[usize]) -> Vec<bool> {
    let n = shape.placements.len();
    let mut below = vec![false; n];
    for id in 0..n {
        below[id] = shape.preds[id]
            .iter()
            .any(|&p| below[p] || from.contains(&p));
    }
    below
}

#[test]
fn every_node_runs_once_after_its_predecessors_within_the_worker_counts() {
    for seed in 1..=300u64 {
        let mut rng = Rng(seed);
        let shape = random_shape(&mut rng);
        let n_cpu = 1 + rng.below(6);
        let (obs, result) = run_shape(&shape, n_cpu, &[]);
        assert!(result.is_ok(), "seed {seed}: {result:?}");
        let runs = obs.runs.lock().unwrap();
        for (id, r) in runs.iter().enumerate() {
            assert_eq!(r.len(), 1, "seed {seed}: node {id} ran {} times", r.len());
            for &p in &shape.preds[id] {
                assert!(
                    runs[p][0].1 < r[0].0,
                    "seed {seed}: node {id} started before its predecessor {p} ended"
                );
            }
        }
        assert!(obs.cpu_peak.load(Ordering::SeqCst) <= n_cpu, "seed {seed}");
        for lane in &obs.lane_peak {
            assert!(lane.load(Ordering::SeqCst) <= 1, "seed {seed}");
        }
    }
}

#[test]
fn a_failure_skips_its_descendants_and_reports_the_lowest_rank() {
    for seed in 1..=300u64 {
        let mut rng = Rng(seed);
        let shape = random_shape(&mut rng);
        let n = shape.placements.len();
        let fail: Vec<usize> = (0..n).filter(|_| rng.chance(10)).collect();
        let below = descendants(&shape, &fail);
        // A failing node below another failure is skipped, so never fails.
        let expected = fail.iter().copied().find(|&f| !below[f]);
        for n_cpu in [1, 4] {
            let (obs, result) = run_shape(&shape, n_cpu, &fail);
            let runs = obs.runs.lock().unwrap();
            for id in 0..n {
                let ran = runs[id].len();
                assert_eq!(
                    ran,
                    usize::from(!below[id]),
                    "seed {seed}, n_cpu {n_cpu}, node {id}"
                );
            }
            match expected {
                None => assert!(result.is_ok(), "seed {seed}"),
                Some(f) => {
                    let error = result.unwrap_err();
                    assert_eq!(error.message, format!("n{f} failed"), "seed {seed}");
                    assert!(
                        error.context[0].starts_with(&format!("n{f}@")),
                        "seed {seed}"
                    );
                }
            }
        }
    }
}

#[test]
fn an_error_names_its_task_and_worker_and_carries_the_others() {
    let mut plan = Plan::new();
    plan.add("first", Placement::Cpu, &[], |_| Err(Error::io("disk")));
    plan.add("second", Placement::Stream(0), &[], |_: Lane<'_, ()>| {
        Err(Error::numerical("nan"))
    });
    let error = run(&mut plan, &mut HostOnly::new(1), 1).result.unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
    assert_eq!(
        error.to_string(),
        "disk [first@cpu0] [also: nan [second@stream0]]"
    );
}

#[test]
fn a_node_runs_inside_its_task_record() {
    let seen = Mutex::new(Vec::new());
    let mut plan = Plan::new();
    let a = plan.add("a", Placement::Cpu, &[], |_| {
        seen.lock().unwrap().push(log::current_task());
        Ok(())
    });
    plan.add("b", Placement::Stream(0), &[a], |_: Lane<'_, ()>| {
        seen.lock().unwrap().push(log::current_task());
        Ok(())
    });
    run(&mut plan, &mut HostOnly::new(1), 1).result.unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        [Some("a@cpu0".to_string()), Some("b@stream0".to_string())]
    );
    assert_eq!(log::current_task(), None);
}

#[test]
#[should_panic(expected = "node panicked")]
fn a_panicking_node_fails_the_run_instead_of_hanging_it() {
    let mut plan = Plan::new();
    let a = plan.add("a", Placement::Cpu, &[], |_: Lane<'_, ()>| {
        panic!("node panicked")
    });
    // Waiting on `a`: without the abort these workers would wait forever.
    plan.add("b", Placement::Cpu, &[a], |_| Ok(()));
    plan.add("c", Placement::Stream(0), &[a], |_| Ok(()));
    let _ = run(&mut plan, &mut HostOnly::new(1), 3);
}

#[test]
fn a_plan_the_workers_cannot_run_is_refused_before_anything_runs() {
    let ran = AtomicUsize::new(0);
    let mut plan = Plan::new();
    plan.add("a", Placement::Stream(1), &[], |_: Lane<'_, ()>| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let error = run(&mut plan, &mut HostOnly::new(1), 1).result.unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);

    let mut plan = Plan::new();
    plan.add("a", Placement::Cpu, &[], |_: Lane<'_, ()>| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let error = run(&mut plan, &mut HostOnly::new(0), 0).result.unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}

#[test]
fn an_empty_plan_runs() {
    let mut plan: Plan<'_, ()> = Plan::new();
    run(&mut plan, &mut HostOnly::new(2), 1).result.unwrap();
}

#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "every predecessor is in the plan")]
fn a_predecessor_from_elsewhere_is_rejected() {
    let mut other: Plan<'_, ()> = Plan::new();
    let foreign = other.add("x", Placement::Cpu, &[], |_| Ok(()));
    let mut plan: Plan<'_, ()> = Plan::new();
    plan.add("a", Placement::Cpu, &[foreign], |_| Ok(()));
}

/// Each node sleeps a known time: one entry per node, named for its node
/// and worker, each host time at least its sleep, the device times empty
/// with no device, and the wall time at least the longest chain.
#[test]
fn every_node_is_timed_by_its_worker() {
    fn ms(n: u64) -> std::time::Duration {
        std::time::Duration::from_millis(n)
    }
    fn sleep(n: u64) -> impl Fn(Lane<'_, ()>) -> Status + Sync {
        move |_| {
            std::thread::sleep(ms(n));
            Ok(())
        }
    }
    let mut plan = Plan::new();
    let a = plan.add("a", Placement::Cpu, &[], sleep(5));
    let b = plan.add("b", Placement::Stream(0), &[a], sleep(10));
    let c = plan.add("c", Placement::Stream(1), &[a], sleep(15));
    plan.add("d", Placement::Cpu, &[b, c], sleep(20));
    let record = run(&mut plan, &mut HostOnly::new(2), 1);
    record.result.unwrap();
    let got: Vec<_> = record
        .tasks
        .iter()
        .map(|t| (t.node.index(), t.name.as_str(), t.worker.as_str(), t.device))
        .collect();
    assert_eq!(
        got,
        [
            (0, "a", "cpu0", None),
            (1, "b", "stream0", None),
            (2, "c", "stream1", None),
            (3, "d", "cpu0", None),
        ]
    );
    for (t, n) in record.tasks.iter().zip([5, 10, 15, 20]) {
        assert!(t.host >= ms(n), "{}: {:?} < {n} ms", t.name, t.host);
    }
    assert!(record.wall >= ms(5 + 15 + 20), "{:?}", record.wall);
}

/// A failed run returns the entries of the nodes that ran, the failed one
/// among them, and none for the skipped.
#[test]
fn a_failed_run_still_returns_its_times() {
    let ms = std::time::Duration::from_millis(5);
    let mut plan = Plan::new();
    let a = plan.add("a", Placement::Cpu, &[], |_: Lane<'_, ()>| {
        std::thread::sleep(ms);
        Ok(())
    });
    let b = plan.add("b", Placement::Stream(0), &[a], |_| {
        std::thread::sleep(ms);
        Err(Error::numerical("nan"))
    });
    plan.add("c", Placement::Cpu, &[b], |_| Ok(()));
    plan.add("d", Placement::Cpu, &[a], |_| {
        std::thread::sleep(ms);
        Ok(())
    });
    let record = run(&mut plan, &mut HostOnly::new(1), 2);
    assert_eq!(record.result.unwrap_err().message, "nan");
    let got: Vec<_> = record.tasks.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(got, ["a", "b", "d"]);
    assert!(record.tasks.iter().all(|t| t.host >= ms));
}

/// A fixed DAG's record as `timing.json`: it parses back to an equal
/// record, with one entry per task and its dimensions, a sync point for
/// each CPU node's wait on each GPU predecessor and for each stream
/// worker's exit, each naming the node waited for, and the profile this
/// test was built in.
#[test]
fn a_record_reads_back_from_its_json() {
    let mut plan = Plan::new();
    let a = plan.add("a", Placement::Cpu, &[], |_: Lane<'_, ()>| Ok(()));
    let b = plan.add("b", Placement::Stream(0), &[a], |_| Ok(()));
    let c = plan.add("c", Placement::Stream(1), &[a], |_| Ok(()));
    let d = plan.add("d", Placement::Cpu, &[b, c], |_| Ok(()));
    plan.set_dims(b, &[("n", 24), ("m", 3)]);
    plan.set_dims(d, &[("n", 24)]);
    let record = run(&mut plan, &mut HostOnly::new(2), 1);
    assert_eq!(record.result, Ok(()));

    let json = record.to_json();
    let back: Record = serde_json::from_str(&json).unwrap();
    assert_eq!(back, record);

    let tasks: Vec<_> = back
        .tasks
        .iter()
        .map(|t| {
            let dims: Vec<_> = t.dims.iter().map(|(k, &v)| (k.as_str(), v)).collect();
            (t.node, t.name.as_str(), dims)
        })
        .collect();
    assert_eq!(
        tasks,
        [
            (a, "a", vec![]),
            (b, "b", vec![("m", 3), ("n", 24)]),
            (c, "c", vec![]),
            (d, "d", vec![("n", 24)]),
        ]
    );
    let syncs: Vec<_> = back
        .syncs
        .iter()
        .map(|s| (s.node, s.worker.as_str(), s.waited_for))
        .collect();
    assert_eq!(
        syncs,
        [
            (Some(d), "cpu0", b),
            (Some(d), "cpu0", c),
            (None, "stream0", b),
            (None, "stream1", c),
        ]
    );
    let expected = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    assert_eq!(back.profile, expected);
    assert!(back.arena.is_empty());

    // The schema the type's doc states: times in integer nanoseconds,
    // nodes as their ranks, no error.
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(value["error"].is_null());
    assert!(value["wall_ns"].is_u64());
    assert_eq!(value["tasks"][1]["node"], 1);
    assert!(value["tasks"][1]["host_ns"].is_u64());
    assert!(value["tasks"][1]["device_ns"].is_null());
    assert_eq!(value["syncs"][0]["node"], 3);
    assert_eq!(value["syncs"][0]["waited_for"], 1);
    assert!(value["syncs"][2]["node"].is_null());
}

/// A failed run's record reads back with its error's kind, message and
/// context, and everything else equal; the origin is not in the file.
#[test]
fn a_failed_record_reads_back_without_its_origin() {
    let mut plan = Plan::new();
    let a = plan.add("a", Placement::Stream(0), &[], |_: Lane<'_, ()>| Ok(()));
    plan.add("b", Placement::Cpu, &[a], |_| Err(Error::numerical("nan")));
    let mut record = run(&mut plan, &mut HostOnly::new(1), 1);
    let mut back: Record = serde_json::from_str(&record.to_json()).unwrap();
    let (got, want) = (back.result.unwrap_err(), record.result.unwrap_err());
    assert_eq!(
        (got.kind, &got.message, &got.context),
        (want.kind, &want.message, &want.context)
    );
    assert_eq!(want.context, ["b@cpu0"]);
    (back.result, record.result) = (Ok(()), Ok(()));
    assert_eq!(back, record);
}

/// The committed contraction of the loop tests: `x <- A x + b` from
/// `x = 0`, with `b` on `A`'s dominant eigenvector, `(2, 1)` for the
/// eigenvalue 0.4 (the other is 0.1), and `|A|_inf = 0.5`.
const A: [[f64; 2]; 2] = [[0.3, 0.2], [0.1, 0.2]];
const B: [f64; 2] = [0.2, 0.1];
/// The decision's stop: `|x_k - x_{k-1}| <= TOL`.
const TOL: f64 = 1e-12;

/// The iterate the loop body moves along.
#[derive(Default)]
struct Iterate {
    x: [f64; 2],
    ax: [f64; 2],
    next: [f64; 2],
    moved: f64,
}

/// A plan of `x0`, a loop whose body is `A x` (stream 0), `+ b` (CPU) and
/// `|dx|` (stream 1) under the decision `converged`, bounded at
/// `max_iterations`, and `report` after it. `A x` fails on its call number
/// `fail_on` (from 1), if given. Returns the plan and the decision.
fn fixed_point_plan<'a>(
    it: &'a Mutex<Iterate>,
    calls: &'a AtomicUsize,
    max_iterations: usize,
    fail_on: Option<usize>,
) -> (Plan<'a, ()>, NodeId) {
    let mut plan = Plan::new();
    let x0 = plan.add("x0", Placement::Cpu, &[], |_: Lane<'_, ()>| {
        *it.lock().unwrap() = Iterate::default();
        Ok(())
    });
    let ax = plan.add("A x", Placement::Stream(0), &[x0], move |_| {
        if Some(calls.fetch_add(1, Ordering::SeqCst) + 1) == fail_on {
            return Err(Error::numerical("injected"));
        }
        let mut it = it.lock().unwrap();
        let x = it.x;
        it.ax = A.map(|row| row[0] * x[0] + row[1] * x[1]);
        Ok(())
    });
    let plus = plan.add("+ b", Placement::Cpu, &[ax], |_| {
        let mut it = it.lock().unwrap();
        it.next = [it.ax[0] + B[0], it.ax[1] + B[1]];
        Ok(())
    });
    plan.add("|dx|", Placement::Stream(1), &[plus, x0], |_| {
        let mut it = it.lock().unwrap();
        let (x, next) = (it.x, it.next);
        it.moved = (next[0] - x[0]).hypot(next[1] - x[1]);
        it.x = next;
        Ok(())
    });
    let decision = plan.add_loop("converged", ax, &[], max_iterations, |_| {
        Ok(if it.lock().unwrap().moved <= TOL {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    });
    plan.add("report", Placement::Cpu, &[decision, x0], |_| Ok(()));
    (plan, decision)
}

/// Each node's iterations in the record, by name.
fn iterations_of(record: &Record, name: &str) -> Vec<Option<usize>> {
    record
        .tasks
        .iter()
        .filter(|t| t.name == name)
        .map(|t| t.iteration)
        .collect()
}

/// Tier 2, one run: the loop replays its body until the decision stops it,
/// after the number of iterations the contraction's spectral radius gives,
/// at the closed-form fixed point.
///
/// With `x_0 = 0` and `b` an eigenvector of `A` for its spectral radius
/// `rho`, `x_k - x_{k-1} = A^{k-1} b = rho^{k-1} b`, so the decision first
/// stops at the smallest `k` with `rho^{k-1} |b| <= TOL`, which the test
/// computes from `rho` (the larger root of `A`'s characteristic
/// polynomial). The exponent falls 0.52 from an integer, far beyond the
/// rounding of a difference near `1e-12` of iterates near 0.3 (a few
/// `1e-4` of it). There `x* - x_k = rho^k x* = rho / (1 - rho) (x_k -
/// x_{k-1})`, so `|x_k - x*| <= 2/3 TOL` with `rho = 0.4`, and `x* = (I -
/// A)^{-1} b` in closed form; the rounding of the iterates is `~1e-17`.
#[test]
fn a_loop_replays_its_body_to_the_fixed_point() {
    let (trace, det) = (A[0][0] + A[1][1], A[0][0] * A[1][1] - A[0][1] * A[1][0]);
    let rho = (trace + (trace * trace - 4.0 * det).sqrt()) / 2.0;
    let exponent = (TOL / B[0].hypot(B[1])).ln() / rho.ln();
    assert!(
        (exponent - exponent.round()).abs() > 0.1,
        "the stop is too close to a boundary for the count to be exact: {exponent}"
    );
    let expected = 1 + exponent.ceil() as usize;

    let it = Mutex::new(Iterate::default());
    let calls = AtomicUsize::new(0);
    let (mut plan, decision) = fixed_point_plan(&it, &calls, 1000, None);
    let record = run(&mut plan, &mut HostOnly::new(2), 2);
    record.result.as_ref().unwrap();

    assert_eq!(record.iterations, BTreeMap::from([(decision, expected)]));
    let each: Vec<_> = (0..expected).map(Some).collect();
    for name in ["A x", "+ b", "|dx|", "converged"] {
        assert_eq!(iterations_of(&record, name), each, "{name}");
    }
    assert_eq!(iterations_of(&record, "x0"), [None]);
    assert_eq!(iterations_of(&record, "report"), [None]);

    let m = [[1.0 - A[0][0], -A[0][1]], [-A[1][0], 1.0 - A[1][1]]];
    let d = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let fixed = [
        (m[1][1] * B[0] - m[0][1] * B[1]) / d,
        (m[0][0] * B[1] - m[1][0] * B[0]) / d,
    ];
    let x = it.lock().unwrap().x;
    assert!(
        (x[0] - fixed[0]).hypot(x[1] - fixed[1]) <= TOL,
        "{x:?} against {fixed:?}"
    );

    // Each iteration's entries read back from `timing.json`.
    let json = record.to_json();
    assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), record);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(value["iterations"][decision.index().to_string()], expected);
    assert!(value["tasks"][0]["iteration"].is_null());
    assert_eq!(value["tasks"][2]["iteration"], 1);
}

/// Tier 2: a decision still saying `Continue` at its bound fails the run
/// with `MaxIterationsExceeded`, after the bound's iterations, and the node
/// after the loop never runs.
#[test]
fn a_loop_that_reaches_its_bound_fails_the_run() {
    let it = Mutex::new(Iterate::default());
    let calls = AtomicUsize::new(0);
    let (mut plan, decision) = fixed_point_plan(&it, &calls, 5, None);
    let record = run(&mut plan, &mut HostOnly::new(2), 2);
    let error = record.result.as_ref().unwrap_err();
    assert_eq!(error.kind, ErrorKind::MaxIterationsExceeded);
    assert_eq!(error.context.len(), 1);
    assert!(error.context[0].starts_with("converged@cpu"), "{error}");
    assert_eq!(record.iterations, BTreeMap::from([(decision, 5)]));
    assert_eq!(iterations_of(&record, "converged").len(), 5);
    assert!(iterations_of(&record, "report").is_empty());
}

/// Tier 2: a body node failing in the second iteration returns its error,
/// ends the loop there, and the decision of that iteration and the node
/// after the loop have no entry.
#[test]
fn a_failed_body_node_ends_the_loop_and_the_run() {
    let it = Mutex::new(Iterate::default());
    let calls = AtomicUsize::new(0);
    let (mut plan, decision) = fixed_point_plan(&it, &calls, 1000, Some(2));
    let record = run(&mut plan, &mut HostOnly::new(2), 2);
    let error = record.result.as_ref().unwrap_err();
    assert_eq!(
        (error.kind, error.message.as_str()),
        (ErrorKind::Numerical, "injected")
    );
    assert_eq!(error.context, ["A x@stream0"]);
    assert_eq!(record.iterations, BTreeMap::from([(decision, 2)]));
    assert_eq!(iterations_of(&record, "A x"), [Some(0), Some(1)]);
    assert_eq!(iterations_of(&record, "converged"), [Some(0)]);
    assert!(iterations_of(&record, "report").is_empty());
}

/// Tier 2: a decision runs after the predecessors it is given as well as
/// its body: a node before the body that no body node follows has finished
/// before the first decision, and is not run again by a replay.
#[test]
fn a_decision_runs_after_its_own_predecessors() {
    let before = AtomicUsize::new(0);
    let mut plan: Plan<'_, ()> = Plan::new();
    let slow = plan.add("slow", Placement::Cpu, &[], |_| {
        std::thread::sleep(std::time::Duration::from_millis(20));
        before.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let body = plan.add("body", Placement::Cpu, &[], |_| Ok(()));
    let decision = plan.add_loop("decide", body, &[slow], 3, |i| {
        assert_eq!(before.load(Ordering::SeqCst), 1, "iteration {i}");
        Ok(if i + 1 < 3 {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    });
    let record = run(&mut plan, &mut HostOnly::new(0), 2);
    record.result.unwrap();
    assert_eq!(record.iterations, BTreeMap::from([(decision, 3)]));
    assert_eq!(before.load(Ordering::SeqCst), 1);
}

#[test]
#[cfg(debug_assertions)] // the contract is compiled out of release
#[should_panic(expected = "no predecessor is in a closed loop's body")]
fn a_node_after_a_loop_cannot_read_its_body() {
    let mut plan: Plan<'_, ()> = Plan::new();
    let body = plan.add("body", Placement::Cpu, &[], |_| Ok(()));
    plan.add_loop("decide", body, &[], 2, |_| Ok(ControlFlow::Break(())));
    plan.add("after", Placement::Cpu, &[body], |_| Ok(()));
}

#[test]
fn default_n_cpu_leaves_the_stream_threads_and_keeps_one() {
    assert!(default_n_cpu(0) >= 1);
    assert_eq!(default_n_cpu(usize::MAX), 1);
}

/// Device 0: each stream worker's device is current, work still running on
/// one stream is waited for by a node on another stream and by a CPU node,
/// and the run returns with nothing outstanding.
///
/// Stream 0's node enqueues a chain of device-to-device copies that ends by
/// writing `src`, so it is still running when the node on stream 1 copies
/// `src` on. Without the wait the copy reads the zeros `src` started as;
/// without the CPU node's event sync the read sees `dst` before the copy
/// lands. Either fault fails the comparison.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_edges_wait_for_the_work_they_consume() {
    use fb_core_wwr::handle::DeviceHandleStream;
    use fb_core_wwr::runtime::Stream;
    use fb_core_wwr::runtime::tests::{DeviceBytes, ok};
    use wwr_sys::wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE;

    const N: usize = 64 << 20;
    const HOPS: usize = 32;
    let pattern: Vec<u8> = (0..N).map(|i| (i % 251) as u8).collect();
    let zeros = vec![0u8; N];
    // Each upload is drained before it returns, so all of it is before the run.
    let setup = Stream::new(0);
    let a = DeviceBytes::upload(&pattern, &setup);
    let b = DeviceBytes::new(N);
    let [src, dst] = [(); 2].map(|_| DeviceBytes::upload(&zeros, &setup));
    // The copies under test: enqueued on a worker's stream and left in
    // flight, which is what the edges must wait for. A helper that synced
    // would hide a missing wait, so they are written out.
    let d2d = |to: &DeviceBytes, from: &DeviceBytes, stream: &DeviceStream| {
        // SAFETY: both device ranges are N bytes and outlive the run.
        ok(unsafe {
            wwr_sys::wwr_memcpy_async(
                to.0,
                from.0,
                N,
                WWR_MEMCPY_DEVICE_TO_DEVICE,
                stream.stream(),
            )
        });
    };
    let seen_by_cpu = Mutex::new(vec![0u8; N]);
    let devices_seen = Mutex::new(Vec::new());

    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    let produce = plan.add("produce", Placement::Stream(0), &[], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        let mut dev = -1;
        // SAFETY: writes one i32 through a valid local.
        ok(unsafe { wwr_sys::wwr_get_device(&mut dev) });
        devices_seen.lock().unwrap().push(dev);
        for _ in 0..HOPS / 2 {
            d2d(&b, &a, stream);
            d2d(&a, &b, stream);
        }
        d2d(&src, &a, stream);
        Ok(())
    });
    let copy = plan.add("copy", Placement::Stream(1), &[produce], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        d2d(&dst, &src, stream);
        Ok(())
    });
    plan.add("read", Placement::Cpu, &[copy], |_| {
        // A blocking copy on the legacy stream would hide a missing wait, so
        // read on a fresh non-blocking stream, synced alone; the edge's
        // event sync is all that orders the copy before this read.
        *seen_by_cpu.lock().unwrap() = dst.read(&Stream::new(0));
        Ok(())
    });
    let mut device = Device::new(&[0, 0], &plan);
    run(&mut plan, &mut device, 1).result.unwrap();
    assert_eq!(*devices_seen.lock().unwrap(), [0]);
    assert!(
        *seen_by_cpu.lock().unwrap() == pattern,
        "a device edge did not wait"
    );
}

/// Device 0: each stream worker's BLAS and dense-solver handles are made
/// with the backend, on the worker's device and bound to its stream, and
/// are the same handles at every run, so no task makes one.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_library_handles_are_the_workers_made_before_the_run() {
    use fb_core_wwr::handle::{DeviceHandle, DeviceHandleStream};
    use wwr_sys::{wwr_blas_status, wwr_solver_status, wwr_stream};

    let seen = Mutex::new(Vec::new());
    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    for s in 0..2 {
        plan.add(format!("blas{s}"), Placement::Stream(s), &[], |lane| {
            let Lane::Stream { index, stream } = lane else {
                unreachable!()
            };
            let blas = stream.blas();
            let mut bound: wwr_stream = std::ptr::null_mut();
            // SAFETY: the handle is live for the backend's life; writes one
            // stream through a valid local.
            let status = unsafe { wwr_sys::wwr_blas_get_stream(blas.get(), &mut bound) };
            assert_eq!(status, wwr_blas_status::WWR_BLAS_STATUS_SUCCESS);
            assert_eq!(bound, stream.stream());
            assert_eq!(blas.dev_idx(), stream.dev_idx());
            let solver = stream.solver();
            let mut bound: wwr_stream = std::ptr::null_mut();
            // SAFETY: as above, for the solver handle.
            let status = unsafe { wwr_sys::wwr_solver_dn_get_stream(solver.get(), &mut bound) };
            assert_eq!(status, wwr_solver_status::WWR_SOLVER_STATUS_SUCCESS);
            assert_eq!(bound, stream.stream());
            assert_eq!(solver.dev_idx(), stream.dev_idx());
            seen.lock()
                .unwrap()
                .push((index, blas.get() as usize, solver.get() as usize));
            Ok(())
        });
    }
    let mut device = Device::new(&[0, 0], &plan);
    let mut runs = Vec::new();
    for _ in 0..2 {
        run(&mut plan, &mut device, 1).result.unwrap();
        let mut handles = std::mem::take(&mut *seen.lock().unwrap());
        handles.sort_unstable();
        runs.push(handles);
    }
    assert_eq!(runs[0].len(), 2);
    assert_ne!(
        runs[0][0].1, runs[0][1].1,
        "one BLAS handle per stream worker"
    );
    assert_ne!(
        runs[0][0].2, runs[0][1].2,
        "one solver handle per stream worker"
    );
    assert_eq!(runs[0], runs[1], "a run made a handle of its own");
}

/// Device 0, one run: a loop whose body is a stream node and a CPU node
/// after it reads each iteration's stream scope back, one per iteration.
///
/// Iteration `k` enqueues `k + 1` times the same chain of device-to-device
/// copies, so its device time is `k + 1` times the first's: a scope read
/// after a later iteration re-recorded the events, or one iteration's time
/// copied to another, breaks the strict increase. The increase per
/// iteration is a whole chain, 16 copies of 32 MiB, far above an event
/// pair's resolution.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_loop_reads_each_iterations_stream_scope() {
    use fb_core_wwr::handle::DeviceHandleStream;
    use fb_core_wwr::runtime::Stream;
    use fb_core_wwr::runtime::tests::{DeviceBytes, ok};
    use wwr_sys::wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE;

    const N: usize = 32 << 20;
    const HOPS: usize = 16;
    const ITERATIONS: usize = 3;
    let setup = Stream::new(0);
    let a = DeviceBytes::upload(&vec![7u8; N], &setup);
    let b = DeviceBytes::new(N);
    let calls = AtomicUsize::new(0);
    let host_ran = AtomicUsize::new(0);

    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    let copies = plan.add("copies", Placement::Stream(0), &[], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        let k = calls.fetch_add(1, Ordering::SeqCst);
        for hop in 0..(k + 1) * HOPS {
            let (to, from) = if hop % 2 == 0 { (&b, &a) } else { (&a, &b) };
            // SAFETY: both device ranges are N bytes and outlive the run.
            ok(unsafe {
                wwr_sys::wwr_memcpy_async(
                    to.0,
                    from.0,
                    N,
                    WWR_MEMCPY_DEVICE_TO_DEVICE,
                    stream.stream(),
                )
            });
        }
        Ok(())
    });
    plan.add("host", Placement::Cpu, &[copies], |_| {
        host_ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let decision = plan.add_loop("decide", copies, &[], ITERATIONS, |i| {
        Ok(if i + 1 < ITERATIONS {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    });
    let mut device = Device::new(&[0], &plan);
    let record = run(&mut plan, &mut device, 1);
    record.result.as_ref().unwrap();

    assert_eq!(record.iterations, BTreeMap::from([(decision, ITERATIONS)]));
    assert_eq!(host_ran.load(Ordering::SeqCst), ITERATIONS);
    let scopes: Vec<_> = record
        .tasks
        .iter()
        .filter(|t| t.name == "copies")
        .map(|t| (t.iteration, t.device))
        .collect();
    assert_eq!(scopes.len(), ITERATIONS, "{scopes:?}");
    for (k, &(iteration, device)) in scopes.iter().enumerate() {
        assert_eq!(iteration, Some(k));
        assert!(device.is_some(), "iteration {k}'s scope was not read");
    }
    assert!(
        scopes.windows(2).all(|w| w[0].1 < w[1].1),
        "each iteration's scope is its own: {scopes:?}"
    );
    // The CPU node waited on the stream node once per iteration.
    let waits = record
        .syncs
        .iter()
        .filter(|s| s.node.is_some() && s.waited_for == copies)
        .count();
    assert_eq!(waits, ITERATIONS, "{:?}", record.syncs);
}
