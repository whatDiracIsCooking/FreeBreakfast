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
    n_gpu: usize,
}

/// A lane: `lane` of GPU worker `worker`.
fn on(worker: usize, lane: usize) -> Placement {
    Placement::Stream(LaneId { worker, lane })
}

fn random_shape(rng: &mut Rng) -> Shape {
    let n = 1 + rng.below(40);
    let n_gpu = rng.below(4);
    let mut placements = Vec::new();
    let mut preds = Vec::new();
    for id in 0..n {
        placements.push(if n_gpu > 0 && rng.chance(50) {
            on(rng.below(n_gpu), rng.below(3))
        } else {
            Placement::Cpu
        });
        preds.push((0..id).filter(|_| rng.chance(15)).collect());
    }
    Shape {
        placements,
        preds,
        n_gpu,
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
    /// Per GPU worker, the nodes running on its thread now and at most.
    gpu_running: [AtomicUsize; 4],
    gpu_peak: [AtomicUsize; 4],
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
                        Lane::Cpu { .. } => (&obs.cpu_running, &obs.cpu_peak),
                        Lane::Stream { lane, .. } => {
                            (&obs.gpu_running[lane.worker], &obs.gpu_peak[lane.worker])
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
        run(&mut plan, &mut HostOnly::new(shape.n_gpu), n_cpu).result
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
        // One thread per GPU worker, whatever its lanes.
        for worker in &obs.gpu_peak {
            assert!(worker.load(Ordering::SeqCst) <= 1, "seed {seed}");
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
    plan.add("second", on(0, 0), &[], |_: Lane<'_, ()>| {
        Err(Error::numerical("nan"))
    });
    let error = run(&mut plan, &mut HostOnly::new(1), 1).result.unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
    assert_eq!(
        error.to_string(),
        "disk [first@cpu0] [also: nan [second@gpu0.0]]"
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
    plan.add("b", on(0, 0), &[a], |_: Lane<'_, ()>| {
        seen.lock().unwrap().push(log::current_task());
        Ok(())
    });
    run(&mut plan, &mut HostOnly::new(1), 1).result.unwrap();
    assert_eq!(
        *seen.lock().unwrap(),
        [Some("a@cpu0".to_string()), Some("b@gpu0.0".to_string())]
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
    plan.add("c", on(0, 0), &[a], |_| Ok(()));
    let _ = run(&mut plan, &mut HostOnly::new(1), 3);
}

#[test]
fn a_plan_the_workers_cannot_run_is_refused_before_anything_runs() {
    let ran = AtomicUsize::new(0);
    let mut plan = Plan::new();
    plan.add("a", on(1, 0), &[], |_: Lane<'_, ()>| {
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

/// Lanes of one worker: an edge between two of them is a wait on the
/// device, not a sync point, so the record holds only the CPU node's wait
/// on the lane it reads and each lane's exit sync; and every node is timed
/// under its lane, `gpu<w>.<l>`, though one thread ran them all.
#[test]
fn an_edge_between_lanes_is_no_sync_point() {
    let threads = Mutex::new(Vec::new());
    let thread = |_: Lane<'_, ()>| {
        let name = std::thread::current().name().map(str::to_owned);
        threads.lock().unwrap().push(name);
        Ok(())
    };
    let mut plan = Plan::new();
    let a = plan.add("a", on(0, 0), &[], thread);
    let b = plan.add("b", on(0, 1), &[a], thread);
    let c = plan.add("c", on(0, 0), &[a, b], thread);
    let d = plan.add("d", Placement::Cpu, &[b], |_| Ok(()));
    assert_eq!(plan.lanes(), [2]);
    let record = run(&mut plan, &mut HostOnly::new(1), 1);
    record.result.unwrap();
    let workers: Vec<_> = record.tasks.iter().map(|t| t.worker.as_str()).collect();
    assert_eq!(workers, ["gpu0.0", "gpu0.1", "gpu0.0", "cpu0"]);
    assert_eq!(
        *threads.lock().unwrap(),
        vec![Some("fb-gpu0".to_string()); 3],
        "one thread per GPU worker"
    );
    let syncs: Vec<_> = record
        .syncs
        .iter()
        .map(|s| (s.node, s.worker.as_str(), s.waited_for))
        .collect();
    assert_eq!(
        syncs,
        [
            (Some(d), "cpu0", b),
            (None, "gpu0.1", b),
            (None, "gpu0.0", c)
        ]
    );
}

/// A lane the backend lacks is refused before anything runs, as a worker
/// it lacks is: a `HostOnly` worker has every lane, so the case is a
/// worker past its count.
#[test]
fn a_lane_the_backend_lacks_is_refused() {
    let mut plan = Plan::new();
    plan.add("a", on(0, 3), &[], |_: Lane<'_, ()>| Ok(()));
    plan.add("b", on(2, 1), &[], |_| Ok(()));
    let error = run(&mut plan, &mut HostOnly::new(1), 1).result.unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(error.message.contains("lane gpu2.1"), "{error}");
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
    let b = plan.add("b", on(0, 0), &[a], sleep(10));
    let c = plan.add("c", on(1, 0), &[a], sleep(15));
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
            (1, "b", "gpu0.0", None),
            (2, "c", "gpu1.0", None),
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
    let b = plan.add("b", on(0, 0), &[a], |_| {
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
    let b = plan.add("b", on(0, 0), &[a], |_| Ok(()));
    let c = plan.add("c", on(1, 0), &[a], |_| Ok(()));
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
            (None, "gpu0.0", b),
            (None, "gpu1.0", c),
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
    let a = plan.add("a", on(0, 0), &[], |_: Lane<'_, ()>| Ok(()));
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
    let ax = plan.add("A x", on(0, 0), &[x0], move |_| {
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
    plan.add("|dx|", on(1, 0), &[plus, x0], |_| {
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
    assert_eq!(error.context, ["A x@gpu0.0"]);
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

/// A counted loop's run, as one random case observed it: each node's
/// (trip, start, end) ticks.
type Ticks = Vec<Vec<(usize, usize, usize)>>;

/// A random counted loop: a node before it, a body of random shape over
/// the CPU pool and up to three lanes of one GPU worker, and a node after
/// it following a random body node, `tail`, with a random trip count and
/// depth. Returns the shape, the trip count, the depth, `tail`, what ran by
/// node (the node before first and the node after last), and the record.
#[allow(clippy::type_complexity)]
fn run_counted(rng: &mut Rng, n_cpu: usize) -> (Shape, usize, usize, usize, Ticks, Record) {
    let shape = Shape {
        n_gpu: 1,
        ..random_shape(rng)
    };
    let shape = Shape {
        placements: shape
            .placements
            .iter()
            .map(|&p| match p {
                Placement::Stream(at) => on(0, at.lane),
                Placement::Cpu => Placement::Cpu,
            })
            .collect(),
        ..shape
    };
    let trips = 1 + rng.below(6);
    let depth = 1 + rng.below(trips);
    let tail = rng.below(shape.placements.len());
    let clock = AtomicUsize::new(0);
    let ticks: Mutex<Ticks> = Mutex::new(vec![Vec::new(); shape.placements.len() + 2]);
    let record = {
        let (clock, ticks) = (&clock, &ticks);
        let tick = move |node: usize, trip: usize| {
            let start = clock.fetch_add(1, Ordering::SeqCst);
            std::thread::yield_now();
            let end = clock.fetch_add(1, Ordering::SeqCst);
            ticks.lock().unwrap()[node].push((trip, start, end));
        };
        let mut plan = Plan::new();
        let before = plan.add("before", Placement::Cpu, &[], move |_: Lane<'_, ()>| {
            tick(0, 0);
            Ok(())
        });
        let mut ids = Vec::new();
        for (j, (&placement, preds)) in shape.placements.iter().zip(&shape.preds).enumerate() {
            let mut deps: Vec<NodeId> = preds.iter().map(|&p| ids[p]).collect();
            if deps.is_empty() {
                deps.push(before);
            }
            ids.push(plan.add(format!("b{j}"), placement, &deps, move |lane| {
                let trip = match lane {
                    Lane::Cpu { iteration } | Lane::Stream { iteration, .. } => iteration,
                };
                tick(j + 1, trip);
                Ok(())
            }));
        }
        plan.add_counted(ids[0], trips, depth);
        let after = shape.placements.len() + 1;
        plan.add("after", Placement::Cpu, &[ids[tail]], move |_| {
            tick(after, 0);
            Ok(())
        });
        run(&mut plan, &mut HostOnly::new(1), n_cpu)
    };
    (
        shape,
        trips,
        depth,
        tail,
        ticks.into_inner().unwrap(),
        record,
    )
}

/// Tier 3, random counted loops: every body node runs once per trip, handed
/// its trip; within a trip after its predecessors, and after itself in the
/// trip before; every node of trip `k` after every node of trip
/// `k - depth`, the ring; the node before the loop once, before it, and
/// the node after it once, after its predecessor's last trip, and so after
/// that node's every trip. Each body node has an entry per trip, with its
/// trip as its iteration, and the record counts no decision loop.
#[test]
fn a_counted_loop_runs_each_trip_after_the_trip_depth_before() {
    for seed in 1..=200u64 {
        let mut rng = Rng(seed);
        let n_cpu = 1 + rng.below(3);
        let (shape, trips, depth, tail, ticks, record) = run_counted(&mut rng, n_cpu);
        let ctx = format!("seed {seed}, {trips} trips, depth {depth}");
        record.result.as_ref().unwrap();
        assert!(record.iterations.is_empty(), "{ctx}");
        let n = shape.placements.len();
        // Per body node, per trip, its (start, end).
        let mut at = vec![vec![(0, 0); trips]; n];
        for j in 0..n {
            let mut seen: Vec<usize> = ticks[j + 1].iter().map(|&(k, _, _)| k).collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..trips).collect::<Vec<_>>(), "{ctx}: b{j}");
            for &(k, s, e) in &ticks[j + 1] {
                at[j][k] = (s, e);
            }
            let entries: Vec<_> = record
                .tasks
                .iter()
                .filter(|t| t.name == format!("b{j}"))
                .map(|t| t.iteration)
                .collect();
            assert_eq!(entries, (0..trips).map(Some).collect::<Vec<_>>(), "{ctx}");
        }
        let (before, after) = (ticks[0][0], ticks[n + 1][0]);
        assert_eq!((ticks[0].len(), ticks[n + 1].len()), (1, 1), "{ctx}");
        for j in 0..n {
            for k in 0..trips {
                let start = at[j][k].0;
                assert!(before.2 < start, "{ctx}: b{j}@{k} before the node before");
                for &p in &shape.preds[j] {
                    assert!(at[p][k].1 < start, "{ctx}: b{j}@{k} before b{p}@{k}");
                }
                if k >= 1 {
                    assert!(at[j][k - 1].1 < start, "{ctx}: b{j}@{k} before itself");
                }
                if k >= depth {
                    for (i, runs) in at.iter().enumerate() {
                        assert!(
                            runs[k - depth].1 < start,
                            "{ctx}: b{j}@{k} before b{i}@{}",
                            k - depth
                        );
                    }
                }
            }
        }
        for (k, &(_, end)) in at[tail].iter().enumerate() {
            assert!(end < after.1, "{ctx}: the node after before b{tail}@{k}");
        }
    }
}
/// Tier 3: a counted loop whose body is two lanes of one worker, a stage
/// on lane 1 and an add on lane 0 reading it, then a node after the loop on
/// lane 0: the record's sync points are the lanes' exits alone, so the loop
/// added no host wait for its GPU nodes, and each body node has an entry
/// per trip.
#[test]
fn a_counted_loop_adds_no_host_wait_for_its_gpu_nodes() {
    let mut plan: Plan<'_, ()> = Plan::new();
    let stage = plan.add("stage", on(0, 1), &[], |_| Ok(()));
    let add = plan.add("add", on(0, 0), &[stage], |_| Ok(()));
    plan.add_counted(stage, 6, 2);
    plan.add("after", on(0, 0), &[add], |_| Ok(()));
    let record = run(&mut plan, &mut HostOnly::new(1), 1);
    record.result.as_ref().unwrap();
    assert!(
        record.syncs.iter().all(|s| s.node.is_none()),
        "{:?}",
        record.syncs
    );
    assert_eq!(record.syncs.len(), 2, "one exit per lane");
    for name in ["stage", "add"] {
        assert_eq!(
            iterations_of(&record, name),
            (0..6).map(Some).collect::<Vec<_>>()
        );
    }
    assert_eq!(iterations_of(&record, "after"), [None]);
}

/// Tier 3: a counted body node failing in trip 2 fails the run with its
/// name, and skips its descendants: its own later trips, the other body
/// node's trips that follow it, and the node after the loop.
#[test]
fn a_failed_trip_skips_the_later_trips_and_what_follows() {
    let mut plan: Plan<'_, ()> = Plan::new();
    let stage = plan.add("stage", on(0, 1), &[], |lane| {
        let Lane::Stream { iteration, .. } = lane else {
            unreachable!()
        };
        if iteration == 2 {
            return Err(Error::numerical("injected"));
        }
        Ok(())
    });
    let add = plan.add("add", Placement::Cpu, &[stage], |_| Ok(()));
    plan.add_counted(stage, 6, 2);
    plan.add("after", Placement::Cpu, &[add], |_| Ok(()));
    let record = run(&mut plan, &mut HostOnly::new(1), 2);
    let error = record.result.as_ref().unwrap_err();
    assert_eq!(error.context, ["stage@gpu0.1"]);
    assert_eq!(iterations_of(&record, "stage"), [Some(0), Some(1), Some(2)]);
    assert_eq!(iterations_of(&record, "add"), [Some(0), Some(1)]);
    assert!(iterations_of(&record, "after").is_empty());
}

/// A nested case's run, as observed: per node, each (outer iteration,
/// trip, start, end) tick it ran at.
type NestedTicks = Vec<Vec<(usize, usize, usize, usize)>>;

/// Tier 3, random counted loops inside a decision loop: a node before the
/// decision loop; its body a node `pre`, a counted loop of random shape
/// over the CPU pool and three lanes of one worker, and a node `post`
/// after a random counted body node; and the decision, stopping after a
/// random number of iterations. In each outer iteration every counted
/// body node runs once per trip, handed its trip, after `pre` of that
/// iteration and the decision of the one before, within the trip after
/// its predecessors and after itself in the trip before, after every node
/// of the trip `depth` back, and before `post` (for the node it follows)
/// and the decision of its own iteration: the counted loop replayed whole.
/// Each counted body node has one entry per (iteration, trip), the
/// iteration as `outer`, and the record counts the decision loop's
/// iterations alone. The outer iteration a counted body node ran in is
/// read from a counter the decision moves, so a trip run in the wrong
/// iteration shows as a missing (iteration, trip).
#[test]
fn a_counted_loop_in_a_decision_loop_replays_whole_each_iteration() {
    for seed in 1..=200u64 {
        let mut rng = Rng(seed);
        let n_cpu = 1 + rng.below(3);
        let shape = random_shape(&mut rng);
        let placements: Vec<Placement> = shape
            .placements
            .iter()
            .map(|&p| match p {
                Placement::Stream(at) => on(0, at.lane),
                Placement::Cpu => Placement::Cpu,
            })
            .collect();
        let n = placements.len();
        let trips = 1 + rng.below(4);
        let depth = 1 + rng.below(trips);
        let outer = 1 + rng.below(4);
        let tail = rng.below(n);
        let pre_at = if rng.chance(50) {
            on(0, rng.below(3))
        } else {
            Placement::Cpu
        };
        let post_at = if rng.chance(50) {
            on(0, rng.below(3))
        } else {
            Placement::Cpu
        };
        let ctx = format!("seed {seed}, {trips} trips, depth {depth}, {outer} iterations");
        // Nodes: 0 before, 1 pre, 2..2+n the counted body, n+2 post, n+3
        // the decision.
        let (pre, post, decision) = (1, n + 2, n + 3);
        let clock = AtomicUsize::new(0);
        let round = AtomicUsize::new(0);
        let ticks: Mutex<NestedTicks> = Mutex::new(vec![Vec::new(); n + 4]);
        let record = {
            let (clock, round, ticks) = (&clock, &round, &ticks);
            let tick = move |node: usize, trip: usize| {
                let k = round.load(Ordering::SeqCst);
                let start = clock.fetch_add(1, Ordering::SeqCst);
                std::thread::yield_now();
                let end = clock.fetch_add(1, Ordering::SeqCst);
                ticks.lock().unwrap()[node].push((k, trip, start, end));
            };
            let trip_of = |lane: Lane<'_, ()>| match lane {
                Lane::Cpu { iteration } | Lane::Stream { iteration, .. } => iteration,
            };
            let mut plan = Plan::new();
            let before = plan.add("before", Placement::Cpu, &[], move |_: Lane<'_, ()>| {
                tick(0, 0);
                Ok(())
            });
            let first = plan.add("pre", pre_at, &[before], move |_| {
                tick(pre, 0);
                Ok(())
            });
            let mut ids = Vec::new();
            for (j, (&placement, preds)) in placements.iter().zip(&shape.preds).enumerate() {
                let mut deps: Vec<NodeId> = preds.iter().map(|&p| ids[p]).collect();
                deps.push(first);
                ids.push(plan.add(format!("b{j}"), placement, &deps, move |lane| {
                    tick(j + 2, trip_of(lane));
                    Ok(())
                }));
            }
            plan.add_counted(ids[0], trips, depth);
            plan.add("post", post_at, &[ids[tail]], move |_| {
                tick(post, 0);
                Ok(())
            });
            plan.add_loop("decide", first, &[], outer, move |i| {
                tick(decision, 0);
                round.fetch_add(1, Ordering::SeqCst);
                Ok(if i + 1 < outer {
                    ControlFlow::Continue(())
                } else {
                    ControlFlow::Break(())
                })
            });
            run(&mut plan, &mut HostOnly::new(1), n_cpu)
        };
        record.result.as_ref().unwrap();
        assert_eq!(
            record.iterations,
            BTreeMap::from([(NodeId(decision), outer)]),
            "{ctx}"
        );
        let ticks = ticks.into_inner().unwrap();
        // Per outer node, per iteration, its (start, end); per counted
        // body node, per (iteration, trip).
        let once = |node: usize| -> Vec<(usize, usize)> {
            let mut runs = ticks[node].clone();
            runs.sort_unstable();
            let seen: Vec<usize> = runs.iter().map(|&(k, ..)| k).collect();
            assert_eq!(seen, (0..outer).collect::<Vec<_>>(), "{ctx}: node {node}");
            runs.iter().map(|&(_, _, s, e)| (s, e)).collect()
        };
        let (pre_t, post_t, decide_t) = (once(pre), once(post), once(decision));
        let mut at = vec![vec![vec![(0, 0); trips]; outer]; n];
        for j in 0..n {
            let mut seen: Vec<(usize, usize)> =
                ticks[j + 2].iter().map(|&(k, t, ..)| (k, t)).collect();
            seen.sort_unstable();
            let each: Vec<(usize, usize)> = (0..outer)
                .flat_map(|k| (0..trips).map(move |t| (k, t)))
                .collect();
            assert_eq!(seen, each, "{ctx}: b{j}");
            for &(k, t, s, e) in &ticks[j + 2] {
                at[j][k][t] = (s, e);
            }
            let entries: Vec<_> = record
                .tasks
                .iter()
                .filter(|e| e.name == format!("b{j}"))
                .map(|e| (e.outer, e.iteration))
                .collect();
            let want: Vec<_> = each.iter().map(|&(k, t)| (Some(k), Some(t))).collect();
            assert_eq!(entries, want, "{ctx}: b{j}'s entries");
        }
        for name in ["pre", "post", "decide"] {
            let entries: Vec<_> = record
                .tasks
                .iter()
                .filter(|e| e.name == name)
                .map(|e| (e.outer, e.iteration))
                .collect();
            let want: Vec<_> = (0..outer).map(|k| (None, Some(k))).collect();
            assert_eq!(entries, want, "{ctx}: {name}'s entries");
        }
        for k in 0..outer {
            for j in 0..n {
                for t in 0..trips {
                    let (start, end) = at[j][k][t];
                    let here = format!("{ctx}: b{j}@{k}.{t}");
                    assert!(pre_t[k].1 < start, "{here} before pre@{k}");
                    assert!(
                        k == 0 || decide_t[k - 1].1 < start,
                        "{here} before the decision"
                    );
                    assert!(end < decide_t[k].0, "{here} after its decision");
                    for &p in &shape.preds[j] {
                        assert!(at[p][k][t].1 < start, "{here} before b{p}");
                    }
                    if t >= 1 {
                        assert!(at[j][k][t - 1].1 < start, "{here} before itself");
                    }
                    if t >= depth {
                        for (i, runs) in at.iter().enumerate() {
                            assert!(runs[k][t - depth].1 < start, "{here} before b{i}");
                        }
                    }
                }
            }
            for &(_, end) in &at[tail][k] {
                assert!(end < post_t[k].0, "{ctx}: post@{k} before b{tail}");
            }
        }
    }
}

/// Tier 3: a counted loop on two lanes inside a decision loop, a zero
/// before it and a download after it on lane 0, the shape of J and K over
/// a streamed tensor every SCF iteration: the record's only host waits are
/// the decision's on the download, one per iteration, and the lanes'
/// exits, so the counted loop adds no host wait for its GPU nodes even
/// though the decision follows them; and each body node's entries carry
/// both indices, which read back from `timing.json`.
#[test]
fn a_counted_loop_in_a_decision_loop_adds_no_host_wait() {
    const OUTER: usize = 3;
    let mut plan: Plan<'_, ()> = Plan::new();
    let zero = plan.add("zero", on(0, 0), &[], |_| Ok(()));
    let stage = plan.add("stage", on(0, 1), &[zero], |_| Ok(()));
    let add = plan.add("add", on(0, 0), &[stage], |_| Ok(()));
    plan.add_counted(stage, 4, 2);
    let down = plan.add("download", on(0, 0), &[add], |_| Ok(()));
    // The planner names every body node to the decision: they add nothing.
    let decision = plan.add_loop("decide", zero, &[zero, stage, add], OUTER, |i| {
        Ok(if i + 1 < OUTER {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    });
    let record = run(&mut plan, &mut HostOnly::new(1), 1);
    record.result.as_ref().unwrap();
    let syncs: Vec<_> = record
        .syncs
        .iter()
        .map(|s| (s.node, s.iteration, s.outer, s.waited_for))
        .collect();
    let mut want: Vec<_> = (0..OUTER)
        .map(|k| (Some(decision), Some(k), None, down))
        .collect();
    want.extend([(None, None, None, stage), (None, None, None, down)]);
    assert_eq!(syncs, want);
    for name in ["stage", "add"] {
        let entries: Vec<_> = record
            .tasks
            .iter()
            .filter(|t| t.name == name)
            .map(|t| (t.outer, t.iteration))
            .collect();
        let each: Vec<_> = (0..OUTER)
            .flat_map(|k| (0..4).map(move |t| (Some(k), Some(t))))
            .collect();
        assert_eq!(entries, each, "{name}");
    }
    assert_eq!(record.iterations, BTreeMap::from([(decision, OUTER)]));
    let json = record.to_json();
    assert_eq!(serde_json::from_str::<Record>(&json).unwrap(), record);
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    // Entries by rank, then outer iteration, then trip: zero's three,
    // then stage's (0, 0) and (0, 1).
    assert_eq!(value["tasks"][3]["name"], "stage");
    assert_eq!(value["tasks"][4]["outer"], 0);
    assert_eq!(value["tasks"][4]["iteration"], 1);
    assert!(value["tasks"][0]["outer"].is_null());
    assert!(value["syncs"][0]["outer"].is_null());
}

#[test]
fn default_n_cpu_leaves_the_gpu_threads_and_keeps_one() {
    assert!(default_n_cpu(0) >= 1);
    assert_eq!(default_n_cpu(usize::MAX), 1);
}

/// Device 0: the GPU worker's device is current, work still running on one
/// lane is waited for by a node on another lane of the same worker, on the
/// device, and by a CPU node, on the host, and the run returns with nothing
/// outstanding.
///
/// Lane 0's node enqueues a chain of device-to-device copies that ends by
/// writing `src`, so it is still running when the node on lane 1 copies
/// `src` on. Without the wait the copy reads the zeros `src` started as;
/// without the CPU node's event sync the read sees `dst` before the copy
/// lands. Either fault fails the comparison. The lane-to-lane edge adds no
/// sync point: the record's are the CPU node's and the two lanes' exits.
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
    let produce = plan.add("produce", on(0, 0), &[], |lane| {
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
    let copy = plan.add("copy", on(0, 1), &[produce], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        d2d(&dst, &src, stream);
        Ok(())
    });
    let read = plan.add("read", Placement::Cpu, &[copy], |_| {
        // A blocking copy on the legacy stream would hide a missing wait, so
        // read on a fresh non-blocking stream, synced alone; the edge's
        // event sync is all that orders the copy before this read.
        *seen_by_cpu.lock().unwrap() = dst.read(&Stream::new(0));
        Ok(())
    });
    let mut device = Device::new(&[0], &plan, &[], &[]);
    let record = run(&mut plan, &mut device, 1);
    record.result.as_ref().unwrap();
    assert_eq!(*devices_seen.lock().unwrap(), [0]);
    assert!(
        *seen_by_cpu.lock().unwrap() == pattern,
        "a device edge did not wait"
    );
    let syncs: Vec<_> = record
        .syncs
        .iter()
        .map(|s| (s.node, s.worker.as_str(), s.waited_for))
        .collect();
    assert_eq!(
        syncs,
        [
            (Some(read), "cpu0", copy),
            (None, "gpu0.0", produce),
            (None, "gpu0.1", copy),
        ]
    );
}

/// `n` bytes of `(i + seed) mod 251`: a pattern a zeroed buffer cannot
/// match, distinct per seed.
#[cfg(feature = "device-tests")]
fn pattern(n: usize, seed: usize) -> Vec<u8> {
    (0..n).map(|i| ((i + seed) % 251) as u8).collect()
}

/// Enqueue a device-to-device copy of `n` bytes on `stream`, left in
/// flight: a helper that synced would hide a missing wait.
#[cfg(feature = "device-tests")]
fn d2d_on(
    to: &fb_core_wwr::runtime::tests::DeviceBytes,
    from: &fb_core_wwr::runtime::tests::DeviceBytes,
    n: usize,
    stream: &fb_core_wwr::runtime::Stream,
) {
    use fb_core_wwr::handle::DeviceHandleStream;
    use wwr_sys::wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE;
    // SAFETY: both device ranges are at least n bytes and outlive the run.
    fb_core_wwr::runtime::tests::ok(unsafe {
        wwr_sys::wwr_memcpy_async(
            to.0,
            from.0,
            n,
            WWR_MEMCPY_DEVICE_TO_DEVICE,
            stream.stream(),
        )
    });
}

/// Device 0: the work a node forks ([`DeviceStream::fork_join`]) is
/// ordered before what the executor queues after the node on its lane and
/// before a node on another lane that reads its output, both on the
/// device, and the forks are the worker's own streams, not a lane's.
///
/// The node forks three items over a pool of two, so one fork carries two.
/// Item `c` enqueues a chain of device-to-device copies that ends by
/// writing `src[c]`, so it is still running when the node returns. A node
/// after it on the same lane, and one on another lane, each copy every
/// `src[c]` on; without the joins either reads the zeros `src` started as,
/// and the comparison on the host, after the CPU node's event syncs,
/// fails.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn forked_work_precedes_the_lane_and_the_readers_after_the_node() {
    use fb_core_wwr::handle::DeviceHandleStream;
    use fb_core_wwr::runtime::Stream;
    use fb_core_wwr::runtime::tests::DeviceBytes;

    const N: usize = 32 << 20;
    const HOPS: usize = 32;
    const ITEMS: usize = 3;
    let patterns: Vec<Vec<u8>> = (0..ITEMS).map(|c| pattern(N, c)).collect();
    let zeros = vec![0u8; N];
    // Each upload is drained before it returns, so all of it is before the run.
    let setup = Stream::new(0);
    let scratch: Vec<(DeviceBytes, DeviceBytes)> = patterns
        .iter()
        .map(|p| (DeviceBytes::upload(p, &setup), DeviceBytes::new(N)))
        .collect();
    let [src, same, other] = [(); 3].map(|_| {
        (0..ITEMS)
            .map(|_| DeviceBytes::upload(&zeros, &setup))
            .collect::<Vec<_>>()
    });
    let streams = Mutex::new(Vec::new());
    let seen = Mutex::new(Vec::new());

    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    let fork = plan.add("fork", on(0, 0), &[], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        let mut used = vec![stream.stream() as usize];
        stream.fork_join(ITEMS, |c, s| {
            let (a, b) = &scratch[c];
            for _ in 0..HOPS / 2 {
                d2d_on(b, a, N, s);
                d2d_on(a, b, N, s);
            }
            d2d_on(&src[c], a, N, s);
            used.push(s.stream() as usize);
            Ok(())
        })?;
        *streams.lock().unwrap() = used;
        Ok(())
    });
    let after = plan.add("after", on(0, 0), &[fork], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        (0..ITEMS).for_each(|c| d2d_on(&same[c], &src[c], N, stream));
        Ok(())
    });
    let reader = plan.add("other lane", on(0, 1), &[fork], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        (0..ITEMS).for_each(|c| d2d_on(&other[c], &src[c], N, stream));
        Ok(())
    });
    plan.add("read", Placement::Cpu, &[after, reader], |_| {
        // A fresh non-blocking stream, synced alone: the edges' event
        // syncs are all that order the copies before this read.
        let fresh = Stream::new(0);
        *seen.lock().unwrap() = [&same, &other]
            .map(|bufs| bufs.iter().map(|b| b.read(&fresh)).collect::<Vec<_>>())
            .to_vec();
        Ok(())
    });
    let mut device = Device::new(&[0], &plan, &[], &[2]);
    run(&mut plan, &mut device, 1).result.unwrap();

    let used = streams.lock().unwrap().clone();
    let lanes = [0, 1].map(|l| device.stream(LaneId { worker: 0, lane: l }).stream() as usize);
    assert_eq!(used.len(), 1 + ITEMS);
    assert_eq!(used[0], lanes[0]);
    let (f0, f1) = (used[1], used[2]);
    assert_ne!(f0, f1, "two items, two forks");
    assert_eq!(used[3], f0, "item 2 on fork 2 mod 2");
    for f in [f0, f1] {
        assert!(!lanes.contains(&f), "a fork is no lane's stream");
    }
    let seen = seen.lock().unwrap();
    for (which, bufs) in ["the lane after the node", "another lane"]
        .iter()
        .zip(seen.iter())
    {
        for (c, got) in bufs.iter().enumerate() {
            assert!(*got == patterns[c], "{which} ran ahead of fork item {c}");
        }
    }
}

/// Device 0: a [`DeviceStream::fork_join`] whose item fails runs no item
/// after it, returns that failure, and still joins every fork it made, so
/// the forking stream does not run ahead of the items in flight.
///
/// Items 0 and 1 write `src[0]` and `src[1]` after a chain of copies, item
/// 2 fails. The node then copies both on, on its own stream; the host
/// reads the copies after the run.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_failed_fork_item_still_joins_the_forks_made() {
    use fb_core_wwr::runtime::Stream;
    use fb_core_wwr::runtime::tests::DeviceBytes;

    const N: usize = 32 << 20;
    const HOPS: usize = 32;
    let patterns: Vec<Vec<u8>> = (0..2).map(|c| pattern(N, c)).collect();
    let zeros = vec![0u8; N];
    let setup = Stream::new(0);
    let scratch: Vec<(DeviceBytes, DeviceBytes)> = patterns
        .iter()
        .map(|p| (DeviceBytes::upload(p, &setup), DeviceBytes::new(N)))
        .collect();
    let [src, out] = [(); 2].map(|_| {
        (0..2)
            .map(|_| DeviceBytes::upload(&zeros, &setup))
            .collect::<Vec<_>>()
    });
    let ran = Mutex::new(Vec::new());

    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    plan.add("fork", on(0, 0), &[], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        let failed = stream.fork_join(4, |c, s| {
            ran.lock().unwrap().push(c);
            if c == 2 {
                return Err(Error::gpu("item 2"));
            }
            let (a, b) = &scratch[c];
            for _ in 0..HOPS / 2 {
                d2d_on(b, a, N, s);
                d2d_on(a, b, N, s);
            }
            d2d_on(&src[c], a, N, s);
            Ok(())
        });
        assert_eq!(failed.unwrap_err().message, "item 2");
        (0..2).for_each(|c| d2d_on(&out[c], &src[c], N, stream));
        Ok(())
    });
    let mut device = Device::new(&[0], &plan, &[], &[4]);
    run(&mut plan, &mut device, 1).result.unwrap();

    assert_eq!(*ran.lock().unwrap(), [0, 1, 2], "no item after the failure");
    let fresh = Stream::new(0);
    for (c, buf) in out.iter().enumerate() {
        assert!(
            buf.read(&fresh) == patterns[c],
            "the stream ran ahead of item {c}"
        );
    }
}

/// Device 0: a worker with fewer than two fork streams declared has none,
/// and [`DeviceStream::fork_join`] runs every item on the forking stream
/// itself, in order.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_worker_with_no_forks_runs_the_items_on_the_lane() {
    use fb_core_wwr::handle::DeviceHandleStream;

    let plan: Plan<'_, DeviceStream> = Plan::new();
    for forks in [&[][..], &[1]] {
        let device = Device::new(&[0], &plan, &[], forks);
        let stream = device.stream(LaneId { worker: 0, lane: 0 });
        let mut items = Vec::new();
        stream
            .fork_join(3, |c, s| {
                items.push((c, s.stream()));
                Ok(())
            })
            .unwrap();
        let lane = stream.stream();
        assert_eq!(items, [(0, lane), (1, lane), (2, lane)], "{forks:?}");
    }
}

/// Device 0: each lane's BLAS and dense-solver handles are made with the
/// backend, on the worker's device and bound to the lane's stream, one per
/// lane, and are the same handles at every run, so no task makes one.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_library_handles_are_the_workers_made_before_the_run() {
    use fb_core_wwr::handle::{DeviceHandle, DeviceHandleStream};
    use wwr_sys::{wwr_blas_status, wwr_solver_status, wwr_stream};

    let seen = Mutex::new(Vec::new());
    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    for s in 0..2 {
        plan.add(format!("blas{s}"), on(0, s), &[], |lane| {
            let Lane::Stream { lane, stream, .. } = lane else {
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
                .push((lane, blas.get() as usize, solver.get() as usize));
            Ok(())
        });
    }
    let mut device = Device::new(&[0], &plan, &[vec![true, true]], &[]);
    let mut runs = Vec::new();
    for _ in 0..2 {
        run(&mut plan, &mut device, 1).result.unwrap();
        let mut handles = std::mem::take(&mut *seen.lock().unwrap());
        handles.sort_unstable();
        runs.push(handles);
    }
    assert_eq!(runs[0].len(), 2);
    assert_ne!(runs[0][0].1, runs[0][1].1, "one BLAS handle per lane");
    assert_ne!(runs[0][0].2, runs[0][1].2, "one solver handle per lane");
    assert_eq!(runs[0], runs[1], "a run made a handle of its own");
}

/// Device 0: a lane the libraries leave unmarked, or do not reach, is made
/// with no library handles; a marked lane has both, so a lane whose nodes
/// call no library costs no handle (`docs/gpu_layer.md`, "Layering").
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_lane_with_no_library_user_has_no_handles() {
    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    for s in 0..4 {
        plan.add(format!("kernel{s}"), on(0, s), &[], |_| Ok(()));
    }
    let device = Device::new(&[0], &plan, &[vec![true, false, true]], &[]);
    let made = |f: &dyn Fn()| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_ok();
    let has: Vec<(bool, bool)> = (0..4)
        .map(|l| {
            let stream = device.stream(LaneId { worker: 0, lane: l });
            (
                made(&|| {
                    let _ = stream.blas();
                }),
                made(&|| {
                    let _ = stream.solver();
                }),
            )
        })
        .collect();
    assert_eq!(
        has,
        [(true, true), (false, false), (true, true), (false, false)]
    );
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
    let copies = plan.add("copies", on(0, 0), &[], |lane| {
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
    let mut device = Device::new(&[0], &plan, &[], &[]);
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

/// Device 0: a wait on an event binds to the recording the event holds
/// when the wait is enqueued, not to one recorded after it and before the
/// device reaches the wait (`docs/programming_model.md`, "Plan once, run
/// many"): record `e` on stream `a` after a long chain of copies, enqueue
/// on stream `b` a wait for `e` and an event `f` after it, then record `e`
/// again on `a` after a second, longer chain. Bound at its enqueue, the
/// wait is released by the first recording, and `f` lands near the second
/// chain's start; bound when the device reaches it, by the second, and
/// `f` lands after that chain's end.
///
/// The bar, by analysis: `f` minus the second chain's start is the first
/// recording's lead over that start (the two are back to back on `a`, so
/// about zero) under the first reading, and at least the chain's length
/// under the second; half the chain's length lies between them by
/// milliseconds, against an event pair's resolution of about a
/// microsecond. The case discriminates only if every enqueue finished
/// before the device reached the first recording, so the host's enqueue
/// time is held under the first chain's device time first.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
#[allow(
    clippy::disallowed_methods,
    reason = "a test reads its events back on the host: the harness waits, not a task"
)]
fn a_wait_binds_to_the_recording_enqueued_before_it() {
    use fb_core_wwr::handle::DeviceHandleStream;
    use fb_core_wwr::runtime::tests::{DeviceBytes, ok};
    use fb_core_wwr::runtime::{Event, Stream};
    use wwr_sys::wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE;

    const N: usize = 32 << 20;
    const HOPS: usize = 64;
    let (a, b) = (Stream::new(0), Stream::new(0));
    let x = DeviceBytes::upload(&vec![3u8; N], &a);
    let y = DeviceBytes::new(N);
    let chain = |hops: usize| {
        for hop in 0..hops {
            let (to, from) = if hop % 2 == 0 { (&y, &x) } else { (&x, &y) };
            // SAFETY: both device ranges are N bytes and outlive the case.
            ok(unsafe {
                wwr_sys::wwr_memcpy_async(to.0, from.0, N, WWR_MEMCPY_DEVICE_TO_DEVICE, a.stream())
            });
        }
    };
    // Timing-capable events, all of them.
    let [first_start, e, f, second_start] = [(); 4].map(|_| Event::with_flags(0, 0));
    let host = std::time::Instant::now();
    ok(first_start.record(&a));
    chain(HOPS);
    ok(e.record(&a));
    ok(b.wait_event(&e, 0));
    ok(f.record(&b));
    ok(second_start.record(&a));
    chain(2 * HOPS);
    ok(e.record(&a));
    let host = host.elapsed();
    ok(a.sync());
    ok(b.sync());

    let ms = |end: &Event, start: &Event| end.elapsed_since(start).expect("both recorded");
    let first = ms(&second_start, &first_start);
    let second = ms(&e, &second_start);
    let lead = ms(&f, &second_start);
    println!(
        "enqueues {host:?}; first chain {first} ms; released {lead} ms into a {second} ms chain"
    );
    assert!(
        host.as_secs_f32() * 1e3 < first,
        "the enqueues took {host:?}, the first chain {first} ms on the device: the case cannot tell"
    );
    assert!(
        lead < second / 2.0,
        "the wait was released {lead} ms into a {second} ms chain recorded after it was enqueued"
    );
}

/// Device 0, one run: a counted loop of three trips at depth 2 whose body
/// is one stream node, trip `k` enqueueing `k + 1` times one chain of
/// copies: each trip's stream scope is its own, read after the run, so
/// the device times rise strictly; and the loop adds no sync point, the
/// lane's exit the record's only one.
#[cfg(feature = "device-tests")]
#[test]
#[ignore = "tier 4: needs a GPU"]
fn device_counted_loop_times_each_trip() {
    use fb_core_wwr::handle::DeviceHandleStream;
    use fb_core_wwr::runtime::Stream;
    use fb_core_wwr::runtime::tests::{DeviceBytes, ok};
    use wwr_sys::wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE;

    const N: usize = 32 << 20;
    const HOPS: usize = 16;
    const TRIPS: usize = 3;
    let setup = Stream::new(0);
    let a = DeviceBytes::upload(&vec![7u8; N], &setup);
    let b = DeviceBytes::new(N);

    let mut plan: Plan<'_, DeviceStream> = Plan::new();
    let copies = plan.add("copies", on(0, 0), &[], |lane| {
        let Lane::Stream {
            stream, iteration, ..
        } = lane
        else {
            unreachable!()
        };
        for hop in 0..(iteration + 1) * HOPS {
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
    plan.add_counted(copies, TRIPS, 2);
    let mut device = Device::new(&[0], &plan, &[], &[]);
    let record = run(&mut plan, &mut device, 1);
    record.result.as_ref().unwrap();

    let scopes: Vec<_> = record
        .tasks
        .iter()
        .map(|t| (t.iteration, t.device))
        .collect();
    assert_eq!(scopes.len(), TRIPS, "{scopes:?}");
    for (k, &(iteration, device)) in scopes.iter().enumerate() {
        assert_eq!(iteration, Some(k));
        assert!(device.is_some(), "trip {k}'s scope was not read");
    }
    assert!(
        scopes.windows(2).all(|w| w[0].1 < w[1].1),
        "each trip's scope is its own: {scopes:?}"
    );
    let syncs: Vec<_> = record
        .syncs
        .iter()
        .map(|s| (s.node, s.waited_for))
        .collect();
    assert_eq!(syncs, [(None, copies)]);
}
