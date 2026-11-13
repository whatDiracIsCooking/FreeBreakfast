//! The executor's promises over random DAGs, with [`HostOnly`] workers (tier
//! 3), and the device edges once on device 0 (tier 4).

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
        run(&plan, &HostOnly::new(shape.n_streams), n_cpu)
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
    let error = run(&plan, &HostOnly::new(1), 1).unwrap_err();
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
    run(&plan, &HostOnly::new(1), 1).unwrap();
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
    let _ = run(&plan, &HostOnly::new(1), 3);
}

#[test]
fn a_plan_the_workers_cannot_run_is_refused_before_anything_runs() {
    let ran = AtomicUsize::new(0);
    let mut plan = Plan::new();
    plan.add("a", Placement::Stream(1), &[], |_: Lane<'_, ()>| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let error = run(&plan, &HostOnly::new(1), 1).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);

    let mut plan = Plan::new();
    plan.add("a", Placement::Cpu, &[], |_: Lane<'_, ()>| {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    let error = run(&plan, &HostOnly::new(0), 0).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert_eq!(ran.load(Ordering::SeqCst), 0);
}

#[test]
fn an_empty_plan_runs() {
    let plan: Plan<'_, ()> = Plan::new();
    run(&plan, &HostOnly::new(2), 1).unwrap();
}

#[test]
#[should_panic(expected = "predecessor 0 is not in the plan")]
fn a_predecessor_from_elsewhere_is_rejected() {
    let mut other: Plan<'_, ()> = Plan::new();
    let foreign = other.add("x", Placement::Cpu, &[], |_| Ok(()));
    let mut plan: Plan<'_, ()> = Plan::new();
    plan.add("a", Placement::Cpu, &[foreign], |_| Ok(()));
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
fn device_edges_wait_for_the_work_they_consume() {
    use fb_core_wwr::handle::DeviceHandleStream;
    use fb_core_wwr::runtime::tests::{DeviceBytes, ok};
    use wwr_sys::wwr_memcpy_kind::{
        WWR_MEMCPY_DEVICE_TO_DEVICE, WWR_MEMCPY_DEVICE_TO_HOST, WWR_MEMCPY_HOST_TO_DEVICE,
    };

    const N: usize = 64 << 20;
    const HOPS: usize = 32;
    let pattern: Vec<u8> = (0..N).map(|i| (i % 251) as u8).collect();
    let [a, b, src, dst] = [(); 4].map(|_| DeviceBytes::new(N));
    // SAFETY: each range is N bytes; synchronous, before the run.
    unsafe {
        ok(wwr_sys::wwr_memcpy(
            a.0,
            pattern.as_ptr().cast(),
            N,
            WWR_MEMCPY_HOST_TO_DEVICE,
        ));
        ok(wwr_sys::wwr_memset(src.0, 0, N));
        ok(wwr_sys::wwr_memset(dst.0, 0, N));
    }
    let [a, b, src, dst] = [&a, &b, &src, &dst].map(|d| d.0 as usize);
    let d2d = |to: usize, from: usize, stream: &DeviceStream| {
        // SAFETY: both device ranges are N bytes and outlive the run.
        ok(unsafe {
            wwr_sys::wwr_memcpy_async(
                to as _,
                from as _,
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
            d2d(b, a, stream);
            d2d(a, b, stream);
        }
        d2d(src, a, stream);
        Ok(())
    });
    let copy = plan.add("copy", Placement::Stream(1), &[produce], |lane| {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!()
        };
        d2d(dst, src, stream);
        Ok(())
    });
    plan.add("read", Placement::Cpu, &[copy], |_| {
        let mut host = seen_by_cpu.lock().unwrap();
        // A blocking copy on the legacy stream would hide a missing wait, so
        // read with an async copy on a fresh non-blocking stream and sync it
        // alone.
        let s = DeviceStream::new(0);
        // SAFETY: both ranges are N bytes; the edge's event sync ordered the
        // copy before this read.
        ok(unsafe {
            wwr_sys::wwr_memcpy_async(
                host.as_mut_ptr().cast(),
                dst as _,
                N,
                WWR_MEMCPY_DEVICE_TO_HOST,
                s.stream(),
            )
        });
        ok(s.sync());
        Ok(())
    });
    let device = Device::new(&[0, 0], &plan);
    run(&plan, &device, 1).unwrap();
    assert_eq!(*devices_seen.lock().unwrap(), [0]);
    assert!(
        *seen_by_cpu.lock().unwrap() == pattern,
        "a device edge did not wait"
    );
}
