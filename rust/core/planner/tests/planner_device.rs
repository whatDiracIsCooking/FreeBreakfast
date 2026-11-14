//! The planner on a real card (device 0): tasks declared once, planned,
//! instantiated and run on a stream worker and the CPU pool.
//!
//! A binary of its own: a case measures the card's free memory, which
//! another test allocating beside it would move, so the cases take turns.
//!
//! The reading is the device's, every tenant of the card counted, and a
//! process outside the gates' card lock can free memory mid-case (#1553).
//! So the checks that a run allocates nothing ask that free memory never
//! fall: an allocation of this process lowers it, another tenant's free
//! cannot. `fb-pipeline-scf`'s `tests/scf.rs` says what that gives up
//! against equality, and why nothing narrower can be read.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fb_core_arena::{
    Bufs, DEVICE_ALIGNMENT, Device, Layout, Pinned, RowId, memcpy_d2d, memcpy_d2p, memcpy_p2d,
};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Decision, Task};
use fb_core_thread_pool::{DeviceStream, Lane, Placement};
use fb_core_wwr::blas::gemm;
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_core_wwr::memset;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;
use wwr_sys::wwr_status;

const N: usize = 4096;

/// Held by each case for its whole length.
static TURN: Mutex<()> = Mutex::new(());

/// Device 0's free memory, with it made current.
#[allow(
    clippy::disallowed_methods,
    reason = "a test measuring a plan's blocks on the card reads free memory; no width is sized from it"
)]
fn free_bytes() -> usize {
    let _device = ScopedDeviceIndex::new(0);
    let (mut free, mut total) = (0, 0);
    // SAFETY: a plain runtime call writing two locals.
    let status = unsafe { wwr_sys::wwr_mem_get_info(&mut free, &mut total) };
    assert_eq!(status, wwr_status::WWR_STATUS_SUCCESS);
    free
}

fn value(i: usize) -> f64 {
    i as f64 * 0.5
}

/// CPU: fills a pinned row.
struct Fill;

impl Task for Fill {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        h.output("up", N)
    }

    fn run(
        &self,
        up: &Self::Rows,
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let up = h.take_mut(*up);
        assert_eq!(up.as_ptr().addr() % DEVICE_ALIGNMENT, 0);
        up.iter_mut().enumerate().for_each(|(i, e)| *e = value(i));
        Ok(())
    }
}

/// GPU: copies a pinned row to a device one, or a device row to a pinned
/// one, on its stream.
struct Upload(RowId<Pinned, f64>);
struct Download(RowId<Device, f64>);

impl Task for Upload {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        h.read(self.0);
        d.output("dev", N)
    }

    fn run(
        &self,
        dev: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let dst = d.take_mut(*dev);
        assert_eq!(dst.as_ptr().addr() % DEVICE_ALIGNMENT, 0);
        // The source is not rewritten before this copy completes: its
        // bytes' next owner is ordered after this task's event by a handoff.
        memcpy_p2d(&dst, h, self.0, stream)
    }
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0);
        h.output("down", N)
    }

    fn run(
        &self,
        down: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        // No host write here: `down` is `up`'s bytes, and the upload out of
        // them may still be in flight. A stream task only enqueues.
        memcpy_d2p(h, *down, d.get(self.0), stream)
    }
}

/// CPU: compares, after the executor waited on the download's event.
struct Check(RowId<Pinned, f64>);

impl Task for Check {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) {
        h.read(self.0);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let got = h.get(self.0);
        assert!(got.iter().enumerate().all(|(i, &e)| e == value(i)));
        Ok(())
    }
}

/// A round trip through the planner: a CPU task fills a pinned row, a
/// stream task copies it up, another copies it down into a second pinned
/// row that takes the first's bytes, and a CPU task compares. Nothing but
/// the derived dependencies orders the reuse. The plan runs twice with the
/// card's free memory not fallen between: a run allocates nothing.
///
/// The record carries the plan's device and pinned block sizes, the
/// tables' own, and a host wait for every CPU task's GPU predecessor: the
/// check waits on the download's event.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_round_trip_through_the_planner() {
    let _turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    let mut dag = Dag::new();
    let up = dag.add("fill", 0, Fill);
    let dev = dag.add("upload", 0, Upload(up));
    let down = dag.add("download", 0, Download(dev));
    dag.add("check", 0, Check(down));
    let planned = dag.plan();
    assert_eq!(
        planned.host_table(0).offset(down),
        planned.host_table(0).offset(up),
        "down reuses up's bytes, which only the handoffs order"
    );
    assert_eq!(planned.predecessors(2), &[0, 1]);
    let sizes = [
        (0, "device", planned.device_table(0).total()),
        (0, "pinned", planned.host_table(0).total()),
    ];
    // Every (CPU task, GPU predecessor) pair: each is a host wait.
    let waits: Vec<(usize, usize)> = (0..planned.len())
        .filter(|&pos| planned.placement(pos) == Placement::Cpu)
        .flat_map(|pos| {
            let gpu = planned
                .predecessors(pos)
                .iter()
                .filter(|&&u| planned.placement(u) != Placement::Cpu);
            gpu.map(move |&u| (pos, u)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(waits, [(3, 2)]);

    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    record.result.unwrap();
    let arena: Vec<_> = record
        .arena
        .iter()
        .map(|a| (a.worker, a.kind.as_str(), a.bytes))
        .collect();
    assert_eq!(arena, sizes);
    for (pos, u) in waits {
        let sync = record
            .syncs
            .iter()
            .find(|s| s.node.map(|n| n.index()) == Some(pos) && s.waited_for.index() == u)
            .unwrap_or_else(|| panic!("no host wait of task {pos} on task {u}"));
        assert!(sync.host >= Duration::ZERO, "{sync:?}");
    }
    let between = free_bytes();
    instance.run(&mut backend, 1).result.unwrap();
    let after = free_bytes();
    assert!(
        after >= between,
        "a run allocated device memory: free {between} before, {after} after"
    );
}

/// The length of a [`Stage`]'s row: big enough that its copy is real device
/// time.
const BIG: usize = 4 << 20;

/// GPU: writes a device row, by a memset when it reads nothing, else by
/// copying the row it reads.
struct Stage(Option<RowId<Device, f64>>);

impl Task for Stage {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        if let Some(input) = self.0 {
            d.read(input);
        }
        d.output("out", BIG)
    }

    fn run(
        &self,
        out: &Self::Rows,
        d: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut dst = d.take_mut(*out);
        match self.0 {
            None => memset::zero(&mut dst, stream),
            Some(input) => memcpy_d2d(&dst, d.get(input), stream),
        }
    }
}

/// A chain of GPU tasks on one stream, timed by the executor alone: one
/// entry per task, each with its stream scope's device time, and those
/// sum to at most the run's wall time, since one stream runs them one
/// after another inside the run. Every bound holds by construction.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn every_gpu_task_has_a_stream_scope() {
    let _turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    let mut dag = Dag::new();
    let mut row = dag.add("stage0", 0, Stage(None));
    for i in 1..6 {
        row = dag.add(format!("stage{i}"), 0, Stage(Some(row)));
    }
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    record.result.unwrap();
    let names: Vec<_> = record
        .tasks
        .iter()
        .map(|t| (t.name.as_str(), t.worker.as_str()))
        .collect();
    let expected: Vec<String> = (0..6).map(|i| format!("stage{i}")).collect();
    assert_eq!(
        names,
        expected
            .iter()
            .map(|n| (n.as_str(), "stream0"))
            .collect::<Vec<_>>()
    );
    let device: Vec<Duration> = record
        .tasks
        .iter()
        .map(|t| t.device.expect("a GPU task's stream scope"))
        .collect();
    let sum: Duration = device.iter().sum();
    assert!(
        sum <= record.wall,
        "stream scopes {device:?} sum past the wall {:?}",
        record.wall
    );
}

/// The order of the affine iteration's matrix.
const M: usize = 16;

/// `A`, column-major, with every entry of magnitude at most `5 / (8 M)`, so
/// each row's magnitudes sum to at most `q = 5/8` and `x <- A x + b` is a
/// contraction in the max norm.
fn contraction() -> Vec<f64> {
    (0..M * M)
        .map(|k| ((k * 7 % 11) as f64 - 5.0) / (8.0 * M as f64))
        .collect()
}

/// `b`.
fn offset() -> Vec<f64> {
    (0..M).map(|i| (i % 5) as f64 - 2.0).collect()
}

/// The affine iteration's rows before the loop, on the host.
#[derive(Clone, Copy)]
struct HostStart {
    a: RowId<Pinned, f64>,
    b: RowId<Pinned, f64>,
    x0: RowId<Pinned, f64>,
    /// The decision's carried last iterate.
    prev: RowId<Pinned, f64>,
}

/// CPU: `A`, `b`, and zeros for `x` and the decision's last iterate.
struct Start;

impl Task for Start {
    type Rows = HostStart;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> HostStart {
        HostStart {
            a: h.output("A", M * M),
            b: h.output("b", M),
            x0: h.output("x0", M),
            prev: h.output("prev", M),
        }
    }

    fn run(
        &self,
        r: &HostStart,
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        h.take_mut(r.a).copy_from_slice(&contraction());
        h.take_mut(r.b).copy_from_slice(&offset());
        h.take_mut(r.x0).fill(0.0);
        h.take_mut(r.prev).fill(0.0);
        Ok(())
    }
}

/// The device rows before the loop: `A`, `b`, and `x`, which the loop
/// carries.
#[derive(Clone, Copy)]
struct DeviceStart {
    a: RowId<Device, f64>,
    b: RowId<Device, f64>,
    x: RowId<Device, f64>,
}

/// GPU: `A`, `b` and `x0` up to the device.
struct Up(HostStart);

impl Task for Up {
    type Rows = DeviceStart;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> DeviceStart {
        h.read(self.0.a);
        h.read(self.0.b);
        h.read(self.0.x0);
        DeviceStart {
            a: d.output("dA", M * M),
            b: d.output("db", M),
            x: d.output("dx", M),
        }
    }

    fn run(
        &self,
        r: &DeviceStart,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_p2d(&d.take_mut(r.a), h, self.0.a, stream)?;
        memcpy_p2d(&d.take_mut(r.b), h, self.0.b, stream)?;
        memcpy_p2d(&d.take_mut(r.x), h, self.0.x0, stream)
    }
}

/// GPU, the body: `y = A x + b`, one gemm onto a copy of `b`, and `y`'s
/// pinned copy for the decision.
struct Affine(DeviceStart);

impl Task for Affine {
    type Rows = (RowId<Device, f64>, RowId<Pinned, f64>);
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0.a);
        d.read(self.0.b);
        d.read(self.0.x);
        (d.output("y", M), h.output("y host", M))
    }

    fn run(
        &self,
        &(y, y_host): &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut dy = d.take_mut(y);
        memcpy_d2d(&dy, d.get(self.0.b), stream)?;
        let m = M as i32;
        gemm(
            stream.blas(),
            OP_N,
            OP_N,
            m,
            1,
            m,
            1.0,
            &d.get(self.0.a),
            m,
            &d.get(self.0.x),
            m,
            1.0,
            &mut dy,
            m,
        )?;
        memcpy_d2p(h, y_host, &dy, stream)
    }
}

/// GPU, the body's last device task: `x = y`, the carried row's writer.
struct CarryX {
    y: RowId<Device, f64>,
    x: RowId<Device, f64>,
}

impl Task for CarryX {
    type Rows = ();
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        d.read(self.y);
        d.write(self.x);
    }

    fn run(
        &self,
        _: &(),
        d: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2d(&d.take_mut(self.x), d.get(self.y), stream)
    }
}

/// The decision: stops when `y` moved at most [`STEP`] from the last
/// iterate, which it carries in `prev`. Also reads the card's free memory
/// every iteration.
struct Moved {
    y: RowId<Pinned, f64>,
    prev: RowId<Pinned, f64>,
    free: Arc<Mutex<Vec<usize>>>,
}

/// The step below which the decision stops: the iterate is then within
/// `q / (1 - q) STEP = 5/3 STEP` of the fixed point.
const STEP: f64 = 1e-13;

impl Decision for Moved {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>) {
        h.read(self.y);
        h.write(self.prev);
    }

    fn decide(&self, _: &(), h: &Bufs<'_, Pinned>, _: usize) -> Result<ControlFlow<()>> {
        self.free.lock().unwrap().push(free_bytes());
        let y = h.get(self.y);
        let prev = h.take_mut(self.prev);
        let moved = y
            .iter()
            .zip(prev.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        prev.copy_from_slice(y);
        Ok(if moved > STEP {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        })
    }
}

/// GPU, after the loop: `x` down to a pinned row.
struct DownX(RowId<Device, f64>);

impl Task for DownX {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0);
        h.output("x", M)
    }

    fn run(
        &self,
        &x: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2p(h, x, d.get(self.0), stream)
    }
}

/// CPU: `x` out of its pinned row.
struct Keep(RowId<Pinned, f64>, Arc<Mutex<Vec<f64>>>);

impl Task for Keep {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) {
        h.read(self.0);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        h: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.1.lock().unwrap() = h.get(self.0).to_vec();
        Ok(())
    }
}

/// A loop on the card: the body is a device `x <- A x + b`, one gemm on
/// the worker's BLAS handle into a body row, then `x`, a carried device
/// row, overwritten from it; the decision reads the iterate's pinned copy
/// and carries the last one. Run once, to the fixed point.
///
/// The bar is the fixed point's own: stopping at a step of `1e-13`, the
/// iterate is within `5/3 1e-13` of `x* = A x* + b` (`q = 5/8`), so the
/// residual `A x + b - x = (A - I)(x - x*)`, taken on the host, is at most
/// `(1 + q) 5/3 1e-13 < 2.8e-13` plus its own rounding, `M` products of
/// at most `5/128 |x|` each, `|x| <= |b| / (1 - q) < 6`: under `1e-15`.
/// So `1e-12` holds with margin. The iteration count is the plan's bound's
/// business, not a bar.
///
/// And replay allocates nothing: the card's free memory, read by the
/// decision every iteration, never falls from one iteration to the next,
/// with the gemm's workspace the plan's declared row.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_device_loop_to_its_fixed_point() {
    let _turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    let (free, out) = (
        Arc::new(Mutex::new(Vec::new())),
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let host = dag.add("start", 0, Start);
    let dev = dag.add("up", 0, Up(host));
    let body = dag.len();
    let (y, y_host) = dag.add("A x + b", 0, Affine(dev));
    dag.add("x = y", 0, CarryX { y, x: dev.x });
    dag.add_loop(
        "moved",
        0,
        body,
        200,
        Moved {
            y: y_host,
            prev: host.prev,
            free: Arc::clone(&free),
        },
    );
    let x = dag.add("down", 0, DownX(dev.x));
    dag.add("keep", 0, Keep(x, Arc::clone(&out)));
    let planned = dag.plan();
    let text = planned.to_string();
    assert!(text.contains("loop 2-4: region of"), "{text}");

    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    record.result.unwrap();
    let iterations: Vec<usize> = record.iterations.values().copied().collect();
    assert_eq!(iterations.len(), 1);
    assert!(iterations[0] > 1, "{iterations:?}");

    let (a, b, x) = (contraction(), offset(), out.lock().unwrap().clone());
    let residual = (0..M)
        .map(|i| {
            let ax: f64 = (0..M).map(|k| a[i + k * M] * x[k]).sum();
            (ax + b[i] - x[i]).abs()
        })
        .fold(0.0, f64::max);
    assert!(residual <= 1e-12, "|A x + b - x| = {residual:e}");

    let free = free.lock().unwrap();
    assert_eq!(free.len(), iterations[0]);
    let fell = free.windows(2).position(|w| w[1] < w[0]);
    assert!(
        fell.is_none(),
        "free memory fell after iteration {}: {free:?}",
        fell.map_or(0, |i| i + 1)
    );
}
