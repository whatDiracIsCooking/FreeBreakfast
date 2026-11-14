//! The planner on a real card (device 0): tasks declared once, planned,
//! instantiated and run on a GPU worker's lanes and the CPU pool.
//!
//! A binary of its own, its cases taking turns: the checks that a run
//! allocates nothing read `fb_core_arena::allocations`, this process's
//! count of the arena's blocks, which a case allocating beside them would
//! move.
//!
//! Not the card's free memory (#1589): that reading is the device's, every
//! tenant counted, and on the RTX 3080, which drives a display, a client
//! outside the container moves it mid-case in either direction (#1573). It
//! is still read and printed beside the count, to be read against the
//! card's other tenants, but nothing is asserted of it. What the count does
//! not see is a library allocating behind the arena, the gemm's cuBLAS
//! among them: #1574.

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fb_core_arena::{
    Bufs, CopyOnly, DEVICE_ALIGNMENT, Device, Heap, Layout, Pinned, RowId, memcpy_d2d, memcpy_d2p,
    memcpy_p2d,
};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{CpuTask, Dag, Decision, GpuTask};
use fb_core_thread_pool::{DeviceStream, LaneId, NodeId, Placement};
use fb_core_wwr::blas::gemm;
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_core_wwr::copy;
use fb_core_wwr::device_slice::split_at_mut;
use fb_core_wwr::memset;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_N as OP_N;
use wwr_sys::wwr_status;

const N: usize = 4096;

/// Held by each case for its whole length.
static TURN: Mutex<()> = Mutex::new(());

/// Device 0's free memory, with it made current: printed, never asserted.
#[allow(
    clippy::disallowed_methods,
    reason = "a test prints the card's free memory beside the arena's count, to read against other tenants; no width is sized from it"
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

impl CpuTask for Fill {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        h.output("up", N)
    }

    fn run(&self, up: &Self::Rows, h: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
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

impl GpuTask for Upload {
    type Rows = RowId<Device, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        h.read(self.0);
        d.output("dev", N)
    }

    fn run(
        &self,
        dev: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let mut dst = d.take_mut(*dev);
        assert_eq!(dst.as_ptr().addr() % DEVICE_ALIGNMENT, 0);
        // The source is not rewritten before this copy completes: its
        // bytes' next owner is ordered after this task's event by a handoff.
        memcpy_p2d(&mut dst, h, self.0, stream)
    }
}

impl GpuTask for Download {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0);
        h.output("down", N)
    }

    fn run(
        &self,
        down: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        // No host write here: `down` is `up`'s bytes, and the upload out of
        // them may still be in flight. A stream task only enqueues.
        memcpy_d2p(h, *down, d.get(self.0), stream)
    }
}

/// CPU: compares, after the executor waited on the download's event.
struct Check(RowId<Pinned, f64>);

impl CpuTask for Check {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        h.read(self.0);
    }

    fn run(&self, _: &(), h: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let got = h.get(self.0);
        assert!(got.iter().enumerate().all(|(i, &e)| e == value(i)));
        Ok(())
    }
}

/// A round trip through the planner: a CPU task fills a pinned row, a
/// stream task copies it up, another copies it down into a second pinned
/// row that takes the first's bytes, and a CPU task compares. Nothing but
/// the derived dependencies orders the reuse. The plan runs twice, the
/// arena's count of this process's blocks the same across the second run:
/// a run allocates nothing.
///
/// The record carries the plan's device, pinned and heap block sizes, the
/// tables' own, and a host wait for every CPU task's GPU predecessor: the
/// check waits on the download's event.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_round_trip_through_the_planner() {
    let _turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    let mut dag = Dag::new();
    let up = dag.add_cpu("fill", 0, Fill);
    let dev = dag.add_gpu("upload", 0, Upload(up));
    let down = dag.add_gpu("download", 0, Download(dev));
    dag.add_cpu("check", 0, Check(down));
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
        (0, "heap", planned.heap_table().total()),
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
    let (blocks, free) = (fb_core_arena::allocations(), free_bytes());
    instance.run(&mut backend, 1).result.unwrap();
    println!(
        "the card's free memory across the second run: {free} before, {} after",
        free_bytes()
    );
    assert_eq!(
        fb_core_arena::allocations(),
        blocks,
        "the second run allocated a block"
    );
}

/// The elements of each half of a [`delay`] row, and the copies between
/// them: 32 copies of 32 MiB, milliseconds of device time on any card, so
/// the work enqueued after it on its lane is still pending when the other
/// lane's next node is enqueued.
const DELAY: usize = 4 << 20;
const DELAY_HOPS: usize = 32;

/// Copies between the two halves of the worker's delay row, `DELAY_HOPS`
/// times, on `stream`: device time and nothing else, before the work that
/// follows it on its lane.
fn delay(row: RowId<Device, f64>, d: &Bufs<'_, Device>, stream: &DeviceStream) -> Status {
    let mut row = d.take_mut(row);
    let (mut a, mut b) = split_at_mut(&mut row, DELAY);
    for hop in 0..DELAY_HOPS {
        if hop % 2 == 0 {
            copy::d2d(&mut b, &a, stream)?;
        } else {
            copy::d2d(&mut a, &b, stream)?;
        }
    }
    Ok(())
}

/// GPU: a delay, then a pinned row copied up into a device row. The delay
/// row is the worker's, so no row of the plan takes its bytes.
struct SlowUpload(RowId<Device, f64>, RowId<Pinned, f64>);

impl GpuTask for SlowUpload {
    type Rows = RowId<Device, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        h.read(self.1);
        d.output("x", N)
    }

    fn run(
        &self,
        x: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        delay(self.0, d, stream)?;
        memcpy_p2d(&mut d.take_mut(*x), h, self.1, stream)
    }
}

/// GPU: a delay, then a device row another lane wrote copied down.
struct SlowDownload(RowId<Device, f64>, RowId<Device, f64>);

impl GpuTask for SlowDownload {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.1);
        h.output("down", N)
    }

    fn run(
        &self,
        down: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        delay(self.0, d, stream)?;
        memcpy_d2p(h, *down, d.get(self.1), stream)
    }
}

/// GPU: zeroes a row of its own, which takes the bytes of the row the
/// download reads.
struct Overwrite;

impl GpuTask for Overwrite {
    type Rows = RowId<Device, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.output("y", N)
    }

    fn run(
        &self,
        y: &Self::Rows,
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memset::zero(&mut d.take_mut(*y), stream)
    }
}

/// Two lanes of one worker sharing a row, twice, with no host wait between
/// them: lane 0 uploads `x` after a delay, lane 1 reads `x` in place after
/// a delay of its own and copies it down, and lane 0 then zeroes `y`, which
/// takes `x`'s bytes. Only the read orders the download after the upload,
/// an event on the device, and only the handoff orders the zero after the
/// download, an event from lane 1 back to lane 0; the CPU check reads what
/// came down.
///
/// The delays make either fault show: without the read's wait, lane 1's
/// copy runs while lane 0 is still in its delay, and brings down what `x`
/// held before the upload; without the handoff, lane 0 zeroes `x` while
/// lane 1 is still in its delay, and lane 1 brings down zeros. `x` holds
/// zeros at the start of the second run, the first run's `y`, so the second
/// run checks the read again. The record holds no sync point for either
/// edge between the lanes: its only host waits are the check's on the
/// download and each lane's exit.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn two_lanes_share_a_row_with_no_host_wait() {
    let _turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    let lane = |lane| LaneId { worker: 0, lane };
    let mut dag = Dag::new();
    // The two delays are ordered by the read, so they may share a row.
    let wait = dag.device_worker_row(0, "delay", 2 * DELAY);
    let up = dag.add_cpu("fill", 0, Fill);
    let x = dag.add_gpu("upload", lane(0), SlowUpload(wait, up));
    let down = dag.add_gpu("download", lane(1), SlowDownload(wait, x));
    let y = dag.add_gpu("overwrite", lane(0), Overwrite);
    dag.add_cpu("check", 0, Check(down));
    let planned = dag.plan();
    assert_eq!(planned.n_lanes(0), 2);
    let table = planned.device_table(0);
    assert_eq!(table.offset(y), table.offset(x), "y takes x's bytes");
    // The download's pinned row takes the fill's bytes: a handoff from the
    // CPU, beside the read.
    assert_eq!(
        planned.predecessors(2),
        &[0, 1],
        "the read, lane 0 to lane 1"
    );
    assert!(
        planned.predecessors(3).contains(&2),
        "the handoff, lane 1 to lane 0: {:?}",
        planned.predecessors(3)
    );

    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    for run in 0..2 {
        let record = instance.run(&mut backend, 1);
        record.result.as_ref().unwrap();
        let workers: Vec<_> = record
            .tasks
            .iter()
            .map(|t| (t.name.as_str(), t.worker.as_str()))
            .collect();
        assert_eq!(
            workers,
            [
                ("fill", "cpu0"),
                ("upload", "gpu0.0"),
                ("download", "gpu0.1"),
                ("overwrite", "gpu0.0"),
                ("check", "cpu0"),
            ],
            "run {run}"
        );
        let syncs: Vec<_> = record
            .syncs
            .iter()
            .map(|s| {
                (
                    s.node.map(NodeId::index),
                    s.worker.as_str(),
                    s.waited_for.index(),
                )
            })
            .collect();
        // Ranks: the download 2, the overwrite 3, the check 4.
        assert_eq!(
            syncs,
            [
                (Some(4), "cpu0", 2),
                (None, "gpu0.1", 2),
                (None, "gpu0.0", 3)
            ],
            "run {run}: no host wait between the lanes"
        );
    }
}

/// The length of a [`Stage`]'s row: big enough that its copy is real device
/// time.
const BIG: usize = 4 << 20;

/// GPU: writes a device row, by a memset when it reads nothing, else by
/// copying the row it reads.
struct Stage(Option<RowId<Device, f64>>);

impl GpuTask for Stage {
    type Rows = RowId<Device, f64>;

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
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
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
    let mut row = dag.add_gpu("stage0", 0, Stage(None));
    for i in 1..6 {
        row = dag.add_gpu(format!("stage{i}"), 0, Stage(Some(row)));
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
            .map(|n| (n.as_str(), "gpu0.0"))
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

impl CpuTask for Start {
    type Rows = HostStart;

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> HostStart {
        HostStart {
            a: h.output("A", M * M),
            b: h.output("b", M),
            x0: h.output("x0", M),
            prev: h.output("prev", M),
        }
    }

    fn run(&self, r: &HostStart, h: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
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

impl GpuTask for Up {
    type Rows = DeviceStart;

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
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_p2d(&mut d.take_mut(r.a), h, self.0.a, stream)?;
        memcpy_p2d(&mut d.take_mut(r.b), h, self.0.b, stream)?;
        memcpy_p2d(&mut d.take_mut(r.x), h, self.0.x0, stream)
    }
}

/// GPU, the body: `y = A x + b`, one gemm onto a copy of `b`, and `y`'s
/// pinned copy for the decision.
struct Affine(DeviceStart);

impl GpuTask for Affine {
    type Rows = (RowId<Device, f64>, RowId<Pinned, f64>);

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
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
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

impl GpuTask for CarryX {
    type Rows = ();

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        d.read(self.y);
        d.write(self.x);
    }

    fn run(
        &self,
        _: &(),
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2d(&d.take_mut(self.x), d.get(self.y), stream)
    }
}

/// The decision: stops when `y` moved at most [`STEP`] from the last
/// iterate, which it carries in `prev`. Also reads the arena's count of
/// this process's blocks and the card's free memory every iteration.
struct Moved {
    y: RowId<Pinned, f64>,
    prev: RowId<Pinned, f64>,
    samples: Arc<Mutex<Vec<(u64, usize)>>>,
}

/// The step below which the decision stops: the iterate is then within
/// `q / (1 - q) STEP = 5/3 STEP` of the fixed point.
const STEP: f64 = 1e-13;

impl Decision for Moved {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        h.read(self.y);
        h.write(self.prev);
    }

    fn decide(
        &self,
        _: &(),
        h: &Bufs<'_, Pinned>,
        _: &Bufs<'_, Heap>,
        _: usize,
    ) -> Result<ControlFlow<()>> {
        self.samples
            .lock()
            .unwrap()
            .push((fb_core_arena::allocations(), free_bytes()));
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

impl GpuTask for DownX {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0);
        h.output("x", M)
    }

    fn run(
        &self,
        &x: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(h, x, d.get(self.0), stream)
    }
}

/// CPU: `x` out of its pinned row.
struct Keep(RowId<Pinned, f64>, Arc<Mutex<Vec<f64>>>);

impl CpuTask for Keep {
    type Rows = ();

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        h.read(self.0);
    }

    fn run(&self, _: &(), h: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
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
/// And replay allocates nothing: the arena's count of this process's
/// blocks, read by the decision every iteration, is the same at every one,
/// with the gemm's workspace the plan's declared row. A cuBLAS allocation
/// behind it is not in the count (#1574).
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_device_loop_to_its_fixed_point() {
    let _turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    let (samples, out) = (
        Arc::new(Mutex::new(Vec::new())),
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let host = dag.add_cpu("start", 0, Start);
    let dev = dag.add_gpu("up", 0, Up(host));
    let body = dag.len();
    let (y, y_host) = dag.add_gpu("A x + b", 0, Affine(dev));
    dag.add_gpu("x = y", 0, CarryX { y, x: dev.x });
    dag.add_loop(
        "moved",
        0,
        body,
        200,
        Moved {
            y: y_host,
            prev: host.prev,
            samples: Arc::clone(&samples),
        },
    );
    let x = dag.add_gpu("down", 0, DownX(dev.x));
    dag.add_cpu("keep", 0, Keep(x, Arc::clone(&out)));
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

    let samples = samples.lock().unwrap();
    assert_eq!(samples.len(), iterations[0]);
    let (blocks, free): (Vec<u64>, Vec<usize>) = samples.iter().copied().unzip();
    println!("the card's free memory at each iteration: {free:?}");
    let grew = blocks.windows(2).position(|w| w[1] != w[0]);
    assert!(
        grew.is_none(),
        "the arena allocated after iteration {}: blocks {blocks:?}",
        grew.map_or(0, |i| i + 1)
    );
}

/// The elements of one batch of the tensor [`batches_reduce_bitwise`]
/// streams, and its batch count: 8 MiB a batch, 64 MiB in all.
const BATCH: usize = 1 << 20;
const TRIPS: usize = 8;

/// Element `i` of batch `k`: distinct in every batch and in most bits, so
/// a batch added twice, or one taken from another batch's copy, changes
/// the sum's bits.
fn batch_value(k: usize, i: usize) -> f64 {
    ((k * BATCH + i) as f64 * 0.37).sin() * 1e3 + 1.0 / (k + 1) as f64
}

/// CPU: the tensor, cut into its batches, a pinned row each, and a one.
struct Tensor;

impl CpuTask for Tensor {
    type Rows = (Vec<RowId<Pinned, f64>>, RowId<Pinned, f64>);

    fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        let batches = (0..TRIPS).map(|_| h.output("batch", BATCH)).collect();
        (batches, h.output("one", 1))
    }

    fn run(&self, (batches, one): &Self::Rows, h: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        for (k, &row) in batches.iter().enumerate() {
            let row = h.take_mut(row);
            row.iter_mut()
                .enumerate()
                .for_each(|(i, x)| *x = batch_value(k, i));
        }
        h.take_mut(*one)[0] = 1.0;
        Ok(())
    }
}

/// The device rows a reduce starts from: the accumulator and the one.
#[derive(Clone, Copy)]
struct Sum {
    acc: RowId<Device, f64>,
    one: RowId<Device, f64>,
}

/// GPU, before the loop: the accumulator zeroed and the one uploaded.
struct Zero(RowId<Pinned, f64>);

impl GpuTask for Zero {
    type Rows = Sum;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Sum {
        h.read(self.0);
        Sum {
            acc: d.output("acc", BATCH),
            one: d.output("one", 1),
        }
    }

    fn run(
        &self,
        r: &Sum,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memset::zero(&mut d.take_mut(r.acc), stream)?;
        memcpy_p2d(&mut d.take_mut(r.one), h, self.0, stream)
    }
}

/// GPU, the copy lane: this iteration's batch uploaded into this
/// iteration's copy of `staged`.
struct StageBatch(Vec<RowId<Pinned, f64>>);

impl GpuTask for StageBatch {
    type Rows = RowId<Device, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        for &batch in &self.0 {
            h.read(batch);
        }
        d.output("staged", BATCH)
    }

    fn run(
        &self,
        staged: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let batch = self.0[d.iteration()];
        memcpy_p2d(&mut d.take_mut(*staged), h, batch, stream)
    }
}

/// GPU, the compute lane: a delay, then the staged batch added into the
/// accumulator, `acc <- 1 staged 1 + 1 acc`, a gemm of inner dimension 1:
/// each element's one product is exact, so each element is `acc + staged`
/// rounded once, the host's sum in the batches' order.
struct Reduce {
    delay: RowId<Device, f64>,
    staged: RowId<Device, f64>,
    sum: Sum,
}

impl GpuTask for Reduce {
    type Rows = ();

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        d.read(self.staged);
        d.read(self.sum.one);
        d.write(self.sum.acc);
    }

    fn run(
        &self,
        _: &(),
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        delay(self.delay, d, stream)?;
        let m = BATCH as i32;
        gemm(
            stream.blas(),
            OP_N,
            OP_N,
            m,
            1,
            1,
            1.0,
            &d.get(self.staged),
            m,
            &d.get(self.sum.one),
            1,
            1.0,
            &mut d.take_mut(self.sum.acc),
            m,
        )
    }
}

/// GPU, after the loop: the accumulator copied down.
struct DownSum(RowId<Device, f64>);

impl GpuTask for DownSum {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Self::Rows {
        d.read(self.0);
        h.output("sum", BATCH)
    }

    fn run(
        &self,
        sum: &Self::Rows,
        d: &Bufs<'_, Device>,
        h: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(h, *sum, d.get(self.0), stream)
    }
}

/// A tensor cut into batches, uploaded and reduced in a counted loop on
/// two lanes of one worker, bitwise against the one-pass ordered reduce
/// of the whole tensor on the host (`docs/programming_model.md`, "Plan once,
/// run many"). Each iteration's upload, on lane 1, stages its batch in its
/// copy of the body's one row, `staged`; the reduce, on lane 0, adds it
/// into the accumulator after a delay. Two copies (depth 2), eight trips.
///
/// The bar is exact, by analysis: each element's sum is the batches added
/// in order, each addition rounded once, on the card as on the host.
/// The delay keeps the reduce lane behind the copy lane, so the ring is
/// load-bearing: without it an upload two iterations on overwrites a batch
/// its reduce has not yet read, and every element's bits change (the ring
/// dropped from the executor, this case fails on all of them). The read's
/// wait between the lanes is `two_lanes_share_a_row_with_no_host_wait`'s.
/// The loop adds no host wait: the record's sync points are the
/// check's on the download, after the loop, and each lane's exit. Each
/// body task has an entry per trip with its device time. The plan runs
/// twice, the arena's count of this process's blocks the same across the
/// second run: a counted loop allocates nothing.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn batches_reduce_bitwise() {
    let _turn = TURN.lock().unwrap_or_else(|p| p.into_inner());
    let lane = |lane| LaneId { worker: 0, lane };
    let got = Arc::new(Mutex::new(Vec::new()));
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let wait = dag.device_worker_row(0, "delay", 2 * DELAY);
    let (batches, one) = dag.add_cpu("tensor", 0, Tensor);
    let sum = dag.add_gpu("zero", lane(0), Zero(one));
    let body = dag.len();
    let staged = dag.add_gpu("stage", lane(1), StageBatch(batches));
    dag.add_gpu(
        "reduce",
        lane(0),
        Reduce {
            delay: wait,
            staged,
            sum,
        },
    );
    dag.add_counted(body, TRIPS, 2);
    let down = dag.add_gpu("download", lane(0), DownSum(sum.acc));
    dag.add_cpu("check", 0, Keep(down, Arc::clone(&got)));
    let planned = dag.plan();
    let text = planned.to_string();
    assert!(
        text.contains("reduce: after 1 2; closes counted loop 2-3, 8 trips, depth 2"),
        "{text}"
    );
    assert!(text.contains("2 copies 8388608 bytes apart"), "{text}");

    let mut expect = vec![0.0f64; BATCH];
    for k in 0..TRIPS {
        for (i, e) in expect.iter_mut().enumerate() {
            *e += batch_value(k, i);
        }
    }
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    for run in 0..2 {
        let blocks = fb_core_arena::allocations();
        let record = instance.run(&mut backend, 1);
        record.result.as_ref().unwrap();
        assert_eq!(fb_core_arena::allocations(), blocks, "run {run} allocated");
        let sum = std::mem::take(&mut *got.lock().unwrap());
        assert_eq!(sum.len(), BATCH);
        let wrong = (0..BATCH)
            .filter(|&i| sum[i].to_bits() != expect[i].to_bits())
            .count();
        assert_eq!(
            wrong, 0,
            "run {run}: {wrong} elements differ from the ordered reduce"
        );

        for name in ["stage", "reduce"] {
            let trips: Vec<_> = record
                .tasks
                .iter()
                .filter(|t| t.name == name)
                .map(|t| (t.iteration, t.device.is_some()))
                .collect();
            let each: Vec<_> = (0..TRIPS).map(|k| (Some(k), true)).collect();
            assert_eq!(trips, each, "run {run}: {name}");
        }
        let syncs: Vec<_> = record
            .syncs
            .iter()
            .map(|s| {
                (
                    s.node.map(NodeId::index),
                    s.worker.as_str(),
                    s.waited_for.index(),
                )
            })
            .collect();
        // Ranks: the download 4, the check 5, the last upload 2.
        assert_eq!(
            syncs,
            [
                (Some(5), "cpu0", 4),
                (None, "gpu0.1", 2),
                (None, "gpu0.0", 4)
            ],
            "run {run}: no host wait in the loop"
        );
    }
}
