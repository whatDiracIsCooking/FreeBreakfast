//! The planner on a real card (device 0): tasks declared once, planned,
//! instantiated and run on a stream worker and the CPU pool.
//!
//! A binary of its own: a case measures the card's free memory, which
//! another test allocating beside it would move, so the cases take turns.

use std::ffi::c_void;
use std::sync::Mutex;
use std::time::Duration;

use fb_core_arena::{Bufs, DEVICE_ALIGNMENT, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{self as thread_pool, DeviceStream, Lane, Placement};
use fb_core_wwr::common::ScopedDeviceIndex;
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use wwr_sys::{wwr_memcpy_kind, wwr_status};

const N: usize = 4096;
const BYTES: usize = N * size_of::<f64>();

/// Held by each case for its whole length.
static TURN: Mutex<()> = Mutex::new(());

/// Device 0's free memory, with it made current.
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
        // SAFETY: N f64s between two rows of N, on the task's stream. The
        // source is not rewritten before this copy completes: its bytes'
        // next owner is ordered after this task's event by a handoff.
        gpu_status(unsafe {
            wwr_sys::wwr_memcpy_async(
                dst.as_mut_ptr().cast(),
                h.get(self.0).as_ptr().cast::<c_void>(),
                BYTES,
                wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
                stream.stream(),
            )
        })
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
        let dst = h.take_mut(*down);
        // SAFETY: as in `Upload`, device to pinned.
        gpu_status(unsafe {
            wwr_sys::wwr_memcpy_async(
                dst.as_mut_ptr().cast(),
                d.get(self.0).as_ptr().cast::<c_void>(),
                BYTES,
                wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
                stream.stream(),
            )
        })
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
/// card's free memory unchanged between: a run allocates nothing.
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

    let instance = planned.instantiate(&[0]);
    let backend = thread_pool::Device::new(&[0], instance.plan());
    let record = instance.run(&backend, 1);
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
    instance.run(&backend, 1).result.unwrap();
    assert_eq!(free_bytes(), between, "a run allocated device memory");
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
        let dst = d.take_mut(*out);
        let bytes = BIG * size_of::<f64>();
        // SAFETY: `bytes` within one row, or between two rows of BIG, on the
        // task's stream.
        gpu_status(unsafe {
            match self.0 {
                None => {
                    wwr_sys::wwr_memset_async(dst.as_mut_ptr().cast(), 0, bytes, stream.stream())
                }
                Some(input) => wwr_sys::wwr_memcpy_async(
                    dst.as_mut_ptr().cast(),
                    d.get(input).as_ptr().cast::<c_void>(),
                    bytes,
                    wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE,
                    stream.stream(),
                ),
            }
        })
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
    let instance = dag.plan().instantiate(&[0]);
    let backend = thread_pool::Device::new(&[0], instance.plan());
    let record = instance.run(&backend, 1);
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
