//! The planner on a real card (device 0): tasks declared once, planned,
//! instantiated and run on a stream worker and the CPU pool.
//!
//! A binary of its own: its case measures the card's free memory, which
//! another test allocating beside it would move.

use std::ffi::c_void;

use fb_rebuild::core::arena::{Bufs, DEVICE_ALIGNMENT, Device, Layout, Pinned, RowId};
use fb_rebuild::core::error_handling::Status;
use fb_rebuild::core::planner::{Dag, Task};
use fb_rebuild::core::thread_pool::{self, DeviceStream, Lane};
use fb_rebuild::core::wwr::common::ScopedDeviceIndex;
use fb_rebuild::core::wwr::gpu_status;
use fb_rebuild::core::wwr::handle::DeviceHandleStream;
use wwr_sys::{wwr_memcpy_kind, wwr_status};

const N: usize = 4096;
const BYTES: usize = N * size_of::<f64>();

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
#[test]
fn a_round_trip_through_the_planner() {
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

    let instance = planned.instantiate(&[0]);
    let backend = thread_pool::Device::new(&[0], instance.plan());
    instance.run(&backend, 1).unwrap();
    let between = free_bytes();
    instance.run(&backend, 1).unwrap();
    assert_eq!(free_bytes(), between, "a run allocated device memory");
}
