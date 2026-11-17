//! A heap row whose length is no multiple of its ranges' streamed to a
//! real card (device 0) and back in `total.div_ceil(len)` trips, the last
//! range clamped to the row's end by `Dag::upload_heap` and
//! `Dag::download_heap` (`docs/memory.md`, "Adaptive widths: one planner
//! routine").
//!
//! Its own binary, beside `heap_copies.rs`, whose case reads the arena's
//! count of this process's blocks.

#![cfg(feature = "device-tests")]

use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_p2d};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{CpuTask, Dag, Decision, GpuTask};
use fb_core_thread_pool::{DeviceStream, LaneId};
use fb_core_wwr::blas::gemv;
use fb_core_wwr::device_slice::split_at;
use fb_core_wwr::memset;
use wwr_sys::wwr_blas_operation::WWR_BLAS_OP_T as OP_T;

/// A range's elements, and the row's: four full ranges and a ragged last
/// one of 337, five trips.
const PIECE: usize = 1000;
const TOTAL: usize = 4 * PIECE + 337;

/// Element `i` of the source row: distinct, so a range copied from or to
/// the wrong place changes the row's bits, and a multiple of a half below
/// 2¹², so every partial sum of the row, in any order, is exact.
fn value(i: usize) -> f64 {
    i as f64 * 0.5 - 1000.0
}

/// CPU: the source heap row, filled; the heap row the ranges come back
/// into, left as the block's fill; and a pinned row of ones, the sum's
/// vector.
struct Source;

impl CpuTask for Source {
    type Rows = (RowId<Heap, f64>, RowId<Heap, f64>, RowId<Pinned, f64>);

    fn layout(&self, pinned: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        (
            heap.output("b", TOTAL),
            heap.output("b back", TOTAL),
            pinned.output("ones", PIECE),
        )
    }

    fn run(&self, rows: &Self::Rows, pinned: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let src = heap.take_mut(rows.0);
        src.iter_mut().enumerate().for_each(|(i, x)| *x = value(i));
        pinned.take_mut(rows.2).fill(1.0);
        Ok(())
    }
}

/// The sum's device rows: the accumulator and the ones.
#[derive(Clone, Copy)]
struct Sum {
    acc: RowId<Device, f64>,
    ones: RowId<Device, f64>,
}

/// GPU: the accumulator zeroed and the ones uploaded.
struct Zero(RowId<Pinned, f64>);

impl GpuTask for Zero {
    type Rows = Sum;

    fn layout(&self, d: &mut Layout<'_, Device>, h: &mut Layout<'_, Pinned>) -> Sum {
        h.read(self.0);
        Sum {
            acc: d.output("acc", 1),
            ones: d.output("ones", PIECE),
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
        memcpy_p2d(&mut d.take_mut(r.ones), h, self.0, stream)
    }
}

/// GPU: the trip's elements of the uploaded batch added into the
/// accumulator, `acc += batchᵀ ones` by one gemv over the head of the
/// device row the trip holds, `PIECE.min(TOTAL - k PIECE)` elements in
/// trip `k`. The device row of the ragged last trip keeps, past that
/// head, the batch of the trip two back (depth 2), which a reader that
/// took the whole row would add.
struct Add {
    batch: RowId<Device, f64>,
    sum: Sum,
}

impl GpuTask for Add {
    type Rows = ();

    fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        d.read(self.batch);
        d.read(self.sum.ones);
        d.write(self.sum.acc);
    }

    fn run(
        &self,
        _: &(),
        d: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let count = PIECE.min(TOTAL - d.iteration() * PIECE);
        let batch = d.get(self.batch);
        let (head, _) = split_at(&batch, count);
        let m = count as i32;
        gemv(
            stream.blas(),
            OP_T,
            m,
            1,
            1.0,
            &head,
            m,
            &d.get(self.sum.ones),
            1,
            1.0,
            &mut d.take_mut(self.sum.acc),
            1,
        )
    }
}

/// CPU: the sum's bits and the count of elements of the second heap row
/// that differ in their bits from the source's, kept for the test.
struct Check {
    sum: RowId<Pinned, f64>,
    back: RowId<Heap, f64>,
    out: Arc<Mutex<Option<(u64, usize)>>>,
}

impl CpuTask for Check {
    type Rows = ();

    fn layout(&self, pinned: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        pinned.read(self.sum);
        heap.read(self.back);
    }

    fn run(&self, _: &(), pinned: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let wrong = heap
            .get(self.back)
            .iter()
            .enumerate()
            .filter(|&(i, x)| x.to_bits() != value(i).to_bits())
            .count();
        *self.out.lock().unwrap() = Some((pinned.get(self.sum)[0].to_bits(), wrong));
        Ok(())
    }
}

/// The decision of a loop of a fixed count of iterations.
struct Times(usize);

impl Decision for Times {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {}

    fn decide(
        &self,
        _: &(),
        _: &Bufs<'_, Pinned>,
        _: &Bufs<'_, Heap>,
        iteration: usize,
    ) -> Result<ControlFlow<()>> {
        Ok(if iteration + 1 == self.0 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        })
    }
}

/// The row streamed once, at the top level (`iterations` 0) or as the
/// body of a decision loop of `iterations` iterations, the zero with it:
/// a counted loop of `TOTAL.div_ceil(PIECE)` trips, depth 2, of
/// `stage b` and `upload b` (lane 1), `add` (lane 0) and `download b
/// back` and `unstage b back` (lane 0), then the sum down and the check.
/// Returns the sum's bits and the second row's wrong elements.
fn run(iterations: usize) -> (u64, usize) {
    let out = Arc::new(Mutex::new(None));
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let (src, back, ones) = dag.add_cpu("source", 0, Source);
    let outer = dag.len();
    let sum = dag.add_gpu("zero", 0, Zero(ones));
    let body = dag.len();
    let batch = dag.upload_heap(LaneId { worker: 0, lane: 1 }, src, 0..PIECE, "b");
    dag.add_gpu("add", 0, Add { batch, sum });
    dag.download_heap(0, batch, back, 0..PIECE, "b back");
    dag.add_counted(body, TOTAL.div_ceil(PIECE), 2);
    if iterations > 0 {
        dag.add_loop("again", 0, outer, iterations, Times(iterations));
    }
    let down = dag.download(0, sum.acc, "sum", 1);
    dag.add_cpu(
        "check",
        0,
        Check {
            sum: down,
            back,
            out: Arc::clone(&out),
        },
    );
    let planned = dag.plan();
    let text = planned.to_string();
    assert!(text.contains("5 trips, depth 2"), "{text}");
    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 2);
    record.result.as_ref().unwrap();
    let trips = record.tasks.iter().filter(|t| t.name == "add").count();
    assert_eq!(trips, 5 * iterations.max(1), "the add's trips");
    out.lock().unwrap().take().expect("the check ran")
}

/// A row of 4337 elements in ranges of 1000 goes up in five trips of a
/// counted loop of depth 2, the last clamped to 337, and is summed on the
/// device and comes back: the sum is the whole row's, and the second heap
/// row is bitwise the first. In a decision loop of three iterations,
/// zeroed each, the sum and the row are bitwise the top level's.
///
/// The bar is exact, by analysis: every element is a multiple of a half
/// below 2¹², so every partial sum is exact in either order, and every
/// other step is a copy. A last trip that copied past the row's end
/// panics; one that copied too little leaves the block's fill in the
/// second row; an add that read the whole device row in the last trip
/// adds the stale 663 elements of trip 2's batch; a range or a count
/// taken from the decision loop's iteration rather than the trip moves the
/// looped sum off the top level's. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_ragged_last_trip_is_clamped() {
    let whole: f64 = (0..TOTAL).map(value).sum();
    let top = run(0);
    assert_eq!(f64::from_bits(top.0), whole, "the sum of the row");
    assert_eq!(
        top.1, 0,
        "elements of the second row differ from the source"
    );
    let looped = run(3);
    assert_eq!(
        looped, top,
        "the decision loop's sum and row, of the top level's"
    );
}
