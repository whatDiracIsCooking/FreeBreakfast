//! A heap row to the device and back through pinned windows on a real
//! card (device 0): `Dag::upload_heap` and `Dag::download_heap`, each a
//! CPU node and a stream node, in a counted loop.
//!
//! Its own binary, so the arena's count of this process's blocks it reads
//! sees no other case's.

#![cfg(feature = "device")]

use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Heap, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag};
use fb_core_thread_pool::{LaneId, NodeId};

/// The elements of one range, and the ranges the row is cut into: 4 MiB
/// a range, 32 MiB in all.
const PIECE: usize = 1 << 19;
const TRIPS: usize = 8;

/// Element `i` of the source row: distinct in most bits, so a range
/// copied from or to the wrong place, or a window a trip did not fill,
/// changes the row's bits.
fn value(i: usize) -> f64 {
    (i as f64 * 0.37).sin() * 1e3 + 1.0 / (i + 1) as f64
}

/// CPU: the source heap row, filled, and the heap row the ranges come
/// back into, left as the block's fill.
struct Source;

impl CpuTask for Source {
    type Rows = (RowId<Heap, f64>, RowId<Heap, f64>);

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        let n = TRIPS * PIECE;
        (heap.output("b", n), heap.output("b back", n))
    }

    fn run(&self, rows: &Self::Rows, _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let src = heap.take_mut(rows.0);
        src.iter_mut().enumerate().for_each(|(i, x)| *x = value(i));
        Ok(())
    }
}

/// CPU: the elements of a heap row that differ in their bits from the
/// source's, counted for the test.
struct Check(RowId<Heap, f64>, Arc<Mutex<Option<usize>>>);

impl CpuTask for Check {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.0);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        let wrong = heap
            .get(self.0)
            .iter()
            .enumerate()
            .filter(|&(i, x)| x.to_bits() != value(i).to_bits())
            .count();
        *self.1.lock().unwrap() = Some(wrong);
        Ok(())
    }
}

/// A heap row cut into eight ranges goes up and comes back, range `k` in
/// trip `k` of a counted loop of depth 2: each trip's `stage b` copies its
/// range into its copy of the pinned window, `upload b` on lane 1 copies
/// the window up, `download b back` on lane 0 copies the device row into
/// the second window, and `unstage b back` copies that into range `k` of
/// the second heap row, which after the loop is bitwise the first.
///
/// The bar is exact, by analysis: every step is a copy. Two copies of the
/// windows and the device row (depth 2) are in flight at once, so a trip
/// that reached another's copy, or a range taken at the wrong offset,
/// changes the bits of a whole range.
///
/// The GPU halves add no sync point of their own: the record's waits are
/// the CPU halves' own, `unstage` on its trip's download, and then each
/// lane's exit. `stage` reuses the copy of the window the upload two trips
/// back read, and waits for it on the host through nothing of its own:
/// it runs after that trip's `unstage`, which waited for its download,
/// which ran after the upload on the device. Each half has an entry per trip. The
/// plan runs twice, the arena's count of this process's blocks the same
/// across the second run: the operations allocate nothing.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn heap_ranges_round_trip_bitwise() {
    let wrong = Arc::new(Mutex::new(None));
    let mut dag = Dag::new();
    let (src, dst) = dag.add_cpu("source", 0, Source);
    let body = dag.len();
    let d = dag.upload_heap(LaneId { worker: 0, lane: 1 }, src, 0..PIECE, "b");
    dag.download_heap(0, d, dst, 0..PIECE, "b back");
    dag.add_counted(body, TRIPS, 2);
    dag.add_cpu("check", 0, Check(dst, Arc::clone(&wrong)));
    let planned = dag.plan();
    let text = planned.to_string();
    assert!(
        text.contains("unstage b back: after 0 3; closes counted loop 1-4, 8 trips, depth 2"),
        "{text}"
    );

    let mut instance = planned.instantiate(&[0]);
    let mut backend = instance.backend();
    for run in 0..2 {
        let blocks = fb_core_arena::allocations();
        let record = instance.run(&mut backend, 2);
        record.result.as_ref().unwrap();
        assert_eq!(fb_core_arena::allocations(), blocks, "run {run} allocated");
        assert_eq!(
            wrong.lock().unwrap().take(),
            Some(0),
            "run {run}: elements differ from the source"
        );

        for name in ["stage b", "upload b", "download b back", "unstage b back"] {
            let trips: Vec<_> = record
                .tasks
                .iter()
                .filter(|t| t.name == name)
                .map(|t| t.iteration)
                .collect();
            let each: Vec<_> = (0..TRIPS).map(Some).collect();
            assert_eq!(trips, each, "run {run}: {name}");
        }
        let syncs: Vec<_> = record
            .syncs
            .iter()
            .map(|s| {
                (
                    s.node.map(NodeId::index),
                    s.iteration,
                    s.worker.as_str(),
                    s.waited_for.index(),
                )
            })
            .collect();
        // Ranks: stage 1, upload 2, download 3, unstage 4.
        let want: Vec<_> = (0..TRIPS).map(|k| (Some(4), Some(k), 3)).collect();
        let got: Vec<_> = syncs
            .iter()
            .filter(|s| s.0.is_some())
            .map(|&(n, i, _, u)| (n, i, u))
            .collect();
        assert_eq!(got, want, "run {run}: the CPU halves' waits alone");
        let exits: Vec<_> = syncs
            .iter()
            .filter(|s| s.0.is_none())
            .map(|&(_, _, w, u)| (w, u))
            .collect();
        assert_eq!(exits, [("gpu0.1", 2), ("gpu0.0", 3)], "run {run}");
    }
}
