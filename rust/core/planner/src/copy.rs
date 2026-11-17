//! Copy nodes: a copy between kinds that is a node's whole work, added by
//! the planner in one line (`docs/memory.md`, "The table"). A task never
//! writes one by hand; the planner's three are the only `GpuTask`s whose
//! `run` is a copy alone, and the two halves of a heap row's copy through
//! its pinned window the only `CpuTask`s (`doc_check/tests/code_rules.rs`).

use std::ops::Range;

use contracts::debug_requires;
use fb_core_arena::{
    Bufs, CopyOnly, Device, Element, Heap, Layout, Pinned, RowId, StagingKind, memcpy_d2p,
    memcpy_p2d,
};
use fb_core_error_handling::Status;
use fb_core_thread_pool::{DeviceStream, LaneId};

use super::dag::Dag;
use super::task::{CpuTask, GpuTask};

impl<'a> Dag<'a, Pinned, DeviceStream> {
    /// Add the upload of the pinned row `src`, `len` elements, to a new
    /// device output row `name` of the same length, which it returns: one
    /// node on lane `at` of a GPU worker (a bare worker index is its lane
    /// 0, [`LaneId`]), `upload <name>`, timed, admitted and ordered like
    /// any other, with the dimension `len`. A copy may have a lane of its
    /// own, so it overlaps the worker's compute.
    ///
    /// # Panics
    /// If `src` is a row of another worker's table ([`Dag::add_gpu`]); and
    /// when the plan runs, if `src` is not `len` long (`memcpy_p2d`).
    pub fn upload<T: Element>(
        &mut self,
        at: impl Into<LaneId>,
        src: RowId<Pinned, T>,
        name: &str,
        len: usize,
    ) -> RowId<Device, T> {
        self.add_gpu(
            format!("upload {name}"),
            at,
            Upload {
                src,
                name: name.to_owned(),
                len,
            },
        )
    }

    /// Add the download of the device row `src`, `len` elements, to a new
    /// pinned output row `name` of the same length, which it returns: one
    /// node on lane `at`, `download <name>`, as [`Dag::upload`]. A CPU task that reads the row runs after the
    /// executor waited for the copy.
    ///
    /// # Panics
    /// If `src` is a row of another worker's table ([`Dag::add_gpu`]); and
    /// when the plan runs, if `src` is not `len` long (`memcpy_d2p`).
    pub fn download<T: Element>(
        &mut self,
        at: impl Into<LaneId>,
        src: RowId<Device, T>,
        name: &str,
        len: usize,
    ) -> RowId<Pinned, T> {
        self.add_gpu(
            format!("download {name}"),
            at,
            Download {
                src,
                name: name.to_owned(),
                len,
            },
        )
    }

    /// Add the download of the device rows `rows`, each `(src, name, len)`
    /// as [`Dag::download`] takes its three, to new pinned output rows of
    /// those names and lengths, which it returns in their order: one node
    /// on lane `at`, `download <name>`, with the dimensions `rows` (their
    /// count) and `len` (their elements together). Its copies share one
    /// stream scope and one event, so a reader of every row waits once,
    /// where a [`Dag::download`] per row is a node, an event and a wait
    /// each. Rows of two element types are two of these nodes.
    ///
    /// # Panics
    /// If a source is a row of another worker's table ([`Dag::add_gpu`]);
    /// and when the plan runs, if a source is not its `len` long
    /// (`memcpy_d2p`).
    pub fn download_rows<T: Element>(
        &mut self,
        at: impl Into<LaneId>,
        name: &str,
        rows: impl IntoIterator<Item = (RowId<Device, T>, String, usize)>,
    ) -> Vec<RowId<Pinned, T>> {
        self.add_gpu(
            format!("download {name}"),
            at,
            DownloadRows {
                rows: rows.into_iter().collect(),
            },
        )
    }

    /// Add the upload of `range` of the heap row `src` to a new device
    /// output row `name` of the range's length, which it returns, through
    /// a pinned window of that length: two nodes, `stage <name>` on the
    /// CPU pool, the range copied into the window `<name> window`, a
    /// pinned output row of worker `at`'s, and `upload <name>` on lane
    /// `at`, the window copied up as [`Dag::upload`] copies. Two nodes,
    /// since the device copy is enqueued only once the host copy has
    /// finished, which inside one task would be a host sync
    /// (`docs/programming_model.md`, "Granularity"); as two, one batch's
    /// host copy runs under another's device work. A heap row reaches the
    /// device by this alone (`docs/memory.md`, "The table").
    ///
    /// In a loop, iteration `k` copies the range moved on by `k` of its
    /// lengths, `start + k len .. end + k len`, from its views' iteration
    /// ([`Bufs::iteration`]); outside every loop, the range itself. So a
    /// row cut into `n` ranges of one length goes up in a counted loop of
    /// `n` trips as one call over the first range, range `k` in trip `k`,
    /// the window and the device row the body's own and held once per
    /// iteration in flight; or, outside a loop, as `n` calls, one per
    /// range.
    ///
    /// A range that runs past the end of `src` is clamped to it: the
    /// iteration copies `min(len, src.len() - start_k)` elements, `start_k`
    /// its range's start, into the head of the window and the device row,
    /// which keep the full length, the rest of them left as an earlier
    /// trip or the block left it. So a row of `total` elements cut into
    /// ranges of `len` goes up in `total.div_ceil(len)` trips, the last
    /// one ragged (`docs/memory.md`, "Adaptive widths: one planner
    /// routine"), and the task that reads the device row reads that count
    /// alone, `len.min(total - k len)` in trip `k`, from its own views'
    /// iteration.
    ///
    /// # Panics
    /// When the plan runs, if an iteration's range starts past the end of
    /// `src`.
    #[debug_requires(range.start <= range.end, "a range runs forward")]
    pub fn upload_heap<T: Element>(
        &mut self,
        at: impl Into<LaneId>,
        src: RowId<Heap, T>,
        range: Range<usize>,
        name: &str,
    ) -> RowId<Device, T> {
        let at = at.into();
        let len = range.len();
        let window = self.add_cpu(
            format!("stage {name}"),
            at.worker,
            Stage {
                src,
                start: range.start,
                len,
                window: format!("{name} window"),
            },
        );
        self.add_gpu(
            format!("upload {name}"),
            at,
            Upload {
                src: window,
                name: name.to_owned(),
                len,
            },
        )
    }

    /// Add the download of the device row `src` into `range` of the heap
    /// row `dst`, through a pinned window of the range's length: two
    /// nodes, `download <name>` on lane `at`, `src` copied into the window
    /// `<name> window`, a pinned output row of worker `at`'s, as
    /// [`Dag::download`] copies, and `unstage <name>` on the CPU pool, the
    /// window copied into the range once the executor has waited for the
    /// download's event. Two nodes for the reason [`Dag::upload_heap`]
    /// gives.
    ///
    /// `dst` is written, not made: a row produced before the loop this
    /// call is in, which it writes as the loop's carried row
    /// ([`Layout::write`]), with its table's contracts; in a counted loop
    /// its one user in the body, whose writes from the trips in flight run
    /// in trip order. In a loop, iteration `k` writes the range moved on
    /// by `k` of its lengths, as [`Dag::upload_heap`] reads: a row cut into
    /// `n` ranges comes down in a counted loop of `n` trips, range `k` in
    /// trip `k`. A task after the loop that reads `dst` runs after the
    /// last trip's write. A range that runs past the end of `dst` is
    /// clamped to it, as [`Dag::upload_heap`] clamps: the whole of `src`
    /// comes down into the window, and the head of the window that the
    /// clamped range holds goes into it, so a row of `total` elements
    /// comes down in `total.div_ceil(len)` trips.
    ///
    /// # Panics
    /// When the plan runs, if `src` is not the range's length long
    /// (`memcpy_d2p`), or if an iteration's range starts past the end of
    /// `dst`.
    #[debug_requires(range.start <= range.end, "a range runs forward")]
    pub fn download_heap<T: Element>(
        &mut self,
        at: impl Into<LaneId>,
        src: RowId<Device, T>,
        dst: RowId<Heap, T>,
        range: Range<usize>,
        name: &str,
    ) {
        let at = at.into();
        let len = range.len();
        let window = self.add_gpu(
            format!("download {name}"),
            at,
            Download {
                src,
                name: format!("{name} window"),
                len,
            },
        );
        self.add_cpu(
            format!("unstage {name}"),
            at.worker,
            Unstage {
                window,
                dst,
                start: range.start,
                len,
            },
        );
    }
}

/// CPU: a range of a heap row copied into a pinned window, the host half
/// of [`Dag::upload_heap`].
struct Stage<T> {
    src: RowId<Heap, T>,
    start: usize,
    len: usize,
    window: String,
}

impl<T: Element, P: StagingKind> CpuTask<P> for Stage<T> {
    type Rows = RowId<P, T>;

    fn layout(&self, pinned: &mut Layout<'_, P>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        heap.read(self.src);
        pinned.output(&self.window, self.len)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len)]
    }

    fn run(&self, window: &Self::Rows, pinned: &Bufs<'_, P>, heap: &Bufs<'_, Heap>) -> Status {
        // This iteration's range: the first moved on by its lengths, its
        // end clamped to the row's.
        let src = heap.get(self.src);
        let from = self.start + heap.iteration() * self.len;
        let to = src.len().min(from + self.len);
        let range = &src[from..to];
        pinned.take_mut(*window)[..range.len()].copy_from_slice(range);
        Ok(())
    }
}

/// CPU: a pinned window copied into a range of a heap row, the host half
/// of [`Dag::download_heap`].
struct Unstage<T, P: StagingKind> {
    window: RowId<P, T>,
    dst: RowId<Heap, T>,
    start: usize,
    len: usize,
}

impl<T: Element, P: StagingKind> CpuTask<P> for Unstage<T, P> {
    type Rows = ();

    fn layout(&self, pinned: &mut Layout<'_, P>, heap: &mut Layout<'_, Heap>) {
        pinned.read(self.window);
        heap.write(self.dst);
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len)]
    }

    fn run(&self, _: &(), pinned: &Bufs<'_, P>, heap: &Bufs<'_, Heap>) -> Status {
        let dst = heap.take_mut(self.dst);
        let from = self.start + heap.iteration() * self.len;
        let to = dst.len().min(from + self.len);
        let range = &mut dst[from..to];
        range.copy_from_slice(&pinned.get(self.window)[..range.len()]);
        Ok(())
    }
}

/// GPU: a pinned row copied into a device row on the worker's stream.
struct Upload<T> {
    src: RowId<Pinned, T>,
    name: String,
    len: usize,
}

impl<T: Element> GpuTask for Upload<T> {
    type Rows = RowId<Device, T>;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        pinned: &mut Layout<'_, Pinned>,
    ) -> Self::Rows {
        pinned.read(self.src);
        device.output(&self.name, self.len)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len)]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        dst: &Self::Rows,
        device: &Bufs<'_, Device>,
        pinned: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_p2d(&mut device.take_mut(*dst), pinned, self.src, stream)
    }
}

/// GPU: a device row copied into a pinned row on the worker's stream.
struct Download<T> {
    src: RowId<Device, T>,
    name: String,
    len: usize,
}

impl<T: Element> GpuTask for Download<T> {
    type Rows = RowId<Pinned, T>;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        pinned: &mut Layout<'_, Pinned>,
    ) -> Self::Rows {
        device.read(self.src);
        pinned.output(&self.name, self.len)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len)]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        dst: &Self::Rows,
        device: &Bufs<'_, Device>,
        pinned: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(pinned, *dst, device.get(self.src), stream)
    }
}

/// GPU: device rows copied into pinned rows on the worker's stream, one
/// node for them all.
struct DownloadRows<T> {
    rows: Vec<(RowId<Device, T>, String, usize)>,
}

impl<T: Element> GpuTask for DownloadRows<T> {
    type Rows = Vec<RowId<Pinned, T>>;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        pinned: &mut Layout<'_, Pinned>,
    ) -> Self::Rows {
        self.rows
            .iter()
            .map(|&(src, ref name, len)| {
                device.read(src);
                pinned.output(name, len)
            })
            .collect()
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("rows", self.rows.len()),
            ("len", self.rows.iter().map(|r| r.2).sum()),
        ]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        dst: &Self::Rows,
        device: &Bufs<'_, Device>,
        pinned: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        for (&(src, _, _), &dst) in self.rows.iter().zip(dst) {
            memcpy_d2p(pinned, dst, device.get(src), stream)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use fb_core_arena::PinnedStandIn;
    use fb_core_thread_pool::HostOnly;

    /// CPU: a pinned output of `len` doubles.
    struct Stage(usize);

    impl CpuTask for Stage {
        type Rows = RowId<Pinned, f64>;
        fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
            h.output("x", self.0)
        }
        fn run(&self, _: &Self::Rows, _: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
            Ok(())
        }
    }

    /// CPU: reads a pinned row.
    struct Read(RowId<Pinned, f64>);

    impl CpuTask for Read {
        type Rows = ();
        fn layout(&self, h: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
            h.read(self.0);
        }
        fn run(&self, _: &(), _: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
            Ok(())
        }
    }

    /// Each copy is one stream node named after its row, the new row of
    /// that name and length, after the producer of its source, and before
    /// its reader.
    #[test]
    fn a_copy_is_one_node_named_after_its_row() {
        let mut dag = Dag::new();
        let x = dag.add_cpu("stage", 0, Stage(8));
        let d = dag.upload(0, x, "x", 8);
        let back = dag.download(0, d, "x back", 8);
        dag.add_cpu("read", 0, Read(back));
        let planned = dag.plan();
        assert_eq!(planned.predecessors(1), &[0]);
        // The download's row may take the staged row's bytes, dead once
        // the upload has read them: a handoff after the stage too.
        assert!(planned.predecessors(2).contains(&1));
        assert!(planned.predecessors(3).contains(&2));
        assert_eq!(planned.device_table(0).len(d), 8);
        assert_eq!(planned.host_table(0).len(back), 8);
        let text = planned.to_string();
        assert!(
            text.contains("    1       0  stream  upload x: after 0\n"),
            "{text}"
        );
        assert!(
            text.contains("    2       0  stream  download x back: after 0 1\n"),
            "{text}"
        );
        assert!(text.contains("  edge    1-2      x: 1w 2r\n"), "{text}");
        assert!(text.contains("x back: 2w 3r\n"), "{text}");
    }

    /// A copy goes on the lane it is given, so it can overlap the worker's
    /// compute on another, and its reader on lane 0 runs after it.
    #[test]
    fn a_copy_goes_on_its_lane() {
        let mut dag = Dag::new();
        let x = dag.add_cpu("stage", 0, Stage(8));
        let d = dag.upload(LaneId { worker: 0, lane: 1 }, x, "x", 8);
        let back = dag.download(0, d, "x back", 8);
        dag.add_cpu("read", 0, Read(back));
        let planned = dag.plan();
        assert!(planned.predecessors(2).contains(&1));
        let text = planned.to_string();
        assert!(text.contains("  stream1  upload x: after 0\n"), "{text}");
        assert!(
            text.contains("  stream0  download x back: after "),
            "{text}"
        );
    }

    /// A download of several rows is one stream node named after its
    /// label, a pinned row of each name and length in their order, after
    /// the producers of its sources, and before a reader of any of them.
    #[test]
    fn a_download_of_rows_is_one_node() {
        let mut dag = Dag::new();
        let x = dag.add_cpu("stage", 0, Stage(8));
        let y = dag.add_cpu("stage", 0, Stage(3));
        let dx = dag.upload(0, x, "x", 8);
        let dy = dag.upload(0, y, "y", 3);
        let back = dag.download_rows(
            0,
            "both",
            [(dx, "x back".into(), 8), (dy, "y back".into(), 3)],
        );
        dag.add_cpu("read", 0, Read(back[1]));
        let planned = dag.plan();
        assert!(planned.predecessors(4).contains(&2));
        assert!(planned.predecessors(4).contains(&3));
        assert!(planned.predecessors(5).contains(&4));
        assert_eq!(back.len(), 2);
        assert_eq!(planned.host_table(0).len(back[0]), 8);
        assert_eq!(planned.host_table(0).len(back[1]), 3);
        let text = planned.to_string();
        assert!(
            text.contains("    4       0  stream  download both: after 0 1 2 3\n"),
            "{text}"
        );
        assert!(text.contains("x back: 4w\n"), "{text}");
        assert!(text.contains("y back: 4w 5r\n"), "{text}");
    }

    /// The ranges a heap row is cut into, and each one's length.
    const RANGES: usize = 4;
    const PIECE: usize = 5;

    /// Element `i` of the source row: distinct in most bits, so a range
    /// copied from or to the wrong place changes the row's bits.
    fn value(i: usize) -> f64 {
        (i as f64 * 0.37).sin() * 1e3 + 1.0 / (i + 1) as f64
    }

    /// CPU: the source heap row of `.0` elements, filled, and the
    /// destination, the heap row the ranges come back into, left as the
    /// block's fill.
    struct Source(usize);

    impl<P: StagingKind> CpuTask<P> for Source {
        type Rows = (RowId<Heap, f64>, RowId<Heap, f64>);
        fn layout(&self, _: &mut Layout<'_, P>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
            (heap.output("b", self.0), heap.output("b back", self.0))
        }
        fn run(&self, rows: &Self::Rows, _: &Bufs<'_, P>, heap: &Bufs<'_, Heap>) -> Status {
            let src = heap.take_mut(rows.0);
            src.iter_mut().enumerate().for_each(|(i, x)| *x = value(i));
            Ok(())
        }
    }

    /// CPU: a heap row's bits, kept for the test.
    struct Keep(RowId<Heap, f64>, Arc<Mutex<Vec<u64>>>);

    impl<P: StagingKind> CpuTask<P> for Keep {
        type Rows = ();
        fn layout(&self, _: &mut Layout<'_, P>, heap: &mut Layout<'_, Heap>) {
            heap.read(self.0);
        }
        fn run(&self, _: &(), _: &Bufs<'_, P>, heap: &Bufs<'_, Heap>) -> Status {
            *self.1.lock().unwrap() = heap.get(self.0).iter().map(|x| x.to_bits()).collect();
            Ok(())
        }
    }

    /// A heap row cut into ranges goes up and comes back in a counted
    /// loop of a trip per range, depth 2: each operation is a CPU node
    /// and a stream node named after its row, the CPU half of the upload
    /// before its stream half and the CPU half of the download after its
    /// stream half; the windows and the device row are the body's, held
    /// twice; the destination is written by its one user in the body and
    /// read after the loop. Outside a loop, a range per call.
    #[test]
    fn a_heap_copy_is_two_nodes_named_after_its_row() {
        let mut dag = Dag::new();
        let (src, dst) = dag.add_cpu("source", 0, Source(RANGES * PIECE));
        let body = dag.len();
        let d = dag.upload_heap(LaneId { worker: 0, lane: 1 }, src, 0..PIECE, "b");
        dag.download_heap(0, d, dst, 0..PIECE, "b back");
        dag.add_counted(body, RANGES, 2);
        let keep = Arc::new(Mutex::new(Vec::new()));
        dag.add_cpu("keep", 0, Keep(dst, Arc::clone(&keep)));
        let first = dag.upload_heap(0, src, 0..PIECE, "b first");
        let last = dag.upload_heap(0, src, 3 * PIECE..4 * PIECE, "b last");
        let planned = dag.plan();
        assert_eq!(planned.placement(1), fb_core_thread_pool::Placement::Cpu);
        assert_eq!(planned.predecessors(1), &[0]);
        assert_eq!(planned.predecessors(2), &[1]);
        assert!(planned.predecessors(3).contains(&2));
        assert!(planned.predecessors(4).contains(&3));
        assert!(planned.predecessors(5).contains(&4));
        assert_eq!(planned.device_table(0).len(d), PIECE);
        assert_eq!(planned.device_table(0).len(first), PIECE);
        assert_eq!(planned.device_table(0).len(last), PIECE);
        let text = planned.to_string();
        // The window's bytes pass on once their reader is done, so the
        // handoffs add to the edges above: each line up to its first
        // dependency.
        for line in [
            "    1       0  cpu     stage b: after 0\n",
            "    2       0  stream1  upload b: after 1\n",
            "    3       0  stream0  download b back: after ",
            "    4       0  cpu     unstage b back: after 0 3; closes counted loop 1-4, 4 trips, depth 2\n",
            "    5       0  cpu     keep: after 0 4\n",
            "    6       0  cpu     stage b first: after 0 ",
            "    7       0  stream0  upload b first: after ",
            "    8       0  cpu     stage b last: after 0 ",
            "    9       0  stream0  upload b last: after ",
        ] {
            assert!(text.contains(line), "{line:?} in\n{text}");
        }
        assert!(planned.predecessors(7).contains(&6));
        assert!(planned.predecessors(9).contains(&8));
        assert!(text.contains("  b back: 0w 4w 5r\n"), "{text}");
        for row in [
            "b window",
            "b back window",
            "b first window",
            "b last window",
        ] {
            assert!(text.contains(&format!("{row}: ")), "{row} in\n{text}");
        }
    }

    /// GPU, the stand-in's: the upload's window carried into a second
    /// window, the download's, as the device round trip leaves it. The
    /// stand-in's stream runs a node at once, so the host access is a
    /// finished copy (`CopyOnly::bufs`).
    struct RoundTrip(RowId<PinnedStandIn, f64>);

    impl GpuTask<PinnedStandIn, ()> for RoundTrip {
        type Rows = RowId<PinnedStandIn, f64>;
        fn layout(
            &self,
            _: &mut Layout<'_, Device>,
            pinned: &mut Layout<'_, PinnedStandIn>,
        ) -> Self::Rows {
            pinned.read(self.0);
            pinned.output("b back window", PIECE)
        }
        fn run(
            &self,
            back: &Self::Rows,
            _: &Bufs<'_, Device>,
            pinned: &CopyOnly<'_, PinnedStandIn>,
            _: &(),
        ) -> Status {
            let pinned = pinned.bufs();
            pinned.take_mut(*back).copy_from_slice(pinned.get(self.0));
            Ok(())
        }
    }

    /// The host halves over the pinned stand-in on `HostOnly` workers,
    /// a stream node standing in for the device round trip between them:
    /// a heap row cut into ranges, staged range `k` in trip `k` of a
    /// counted loop of depth 2 and unstaged into range `k` of a second
    /// heap row, which comes back bitwise the first. Each half is one
    /// entry per trip in the record, under its own name. A row whose
    /// length is no multiple of the range's comes back the same way in
    /// `div_ceil` trips, the last range clamped to the row's end: a clamp
    /// that copied too few elements leaves the block's fill in the
    /// second row, and one that copied past the end panics.
    #[test]
    fn heap_ranges_round_trip_bitwise() {
        for total in [RANGES * PIECE, RANGES * PIECE - 2] {
            round_trip(total);
        }
    }

    /// [`heap_ranges_round_trip_bitwise`] over a row of `total` elements.
    fn round_trip(total: usize) {
        let trips = total.div_ceil(PIECE);
        let keep = Arc::new(Mutex::new(Vec::new()));
        let mut dag: Dag<'_, PinnedStandIn, ()> = Dag::new();
        let (src, dst) = dag.add_cpu("source", 0, Source(total));
        let body = dag.len();
        let window = dag.add_cpu(
            "stage b",
            0,
            super::Stage {
                src,
                start: 0,
                len: PIECE,
                window: "b window".to_owned(),
            },
        );
        let back = dag.add_gpu("round trip b", 0, RoundTrip(window));
        dag.add_cpu(
            "unstage b back",
            0,
            Unstage {
                window: back,
                dst,
                start: 0,
                len: PIECE,
            },
        );
        dag.add_counted(body, trips, 2);
        dag.add_cpu("keep", 0, Keep(dst, Arc::clone(&keep)));
        let mut instance = dag.plan().instantiate(&[0]);
        for run in 0..2 {
            let record = instance.run(&mut HostOnly::new(1), 2);
            record.result.as_ref().unwrap();
            let got = std::mem::take(&mut *keep.lock().unwrap());
            let want: Vec<u64> = (0..total).map(|i| value(i).to_bits()).collect();
            assert_eq!(got, want, "total {total}, run {run}");
            for name in ["stage b", "unstage b back"] {
                let got: Vec<_> = record
                    .tasks
                    .iter()
                    .filter(|t| t.name == name)
                    .map(|t| (t.iteration, t.dims.get("len").copied()))
                    .collect();
                let each: Vec<_> = (0..trips).map(|k| (Some(k), Some(PIECE))).collect();
                assert_eq!(got, each, "total {total}, run {run}: {name}");
            }
        }
    }
}
