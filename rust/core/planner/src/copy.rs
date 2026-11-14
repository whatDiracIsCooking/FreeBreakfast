//! Copy nodes: a copy between kinds that is a node's whole work, added by
//! the planner in one line (`docs/memory.md`, "The table"). A task never
//! writes one by hand; the planner's three are the only `GpuTask`s whose
//! `run` is a copy alone (`doc_check/tests/code_rules.rs`).

use fb_core_arena::{
    Bufs, CopyOnly, Device, Element, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d,
};
use fb_core_error_handling::Status;
use fb_core_thread_pool::{DeviceStream, LaneId};

use super::dag::Dag;
use super::task::GpuTask;

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
    use super::*;
    use crate::CpuTask;
    use fb_core_arena::Heap;

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
}
