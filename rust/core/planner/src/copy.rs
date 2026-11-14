//! Copy nodes: a copy between kinds that is a node's whole work, added by
//! the planner in one line (`docs/memory.md`, "The table"). A task never
//! writes one by hand; the planner's two are the only `GpuTask`s whose
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
}
