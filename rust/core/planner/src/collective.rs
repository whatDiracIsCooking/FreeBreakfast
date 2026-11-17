//! Collective nodes: a row moved between the GPU workers, added by the
//! planner in one line, each a collective of a part per worker
//! ([`Dag::add_collective`]) over the backend's clique
//! (`docs/programming_model.md`, "Collectives").

use contracts::debug_requires;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::ccl;
use fb_core_wwr::copy::d2d;
use fb_core_wwr::device_slice::split_at_mut;

use super::dag::Dag;
use super::partition::Partition;
use super::task::GpuTask;

impl<'a> Dag<'a, Pinned, DeviceStream> {
    /// Add the broadcast of `src`, `len` elements on worker `root`, to
    /// every worker of `ranks`: a collective `broadcast <name>`
    /// ([`Dag::add_collective`]). Returns the row each worker reads, in
    /// worker order: `src` itself on the root, which only reads it, and on
    /// every other worker a new output row `name` of `len` elements. The
    /// copy is exact, so a reader on any worker reads the root's bytes.
    ///
    /// # Panics
    /// If `src` is not a row of `root`'s table ([`Dag::add_gpu`]); and when
    /// the plan runs, if `src` is not `len` long.
    #[debug_requires(root < ranks, "the root is one of the workers")]
    pub fn broadcast(
        &mut self,
        name: &str,
        root: usize,
        src: RowId<Device, f64>,
        len: usize,
        ranks: usize,
    ) -> Vec<RowId<Device, f64>> {
        let parts = (0..ranks)
            .map(|w| Broadcast {
                src: (w == root).then_some(src),
                root,
                name: name.to_owned(),
                len,
            })
            .collect();
        self.add_collective(&format!("broadcast {name}"), parts)
    }

    /// Add the gather of `pieces`, worker `w`'s piece the rows
    /// `split.range(w)` of the whole, onto worker `root`: a collective
    /// `gather <name>` ([`Dag::add_collective`]), each piece sent to the
    /// root and written at its range of a new output row `name` of
    /// `split.total()` elements there, which it returns. The pieces are
    /// disjoint, so the gather is exact: the root's row is bitwise the
    /// rows one worker would have written whole. A piece of no rows sends
    /// nothing.
    ///
    /// # Panics
    /// If a piece is not a row of its worker's table ([`Dag::add_gpu`]);
    /// and when the plan runs, if a piece is not its range's length.
    #[debug_requires(pieces.len() == split.parts(), "a piece per part")]
    #[debug_requires(root < pieces.len(), "the root is one of the workers")]
    pub fn gather(
        &mut self,
        name: &str,
        root: usize,
        pieces: &[RowId<Device, f64>],
        split: &Partition,
    ) -> RowId<Device, f64> {
        let parts = pieces
            .iter()
            .enumerate()
            .map(|(w, &piece)| Gather {
                piece,
                root,
                worker: w,
                name: name.to_owned(),
                split: split.clone(),
            })
            .collect();
        let mut rows = self.add_collective(&format!("gather {name}"), parts);
        rows.swap_remove(root)
            .expect("the root's part outputs the gathered row")
    }

    /// Add the sum of `partials`, one of `len` elements per worker, onto
    /// worker `root`: a collective `reduce <name>`
    /// ([`Dag::add_collective`]), into a new output row `name` of `len`
    /// elements there, which it returns. The sum's order is the
    /// library's, fixed by its algorithm and the worker count, so it is an
    /// unordered reduce (`docs/programming_model.md`, "An edge declares"):
    /// its result differs from one worker's sum by rounding, which a
    /// caller's bar fixes by analysis.
    ///
    /// # Panics
    /// If a partial is not a row of its worker's table ([`Dag::add_gpu`]);
    /// and when the plan runs, if a partial is not `len` long.
    #[debug_requires(root < partials.len(), "the root is one of the workers")]
    pub fn reduce(
        &mut self,
        name: &str,
        root: usize,
        partials: &[RowId<Device, f64>],
        len: usize,
    ) -> RowId<Device, f64> {
        let parts = partials
            .iter()
            .enumerate()
            .map(|(w, &partial)| Reduce {
                partial,
                root,
                worker: w,
                name: name.to_owned(),
                len,
            })
            .collect();
        let mut rows = self.add_collective(&format!("reduce {name}"), parts);
        rows.swap_remove(root)
            .expect("the root's part outputs the sum")
    }
}

/// A broadcast's part: the root's (`src` some) reads its row, every
/// other worker's writes a new one.
struct Broadcast {
    src: Option<RowId<Device, f64>>,
    root: usize,
    name: String,
    len: usize,
}

impl GpuTask for Broadcast {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        match self.src {
            Some(src) => device.read(src),
            None => device.output(&self.name, self.len),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len)]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        row: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        if self.len == 0 {
            return Ok(());
        }
        let comm = stream.comm();
        match self.src {
            Some(src) => ccl::broadcast_send(comm, &device.get(src), stream),
            None => ccl::broadcast_recv(comm, &mut device.take_mut(*row), self.root, stream),
        }
    }
}

/// A gather's part: worker `worker`'s piece, sent to the root; the root's
/// copies its own into place and receives every other one.
struct Gather {
    piece: RowId<Device, f64>,
    root: usize,
    worker: usize,
    name: String,
    split: Partition,
}

impl GpuTask for Gather {
    type Rows = Option<RowId<Device, f64>>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.piece);
        (self.worker == self.root).then(|| device.output(&self.name, self.split.total()))
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.split.range(self.worker).len())]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        whole: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let comm = stream.comm();
        let Some(whole) = *whole else {
            if self.split.range(self.worker).is_empty() {
                return Ok(());
            }
            return ccl::send(comm, &device.get(self.piece), self.root, stream);
        };
        let mut whole = device.take_mut(whole);
        let own = self.split.range(self.root);
        {
            let (_, mut rest) = split_at_mut(&mut whole, own.start);
            let (mut at, _) = split_at_mut(&mut rest, own.len());
            d2d(&mut at, &device.get(self.piece), stream)?;
        }
        ccl::group(comm, || {
            for w in (0..self.split.parts()).filter(|&w| w != self.root) {
                let range = self.split.range(w);
                if range.is_empty() {
                    continue;
                }
                let (_, mut rest) = split_at_mut(&mut whole, range.start);
                let (mut at, _) = split_at_mut(&mut rest, range.len());
                ccl::recv(comm, &mut at, w, stream)?;
            }
            Ok(())
        })
    }
}

/// A reduce's part: worker `worker`'s partial sent into the sum, the
/// root's into a new row.
struct Reduce {
    partial: RowId<Device, f64>,
    root: usize,
    worker: usize,
    name: String,
    len: usize,
}

impl GpuTask for Reduce {
    type Rows = Option<RowId<Device, f64>>;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.partial);
        (self.worker == self.root).then(|| device.output(&self.name, self.len))
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("len", self.len)]
    }

    fn uses_libraries(&self) -> bool {
        false
    }

    fn run(
        &self,
        sum: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        if self.len == 0 {
            return Ok(());
        }
        let comm = stream.comm();
        let partial = device.get(self.partial);
        match *sum {
            Some(sum) => ccl::reduce_recv(comm, &partial, &mut device.take_mut(sum), stream),
            None => ccl::reduce_send(comm, &partial, self.root, stream),
        }
    }
}
