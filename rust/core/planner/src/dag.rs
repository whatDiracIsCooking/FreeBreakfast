//! Declaring a DAG of tasks, and planning it: the tables and the
//! dependencies.

use std::collections::BTreeSet;
use std::fmt;

use super::BLAS_WORKSPACE_BYTES;
use super::task::Task;
use fb_core_arena::{Bufs, Device, Element, HostKind, Pinned, RowId, Table, TableBuilder};
use fb_core_error_handling::Status;
use fb_core_thread_pool::{DeviceStream, Lane, Placement};

/// A task's `run` with its rows bound: what the executor's node calls.
pub(super) type Run<'a, H, S> =
    Box<dyn Fn(&Bufs<'_, Device>, &Bufs<'_, H>, Lane<'_, S>) -> Status + Send + Sync + 'a>;

/// Bind the closure's signature where it is written: a closure passed
/// straight to `Box::new` does not take its higher-ranked signature from
/// the box's type.
fn erase<'a, H: HostKind, S, F>(f: F) -> Run<'a, H, S>
where
    F: Fn(&Bufs<'_, Device>, &Bufs<'_, H>, Lane<'_, S>) -> Status + Send + Sync + 'a,
{
    Box::new(f)
}

/// A node as declared: its name, its worker, where it runs, its
/// dimensions, and its run.
pub(super) struct Declared<'a, H: HostKind, S> {
    pub(super) name: String,
    pub(super) dims: Vec<(&'static str, usize)>,
    pub(super) worker: usize,
    pub(super) placement: Placement,
    pub(super) run: Run<'a, H, S>,
}

/// One GPU worker's rows, of both kinds, and its BLAS workspace row if
/// one is declared.
struct Worker<H> {
    device: TableBuilder<Device>,
    host: TableBuilder<H>,
    blas_workspace: Option<RowId<Device, u8>>,
}

/// Tasks declared in plan order, each on a GPU worker: the planner's input.
///
/// A task's position is the number added before it, so the order of
/// [`Dag::add`] is the plan's rank order, which is a topological order: a
/// task can only read what an earlier one produced.
pub struct Dag<'a, H: HostKind = Pinned, S = DeviceStream> {
    workers: Vec<Worker<H>>,
    nodes: Vec<Declared<'a, H, S>>,
}

impl<H: HostKind, S> Default for Dag<'_, H, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, H: HostKind, S> Dag<'a, H, S> {
    /// An empty DAG.
    pub fn new() -> Self {
        Dag {
            workers: Vec::new(),
            nodes: Vec::new(),
        }
    }

    /// Add `task` on GPU worker `worker`, call its layout, and return its
    /// rows. A GPU task runs on the worker's stream; a CPU task runs on the
    /// CPU pool, with its host rows in that worker's block (a pinned row
    /// belongs to the GPU worker whose copies it stages).
    ///
    /// # Panics
    /// If a CPU task declares or reads a device row, or the layout reads a
    /// row of another worker's table ("a row of another table": an edge
    /// between two workers needs a transport row, which placement does not
    /// make yet), or a read the table refuses
    /// ([`TableBuilder::read`]).
    pub fn add<T>(&mut self, name: impl Into<String>, worker: usize, task: T) -> T::Rows
    where
        T: Task<H, S> + 'a,
        T::Rows: 'a,
    {
        let name = name.into();
        let pos = self.nodes.len();
        let w = self.worker(worker);
        let mut device = w.device.node(pos);
        let mut host = w.host.node(pos);
        let rows = task.layout(&mut device, &mut host);
        let device_rows = device.used();
        assert!(
            T::GPU || device_rows == 0,
            "{name}: a CPU task holds no device row, but its layout uses {device_rows}"
        );
        let kept = rows.clone();
        self.nodes.push(Declared {
            name,
            dims: task.dims(),
            worker,
            placement: if T::GPU {
                Placement::Stream(worker)
            } else {
                Placement::Cpu
            },
            run: erase(move |d, h, lane| task.run(&kept, d, h, lane)),
        });
        rows
    }

    /// A device row `worker` holds for the whole plan (a BLAS handle's
    /// workspace is [`Dag::blas_workspace`]'s, not this). Any task on the worker
    /// may write it through the returned id, with no declaration in its
    /// layout; two such tasks are not ordered by it, so they must be
    /// ordered by something else (one stream does).
    #[must_use]
    pub fn device_worker_row<T: Element>(
        &mut self,
        worker: usize,
        name: &str,
        len: usize,
    ) -> RowId<Device, T> {
        self.worker(worker).device.worker(name, len)
    }

    /// Declare worker `worker`'s BLAS workspace: a worker row of
    /// [`BLAS_WORKSPACE_BYTES`], `blas workspace` in its device table,
    /// which admission counts like any other row, and which the instance's
    /// backend hands the worker's BLAS handle when it is made
    /// (`Instance::backend`), so no BLAS call allocates behind the table.
    /// No task can name the row, so it is the handle's alone. Declaring it
    /// again for the same worker does nothing.
    ///
    /// A plan that declares none still runs, every BLAS call on the
    /// workspace the handle allocated for itself, outside the count.
    pub fn blas_workspace(&mut self, worker: usize) {
        let w = self.worker(worker);
        if w.blas_workspace.is_none() {
            w.blas_workspace = Some(w.device.worker("blas workspace", BLAS_WORKSPACE_BYTES));
        }
    }

    /// A host row `worker` holds for the whole plan.
    #[must_use]
    pub fn host_worker_row<T: Element>(
        &mut self,
        worker: usize,
        name: &str,
        len: usize,
    ) -> RowId<H, T> {
        self.worker(worker).host.worker(name, len)
    }

    /// Lay every worker's two tables out, and derive each task's
    /// dependencies: the producer of every row it reads, and every handoff
    /// into it. Pure: no device.
    pub fn plan(self) -> Planned<'a, H, S> {
        let blas_workspaces = self.workers.iter().map(|w| w.blas_workspace).collect();
        let tables: Vec<(Table<Device>, Table<H>)> = self
            .workers
            .into_iter()
            .map(|w| (w.device.build(), w.host.build()))
            .collect();
        let mut deps = vec![BTreeSet::new(); self.nodes.len()];
        for (device, host) in &tables {
            let pairs = [
                device.reads(),
                device.handoffs(),
                host.reads(),
                host.handoffs(),
            ];
            for (u, v) in pairs.into_iter().flatten() {
                // INVARIANT: the tables number nodes by rank, a row's
                // producer is added before its readers, and a handoff
                // points forward in rank (the arena's property test), so
                // no caller can make this fail.
                debug_assert!(u < v, "a dependency {u} -> {v} against rank");
                deps[v].insert(u);
            }
        }
        Planned {
            tables,
            blas_workspaces,
            nodes: self.nodes,
            deps: deps.into_iter().map(|d| d.into_iter().collect()).collect(),
        }
    }

    fn worker(&mut self, worker: usize) -> &mut Worker<H> {
        while self.workers.len() <= worker {
            self.workers.push(Worker {
                device: TableBuilder::new(),
                host: TableBuilder::new(),
                blas_workspace: None,
            });
        }
        &mut self.workers[worker]
    }
}

/// A planned DAG: each worker's tables laid out and each task's
/// dependencies derived, with nothing allocated. Admission reads the tables
/// ([`admit`](fb_core_arena::admit)); [`Planned::instantiate`] makes
/// the blocks.
///
/// Its text ([`fmt::Display`]) is the plan's golden: each task's worker,
/// lane and dependencies, then every non-empty table.
pub struct Planned<'a, H: HostKind = Pinned, S = DeviceStream> {
    pub(super) tables: Vec<(Table<Device>, Table<H>)>,
    /// Per worker, its BLAS workspace row, if declared.
    pub(super) blas_workspaces: Vec<Option<RowId<Device, u8>>>,
    pub(super) nodes: Vec<Declared<'a, H, S>>,
    pub(super) deps: Vec<Vec<usize>>,
}

impl<H: HostKind, S> Planned<'_, H, S> {
    /// The number of tasks.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The GPU workers the plan uses: one past the highest declared.
    pub fn n_workers(&self) -> usize {
        self.tables.len()
    }

    /// Worker `worker`'s device table.
    pub fn device_table(&self, worker: usize) -> &Table<Device> {
        &self.tables[worker].0
    }

    /// Worker `worker`'s host table.
    pub fn host_table(&self, worker: usize) -> &Table<H> {
        &self.tables[worker].1
    }

    /// The tasks `pos` runs after, by position, ascending.
    pub fn predecessors(&self, pos: usize) -> &[usize] {
        &self.deps[pos]
    }

    /// Where `pos` runs.
    pub fn placement(&self, pos: usize) -> Placement {
        self.nodes[pos].placement
    }
}

impl<H: HostKind, S> fmt::Display for Planned<'_, H, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "plan: {} tasks, GPU workers: {}",
            self.nodes.len(),
            self.tables.len()
        )?;
        writeln!(
            f,
            "{:>5}  {:>6}  {:<6}  name: after",
            "task", "worker", "lane"
        )?;
        for (pos, (node, deps)) in self.nodes.iter().zip(&self.deps).enumerate() {
            let lane = match node.placement {
                Placement::Stream(_) => "stream",
                Placement::Cpu => "cpu",
            };
            write!(f, "{pos:>5}  {:>6}  {lane:<6}  {}", node.worker, node.name)?;
            if deps.is_empty() {
                writeln!(f)?;
            } else {
                let deps: Vec<String> = deps.iter().map(usize::to_string).collect();
                writeln!(f, ": after {}", deps.join(" "))?;
            }
        }
        for (w, (device, host)) in self.tables.iter().enumerate() {
            if device.n_rows() > 0 {
                write!(f, "worker {w} {device}")?;
            }
            if host.n_rows() > 0 {
                write!(f, "worker {w} {host}")?;
            }
        }
        Ok(())
    }
}
