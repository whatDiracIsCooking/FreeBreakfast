//! Declaring a DAG of tasks, and planning it: the tables and the
//! dependencies.

use std::collections::BTreeSet;
use std::fmt;
use std::ops::ControlFlow;

use super::BLAS_WORKSPACE_BYTES;
use super::task::{CpuTask, Decision, GpuTask};
use contracts::debug_requires;
use fb_core_arena::{
    Bufs, CopyOnly, Device, Element, Heap, Pinned, RowId, StagingKind, Table, TableBuilder,
};
use fb_core_error_handling::{Result, Status};
use fb_core_thread_pool::{DeviceStream, LaneId, Placement};

/// A GPU task's `run` with its rows bound: what the executor's stream
/// node calls.
pub(super) type GpuRun<'a, H, S> =
    Box<dyn Fn(&Bufs<'_, Device>, &CopyOnly<'_, H>, &S) -> Status + Send + Sync + 'a>;

/// A CPU task's `run` with its rows bound: what the executor's CPU node
/// calls.
pub(super) type CpuRun<'a, H> =
    Box<dyn Fn(&Bufs<'_, H>, &Bufs<'_, Heap>) -> Status + Send + Sync + 'a>;

/// A decision's `decide` with its rows bound: what the executor's
/// decision node calls with the iteration's index.
pub(super) type Decide<'a, H> =
    Box<dyn Fn(&Bufs<'_, H>, &Bufs<'_, Heap>, usize) -> Result<ControlFlow<()>> + Send + Sync + 'a>;

/// Bind the closure's signature where it is written: a closure passed
/// straight to `Box::new` does not take its higher-ranked signature from
/// the box's type.
fn erase_gpu<'a, H: StagingKind, S, F>(f: F) -> GpuRun<'a, H, S>
where
    F: Fn(&Bufs<'_, Device>, &CopyOnly<'_, H>, &S) -> Status + Send + Sync + 'a,
{
    Box::new(f)
}

/// As [`erase_gpu`], for a CPU task.
fn erase_cpu<'a, H: StagingKind, F>(f: F) -> CpuRun<'a, H>
where
    F: Fn(&Bufs<'_, H>, &Bufs<'_, Heap>) -> Status + Send + Sync + 'a,
{
    Box::new(f)
}

/// As [`erase_gpu`], for a decision.
fn erase_decide<'a, H: StagingKind, F>(f: F) -> Decide<'a, H>
where
    F: Fn(&Bufs<'_, H>, &Bufs<'_, Heap>, usize) -> Result<ControlFlow<()>> + Send + Sync + 'a,
{
    Box::new(f)
}

/// What a node runs: a task of either sort, or a loop's decision. The
/// sort is where it runs: a GPU task on its lane's stream, the rest on
/// the CPU pool.
pub(super) enum Work<'a, H: StagingKind, S> {
    Gpu(GpuRun<'a, H, S>),
    Cpu(CpuRun<'a, H>),
    /// The decision of the loop over the tasks from `body` to the one
    /// before it, replayed at most `max_iterations` times.
    Decide {
        body: usize,
        max_iterations: usize,
        decide: Decide<'a, H>,
    },
}

/// A node as declared: its name, its worker, its dimensions, and its work.
pub(super) struct Node<'a, H: StagingKind, S> {
    pub(super) name: String,
    pub(super) dims: Vec<(&'static str, usize)>,
    /// The worker whose tables hold its rows, and for a GPU task the lane
    /// it runs on; 0 for the rest.
    pub(super) at: LaneId,
    pub(super) work: Work<'a, H, S>,
}

impl<H: StagingKind, S> Node<'_, H, S> {
    /// Where it runs, which its work's sort decides.
    pub(super) fn placement(&self) -> Placement {
        match self.work {
            Work::Gpu(_) => Placement::Stream(self.at),
            Work::Cpu(_) | Work::Decide { .. } => Placement::Cpu,
        }
    }
}

/// One GPU worker's rows, of both kinds, shared by its lanes; its lane
/// count, one past the highest a GPU task named; per lane, whether a GPU
/// task on it uses the library handles ([`GpuTask::uses_libraries`]); its
/// fork streams, the most any of its GPU tasks forks onto
/// ([`GpuTask::forks`]); and its BLAS workspace rows, each with its lane,
/// once declared.
struct Worker<H> {
    device: TableBuilder<Device>,
    host: TableBuilder<H>,
    lanes: usize,
    libraries: Vec<bool>,
    forks: usize,
    blas_workspace: Vec<(usize, RowId<Device, u8>)>,
}

/// Tasks declared in plan order, each on a GPU worker: the planner's input.
///
/// A task's position is the number added before it ([`Dag::len`]), so the
/// order of [`Dag::add_gpu`] and [`Dag::add_cpu`] is the plan's rank order, which is a topological
/// order: a task can only read what an earlier one produced. A loop
/// ([`Dag::add_loop`]) is a run of them replayed between host decisions; a
/// counted loop ([`Dag::add_counted`]) one run a trip count fixed here,
/// with no decision, perhaps inside a decision loop's body.
///
/// Its tables are each GPU worker's device and pinned ones, which its
/// lanes share, and the plan's one heap table, whose rows only CPU nodes
/// hold (`docs/memory.md`, "The three kinds"). `H` is the staging kind of
/// the workers' host tables, never the heap's own, so a heap row cannot
/// be asked of a worker's table:
///
/// ```compile_fail,E0277
/// use fb_core_arena::Heap;
/// use fb_core_planner::Dag;
///
/// let dag: Dag<'_, Heap, ()> = Dag::new();
/// ```
pub struct Dag<'a, H: StagingKind = Pinned, S = DeviceStream> {
    workers: Vec<Worker<H>>,
    heap: TableBuilder<Heap>,
    nodes: Vec<Node<'a, H, S>>,
    /// Each loop, in plan order.
    loops: Vec<Loop>,
}

/// A loop of a [`Dag`]: its positions, from its body's first to its
/// decision or, for a counted loop, its body's last; and for a counted
/// loop its trip count and depth.
#[derive(Clone, Copy)]
pub(super) struct Loop {
    pub(super) first: usize,
    pub(super) last: usize,
    pub(super) counted: Option<(usize, usize)>,
}

impl<H: StagingKind, S> Default for Dag<'_, H, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, H: StagingKind, S> Dag<'a, H, S> {
    /// An empty DAG.
    pub fn new() -> Self {
        Dag {
            workers: Vec::new(),
            heap: TableBuilder::new(),
            nodes: Vec::new(),
            loops: Vec::new(),
        }
    }

    /// The number of nodes added so far, loops' decisions included: the
    /// position the next one will have, which is how a loop's body is
    /// named to [`Dag::add_loop`].
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether nothing has been added.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Add a GPU task on lane `at` of a GPU worker, to run on that lane's
    /// stream, call its layout, and return its rows. A bare worker index is
    /// its lane 0 ([`LaneId`]).
    ///
    /// Its rows are the worker's, in tables its lanes share, so a task may
    /// read a row a task on another lane of the worker wrote: the executor
    /// orders the two by an event on the device, with no host wait, and the
    /// reader reads the row in place. The lanes a worker has are the ones
    /// its tasks name, fixed when the DAG is planned.
    ///
    /// The layout's declarations are held to the worker's tables'
    /// contracts ([`TableBuilder::read`], [`TableBuilder::write`]).
    ///
    /// # Panics
    /// If the layout reads a row of another worker's table (an edge between
    /// two workers needs a transport row, which placement does not make
    /// yet): here, by the table's contract, in a debug build; in every
    /// build, when the task's run asks for the row, by the view's checks
    /// ([`Bufs`]).
    pub fn add_gpu<T>(&mut self, name: impl Into<String>, at: impl Into<LaneId>, task: T) -> T::Rows
    where
        T: GpuTask<H, S> + 'a,
        T::Rows: 'a,
    {
        let at = at.into();
        let pos = self.nodes.len();
        let w = self.worker(at.worker);
        w.lanes = w.lanes.max(at.lane + 1);
        if w.libraries.len() < w.lanes {
            w.libraries.resize(w.lanes, false);
        }
        w.libraries[at.lane] |= task.uses_libraries();
        w.forks = w.forks.max(task.forks());
        let rows = task.layout(&mut w.device.node(pos), &mut w.host.node(pos));
        let kept = rows.clone();
        self.nodes.push(Node {
            name: name.into(),
            dims: task.dims(),
            at,
            work: Work::Gpu(erase_gpu(move |d, p, stream| task.run(&kept, d, p, stream))),
        });
        rows
    }

    /// Add a CPU task, to run on the CPU pool with its pinned rows in GPU
    /// worker `worker`'s block (a pinned row belongs to the GPU worker whose
    /// copies it stages) and its heap rows in the plan's, call its layout,
    /// and return its rows.
    ///
    /// The layout's declarations are held to the worker's host table's
    /// and the heap table's contracts, as in [`Dag::add_gpu`].
    ///
    /// # Panics
    /// As [`Dag::add_gpu`], for a row of another worker's table.
    pub fn add_cpu<T>(&mut self, name: impl Into<String>, worker: usize, task: T) -> T::Rows
    where
        T: CpuTask<H> + 'a,
        T::Rows: 'a,
    {
        let pos = self.nodes.len();
        self.worker(worker);
        let rows = task.layout(
            &mut self.workers[worker].host.node(pos),
            &mut self.heap.node(pos),
        );
        let kept = rows.clone();
        self.nodes.push(Node {
            name: name.into(),
            dims: task.dims(),
            at: worker.into(),
            work: Work::Cpu(erase_cpu(move |h, heap| task.run(&kept, h, heap))),
        });
        rows
    }

    /// Close a loop: the tasks from position `body` to the last one added
    /// are its body, replayed in the same run, on the same workers, until
    /// `decision`, a CPU node added after them on GPU worker `worker`'s
    /// host table and the plan's heap table, says `Break`, at most `max_iterations` times
    /// (`docs/programming_model.md`, "Plan once, run many";
    /// `fb_core_thread_pool::TaskGraph::add_loop`, which runs it). Calls the
    /// decision's layout, and returns its rows.
    ///
    /// The body's own rows, those every user of which is in the loop, are
    /// laid out in each worker's table as a table of their own, in a region
    /// the rest of the table holds while the loop runs, sized to the body's
    /// peak once (`docs/memory.md`, "Nested plans"): a replay allocates
    /// nothing and is admitted with the plan. Every other row the body
    /// uses, S read every iteration or the loop's output read after it, is
    /// the parent's, live for the whole loop. A row the loop carries from
    /// one iteration to the next, a density or a DIIS history, is produced
    /// before the loop, read in the body, and written by its last user
    /// there ([`Layout::write`](fb_core_arena::Layout::write)); the heap
    /// table's region and carried rows work the same way.
    ///
    /// A task after the loop that reads a row the body wrote runs after the
    /// decision's `Break`, never after a single iteration.
    ///
    /// The body may hold counted loops ([`Dag::add_counted`]), closed
    /// before it, J and K over a streamed tensor every SCF iteration: each
    /// one's region is a row of the body's table, live over its span, and
    /// each is replayed whole every iteration, its trips overlapped with no
    /// host wait between them (`docs/memory.md`, "Nested plans").
    ///
    /// The decision's layout is held to the table's contracts, as a
    /// task's is in [`Dag::add_cpu`].
    #[debug_requires(body < self.nodes.len(), "the body has a task")]
    #[debug_requires(
        self.loops.iter().all(|l| l.last < body || l.counted.is_some()),
        "a decision loop holds no decision loop"
    )]
    #[debug_requires(
        self.loops.iter().all(|l| l.last < body || l.first >= body),
        "a loop holds another whole or not at all"
    )]
    #[debug_requires(max_iterations > 0, "the bound admits an iteration")]
    pub fn add_loop<D>(
        &mut self,
        name: impl Into<String>,
        worker: usize,
        body: usize,
        max_iterations: usize,
        decision: D,
    ) -> D::Rows
    where
        D: Decision<H> + 'a,
        D::Rows: 'a,
    {
        let pos = self.nodes.len();
        self.worker(worker);
        let rows = decision.layout(
            &mut self.workers[worker].host.node(pos),
            &mut self.heap.node(pos),
        );
        let kept = rows.clone();
        self.nodes.push(Node {
            name: name.into(),
            dims: Vec::new(),
            at: worker.into(),
            work: Work::Decide {
                body,
                max_iterations,
                decide: erase_decide(move |h, heap, iteration| {
                    decision.decide(&kept, h, heap, iteration)
                }),
            },
        });
        self.loops.push(Loop {
            first: body,
            last: pos,
            counted: None,
        });
        rows
    }

    /// Close a counted loop: the tasks from position `body` to the last
    /// one added are its body, run `trips` times in the same run with no
    /// decision and no host wait between iterations, up to `depth` of them
    /// in flight at once (`docs/programming_model.md`, "Plan once, run
    /// many"; `fb_core_thread_pool::TaskGraph::add_counted`, which runs it). A
    /// body task reads its iteration's index from its views
    /// ([`Bufs::iteration`]), by which it picks its batch.
    ///
    /// The body's own rows are laid out once as a table of their own and
    /// held `depth` times in a region of each worker's tables, iteration
    /// `k` in copy `k mod depth`, and iteration `k` starts only after
    /// iteration `k - depth` (`docs/memory.md`, "Nested plans"): for
    /// batched work, an upload on a copy lane and its GEMMs on a compute
    /// lane, two copies let the next batch's upload run under this one's
    /// GEMMs. A row the body accumulates into across iterations, K summed
    /// over batches, is the parent's, produced before the loop and written
    /// by its one user in the body ([`Layout::write`]); an output read
    /// after the loop is its last iteration's, from its one user in the
    /// body ([`TableBuilder::add_counted`]).
    ///
    /// [`Layout::write`]: fb_core_arena::Layout::write
    /// [`TableBuilder::add_counted`]: fb_core_arena::TableBuilder::add_counted
    ///
    /// A task after the loop that reads a row the body wrote runs after
    /// the writer's last iteration. The body's GPU tasks add no sync
    /// point: between lanes every edge, within an iteration and across
    /// the ring, is a wait on the device.
    ///
    /// It may lie in the body of a decision loop closed after it
    /// ([`Dag::add_loop`]), and holds no loop itself; there the
    /// accumulator may be the decision loop's own row, zeroed in its body
    /// before this loop and read after it.
    #[debug_requires(body < self.nodes.len(), "the body has a task")]
    #[debug_requires(
        self.loops.last().is_none_or(|l| l.last < body),
        "a counted loop holds no loop"
    )]
    #[debug_requires(trips > 0, "the loop has an iteration")]
    #[debug_requires(0 < depth && depth <= trips, "the depth is from 1 to the trip count")]
    pub fn add_counted(&mut self, body: usize, trips: usize, depth: usize) {
        self.loops.push(Loop {
            first: body,
            last: self.nodes.len() - 1,
            counted: Some((trips, depth)),
        });
    }

    /// A device row `worker` holds for the whole plan (a BLAS handle's
    /// workspace is [`Dag::blas_workspace`]'s, not this). Any task on the worker
    /// may write it through the returned id, with no declaration in its
    /// layout; two such tasks are not ordered by it, so they must be
    /// ordered by something else (one lane does; two lanes of the worker
    /// need an edge).
    #[must_use]
    pub fn device_worker_row<T: Element>(
        &mut self,
        worker: usize,
        name: &str,
        len: usize,
    ) -> RowId<Device, T> {
        self.worker(worker).device.worker(name, len)
    }

    /// Declare worker `worker`'s BLAS workspaces: one worker row of
    /// [`BLAS_WORKSPACE_BYTES`] per lane that has a BLAS handle, since each
    /// such lane has one of its own and two lanes run at once. Lane 0's is
    /// `blas workspace` in the worker's device table, declared here; lane
    /// `l`'s, `blas workspace lane <l>`, when the DAG is planned and its
    /// lanes are known, and only if a task on it uses the library handles
    /// ([`GpuTask::uses_libraries`]): a lane whose tasks only launch kernels
    /// or copy has no handles and no row. Admission counts them like any other row, and the instance's
    /// backend hands each to its lane's BLAS handle when it is made
    /// (`Instance::backend`), so no BLAS call allocates behind the table.
    /// No task can name the rows, so each is its handle's alone. Declaring
    /// them again for the same worker does nothing.
    ///
    /// A plan that declares none still runs, every BLAS call on the
    /// workspace the handle allocated for itself, outside the count.
    pub fn blas_workspace(&mut self, worker: usize) {
        let w = self.worker(worker);
        if w.blas_workspace.is_empty() {
            let row = w.device.worker("blas workspace", BLAS_WORKSPACE_BYTES);
            w.blas_workspace.push((0, row));
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

    /// Lay every worker's two tables and the heap table out, each loop's
    /// body in its region,
    /// and derive each task's dependencies: the producer of every row it
    /// reads, the writer a loop's carried row is ordered against, and
    /// every handoff into it; a loop's decision also runs after every task
    /// of its body. Pure: no device. The rows a loop carries are held to
    /// the tables' contracts ([`TableBuilder::build`]).
    pub fn plan(mut self) -> Plan<'a, H, S> {
        for w in &mut self.workers {
            // Lane 0 has its handles whatever its tasks say; a lane above
            // it, only if a task on it uses them.
            w.libraries.resize(w.lanes.max(1), false);
            w.libraries[0] = true;
            if !w.blas_workspace.is_empty() {
                for lane in (1..w.lanes).filter(|&l| w.libraries[l]) {
                    let name = format!("blas workspace lane {lane}");
                    let row = w.device.worker(&name, BLAS_WORKSPACE_BYTES);
                    w.blas_workspace.push((lane, row));
                }
            }
        }
        let blas_workspaces = self
            .workers
            .iter_mut()
            .map(|w| std::mem::take(&mut w.blas_workspace))
            .collect();
        let libraries = self
            .workers
            .iter_mut()
            .map(|w| std::mem::take(&mut w.libraries))
            .collect();
        let lanes = self.workers.iter().map(|w| w.lanes).collect();
        let forks = self.workers.iter().map(|w| w.forks).collect();
        let loops = &self.loops;
        let mut heap = self.heap;
        for l in loops {
            match l.counted {
                Some((_, depth)) => heap.add_counted(l.first..=l.last, depth),
                None => heap.add_loop(l.first..=l.last),
            }
        }
        let heap = heap.build();
        let tables: Vec<(Table<Device>, Table<H>)> = self
            .workers
            .into_iter()
            .map(|mut w| {
                for l in loops {
                    match l.counted {
                        Some((_, depth)) => {
                            w.device.add_counted(l.first..=l.last, depth);
                            w.host.add_counted(l.first..=l.last, depth);
                        }
                        None => {
                            w.device.add_loop(l.first..=l.last);
                            w.host.add_loop(l.first..=l.last);
                        }
                    }
                }
                (w.device.build(), w.host.build())
            })
            .collect();
        let mut deps = vec![BTreeSet::new(); self.nodes.len()];
        for l in loops.iter().filter(|l| l.counted.is_none()) {
            deps[l.last].extend(l.first..l.last);
        }
        let mut pairs = vec![heap.reads(), heap.handoffs()];
        for (device, host) in &tables {
            pairs.extend([
                device.reads(),
                device.handoffs(),
                host.reads(),
                host.handoffs(),
            ]);
        }
        for (u, v) in pairs.into_iter().flatten() {
            // INVARIANT: the tables number nodes by rank, a row's producer
            // is added before its readers, and a handoff points forward in
            // rank (the arena's property test), so no caller can make this
            // fail.
            debug_assert!(u < v, "a dependency {u} -> {v} against rank");
            deps[v].insert(u);
        }
        Plan {
            tables,
            heap,
            lanes,
            libraries,
            forks,
            blas_workspaces,
            nodes: self.nodes,
            deps: deps.into_iter().map(|d| d.into_iter().collect()).collect(),
            loops: self.loops,
        }
    }

    fn worker(&mut self, worker: usize) -> &mut Worker<H> {
        while self.workers.len() <= worker {
            self.workers.push(Worker {
                device: TableBuilder::new(),
                host: TableBuilder::new(),
                lanes: 0,
                libraries: Vec::new(),
                forks: 0,
                blas_workspace: Vec::new(),
            });
        }
        &mut self.workers[worker]
    }
}

/// A planned DAG: each worker's tables laid out and each task's
/// dependencies derived, with nothing allocated. Admission reads the tables
/// ([`admit`](fb_core_arena::admit)); [`Plan::instantiate`] makes
/// the blocks.
///
/// Its text ([`fmt::Display`]) is the plan's golden: each task's worker,
/// lane and dependencies, each loop's span and bound on its decision's
/// line and each counted loop's span, trips and depth on its body's last,
/// then every non-empty table, a loop's region and body in it: each
/// worker's, then the heap's. A GPU
/// task's lane reads `stream` on a worker of one lane and `stream<l>` on a
/// worker of several; any other task's, `cpu`.
pub struct Plan<'a, H: StagingKind = Pinned, S = DeviceStream> {
    pub(super) tables: Vec<(Table<Device>, Table<H>)>,
    /// The plan's heap table, one for every worker.
    pub(super) heap: Table<Heap>,
    /// Per worker, its lane count: one past the highest a GPU task named.
    pub(super) lanes: Vec<usize>,
    /// Per worker, per lane, whether the lane has library handles: lane 0
    /// always, a lane above it if a task on it uses them.
    pub(super) libraries: Vec<Vec<bool>>,
    /// Per worker, its fork streams: the most any of its GPU tasks forks
    /// onto.
    pub(super) forks: Vec<usize>,
    /// Per worker, its BLAS workspace rows, each with its lane: one per
    /// lane with library handles, if declared.
    pub(super) blas_workspaces: Vec<Vec<(usize, RowId<Device, u8>)>>,
    pub(super) nodes: Vec<Node<'a, H, S>>,
    pub(super) deps: Vec<Vec<usize>>,
    pub(super) loops: Vec<Loop>,
}

impl<H: StagingKind, S> Plan<'_, H, S> {
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

    /// Worker `worker`'s lanes: one past the highest a GPU task on it
    /// named, 0 for a worker with no GPU task.
    pub fn n_lanes(&self, worker: usize) -> usize {
        self.lanes[worker]
    }

    /// Worker `worker`'s device table.
    pub fn device_table(&self, worker: usize) -> &Table<Device> {
        &self.tables[worker].0
    }

    /// Worker `worker`'s host table.
    pub fn host_table(&self, worker: usize) -> &Table<H> {
        &self.tables[worker].1
    }

    /// The plan's heap table: the rows its CPU nodes hold on the heap.
    pub fn heap_table(&self) -> &Table<Heap> {
        &self.heap
    }

    /// The tasks `pos` runs after, by position, ascending.
    pub fn predecessors(&self, pos: usize) -> &[usize] {
        &self.deps[pos]
    }

    /// Where `pos` runs.
    pub fn placement(&self, pos: usize) -> Placement {
        self.nodes[pos].placement()
    }
}

impl<H: StagingKind, S> fmt::Display for Plan<'_, H, S> {
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
            let lane = match node.placement() {
                Placement::Stream(at) if self.lanes[at.worker] > 1 => format!("stream{}", at.lane),
                Placement::Stream(_) => "stream".to_string(),
                Placement::Cpu => "cpu".to_string(),
            };
            write!(
                f,
                "{pos:>5}  {:>6}  {lane:<6}  {}",
                node.at.worker, node.name
            )?;
            if !deps.is_empty() {
                let deps: Vec<String> = deps.iter().map(usize::to_string).collect();
                write!(f, ": after {}", deps.join(" "))?;
            }
            if let Work::Decide {
                body,
                max_iterations,
                ..
            } = node.work
            {
                write!(
                    f,
                    "; decides loop {body}-{pos}, at most {max_iterations} iterations"
                )?;
            }
            for l in self.loops.iter().filter(|l| l.last == pos) {
                if let Some((trips, depth)) = l.counted {
                    write!(
                        f,
                        "; closes counted loop {}-{pos}, {trips} trips, depth {depth}",
                        l.first
                    )?;
                }
            }
            writeln!(f)?;
        }
        for (w, (device, host)) in self.tables.iter().enumerate() {
            if device.n_rows() > 0 {
                write!(f, "worker {w} {device}")?;
            }
            if host.n_rows() > 0 {
                write!(f, "worker {w} {host}")?;
            }
        }
        if self.heap.n_rows() > 0 {
            write!(f, "plan {}", self.heap)?;
        }
        Ok(())
    }
}
