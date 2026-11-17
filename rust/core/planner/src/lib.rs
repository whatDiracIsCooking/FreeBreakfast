//! The planner: tasks declared once, turned into the arena's tables and the
//! executor's plan (`docs/programming_model.md`, "Three owners").
//!
//! No C++ counterpart: the C++ pipeline's steps allocate as they go and
//! run one at a time. This is the glue between [`arena`](fb_core_arena)
//! and [`thread_pool`](fb_core_thread_pool); without it a caller wrote
//! each row's position, each node's dependencies and each `partition` call
//! by hand.
//!
//! - [`GpuTask`] and [`CpuTask`], a task of each sort: `layout`, which
//!   declares every row the task holds or reads, and `run`, which computes
//!   through them. A GPU task declares device and pinned rows and runs on
//!   a lane of its worker, one of the worker's streams, its pinned rows a
//!   [`CopyOnly`] only its copies take; a CPU task declares host rows alone,
//!   pinned and on the heap, and runs on the CPU pool.
//! - [`Dag`]: tasks added in plan order, each on a GPU worker, by
//!   [`Dag::add_gpu`] (on a lane the caller names, a [`LaneId`]) or
//!   [`Dag::add_cpu`], the sort deciding where it runs. A worker's tables
//!   and blocks are its lanes', so a task reads a row another lane wrote in
//!   place. Each calls the layout at once and
//!   returns the task's rows, whose output ids the caller passes to the
//!   tasks that read them. [`Dag::blas_workspace`] declares a worker's
//!   BLAS workspaces, a worker row per lane no task can name. [`Dag::add_loop`]
//!   closes a loop over the tasks from a position on, with a [`Decision`]
//!   after them: the body replayed in one run until the decision says
//!   `Break`, its rows laid out once in a region of each worker's tables,
//!   the rows it carries between iterations and its outputs the parent's
//!   (`docs/memory.md`, "Nested plans"). [`Dag::add_counted`] closes a
//!   counted loop: the body run a trip count fixed at planning, no
//!   decision and no host wait between iterations, its rows held once per
//!   iteration in flight, each body task reading its iteration from its
//!   views ([`Bufs::iteration`]) to pick its batch; it may lie in a
//!   decision loop's body, replayed whole every iteration. With `device`,
//!   `Dag::upload` and `Dag::download` add a copy node, a pinned row to a
//!   new device row or back, named after the new row, and
//!   `Dag::download_rows` one node for several device rows down, one
//!   event for them all, and `Dag::upload_heap` / `Dag::download_heap` a
//!   range of a heap row through a pinned window, a CPU and a GPU node
//!   each: a copy that is a node's whole work is never a
//!   hand-written task (`docs/memory.md`, "The table").
//! - [`Dag::add_collective`]: one step with a part on every GPU worker,
//!   each on its worker's lane 0 holding its worker's rows, the parts
//!   exchanging rows on the devices through the backend's clique, and
//!   each worker's collectives issued in plan order
//!   (`docs/programming_model.md`, "Collectives"). With `device`,
//!   `Dag::broadcast`, `Dag::gather` and `Dag::reduce` are the planner's
//!   own: a row from a root to every worker, the pieces of a
//!   [`Partition`] onto a root, and partials summed onto one.
//!
//! [`Bufs::iteration`]: fb_core_arena::Bufs::iteration
//! - [`Adaptive`]: a part of the plan whose width the planner picks, a
//!   batch, declared by `width_max` and `bytes(width)`. [`Dag::width`]
//!   asks for it before the part's tasks are added, and
//!   [`Dag::plan_within`] plans a DAG declared by a closure against each
//!   worker's device room, bisecting each part's bytes once, by
//!   [`widest`], in the room the schedule leaves at its position
//!   (`docs/memory.md`, "Adaptive widths: one planner routine"). The width
//!   is a dimension of the plan, on its text's `width` lines and logged at
//!   `debug`, at `warn` when narrowed below its floor; `run` never reads
//!   free memory.
//! - [`Plan`] ([`Dag::plan`]): each worker's device and host tables and
//!   the plan's one heap table laid out, and each task's dependencies
//!   derived, all pure. A task runs after
//!   the producer of every row it reads ([`Table::reads`]) and after every
//!   user of the bytes it takes over ([`Table::handoffs`]). Admission is
//!   [`admit`](fb_core_arena::admit) on its tables; its text is the
//!   plan's golden.
//! - [`Graph`] ([`Plan::graph`]): the plan's tasks, dependencies, loops
//!   and memory, owned, so it outlives [`Plan::instantiate`];
//!   [`Graph::dot`] writes it as Graphviz, each edge labelled with the rows
//!   behind it and each task with the bytes it produces and the bytes live
//!   while it runs, and with a run's [`Record`](fb_core_thread_pool::Record)
//!   each task's times.
//! - [`Instance`] ([`Plan::instantiate`]): the blocks, each worker's two
//!   and the plan's heap block, and a
//!   [`thread_pool::TaskGraph`] whose node for each task partitions the blocks
//!   for it and calls its `run`. Run any number of times, one at a time:
//!   [`Instance::run`] takes the instance and the backend by `&mut`
//!   (`fb_core_thread_pool`, "One run at a time"). Each run returns the
//!   executor's [`Record`](fb_core_thread_pool::Record), every task timed
//!   by the executor and none by itself. With `device`,
//!   `Instance::backend` makes the `thread_pool::Device` it runs on and
//!   hands each lane's BLAS handle its declared workspace, held for as
//!   long as the backend lives.
//!
//! [`CopyOnly`]: fb_core_arena::CopyOnly
//! [`LaneId`]: fb_core_thread_pool::LaneId
//! [`Table::reads`]: fb_core_arena::Table::reads
//! [`Table::handoffs`]: fb_core_arena::Table::handoffs
//! [`thread_pool::TaskGraph`]: fb_core_thread_pool::TaskGraph
//!
//! # What a plan promises
//!
//! Whatever order the executor picks among ready tasks: a reader starts
//! after its producer has finished in the sense the edge needs (the
//! executor's "When an edge is satisfied"), and a row's bytes pass to the
//! next row only after every user of the previous one. So no task reads a
//! row before it is written, and no two tasks hold the same bytes at once,
//! on one lane or several; the arena's borrow flags would panic if one did
//! on the host. Between two lanes of a worker both orderings are events on
//! the device, so a plan of several lanes adds no host wait.
//!
//! This module has no `unsafe`: the views are what make sharing the blocks
//! across workers sound.
//!
//! ```
//! use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
//! use fb_core_error_handling::Status;
//! use fb_core_planner::{Dag, GpuTask};
//! use fb_core_thread_pool::DeviceStream;
//!
//! /// Writes an `n`-element output, reading `input` if it has one.
//! struct Step {
//!     n: usize,
//!     input: Option<RowId<Device, f64>>,
//! }
//!
//! impl GpuTask for Step {
//!     type Rows = RowId<Device, f64>;
//!     fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
//!         if let Some(input) = self.input {
//!             d.read(input);
//!         }
//!         d.output("out", self.n)
//!     }
//!     fn run(
//!         &self,
//!         _: &Self::Rows,
//!         _: &Bufs<'_, Device>,
//!         _: &CopyOnly<'_, Pinned>,
//!         _: &DeviceStream,
//!     ) -> Status {
//!         Ok(()) // enqueue on the stream
//!     }
//! }
//!
//! let n = 24 * 24;
//! let mut dag = Dag::new();
//! let s = dag.add_gpu("S", 0, Step { n, input: None });
//! let x = dag.add_gpu("X", 0, Step { n, input: Some(s) });
//! let f = dag.add_gpu("F", 0, Step { n, input: Some(x) });
//! let planned = dag.plan();
//! // F takes S's bytes, which are dead once X has read them: the reads
//! // order X after S and F after X, and the handoff F after S's users.
//! assert_eq!(planned.device_table(0).offset(f), planned.device_table(0).offset(s));
//! assert_eq!(planned.predecessors(1), &[0]);
//! assert_eq!(planned.predecessors(2), &[0, 1]);
//! // planned.instantiate(&[0]) allocates; Instance::run runs.
//! ```
//!
//! # Not yet
//!
//! The caller places each task, on a worker and a lane, and the plan order
//! is the order of the adds: no placement or schedule search, and no lanes
//! derived from the DAG's shape. The assigner packs to the fewest bytes,
//! so every reuse between two lanes serializes them; nothing gives a reuse
//! up for concurrency (`docs/memory.md`, "Open questions"). A task
//! reads only rows of its own worker; a row crosses between workers by a
//! collective alone, whose parts the caller places on every worker, and no
//! edge between workers is priced. A partition is the caller's tasks over
//! a [`Partition`]'s ranges, not a declaration the planner slices. An
//! adaptive width is bisected against a worker's device room alone, the
//! pinned rows that grow with it admitted after; no sub-DAG with
//! dimensions of its own, and of loops in a loop's body only a counted
//! loop in a decision loop's. Every
//! dependency is a row: an ordering with no buffer behind it has no way to
//! be declared until a task needs one.

#![forbid(unsafe_code)]

#[cfg(feature = "device")]
mod collective;
#[cfg(feature = "device")]
mod copy;
mod dag;
#[cfg(all(test, feature = "device-tests"))]
mod device_tests;
mod graph;
#[cfg(test)]
mod graph_tests;
mod instance;
mod partition;
mod task;
#[cfg(test)]
mod tests;
mod width;
#[cfg(test)]
mod width_tests;

pub use dag::{Dag, Plan};
pub use graph::Graph;
pub use instance::Instance;
pub use partition::Partition;
pub use task::{CpuTask, Decision, GpuTask};
pub use width::{Adaptive, widest};

/// The bytes of device memory each worker's BLAS handle is handed as its
/// workspace ([`Dag::blas_workspace`]): 32 MiB, cuBLAS's own default on
/// sm_90, the largest it allocates for itself on any architecture (4 MiB
/// below sm_90). Too little makes cuBLAS pick slower kernels rather than
/// fail.
pub const BLAS_WORKSPACE_BYTES: usize = 32 << 20;
