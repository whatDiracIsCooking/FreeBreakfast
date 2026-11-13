//! The planner: tasks declared once, turned into the arena's tables and the
//! executor's plan (`docs/programming_model.md`, "Three owners").
//!
//! No C++ counterpart: the C++ pipeline's steps allocate as they go and
//! run one at a time. This is the glue between [`arena`](crate::core::arena)
//! and [`thread_pool`](crate::core::thread_pool); without it a caller wrote
//! each row's position, each node's dependencies and each `partition` call
//! by hand.
//!
//! - [`Task`]: `layout`, which declares every row the task holds or reads,
//!   and `run`, which computes through them.
//! - [`Dag`]: tasks added in plan order, each on a GPU worker, on its
//!   stream or the CPU pool. [`Dag::add`] calls the layout at once and
//!   returns the task's rows, whose output ids the caller passes to the
//!   tasks that read them.
//! - [`Planned`] ([`Dag::plan`]): each worker's device and host tables laid
//!   out, and each task's dependencies derived, all pure. A task runs after
//!   the producer of every row it reads ([`Table::reads`]) and after every
//!   user of the bytes it takes over ([`Table::handoffs`]). Admission is
//!   [`admit`](crate::core::arena::admit) on its tables; its text is the
//!   plan's golden.
//! - [`Instance`] ([`Planned::instantiate`]): the blocks, and a
//!   [`thread_pool::Plan`] whose node for each task partitions the blocks
//!   for it and calls its `run`. Run any number of times.
//!
//! [`Table::reads`]: crate::core::arena::Table::reads
//! [`Table::handoffs`]: crate::core::arena::Table::handoffs
//! [`thread_pool::Plan`]: crate::core::thread_pool::Plan
//!
//! # What a plan promises
//!
//! Whatever order the executor picks among ready tasks: a reader starts
//! after its producer has finished in the sense the edge needs (the
//! executor's "When an edge is satisfied"), and a row's bytes pass to the
//! next row only after every user of the previous one. So no task reads a
//! row before it is written, and no two tasks hold the same bytes at once;
//! the arena's borrow flags would panic if one did.
//!
//! This module has no `unsafe`: the views are what make sharing the blocks
//! across workers sound.
//!
//! ```
//! use fb_rebuild::core::arena::{Bufs, Device, Layout, Pinned, RowId};
//! use fb_rebuild::core::error_handling::Status;
//! use fb_rebuild::core::planner::{Dag, Task};
//! use fb_rebuild::core::thread_pool::{DeviceStream, Lane};
//!
//! /// Writes an `n`-element output, reading `input` if it has one.
//! struct Step {
//!     n: usize,
//!     input: Option<RowId<Device, f64>>,
//! }
//!
//! impl Task for Step {
//!     type Rows = RowId<Device, f64>;
//!     const GPU: bool = true;
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
//!         _: &Bufs<'_, Pinned>,
//!         _: Lane<'_, DeviceStream>,
//!     ) -> Status {
//!         Ok(()) // enqueue on the lane's stream
//!     }
//! }
//!
//! let n = 24 * 24;
//! let mut dag = Dag::new();
//! let s = dag.add("S", 0, Step { n, input: None });
//! let x = dag.add("X", 0, Step { n, input: Some(s) });
//! let f = dag.add("F", 0, Step { n, input: Some(x) });
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
//! is the order of [`Dag::add`]: no placement or schedule search. A task
//! reads only rows of its own worker; an edge between workers needs a
//! transport row, which placement would add. No partition or reduce: every
//! edge is one producer's output. No adaptive widths, nested plans,
//! replaying a sub-DAG between host decisions, or stream scopes. Every
//! dependency is a row: an ordering with no buffer behind it has no way to
//! be declared until a task needs one. That a stream task does not touch a
//! pinned row from the host is a rule, not a type.

#![forbid(unsafe_code)]

mod dag;
mod instance;
mod task;
#[cfg(test)]
mod tests;

pub use dag::{Dag, Planned};
pub use instance::Instance;
pub use task::Task;
