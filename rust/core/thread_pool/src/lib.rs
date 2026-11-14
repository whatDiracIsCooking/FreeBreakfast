//! The run's workers: a fixed set of threads that executes a placed DAG
//! (`docs/programming_model.md`, "CPU workers").
//!
//! No C++ counterpart: the C++ pipeline runs its steps one at a time on one
//! thread. This is the executor half of the orchestrator; choosing the order
//! and the placement is the planner's, and arrives here already done.
//!
//! - [`Plan`]: the nodes, each with its predecessors, its [`Placement`] (the
//!   CPU pool or one stream worker) and the callable that runs it. Nodes are
//!   added in topological order, so a node's [`NodeId`] is its rank and a
//!   plan cannot have a cycle.
//! - [`Backend`]: the device side of a run: the stream workers' streams, the
//!   events at the edges that leave one, and the syncs. [`HostOnly`] has no
//!   device and is what the device-free tests run on; `Device` (the
//!   `device` feature) is the real one, over `fb_core_wwr::runtime`: each
//!   stream worker's stream and its library handles ([`DeviceStream`], a
//!   BLAS and a dense-solver handle), made with the backend before any run
//!   and reached by a GPU node through its [`Lane`].
//! - [`run`]: open one [`std::thread::scope`] with `n_cpu` CPU workers and
//!   one thread per stream worker, run every node once, join, and return
//!   the [`Record`]: the result, a [`TaskTime`] per node that ran and a
//!   [`SyncPoint`] per host wait, and as `timing.json` its
//!   [`Record::to_json`].
//!
//! # What a run promises
//!
//! 1. Every node runs exactly once, unless one of its ancestors failed, in
//!    which case it never runs.
//! 2. No node starts before each predecessor has finished in the sense its
//!    edge needs (the table below).
//! 3. At most `n_cpu` CPU nodes run at once, and each stream worker runs one
//!    node at a time, its ready nodes in rank order.
//! 4. The run's threads are `n_cpu` plus one per stream worker, created when
//!    the run opens and joined before [`run`] returns. Nothing else spawns:
//!    `clippy.toml` disallows the spawning functions everywhere but here.
//!
//! # When an edge is satisfied
//!
//! A GPU node *returns* once it has enqueued its work; that work *completes*
//! later, on its stream. The executor tracks returns, and the backend makes
//! up the difference per edge:
//!
//! | Producer → consumer | Before the consumer runs |
//! |---|---|
//! | stream `s` → stream `s` | nothing: the stream's order |
//! | stream `s` → stream `t ≠ s` | `t` waits on the producer's event; no host wait |
//! | stream → CPU | the CPU worker waits on the producer's event: a sync point |
//! | CPU → anything | nothing: the host work is done when it returns |
//!
//! A CPU node that waits on the device holds its CPU worker while it waits.
//!
//! # Failure
//!
//! A node's `Err` gets `name@worker` pushed onto its
//! [`context`](fb_core_error_handling::Error::context), and every
//! descendant is skipped; siblings already running run on. [`run`] returns the
//! failure lowest in rank, so the same failure reports the same way at every
//! worker count, with any others pushed onto its context as `also: ...`. Each
//! stream worker syncs its stream before it exits, so a run never returns
//! with work outstanding, and a fault surfacing there is reported against the
//! stream's last node.
//!
//! **A panic fails the run rather than hanging it.** A worker that panics
//! mid-node never marks the node finished, and the other workers would wait
//! for it forever, so the scope would never join. Each worker holds a guard
//! that, while unwinding, marks the run aborted and wakes every worker; they
//! exit, the scope joins and re-raises the panic. A release build aborts on
//! the panic first.
//!
//! # Timing
//!
//! The executor times every node, and the node does nothing for it
//! (`docs/diagnostics.md`, "What the orchestrator times on its own"):
//!
//! - **A wall-clock scope around every node's callable**, not the waits
//!   before it: a CPU node's compute, a GPU node's enqueueing.
//! - **A stream scope around every GPU node**: [`Backend::begin`] records an
//!   event on its stream after the waits and before it runs, and the event
//!   [`Backend::record`] records after it returns, which its consumers wait
//!   on, closes the scope. No scope syncs: they are read after the run,
//!   when each stream worker has synced its stream on the way out.
//! - **A wall-clock scope around every sync point**: a CPU node's
//!   [`Backend::host_wait`] on each GPU predecessor, and each stream
//!   worker's [`Backend::sync`] as it exits, each naming the node it
//!   waited for.
//! - **The run's wall time**, from the workers' start to their join.
//!
//! The record also carries each node's dimensions ([`Plan::set_dims`]) and
//! the profile that built the executor; the plan's arena sizes
//! ([`ArenaSize`]) are the planner's to add.
//!
//! A failed run still returns its nodes' times: a failed node has an entry
//! (with no device time), a skipped one has none. A stream scope that
//! cannot be read fails a run that otherwise succeeded, and is dropped from
//! one that already failed, since it echoes that failure.
//!
//! # Logging
//!
//! Each node runs inside a [`TaskScope`](fb_core_log::TaskScope), so its
//! log lines read `[name@worker]`. The executor logs a run's open and close
//! and each node's start and end at `debug`, and nothing above it: a failure
//! is returned, and `main` logs it once.
//!
//! # Not yet
//!
//! Replaying a sub-DAG on the run's workers (an SCF iteration) is not here.
//! The run ends when no node remains, not when a list is exhausted, so a
//! decision node can later enqueue the next iteration's nodes without a
//! second scope. `scope!` inside a node and the sync-after-every-task switch
//! are `diagnostics.md`'s and not here.
//!
//! ```
//! use std::sync::atomic::{AtomicU32, Ordering};
//! use fb_core_thread_pool::{HostOnly, Placement, Plan, run};
//!
//! let total = AtomicU32::new(0);
//! let mut plan = Plan::new();
//! let a = plan.add("a", Placement::Cpu, &[], |_| {
//!     total.fetch_add(1, Ordering::Relaxed);
//!     Ok(())
//! });
//! let b = plan.add("b", Placement::Stream(0), &[a], |_| {
//!     total.fetch_add(10, Ordering::Relaxed);
//!     Ok(())
//! });
//! plan.add("c", Placement::Cpu, &[a, b], |_| {
//!     assert_eq!(total.load(Ordering::Relaxed), 11);
//!     Ok(())
//! });
//! let record = run(&plan, &HostOnly::new(1), 2);
//! record.result.unwrap();
//! assert_eq!(record.tasks.len(), 3);
//! ```

mod backend;
mod plan;
mod run;
#[cfg(test)]
mod tests;

#[cfg(feature = "device")]
pub use backend::Device;
pub use backend::{Backend, DeviceStream, HostOnly};
pub use plan::{Lane, NodeId, Placement, Plan};
pub use run::{ArenaSize, Record, SyncPoint, TaskTime, default_n_cpu, run};
