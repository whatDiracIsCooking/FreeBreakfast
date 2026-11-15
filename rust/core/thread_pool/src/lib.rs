//! The run's workers: a fixed set of threads that executes a placed DAG
//! (`docs/programming_model.md`, "CPU workers").
//!
//! No C++ counterpart: the C++ pipeline runs its steps one at a time on one
//! thread. This is the executor half of the orchestrator; choosing the order
//! and the placement is the planner's, and arrives here already done.
//!
//! - [`Plan`]: the nodes, each with its predecessors, its [`Placement`] (the
//!   CPU pool or one lane of a GPU worker, a [`LaneId`]) and the callable
//!   that runs it. Nodes are
//!   added in topological order, so a node's [`NodeId`] is its rank and a
//!   plan cannot have a cycle; and its loops, each a run of nodes, its
//!   body, replayed on the same workers: until a CPU decision node after
//!   it stops it, at most a bound fixed at planning ([`Plan::add_loop`]),
//!   or a trip count fixed at planning with up to a depth of iterations in
//!   flight and no decision ([`Plan::add_counted`]).
//! - [`Backend`]: the device side of a run: the GPU workers' lanes, the
//!   events at the edges that leave one, and the syncs. [`HostOnly`] has no
//!   device and is what the device-free tests run on; `Device` (the
//!   `device` feature) is the real one, over `fb_core_wwr::runtime`: each
//!   lane's stream and its library handles ([`DeviceStream`], a BLAS and a
//!   dense-solver handle), made with the backend before any run and reached
//!   by a GPU node through its [`Lane`]; and each worker's fork streams,
//!   outside its lanes, over which a node forks its own work and joins it
//!   back before it returns (`DeviceStream::fork_join`).
//! - [`run`]: open one [`std::thread::scope`] with `n_cpu` CPU workers and
//!   one thread per GPU worker, run every node once (a loop's body and
//!   decision once per iteration, a counted loop's body once per trip),
//!   join, and return
//!   the [`Record`]: the result, a [`TaskTime`] per node that ran and a
//!   [`SyncPoint`] per host wait, and as `timing.json` its
//!   [`Record::to_json`].
//!
//! # What a run promises
//!
//! 1. Every node runs exactly once, unless one of its ancestors failed, in
//!    which case it never runs. A loop's body nodes and decision run once
//!    per iteration, and the next iteration's start only after the
//!    decision returns `Continue`, so once the whole body has completed. A
//!    counted loop's body nodes run once per trip, trip `k` after trip
//!    `k - depth`.
//! 2. No node starts before each predecessor has finished in the sense its
//!    edge needs (the table below).
//! 3. At most `n_cpu` CPU nodes run at once, and each GPU worker runs one
//!    node at a time, its ready nodes in rank order across all its lanes.
//!    Running a GPU node is enqueueing it: its lanes' work overlaps on the
//!    card, and the worker's thread never waits on the device until it
//!    exits.
//! 4. The run's threads are `n_cpu` plus one per GPU worker, whatever its
//!    lanes, created when
//!    the run opens and joined before [`run`] returns. Nothing else spawns:
//!    `clippy.toml` disallows the spawning functions everywhere but here.
//!
//! # One run at a time
//!
//! [`run`] takes the plan and the backend by `&mut`, so the borrow checker
//! holds what the run model needs: neither is in two runs at once. A
//! `Device` backend's events are its plan's nodes', recorded and waited on
//! within one run, and its streams and library handles are each one lane's;
//! and a plan's nodes reach what their closures captured, which
//! for a planner instance is its blocks, whose rows' borrow flags are
//! released when a node returns while its device work may still be in
//! flight. Two runs sharing either would race on both. The two kinds of
//! concurrency the model has keep working: within one run the workers share
//! the plan and the backend by reference (which is why [`Backend`] is
//! `Sync`), and independent (plan, backend) pairs run on threads of their
//! own. Several GPUs are one run: a `Device` spans its GPU workers'
//! devices.
//!
//! # When an edge is satisfied
//!
//! A GPU node *returns* once it has enqueued its work; that work *completes*
//! later, on its stream. The executor tracks returns, and the backend makes
//! up the difference per edge:
//!
//! | Producer → consumer | Before the consumer runs |
//! |---|---|
//! | lane `l` → lane `l` | nothing: the stream's order |
//! | lane `l` → lane `m ≠ l`, of the same worker or another | `m` waits on the producer's event on the device; no host wait |
//! | lane → CPU | the CPU worker waits on the producer's event: a sync point |
//! | CPU → anything | nothing: the host work is done when it returns |
//!
//! A row read across two lanes of one worker is read in place, the
//! pointer the producer wrote: the worker's blocks are its lanes'.
//!
//! A CPU node that waits on the device holds its CPU worker while it waits.
//!
//! # Loops
//!
//! A loop is a sub-DAG replayed between host decisions
//! (`docs/programming_model.md`, "Plan once, run many"). Its decision is a
//! CPU node after the body's sinks, so after every body node, and after
//! any node before the body it is given, so a CPU worker has waited on the
//! host for the sinks' work, which every GPU body node's precedes on the
//! device, before it runs: the body is quiescent,
//! its events complete, when the decision readies it again. The replay
//! happens under the run's one lock, in the same scope, on the same
//! workers: nothing is spawned and nothing allocated but the record's
//! entries. A node after the loop depends on the decision, and is ready
//! only when the decision returns `Break`. The bound is the plan's: a
//! decision that would start iteration `max_iterations` fails the run
//! with [`MaxIterationsExceeded`](fb_core_error_handling::ErrorKind::MaxIterationsExceeded).
//!
//! A counted loop ([`Plan::add_counted`]) has no decision: its body runs a
//! trip count fixed at planning, each trip a run of instances of the body's
//! nodes with edges of its own, the body's within the trip, each node's
//! from itself in the trip before, and the ring, from the trip `depth`
//! back. The run schedules instances, a node in one trip, in rank order
//! with the body unrolled trip by trip, so a GPU worker enqueues the trips
//! back to back and nothing waits on the host between them. Each node is
//! handed its trip ([`Lane`]), and a GPU node has events per trip, so its
//! consumers wait on its own trip's.
//!
//! A counted loop may lie in a decision loop's body, the batches of a
//! streamed tensor reduced every SCF iteration: its instances unroll
//! within the decision loop's run of them, so the decision's replay
//! readies the counted loop whole, its trips overlapped again with no
//! host wait between them. A counted loop holds no loop, and a decision
//! loop no decision loop.
//!
//! # Failure
//!
//! A node's `Err` gets `name@worker` pushed onto its
//! [`context`](fb_core_error_handling::Error::context), and every
//! descendant is skipped; siblings already running run on. [`run`] returns the
//! failure lowest in rank, so the same failure reports the same way at every
//! worker count, with any others pushed onto its context as `also: ...`. Each
//! GPU worker syncs each lane it ran on before it exits, so a run never returns
//! with work outstanding, and a fault surfacing there is reported against the
//! lane's last node.
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
//!   when each GPU worker has synced its lanes on the way out.
//! - **A wall-clock scope around every sync point**: a CPU node's
//!   [`Backend::host_wait`] on each GPU predecessor, and each GPU
//!   worker's [`Backend::sync`] of each lane as it exits, each naming the
//!   node it waited for.
//! - **The run's wall time**, from the workers' start to their join.
//!
//! A loop's body nodes and decision have an entry per iteration, each with
//! its iteration's index ([`TaskTime::iteration`]), and the record carries
//! each decision loop's iteration count ([`Record::iterations`]). A body's
//! stream scopes are reused every iteration, so its decision reads them,
//! after the host waits that make the body complete and before it
//! replays: a sync point the loop has anyway, so still no scope adds one.
//! A scope that cannot be read there fails the decision. A counted loop's
//! body node has an event pair per trip, each read after the run like any
//! other node's, and an entry per trip; inside a decision loop, the
//! decision reads each trip's before the replay, and the node has an entry
//! per iteration and trip, the iteration as [`TaskTime::outer`].
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
//! A decision loop inside a decision loop's body, a loop inside a counted
//! loop's, and a node after a decision loop reading a body node directly:
//! each is refused by contract. `scope!` inside a node and the
//! sync-after-every-task switch are `diagnostics.md`'s and not here.
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
//! let b = plan.add("b", Placement::Stream(0.into()), &[a], |_| {
//!     total.fetch_add(10, Ordering::Relaxed);
//!     Ok(())
//! });
//! plan.add("c", Placement::Cpu, &[a, b], |_| {
//!     assert_eq!(total.load(Ordering::Relaxed), 11);
//!     Ok(())
//! });
//! let record = run(&mut plan, &mut HostOnly::new(1), 2);
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
pub use plan::{Lane, LaneId, NodeId, Placement, Plan};
pub use run::{ArenaSize, Record, SyncPoint, TaskTime, default_n_cpu, run};
