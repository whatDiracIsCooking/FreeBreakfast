# Rust rebuild: programming model

We design a feature as a DAG of tasks before we write it as code. A node is a
task. An edge is data that one task produces and another consumes. The
orchestrator admits, places, schedules and joins the nodes. The tasks only
compute.

The model is not new to this tree. The C++ pipeline is already an instance of
it, in its most restricted form, and the first section says exactly which
restrictions the Rust rebuild lifts. The rest of the document is the contract:
what a task declares, what an edge declares, what the orchestrator derives, and
how one DAG runs unchanged on one GPU, on a node of GPUs, and across nodes.

## The pipeline is already an instance

A C++ pipeline step declares its memory footprint as a formula in the plan's
dimensions, `footprint(dims)`. The plan fold, `then` over the steps in plan
order with `drop` for what each one frees, finds the peak and the step that
reaches it, and `check_plan` refuses the plan against the `[memory]` caps
before the first step runs
([`src/pipeline/footprint`](../../src/pipeline/footprint/README.md)).
Steps run one at a time, on one device.

Two things are fixed there that the model leaves free:

- **The schedule is a list.** The plan order is the config's order. Formally
  the steps are a DAG: the overlap and the core Hamiltonian share no edge,
  and either may run first. A step is itself a sub-DAG, and some of those are
  replayed: the RHF step runs one iteration's DAG until convergence.
- **The placement is one device.** Every step runs on the one card the plan
  was given.

The rebuild keeps the contract and relaxes both: the list becomes a DAG, and
the device becomes a topology. That relaxation also says what the
orchestrator's first job is, on one GPU: not running nodes in parallel, but
finding an order whose peak fits the budget. Admission and scheduling are one
problem, not two. Parallelism is whatever room the chosen order leaves.

## Three owners

The things one has to decide about a feature belong to three owners. A task
answers only for itself; an edge answers for the data between two tasks; the
orchestrator derives the rest. Keeping them apart is what stops two tasks from
disagreeing about the edge between them, and keeps each task's spec short.

### A task declares

1. **Device.** CPU or GPU.
2. **Layout.** Every buffer it will hold, as a function of its dimensions
   (`nbf`, `naux`, `nocc`, ...): each one's size, and whether it is scratch,
   freed when the task ends, or an output that outlives it. The footprint is
   derived from the layout; nobody writes it. A kernel's share comes from its
   `*_workspace_bytes(dims)` query ([`gpu_layer.md`](gpu_layer.md)), so the
   sizes are numbers the kernels own rather than guesses.
   "Memory: layouts and arenas" below has the whole of it.
3. **Boundary.** Its inputs, its outputs, and what it promises not to touch.
4. **Partition**, optionally. An axis along which its work can be sliced over
   workers, and the reduction that recombines the partials. A task that
   declares none still runs anywhere; it just cannot be split. Most tasks
   declare none.

A task also obeys three rules, which are what make it the unit of admission,
timing, testing and retry:

- **It allocates nothing of its own.** Its buffers are carved from an arena
  the orchestrator planned from its layout, and there is no handle through
  which a task could allocate anything else. A task asking for a buffer its
  layout did not name is a contract failure, not a surprise at runtime.
- **It has no internal host sync.** It enqueues work on the stream it was
  given and returns. If part of a computation must wait for another part,
  that wait is an edge, and the computation is two tasks.
- **Retry means re-admission.** A failed GPU allocation aborts in this tree.
  The place a task can be refused and tried again later is admission, before
  anything is launched, and nowhere else.

### An edge declares

1. **Size.** Its bytes, as a formula in the same dimensions. An edge is a
   known number of bytes, which is the one requirement for it to cross any
   boundary: a pointer within a GPU, a copy across cards, a transfer across
   nodes.
2. **Kind.** Broadcast (one producer, every consumer reads), reduce (every
   producer contributes, one consumer reads the combination), gather, or
   point-to-point.
3. **Reduction order**, for a reduce. Ordered, with a fixed combination
   order and so a bitwise-reproducible result at any worker count, or
   unordered, with a declared tolerance. The same J task reduced over one
   slice, four streams or sixteen GPUs gives three results if the reduce is
   unordered, and the acceptance rule in this tree fixes its bars by analysis.
4. **Lifetime.** An edge's buffer lives from its producer's start to its last
   consumer's end. With the tasks' layouts, that is everything the arena
   needs to lay a DAG out.

An edge never says which level of the worker hierarchy it crosses. That is
placement, and placement is the orchestrator's.

### The orchestrator derives

- **Fork and sync points**, from the DAG's shape. A node with several
  successors is a fork; a node with several predecessors is a join.
- **Concurrency**, from the shape too. Two nodes with no path between them
  may run at once, as may a transfer and the compute it does not feed.
- **Placement**: which worker each node, or each slice of a partitioned node,
  lands on.
- **Admission**: a schedule, and an arena laid out over every layout and
  edge lifetime under that schedule, that fits each worker's budget.
- **Transport**: what an edge costs given where its two ends landed, and the
  mechanism that moves it.
- **Error propagation**: a failed node fails its successors and nothing else.

## Memory: layouts and arenas

The C++ tree's footprints
([`src/core/footprint`](../../src/core/footprint/README.md),
[`src/pipeline/footprint`](../../src/pipeline/footprint/README.md))
were hard to write, and the reason is worth naming, because the model is
built to remove it. A footprint there is a second program about allocation:
the routine allocates, and beside it a hand-written formula describes what
it allocated, held in step with it only by tests against the measured pool
peak. Every difficulty follows from that. Scratch lifetimes are implicit in
where the code happens to free, so the `then` / `drop` algebra reconstructs
them. The pool is dynamic, so a fragmentation headroom is left. Adaptive
widths are chosen at run time from free memory, so the formula can only
state the least. Nested pipelines fold inside outer steps. The shared
`sizes` struct that keeps a routine and its footprint agreeing is a
suggestion, not a contract.

This section and the three after it are the principles. The level below,
the table's rows and their owners, the lifetime order, the admission
arithmetic, the adaptive-width routine and the two blocks' raw calls, is
[`memory.md`](memory.md).

The rebuild makes allocation and footprint one program, in three moves.

**A task is two functions.** `layout(dims)` returns the buffers the task
will hold: each with a name, a size in bytes and a lifetime (scratch or
output), the kernel workspace queries and the solver's `lwork` queries called
here, at planning, as the pipeline already treats `syevd_lwork` as a
dimension, and a BLAS handle's workspace among them
([`gpu_layer.md`](gpu_layer.md), "Layering"). `run(bufs, stream)` computes,
and reaches a buffer only through the layout it declared. The footprint is
read off the layout by the orchestrator and never written by the author. The
layout is what admission reads, what a tier-3 test checks with no device, and
what the arena is built from. In the crate the two are `core::planner`'s
`Task`: the task's value is its dimensions, so `dims` is `&self`, and
`layout` declares through one `Layout` per memory kind, reading an earlier
task's output by the row id that task's layout returned. `run` is handed
those rows, one `Bufs` per kind, and the lane it runs on.

**Allocation is a capability, held by one module.** The raw allocation
bindings in `wwr-sys` are `unsafe`, one arena module wraps them, and a
task's `run` receives its buffers already carved. There is no handle a task
could allocate through, so "a task allocates nothing of its own" is a
compile error rather than a convention. The contract tier's check becomes a
`debug_requires` on the arena: a task may ask only for a buffer its layout
named.

**The orchestrator lays an arena out at planning.** Every buffer's size and
lifetime is known once the schedule is chosen, so assigning offsets in one
block per worker is an interval-overlap problem solved once, the memory
planning a tensor compiler does. What that replaces, from the C++:

| C++ today | Under an arena |
|---|---|
| `then` / `drop` and the peak model | The arena's size. The peak is not estimated; it is the plan. |
| Footprint tests against the measured pool peak | Nothing to measure. The run performs no allocation, so measured equals declared by construction. |
| The 64 MiB fragmentation headroom | None. There is no dynamic pool to fragment. |
| An adaptive routine's width chosen at run time from free memory | Chosen at planning: `layout` takes the room the schedule leaves and returns the width. Deterministic, and known before the first run. |
| A nested pipeline's footprint folded into the outer step's | A sub-DAG's arena is carved from its parent's. Outputs that outlive a task live in the parent's layout. |
| `Footprint<Device>` and `Footprint<Pinned>` | Two arenas, one per memory kind, and the kind stays in the type. |

Plan-once-run-many is what makes this physical: one arena per planned DAG,
carved once, reused every run. An iteration, a geometry or a displacement
allocates nothing.

**Where offsets may be reused.** Two buffers may share an offset only when
the schedule guarantees their lifetimes do not overlap. On one stream that is
the stream's order. Across streams it is a path in the DAG from the last use
of one to the first use of the other, so offset reuse is a scheduling
decision, which is fine because the planner owns both.

**Sizes the dimensions do not fix.** A screened shell-pair count, a Cholesky
rank or a Davidson subspace's final size is data-dependent and unknown at
planning. There are two honest answers, and the model allows both: a bound
from the dimensions, which is the pipeline's answer today (worst-case `ncas`,
the Davidson's largest subspace), and a host sizing node that fixes a
sub-DAG's dimensions before that sub-DAG is planned. The bound is preferred;
the sizing node is for the case where the bound is too loose to admit.

### The arena is a table and a block

Two pieces, and the first is the one with the content.

**The offset assigner is a pure planner function.** Its input is every
buffer in the planned DAG with its size, its alignment and its lifetime, an
interval of the schedule on one stream or a DAG-path relation across streams.
Its output is an offset per buffer and the block's total. Greedy in start
order with best-fit placement, the static memory planning a tensor compiler
does, is enough for the DAGs this tree has; the general problem is hard, and
"The schedule search" under the open questions says when greedy may not be.
It touches no device, so it is tier 3
([`testing.md`](testing.md), "Tiers: where a check lives"), it is fuzzed as a
pure function ([`testing.md`](testing.md), "Fuzzing"), and its total is the
committed golden ([`testing.md`](testing.md), "Regression").

**The block is one allocation plus that table.** One contiguous allocation
per GPU worker per memory kind, made when the plan is instantiated and freed
when the plan drops, and a `buffer(id)` accessor that returns a slice of the
length the layout declared, under the debug contract that the id is in the
calling task's layout. It is the one module that calls the raw allocation
bindings ("Capabilities are enforced by lints"), and the call is the plain
device allocation, not the stream-ordered pool the crate also wraps: one
allocation per plan has nothing for a pool to amortise, and the crate's
README records `hipMallocFromPoolAsync` deadlocking beside other threads'
stream work on ROCm 7.2.4, a hazard a plain allocation made before any
stream work never meets. It is not an allocator.
There is no allocate or free at run time, so there are no size classes, no
free lists, no coalescing and no compaction, and nothing for a pool library
to speed up. A pool is the right tool when sizes arrive in an order nobody
knows; here every size and lifetime is known before the first run, and the
headroom and the fragmentation a pool carries are what the table above
removes.

### Blocks, plans and the process

Three lifetimes, each owning the next.

| | What it is | How many | Lives |
|---|---|---|---|
| **Process** | The running binary: one config, one logger, the devices, the worker budget | One | `main` to exit |
| **Plan** | A DAG with its dimensions fixed, scheduled, placed and admitted: the offset tables, the worker assignment, the sync points | One per calculation; the pipeline's, with its steps' and their iterations' nested inside | Planning to drop, run any number of times between |
| **Block** | One contiguous allocation of one memory kind on one GPU worker, sized to the plan's peak for that worker and kind | Per top-level plan, one per GPU worker it uses, per kind | The plan's instantiation to its drop |

A run is a plan being executed. It owns nothing: it borrows the plan's
blocks and workers for its duration and allocates nothing. The process owns
the devices, the budgets the planner admits against and the threads the
orchestrator opens around a run, and no memory of its own. There is no
process-wide pool, because a process block could be sized only two ways:
by knowing every plan it will serve before allocating, which is a plan of
plans, and that is the pipeline plan; or by allocating a guess and growing
when a plan does not fit, which is a pool, with the reallocation and the
headroom the arena exists to remove. The config fixes the whole calculation
up front, so the first way is available and is the model as written.

**A sub-DAG has a table but no block.** Its offsets are laid out inside its
parent's block, in the region the parent's schedule leaves free while the
sub-DAG runs, and the outputs that outlive it are in the parent's table.
Only top-level plans allocate. The semi-numerical Hessian's pinned block is
the top-level plan's, sized to one gradient sub-DAG's peak once, not
`6 natoms` times.

**The peak is a maximum over the schedule, not a sum.** A block equals its
plan's peak, the largest total of simultaneously live buffers under the
schedule, and every run reaches it at some instant, so no design runs the
plan in less. The SCF step's scratch and the gradient step's alias in time
inside the pipeline plan because their lifetimes are disjoint, which is the
process-wide sharing a pool would do by accident, done on purpose. Two
top-level plans alive at once hold their sum; the answer is to make them one
plan, and nesting is how. A plan kept alive after its last run holds its
block while something else runs, and the answer there is that plans are
values: drop the finished one.

**Replay allocates nothing.** The coordinates are an edge, not a dimension.
A geometry optimization, an IRC walk or a molecular-dynamics trajectory is
one plan run once per geometry, with the same offsets and the same pages,
and its two allocations are at the start and the end of the calculation.
What varies with geometry, a screened shell-pair count or a Cholesky rank,
takes the bound from the dimensions ("Sizes the dimensions do not fix"), so
memory sits idle at most geometries in exchange for a trajectory that never
calls the driver, which is the right trade inside a loop. Even a sizing node
that re-plans a sub-DAG per run allocates nothing: the parent reserves the
sub-DAG's region at its bound, and re-planning recomputes a table inside it.

**Concurrency is a width the planner admits.** Two sub-DAGs in flight cannot
alias, so each one adds its peak to the parent's block. How many of the
independent sub-DAGs run at once, the displacements, the fragments, the
roots, is the smaller of what memory admits and what the workers allow, from
one up, and both numbers are the plan's. Serializing is how a plan fits;
parallelism is the room the chosen order leaves, as the first section says.
The reduce at the join combines the partials in the declared order whatever
the width, so the width is a planning decision and not a correctness one.

### Memory kinds

| Kind | Owner | Admitted against | How it is held |
|---|---|---|---|
| Device | The GPU worker | The per-GPU device cap | One block per plan |
| Pinned host | The GPU worker whose copies it stages | The per-GPU pinned cap, and the sum over a node's workers within the host's available memory | One block per plan, allocated portable |
| Pageable host | Nobody; the operating system's | Not modelled | `Vec`, owned by the plan when it is part of the DAG |

**Pinned memory is per GPU.** Physically it is the node's, page-locked host
RAM any card can read, and the C ABI's portable flag is set on both backends
so that fact does not depend on the unified address space. But every pinned
buffer exists to feed one GPU's copy engines, and its lifetime is bounded by
a copy on that GPU's stream, so its owner is that GPU worker, as the C++
tree's `DeviceResources` already has it. A GPU worker is then one
self-contained unit, its streams, its device block and its pinned block,
admitted together against the per-GPU caps, and NUMA locality comes free
from allocating the block on the worker's own thread. The node is a cap and
not an owner: the sum of a node's pinned blocks must fit the host's memory
less a margin, which is the C++ `check_pinned_node` check carried over. Host
data broadcast to several cards is placed in one consumer's pinned block and
read across, which the portable flag makes legal and the "GPUs on one node"
row of the worker table prices, or staged once per card; both are placement.
Pinning costs milliseconds and locks pages, so once per plan, reused every
run, is where plan-once-run-many pays most.

**Pageable host memory is not an arena.** The arena exists for memory that
is admitted against a budget and whose peak must be a planned number.
Pageable host memory is neither: the config, the basis, the planner's own
tables and a file writer's buffer are the operating system's business, and
`Vec` is the right tool. Making an arena the backend for `Vec` means the
unstable allocator API or a third-party `Vec` across every host type, for no
property the model asks for. Plan-once-run-many gives the host side what
matters without an allocator: a host buffer that is part of the DAG is owned
by the plan, allocated when the plan is built and reused every run. The C++
footprint's rule carries over unchanged: large host-side data is pinned,
where it is counted, and a step that would move something big to pageable
memory moves it to pinned memory instead.

**What no arena reaches.** A vendor routine that allocates internally, as
some solver paths do whatever workspace they are handed, is outside every
arena. The answer is the one the C++ tree uses, the `lwork` and
`*_workspace_bytes` queries so the routine has no reason to allocate, and
the memory golden is what notices when one does anyway.

## Plan once, run many

The replayed computations in this tree share one property: their dimensions
do not change between runs. An SCF iteration, the gradient at each geometry of
an optimization or an IRC walk, and each displacement of a semi-numerical
Hessian all reuse the same basis, so the same `nbf` and `naux`. The model
makes that a first-class split:

- **Planning** fixes the dimensions, evaluates every footprint and edge size,
  chooses a placement and a schedule, and admits the whole against the
  budget. It happens once.
- **Running** moves data through the planned DAG. It happens any number of
  times, and nothing is re-admitted.

**One run at a time, enforced by types.** `Instance::run` takes the
instance and the backend by `&mut`, as `fb_core_thread_pool::run` takes
the plan and the backend, so neither is ever in two runs at once. The run
model needs exactly that: a block's row borrow flags are host-side and a
GPU task's rows are returned while its device work may still be in flight,
and a `Device` backend's events are one plan's nodes' and its streams and
library handles one run's workers'. The concurrency the model does have
keeps working. Within a run the workers share the instance's plan and the
backend by reference, through a wrapper only `run` can make (a `Backend`
is sealed and need not be `Sync`, and a `Device` is not, so outside a run
no backend reaches a second thread at all); independent
(instance, backend) pairs run on threads of their own; and several GPUs
are one run, a `Device` spanning its stream workers' devices.

Stream capture into a replayed graph is the same split one level down, which
is a sign the split is real rather than imposed. A loop is therefore not a
special node: it is a planned sub-DAG run repeatedly, with a host decision
between runs. The decision is a node (convergence, a step accepted, the walk
reaching a minimum) whose output is read on the host.

**The executor replays the body in the run it is in.** In
`fb_core_thread_pool`, a loop is its body, the nodes added after a given
one, and a decision node `Plan::add_loop` adds after the body's sinks, so
after every body node. The decision is a CPU node, so a worker has waited
on the host for every GPU body node before it runs: the body's work has
completed, and its rows and events are free to be used again, when the
decision returns. Its `Continue` readies the body again, under the run's
one lock, in the same scope and on the same workers, so a replay spawns no
thread and allocates nothing; its `Break` readies what follows the loop,
which depends on the decision and never on a body node. The decision is
handed the iteration's index. The iteration count has a bound, fixed at
planning beside the dimensions: a decision that would start one iteration
past it fails the run with `MaxIterationsExceeded`, and a failed body node
ends the loop and skips what follows it, as any failure skips its
descendants. A loop in a loop's body is not built.

**DAGs nest.** A node may be a sub-DAG with dimensions of its own. The
pipeline is a DAG of steps; a step is a DAG of tasks; the RHF step's body is a
sub-DAG replayed per iteration. Dimensions attach to the sub-DAG, not to the
whole plan: fragments of a many-body expansion each have their own `nbf`, and
the pipeline's one global dimensions struct is the one-calculation special
case.

## Two kinds of distribution

The topologies we want are a single GPU, a node of GPUs, and many nodes of
GPUs. A large coupled-cluster calculation would spread one task's work over
every GPU it can reach; a many-body expansion would spread independent
fragments over GPUs and nodes. These are two different kinds of
distribution, and only one of them costs a task's author anything.

- **Across sub-DAGs.** Fragments of an expansion, displacements of a
  semi-numerical Hessian, the roots of a state average: independent sub-DAGs,
  joined by a small reduce at the end (a sum of fragment energies, a Hessian
  assembled from gradients). The orchestrator gets this for free by placement.
  Nothing inside the tasks changes. How many run at once is the width the
  planner admits ("Blocks, plans and the process").
- **Within a task.** The amplitudes of a coupled-cluster iteration, the
  Coulomb build over slices of the shell-pair list: one task's work sliced
  over workers, a broadcast in and a reduce out. This is the partition a task
  may declare, and it is the only place distribution reaches into a task.

Keeping the partition opt-in is what keeps single-GPU code honest: a task
without one is written once, for one stream, and is still placeable on any
worker of any topology.

## Workers are a hierarchy

A worker is a thread, a stream, a GPU, or a node, and the levels nest. An
edge's kind and size are the same at every level; only the cost differs, and
the orchestrator prices it by where the two ends landed:

| Edge crosses | Broadcast is | Reduce is |
|---|---|---|
| Threads on one host | A shared pointer | An ordered accumulate into one buffer |
| Streams on one GPU | A shared pointer | An ordered accumulate into one buffer |
| GPUs on one node | A device-to-device copy per card | Partials copied to one card, or a peer reduce |
| Nodes | A network transfer | A network reduce |

The single-GPU case is the degenerate one: broadcast is a pointer, reduce is
an in-place accumulate, and the workers are streams. One DAG description
serves every row, which is the point of wanting all of them. The transport
behind the lower rows is an orchestrator detail, decided later and not here.

### CPU workers

A CPU node is a file write, a host decision between runs of a sub-DAG, or a
small host-side assembly. It runs on a CPU worker, which is one thread the
orchestrator owns. The needs here are meagre: the products written when a
step finishes, an intermediate tensor dumped for debugging, a convergence
check. The design is sized to them, and it is enforced rather than hoped for,
because the place a thread count gets out of hand is never the one that was
designed.

**A run owns a fixed set of threads.** At the start of a run the orchestrator
opens one `std::thread::scope` and creates `n_cpu` CPU workers and one thread
per stream worker inside it; the scope joins them all when the run ends. A
ready queue of node ids feeds the CPU workers, and a stream worker's thread
has its device made current for the life of the run
([`gpu_layer.md`](gpu_layer.md), "Layering"). No node is ever given a thread
of its own, so the thread count is a property of the plan, fixed at planning,
and not of the DAG's size. "Too many threads" is then impossible for the same
reason "too much memory" is once the arena is laid out: there is no counter to
check because there is no spawn in the loop.

**Sub-DAGs borrow their parent's workers.** This is where a count would blow
up otherwise. The semi-numerical Hessian is `6 natoms` gradient sub-DAGs, and
if each opened a scope of its own, one config line would make hundreds of
threads. The rule is the arena rule restated: a sub-DAG's arena is carved
from its parent's, and a sub-DAG's nodes are scheduled onto its parent's
workers. Only the top-level run owns threads. Nesting adds nodes to the
queue, never workers, and a loop re-adds its body's nodes to it, as the
thread-count test below checks across a replay.

**The count is a budget, admitted beside memory.** `n_cpu` defaults to the
host's `available_parallelism` less the stream workers' threads, floored at
one, and the config overrides it beside the `[memory]` caps. A placement that
needs more CPU workers than the budget is refused in the same admission pass
that refuses an arena that does not fit.

**Fan-out is nodes, not a pool inside a task.** Writing three files when the
RHF step finishes is three CPU nodes with no edge between them, each reading
an edge from the step; the orchestrator sees three ready nodes and runs them
at once. A task that opened a pool of its own to do the same would hide the
parallelism from the planner and would log from threads that carry no task
record ([`diagnostics.md`](diagnostics.md), "The task context"). One thread
runs one node, and the orchestrator sets the record on that thread before the
node starts, so the record needs no forwarding. A debugging dump in the middle
of an iteration has the same shape: a device-to-pinned copy node on the
stream, then a CPU write node that depends on it and feeds nothing. The next
iteration's GPU work has no edge to the write, so it proceeds while the write
runs, and the overlap costs the author nothing. The model makes two things
explicit that a hand-written dump would leave implicit: the pinned buffer is
in the layout, with its lifetime extended to the writer's end, so admission
counts it; and under plan-once-run-many a dump switched on is a different
plan, not a branch inside a task.

**The standard library, and no pool crate.** What the orchestrator needs is a
queue, a condition variable and the scoped threads, under a hundred lines,
most of them the loop that pops a node id and runs it. Shutdown is the
scope's join. A release build aborts on a panic, and a test build propagates
it through the join ([`diagnostics.md`](diagnostics.md), "Panics and
profiles"), but only if the other workers exit: a worker that panics
mid-node never marks it finished, and the rest would wait for it forever.
So each worker holds a guard that, while unwinding, marks the run aborted
and wakes everyone, and the join re-raises the node's own panic. The work
that would justify a dependency is the
scheduler, readiness counts and the choice of which ready node goes to which
worker, and no crate has it in the DAG-with-placement shape the model wants,
so it is written under either choice. The usual crates would each add
something unwanted. Rayon is a work-stealing scheduler, and stealing is
placement, which is the orchestrator's; its idle workers spin before they
sleep; and a stream worker is a thread dedicated to one device, which a
general pool has no notion of. Tokio is for many sockets, and its file I/O is
a blocking thread pool underneath. The `threadpool` crate's `'static` bound on
a job forces every edge buffer behind an `Arc` and defeats the scoped borrow.
The workers live behind one type in one module, so that if a profile ever
asked for stealing on the CPU side a crate could slot in behind the same type.

**What is tested.** Every node runs exactly once, in a topological order, on
no more threads than the plan said. The last is a tier-3 test
([`testing.md`](testing.md), "Tiers: where a check lives") that reads the
`Threads:` line of `/proc/self/status` after a device-free run and compares it
to the plan's worker total plus the main thread, and reads the same count
before a loop's first iteration and after its last. It is the one check that
catches what a lint cannot see: a linked host library opening a pool of its
own at load, as a host BLAS does. If one is ever linked, its thread count is
set to one at initialization, and this test is what notices when that slips.

### Capabilities are enforced by lints

Three things a task may not do are each held by named code: allocating is
the arena's ("Memory: layouts and arenas"), spawning is the worker module's,
and `unsafe` is a listed few crates'. Each is a lint, in the manifest's
`[lints]` table that [`documentation.md`](documentation.md) ("Code:
rustdoc") already uses for `missing_docs` or at a crate's root, so a violation is a build failure in every invocation, the
gates and an editor alike, and not a convention a review has to remember.

**Spawning.** `std::thread` cannot be removed from the standard library, but
clippy's `disallowed_methods` makes calling it a compile error everywhere but
the one module with the `allow`:

```toml
# clippy.toml
disallowed-methods = [
  { path = "std::thread::spawn", reason = "workers belong to the orchestrator" },
  { path = "std::thread::scope", reason = "workers belong to the orchestrator" },
  { path = "std::thread::Builder::spawn", reason = "workers belong to the orchestrator" },
  { path = "std::thread::Builder::spawn_scoped", reason = "workers belong to the orchestrator" },
]
```

The manifest does the rest: rayon and tokio are not dependencies, and a pool
crate arriving later is a `Cargo.toml` diff a review sees.

**`unsafe`.** The rebuild's `unsafe` is the FFI into `wwr-sys` and the
kernel libraries, and the code that wraps it. `unsafe_code` is rustc's own
lint, and `forbid` is the level that cannot be allowed back, so it is set
per crate: a crate on the list below leaves it at its default, and every
other crate forbids it with `#![forbid(unsafe_code)]` at its root. That is
an attribute rather than a manifest line because Cargo refuses a member that
inherits `[workspace.lints]` and adds a `[lints.rust]` table of its own. The
list is in two tiers:

- **Core:** `fb-core-wwr`, `fb-core-arena`, `fb-core-log`,
  `fb-core-thread-pool`.
- **Bindings:** `fb-integrals-sys`, `fb-recipes-sys`.

The core crates hold the wrappers: `fb-core-wwr` over the `wwr-sys`
bindings, the arena over the raw allocation and copy bindings, the two
`libc` calls in `core::log`, and the one `unsafe impl Sync` in
`core::thread_pool`, on `run`'s private wrapper of the backend it shares
with its workers, which rests on a run borrowing the backend exclusively
and on `Backend` being sealed ("Plan once, run many"). The binding crates declare a kernel
library's `extern "C"` functions, check its version at load, and wrap its
launches safely beside the bindings (`fb-integrals-sys` its records tasks
too, since the class launchers trust records only those tasks write). There
was a third tier, the method crates, whose `unsafe` was a direct call into
their kernels' launchers, the BLAS or the solver; each such call has moved
into a safe wrapper in the core and binding crates, and a method crate that
needs a new one adds it there rather than joining the list.
`fb-doc-check`'s `tests/unsafe_code.rs` reads this list: a crate root that
neither forbids `unsafe_code` nor is listed fails it, and so does a listed
crate with no `unsafe` left, so the list only shrinks.

Where `unsafe` is
allowed, clippy's `undocumented_unsafe_blocks` requires a `// SAFETY:`
comment on every block, which turns the comment the crate already writes into
a rule and names the blocks that lack one, and
`multiple_unsafe_ops_per_block` keeps each block to one operation so the
comment justifies exactly one thing. Miri covers what a lint cannot
([`testing.md`](testing.md), "Sanitizers").

A library call reaches the crates above `core` through a safe wrapper, not
`wwr-sys`: `fb_core_wwr::blas::gemm` and `dgmm` and
`fb_core_wwr::solver::syevd` take their operands as `DeviceSlice`s (device
memory with its length, which the arena's rows implement) and check each
extent against them in every build (syevd's work against the solver's own
query), `fb_core_wwr::memset::zero` zeroes one on a stream, and
`fb-recipes-blas`, the BLAS calls' packed shorthand, forbids `unsafe`
itself, as the arena's `memcpy_*` do for the copies. A kernel library's
launch is wrapped the same way in its binding crate, beside the generated
bindings: `fb_recipes_sys::inverse_sqrt`, `merge_sort` and
`merge_sort_keys` take `DeviceSlice`s and map the launch status through
`fb_kernel_abi::launch_status`, and `fb-integrals-sys`'s `overlap_class`,
`kinetic_class` and `potential_en_class` take the output as a
`DeviceSliceMut` whose `n_cart²` they assert, and the records as a
`ClassRows`, a type only its own records tasks make, since the AO offsets
the kernels write at are device data no host check can read. That is the
shape for the rest. The BLAS
workspace, which the library keeps beyond any borrow, is handed over by
value instead: `BlasHandle::bind_workspace` owns an arena `HeldRow` until
the handle is destroyed or rebound ([`gpu_layer.md`](gpu_layer.md),
"Layering"). Every crate above the core and binding crates now has nothing
left to call raw and forbids `unsafe`: `fb-pipeline-scf`, the recipe
crates, `fb-integrals-mmd` and the four `fb-integrals-one-electron-*`
crates among them.

```toml
[workspace.lints.clippy]            # every member: [lints] workspace = true
disallowed_methods = "deny"
undocumented_unsafe_blocks = "deny"
multiple_unsafe_ops_per_block = "deny"
missing_panics_doc = "deny"
```

All four are on. `missing_panics_doc` holds the other half of the
`unsafe` rule: the length an `unsafe` call trusts is an always-on
`assert!`, and the lint fails a public function that can panic and whose
doc has no `# Panics` naming it ([`testing.md`](testing.md), "Contracts are
part of the spec").

## Reproducibility across placements

The same planned DAG must give the same answer, to a declared tolerance,
under every placement. Where every reduce is ordered, that tolerance is zero
and the result is bitwise. Where a reduce is unordered, its declared tolerance
is the bar, and the test tier compares the one-worker result against a
many-worker one at that bar
([`testing.md`](testing.md), "Tiers: where a check lives", tiers 4 and 5).
Tolerances are fixed by analysis and never loosened to make a comparison
pass.

## Example: the Coulomb build under two placements

The DAG, declared once:

```
        D ──broadcast──┬──────────┬──────────┐
                       ▼          ▼          ▼
                   J_0 (w0)   J_1 (w1)   J_2 (w2)    shell-pair slices
                       └────reduce┴──────────┘
                                  ▼
                                  J
```

- **Task J:** GPU. Layout per slice: the partial J (`nbf²`, an output) and
  the integral scratch, sized by the kernel's workspace query. Boundary:
  reads D, writes its partial, touches nothing else. Partition: the
  shell-pair list, recombined by an ordered sum.
- **Edge D:** `nbf²` bytes, broadcast. Freed after the last slice, and after
  K, which reads it too.
- **Edge J:** `nbf²` bytes per slice, reduce, ordered.
- **Derived:** the fork is the partition; the join is the reduce; K is a
  sibling of J, not a successor, because it shares D and nothing else.

The same declarations under two placements:

| | One GPU | Three GPUs |
|---|---|---|
| Workers | Three streams, or one if admission leaves no room for three partials | One card per slice |
| Broadcast | D's pointer | D copied to each card |
| Reduce | Each slice accumulates into J in slice order | Partials copied to one card, summed in slice order |
| Concurrency | K on a fourth stream if its footprint fits beside J's | K on any card with room |

Neither the task nor the edges changed. Only the orchestrator's answers did.

A second example costs one sentence: the semi-numerical Hessian is `6 natoms`
gradient sub-DAGs, each the whole gradient DAG with the same dimensions,
independent until the final assembly. On one GPU they run in sequence; on a
node they spread across cards; the gradient DAG is declared once.

## Granularity

A task is the smallest unit with its own footprint and no internal host sync.
That rules out both ends:

- A kernel launch is below the line. A launch wrapper allocates nothing and
  syncs nothing, so it has no footprint to admit and nothing to time but the
  stream.
- A pipeline step is above the line. It allocates many things with different
  lifetimes and syncs between them, so it is a sub-DAG, and its syncs are its
  edges.

The practical test is the second rule: if a computation must wait for part of
itself, split it there.

## Open questions

- **Where the line is in practice.** A sync that is free on one GPU, two
  kernels on one stream, still splits a task under this rule. Is that the
  right cost, or should a task be allowed a chain of launches on one stream
  with no edge between them? The current answer is that a chain on one
  stream is not a sync, so it stays one task; the rule is about host waits.
- **Adaptive layouts and the schedule.** An adaptive task's width depends on
  the room the schedule leaves, and the schedule's peak depends on the
  widths. Is one pass enough, laying the fixed layouts out first and handing
  each adaptive task the remainder in schedule order, as `check_plan`'s
  room line does today, or does the planner iterate?
- **Reuse across streams.** Offset reuse between buffers on different
  streams needs a DAG path between their lifetimes. Should the planner add an
  event edge to create one when the saving is large, or only exploit the
  paths the DAG already has?
- **How loose is too loose.** A bound from the dimensions is the preferred
  answer to a data-dependent size. At what ratio of bound to typical size
  does a sizing node pay for its extra host round trip?
- **The schedule search.** Finding an order whose peak fits is a scheduling
  problem that is hard in general. A plan order from the config, improved
  greedily, is probably enough for the DAGs this tree has. When is it not?
- **Transport.** NCCL, MPI, or something else moves the lower rows of the
  worker table. Nothing in the model depends on the choice, and nothing here
  decides it.
- **Pinned blocks and NUMA.** On a multi-socket node, pinned pages on the
  GPU's own socket copy faster than pages across the interconnect. A pinned
  block is allocated on its worker's thread, so pinning that thread to the
  GPU's socket would place the block without the planner knowing. Whether
  the gap is worth the pinning is a measurement on such a node.
