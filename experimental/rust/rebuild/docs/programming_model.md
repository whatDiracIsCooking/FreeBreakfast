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
([`src/pipeline/footprint`](../../../../src/pipeline/footprint/README.md)).
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
([`src/core/footprint`](../../../../src/core/footprint/README.md),
[`src/pipeline/footprint`](../../../../src/pipeline/footprint/README.md))
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
what the arena is built from.

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

Stream capture into a replayed graph is the same split one level down, which
is a sign the split is real rather than imposed. A loop is therefore not a
special node: it is a planned sub-DAG run repeatedly, with a host decision
between runs. The decision is a node (convergence, a step accepted, the walk
reaching a minimum) whose output is read on the host.

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
  Nothing inside the tasks changes.
- **Within a task.** The amplitudes of a coupled-cluster iteration, the
  Coulomb build over slices of the shell-pair list: one task's work sliced
  over workers, a broadcast in and a reduce out. This is the partition a task
  may declare, and it is the only place distribution reaches into a task.

Keeping the partition opt-in is what keeps single-GPU code honest: a task
without one is written once, for one stream, and is still placeable on any
worker of any topology.

## Workers are a hierarchy

A worker is a stream, a GPU, or a node, and the levels nest. An edge's kind
and size are the same at every level; only the cost differs, and the
orchestrator prices it by where the two ends landed:

| Edge crosses | Broadcast is | Reduce is |
|---|---|---|
| Streams on one GPU | A shared pointer | An ordered accumulate into one buffer |
| GPUs on one node | A device-to-device copy per card | Partials copied to one card, or a peer reduce |
| Nodes | A network transfer | A network reduce |

The single-GPU case is the degenerate one: broadcast is a pointer, reduce is
an in-place accumulate, and the workers are streams. One DAG description
serves all three rows, which is the point of wanting all three. The transport
behind the lower rows is an orchestrator detail, decided later and not here.

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
