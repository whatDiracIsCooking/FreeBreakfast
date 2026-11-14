# Rust rebuild: memory

The programming model fixes the principles: a task's layout is its
footprint, the arena is a table and a block, the block is the plan's peak,
and a run allocates nothing ([`programming_model.md`](programming_model.md),
"Memory: layouts and arenas", "The arena is a table and a block", "Blocks,
plans and the process", "Memory kinds"). This document is the level below,
the one a planner is written from: what a row of the table is and who owns
it, how a lifetime is represented and when two rows may share bytes, the
admission arithmetic, how an adaptive width is chosen, how the two kinds are
allocated, and what checks each rule. Every rule here is fixed by a case the
C++ tree already has, and the cases are cited rather than invented: the
footprint vocabulary and fold
([`src/core/footprint`](../../src/core/footprint/README.md),
[`src/pipeline/footprint`](../../src/pipeline/footprint/README.md)),
pre-flight ([`src/pipeline/shape`](../../src/pipeline/shape/README.md),
[`src/pipeline`](../../src/pipeline/README.md), "Memory caps"), and
`DeviceResources` ([`src/core/resources`](../../src/core/resources/README.md)).

The C++ footprint was a second program about allocation, held in step with
the first by tests against a measured pool peak. The rules below exist so
that there is one program: the table *is* the allocation, and what the C++
measured is here read off the plan.

## The table

The planner's output for one worker and one memory kind is a table: one row
per buffer, with its owner, its bytes, its alignment, its lifetime and the
offset the assigner gave it, and the block's total. A `run` touches memory
only through a row. Four owners, and the owner is what decides the lifetime:

| Owner | Row | Lifetime | Example |
|---|---|---|---|
| A task | Its scratch, from `layout(dims)` | The task: its first launch to its last | A kernel's `*_workspace_bytes`, a solver's `lwork` and its `info` flag, DIIS's coefficients for one push |
| An edge | Its producer's output, from the producer's `layout(dims)` | The edge: the producer's first write to the last consumer's last read | S, H_core, the converged C, B and L |
| A worker | A buffer the worker holds for every task it runs | The plan | A BLAS handle's workspace |
| The orchestrator | A buffer placement created | The edge it carries | A broadcast's copy on the consumer's card, a reduce's partials on the join's card |

**An output's bytes are declared once.** The task's layout names the output
and its size; the edge declares its kind, its reduce order and its consumers
([`programming_model.md`](programming_model.md), "An edge declares"), and
its size is *read* from the producer's layout, never written a second time.
Two formulas for one buffer is the drift the C++ `sizes` struct could only
suggest against.

**Per-worker rows exist so that nothing is uncounted.** Pre-flight today
subtracts `kOwnPoolBytes` from the room by hand, the solver `info` flag that
`DeviceResources` holds from construction and that no step's footprint can
see. Here each library handle's workspace is a row with the lifetime "the
plan", placed by the assigner like any other ([`gpu_layer.md`](gpu_layer.md),
"Layering"). The flag is not: it is the `syevd` task's own, a device scratch
row syevd writes, copied on the stream into a pinned output row that the
`syevd check` CPU task reads ([`gpu_layer.md`](gpu_layer.md), "Scratch: the
workspace query"). A flag the task declares is counted in its layout and
ordered by its edges, and two eigensolves have nothing to share through a
per-worker one. Its `lwork` scratch is a task row too, and `memcheck` sees
only the arena block's bounds, not a row's: a `syevd` that overruns its
`lwork` row into a neighbouring row of the same block is not caught. A worker's rows are its own
layout, called once at planning with the plan's dimensions, and the C++
`syevd_lwork[k]` table, the solver's query at every order the plan
diagonalizes at, is how those dimensions already carry a library's size.

**Transport rows are placement's.** An edge whose two ends land on different
cards needs a destination on the consumer's card; a reduce whose slices land
apart needs the partials beside the join. The orchestrator adds those rows
when it places, sized by the edge, with the edge's lifetime from the copy's
enqueue. On one card they do not exist: a broadcast is a pointer, and a
reduce accumulates in order into the join's output
([`programming_model.md`](programming_model.md), "Workers are a hierarchy").
The join owns the reduced output; each slice's partial is that slice's
output row, and on one stream the planner may give slice 0's partial the
join's offsets, since the ordered accumulate writes the same bytes.

**Where the C++ lifetimes went.** `persistent` is an edge row; `scratch` is a
task row; `workspace`, the one fixed block every step borrowed from, splits
in two. What a step carved from it for itself is a task row, which the
assigner aliases with every other task's dead scratch, so the "workspace
gap" pre-flight logs, a one-off large carve held at the plan's maximum
through every step, cannot occur. What the handles hold is a worker row. The
rule `DeviceResources` needed, one routine owns the workspace at a time and
a callback is a nested user, guarded against two routines carving from the
same start; here every node has its own offsets and the rule has nothing
left to guard. The Davidson keeps its subspace across its operator's calls
in rows of its own because they are edges of its sub-DAG, not because a
rule says so.

## Lifetimes

A lifetime is an interval in the schedule, and the schedule is a partial
order over launches, so the interval's ends are positions in that order:

- **On one stream**, the stream's order is total, and a lifetime is a pair
  of positions. Reuse is the C++ `then` / `drop` algebra with the implicit
  frees made explicit.
- **Across streams**, two positions are ordered only through an event one
  stream records and the other waits on, or through a chain of edges that
  implies one. Two buffers on different streams may share bytes only when
  such a path runs from one's end to the other's start. Whether the planner
  adds an event edge to create a path is the programming model's open
  question on reuse across streams
  ([`programming_model.md`](programming_model.md), "Open questions").
- **Across a CPU node**, the node's completion is a host event, not a stream
  one. A pinned row read by a file-write node ends when the write returns,
  and the next row to take those bytes is a device-to-pinned copy the
  orchestrator may not enqueue before that. The readiness count that gates a
  node on its predecessors is the mechanism; the memory rule adds that an
  offset's previous owner counts as a predecessor. This is `DeviceResources`'
  "drain the copies that read a region before rewriting it", as a planning
  fact.

**Two rows may share bytes iff one's end precedes the other's start.** That
is the assigner's one invariant, and it is what the fuzz checks
([`testing.md`](testing.md), "Fuzzing"). The assigner is greedy with best
fit, run in three orders and kept at the smallest total: by first position
and then the larger row first, by first position and then the longer-lived
first, and the larger first whatever its position. A schedule it cannot fit
at the given order is re-ordered or refused by admission, never allocated
past the block.

**The block is within 3% of its lower bound on every committed golden.**
The bound is the most that is live at once: at each position, the live
rows' bytes each padded to the alignment, less the largest padding, which
the last row does not need; a loop's region counts as one row of its
parent, and its body is held to the bound alone. No single order meets it:
measured over the goldens when it was set, start order with the larger
first came to 1.20 times the bound on the arena's SCF fixture, the
longer-lived first to 1.005 on the DF-RHF config's plan, where the first
order and the larger first are within 0.02%, and the larger first to 1.015
on DIIS's, where start order meets the bound exactly; the three together
came to 1.026 at worst, the guess config's pinned table. The bound is a
measurement, not a theorem: a plan that breaks it is a finding about the
assigner, read before the bound is moved.

**"Precedes" is the plan's rank order, so reuse adds edges.** The positions
the assigner packs against are the nodes' ranks, the order they were added
in, but a run starts any ready node whatever its rank: on one stream a
higher rank runs first when a lower one is still waiting on a CPU
predecessor. So the table returns its **handoffs** (`Table::handoffs`):
for every pair of rows that share bytes, each user of the earlier paired
with the later one's producer. Each handoff points forward in rank and is
added to the plan as a dependency, which is the rule above, an offset's
previous owner counts as a predecessor, made uniform. On one stream a
handoff between two stream nodes costs nothing, the stream's order already
holds it; from a CPU node it is the wait the rule asks for. A tier-3
property test runs random DAGs in random orders consistent with their
edges and handoffs and checks that no two rows sharing bytes are live at
once.

**An alias is free of synchronization.** The ERI step is the case. Issue
1153 found the DF chain's packing broken because B was allocated while the
metric pair's frees were still pending on the stream, and the pool took new
chunks; the fix was a host synchronize before B. Under the table B's row
starts after the copy into `l` that is the pair's last read, on the same
stream, so B takes the pair's bytes with no wait at all. The step's README
calls that sync a departure from the no-wait ideal; here it is gone.

**No row is modified in place.** A task writes only the rows it produces,
and a loop's last user the row it carries ("Nested plans"); a row it reads
is read-only to it. The C++ fit overwrites the metric with L and the
tensor with B, so here the Cholesky copies M into L and the fit copies
(mn|P) into B, and while each runs both are live. A read-write row, a task
that takes a row and hands it on under a new name, was weighed and
declined (#1536), because it never lowers a peak. On the ERI step's water
fixture (cc-pVDZ, cc-pVDZ-JKFIT, `nao` 24, `naux` 116), with each row
padded to the alignment and beside the 32 MiB BLAS workspace, the
three-center task holds 2,133,248 bytes; the Cholesky 783,104, 675,328
with L in M's bytes; and the fit 1,176,832, 642,304 with B in the
tensor's. The plan's peak is 35,687,088 bytes at the three-center task
either way. The order is structural: the three-center task holds the
Cartesian tensor, its half-transformed copy and the spherical output at
once, three tensor-sized rows, and the fit two and L, whose `naux²` is
less than `naux · nao²` whenever `naux < nao²`. The question is worth
reopening only when the three-center task stops being the peak, a
transform in place or in auxiliary blocks; the row to build then is
`Layout::write` with its loop contract relaxed to "no later user", not a
new kind.

## Admission

Per worker and per kind, the block must fit:

| Kind | Must hold | Terms |
|---|---|---|
| Device | `block + reserve + stack <= cap` | `reserve`: the context, the handles and the loaded modules, outside any block; `stack`: the kernels' stack the driver takes back when the card has no free memory left for it (`kKernelStackBytes` per thread, up to 1.6 GiB on the RTX 3080) |
| Pinned | `block <= pinned_cap`, and `sum over the node's workers <= MemAvailable - margin` | The per-GPU cap, and the C++ `check_pinned_node` carried over |

**No headroom and no eighth.** The 64 MiB fragmentation headroom and the HIP
`ceil(P / 8)` reuse rounding are properties of a pool that reserves in chunks
and serves a request from a larger freed block. One plain allocation has
neither, and both terms are dropped. What stays is what sits outside every
block: the reserve, a floor measured on the 3080 and grown as modules load,
and the stack.

**The reserve is measured, with the constant as its floor.** At process
start, after the devices' handles exist and before any block, each worker
reads its device's free memory against its total; the difference is that
device's reserve, and the C++ `kDeviceReserve` of 256 MiB is the least it may
be taken as. A card another process holds part of shows up here, as a
smaller cap, rather than at the first allocation.

**Without a cap, the cap is the card.** A config with no `[memory]` table is
admitted against the device's free memory at planning less the reserve and
the stack, so an uncapped plan that does not fit is still refused before the
first launch. Today an uncapped plan is checked against nothing but the
device workspace.

**Admission is a search over order.** The orchestrator's first job on one
GPU is an order whose peak fits ([`programming_model.md`](programming_model.md),
"The pipeline is already an instance"). The search is: the plan order from
the config, siblings concurrent where room allows; when the peak is past the
cap, serialize the widest concurrency first, then narrow each adaptive task
in schedule order down to its least; refuse only when the serial plan at
every adaptive task's least does not fit. The refusal names the worker, the
kind and the bytes over; then every row, and every loop's region, larger
than the cap on its own, which no order could fit, live at the peak or not;
then the instant of the peak and every row live at it, which is the C++
message, the step that reaches the peak, with its rows. The peak alone
would miss a row the peak does not hold: the DF step's peak is the
three-center task, before B is live, so a cap short of B would name only
the integrals' rows.

**The block is the pre-warm.** Pre-flight today allocates and frees the
pool's cap once so that a card another process holds part of is refused
before the first step. Here the block allocation is that probe: one call,
and a refusal naming the bytes asked and the free memory
([`diagnostics.md`](diagnostics.md), "Allocation is a planning-time error").

## Adaptive widths: one planner routine

Three C++ routines size a part of themselves from free memory at run time,
and they are one pattern:

| Routine | Width | Bytes it bisects | Least |
|---|---|---|---|
| NEVPT2's `walk_step` | The slab batch | `walk_bytes(step)`, against the budget less the headroom | A batch of one |
| FCI's `set_scratch_limit` | The block of strings whose G and D fit | `sigma_apply_bytes(limit)`: the whole vector where it fits, else the largest block | One string's |
| AVAS's `narrow_avas` | `ncas` | The whole fold at each count | One orbital |

Each writes its own bisection and reads the budget through `memory_budget()`
at run time. Here a task with an adaptive part declares it as two functions
beside `layout`: `bytes(width)`, monotone non-decreasing, and
`width_max(dims)`, the natural full size. The planner bisects `bytes` against
the room the schedule leaves at the task's position, once, at planning. The
width is then a dimension of the plan, logged at `debug` and at `warn` when
it is 1 ([`diagnostics.md`](diagnostics.md), "Levels"), and `run` never reads
free memory. A `bytes` that is not monotone is caught by the tier-3 fuzz
that bisects it and checks the result against a linear scan. clippy's
`disallowed_methods` lists `wwr_mem_get_info`, so a read of free memory is
an `allow` naming its exception: an allocation's refusal ("The two kinds"),
admission and the reserve's measurement ("Admission"), or a tier-4 test
measuring what a block costs.

**AVAS is the adaptive case that changes the DAG's dimensions.** Its width
is `ncas`, and every downstream layout grows with it, so it is bisected over
the whole fold, as `narrow_avas` does, and its result becomes the plan's
`ncas`, with a pick of more refused at the step rather than truncated. It is
the same routine with a different `bytes`.

**One pass, in schedule order.** The fixed layouts are placed first, then
each adaptive task takes the remainder at its position, which is pre-flight's
room line today. Whether the planner should iterate, since the widths change
the peak and the peak the room, is the programming model's open question on
adaptive layouts ([`programming_model.md`](programming_model.md), "Open
questions"); the single pass is the model until a plan shows it is not
enough.

## Sizes the dimensions do not fix

The C++ tree has both of the model's answers, so neither has to be argued
from first principles:

- **A bound from the dimensions.** The Davidson's subspace at 16 `nroots`
  columns, which can pass `nao`; AVAS's worst-case `ncas` when no cap
  narrows it; `syevd_lwork` at every order through the plan's largest
  eigensolve; the CASSCF's worst-case `ncas` and `nelecas`. Each is a layout
  written at the most the task can take, and the table above is sized to it.
- **A sizing node.** The Coulomb pair-class prune copies the kept counts to
  pinned memory, synchronizes, and sizes J's classes to them. In the DAG that
  is a device-to-pinned copy node, a CPU node that reads the counts and fixes
  the sub-DAG's dimensions, and the sub-DAG planned inside the region its
  parent reserved at the bound. The region is the bound; what the sizing
  node saves is not memory but the launches over the pruned classes.

The ERI's own check, B's and L's bytes against the budget before B is
allocated, is admission's: B's row is in the table before the first launch,
and a B that does not fit is refused at planning with its bytes named.

## Nested plans

A sub-DAG has a table and no block; its rows are laid out in the region its
parent's schedule leaves free while it runs
([`programming_model.md`](programming_model.md), "Blocks, plans and the
process"). The Hessian and Optimize steps are the cases, and they add one
rule each:

- **A region is the max over the inner shapes.** The semi-numerical Hessian
  runs the reference chain to `[casscf]` once, then each displacement's
  chain from the carried orbitals, a different sub-DAG with a different
  fold. Its region is the larger of the two peaks, and both are laid out
  inside it. `HessianStep::footprint` folds each with `plan_footprint` today
  and takes the max by hand; here each inner DAG is planned as a table inside
  the one region.
- **Replay has the same dimensions.** Every geometry of an optimization and
  every displacement of a Hessian has the same `nbf`, `naux` and active
  space, so one inner table serves every run, which is why the inner plan is
  laid out once and the outer's two allocations are the calculation's only
  two. What varies with geometry takes the bound.
- **What crosses out is in the parent's table.** The optimizer holds the
  start's Molecule beside the chain and keeps the optimum's; the Hessian
  keeps nothing on the device and assembles on the host. Those rows are the
  parent's, with the parent's lifetimes, and the inner table never holds a
  row that outlives it.

Concurrent sub-DAGs do not share a region: the width the planner admits is
how many regions the parent holds at once, each at the sub-DAG's peak.

**A loop is the nested plan the SCF has.** Its body is one iteration's
tasks and its decision the CPU node after them, and the body's table is the
rows every user of which is in the loop: laid out alone, from the region's
first byte, while the region is one row of the parent's table, live from the
body's first position to the decision, of the body's laid-out total. Every
other row the loop uses, S read every iteration or the energy read after
the loop, is the parent's and live for the whole loop, so no body row takes
its bytes, and admission counts the region once, at each of the loop's
positions, by the name `loop first-last`. Within an iteration the body's
rows share bytes as any table's do, ordered by their handoffs. Across the
iteration edge nothing needs ordering: the decision is after every body
node and the replay after the decision. A dependency from a body node to a
node after the loop is taken from the decision instead, since a body node
runs once per iteration and the decision's `Break` once.

**A carried row is the parent's.** The density, the occupied factor and
DIIS's history pass from one iteration to the next. Such a row is produced
before the loop (the guess, a zero fill, `add_diis`'s `diis history` task),
read in the body, and written by
its last user there, so each iteration reads what the one before wrote, or
the producer's value in the first. The table orders every other user
against that writer: the body's readers before it, a node after the loop
after the decision. It is live across the whole loop, so no handoff crosses
the iteration edge. A decision that keeps something from one iteration to
the next, the last energy, keeps it the same way, as a host row it carries:
nothing outside the rows survives the run.

## The two kinds

| | Device | Pinned host |
|---|---|---|
| Raw call | `wwr_malloc`, plain | `wwr_host_alloc`, portable |
| Thread | The one that makes the plan, its device made current for the call, before any stream work on it; the worker's own thread once one outlives a run | The same; the worker's thread would put the pages on its socket |
| Freed | At the plan's drop, after the last run's final sync | The same |
| Alignment | `FB_DEVICE_ALIGNMENT` for every row ([`gpu_layer.md`](gpu_layer.md), "Scratch: the workspace query") | The same; the copy engines need no more, and `kWorkspaceAlignment` is 256 on both sides today |
| Fill | Every byte `0xFF` at instantiation, in debug builds: a NaN in both float widths, and `-1` in every integer width | The same, in every build |
| Failure | Aborts, as every create does (`check_or_abort`), naming the bytes and the free memory | The same, naming `MemAvailable` |

**Plain, not stream-ordered.** One allocation per plan has nothing for a
pool to amortise, and `hipMallocFromPoolAsync` beside other threads' stream
work has deadlocked on ROCm 7.2.4 (the crate's README). A plain allocation
made before any stream work on the device never meets that.

**Portable, not write-combined, not mapped.** Portable so that a pinned
block one worker owns may be read by another card's copy engine, the
"GPUs on one node" row of the worker table. Not write-combined, because CPU
nodes read pinned rows back, the prune's counts and every product written to
disk, and write-combined memory makes those reads slow. Not mapped, because
no kernel reads host memory directly in this tree, and a mapped pointer would
be a third kind of buffer a task could reach.

**The debug fill is the arena's own check.** A row read before it is
written is a bug the C++ pool, which hands back whatever a freed block held,
cannot show. A block filled with `0xFF` makes every such read a NaN, which
the first comparison or norm turns into a failed test. A release build skips
the device fill, but not the pinned one: a pinned row is handed out as a
`&mut [T]`, which must not be uninitialised memory in any build, and pinned
blocks are small staging, so the fill costs nothing worth measuring. The
same reason restricts a row's type to an `Element`, a primitive for which
every bit pattern is a value, since a reused row is read with whatever its
previous owner left.

**What is not a kind.** Managed memory is not used and not modelled, and
nothing here should grow a `Managed` marker; clippy's `disallowed_methods`
lists `wwr_malloc_managed` and `wwr_stream_attach_mem_async`, and no site
allows them. Pageable host memory is `Vec`,
owned by the plan when it is part of the DAG
([`programming_model.md`](programming_model.md), "Memory kinds").

**The three pinned layouts that exist.** Every pinned use in the C++ tree
is staging sized by the dimensions: mmd's shell tables on their way up, the
Coulomb prune's kept counts on their way down, and the Molecule's charges
and coordinates. `CountedPinned`, the offloaded B streamed through a ring,
has no user yet. So the first pinned rows are small upload and readback
buffers, and the ring, when it comes, is a pinned edge row with the lifetime
of the copies that read it.

### The Rust shape

One module, `core::arena`, one of the few crates with `unsafe_code`
allowed ([`programming_model.md`](programming_model.md), "Capabilities are
enforced by lints"); its raw calls are on a sealed trait nothing outside it
can name. In it:

- `TableBuilder<K>`, the rows declared node by node, and `assign(rows)`,
  the pure offset assigner it lays them out with into a `Table<K>`: tier 3,
  fuzzed, its text the committed golden ([`testing.md`](testing.md),
  "Regression"). The table also returns the handoffs ("Lifetimes"), and
  the reads, each reader after its producer; the two together are every
  dependency it asks of a plan. `TableBuilder::node(pos)` is a `Layout`,
  one node's declarations with its position bound, which is what a task's
  `layout` is handed. `core::planner` calls each task's layout in plan
  order, builds the tables, and adds the reads and handoffs to the plan.
  `TableBuilder::add_loop(positions)` marks a loop and `Layout::write(row)`
  a row it carries ("Nested plans").
- `Block<K>`, with `K` one of two uninhabited markers, `Device` and
  `Pinned`. It holds the raw pointer, the table and one atomic borrow flag
  per row, allocates in `new` and frees in `Drop`, and the kind is in the
  type, as it is in every `RowId<K, T>`, so a device row cannot be handed to
  a host write.
- `Block::<K>::partition(&self, node) -> Bufs`, called by the orchestrator
  once per node per run. A pinned row is a real `&[T]` or `&mut [T]`; a
  device row is a pointer-and-length view with no `Deref`.

**The view's checks are always on, not `debug_requires`.** Two nodes
running at once both need rows of one block, so `partition` takes `&self`,
and the orchestrator that calls it has no `unsafe`. A safe `&self` API that
hands out `&mut [T]` cannot rest on a debug-only contract, so a view takes a
row's flag when the node first asks for it, a writer alone and readers
shared, and returns it when the view drops; a row the node's layout did not
name, or one another view holds in a conflicting mode, panics naming it. In
a plan that honours the handoffs no flag is ever contended, so the cost is
one uncontended atomic per row a node asks for.

**A stream node does not touch a pinned row from the host.** It hands the
row to a copy it enqueues and returns. A handoff between two nodes on one
stream adds no wait, because the stream already orders their device work,
but it does not order a host write against an earlier copy still in flight
out of the same bytes. Host reads and writes of pinned rows are CPU nodes',
which the executor starts only after the events of the stream nodes before
them.

`run(bufs, stream)` receives the `Bufs` and nothing else that can allocate.

## What is checked

| Rule | Check | Tier |
|---|---|---|
| No two live rows alias | The assigner fuzzed as a pure function over random rows and orders | 3 |
| The block is within 3% of its lower bound on every committed golden | Every golden's text read, each table's total against its live rows' padded bytes at their largest, each loop's region against its body's | 3 |
| The block is the plan | The arena's total, committed and compared bitwise per example config | 3 |
| `bytes(width)` is monotone | Bisection against a linear scan over random dims | 3 |
| Admission refuses what does not fit | A cap one byte under the peak refused naming the rows live at it; a cap under a row refused naming that row, the DF step's B not live at its peak among them; one byte over admitted, as the C++ nested cases pin the room to the byte | 3 |
| A task reaches only its rows | The view's borrow flags and layout check, always on | 2 |
| Reusing bytes is ordered | Random DAGs run in random orders consistent with their edges and handoffs, no two rows sharing bytes live at once | 3 |
| A loop's body lies in its region, apart from every parent row live across the loop | Random bodies laid out, then unrolled over iterations and run in random orders consistent with the reads and handoffs, no two rows sharing bytes live at once | 3 |
| Replay allocates nothing | Free memory read by a device loop's decision every iteration, never falling from one iteration to the next: the reading is the device's, so another tenant's free can raise it, and an allocation of this process can only lower it | 4 |
| Declared equals allocated | Free memory before and after instantiation differs by the block, under the reserve's growth | 4 |
| No library allocates behind the arena | The same measurement across a run: free memory never falling from the first launch to the last sync | 4 |
| A kernel assumes no more scratch than it asked | The wrapper handed exactly its query's bytes under `memcheck` ([`testing.md`](testing.md), "The C++ boundary") | 4 |
| A row is written before it is read | The debug fill, through every tier-4 test's comparisons | 4 |

What the C++ footprint tests measure, a pool peak held to a formula at two
bars, has no counterpart: there is no second number to compare the first
against.

## Open questions

- **Bytes against concurrency.** The assigner always packs to the fewest
  bytes, and every byte it reuses is a handoff, an edge the plan did not
  otherwise have. Two independent nodes whose rows share bytes are then
  serialized when a larger block would have let them overlap. On one stream
  that costs nothing; once there are several, admission's search should
  decide which reuse to give up for concurrency when the cap leaves room.
- **Iterating the adaptive pass.** One pass in schedule order is pre-flight's
  model today. A plan whose adaptive tasks are large relative to its fixed
  rows may leave room the pass does not find.
- **The reserve's growth.** Modules load as kernels are first launched, and
  the reserve grows after the block exists. The 256 MiB floor covers what the
  3080 has shown; a card with more kernels resident may need the first run
  measured.
- **The stack constant per card.** `kKernelStackBytes` times the resident
  threads is the 3080's number. It belongs with the device's properties at
  process start, read rather than carried.
- **Pinned pages and NUMA.** Allocating on the worker's thread places the
  pages where that thread runs; whether pinning the thread to the GPU's
  socket is worth it is a measurement on a multi-socket node, the
  programming model's own open question on it
  ([`programming_model.md`](programming_model.md), "Open questions").
- **Transport rows and the width.** A broadcast to `w` cards is `w - 1`
  destination rows; a reduce from `w` cards is `w - 1` partial rows at the
  join. Both grow with the width the planner chooses, so they are part of the
  admission search, and the search described above has not been written with
  them in it.
