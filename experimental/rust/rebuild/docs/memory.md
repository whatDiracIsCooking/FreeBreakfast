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
([`src/core/footprint`](../../../../src/core/footprint/README.md),
[`src/pipeline/footprint`](../../../../src/pipeline/footprint/README.md)),
pre-flight ([`src/pipeline/shape`](../../../../src/pipeline/shape/README.md),
[`src/pipeline`](../../../../src/pipeline/README.md), "Memory caps"), and
`DeviceResources` ([`src/core/resources`](../../../../src/core/resources/README.md)).

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
| A task | Its scratch, from `layout(dims)` | The task: its first launch to its last | The SCF's DIIS history, a kernel's `*_workspace_bytes`, a solver's `lwork` |
| An edge | Its producer's output, from the producer's `layout(dims)` | The edge: the producer's first write to the last consumer's last read | S, H_core, the converged C, B and L |
| A worker | A buffer the worker holds for every task it runs | The plan | A BLAS handle's workspace, the solver's `info` flag |
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
see. Here it is a row with the lifetime "the plan", placed by the assigner
like any other, and the same for each library handle's workspace
([`gpu_layer.md`](gpu_layer.md), "Layering"). A worker's rows are its own
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
([`testing.md`](testing.md), "Fuzzing"). The assigner is greedy in start
order with best fit, and a schedule it cannot fit at the given order is
re-ordered or refused by admission, never allocated past the block.

**An alias is free of synchronization.** The ERI step is the case. Issue
1153 found the DF chain's packing broken because B was allocated while the
metric pair's frees were still pending on the stream, and the pool took new
chunks; the fix was a host synchronize before B. Under the table B's row
starts after the copy into `l` that is the pair's last read, on the same
stream, so B takes the pair's bytes with no wait at all. The step's README
calls that sync a departure from the no-wait ideal; here it is gone.

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
kind, the instant of the peak and every row live at it, which is the C++
message, the step that reaches the peak, with its rows.

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
that bisects it and checks the result against a linear scan.

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

## The two kinds

| | Device | Pinned host |
|---|---|---|
| Raw call | `wwr_malloc`, plain | `wwr_host_alloc`, portable |
| Thread | The worker's, its device current, before any stream work on it | The same, so the pages are the worker's socket's |
| Freed | At the plan's drop, after the last run's final sync | The same |
| Alignment | `FB_DEVICE_ALIGNMENT` for every row ([`gpu_layer.md`](gpu_layer.md), "Scratch: the workspace query") | The same; the copy engines need no more, and `kWorkspaceAlignment` is 256 on both sides today |
| Debug fill | Every byte `0xFF` at instantiation: a NaN in both float widths, and `-1` in every integer width | The same |
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
the first comparison or norm turns into a failed test. Release builds skip
the fill.

**What is not a kind.** Managed memory is not used and not modelled, and
nothing here should grow a `Managed` marker. Pageable host memory is `Vec`,
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

One module, `arena`, the only one outside the `-sys` wrappers with
`unsafe_code` allowed ([`programming_model.md`](programming_model.md),
"Capabilities are enforced by lints"). In it:

- `assign(rows) -> Table`, the pure offset assigner: tier 3, fuzzed, its
  total the committed golden ([`testing.md`](testing.md), "Regression").
- `Block<K>`, with `K` one of two zero-sized markers, `Device` and `Pinned`.
  It holds the raw pointer and the table, allocates in `new` and frees in
  `Drop`, and the kind is in the type so a device row cannot be handed to a
  host write.
- `Block::<K>::partition(&mut self, task) -> Bufs`, called by the
  orchestrator once per node per run: the task's rows as disjoint slices,
  `split_at_mut` fashion, so no task ever holds the block. A device row is a
  pointer-and-length view with no `Deref`; a pinned row is a real
  `&mut [T]`. The `debug_requires` that the id is in the task's layout lives
  here.

`run(bufs, stream)` receives the `Bufs` and nothing else that can allocate.

## What is checked

| Rule | Check | Tier |
|---|---|---|
| No two live rows alias | The assigner fuzzed as a pure function over random rows and orders | 3 |
| The block is the plan | The arena's total, committed and compared bitwise per example config | 3 |
| `bytes(width)` is monotone | Bisection against a linear scan over random dims | 3 |
| Admission refuses what does not fit | A cap one byte under the peak refused naming the rows live at it; one byte over admitted, as the C++ nested cases pin the room to the byte | 3 |
| A task reaches only its rows | `debug_requires` on `partition` | 2 |
| Declared equals allocated | Free memory before and after instantiation differs by the block, under the reserve's growth | 4 |
| No library allocates behind the arena | The same measurement across a run: free memory unchanged from the first launch to the last sync | 4 |
| A kernel assumes no more scratch than it asked | The wrapper handed exactly its query's bytes under `memcheck` ([`testing.md`](testing.md), "The C++ boundary") | 4 |
| A row is written before it is read | The debug fill, through every tier-4 test's comparisons | 4 |

What the C++ footprint tests measure, a pool peak held to a formula at two
bars, has no counterpart: there is no second number to compare the first
against.

## Open questions

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
