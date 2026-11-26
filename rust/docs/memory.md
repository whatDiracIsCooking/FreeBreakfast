# Rust rebuild: memory

Two layers are in the tree, and this document describes both until the old
one is deleted. **The v0.0.4 layer** (`fb_core_budget`, over `fb_core_pack`
and `fb_core_device`) counts what each step allocates, as it allocates it,
against a cap per kind, and packs the steps' scratch by the DAG: "The
budget", "Step scratch" and "The final node" below, ported from the
prototype's design (`../experimental/rust/v0.0.4/README.md`, "The memory
budget", "Step scratch"). **The old layer** (`fb_core_arena` and
`fb_core_planner`) lays every buffer out as a row of a table and admits
the table's block: every section from "The table" on, which stays as the
old layer's until its crates are deleted
([`programming_model.md`](programming_model.md), "Memory: layouts and
arenas").

## The budget

A step's `mem_budget` allocates what its `Output` owns through a `Budget`,
the handle the ledger gives it, and `run` is given none, so a run
allocates nothing. Each call, in order:

1. Adds the bytes to the kind's atomic counter and to the step's row of
   the table.
2. Checks the counter against the cap. Past it, sets the ledger's `over`
   flag. It does not panic.
3. Allocates, unless the flag is set. Under the flag, or on a dry budget,
   the buffer is empty: length 0, nothing promised.

| Kind | Counter | What `Budget` hands back | Rule |
|---|---|---|---|
| Device | A sum, per gpu | `DeviceBuf<T>`, a fixed-size device buffer | An `Output`'s: read by other steps, or carried over a replay |
| Pinned | A sum, per gpu | `PinnedBuf<T>`, page-locked, filled with `0xFF` when made | As device |
| Heap | A sum, one for the host | `HeapSlab<T>`, a `Box<[T]>` | Cannot grow; pages touched in `mem_budget` |
| Scratch | Packed, per gpu and kind | A `Lease` on the step's region for its `run` | "Step scratch" |
| Handle workspace | `fb_core_device::HANDLE_WORKSPACE_BYTES`, 128 MiB per gpu | A `DeviceBuf<u8>` the gpu's `GpuResources` takes (`Ledger::take_workspace`) and binds its handle to once | Allocated by the final node |

**The total is a sum and so order-independent.** The steps budget in
parallel, in whatever order the executor runs them; the counters are
atomic, and the refusal is deferred to the final node, so accept or refuse
is the same every run and the report is the whole table, not whichever
step crossed the cap first.

**The heap is a boxed slice, touched where it is made.** `Box<[T]>` has no
`push`, `reserve` or `extend`, so "no growing" is the type's, and only a
`Budget` makes a `HeapSlab`. Huge pages are asked for first (`madvise`,
advice only), then every page is touched: zeroed memory is handed out
lazily, and an untouched slab would "succeed" and fail at first touch
mid-run. The host cap is checked at the call, before the slab exists, so a
slab past it is counted, sets the flag and is never touched.

**Host-visible bytes are plain data.** A pinned buffer, a heap slab and
every tenant of a lease are read as a `Plain` type, sealed to the
primitives every bit pattern of which is a value, so a NaN-filled or
reused region is still valid to hold. Pinned memory is filled with `0xFF`
when made, a host-backed plan's zeroed, so nothing the host reads is
uninitialised.

**The host end of a copy is held by the copy's token.** `PinnedBuf`,
`HeapSlab` and a lease's `HostPart` implement `fb_core_device::HostSlice`
([`gpu_layer.md`](gpu_layer.md), "The v0.0.4 runtime"), and are slices on
the host too. A copy takes one by value and returns a token holding it
until a sync of the stream has covered the copy, so the host cannot touch
the bytes meanwhile. A token dropped unsynced leaks its host end: a
`HostPart`'s bytes stay carved until the lease drops after the step-end
sync, and an owned buffer is never freed.

### Caps and margins

- **A defaulted device cap** is the gpu's free memory, read once after its
  handles exist, less the larger of 512 MiB and 5% of its total
  (`device_margin`); gpus sharing one device split both evenly.
- **A defaulted host cap** is the host's available memory, read once in
  setup, less the larger of 1 GiB and 5% (`host_margin`). The reading is
  the layer's own, `host_available`, and names nothing of the arena's:
  `MemAvailable`, lowered by every cgroup v2 from the process's own to the
  mount whose `memory.max` less `memory.current` is smaller. Heap and
  pinned bytes are one counter against it.
- **Every cap is clamped to the reading.** An explicit cap is used as
  written, the user owning the margin, unless it is past the machine; then
  it is the reading itself, no margin taken. The table prints each cap as
  written beside the value used (`Cap::resolve`, `Caps::read`). Without the
  clamp an impossible cap ends in an abort at whichever allocation fails
  first, or in the OOM killer.
- **The margins are judgment, not measurement.** They cover what the
  budget leaves out: the device context, the handles' own memory, loaded
  modules, the kernels' stack and small host allocations such as the
  geometry and the basis. An explicit cap leaves those to the user.

## Step scratch

Memory a step needs during its `run` and nothing reads afterwards: a
library call's work buffer, a solver's subspace, build intermediates, the
records and work lists. A step declares it in `mem_budget`,
`budget.scratch(gpu, kind, bytes)`, which records a row and charges and
allocates nothing; the final node packs it, and `run` is lent the region.

**Two steps share scratch bytes only if the DAG orders them**, one
reaching the other (`fb_core_pack`). A finished node is finished device
work, so a path is a happens-before on the device as on the host, under
any schedule; steps the DAG leaves unordered never share, whatever lock
serializes them. The packer is first fit, largest first and then by step
name, each region at the lowest offset clear of every unordered region
already placed, so the offsets are the same every run. Its quality is not
its safety: any placement that keeps unordered regions apart is safe, and
a poorer one costs bytes. Beside the peak it reports the plain sum and the
lower bound, the heaviest set of regions no two of which the DAG orders,
so a DAG that packs badly shows itself.

**The step-end sync carries memory safety.** Pinned scratch needs it
whole: the host writes pinned bytes inside `run`, and the device reads
them through copies done by the time the wrapper's sync returns. Any
change to the sync is a change to this rule.

**One layout per step.** A step's region holds several tenants. A
`Layout` places them, each `Span` at the next 64-byte boundary;
`mem_budget` declares `Layout::bytes` and `run` builds the same layout
and carves its spans from the `Lease`, `device` for a `ScratchBuf` and
`pinned` for a `HostPart`. A span past what was declared is refused at
the carve, always on.

**A lease cannot escape, and its tenants never meet.** A lease borrows
the ledger for one `run`, and what it carves borrows the lease, so no
tenant can be stored in an `Output`. A span is neither `Copy` nor
`Clone`, each carve is recorded, and one that meets a tenant already
carved panics, so two layouts over one region cannot hand out one byte
twice; a step holds one lease at a time.

**Addresses hold over a replay; contents do not.** A step gets the same
region on every run, and what is in it on entry is undefined. With the
ledger's `nan_fill`, the wrapper fills each lease's host regions with
`0xFF` before `run` (`Lease::fill_host`), a NaN to every `f64` read before
it is written, and its device regions on the step's stream
(`Lease::fill_device`).

## The final node

`Ledger::finish` is the budget DAG's last node, over the steps and what
each reads:

1. Takes the DAG's transitive closure and packs each gpu and kind's
   scratch rows into one slab, rows in step-name order. Every region is
   checked inside its slab, always on, whatever the packer did.
2. Adds each packed peak to its counter, and the handle workspace to each
   gpu's device counter, and checks each against its cap again; holds heap
   and pinned, the packed pinned scratch included, against the host.
3. Seals the ledger: a `Budget` call after it panics, naming the step.
4. Refuses or accepts. Over any cap, it returns an `Allocation` error
   whose message is the whole table and allocates nothing more: the one
   place in the layer that error is a value
   ([`diagnostics.md`](diagnostics.md), "Allocation is a planning-time
   error"). Otherwise it allocates the slabs and each gpu's workspace,
   aborting as every create does if the machine refuses bytes the budget
   accepted, and logs the table at `info`.

**The table** is one row per step and column (device and pinned per gpu,
the heap) for what the `Output`s hold, their sum, each slab's packed peak
beside its plain sum and lower bound, the handle workspace, the totals,
each device cap used beside as written, the host's total against its cap
used and as written, each slab's regions with their offsets, and the locks
each step declared. Rows are in step-name order, so the text is the same
under any budget order.

**A lock's rank is its type's.** A step declares the locks its `run` takes
(`Budget::locks`), each a `LockId`: every manager first, then the gpus by
index, `Ord` derived in that order. `Ledger::locks_of` hands a step's set
back in that order, for the wrapper to take ascending.

---

*The sections from here to the end describe the old layer, the arena and
the planner, and go when they do.*

The programming model fixes the principles: a task's layout is its
footprint, the arena is a table and a block, the block is the plan's peak,
and a run allocates nothing ([`programming_model.md`](programming_model.md),
"Memory: layouts and arenas", "The arena is a table and a block", "Blocks,
plans and the process", "Memory kinds"). What follows is the level below,
the one a planner is written from: what a row of the table is and who owns
it, how a lifetime is represented and when two rows may share bytes, the
admission arithmetic, how an adaptive width is chosen, how the three kinds
are allocated, and what checks each rule. Every rule here is fixed by a case the
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

The planner's output for one worker and one memory kind, device or pinned,
and for the plan's heap, is a table: one row per buffer, with its owner, its
bytes, its alignment, its lifetime and the offset the assigner gave it, and
the block's total. A `run` touches memory only through a row. Four owners,
and the owner is what decides the lifetime:

| Owner | Row | Lifetime | Example |
|---|---|---|---|
| A task | Its scratch, from `layout(dims)` | The task: its first launch to its last | A kernel's `*_workspace_bytes`, a solver's `lwork` and its `info` flag, DIIS's coefficients for one push |
| An edge | Its producer's output, from the producer's `layout(dims)` | The edge: the producer's first write to the last consumer's last read | S, H_core, the converged C, B and L; on the heap, the SCF decision's `last build` |
| A worker | A buffer the worker holds for every task it runs | The plan | A BLAS handle's workspace |
| The orchestrator | A buffer placement created | The edge it carries | A broadcast's copy on the consumer's card, a reduce's partials on the join's card |

**The heap's rows are the plan's.** A CPU node's heap rows are in one
table for the whole plan, not one per GPU worker, laid out like any other
and owned as above, a task's scratch or an edge's output; a loop's region
and the rows it carries work as in the other kinds ("Nested plans"). Its
footprint is the table, as every kind's is: there is no second formula
([`programming_model.md`](programming_model.md), "Memory kinds").

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

**A copy between kinds is the planner's.** A copy that is a node's whole
work, a pinned row up to the device or a device row down to a pinned one,
is a copy node the planner adds: `Dag::upload` and `Dag::download`, one
line for the caller, each returning the new row, named after it
(`upload <row>`, `download <row>`), and timed, admitted and ordered like
any other node. Rows that one reader waits for together come down in one
node, `Dag::download_rows`, the rows of one element type each named by
the caller and the node by a label (`download <label>`): one event and
one wait for them all, where a node per row costs an event and a wait
each. Its rows are ordinary rows, the source read and the
destination an output, so admission counts them and the edges order them.
A heap row reaches the device and comes back only through two more:
`Dag::upload_heap`, a range of a heap row up to a new device row, and
`Dag::download_heap`, a device row down into a range of a heap row. Each
is two nodes through a pinned window of the range's length, the
operation's own row (`<row> window`): a CPU node and a stream node
(`stage <row>`, then `upload <row>`; `download <row>`, then `unstage
<row>`), since the device copy can start only once the host copy is done,
and the host copy of a download only once the device copy is, and either
wait inside one task would be a host sync
([`programming_model.md`](programming_model.md), "Granularity"). As two
nodes, one batch's host copy runs under another's device work. In a loop,
iteration `k`'s range is the given one moved on by `k` of its lengths, so
a row cut into ranges streams through a counted loop as one call, range
`k` in trip `k`, its window and device row the body's and held once per
iteration in flight; the heap row a download writes is one its loop
carries, written by its one user in the body. A range that runs past the
row's end is clamped to it, the window and the device row keeping their
full length, so the last of `total.div_ceil(len)` trips moves the
`total - k len` elements left. A device row that is a matrix narrower
than the heap row's, a batch of B's columns into B, comes down through
`Dag::download_heap_pitched`, the same two nodes with the window's runs
written at a pitch, each trip's moved on by its run and clamped to the
matrix's columns, so a short last batch is its loop's last trip, as the
tasks before it take their count from the iteration.
A task never writes such a copy by hand; a copy inside a task's larger
work, a solver's `info` flag down beside its launch, stays the task's.
On one card the copy node is the transport; between cards, placement's
transport rows above are the same node on the consumer's side.

**A CPU task writes into the pinned row.** A pinned row is a real
`&mut [T]`, so a CPU task whose result is bound for the device computes
it in place, never into a `Vec` it then copies in: U is written by the
`harmonics` task straight into its row.

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
- **Across streams**, the lanes of one GPU worker
  ([`programming_model.md`](programming_model.md), "Workers are a
  hierarchy"), two positions are ordered only through an event one lane
  records and the other waits on, or through a chain of edges that implies
  one. Two buffers on different lanes may share bytes only when such a path
  runs from one's end to the other's start, and the planner makes one
  wherever the assigner reused bytes: each handoff below is an edge, and
  between two lanes an edge is that event, so the path is added uniformly,
  not looked for.
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
assigner, read before the bound is moved. Beside the 3%, each laid-out
table, each copy of a counted loop's body, may hold one alignment less a
byte more: the bound credits the largest padding as if its row lay last,
and the assigner orders rows by position and size, not padding, so the
row that ends the block may carry less. Without it the packed B build's
one-column batch (#1691), a body of a few KiB, came to 4,256 bytes against
4,120, the 232 bytes of padding credited less the 96 of the row it ended
on; on every other golden the term is not needed.

**"Precedes" is the plan's rank order, so reuse adds edges.** The positions
the assigner packs against are the nodes' ranks, the order they were added
in, but a run starts any ready node whatever its rank: on one stream a
higher rank runs first when a lower one is still waiting on a CPU
predecessor. So the table returns its **handoffs** (`Table::handoffs`):
for every pair of rows that share bytes, each user of the earlier paired
with the later one's producer. Each handoff points forward in rank and is
added to the plan as a dependency, which is the rule above, an offset's
previous owner counts as a predecessor, made uniform. On one lane a
handoff between two stream nodes costs nothing, the stream's order already
holds it; between two lanes of a worker it is an event on the device, no
host wait, that serializes the two lanes there; from a CPU node it is the
wait the rule asks for. The positions are the ranks whatever lane a node is
on, so the assigner packs to the fewest bytes and pays for every reuse
across lanes in concurrency: bytes win, always, as in the C++ mmd, whose
classes' sorts share one carve, "which is why the forks chain". A tier-3
property test runs random DAGs over one to three lanes, enqueued in random
orders consistent with their edges and handoffs, each node a random time on
the device, its lane's stream and its edges' events alone ordering it, and
checks that no two rows sharing bytes are live at once.

**An alias is free of synchronization.** The ERI step is the case. Issue
1153 found the DF chain's packing broken because B was allocated while the
metric pair's frees were still pending on the stream, and the pool took new
chunks; the fix was a host synchronize before B. Under the table a row
starts after the last read of the bytes it takes, on the same stream, so
it takes them with no wait at all: L after M's last read, each batch of
the streamed build after the last batch's. The step's README calls that
sync a departure from the no-wait ideal; here it is gone.

**No row is modified in place.** A task writes only the rows it produces,
and a loop's last user the row it carries ("Nested plans"); a row it reads
is read-only to it. The C++ fit overwrites the metric with L and the
tensor with B, so here the Cholesky copies M into L and each batch's fit
writes an output of its own, and while each runs both are live. A
read-write row, a task that takes a row and hands it on under a new name,
was weighed and declined (#1536), because it never lowers a peak. The DF
step no longer holds B on the device at all: B is a heap row, built a
batch of kept shell pairs at a time (`add_packed_b`) and streamed back in
aux batches by every J and K (`add_streamed_jk`), so the device holds L,
the records and one batch's ring. On the ERI step's water fixture
(cc-pVDZ, cc-pVDZ-JKFIT, `nao` 24, `naux` 116, threshold 0, every class in
one batch), with each row padded to the alignment and beside the 32 MiB
BLAS workspace, the plan's peak is 34,185,856 bytes at the metric's
spherical transform, where the resident chain's was 35,687,088 at the
three-center task, which held the Cartesian tensor, its half-transformed
copy and the spherical output at once, three `naux · nao²` rows. Nothing
of the step is `naux · nao²` or `naux · n_columns` on the device now; the
peak grows with `naux²` (M, L) and with the batch, which admission
narrows to the room ("Adaptive widths: one planner routine"). The
question is worth reopening only when a transform in place would lower
the metric's peak; the row to build then is `Layout::write` with its
loop contract relaxed to "no later user", not a new kind.

## Admission

Per worker and per kind, the block must fit:

| Kind | Must hold | Terms |
|---|---|---|
| Device | `block + reserve + stack <= cap` | `reserve`: the context, the handles and the loaded modules, outside any block; `stack`: the kernels' stack the driver takes back when the card has no free memory left for it (`kKernelStackBytes` per thread, up to 1.6 GiB on the RTX 3080) |
| Pinned | `block <= pinned_cap`, and `pinned_cap` times the node's workers `<= available` | The per-GPU cap, and the C++ `check_pinned_node` carried over, with no margin, as there |
| Heap | `block <= max_host_mem` | The plan's cap, one block for every worker |
| Host | `sum of the node's pinned blocks + heap block <= available` | Neither kind is paged out for the other: pinned is locked, and the heap block's pages are committed by its fill; no margin, as the pinned row's |

**The available memory is the cgroups' too.** `available` is the node's
`MemAvailable`, which inside a container is the host's, so under cgroup v2
memory limits it is the smallest of that and what each limit leaves:
`memory.max` less the cgroup's usage, `memory.current` less what it could
reclaim (`fb_core_arena::host_available`). Every cgroup from the process's
own up to the root of the mount is read, so a lower limit on an ancestor,
such as a systemd slice above a container's cgroup, wins. A limit of `max`,
or none, leaves `MemAvailable`. The walk stops at a directory it cannot list,
and at the mount's root: a cgroup namespace hides the parents above it, and
a limit there is not read, the one gap left.

**Reclaimable is half the cache and half the slab.** The kernel's own
`MemAvailable` estimate, `si_mem_available` in `mm/show_mem.c`, counts the
page cache on the file LRU lists (`active_file` plus `inactive_file`) and the
reclaimable slab as available, each less the smaller of half of it and the
node's low watermark. The same terms are in a cgroup's `memory.stat`
(`Documentation/admin-guide/cgroup-v2.rst`): `active_file`, `inactive_file`
and `slab_reclaimable`; shmem sits on the anon lists and counts used. A cgroup
has no low watermark of its own, so each term is counted at half, rounded up:
what the kernel counts when the watermark is at least that half, and less than
it counts otherwise. A cgroup's figure is then never more hopeful than the
node's rule, and errs, when it errs, by at most half its cache and slab.

**No headroom and no eighth.** The 64 MiB fragmentation headroom and the HIP
`ceil(P / 8)` reuse rounding are properties of a pool that reserves in chunks
and serves a request from a larger freed block. One plain allocation has
neither, and both terms are dropped. What stays is what sits outside every
block: the reserve, a floor measured on the 3080 and grown as modules load,
and the stack.

**The reserve is measured, with the constant as its floor.** At process
start, after the devices' handles exist and before any block, each worker
reads its device's free memory against its total; the difference is that
device's reserve, and the C++ `kDeviceReserve` of 256 MiB
(`fb_pipeline_config::DEVICE_RESERVE`) is the least it may be taken as. A card
another process holds part of shows up here, as a smaller cap, rather than at
the first allocation. The binary makes the reading at planning, once per
worker, the first time planning (for a worker holding an adaptive part) or
admission (for a worker with a device block) asks for it, with that worker's
stream and handles made for it (the run's own are made with its backend,
after the blocks); it is the one reading of the card
(`fb_pipeline_config::Card`), so the card's bound below, the reserve and the
room the adaptive widths are bisected in ("Adaptive widths: one planner
routine") agree.

**The stack is what it can still take.** The C++ `Device::stack_growth`: what
the kernels' stack can still take for a kernel of `kKernelStackBytes` per
thread (`fb_pipeline_config::KERNEL_STACK_BYTES`, the C++ tree's largest,
which bounds the Rust forks of its kernels and the rust-only conventional
ERI units, at most 3,856 bytes in their low-stack form), 0 once a kernel that large has
run, and 0 under HIP, whose scratch is sized per dispatch. wwr's C ABI has no
device attribute, so the Rust term is measured rather than computed from the
resident threads: the free memory read either side of raising the stack limit
to that size, and the limit set back (`fb_core_wwr::runtime::stack_growth`).

**Each cap's default is its floor.** The caps are the config's `[memory]`
keys, the C++ ones with the C++ size strings: `max_device_mem_per_gpu`,
default 8 GiB, and `max_pinned_mem_per_gpu`, default 2 GiB; and one the C++
does not have, the heap block's `max_host_mem`, default 16 GiB. Each key's one
compiled-in value is both its default and its floor: a config with no key
runs at it, and one that sets a key under it is refused, naming the key and
the value, so a key can only raise a cap. A floor is what lets a compiled-in
size, a streamed tensor's batch, be relied on to fit without a search; as the
default it means a card with more than 8 GiB is used past that only when the
config says so.

**Every cap is bounded by the card.** A device block is admitted against the
smaller of its cap and the card's total memory, less the reserve and the
stack. The reserve being the total less the free memory, a cap past the card
admits its free memory less the stack, so a card under its cap, or one
another process holds part of, still refuses the plan before the first
launch. The binary admits every worker's device and
pinned table and the heap table after planning and before it makes the
blocks, the node's caps before them and its blocks after
(`fb_pipeline_config::admit_plan`).

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
would miss a row the peak does not hold: a row live only before or after
the peak, under a cap short of it, would go unnamed, the resident DF
step's B under its three-center peak the case that found it.

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
an `allow` naming its exception: an allocation's refusal ("The three kinds"),
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

**The Rust shape.** A layout is called as its task is added, before the
lifetimes of the rows live at its position are known, so the widths are
chosen by planning the DAG's declaration more than once. A part's two
functions are a type's `Adaptive` implementation (`fb_core_planner`),
with a floor below which a narrowed width is a `warn` (2 by default, so a
width of 1 is; the streamed J/K batch's is 32) and whether the width is
rounded down to a power of two, which keeps the fit since `bytes` is
monotone; a part with a pinned window that grows with the width declares
its bytes too, `pinned_bytes(width)`, monotone as `bytes` is.
`Dag::plan_within(rooms, build)` takes a closure giving each worker's
`Room`, its device and pinned bytes, and a closure that declares the DAG,
asking `Dag::width` for each part's width just before adding its tasks; the
part's position is the next task's, and a counted loop over its batches
takes `total.div_ceil(width)` trips, the last one ragged when `width`
does not divide `total`: `Dag::upload_heap` clamps that trip's range to
the heap row's end, and the batch task reads the
`width.min(total - k width)` it holds from its iteration `k`. The declaration is run once per part
and once more, and stops at its first error: in run `k` the parts before
part `k` have their widths and the rest are at 1, and part `k`'s fixed
bytes of each kind are that kind's table's `live_at` its position, a loop's
region whole, less the part's own at width 1, the region of a loop that
starts at its position as laid out, else its declared bytes, so a part
whose bytes bound its region from above is measured against the rows
really beside it; its room is the worker's less
those, of each kind, bisected once by `widest`, the largest width whose
bytes fit both. A worker's room is asked once, when the first part on it is
measured, so a DAG with no part reads none and the declaration's refusals
come first. The last run is the plan, its text carrying a `width` line per
part. A part inside a decision loop's body is measured against that body's
region whole, which may leave room the pass does not find. A DAG planned by
`Dag::plan` with no rooms gives each part its `width_max`.

**The rooms are admission's caps, from the one reading.** A worker's device
room is what admission holds its device block to, the smaller of
`[memory] max_device_mem_per_gpu` and the card's total, less the reserve and
the stack, all from the worker's one reading of its card ("Admission"); its
pinned room is `max_pinned_mem_per_gpu`, what admission holds its pinned
block to (`fb_pipeline_config::Card::room`). The binary's planning path,
`fb_pipeline_config::plan`, hands those rooms to `Dag::plan_within`, and
`admit_plan` takes the same reading, so a plan built inside its rooms is
admitted, and planning and admission cannot disagree about a card whose
other tenants moved between two readings. The pinned room is counted, not
left to admission, because the streamed J/K fold's window is pinned and
grows with its batch: a ring of two batches of packed B, `2 width npair`
doubles, beside an uploaded copy of each on the device and the batch's
square slab. Its pinned bytes are never more than its device bytes, but the
pinned cap's floor, 2 GiB, is a quarter of the device cap's, so past about
2,000 AOs at a batch of 64 the window, not the device, binds. What is left
to admission is the node: the workers' pinned caps together are refused past
the host's available memory before any table, and the pinned blocks with the
heap block, B itself in the streamed fold, are admitted after planning, so a
heap block that leaves the node less than the windows refuses the plan
rather than narrowing them.

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
allocated, is admission's: L's row and the build's and the streamed
pass's rings are in the device table before the first launch, B's heap row
in the heap block, and a fitting set that does not fit is refused at
planning with its bytes named.

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

**A counted loop's region is `R` copies of its body's table.** A counted
loop ([`programming_model.md`](programming_model.md), "Plan once, run
many") has no decision, and up to `R`, its depth, of its iterations are in
flight at once. Its body's table is laid out once, alone, as a decision
loop's is, and its region holds it `R` times, a stride apart, the body's
total aligned: iteration `k` runs in copy `k mod R`, so admission counts
`R` bodies once at each of the loop's positions, by the same name. Two
iterations in flight hold different copies, and the copy iteration `k`
takes back was last used by iteration `k - R`, on which iteration `k`
waits, the ring. Within an iteration the body's rows share bytes as any
table's do. A row outside the body that shares bytes with any copy is
ordered against it as against one, whatever copy it meets. A body row has
a borrow flag per copy, so two iterations' views hold their own.

**What a counted loop writes and does not own has one user in its body.**
Two iterations in flight would otherwise reach the row's one copy in no
order. An accumulator, K summed over batches, is produced before the loop,
read and written by its one body node, which runs after itself in the
iteration before, so the batches add in order; an output read after the
loop is its producer's in the last iteration. `TableBuilder::build`
refuses any other.

**Regions nest: a counted loop in a decision loop's body has its region
in that body's table.** The one nesting built is a counted loop in a
decision loop's body ([`programming_model.md`](programming_model.md),
"Plan once, run many"), J and K over a streamed B every SCF iteration. A
row is the body of the innermost loop that holds all its users. The
counted loop's body is laid out alone, as at the top level, its region of
`R` copies is one row of the decision loop's body table, live over the
counted loop's span, and that body is laid out alone around it, its region
a row of the parent. A row's lifetime is extended over every loop it
overlaps that is not its own or one holding its own: a row of the decision
loop's body used before and after the counted loop is live across the
whole of it, so no copy takes its bytes, and a row outside both is live
across the decision loop. Admission counts the outer region alone, the
inner inside it. A handoff or read leaving the decision loop is taken from
its decision; one leaving only the counted loop, to a later node of the
decision loop's body, from the writer's last trip. The accumulator above
may be a row of the decision loop's body: K zeroed in the SCF's body before the batches, written by the counted
loop's one user, and read after it, every iteration.

**A loop's body may span workers.** Its tasks may be on several GPU
workers, on several devices, as the body of a multi-GPU SCF is: each
worker's table lays out that worker's body rows in a region of its own,
the decision, a CPU node, waits for every worker's body nodes, and a row a
worker carries from one iteration to the next is that worker's, in its
table, by the rule below. Rows cross between the workers only through
collectives ([`programming_model.md`](programming_model.md),
"Collectives").

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

## The three kinds

| | Device | Pinned host | Heap |
|---|---|---|---|
| Owner | The GPU worker | The GPU worker whose copies it stages | The plan |
| Raw call | `wwr_malloc`, plain | `wwr_host_alloc`, portable | The global allocator, once |
| Thread | The one that makes the plan, its device made current for the call, before any stream work on it; the worker's own thread once one outlives a run | The same; the worker's thread would put the pages on its socket | The one that makes the plan |
| Freed | At the plan's drop, after the last run's final sync | The same | At the plan's drop |
| Alignment | `FB_DEVICE_ALIGNMENT` for every row ([`gpu_layer.md`](gpu_layer.md), "Scratch: the workspace query") | The same; the copy engines need no more, and `kWorkspaceAlignment` is 256 on both sides today | The pinned alignment |
| Fill | Every byte `0xFF` at instantiation, in debug builds: a NaN in both float widths, and `-1` in every integer width | The same, in every build | The same, in every build, which also commits its pages when it is made |
| Failure | Aborts, as every create does (`check_or_abort`), naming the bytes and the free memory | The same, naming `MemAvailable` | The same |
| Held by | A GPU node | A GPU node, for its copies, and a CPU node | A CPU node |

**Pinned is staging; large flat host data is a heap row.** A pinned row
exists to feed or drain a copy on a stream: page-locked memory is for the
copy engines, and host state no copy meets, a decision's carried energy, is
locked for no reason. Host data too large to lock, an offloaded B past the
pinned carve, needs a counted place too. So large flat host data is a heap
row, a slice of an `Element` sized at planning; when it must reach the
device, a pinned row is the window it streams through. **Structured
planning input stays `Vec`**: the config, a `BasisSet` and the planner's own
tables exist before any table can be sized, and fix its dimensions, so they
are not rows and are not admitted. **No allocator**: the heap block is one
global-allocator call at instantiation and freed at the plan's drop, as the
other blocks are, so a run allocates nothing on the heap either.

**No node holds all three kinds.** A GPU node holds device and pinned rows,
a CPU node pinned and heap rows ([`programming_model.md`](programming_model.md),
"A task declares"). A heap row reaches the device only through a
CPU node's copy into a pinned row, which a copy node then uploads.

**A plan's exit is the one copy a run makes.** What a run hands its caller
after it, the SCF's orbitals or the overlap's S as a `Vec`, is copied out of
its row by the CPU node that reads it last (`memcpy_p2h`): that copy is the
one allocation a run makes, outside every block, once per run and never per
iteration. A product written to disk is not an exit: the writer borrows its
row (`fb_io_npy::write_npy` takes a slice), so the row's bytes are the
file's.

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
the device fill, but not the host ones: a pinned or heap row is handed out
as a `&mut [T]`, which must not be uninitialised memory in any build.
Pinned blocks are small staging, so their fill costs nothing worth
measuring; the heap block's commits its pages at instantiation, so the
memory admission counted is held before the first run, not found missing
at a first touch inside one. The
same reason restricts a row's type to an `Element`, a primitive for which
every bit pattern is a value, since a reused row is read with whatever its
previous owner left.

**What is not a kind.** Managed memory is not used and not modelled, and
nothing here should grow a `Managed` marker; clippy's `disallowed_methods`
lists `wwr_malloc_managed` and `wwr_stream_attach_mem_async`, and no site
allows them.

**The three pinned layouts that exist.** Every pinned use in the C++ tree
is staging sized by the dimensions: mmd's shell tables on their way up, the
Coulomb prune's kept counts on their way down, and the Molecule's charges
and coordinates. `CountedPinned`, the offloaded B streamed through a ring,
has no user yet. So the first pinned rows are small upload and readback
buffers, and the ring, when it comes, is a pinned edge row with the lifetime
of the copies that read it, the window a heap row of B streams through
(`experimental/gemm_host_offload`).

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
- `Block<K>`, with `K` one of three uninhabited markers, `Device`,
  `Pinned` and `Heap`. It holds the raw pointer, the table and one atomic borrow flag
  per row, allocates in `new` and frees in `Drop`, and the kind is in the
  type, as it is in every `RowId<K, T>`, so a device row cannot be handed to
  a host write. A plan's per-worker host table is of a `StagingKind`,
  `Pinned`, or in the device-free tests `PinnedStandIn` (the arena's
  `test-support` feature), a staging kind on the global allocator; never
  `Heap`, so no two tables of a plan share a kind and a heap row cannot be
  asked of a worker's table.
- `Block::<K>::partition(&self, node) -> Bufs`, called by the orchestrator
  once per node per run. A pinned row is a real `&[T]` or `&mut [T]`; a
  device row is a pointer-and-length view with no `Deref`. A stream node's
  pinned view is wrapped in a `CopyOnly`, which yields no slice and which
  only `memcpy_p2d` and `memcpy_d2p` take.
- `core::planner`'s `Dag::upload`, `Dag::download` and
  `Dag::download_rows`, the copy nodes ("The table"): the only `GpuTask`s
  whose `run` is a copy between kinds alone, so a pinned-to-device or device-to-pinned move is one line at
  its caller, on whichever lane of the worker the caller names. Its
  `Dag::upload_heap` and `Dag::download_heap` move a range of a heap row
  through a pinned window, and `Dag::download_heap_pitched` runs of one
  at a pitch, a CPU node and a copy node each, whose host
  halves are the only `CpuTask`s whose `run` is a copy alone: a heap row
  reaches the device through them and nothing else.

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
them. The type holds it: a GPU task's `run` is handed its pinned rows as a
`CopyOnly`, never as a `Bufs`, so a host read of one does not build.

**A CPU task holds no device row.** Its `layout` is handed a pinned and a
heap `Layout` and its `run` the two `Bufs`, so a device row is not
something it can declare or ask for. A GPU task is handed no heap view.

A GPU task's `run(device, pinned, stream)` and a CPU task's
`run(pinned, heap)` receive their views and nothing else that can allocate.

## What is checked

| Rule | Check | Tier |
|---|---|---|
| No two live rows alias | The assigner fuzzed as a pure function over random rows and orders | 3 |
| The block is within 3% of its lower bound, and one alignment per table, on every committed golden | Every golden's text read, each table's total against its live rows' padded bytes at their largest, each loop's region against its body's | 3 |
| The block is the plan | The arena's total, committed and compared bitwise per example config | 3 |
| `bytes(width)` is monotone | Bisection against a linear scan over random dims | 3 |
| Admission refuses what does not fit | A cap one byte under the peak refused naming the rows live at it; a cap under a row refused naming that row, live at the peak or not; one byte over admitted, as the C++ nested cases pin the room to the byte | 3 |
| Admission counts pinned and heap together | The node's pinned blocks and the heap block admitted at their sum and refused one byte short of it | 3 |
| No two tables of a plan share a kind | `Dag`'s host kind is a `StagingKind`, which `Heap` is not | 1 |
| A task reaches only its rows | The view's borrow flags and layout check, always on | 2 |
| Reusing bytes is ordered | Random DAGs over random lanes run in random orders consistent with their edges and handoffs, each node a random time on its lane, no two rows sharing bytes live at once | 3 |
| A loop's body lies in its region, apart from every parent row live across the loop | Random bodies laid out, then unrolled over iterations and run in random orders consistent with the reads and handoffs, no two rows sharing bytes live at once | 3 |
| Replay allocates nothing | The arena's count of this process's blocks (`fb_core_arena::allocations`), read by a device loop's decision every iteration, the same at every one: no other tenant of the card moves it, as one moves the device's free memory (#1573). A library allocating behind the arena at a launch is not in it: the next row's | 4 |
| Declared equals allocated | Free memory before and after instantiation differs by the block, under the reserve's growth: the device's reading, the one measure of the driver's cost of a block, over a window of one allocation, which a display client allocating inside it can fail (#1589) | 4 |
| No library allocates behind the arena | On an AMD card, this process's VRAM from amdgpu's DRM fdinfo (`drm-memory-vram`, per open file, once per client), which the kernel keeps per process so no other tenant moves it, and which counts every buffer of ours, the arena's and a library's: the same from the DF-RHF loop's second iteration to its last: once every kernel of the body has run, the DIIS solve's from the second, a replay takes nothing. What a kernel's first launch takes (24 KiB on the gfx1200, at the first DIIS solve) is not told apart from the code the runtime loads at a first launch, so it is outside the bar. On CUDA unenforced: nothing counts a library's allocations per process in a container, and the device-wide free memory moves with the card's other tenants (#1573), so the planner's and the SCF's cases print it and assert nothing of it (#1589) | 4 |
| A kernel assumes no more scratch than it asked | The wrapper handed exactly its query's bytes under `memcheck` ([`testing.md`](testing.md), "The C++ boundary") | 4 |
| A row is written before it is read | The debug fill, through every tier-4 test's comparisons | 4 |
| The heap block is filled in every build | A heap block read back all NaN | 4 |
| Large flat host data is a heap or pinned row, and a pinned row is read or written by a copy on a stream | Review alone: nothing tells a `Vec` of planning input from one of host data, or a pinned row a copy meets from one it does not ([`rules.md`](rules.md), "Memory") | none |

What the C++ footprint tests measure, a pool peak held to a formula at two
bars, has no counterpart: there is no second number to compare the first
against.

## Open questions

- **Bytes against concurrency.** The assigner always packs to the fewest
  bytes, and every byte it reuses is a handoff, an edge the plan did not
  otherwise have. Two independent nodes on two lanes whose rows share bytes
  are then serialized when a larger block would have let them overlap. That
  is the rule today ("Lifetimes"): on one lane it costs nothing, and across
  lanes the planner pays it. The open part is admission's search deciding
  which reuse to give up for concurrency when the cap leaves room.
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
