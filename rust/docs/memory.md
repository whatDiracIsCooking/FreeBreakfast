# Rust rebuild: memory

`fb_core_budget`, over `fb_core_pack` and `fb_core_device`, counts what
each step allocates, as it allocates it, against a cap per kind, and packs
the steps' scratch by the DAG: the design the v0.0.4 prototype was built to
try. A step's `mem_budget` allocates; its `run` is given no `Budget` and
allocates nothing ([`programming_model.md`](programming_model.md), "Steps
on rustflow").

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
| ERI buffer | The device and pinned sums, per gpu | A `DeviceBuf<u8>` and a `PinnedBuf<u8>` the `EriManager` owns, carved into tenants ("Tenants of an Output's buffers") | Within `[eri] mem_per_gpu` (2 GiB by default, a constant so a dry budget is exact), charged at what the build's ring or a J/K pass's uses, whichever is larger; the batch widths are the widest that fit, a pinned ring buffer at most 64 MiB (build) or 128 MiB (pass); under the least that holds one batch the plan is refused, naming the key, its value and the bytes needed. B is the manager's `HeapSlab`, at the unscreened bound; the pair batches partition its pair columns, checked when the list is made, and each GPU takes a contiguous run of them, so it writes its own columns of every row through slices `split_at_mut` cut, beside the other GPUs, with no shared pointer |

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

**Only a `Budget` allocates device, pinned and budgeted heap memory.**
The raw `wwr-sys` allocators are called in `fb-core-budget` alone, which
`clippy.toml` bans everywhere else, a test's own buffers each an `allow`
with its reason. Each buffer is one plain allocation, not the stream-ordered
pool; pinned memory takes the default flags, never mapped or
write-combined; and managed memory is not used, its calls on the same
list.

**Small host scratch is a policy, not a kind.** A `Vec` or a `HashMap`
that lives within a `run` is assumed small and is outside the budget: it
stays within O(n_ao²), and large host data is a `HeapSlab` on the bus. Two
checks hold it. In `fb-steps`, whose `clippy.toml` repeats the
workspace's list, `vec!`, `Vec::with_capacity`, `reserve`, `reserve_exact`,
`resize`, `extend_from_slice` and `String::with_capacity` are banned,
each exception an `allow` with its reason: setup's counts, the config's
defaults, a test's reference. And the allocation record,
`fb_core_budget::AllocationRecord`, a `#[global_allocator]` over the
system's that every test binary of the steps, and `fb-core-flow`'s
`tests/flow_device.rs`, installs: the wrapper watches
each `run`, and a step that made one allocation past
`fb_core_budget::RECORD_THRESHOLD`, 1 MiB, on the thread running it fails
the execute, named. The record records and never panics, since unwinding
out of a global allocator is undefined behaviour; it sees Rust's heap
alone, not a C library's `malloc`, and the thread running `run` alone, not
a pool's or a fan-out's. So a library that returns an O(n_ao²) `Vec`
is not called from a `run`: the host integrals and the SAD guess each have
an `_into` form that writes into a caller's slice, and a host step hands
them its pinned staging. The budget table prints an execute's peak heap
growth, as the record saw it, as its unbudgeted row. The growth is the
process's, so another thread's allocations and frees meanwhile move it,
but a second execute's start does not reset it: an execute that starts
during another leaves the peak the other reads alone, so several executes
may share one binary.

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
  handles exist, less the larger of 2 GiB and 5% of its total
  (`device_margin`); gpus sharing one device split both evenly.
- **A defaulted host cap** is the host's available memory, read once in
  setup, less the larger of 1 GiB and 5% (`host_margin`). The reading is
  the layer's own, `host_available`:
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
- **The kernel stack is not in the budget; the device margin absorbs
  it.** The runtime gives every kernel a per-thread stack for its locals
  (ROCm calls it scratch, CUDA local memory), sized at the kernel's bytes
  per lane times every lane the card can keep resident, not at the
  dispatch, and keeps it. No `mem_budget` declares it and no row of the
  table counts it: a library's kernels cannot be asked their size, and
  the bound that is true on HIP is too loose to plan by. "Scratch" in
  this document means only what a step declares, "Step scratch" below.
  - **HIP holds it per hardware queue.** There are up to
    `GPU_MAX_HW_QUEUES` of them (the runtime's environment variable,
    default 4), and a stream is not bound to one: it takes the
    least-loaded queue each time it goes from idle to busy, so one
    stream's successive batches land on different queues. Each queue
    keeps every allocation it has made, and a launch needing more than
    its queue holds takes a new one. Past the single-allocation ceiling,
    140 MiB, a launch is sized to its dispatch instead, and the next
    launch under the ceiling takes a new allocation again.
  - **CUDA holds one per context**, sized for the largest kernel run.
  - **The eri step launches its largest class first** on each ring
    stream (`stack_rank`, by `(la + lb, la)` and then the aux momentum,
    both descending; B is bitwise the same in either order). A queue
    that has run a larger launch runs every smaller one in what it
    holds, so the order takes away most of the staircase a rising order
    climbs. It bounds nothing: a queue whose first launch is a small one
    still climbs.
  - **Measured on the gfx1200** (32,768 resident lanes; `ssperi3`'s
    stack per lane from 0 to 13,072 bytes, `<3, 3, 6>`), beyond the
    budget's allocation. Water's chain in cc-pVDZ / cc-pVDZ-JKFIT takes
    128.6 to 197.1 MiB in its first execute, by which queues its
    batches landed on, and a second execute can add the largest class's
    125 MiB on a queue that had not run it: 322 MiB. In the classes'
    order it took 492.6 MiB, 246 MiB on each of two streams. All 70
    classes on one stream took 950 MiB largest first and 3.7 GiB in the
    classes' order.
  - **The bound per queue is the whole ascending staircase**: the sum
    of the plan's distinct stack sizes under the ceiling, each times
    the resident lanes, reached when a queue meets them smallest first.
    For water that is 17,428 bytes per lane, 544.6 MiB a queue and
    2.1 GiB on four (`steps/tests/chain_vram.rs` asserts it of the
    first execute and a replay together).
  - **What the 2 GiB floor of `device_margin` covers**: every figure
    measured above, and the estimate for CUDA at f, about 1.3 GiB in
    one allocation on an RTX 3080 (nvcc's local sizes were not read).
    It is not sized for the bound, and it does not cover an f-shell
    plan on HIP in which every queue ends up holding the full range of
    classes: about 3.8 GiB from the one-stream figure, about 15 GiB in
    the classes' order on all four queues. `GPU_MAX_HW_QUEUES` in the
    environment lowers the queue count. An explicit cap takes no
    margin, so its user leaves the room.
  - **When the room is not there** the run fails with an out-of-memory
    error at the eri build's first launches, before the first SCF
    iteration. Lower the device cap, or `[eri] mem_per_gpu`, by what
    the stack needs.

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

## Tenants of an Output's buffers

A manager reuses its own buffers: its rings carve the same pinned staging
and the same device buffer on every turn, typed pieces of one `Output`'s
bytes, as `run` carves its lease. `PinnedBuf::tenant` and
`DeviceBuf::tenant` take a `Span` of a `Layout` over the buffer's bytes
and return a `PinnedTenant` (a host slice, and the host end of a copy) or
a `DeviceTenant` (a device slice), several live at once.

**Each carve is recorded in the buffer, always on, and a tenant's drop
clears it.** A carve that meets a live tenant panics at the caller's
line, so two tenants never share a byte, and the same span is carved again
once its tenant is dropped. A pinned tenant moves into its copy's token
and comes back by the token's sync; a token dropped unsynced leaks it, its
record stays, and a carve that meets those bytes panics while the copy
may still read them.

**A tenant shares the allocation, and the whole is not reached beside
it.** A tenant holds the buffer's allocation alive, so a leaked one leaks
the memory rather than freeing bytes a copy may use. The carve takes the
buffer by `&mut`, so no slice of the whole is alive at it (a
`compile_fail,E0502` doctest), and while a carve is recorded, live or
leaked, every reach of the whole (the pinned slice, the device pointer a
call takes) panics.

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

## Batched only

Every step over an O(N³) tensor has a batched path and no other. The
unbatched case is the batched one with a single batch on a single GPU.

| What a step holds | Rule |
|---|---|
| Everything O(N²): the n_ao² matrices, the metric and its factor, the records | Resident whole on the GPUs it needs. It fits or the plan is refused: a dense eigensolve at n_ao cannot be batched |
| The O(N³) tensors: (μν\|P), B, a derivative of B | Never resident whole. Built or contracted a batch at a time; B itself on the host heap |
| The batch width | The widest whose bytes fit the step limit, and no wider than its staging buffer holds; none fits: refuse, naming the step's key, its value and its least bytes |
| The step limit | Covers the batches and their rings, not the O(N²) residents, which the cap admits or refuses. A fixed constant by default; the config overrides it, which is also how a test forces many batches on a small molecule. For `eri`, `[eri] mem_per_gpu` |
| What is charged | What is used, never the limit. Limits need not sum to the cap |

- **Every GPU step is built batched and for every GPU, or not built.** A
  feature is not added until it runs on a single node's GPUs, one GPU
  being the case of one.
- **Width stops mattering early.** In the prototype's single-GPU build
  (`experimental/rust/df_batched`) the time is flat past about 20
  batches, and a width of one costs 5.6× to 11.6×.
- **No "what is left".** A width is a function of the step's own limit,
  so `mem_budget` stays local and order-free.
- **A width of one is logged at `warn`**; the user raises the limit.
- **A size fixed only by an earlier step's run is allocated at the bound
  `Dims` gives**: a screened pair count, the orbital count after dropping
  linear dependencies. B's pair count is the count in the tree's own pair
  layout (270,497 pair columns on gly10, where n_ao(n_ao+1)/2 gives
  269,745). On a large sparse molecule that wastes host memory, and a
  replay can never outgrow its buffers.
- **A size a step chooses at run time is allocated at a bound set in
  setup.** No such step is built; the rule is for the first. AVAS's
  active space is bounded by its labels: at most twice the MINAO
  functions they name, capped by `avas_max_ncas`, as
  `src/pipeline/shape`'s `largest_active_space` does, and the CI is sized
  at the largest determinant count at or under that bound. A run that
  selects past the bound fails, as a `Result` naming the count it wanted;
  it never truncates, which would change the chemistry unannounced. A
  bound too large is refused in budget mode, before anything runs.

What can still be refused: a plan whose O(N²) data does not fit its GPUs,
whose B does not fit the host, whose step limit holds no batch, or whose
`Output`s, each fitting alone, do not fit summed down a chain. The last
is the sum's price; the remedy is a narrower batch in the step concerned,
and past that `Output`s packed by the closure, built when a chain is
first refused.

## What the budget measured

The prototype's numbers, of the design `fb-core-budget` and `fb-steps`
port: RTX A6000 (CUDA), 2026-11-26, a release build, one run each, cc-pVDZ
over cc-pVDZ-JKFIT; the chain's times are
[`programming_model.md`](programming_model.md)'s, "What the chain
measured".

**The budget against the old path's block.** One GPU, the same runs through
the Rust path the steps replaced, deleted since (#1875), which laid every
buffer out in one block per kind (`schwarz_threshold = 0`, its blocks as it
logged them) and through the steps, at the default step
limit of 2 GiB and at the least `mem_per_gpu` each molecule accepts (budget
mode alone, the solver work buffers added from the default run).

| Molecule | Old path: device block | pinned block | heap block | The steps at 2 GiB: device | pinned | heap | The steps at the least limit: device | pinned |
|---|---|---|---|---|---|---|---|---|
| Water | 33.35 MiB | 322.00 KiB | 292.02 KiB | 130.73 MiB | 721.06 KiB | 301.00 KiB | 129.21 MiB | 213.34 KiB |
| Ethylene | 37.44 MiB | 1.19 MiB | 2.16 MiB | 143.09 MiB | 4.85 MiB | 2.19 MiB | 130.74 MiB | 663.62 KiB |
| Benzene | 62.26 MiB | 6.90 MiB | 28.39 MiB | 307.19 MiB | 59.86 MiB | 28.59 MiB | 140.05 MiB | 3.36 MiB |
| gly5 | 906.52 MiB | 263.87 MiB | 1.00 GiB | 1009.41 MiB | 293.24 MiB | 1.01 GiB | 251.22 MiB | 37.37 MiB |
| gly10 | 3.01 GiB | 989.67 MiB | 7.27 GiB | 1.31 GiB | 383.90 MiB | 7.28 GiB | 585.07 MiB | 140.30 MiB |

- **The heap is the same**: B, packed, on the host in both.
- **The device total is the batch widths, not the sum.** the steps' is the BLAS
  workspace (128 MiB against the old path's 32 MiB) and the ERI buffer, whose
  batches the staging bounds: 792 MiB at gly5, 881 MiB at gly10. The old
  block was its adaptive widths, chosen at planning.
- **What the sum can cost is what lies outside those two**: the other
  `Output`s and the packed scratch, 89 MiB at gly5 and 329 MiB at gly10. A
  fold over lifetimes could recover part of that and no more.
- **At the least limit the sum is under the old block** on the glycine
  chains, 3.6× and 5.3×. Less each path's workspace it is on the small ones
  too: 1.21 MiB against 1.35 MiB on water, 12.05 MiB against 30.26 MiB on
  benzene.

**Budget mode and the staging**, at gly10 (734 AOs, 3606 aux), where B
is 7.27 GiB:

- **Budget mode is B's pages.** At gly10, 5.7 s on one GPU and 6.5 s on
  four. Touching B's 7.27 GiB takes 5.3 s with huge pages (7.6 s without),
  about 1.4 GiB/s. The rest is page-locking each GPU's 272 MiB of staging,
  about 0.3 s a GPU.
- **That touch is the VM's, not the rule's.** The same touch of a fresh
  7.27 GiB slab on bare metal (a Ryzen 9 5900X, kernel 6.8,
  the prototype's `examples/touch_probe.rs`) takes 0.52 s with huge pages, 14 GiB/s, and
  2.67 s without. More threads help there: 0.39 s on sixteen at 4 KiB,
  where on the VM sixteen took 12.8 s against 7.6 s on one. A 10× gap that
  threads widen points at the hypervisor backing guest memory on its first
  touch. On bare metal, budget mode at gly10 would be about 1 s. Whether a
  second slab in the same process on the VM touches at bare-metal speed is
  open ("Open questions", below).
- **Staging is bounded: 64 MiB a buffer of the build's ring, 128 MiB a
  buffer of the pass's.** A GPU page-locks two of each, and the larger pair
  is what its pinned buffer holds. The build's time does not move with its
  bound.
- **Page-locking does not overlap.** With each GPU's staging page-locked on
  a thread of its own while another touched B, the locks finished one after
  another and the touch slowed beside them: 6.8 s on four GPUs against the
  6.5 s of taking them in turn. They are taken in turn.

## Open questions

- **Packing `Output`s too**, by the same closure from producer to last
  reader: deferred until a chain is refused for its sum
  ([`programming_model.md`](programming_model.md), "What the model
  covers").
- **The default step limit for `eri`**: the measured value in place of the
  2 GiB placeholder.
- **Packing quality** on real configs: the table's lower-bound row answers
  it on each.
- **First touch on a VM**: whether the 1.4 GiB/s of B's touch ("What the
  budget measured", above) is paid once per process or by every slab. The
  prototype's `examples/touch_probe.rs` touches the same size twice in one
  process: if the second is fast, the host was backing guest memory, and a
  replay or a long-lived process pays it once. If both are slow, the cost
  is the guest's own zeroing. Either way, it is not a reason to stop
  touching the pages up front.
- **Pinned pages and NUMA**: a pinned buffer is touched by whichever
  worker runs its step's `mem_budget`.
