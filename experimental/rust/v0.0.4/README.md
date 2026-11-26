# rust/v0.0.4: the design, and the prototype built from scratch

**The design of the Rust rebuild as it stands at v0.0.4, and what v0.0.4 will
build.** A calculation is a DAG of steps on
[rustflow](https://github.com/whatDiracIsCooking/rustflow). Each step has one
function that counts and allocates its memory up front, and one that runs.
There is no planner. The prototype is built, in this directory (`src/`,
`tests/`, `examples/`); nothing under `rust/` is. `rust/docs/` describes the
tree as it is, and changes with the implementation.

**This page is self-contained.** It says what the design is now, not how it
got here. Each prototype's page carries the whole design forward as it stands
at that prototype; the earlier pages (`../v0.0.2`, `../v0.0.3`) and git
history hold the path. Measured claims name the experiment that measured them,
under `experimental/rust/` unless a path says otherwise. *Open* marks what is
not settled; everything else is the design.

What v0.0.4 builds is [the DF-RHF chain as steps](#the-v004-prototype), on
`num_gpus` GPUs, batched. Like its predecessors it is throwaway: a crate and a
Cargo workspace of its own, in no gate, never merged into `rust/`.

**It is the last prototype.** The model covers what `rust/` and the C++ `src/`
compute ([what the model covers](#what-the-model-covers)), and nothing left
open would change its core: the MainBus, `Dims` counted in setup, one budget
mode. The next work is the layer itself, in `rust/`
([the order of work](#the-order-of-work)).

## Vocabulary

Each word below means one thing on this page.

**The calculation**

| Word | Means |
|---|---|
| **step** | One node of the outer graph: a type with `mem_budget` and `run`, one config table and one slot. Something is its own step only if another step reads its `Output`, its `Output` carries over a replay, or it has a config table of its own |
| **inner task** | A unit of a step's own work inside `run`: one GPU's thread of a fan-out, one turn of its loop. Plain code, not a rustflow task, and not a step |
| **node** | A node of the outer graph; one per step, plus budget mode's final node |
| **wrapper** | The one generic function around every step's `mem_budget` and `run`: it makes the device current, takes the step's locks, calls the body, syncs, and releases |
| **`Output`** | What a step leaves on the MainBus: what other steps read, or what carries over a replay |
| **MainBus** | The one shared object: the config, `Dims`, the caps and counters, each GPU's `GpuResources`, and one slot per step |
| **slot** | A step's `RwLock<Option<Output>>` on the MainBus. Used only in this sense |
| **View** | What `run` receives in place of the bus: the `Output`s the step declared it reads, and nothing else |
| **`Dims`** | Every count setup makes: `n_atom`, `n_ao`, `n_aux`, `n_occ`, and every array length a step needs |
| **setup** | Before the DAG: parse the config, the geometry and the bases on the host, fill `Dims`, make each GPU's `GpuResources`. Not budgeted |
| **budget mode** | The DAG run once with each node calling `mem_budget`: count, check, allocate. Then the final node |
| **final node** | The last node of budget mode: packs step scratch, checks the totals, prints the budget table, accepts or refuses |
| **execute mode** | The same DAG with each node calling `run`. Allocates nothing |
| **replay** | Execute mode again at the same `Dims` (a geometry optimization, AIMD), in the same buffers |

**Memory**

| Word | Means |
|---|---|
| **kind** | Where bytes live: **device** (a GPU's memory), **pinned** (page-locked host memory) or **heap** (ordinary host memory) |
| **counter** | The running total of one kind (per GPU for device and pinned), charged as steps allocate |
| **cap** | The most a kind may hold: from the config, or the machine's reading less a margin, always clamped to the machine |
| **margin** | What a defaulted cap leaves unpromised, for what the budget does not count |
| **step limit** | A step's own bound on its batch memory; it picks the batch width. The cap still admits or refuses. For `eri`, `[eri] mem_per_gpu` |
| **flag** | The shared over-budget flag: set by the first over-cap charge, read by the final node |
| **budget table** | One row per step and kind, plus the packed scratch, the lower bound, the handle workspace and the caps. Logged every run, printed whole on a refusal |
| **step scratch** | Memory a step uses during its `run` and nothing reads afterwards |
| **slab** | One allocation of one kind on one GPU (or the host): a `HeapSlab<T>`, an `Output`'s buffer, or the scratch slab the final node packs |
| **region** | One step's part of the scratch slab, at an offset fixed by the final node |
| **layout** | A step's function from `Dims` and its options to the offsets of its **tenants**, the pieces its region holds |
| **lease** | The borrow of its region that `run` receives, for the length of `run` |
| **closure** | The DAG's transitive closure: which steps a path orders. Two regions may overlap only if the closure orders their steps |
| **lower bound** | The heaviest set of mutually unordered scratch declarations: no packing does better |
| **sum, fold** | For a calculation: every buffer alive at once (sum), and the peak with bytes reused perfectly (fold) |
| **allocation record** | The global allocator's record of large heap allocations made during `run`, checked in the test tiers |

**GPUs**

| Word | Means |
|---|---|
| **GPU** | One device, `gpu0` to `gpu{num_gpus-1}`. "Card" is not used |
| **gpu0** | The GPU that one-GPU steps run on and all-GPU results are reduced onto |
| **all-GPU step** | A step that uses every GPU, its data split across them |
| **logical GPU** | One of several GPU indices mapped onto one physical device, a test facility (`devices = [0, 0]`) |
| **`GpuResources`** | Per GPU, a `Mutex` around that GPU's stream, the BLAS and solver handles made on it, and the handle workspace |
| **the GPU's stream** | The stream inside `GpuResources`, used only by the holder of that GPU's lock |
| **a step's streams** | Streams the wrapper makes for a step, one per GPU it spans by default; the step's kernels and copies go there |
| **guard** | A held lock: on a GPU's `GpuResources`, or on a manager |
| **handle set** | One GPU's BLAS and solver handles |
| **handle workspace** | The fixed 128 MiB per GPU the BLAS handle draws on, bound once |
| **work buffer** | A solver call's `work` argument, sized by its query, a slice of the caller's scratch |
| **recipe** | The tree's own dense routine in place of a vendor call (the blocked `trsm`, the Cholesky), from `src/recipes` |
| **binding** | The new layer's function over one C ABI call or kernel: it takes slices and checks their lengths before enqueueing |

**Density fitting**

| Word | Means |
|---|---|
| **B** | The fitted three-centre tensor B<sub>μν</sub><sup>P</sup>, on the host heap |
| **pair column** | One orbital shell-pair function index μν of B, in spherical functions. Not used bare |
| **pair batch** | A run of pair columns, the unit the build of B works in |
| **aux batch** | A run of aux functions P, the unit a J/K pass works in |
| **J/K pass** | One sweep over all of B building J and K for a set of densities |
| **`EriManager`** | `eri`'s `Output`: B, the metric's factor, and per GPU the ERI buffer, its streams and their events. Reached by its readers under its lock |
| **ERI buffer** | One device buffer per GPU within `[eri] mem_per_gpu`, owned by the `EriManager`: the build's ring of pair batches, then the J and K fragments, one aux batch's work and a J/K pass's ring of aux batches |
| **ring** | The buffers of the batches in flight: two, one filling while the other is used. The build has one, of pair batches; a J/K pass has one, of aux batches |
| **staging buffer** | One pinned buffer that B passes through between a GPU and the host: a fitted pair batch on its way down, of at most 64 MiB, or an aux batch of B's rows on its way up, of at most 128 MiB. One per ring buffer |
| **records** | Per-primitive input data a kernel reads: centre, exponent, weighted coefficient |
| **work list** | Indices, one entry per unit of kernel work. `cells` is one of them, by name |

## The shape

A calculation is a DAG of steps, as in the C++ `src/pipeline`. Every step
implements `MemBudget`: `mem_budget` charges the counters for what the step
will allocate, then allocates it. The same DAG runs in two modes over one
rustflow `Executor`.

1. **Setup.** Parse the config, the geometry and the bases on the host; fill
   `Dims` with every length a step's arrays need (counted, not built); make
   each GPU's stream and handles. Not budgeted.
2. **Budget mode.** Each node calls its step's `mem_budget`: count, check,
   allocate into the MainBus. The final node packs step scratch and accepts or
   refuses the whole plan.
3. **Execute mode.** Each node calls its step's `run`. Allocates nothing.
   Replays write into the same buffers.

There is no planner and no byte reuse between the `Output`s steps pass on. A
kind's peak is the sum of the steps' `Output`s, plus packed step scratch, plus
the fixed handle workspace. Step scratch is packed into one slab per GPU and
kind, and two steps share bytes only where the DAG already orders them. No
edge is added for memory, so work stealing cannot interact with memory.

## Steps and the MainBus

The MainBus is a deliberate god object: one place that holds every step's
`Output`. A step declares the `Output`s it reads; the builder makes the DAG's
edges from that declaration, and `run` receives a View of exactly those and
not the bus. The steps and the MainBus live in one crate, so the bus can name
every `Output` type with no dependency cycle.

```text
FIGURE-01 (fig)
```

- **An `Output` holds what outlives its step's `run`; scratch holds the
  rest.** The DIIS ring, a Davidson subspace, the records and work lists, build
  intermediates: all step scratch.
- **The lock is used plainly.** The owning step takes `write()`, a reader's
  View takes `read()`. On a chain it is never contended. A thin wrapper hides
  the standard lock's poisoning `unwrap()`.
- **A step reads only what it declared.** The declaration is the edge, so a
  missing edge is a missing declaration, and that does not compile: the View
  has no such field.
- **Only the owning step writes**, by the visibility of an `Output`'s mutating
  methods within the steps' crate. The one exception is a manager's mutable
  state, which its readers reach through the manager's lock
  ([`EriManager`](#the-erimanager)).
- **A step is one config table and one slot.** `[gradient]` is one step; what
  it computes depends on the energy method the config names. No step type has
  two instances.
- **A step the config does not ask for is not in the DAG.** A step that
  declares it reads one is refused when the DAG is built, naming both. So no
  declared read finds an empty slot; that the slot is filled is a runtime
  assert.
- **Shared code with per-user state is a type, not a step.** DIIS: each user
  builds its own `Diis` in `run` over a slice of its own scratch, sized in its
  own `mem_budget`. The history lives exactly as long as the `run`.
- **An `Output` can be loaded from a file**, in part or whole, as in
  `src/pipeline`. That is how a step is tested alone. Budget mode still
  allocates; the load fills.
- **A replay allocates nothing.** What lives in scratch never carries over.
  What carries over in an `Output` (the last geometry's orbitals as the next
  guess) and what must be cleared (a converged flag) is each step's own
  business: a reset mode in the step, as v0.0.2 built it. *Open*: whether to
  make it explicit, as an `on_replay` the wrapper calls.
- **`mem_budget` is an associated function** of `Dims` and the step's options
  (`nroots`, a subspace size), both read from the MainBus.

## The EriManager

The density-fitted ERI is a shared product with state of its own: streams, a
ring of buffers on every GPU. That state lives in one object,
`eri`'s `Output`, and its users call its methods.

```rust
pub struct EriManager {
    l: PerGpu<DeviceBuf<f64>>,   // the metric's Cholesky factor, every GPU
    b: HeapSlab<f64>,            // B, n_pair (unscreened bound) × n_aux
    state: Mutex<EriState>,      // per GPU: the ERI buffer, its streams
                                 // and their events
}

impl EriManager {
    /// J and K for each density, on gpu0. Every GPU uploads and contracts
    /// its own aux batches; the fragments are reduced onto gpu0.
    pub fn build_jk(&self, held: &mut Held<'_>, d: &[Density<'_>],
                    j: &mut [DeviceSliceMut<'_, f64>],
                    k: &mut [DeviceSliceMut<'_, f64>]);
}

/// D itself, or the occupied orbitals C of D = 2 C Cᵀ.
pub enum Density<'a> {
    Dense(DeviceSlice<'a, f64>),
    Occupied(DeviceSlice<'a, f64>, usize),
}
```

A sketch, not a signature to hold anyone to. What is fixed:

- **It is `eri`'s `Output`.** `eri`'s `mem_budget` allocates all of it: L on
  every GPU, B's slab on the host heap, and on every GPU the ERI buffer and its
  pinned staging. These are summed with every other `Output`, for the whole
  calculation. `eri`'s `run` factors the metric, builds B and fills the
  manager; as the owner it reaches the state through `&mut`, with no lock.
- **Its readers declare it, like any `Output`.** The edge comes from the
  declaration. A reader that calls its methods also declares its lock, and
  the wrapper takes it ([lock order](#lock-order)). B and L are immutable after
  `eri`; only the state behind the `Mutex` changes, and only under its lock.
- **The caller owns its results.** `build_jk` writes the J and K it is passed,
  reads the densities it is passed, and keeps nothing of the caller's. A
  caller's own state (its DIIS, its Fock matrix) stays in the caller's scratch.
- **It takes a slice of densities.** DF-RHF passes one. UHF passes two, and a
  response solve, a Davidson or a state average pass many. v0.0.4 sizes for
  one.
- **A density is D, or the occupied orbitals of D = 2 C Cᵀ.** From D, K costs
  two n_aux × n_ao³ products; from C, two n_aux × n_ao² × n_occ ones, and J
  comes from the same half-transformed B. A caller that has the orbitals
  passes them: the SCF does on every pass but the guess's. Only the matrix
  given crosses to the other GPUs.
- **It never takes a lock.** The GPU guards it uses are passed in, held by the
  wrapper; it has no path to a `Mutex`.

**The ERI buffer.** One device buffer per GPU within `[eri] mem_per_gpu`,
`eri`'s step limit, charged at what it uses. `eri`'s build and every later J/K pass run in it, one
after the other, since the DAG orders `eri` before every reader:

1. **During `eri`'s run** it holds the build's ring: two pair batches in
   flight per GPU, each with its records and work lists.
2. **Afterwards** it holds, per GPU, the J and K fragments, one aux batch's
   work, and the J/K pass's ring: two aux batches of B's rows in flight.

**(μν|P) is spherical from the kernel.** FelizJueves's `ssperi3<la, lb, lP>`
(#745) writes all three indices as real solid harmonics, each pair's block at
its spherical pair column, the layout the fit reads. No bra transform runs,
and the ring holds no Cartesian batch, no transform scratch and no orbital U.
It is plain FP64: the mixed-precision and screened forms of the same kernels
are not used. No array in the DF-RHF chain has a Cartesian index.

**B is built over pairs and contracted over aux functions.** A pair column of
B depends on every P, so a batch of P cannot be fitted alone; a batch of pairs
can. Each GPU takes its own pair batches, forms (μν|P) in its ring, fits it
against its own L, and downloads it into B's slab. No GPU needs anything from
another. B is pair-fastest, so one pair batch's columns are a strided run in
every row of P, and two GPUs' batches interleave in memory: no slice cut parts
them. Each batch is scattered through a raw pointer, the second `unsafe` of
the memory model, with its invariant beside it: the pair batches partition
the pair columns, so a batch owns its columns and no two write one. The
scatter is a transpose, done in 64 × 64 tiles: element by element it strides B
by a whole row per write. A J/K pass then runs over aux batches: each GPU
contracts its own, and the fragments are summed onto gpu0 through the host.

**A J/K pass uploads its whole share, through a ring of two.** B has one
copy, on the host; no GPU keeps rows of it between passes. A GPU's aux range
is fixed: the aux functions are split across the GPUs once, in budget mode.
Each aux batch goes through one of two ring buffers, each with a pinned end:

1. The host copies the batch's rows of B into the buffer's pinned end, once
   the upload that last read those bytes has run.
2. The upload goes on the manager's own upload stream, after the event the
   GPU's stream recorded behind the contraction that last read the buffer.
3. The contraction goes on the GPU's stream, after the event the upload
   stream recorded behind the upload.

So a batch's host copy and its upload both run while the batch before is
contracted, and the GPU never waits for the host. The pass joins its upload
stream before it returns: the wrapper does not know the manager's stream.
With the ring a pass takes the contraction's time, so nothing is gained by
keeping rows of B on a GPU ([what it showed](#what-it-showed)).

**Pinned staging does not grow with the limit.** A fitted pair batch comes
down through a staging buffer of at most 64 MiB, and an aux batch of B's rows
goes up through one of at most 128 MiB. Neither batch is ever wider than its
staging buffer holds, whatever `mem_per_gpu` is: a batch is one copy. Page-
locking costs about 0.9 s per GiB here, and each ring page-locks a buffer for
each of its depth: the build's time is flat in the width well below 64 MiB
([batched only](#batched-only)).

**What `mem_per_gpu` does, from config alone:**

| Set to | Gives |
|---|---|
| Below the ring at a width of one | A refusal naming `[eri]`, its limit and the least it needs |
| About the ring | Narrow pair batches and narrow aux batches: the tests' many-batch case on water |
| More | Wider batches, up to what a staging buffer holds; past that, nothing |
| Default | 2 GiB: a fixed constant, not a fraction of the cap, so `--budget-only` is exact. A placeholder; *Open*: the measured value |

The buffer is held for the whole calculation, so what it takes is not
available to later steps. The budget table shows it as its own row. With the
batches bounded by their staging it stops growing with the limit: 881 MiB a
GPU at gly10, whatever the limit above that.

**Derivatives of B are scratch.** Gradients and Hessians build the derivative
of B a batch at a time and contract it at once, inside the step that needs it.
They read the metric and L, which is why the metric is a real `Output` and L is
the manager's.

## The memory budget

`MemBudget` the trait, `mem_budget` its function, budget mode the DAG run that
runs it, `Budget` the handle it allocates through.

```rust
pub trait MemBudget {
    /// Charge what this step allocates, allocate it, and put the
    /// `Output` in the bus. Budget mode calls it once per step.
    fn mem_budget(bus: &MainBus, budget: &Budget);
}

impl MemBudget for ScfStep {
    fn mem_budget(bus: &MainBus, budget: &Budget) {
        let n = bus.dims.n_ao;
        let output = ScfOutput {
            c:   budget.device(Gpu(0), n * n),   // summed: later steps read it
            eps: budget.heap(n),                 // a fixed-size host slab
            ..
        };
        // Dies with the run: F, the DIIS ring, syevd's work. One region,
        // packed by the final node, lent to run by the layout.
        budget.scratch(Gpu(0), Kind::Device, ScfLayout::new(bus).bytes());
        // The manager and every GPU; the wrapper takes them in rank order.
        budget.locks(
            [LockId::Manager(ManagerId::Eri)]
                .into_iter()
                .chain(bus.gpus().map(LockId::Gpu)),
        );
        *bus.scf.write() = Some(output);
    }
}
```

What each `Budget` call for an `Output` does, in order:

1. Add the bytes to the kind's atomic counter (per GPU for device and pinned)
   and to this step's row of the budget table.
2. Check the counter against the cap. Over it: set the flag. Do not panic.
3. Allocate, unless the flag is set.

A `budget.scratch` call only records a row (step, GPU, kind, bytes). The final
node then packs each GPU and kind's rows into one slab
([step scratch](#step-scratch)), adds each packed peak and the handle
workspace to its counter, checks again, and holds heap and pinned against the
host. If the flag is set it prints the whole table and exits. Otherwise it
allocates the scratch slabs and each GPU's handle workspace, binds the latter
to that GPU's BLAS handle once, records each step's offset, and logs the
table.

```text
FIGURE-02 (fig)
```

- **Two modes, one run each.** No counting run before an allocating one.
- **The same DAG, in parallel, with atomic counters.** The total is
  order-independent, so accept or refuse is the same every run; deferring the
  exit to the final node makes the report the same every run too, complete and
  not whichever step crossed the line first.
- **Two moments of the device check:** per step for `Output`s, once at the
  end for the packed scratch and the handle workspace. Each is its own row.
- **In parallel** lets independent steps' slabs be touched at once. One slab
  is touched on one thread, with huge pages: on bare metal that already runs
  at 14 GiB/s, and on the VM the glycine runs used, more threads made the
  touch slower ([what it showed](#what-it-showed)).

### What is budgeted, by kind

| Kind | Counter | What `Budget` hands back | Rule |
|---|---|---|---|
| Device | Sum, per GPU | A fixed-size device buffer | An `Output`'s: read by other steps, or carried over a replay |
| Pinned | Sum, per GPU | A fixed-size page-locked slab | As device |
| Heap | Sum, one for the host | A `HeapSlab<T>` over a `Box<[T]>` | Cannot grow; pages touched in `mem_budget` |
| Scratch | Packed, per GPU and kind (device, pinned) | A lease on the step's region for its `run` | Packed and allocated by the final node; regions overlap only between steps the closure orders |
| Handle workspace | A constant, per GPU | Nothing: bound to the GPU's BLAS handle once | 128 MiB, allocated by the final node |

### Host heap

- **A boxed slice, not a `Vec`.** `Box<[T]>` has no `push`, `reserve` or
  `extend`, so "no growing" is the type's. `HeapSlab` is constructible only by
  `Budget`, which separates budgeted memory from an incidental `Vec`.
- **Touch the pages in `mem_budget`.** Zeroed memory is handed out lazily on
  Linux; an untouched slab "succeeds" and fails at first touch mid-run, as a
  bare kill from the OOM killer.
- **Huge pages are asked for first** (`madvise`, which is advice: a kernel
  that gives none changes nothing). A slab is then touched 2 MiB at a time
  where it would be 4 KiB.
- **The host is checked at every call**, before the slab is allocated and
  touched, so a B larger than the host sets the flag and is never touched.
  The final node checks again with the packed pinned scratch added.
- **Not packed.** The heap stays a sum. What a step's `run` needs on the heap
  and drops is small, or it is pinned staging, which is scratch.
- **Small host scratch is a policy, not a kind.** A `Vec` or a `HashMap` that
  lives within a `run` is assumed small and is outside the budget: it stays
  within O(n_ao²), and anything larger is a `HeapSlab` on the MainBus. The
  allocation record checks it ([enforcement](#enforcement)), and the budget
  table prints execute mode's peak heap growth as an unbudgeted row.

### Caps and margins

- **Caps come from the config, per kind.** An explicit cap is used as
  written; the user owns the margin.
- **A defaulted device cap** is the GPU's free memory at start, read once
  after its handles exist, less the margin: the larger of 512 MiB and 5% of
  the GPU's total memory.
- **A defaulted host cap** is the host's available memory, read once in setup
  (cgroup-aware: the smaller of the machine's available memory and the
  cgroup's limit less its use), less the larger of 1 GiB and 5%. The reading
  is the new layer's own function; it names nothing of the arena. With B on the host heap, this is the cap a large density-fitted run
  meets first.
- **Every cap is clamped to the reading.** An explicit cap past the machine is
  clamped to the reading itself, no margin taken. The table prints the cap as
  written beside the value used. Without the clamp an impossible cap ends in an
  abort at whichever allocation fails first, or in the OOM killer.
- **The margins are judgment, not measurement.** One too small costs a failed
  launch that says so ([failure](#failure-diagnostics-determinism)), and is
  raised then.

### What the budget leaves out

For the user docs: **the memory limits cover what steps allocate through
`mem_budget`, step scratch (which holds every library call's work buffer), and
the fixed handle workspace. They exclude the device context, the handles' own
memory and loaded modules, the kernels' stack, and small host allocations such
as reading the geometry and the basis data.** A defaulted cap's margin covers
those; an explicit cap is the user's.

- **The kernels' stack.** CUDA holds a kernel's per-thread stack for every
  thread the GPU can keep resident (104,448 on the RTX 3080) from that
  kernel's first launch on. At the Rust-only kernels' largest frame, 3,856
  bytes, that is about 280 MiB on the 3080. No row, no declaration. The risk
  is one loud failure: a plan that nearly fills the GPU fails at the first
  launch of a kernel with a larger frame than any before, in a named step,
  with an allocation error. Nothing budgeted can be in the way, since
  everything was allocated before the first launch. ROCm reserves nothing
  per frame (`src/core/wwr/README.md`, ROCm 7.2.4). *Open*: raise the stack
  limit at init, or budget the stack, once a kernel with a frame past 8 KiB
  is ported (the Hessian kernels).
- **Modules and a library's first call** (2–10 MiB at a HIP solver handle's,
  up to 14 on cuSOLVER) are the margin's. No warm-up in budget mode. The
  handles themselves exist before the free-memory reading, so a defaulted cap
  already sees them.
- **No free-memory check at run time.** Every vendor call measured to take
  memory by itself is no longer called ([workspace](#library-workspace-and-work-buffers)),
  and the reading moves for reasons that are not the tree's (the stack,
  modules, other processes).

## Lengths: counted in setup, built by the step

A kernel cannot walk a `BasisSet`. It reads flat arrays that say, for every
unit of work, where its inputs are and where its result goes; the host builds
them, stages them in pinned memory and uploads them. **Records** are data, one
entry per primitive. **Work lists** are indices, one entry per unit of kernel
work. Nothing in an entry is a pointer, so every array is sized by counting.

The density-fitting arrays on gly10 in cc-pVDZ / cc-pVDZ-JKFIT, unbatched (from
`v0.0.2/src/steps/eri.rs`):

| Array | Kind | One entry per | Length, from `Dims` | gly10 |
|---|---|---|---|---|
| `aux_records` | records | aux primitive: x, y, z, α, c(π/α)<sup>3/2</sup> | 5 × `n_aux_prim` f64 | 51 kB |
| `metric_pairs` | work list | aux shell pair P ≤ Q, by class (l<sub>P</sub>, l<sub>Q</sub>) | 6 × `n_aux_shell`(`n_aux_shell`+1)/2 i32 | 19.4 MB |
| `cells` | work list | orbital shell pair × aux shell, by (orbital class, l<sub>P</sub>) | 6 × `n_shell_pair` × `n_aux_shell` i32 | 1.79 GB |
| `gather_at`, `gather_weight` | work list | pair column: its μ and ν, and 1 or 2 | 2 × `n_col` i32, `n_col` f64 | 4.3 MB |

Batched, `cells` is a pair batch's share, in the ERI buffer's ring.

```text
FIGURE-03 (fig)
```

- **Setup counts every length that scales; it builds none of those arrays.**
  It has read the geometry and both bases, so it adds `n_aux_prim`,
  `n_aux_shell`, `n_shell_pair` and the per-class counts to `Dims` by a
  counting walk over the shells. The walk is the new layer's own:
  `fb_integrals_mmd::record_tables` builds every shell pair's bra, ket and
  record-prefix arrays to count from, which this rule forbids. Any array whose length scales with a pair
  count, or a product of shell counts, belongs to a step and is budgeted.
- **`mem_budget` charges, `run` builds.** `mem_budget` declares the records
  and work lists as scratch, pinned and device, at their lengths from `Dims`.
  `run` builds them straight into its pinned region (a counting walk gives each
  class's offset, so no temporary `Vec`), uploads them on the step's stream,
  and launches.
- **A replay builds again, into the same regions.** The records hold
  coordinates, which move; the lengths and offsets do not.
- **A refused or dry budget builds nothing and still prints exact lengths**,
  since they are counts.

```text
FIGURE-04 (fig)
```

## Step scratch

Memory a step needs during its `run` and nothing reads afterwards: a library
call's work buffer, a solver's DIIS ring or subspace, build intermediates, the
records and work lists. It is allocated in budget mode like everything else,
since allocation in `run` is ruled out and a replay must allocate nothing.
What lowers the peak is sharing scratch between steps, not releasing it: under
the sum, a release frees bytes no admitted step can use.

- **Declared per step, packed by the closure.** A step declares
  `budget.scratch(gpu, kind, bytes)` for device or pinned memory. The final
  node takes the transitive closure of the DAG (a bitset of successors per
  node) and packs each GPU and kind's rows into one slab: two steps' regions
  may overlap only if one reaches the other.
- **Largest first.** First fit, by size then step id, each region at the
  lowest offset clear of every unordered region already placed. The offsets,
  and the table, are the same every run.
- **The table prints the lower bound** and the steps that make it, beside the
  packed peak and the plain sum, so a config that packs badly shows itself.
  The bound is a floor, not a promise.
- **Safe whatever the packing's quality.** Any placement that keeps unordered
  regions apart is safe; a poorer one costs bytes, and at worst a refusal at
  init, never a collision in a run.
- **Why the closure is enough.** A finished node is finished device work (the
  wrapper's sync), so a path from A to B is a happens-before on the GPU as well
  as on the host, under any schedule the executor picks. Unordered steps never
  share. Two lock holders the DAG leaves unordered also keep their regions
  apart, though the lock serializes them: one safety argument, not two.
- **The step-end sync carries memory safety.** Pinned scratch needs it whole:
  the host writes pinned bytes inside `run`, and the device reads them through
  copies done by the time the wrapper's sync returns. Event chaining would
  order device work along every path but not a host write after another
  step's copy. Any change to the sync is a change to this rule.

```text
FIGURE-05 (fig)
```

```text
FIGURE-06 (fig)
```

- **Computed once.** The graph and every length are fixed, so the closure and
  the offsets are made once in budget mode. A replayed chain's closure is
  taken over one iteration of its body, laid flat into the outer DAG.
- **A condition's exclusive branches are packed as unordered**: safe, at their
  sum rather than their max. Exclusive branches of the outer graph should be
  rare, since a config choice never enters the DAG and a runtime condition
  lives inside a step.
- **Addresses hold over a replay; contents do not.** A step gets the same
  region on every run, and what is in it on entry is undefined. In a test mode
  the wrapper fills each lease with NaN before `run`.
- **One layout per step.** A step's region holds several tenants (the SCF's:
  F, the DIIS ring, syevd's work). The layout is an associated type of the
  step: `mem_budget` declares its total and the wrapper carves the lease by
  it, once.
- **A lease cannot escape.** It borrows for the length of `run`, so a step
  cannot store its scratch in its `Output`. The overlapping slices need one
  `unsafe`, in the arena, with the closure invariant beside it.
- **Inner tasks split their step's region themselves.** The closure is over
  the outer graph only.
- **Bytes in place of time.** No lock, so a scratch user never waits on
  another for memory.

## Batched only

Every step over an O(N³) tensor has a batched path and no other. The unbatched
case is the batched one with a single batch on a single GPU.

| What a step holds | Rule |
|---|---|
| Everything O(N²): the n_ao² matrices, the metric and its factor, the records | Resident whole on the GPUs it needs. It fits or the plan is refused: a dense eigensolve at n_ao cannot be batched |
| The O(N³) tensors: (μν\|P), B, a derivative of B | Never resident whole. Built or contracted a batch at a time; B itself on the host heap |
| The batch width | The widest whose bytes fit the step limit, and no wider than its staging buffer holds; none fits: refuse, naming the step, its limit and its least bytes |
| The step limit | Covers the batches and their rings, not the O(N²) residents, which the cap admits or refuses. A fixed constant by default; the config overrides it, which is also how a test forces many batches on a small molecule |
| What is charged | What is used, never the limit. Limits need not sum to the cap |

- **Every GPU step is built batched and for every GPU, or not built.** A
  feature is not added until it runs on a single node's `num_gpus` GPUs, one
  GPU being the case of one.
- **Width stops mattering early.** In `df_batched/single_gpu` the build's time
  is flat past about 20 batches, and width 1 costs 5.6× to 11.6×.
- **No "what is left".** A width is a function of the step's own limit, so
  `mem_budget` stays local and order-free.
- **A width of one is logged at warn**; the user raises the limit.
- **Sizes fixed only by an earlier step's run** (the screened pair count, the
  orbital count after dropping linear dependencies) are allocated at the bound
  `Dims` gives. The pair count's bound is the count in the tree's own pair
  layout (270,497 pair columns on gly10, where n_ao(n_ao+1)/2 gives 269,745).
  On a large sparse molecule that wastes host memory, and a replay can never
  outgrow its buffers.
- **A size a step chooses at run time is allocated at a bound set in setup.** AVAS's
  active space is bounded by its labels: at most twice the MINAO functions
  they name, capped by `avas_max_ncas`, as `src/pipeline/shape`'s
  `largest_active_space` does. The CI is sized at the largest determinant
  count at or under that bound. A run that selects past the bound fails, as a
  `Result` naming the count it wanted; it never truncates, which would change
  the chemistry unannounced. The rerun loads the SCF's `Output` from disk and
  pays setup and budget mode, not the SCF. A bound too large is refused in
  budget mode, before anything runs.
- **The SCF's `Output` is written on every run.** It is the checkpoint every
  size fixed at run time leans on.

What can still be refused: a plan whose O(N²) data does not fit its GPUs, whose
B does not fit the host, whose step limit holds no batch, or whose `Output`s,
each fitting alone, do not fit summed down a chain. The last is the sum's
price; the remedy is a narrower batch in the step concerned, and past that
`Output`s packed by the closure (*open*, built when a chain is first refused).

## GPUs, streams and locks

### Which GPUs a step runs on

`num_gpus` comes from the config. A step runs on no GPU, on gpu0, or on all of
them; no step runs on GPU k alone or on some of them. The assignment is fixed
when the DAG is built, before budget mode, since `mem_budget` must know which
counters to charge.

- **gpu0 is the privileged GPU.** A one-GPU step runs there, and dense
  diagonalization is single-GPU.
- **An all-GPU step splits its data across every GPU** and reduces its results
  onto gpu0. A step that calls an all-GPU method is an all-GPU step (the SCF,
  through `build_jk`), though its eigensolve runs on gpu0.
- **A device buffer is on gpu0 or on every GPU**, as the `Budget` call that
  allocates it says. gpu0's cap is the one that binds.
- **With one GPU, all is gpu0.** Not a second path.
- **The current device.** Launches and library calls run on the calling
  thread's current device. The wrapper makes the step's device current for the
  body, and each GPU's thread of a fan-out does the same for its own GPU with
  a `ScopedDeviceIndex`.
- **Host steps relieve gpu0.** The one-electron matrices and the metric are
  computed on the host on rayon's threads (`s_overlap_host`, `t_kinetic_host`,
  `v_potential_en_host`, `df_metric`: on gly10, tens of milliseconds) and
  uploaded. An upload calls no library, so these steps take no lock and run
  beside the GPU work.

### Streams

A GPU step has streams of its own on each GPU it spans, one by default, more if
declared. The wrapper makes them in budget mode and keeps them in the step's
slot, beside its `Output`. A step's kernels and copies go there.

- **`run` enqueues; the wrapper syncs.** The DAG builder wraps every GPU step's
  `run` in one generic function: make the device current, take the step's
  declared locks, call `run`, synchronize every stream the step has and each
  held GPU's stream, then release the locks. A step cannot skip it.
- **Whoever fans out, joins.** A step's inner tasks, and a manager method's
  fan-out across GPUs, are finished before `run` returns. The wrapper's sync is
  the last word, not the only one.
- **A finished node is finished device work**, on every stream the step
  touched. So the DAG's edges order everything between steps, with no event
  between steps, and any worker can run any step. Steps that call no library
  overlap on a GPU for free.
- **The sync blocks**, so the waiting thread sleeps. A waiting worker is not
  stealing, so the executor's width covers the GPU steps that can wait at once,
  plus the CPU work: a handful of threads.
- **Two thread pools.** The host integrals run on rayon's threads, called from
  a step on a rustflow worker; each pool has its own thread count from the
  config. An all-GPU step's fan-out adds one scoped thread per GPU for the
  length of the fan-out, each blocked on its device most of the time.

### One handle set per GPU, under its lock

Each GPU's `GpuResources` is a `Mutex` around that GPU's stream, a BLAS handle
and a dense-solver handle made on that stream, and the 128 MiB handle
workspace. The handles keep that stream for the life of the process:
`SetStream` is never called after creation, and the workspace is bound once.

| | A step's own streams | The GPU's `GpuResources` |
|---|---|---|
| What | Streams, and their events for the step's own fan-out | The GPU's stream, the handles made on it, any further library object a step needs on that GPU, the handle workspace |
| How many | One per GPU step per GPU it spans by default | One per GPU |
| Made | By the wrapper, in budget mode | In setup, before budget mode |
| Lives | In the step's slot | On the MainBus |
| Reached through | The step's slot | The GPU's guard |
| What goes on it | The step's kernels and copies | Library calls, and the holder's kernels when it wants one order with no event |
| Who uses it | Its step | The holder of the GPU's lock, for its whole `run` |

- **A library-using step declares the GPUs it calls the library on**, in
  `mem_budget`, and its library calls go on those GPUs' streams. A simple
  library user (the orthogonalizer) puts everything there, in the order it
  enqueues, with no event.
- **The handles are reachable only through the guard.** "No handle is shared
  with work in flight" is the borrow checker's: the previous holder synced the
  GPU's stream before releasing.
- **A recipe cannot be handed the wrong stream.** Library bindings and recipes
  take the GPU's guard, the stream and the handles together. `trsm`'s
  diagonal kernels and its gemms land on one stream because the guard has
  only one.
- **A step's inner tasks share its handle set.** An inner task for a GPU takes
  that GPU's guard from the step, mutably. A step that wants concurrent
  library calls inside itself declares further handles of its own, with
  budgeted workspaces.
- **What it costs:** library-using steps are serial on a GPU, whatever the DAG
  says, because the lock is held for a whole `run`. A long one serializes every
  short one behind it. Heavy steps fill a GPU on their own anyway, and two on
  one GPU time-slice rather than overlap (a CASSCF case measured 2.3 s alone
  and 158 s beside the exchange build, in this tree's test batches).
- **A handle set is not free**: about 20 MiB on CUDA and 52 on HIP
  (`handle_ring`). One per GPU is the floor.

```text
GPU's stream:  eri [ factor L | fit 0 | fit 1 | fit 2 ] sync, release; orthogonalizer [ syevd, gemms ]
```

### Lock order

There are two kinds of lock: one per manager (here only the `EriManager`'s),
and one per GPU. A deadlock needs a cycle: one step holding the manager and
waiting for gpu1 while another holds gpu1 and waits for the manager. The rule
that makes a cycle impossible:

1. **One global rank:** every manager first, by a fixed id, then the GPUs by
   index: `enum LockId { Manager(ManagerId), Gpu(usize) }`, with `Ord`
   derived in that order.
2. **A step declares every lock it takes** in `mem_budget`, beside its
   scratch.
3. **The wrapper takes the declared set in rank order** before `run`, and
   releases it after the sync.
4. **Nothing takes a lock inside `run`.**

If every step takes its locks in increasing rank, nothing can hold a higher
lock while waiting for a lower one, so no cycle forms. Rule 4 is held by type:
`run` never sees the MainBus, so it never sees a `Mutex`. It receives the
guards already held:

```rust
// scf declared {Eri, gpu0, gpu1}; the wrapper locked them in that order
fn run(view: ScfView<'_>, held: &mut Held<'_>, lease: Lease<'_, ScfLayout>) {
    view.eri.build_jk(held, &d, &mut j, &mut k);
}
```

`build_jk` takes the GPU guards as an argument; the manager has no path to a
GPU's `Mutex`, so it cannot lock on its own. The same rank keeps collectives
issued in the same order on every rank. A step re-run by a condition loop takes
its locks each time.

### A step with streams of its own and library calls

`eri` is the case (`df_batched/multi_gpu_spherical` builds it): each GPU
builds its share of B over a ring of two pair batches, and every batch has
kernels, a library fit and a download.

- **Each ring buffer has a stream of its own**, carrying its batch's upload,
  its three-centre kernels and its download.
- **The fit goes on the GPU's stream.** It waits on the event its ring stream
  recorded after its kernels, and the download waits on the event recorded
  after the fit.
- **The host waits on a ring buffer's last event**, not a whole stream, before
  it scatters that batch into B and writes the buffer's pinned bytes again.
- **The metric's factor is on the GPU's stream before the loop**, and overlaps
  the first batch's kernels.
- **Two fits never overlap** on one GPU; the ring overlaps a batch's copies and
  kernels with the previous batch's fit.
- **The events are the step's own fan-out**, all waited on before `run`
  returns.

### Library workspace and work buffers

| | The handle workspace | A call's work buffer |
|---|---|---|
| What | Device memory a BLAS handle draws on for any call | The `work` argument of a solver call, sized by its query (`syevd`, `potrf`, …) |
| Size | 128 MiB per GPU | The query at the call's arguments, declared in `mem_budget` |
| Lives | In `GpuResources`, bound once by the final node | In step scratch |
| Protected by | The GPU's lock | The closure and the step-end sync |

- **Every BLAS handle runs on the caller's workspace, never its own.** On
  cuBLAS the size only picks the algorithm; on rocBLAS a call that needs more
  takes memory of its own, outside the budget, with no error. Setting it on
  hipBLAS also frees rocBLAS's own 32 MiB pool.
- **No vendor `trsm`, `potrs` or `getrs`, on either backend.** On rocBLAS 5.6.0
  `trsm` takes workspace growing with its right-hand sides, up to a copy of B,
  and hipSOLVER's `potrs` and `getrs` take B's size plus about 42 MiB by
  themselves (`blas_workspace_scaling`). The triangular solve is the tree's
  recipe, as in the C++ `src/recipes/trsm`: each diagonal block by a kernel,
  the rest by one gemm, needing only the handle workspace. At the DF solve's
  shape it takes 0.76–0.86 of cuBLAS's time on the RTX 3080 and 0.99–1.04 of
  rocBLAS's on the RX 9060 XT (`src/recipes/trsm/README.md`). Every other call
  measured stays inside the handle workspace or its size query, so the fixed
  128 MiB covers every vendor call that is left.
- **A call's work buffer is the step's, like any kernel's scratch.** The step
  declares it from the size query (`budget.scratch(gpu, Kind::Device,
  syevd_lwork(n))`) and passes a slice of its lease in `run`.
- **The binding checks the slice against the query, before the call.** It
  repeats the query at the same arguments and panics at the caller's line
  (`#[track_caller]`) before anything is enqueued. Not optional: cuSOLVER does
  not check `lwork` itself, and writes past a short buffer
  (`library_handle_workspace`).
- **A work buffer is never null.** A query that returns 0 still gets a minimal
  non-null slice: on HIP a null `work` puts hipSOLVER's rocBLAS handle in
  library-managed mode.
- **The queries need live handles**, so `GpuResources` exists before budget
  mode. A device-free `--budget-only` reports each call's work as zero.
- *Open*: a sentinel test per vendor binding in the nightly, the call run on a
  sentinel-filled workspace with the GPU's free memory read around it, so a
  toolkit bump that changes what a call takes fails there.

## The step DAG on rustflow

One graph is written, over steps, and built twice from the same description:
once with each node's body calling `mem_budget`, once calling `run`. Names are
rustflow's ([README](https://github.com/whatDiracIsCooking/rustflow),
[DESIGN.md](https://github.com/whatDiracIsCooking/rustflow/blob/main/DESIGN.md)),
pinned at `f184bcab`.

| In the calculation | In the Taskflow |
|---|---|
| A step | A static task (`emplace`) whose body is the wrapper |
| A step reads another's `Output` | A dependency (`precede`), made by the builder from the declared View, and nothing else |
| The end of budget mode | One final task every step precedes |
| A step's inner work (integral batches, the SCF's iterations) | Nothing: plain code inside the node's body. A loop on the host, and for a fan-out one scoped thread per GPU, joined before `run` returns |
| A replayed chain | The chain as a module task (`composed_of`) inside an outer condition loop; the scratch closure is over the loop's body |

- **rustflow runs the outer graph and nothing inside a step.** What a step
  does inside `run` gives a task graph nothing to schedule. The SCF's loop is
  serial. A fan-out is one thread per GPU, each waiting on its device. A
  library-using step holds its GPUs' locks for its whole `run`, so nothing
  inside it could overlap another step's library calls. What the outer graph
  buys is the host steps beside the GPU work, the edges from the declared
  reads, and the closure the scratch packing reads.
- **An inner graph enters when one is measured to pay**, and not before.
  rustflow has what it would need: a subflow (`emplace_subflow`) whose
  condition loop runs and joins before its node finishes
  (`v0.0.2/tests/rustflow_pin.rs`, at `f184bcab`), and a `Pipeline`, whose
  back-pressure is host-side, so a streamed loop would still need device
  events.
- **A replay's loop needs no entry node.** A multi-condition
  (`emplace_multi_condition`) names every source of the loop's body.
- **A module cannot be run by two tasks at once**, so concurrent replays would
  be copies with buffers of their own. None is in scope.
- **Budget mode runs the same edges**; the graph is one.
- **A serial mode** is a one-worker executor, `Executor::new(1)`.
- **`'static` bodies** capture an `Arc<MainBus>`, which owns everything a body
  reaches.
- **The executor is the process's**: one, its thread count fixed at init from
  the config.

## Failure, diagnostics, determinism

- **A plan over budget is a refusal, not a panic**: the final node prints the
  table and exits with an error. An allocation the counter admitted and the
  device refused aborts at the failing call, still at init.
- **A launch that fails for memory reports what it can read.** By execute mode
  everything budgeted is allocated, so the failure is about memory outside the
  budget. The error names the step and the kernel, prints the GPU's free
  memory and, on CUDA, the stack limit, lists the causes on that backend (a
  larger frame than any before, modules loading, another process), and
  suggests a lower device cap.
- **A config whose steps do not close** is refused when the DAG is built,
  naming the step and what it needs.
- **A step's failure in execute mode is a `Result`.** A failed step fails its
  successors and nothing else: the wrapper holds a failed flag per node. A panic
  surfaces from `wait` as `RunError::Panicked`.
- **The budget table is the memory diagnostic**: one row per step and kind,
  the ERI buffer and the handle workspace their own rows, the packed scratch
  beside the plain sum and the lower bound, the totals against the caps, each
  cap as written beside the value used. rustflow's `ChromeObserver` and
  `dump()` sit beside it.
- **Determinism.** Results are those of the edges. Accept or refuse, the
  scratch offsets and the table are the same every run; the order steps were
  budgeted in is not.

## Enforcement

| Rule | Check |
|---|---|
| Device, pinned and budgeted heap memory are allocated only in `mem_budget` | By visibility: the allocation calls are callable only from the crate that defines `Budget`, and `run` is never handed a `Budget`. A clippy ban on the raw `wwr-sys` allocators, with one allow there, backs it |
| A budgeted host buffer never grows | By type: `Box<[T]>` |
| Nothing large is allocated in `run` on the plain heap | clippy's `disallowed_methods`/`disallowed_macros` ban `vec!`, `Vec::with_capacity` and the like in the steps' crate, each exception an allow with its reason; and the allocation record, a `#[global_allocator]` wrapper that records any single allocation over a threshold during execute mode, checked by the wrapper after each `run` in the test tiers. *Open*: the threshold (1 MiB, say) |
| Only the owning step writes its `Output` | Visibility of the mutating methods; a manager's state only under its lock |
| A step reads only the `Output`s it declared | By type: `run` receives its View, not the bus |
| Nothing takes a lock inside `run` | By type: `run` never sees a `Mutex`, only the guards the wrapper took |
| Scratch dies with its step | By type: the lease borrows for the length of `run` |
| Two GPUs never write the same columns of B | By construction: the pair batches partition the columns, checked when the batch list is made; the scatter is the one `unsafe` outside the arena |
| A step relies on nothing its scratch held on entry | In a test mode the wrapper fills each lease with NaN |
| Two steps share scratch bytes only if the DAG orders them | By construction: the final node packs from the closure; a test packs DAGs and checks every overlapping pair against reachability |
| A library call's work buffer is at least its size query | By the binding, before the call; a `doc_check` test holds every solver binding to it |
| No library handle manages its own memory | The handle workspace is bound once and `SetStream` is never called; no work buffer is null |
| A GPU's handles are used by one step at a time, with nothing in flight | By type: they are inside the GPU's `Mutex`, and the wrapper syncs the GPU's stream before the guard drops. A `compile_fail` doctest keeps a handle past the guard |
| No vendor call takes memory by itself | By what is bound: `trsm`, `potrs` and `getrs` have no binding |

The allocation record must record and not panic (unwinding out of a global
allocator is undefined behaviour), and it sees Rust's heap only, not a C
library's `malloc`.

## Example: the DF-HF setup DAG

Everything that must exist before the first SCF iteration of a density-fitted
Hartree-Fock, and what each piece depends on. Edges point down, from what is
needed to what needs it. Each node carries its phase and what its
`mem_budget` would charge; the charges show the rules at work, with the ERI
buffer at an example `mem_per_gpu` of 2 GiB, and are not sizes the tree
commits to.

```text
FIGURE-GRAPH (the graph: pan, zoom, and click a node to trace what it needs)
```

### Nodes

```text
FIGURE-NODES (the node table: one row per node, filled by the page's script)
```

### How it maps onto this design

- **The top three levels are setup.** The reads, the electron count, the
  nuclear repulsion and the two basis sets are small host work, unbudgeted,
  and they fill `Dims`. Budget mode starts below them.
- **The one-electron matrices and the metric are host steps.** S, T and
  (P|Q) are built directly in the solid harmonics on rayon's threads; V turns
  each contracted shell pair's Cartesian block spherical through the
  harmonics' nonzeros, so no Cartesian matrix exists and no spherical
  transform is a node. They are uploaded: S and H to gpu0, the metric to
  every GPU. They take no lock and overlap the GPU work.
- **Three steps hold gpu0's lock**: the orthogonalizer (an eigensolve), `eri`
  (the factor and the fits, on every GPU) and the SCF. The orthogonalizer and
  `eri` are unordered, so one waits for the other.
- **The factor, (μν|P) and B are one step, `eri`**, whose `Output` is the
  `EriManager`. (μν|P) exists only a pair batch at a time, in the ERI buffer's
  ring; B is on the host heap.
- **Scratch is packed.** The orthogonalizer's eigensolve work and `eri`'s
  records are unordered, so their regions are disjoint; the SCF follows both
  and may reuse either.
- **The peak moved off the GPU.** B's 7.80 GB slab is on the host. A GPU holds
  its O(N²) data, the metric and its factor (0.21 GB), the ERI buffer and the
  handle workspace. Unbatched, gpu0 held B, and the 1.79 GB `cells` work list
  on each side: 9.86 GB of device memory in all.

```text
FIGURE-12 (budgetfig)
```

```text
FIGURE-13 (fig)
```

### Choices the graph leaves open

- **How (P|Q) is orthogonalized**: a Cholesky factor and a triangular solve, as
  drawn, or a symmetric J<sup>−1/2</sup> that can drop near-dependent fitting
  functions.
- **How X is built**: symmetric S<sup>−1/2</sup>, or canonical, which may give
  fewer orbitals than basis functions.
- **Which initial guess**: a superposition of atomic densities read from a
  table, as drawn; a core-Hamiltonian guess would depend on H, X and the
  electron count.
- **What the screening map bounds**: the Schwarz bound over orbital shell pairs,
  as drawn; one that also uses √(P|P) adds an edge from the auxiliary set.

Symmetry, frozen orbitals and effective core potentials are left out; none
changes the shape.

## The v0.0.4 prototype

Built: [what it showed](#what-it-showed). Item 3 of the order of work: the
DF-RHF chain as steps on `num_gpus` GPUs, batched. The first prototype to run the GPU's stream under
its lock, a step with streams of its own and library calls, the `EriManager`
with a ring and a stream of its own, and the lock order; and the first written from scratch over
the C ABI.

### Built from scratch over the C ABI

The new layer is written as if nothing in `rust/` above the C ABI existed.

| In `rust/` | In the new layer |
|---|---|
| The kernel libraries and the C ABI: what `rust/CMakeLists.txt` builds, `wwr-sys`, the extern declarations in each binding crate's `bindings.rs` | Kept, linked as they are |
| The device-free crates: `fb-basis`, `fb-molecule`, `fb-io-*`, `fb-sad`, `fb-integrals-mmd`, `fb-kernel-abi`, `fb-core-error-handling`, `fb-core-primitives` | Kept, as dependencies |
| `fb-core-wwr`'s streams, events, copies and handles | Written again: here they live inside the GPU's guard |
| `fb-integrals-sys`'s launch functions, `fb-recipes-sys`'s `trsm` | Written again over the same extern declarations, taking slices |
| The three-centre kernels: `rust/`'s are `speri3`, with a Cartesian bra | Not used: `eri` links `ssperi3`'s library as `df_batched/single_gpu_spherical/kernels` builds it, and binds it over that C ABI |
| The planner, the arena, the thread pool, every `add_*` builder, `rust/pipeline` | Not used; retired when the layer lands in `rust/` |

The host steps' integrals are not in `rust/`: they are the library crates of
`s_overlap_host`, `t_kinetic_host`, `v_potential_en_host` and `df_metric`
here, device-free, taken as path dependencies.

**The rule the prototype is held to.** Its code names the C ABI, the
device-free crates and plain data, and nothing else of `rust/`: no type that
owns or names a stream, a library handle or anything of the arena. Anything from the
wrapper or task layers appears only under `[dev-dependencies]`, as the old path
the tests hold the new one against. It takes v0.0.3's flow layer and packer
as they are, which name no device. v0.0.3's bus and budget are written again
over the new module: they name `fb_core_wwr`'s device slices and device
index, `fb_integrals_sys`'s record tables and the arena's host reading. The
rest is new: setup's counting walk, the host reading, one
module holding the stream, the event, the copies, the guard with its BLAS and
solver calls, the `trsm` and Cholesky recipes over the guard, and one launch
function per kernel family the steps use.

### The DF-RHF chain

```text
overlap ──> orthogonalizer ─┐
core_hamiltonian ───────────┤
metric ──> eri ─────────────┼─> scf
guess ──────────────────────┘
```

| Step | Runs on | Locks | Its `Output` | Its scratch |
|---|---|---|---|---|
| `overlap` | host, then gpu0 | none | S on gpu0 | The pinned staging |
| `core_hamiltonian` | host, then gpu0 | none | H = T + V on gpu0 | The pinned staging |
| `metric` | host, then every GPU | none | (P\|Q) on every GPU | The pinned staging |
| `guess` | host, then gpu0 | none | The SAD density on gpu0 | The pinned staging |
| `orthogonalizer` | gpu0 | gpu0 | X = S<sup>−1/2</sup> | syevd's work at n_ao, the eigenvectors |
| `eri` | every GPU | every GPU | The `EriManager`: L on every GPU, B on the host heap, the ERI buffer per GPU | The orbital records |
| `scf` | every GPU | the `EriManager`, every GPU | C, the orbital energies, D, the energy | On gpu0: F, the DIIS ring, syevd's work |

- **The SCF's loop is a plain loop inside `run`.** One iteration is `build_jk`
  over the carried density, F, the energy, the residual, the DIIS step and the
  Roothaan step.
- **`build_jk` runs over aux batches**, each GPU uploading and contracting
  its own through its ring of two, reduced onto gpu0 through the host. The
  host blocks only on a ring buffer's upload event, never on a contraction.
- **K is built from the occupied orbitals.** Every Roothaan step leaves the C
  of its D, so every pass is given C's occupied columns but the first of a
  fresh run, whose D is the guess and has no factor. A replay's first pass has
  the carried C.
- **Crudest forms stay**: no screening, a neutral closed-shell molecule,
  built-in bases.
- **The replay reset** is v0.0.2's reset mode; what carries over is the SCF's
  orbitals and nothing in scratch.

### What it is held against

One run each, in one process per molecule, the bars fixed by analysis. "The
old Rust path" is `rust/` as merged, a dev-dependency of the tests only.

| What | Against | Bar |
|---|---|---|
| The energy of water, benzene and ethylene, cc-pVDZ over cc-pVDZ-JKFIT | The old Rust path in the same process; PySCF's DF-RHF as v0.0.2's README records it | 1e-9 Eh |
| The same at forced batches: `mem_per_gpu` at its least, so every ring buffer is reused many times | The default run | 1e-9 Eh |
| The same with the J/K pass's ring at depth 1 | The default run | 1e-9 Eh |
| The same on two GPUs, NaN-filled | One GPU | 1e-9 Eh |
| B from the ring of two | The same build at ring depth 1; and v0.0.3's B, from `speri3` with a Cartesian bra and one batch in flight, a check across kernels | 1e-10 of the largest element |
| S, H and the metric from the host steps | The old Rust path's, on the GPU | 1e-12 of the largest element |
| A replay | The run before it | The same energy to 1e-9 Eh; the allocation count unchanged |

**Real multi-GPU.** Two GPUs suffice: they show the right device current in
each inner task, the lock order across GPUs, the reduction onto gpu0, a ring
per GPU. v0.0.4 is not done until it has run on a two-GPU machine.
Where a four-GPU machine is at hand, run `num_gpus` = 2, 3 and 4 there too,
which catches code that is right only for two.

### What it runs that no prototype has

| The design says | Held by |
|---|---|
| A library-using step's work is finished on every stream it touched before its lock is released | The SCF's scratch laid over the orthogonalizer's, twenty runs on eight workers, NaN-filled, X held against a host S<sup>−1/2</sup>. **With a negative control**: a test-only wrapper that skips the GPU stream's sync, with the orthogonalizer forced to run after `eri`, must fail |
| A step's own streams and the GPU's are ordered by events inside the step | `eri` at ring depth 2 against depth 1, to the bar above; and a negative control with the event before the fit removed, which must fail under the NaN fill |
| Locks are taken in rank order and released after the sync | The acquisitions the bus records, with waits: a step that found gpu0's lock held is recorded as having waited, and every step's acquisitions are in rank order |
| Nothing takes a lock inside `run` | A `compile_fail` doctest: a `run` that reaches for a GPU's `Mutex` |
| The handles are reachable only through the guard | A `compile_fail` doctest: a `run` that keeps a handle past the guard |
| A recipe cannot be handed a step's stream | A `compile_fail` doctest: `trsm` called with a ring stream in place of the guard |
| The new layer names nothing of `rust/`'s wrapper and task layers | A test over the crate's `src/`: no path into `fb_core_wwr`, `fb_core_arena`, `fb_core_planner` or `fb_core_thread_pool`; those four only under `[dev-dependencies]` |
| Unordered lock holders keep disjoint scratch | The budget table's offsets for the orthogonalizer and `eri` on gpu0 |
| The ERI buffer is refused below the ring | A `mem_per_gpu` one byte short, refused naming `[eri]` |
| A solver call's work buffer is step scratch, checked against the size query | The SCF and the orthogonalizer run on it; an undeclared `syevd` is refused naming the step |
| A kernel launch's slices are checked against its formula | A launch handed a short slice is refused before anything is enqueued, naming the kernel |
| The default cap is read after the handles exist | The table's cap against a reading taken by hand after setup |
| The executor does not starve with workers waiting on locks | The DF-RHF chain on two workers as well as eight |

Two measurements beside them, with no bar: each molecule's budget total
against the old path's planner block for the same run, and the SCF iteration
with a sync where it would end a step, were it three steps
(`[scf] stage_sync`). One-off times are printed, not benchmarked; the question
that matters is at gly10's size, which [what it showed](#what-it-showed) has
on 1, 2 and 4 GPUs.

### What it showed

RTX A6000 (CUDA), 2026-11-26, one run each, cc-pVDZ over cc-pVDZ-JKFIT.
`|ΔE|` in hartree against the bar of 1e-9. The recorded energies are
`../v0.0.2/README.md`'s, each within 1.2e-12 of PySCF's DF-RHF and of the old
Rust path.

| What | Result |
|---|---|
| Water, one GPU | −76.026751140543 in 11 iterations, 4.8e-13 from the recorded energy |
| Benzene, one GPU | −230.721729270921 in 11 iterations, 2.6e-13 |
| Ethylene (#865's geometry), one GPU | −77.867545824336 in 54 iterations, 4.0e-13 |
| Water at forced batches: `mem_per_gpu` at its least, 94.59 KiB, NaN-filled | 18 pair batches, aux batches of 5 of the 116 rows, so each buffer of the pass's ring is reused 12 times a pass; 1.4e-14 from the default run |
| The same with both rings at depth 1 | 5.7e-14 from the default run |
| Water at the default 2 GiB | One aux batch: the whole share |
| B from the ring of two against the ring of one, water, forced batches | The same bits |
| A replay of water | 2.8e-14 from the first run, in 2 iterations from the carried D; the allocation count unchanged |
| `mem_per_gpu` one byte under the least | Refused, naming `[eri] mem_per_gpu`, its value and the 96,864 B it needs |
| Lock order, water, eight workers | Every step's locks taken in rank order and released together after the sync; gpu0's three holders never overlap; the orthogonalizer's and `eri`'s scratch disjoint on gpu0 |
| A `syevd` with a work buffer under its size query | Panics at the caller's line before the solver is called |
| Benzene on two logical GPUs, and on 2, 3 and 4 real ones, NaN-filled, each share one aux batch and forced batches, on 8 workers and on 2 | Each within 1.7e-13 of one GPU |

One-off times on water, release build: setup 0.79 s (the GPU's handles), budget
mode 8 ms, execute mode 262 ms, of which `eri` 102 ms and the SCF's 11
iterations 39 ms.

**The glycine chains on 1, 2 and 4 GPUs.** 4× RTX A6000, release build, one
run each, 19 iterations every time, at the default `mem_per_gpu` of 2 GiB.
PySCF is its DF-RHF over the same two sets. The ERI buffer is what the plan
charges a GPU for it; the aux batches are 231 rows at gly5 and 62 at gly10,
what a 128 MiB staging buffer holds.

| Molecule | ERI buffer | GPUs | E, hartree | \|E − PySCF\| | Budget mode | `eri` | `scf` |
|---|---|---|---|---|---|---|---|
| gly5: 379 AOs, 1861 aux | 792 MiB | 1 | −1110.135779508227 | 3.9e-11 | 1.0 s | 1.1 s | 5.1 s |
| | | 2 | −1110.135779508228 | 3.8e-11 | 1.3 s | 0.7 s | 3.0 s |
| | | 4 | −1110.135779508227 | 3.9e-11 | 1.8 s | 0.5 s | 1.9 s |
| gly10: 734 AOs, 3606 aux | 881 MiB | 1 | −2144.284111689077 | 5.5e-11 | 5.7 s | 10.2 s | 58.3 s |
| | | 2 | −2144.284111689077 | 5.5e-11 | 6.0 s | 5.3 s | 29.9 s |
| | | 4 | −2144.284111689077 | 5.5e-11 | 6.5 s | 2.8 s | 15.6 s |

- **The energy does not depend on the GPU count**: the same to 1e-12 across
  each molecule's three runs, 5.5e-11 from PySCF at gly10, against the bar
  of 1e-9.
- **K from the occupied orbitals is what the SCF's time was.** gly10 has 155
  occupied orbitals of 734. From D, one GPU took 233.6 s for the 19 passes;
  from C it takes 58.3 s, of which the guess's pass, still from D, is about
  12 s.
- **The SCF scales with the GPUs**: 1.9× on two and 3.7× on four at gly10.
  At gly5, 1.7× and 2.6×.
- **The pass's ring hides the copies.** A pass uploads its GPU's whole share
  of B: at gly10, 7.27 GiB, about 0.45 s of host copy and 0.27 s of upload
  beside 2.5 s of contraction (timed apart, with a sync after each). Both
  run under the batch before's contraction, so the pass takes the
  contraction's time and no GPU needs to keep rows of B. The copies grow as
  n_aux × n_ao² and the contraction as that times n_occ, so they hide
  better as the molecule grows.
- **Where it does not hide: one batch.** A share that fits one aux batch
  has no batch before it to hide under. Benzene's iteration is 13.4 ms, of
  which about 2.7 ms is the copy of its 28 MiB of B up.
- **The build of B was the host's scatter, and now scales**: 1.9× on two and
  3.7× on four at gly10. Element by element the scatter was 14.2 s of the
  17.3 s build on one GPU, with the GPU idle behind it; in tiles it is 2.7 s,
  and the build waits on the device, whose time is the fit.
- **Budget mode is B's pages.** At gly10, 5.7 s on one GPU and 6.5 s on
  four. Touching B's 7.27 GiB takes 5.3 s with huge pages (7.6 s without),
  about 1.4 GiB/s. The rest is page-locking each GPU's 272 MiB of staging,
  about 0.3 s a GPU.
- **That touch is the VM's, not the rule's.** The same touch of a fresh
  7.27 GiB slab on bare metal (a Ryzen 9 5900X, kernel 6.8,
  `examples/touch_probe.rs`) takes 0.52 s with huge pages, 14 GiB/s, and
  2.67 s without. More threads help there: 0.39 s on sixteen at 4 KiB,
  where on the VM sixteen took 12.8 s against 7.6 s on one. A 10× gap that
  threads widen points at the hypervisor backing guest memory on its first
  touch. On bare metal, budget mode at gly10 would be about 1 s. Whether a
  second slab in the same process on the VM touches at bare-metal speed is
  [open](#open-questions).
- **Staging is bounded: 64 MiB a buffer of the build's ring, 128 MiB a
  buffer of the pass's.** A GPU page-locks two of each, and the larger pair
  is what its pinned buffer holds. The build's time does not move with its
  bound.
- **Page-locking does not overlap.** With each GPU's staging page-locked on
  a thread of its own while another touched B, the locks finished one after
  another and the touch slowed beside them: 6.8 s on four GPUs against the
  6.5 s of taking them in turn. They are taken in turn.
- **An aux batch is one upload.** Its width is capped at what the staging
  buffer holds, so no batch is split across buffers, and the ERI buffer
  stops growing with the limit: at gly10 the batch is 62 rows and the
  buffer 881 MiB, at 2 GiB and at any limit above.

**The budget against the planner's block.** One GPU, the same runs through
the old Rust path (`rust/`'s `free-breakfast`, `schwarz_threshold = 0`, its
blocks as `admit_plan` logs them) and through v0.0.4, at the default step
limit of 2 GiB and at the least `mem_per_gpu` each molecule accepts (budget
mode alone, the solver work buffers added from the default run).

| Molecule | Old path: device block | pinned block | heap block | v0.0.4 at 2 GiB: device | pinned | heap | v0.0.4 at the least limit: device | pinned |
|---|---|---|---|---|---|---|---|---|
| Water | 33.35 MiB | 322.00 KiB | 292.02 KiB | 130.73 MiB | 721.06 KiB | 301.00 KiB | 129.21 MiB | 213.34 KiB |
| Ethylene | 37.44 MiB | 1.19 MiB | 2.16 MiB | 143.09 MiB | 4.85 MiB | 2.19 MiB | 130.74 MiB | 663.62 KiB |
| Benzene | 62.26 MiB | 6.90 MiB | 28.39 MiB | 307.19 MiB | 59.86 MiB | 28.59 MiB | 140.05 MiB | 3.36 MiB |
| gly5 | 906.52 MiB | 263.87 MiB | 1.00 GiB | 1009.41 MiB | 293.24 MiB | 1.01 GiB | 251.22 MiB | 37.37 MiB |
| gly10 | 3.01 GiB | 989.67 MiB | 7.27 GiB | 1.31 GiB | 383.90 MiB | 7.28 GiB | 585.07 MiB | 140.30 MiB |

- **The heap is the same**: B, packed, on the host in both.
- **The device total is the batch widths, not the sum.** v0.0.4's is the BLAS
  workspace (128 MiB against the old path's 32 MiB) and the ERI buffer, whose
  batches the staging bounds: 792 MiB at gly5, 881 MiB at gly10. The old
  block is its planner's adaptive widths.
- **What the sum can cost is what lies outside those two**: the other
  `Output`s and the packed scratch, 89 MiB at gly5 and 329 MiB at gly10. A
  fold over lifetimes could recover part of that and no more.
- **At the least limit the sum is under the fold's block** on the glycine
  chains, 3.6× and 5.3×. Less each path's workspace it is on the small ones
  too: 1.21 MiB against 1.35 MiB on water, 12.05 MiB against 30.26 MiB on
  benzene.

**The step-end sync on the SCF iteration.** The loop is one step, so the
wrapper syncs once after it. `[scf] stage_sync` adds a sync of gpu0's stream
where the iteration would end a step were it three: after the J/K build, after
F with its energy and residual, after the DIIS step. Time per iteration, each
run listed.

| Run | Iterations | As built | With the three syncs |
|---|---|---|---|
| Water | 11 | 3.47, 3.53, 3.51 ms | 3.61, 3.60, 3.58 ms |
| Ethylene | 54 | 2.98, 2.86, 2.87 ms | 3.10, 2.94, 2.98 ms |
| Benzene | 11 | 13.74, 13.44, 13.41 ms | 13.68, 13.55, 13.53 ms |
| gly5 | 19 | 268.4, 269.6, 270.0 ms | 268.3, 270.1, 270.2 ms |
| gly10 | 19 | 3.066, 3.075 s | 3.070, 3.076 s |
| gly10, four GPUs | 19 | 825.3, 822.2 ms | 824.2, 824.8 ms |

- **The sync costs nothing that shows above benzene's size**, and at water
  and ethylene about 0.1 ms of 3 ms: some 30 µs for each of the three, what
  a sync of an idle stream costs (the wrapper's own, after each step of the
  chain, took 40 to 110 µs with the lock release).
- **The iteration already waits for the GPU**, whatever the wrapper does:
  the pass's ring keeps the host at most two aux batches ahead of the
  device, the decision reads E and the residual on the host, and the
  eigensolver's call returns only when the stream has drained. At gly10 the
  20 stream syncs of the loop as built blocked for 0.13 s of its 58.3 s.
  With the three syncs, 1.44 s is spent in them, 0.07 s an iteration, and
  the loop takes the same time: the wait moved, from `syevd`'s call to the
  sync after the J/K build.
- **So the host is never far ahead of the device in this loop**, and a sync at
  a step's end takes nothing from it. A loop whose host work could overlap the
  device's is where it would cost; none of the DF-RHF chain's is.

### Where the code departs from this page

- **The manager's lock is a `Mutex<()>` on the MainBus**, taken by the wrapper
  in rank order; the state inside the `EriManager` sits behind a `Mutex` of its
  own that is only ever `try_lock`ed under it. `build_jk` takes the `Held`
  guards and has no path to a GPU's `Mutex`.
- **The build's ring streams are `eri`'s own** (two per GPU, declared by the
  step), not the manager's. A J/K pass's upload stream is the manager's, one
  per GPU, made on `eri`'s first run beside the events; the wrapper does not
  know it, so the pass joins it.
- **`[eri] ring` sets both rings' depth**, 2 or 1: the build's and a J/K
  pass's.
- **The ERI buffer is charged at what the two layouts use**, the build's ring or
  the J/K pass's, whichever is larger; `mem_per_gpu` only bounds it.
- **Two changes to v0.0.3's flow layer**: `run` receives the guards, and a
  step declares how many streams of its own it has per GPU.
- **`--budget-only` reports each solver call's work buffer as zero**, as the
  page says; every other length is exact.
- **`Stream::sync` counts and times itself**, and the SCF's `Output` carries
  its loop's wall time, sync count and time blocked: the measurement above,
  with `[scf] stage_sync`, off by default.

### What is not built

Rows of the two tables above that have no test here:

- The energies against the old Rust path in the same process: they are held
  against the recorded ones instead. S, H and the metric against the old
  path's. B against v0.0.3's.
- The two negative controls (the skipped sync, the removed event).
- The three `compile_fail` doctests.
- The default cap against a reading taken by hand.
- An undeclared `syevd` refused naming the step: the binding panics naming
  the call and its size query.

### Build and run

`rust/`'s CMake project and the `ssperi3` kernel library first (`build.rs` has
the two directories and their environment overrides), then, from this
directory:

```sh
cargo test                                                        # the device-free tests
cargo test --test chain -- --include-ignored --test-threads=1     # with the device cases
cargo run --release --example run -- examples/h2o.toml            # setup, budget mode, execute mode
cargo run --release --example run -- --budget-only examples/h2o.toml
cargo run --release --example run -- examples/h2o.toml --devices 0,1 --mem-per-gpu-mib 1
cargo run --release --example run -- examples/h2o.toml --stage-sync   # the SCF iteration as three steps' syncs
cargo run --release --example touch_probe -- 7.27 16              # first touch of a B-sized slab: no device, no config
```

`touch_probe` uses only `std`. Where the two CMake builds are missing,
`rustc --edition 2024 -O examples/touch_probe.rs` builds it alone.

### Not in it

- Any change under `rust/`.
- HIP. CUDA only.
- A step with further handles of its own; two fits in flight on one GPU.
- More than one density in `build_jk`.
- The enforcement lints: the clippy bans, the allocation record, the
  unbudgeted-heap row.
- Screening, measuring the default step limit (2 GiB stands in), packing
  `Output`s, several nodes.

## What the model covers

**`rust/`.** `rust/pipeline` is the DF-RHF chain (config, eri, guess, scf),
over integrals with Schwarz screening, Pipek-Mezey localization and the
recipes. v0.0.4 runs that chain batched on one to four GPUs, with B on the
host, to the same energies. Screening, localization and HIP are ports; none
asks the model anything new.

**`src/`.** Each of the model's constraints, against what the C++ tree needs:

| The model holds | What `src/` needs |
|---|---|
| Every length is counted before anything runs | Already so: `src/pipeline/shape` and `footprint` size every step from its dimensions at plan time, AVAS included. A size known only at run time is allocated at its bound ([batched only](#batched-only)) |
| One config table, one slot, one instance per step | The C++ step contract (`kTable`, `kKeys`). Two states' gradients for an MECI, or a QD pair's, live inside one step |
| Nothing is allocated in `run` | Steps sized from footprints already allocate up front |
| A library-using step holds its GPUs for its whole run | Two heavy steps on one GPU time-slice anyway |
| Every GPU step is batched and spans every GPU | A cost of each port, not a limit of the model |

**Repeated evaluations are replays.** Finite-difference gradients and
Hessians, geometry optimization, IRC, and molecular dynamics over a fixed
atom list, basis and active space (Born-Oppenheimer, surface hopping over the
NAC steps) run execute mode again at the same `Dims`, in the same buffers,
each from the orbitals of the evaluation before. Displacements run one after
another: every heavy step already spans every GPU, so a serial replay leaves
none idle. Concurrent replays would need copies of the chain's buffers and pay
only on molecules too small to fill the GPUs; none is in scope.

**A calculation whose `Dims` change mid-run is a new DAG per segment**: a
QM/MM run whose QM region moves, an active space that changes along a
trajectory. Each segment pays setup and budget mode, seconds against its run.

**The sum is bounded by batching.** Every O(N³) tensor is batched under a
step limit, so what stays alive as an `Output` is O(N²) matrices (32 MB each
at n_ao = 2000) and active-space tensors (a 3-RDM at 16 orbitals is 134 MB).
A Hessian's per-perturbation response, 3 n_atom × n_ao × n_occ, is its step's
scratch, batched over perturbations, and does not stack. A chain the sum
refuses has a remedy within the model: `Output`s packed by the same closure.

## The order of work

1. **The core, device-free**: the MainBus, the declared View and the builder,
   `MemBudget`, `Budget` with its counters, flag, table and final node, the heap
   slab, the closure and the packing, the two-mode DAG on rustflow. Built in
   v0.0.3.
2. **One step, batched, on `num_gpus` GPUs**: built in v0.0.3 (`eri` and the
   J/K build, on one GPU and two logical GPUs).
3. **The DF-RHF chain as steps**: built in v0.0.4.
4. **Measure**: the budget's total against the planner's block for the same
   runs, and what the step-end sync costs on the SCF iteration. Measured in
   v0.0.4: the total is set by the batch widths, with at most 329 MiB at
   gly10 for a fold to recover; the sync costs about 30 µs each.
5. **The layer in `rust/`**, beside the old crates: the DF-RHF chain first,
   with the enforcement checks and the tests v0.0.4 left unbuilt (the negative
   controls, the `compile_fail` doctests) as its acceptance. No further
   prototype comes before it.
6. **HIP, with the gate's card.** The vendor behaviour the design's rules rest
   on is measured on both backends (`../library_handle_workspace`,
   `blas_workspace_scaling`, `handle_ring`); what is left is plumbing, which
   the cross-backend check and the HIP tier see.

Batching and every GPU are not on this list: each step has them from its
first form.

Development on the old task layer has stopped. The old crates are deleted once
the gates pass on the new layer alone. `rust/docs/memory.md` and
`programming_model.md` change with the implementation, not before.

## Open questions

**To discuss**

- **Several nodes.** The commitment is to one node's GPUs. Keeping several
  nodes open cheaply is one rule: work on different GPUs communicates only
  through the collectives layer. B on one host's heap is the first thing that
  rule would have to answer.
- **Packing `Output`s too**, by the same closure from producer to last reader:
  deferred until a chain is refused for its sum
  ([what the model covers](#what-the-model-covers)).
- **What is cleared between replays, and by whom**: a reset mode per step (as
  built), an explicit `on_replay`, or a node at the loop's entry or end.
- **rustflow and `doc_check`'s thread-pool rule.** rustflow depends on
  `crossbeam-deque`, which `rust/docs/programming_model.md` ("CPU workers")
  forbids any member to reach. The first member of `rust/` on rustflow changes
  that rule with it; a prototype that is a Cargo workspace of its own needs no
  change.

**To measure, verify or pick a value for**

- **The default step limit** for `eri`: the value in place of the 2 GiB
  placeholder; the threshold of the
  allocation record.
- **The blocking sync**: which scheduling flag makes a waiting worker sleep, on
  each backend.
- **The executor's width** against the GPU steps that can wait at once, on a
  sync or a lock.
- **Packing quality** on real configs: the table's lower-bound row answers it
  on each.
- **The rest of the workspace sweep**: upper solves, rectangular `syrk` and
  `gemv`, the batched solver calls, hipBLASLt. What the sentinel tests would
  run.
- **Scratch on HIP while a kernel runs**, on ROCm 10, sampled during a large
  launch: the HIP share of the device margin.
- **An inner graph on rustflow**: none is built. The case to measure is a
  step whose inner work has independent host-heavy parts, which no step of the
  DF-RHF chain has.
- **The `cells` work list as a kernel's own**: a dense cross product a kernel
  could derive from two short lists (58,653 and 1,272 entries on gly10), a
  FelizJueves change. It lowers the ERI buffer a batch needs.
- **Rebuilding only the records on a replay**, if the full rebuild costs.
- **Collectives**: whether NCCL and RCCL mind a rank moving between steps'
  streams.
- **Idle workers**: what rustflow's parking costs while a run waits on the
  device (`src/executor/notifier.rs`; verify).
- **First touch on a VM**: whether the 1.4 GiB/s of B's touch is paid once
  per process or by every slab. `examples/touch_probe.rs` touches the same
  size twice in one process: if the second is fast, the host was backing
  guest memory, and a replay or a long-lived process pays it once. If both
  are slow, the cost is the guest's own zeroing. Either way, it is not a
  reason to stop touching the pages up front.
- **Pinned pages and NUMA**: a pinned slab is touched by whichever worker runs
  its step's `mem_budget`.
- **cuBLAS and rocBLAS handle thread-safety**: the lock answers it either way;
  verify what each documents.

## This page as HTML

`v0.0.4.html` beside this file is this README rendered by
`rust/docs/md2html.py`, with the figures of `README.figs` spliced in: each text
block above that names a figure, and the two text diagrams, is replaced by one.
`README.figs` also holds the figures' style sheet and the script that draws the
example's graph, its node table and its budget chart. The `.md` and the
`.figs` are the source; after editing either, from this directory:

```sh
uv run --no-project --with markdown-it-py python ../../../rust/docs/md2html.py README.md v0.0.4.html README.figs
```

The script is spliced in after this paragraph, at the page's end.
