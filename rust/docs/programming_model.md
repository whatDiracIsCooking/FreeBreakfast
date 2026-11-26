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

## Steps on rustflow

Two layers are in the tree, as in [`memory.md`](memory.md). **The v0.0.4
layer** (`fb_core_flow`, over `fb_core_budget` and `fb_core_device`) is
this section, the design the v0.0.4 prototype was built to try; its
streams, locks and handles are [`gpu_layer.md`](gpu_layer.md)'s, "The
v0.0.4 runtime". **The old layer**, tasks,
edges and the orchestrator in `fb_core_planner` and `fb_core_thread_pool`,
is every section from "Three owners" on, until its crates are deleted;
"CPU workers" and "Capabilities are enforced by lints" say what holds for
both.

**A step is one config table, one slot, `mem_budget` and `run`.** A
`Step` is a unit struct named by value. Its `Output` is what outlives its
`run`, behind a lock used plainly: the owner writes, a reader reads. Its
`mem_budget` allocates the `Output` through a `Budget`, declares its
scratch and the locks its `run` takes, and reads only what setup filled
([`memory.md`](memory.md), "The budget"). Its `run` is handed the
`Output`, a `View`, the guards the wrapper took (`Held`), its streams
(`Lane`) and its scratch (`Lease`), and no `Budget` and no bus.

**The declaration is the edge.** A step names the steps it reads as a
tuple, `Step::Reads`. The builder makes the DAG's edges from those tuples
and nothing else, and `View::read` compiles only for a step in the
reader's tuple, so a read with no edge behind it does not build. A step
the config does not ask for is not in the DAG, and `Dag::build` refuses a
step that reads one, naming both, before anything is budgeted; so is a
step added twice, and a cycle.

**One graph, two modes.** `Plan::budget` runs every step's `mem_budget`
over the DAG's edges on rustflow's `Executor`, then one final node,
`Ledger::finish` ([`memory.md`](memory.md), "The final node").
`Plan::execute` runs every `run` over the same edges, allocates nothing
budgeted, and may be called again: a replay writes into the same buffers
and the same scratch regions. A step's inner work is plain code inside its
`run`: rustflow schedules the outer graph and nothing inside a step.

**The wrapper.** Every `run` is entered and left by one function a step
cannot skip:

1. Make the step's device current: gpu0's for a GPU step, and each thread
   of a fan-out makes its own GPU's current.
2. Take the locks its `mem_budget` declared, in rank order (`LockId`'s
   derived order: every manager, then the GPUs by index), and bind a
   GPU's handle workspace the first time its lock is taken.
3. In the ledger's NaN mode, fill the lease: its host regions, and its
   device regions on the step's first stream on each GPU, synced.
4. Call `run` with the held guards.
5. Synchronize every stream the step has and each held GPU's stream,
   failed or not.
6. Release the locks.

So a finished node is finished device work, and the edges order
everything between steps with no event between them. Each execute
records every step's acquisitions in its `StepTime`: the locks in the
order taken, whether one was found held so that the step waited, and when
it held them all and when it began to release them. **Nothing takes a
lock inside `run`:** a step reaches a GPU's handles only through the
guards in its `Held`, and `Locks` gives no path to a `Mutex`; its one
locking method, `query`, takes a `Budget`, which only `mem_budget` holds.
A step that declares a GPU's lock it does not run on is refused in budget
mode.

**A failed step fails its successors and nothing else.** The wrapper
keeps a failed flag per node; a node with a failed predecessor is skipped
and flagged in turn, and the plan returns the first failure in its one
topological order, the others on its context
([`diagnostics.md`](diagnostics.md), "One worker fails").

### The MainBus and its steps

`fb-steps` is the run's bus and its steps, in one crate so the bus names
every step's `Output` with no dependency cycle; the method code a step calls
stays in its module's crate.

- **The config is one table per step family.** `[molecule]` and `[basis]`
  bring the overlap, the core Hamiltonian, the guess and the orthogonalizer;
  `[eri]` the metric and the ERI; `[scf]` the SCF; `[resources]` the GPUs,
  the executor's width and the caps. A step whose table is absent is not in
  the DAG, and a step that reads one the config leaves out is refused.
- **`[molecule]` sets the charge and the multiplicity, and setup counts
  the occupations.** `charge` (default 0) and `multiplicity` (default 1)
  give `Dims`' `n_alpha` and `n_beta`, `n_alpha - n_beta = multiplicity -
  1`, beside `n_occ`, the doubly occupied, `n_beta`. A multiplicity the
  electron count cannot have is refused `InvalidConfig`, and an open shell
  `Unsupported`, naming the key, until the ROHF step: the steps run a
  closed shell, a closed-shell ion among them. A refusal of the config
  names the key, the file and the line ([`documentation.md`](documentation.md),
  "Configuration: the reference is emitted from the type"). No
  `mem_budget` reads the occupations, so an ion's budget is the neutral's.
- **Setup is the unbudgeted host phase, and it counts.**
  `MainBus::setup` reads the geometry and both bases, fills `Dims` by a
  walk over the shells that builds no array (`for_each_pair`), makes each
  GPU's `GpuResources` before it reads the device caps, and returns the bus.
  Every `mem_budget` is then a function of `Dims`, the step's options and
  the caps, so a dry budget, with no device behind it, prints exact lengths:
  only a solver's work buffer, which needs a live handle to query, is
  counted as zero. A GPU's lock is declared and checked in a dry plan too,
  against the plan's GPUs rather than the devices behind them; a
  host-backed plan refuses to run a step that takes one.
- **A host step builds on the host and uploads.** S, H = T + V, (P|Q) and
  the SAD density are built on rayon's threads ("CPU workers"), written
  into the step's pinned scratch and uploaded on the step's own stream,
  each copy's token dropped, since the staging is a tenant of the lease and
  the wrapper's step-end sync covers the copy. A copy holds its host end
  until a sync covers it, so a step that uploads to every GPU stages once
  per GPU. A host step takes no lock and runs beside the GPU work.
- **A library step takes its GPU's lock and enqueues on the guard's
  stream.** The orthogonalizer, X = S^(-1/2), runs syevd, the inverse
  square root, a `dgmm` and a `gemm` on gpu0's guarded stream, with no
  event, its eigenvectors and work buffer in its device scratch; it reads
  the solver's flag on the host after its last enqueue
  ([`gpu_layer.md`](gpu_layer.md), "The v0.0.4 runtime").
- **The SCF is one all-GPU step whose loop is plain code in `run`.** Each
  iteration is the `EriManager`'s `build_jk` over a factor of the carried
  density (its occupied orbitals, or on a fresh run's first pass the
  guess's eigen-factor above a floor), F, the energy,
  the residual, the DIIS step and the Roothaan step, on gpu0's guarded
  stream in its scratch; the turn's scalars are read on the host through
  their download's token once, after the turn's last enqueue. `fb_steps::run`
  is setup, budget mode and execute mode over the run's one executor, and
  `Run::replay` execute mode again in the same buffers: the SCF's orbitals
  carry over in its `Output`, nothing in scratch does.

### The EriManager

The density-fitted ERI is a shared product with state of its own: streams
and events, and a ring of buffers on every GPU. That state lives in one
object, `fb_steps::EriManager`, which is `eri`'s `Output`, and its users
call its methods. `build_jk(shared, held, d, j, k)` is the one that works:
J and K for each density of `d`, on gpu0, into the caller's `j` and `k`.

- **It is `eri`'s `Output`.** `eri`'s `mem_budget` allocates all of it: the
  metric's Cholesky factor L on every GPU, B's slab on the host heap, and
  on every GPU the ERI buffer and its pinned staging
  ([`memory.md`](memory.md), "The budget"). They are summed with every
  other `Output`, for the whole calculation. `eri`'s `run` factors the
  metric, builds B and fills the manager.
- **Its readers declare it, like any `Output`.** The edge comes from the
  declaration. A reader that calls `build_jk` also declares the manager's
  lock and every GPU's, and the wrapper takes them, the manager's first.
  B and L are immutable after `eri`; the per-GPU state changes only under
  the manager's lock.
- **It never takes a lock.** The guards it uses come in `held`, taken by
  the wrapper, and it has no path to a GPU's `Mutex`. Its own state sits
  behind a `Mutex` that is only ever `try_lock`ed, under the manager's
  lock: a caller that did not declare that lock panics there, and fails
  the method's contract first in a debug build.
- **The caller owns its results.** `build_jk` writes the J and K it is
  passed, reads the densities it is passed, and keeps nothing of the
  caller's. A caller's own state (its DIIS, its Fock matrix) stays in the
  caller's scratch.
- **It takes a slice of densities.** DF-RHF passes one. UHF passes two,
  and a response solve, a Davidson or a state average pass many. The
  buffers are sized for one today, a contract of `build_jk`.
- **A density is D, or a factor C of D = 2 C Cᵀ** (`Density::Dense`,
  `Density::Occupied`). From D, K costs two n_aux × n_ao³ products; from
  C, two n_aux × n_ao² × n_occ ones, and J comes from the same
  half-transformed B. A caller that has a factor passes it: the SCF does
  on every pass, its first the guess's eigen-factor. Only the matrix
  given crosses to the other GPUs: downloaded from gpu0 once, and one host
  end uploaded to each of them under one token.

**B is built over pairs and contracted over aux functions.** A pair
column of B depends on every P, so a batch of P cannot be fitted alone; a
batch of pairs can. The pair batches, each a run of one class's pairs,
partition the pair columns, and each GPU takes a contiguous range of
them: it forms (μν|P) in its ring, fits it against its own L, downloads
it and scatters it into its own columns of every row of B, which
`split_at_mut` cut from the other GPUs', so no GPU needs anything from
another and no pointer is shared. A J/K pass then runs over aux batches:
the aux functions are split across the GPUs once, in budget mode, each
GPU contracts its own share, and the fragments are summed onto gpu0
through the host.

**(μν|P) is spherical from the kernel.** FelizJueves's
`ssperi3<la, lb, lP>` writes all three indices as real solid harmonics,
each pair's block at its spherical pair column, the layout the fit reads.
No bra transform runs, and the ring holds no Cartesian batch and no
transform scratch. It is plain FP64, unscreened. No array in the DF-RHF
chain has a Cartesian index.

**The build's ring.** `[eri] ring` pair batches are in flight per GPU,
two or one. Each ring buffer has a stream of its own, which carries its
batch's cells, its three-centre kernels and its download. The fit, a
blocked triangular solve, goes on the GPU's stream: it waits on the event
the ring stream recorded after the kernels, and the download waits on the
event recorded after the fit. The host waits on a buffer's download token,
which syncs that buffer's stream alone, before it scatters the batch into
B and writes the buffer's pinned bytes again
([`gpu_layer.md`](gpu_layer.md), "The v0.0.4 runtime").

**A J/K pass uploads its whole share, through a ring of the same depth.**
B has one copy, on the host; no GPU keeps rows of it between passes. Each
aux batch goes through one of the pass's ring buffers, each with a pinned
end and an upload stream of the manager's own:

1. The host copies the batch's rows of B into the buffer's pinned end,
   once the upload that last read those bytes has run.
2. The upload, and the gather of the packed rows into square matrices, go
   on the buffer's stream, after the event the GPU's stream recorded
   behind the contraction that last read the buffer.
3. The contraction goes on the GPU's stream, after the event the buffer's
   stream recorded behind the gather.

So a batch's host copy and its upload both run while the batch before is
contracted, and the GPU never waits for the host. The pass joins its
streams before it returns: the wrapper does not know the manager's
streams. With the ring a pass takes the contraction's time, so nothing is
gained by keeping rows of B on a GPU.

**The ERI buffer.** One device buffer per GPU within `[eri] mem_per_gpu`,
`eri`'s step limit, charged at what it uses, with its pinned staging.
`eri`'s build and every later J/K pass run in it, one after the other,
since the DAG orders `eri` before every reader: during `eri`'s run it
holds the build's ring; afterwards a pass's gather tables, a copy of the
density, the J and K fragments, one aux batch's work and the pass's ring.
It is held for the whole calculation, so what it takes is not available
to later steps, and the budget table shows it in `eri`'s row.

**Pinned staging does not grow with the limit.** A fitted pair batch
comes down through a staging buffer of at most 64 MiB, and an aux batch
of B's rows goes up through one of at most 128 MiB. Neither batch is ever
wider than its staging buffer holds, whatever `mem_per_gpu` is: a batch
is one copy. Page-locking costs about 0.9 s per GiB, and each ring
page-locks a buffer for each of its depth, while the build's time is flat
in the width well below 64 MiB ([`memory.md`](memory.md), "Batched
only"). So the buffer stops growing with the limit.

**What `mem_per_gpu` does, from config alone:**

| Set to | Gives |
|---|---|
| Below the ring at a width of one | A refusal naming `[eri]`'s key, its value and the least it needs |
| About the ring | Narrow pair batches and narrow aux batches, a width of one logged at `warn`: the tests' many-batch case on water |
| More | Wider batches, up to what a staging buffer holds; past that, nothing |
| Default | 2 GiB: a fixed constant, not a fraction of the cap, so a dry budget is exact. A placeholder for a measured value |

**Derivatives of B are scratch.** A gradient or a Hessian step builds the
derivative of B a batch at a time and contracts it at once, inside the
step that needs it. It reads the metric and L, which is why the metric is
an `Output` of its own and L is the manager's. No such step is built yet.

### What the model covers

The layer runs the DF-RHF chain batched on one to four GPUs, with B on
the host. What follows is the argument that the same model carries the
rest of the C++ tree: each of its constraints, against what `src/` needs.
None of it past the chain is built.

| The model holds | What `src/` needs |
|---|---|
| Every length is counted before anything runs | Already so: `src/pipeline/shape` and `footprint` size every step from its dimensions at plan time, AVAS included. A size known only at run time is allocated at its bound ([`memory.md`](memory.md), "Batched only") |
| One config table, one slot, one instance per step | The C++ step contract (`kTable`, `kKeys`). Two states' gradients for an MECI, or a QD pair's, live inside one step |
| Nothing is allocated in `run` | Steps sized from footprints already allocate up front |
| A library-using step holds its GPUs for its whole run | Two heavy steps on one GPU time-slice anyway |
| Every GPU step is batched and spans every GPU | A cost of each port, not a limit of the model |

**Repeated evaluations are replays.** Finite-difference gradients and
Hessians, geometry optimization, IRC, and molecular dynamics over a fixed
atom list, basis and active space run execute mode again at the same
`Dims`, in the same buffers, each from the orbitals of the evaluation
before. Displacements run one after another: every heavy step already
spans every GPU, so a serial replay leaves none idle. Concurrent replays
would need copies of the chain's buffers and pay only on molecules too
small to fill the GPUs; none is in scope.

**A calculation whose `Dims` change mid-run is a new DAG per segment**: a
QM/MM run whose QM region moves, an active space that changes along a
trajectory. Each segment pays setup and budget mode, seconds against its
run.

**The sum is bounded by batching.** Every O(N³) tensor is batched under a
step limit, so what stays alive as an `Output` is O(N²) matrices (32 MB
each at n_ao = 2000) and active-space tensors (a 3-RDM at 16 orbitals is
134 MB). A Hessian's per-perturbation response, 3 n_atom × n_ao × n_occ,
is its step's scratch, batched over perturbations, and does not stack. A
chain the sum refuses has a remedy within the model: `Output`s packed by
the same closure that packs scratch.

### What the chain measured

The prototype's numbers, of the design `fb-steps` ports: 4× RTX A6000
(CUDA), 2026-11-26, a release build, one run each, cc-pVDZ over
cc-pVDZ-JKFIT. They are measurements, not bars; the steps' tests hold the
energies ([`testing.md`](testing.md), "The steps' tests"). Since then the
SCF's first pass is from the guess's factor, not from D, and on CUDA K's
update is a syrk ([`gpu_layer.md`](gpu_layer.md), "The v0.0.4 runtime"),
so the `scf` column is an upper bound on today's.

**The glycine chains on 1, 2 and 4 GPUs.** 19 iterations every time, at
the default `mem_per_gpu` of 2 GiB. PySCF is its DF-RHF over the same two sets. The ERI buffer is what the plan
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
- **An aux batch is one upload.** Its width is capped at what the staging
  buffer holds, so no batch is split across buffers, and the ERI buffer
  stops growing with the limit: at gly10 the batch is 62 rows and the
  buffer 881 MiB, at 2 GiB and at any limit above.

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

### Vocabulary

Each word below means one thing in the v0.0.4 layer's sections of these
documents.

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
| **View** | What `run` receives in place of the bus: the shared setup and the `Output`s the step declared it reads, and nothing else |
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
| **GPU** | One device of the run, `gpu0` on: an index into `[resources] devices` |
| **gpu0** | The GPU that one-GPU steps run on and all-GPU results are reduced onto |
| **all-GPU step** | A step that uses every GPU, its data split across them |
| **logical GPU** | One of several GPU indices mapped onto one physical device, a test facility (`[resources] devices = [0, 0]`) |
| **`GpuResources`** | Per GPU, a `Mutex` around that GPU's stream, the BLAS and solver handles made on it, and the handle workspace |
| **the GPU's stream** | The stream inside `GpuResources`, used only by the holder of that GPU's lock |
| **a step's streams** | Streams the wrapper makes for a step, one per GPU it spans by default; the step's kernels and copies go there |
| **guard** | A held lock: on a GPU's `GpuResources`, or on a manager |
| **handle set** | One GPU's BLAS and solver handles |
| **handle workspace** | The fixed 128 MiB per GPU the BLAS handle draws on, bound once |
| **work buffer** | A solver call's `work` argument, sized by its query, a slice of the caller's scratch |
| **recipe** | The tree's own dense routine in place of a vendor call (the blocked `trsm`, the Cholesky), from the recipes' kernel library |
| **binding** | The layer's function over one C ABI call or kernel: it takes slices and checks their lengths before enqueueing |

**Density fitting**

| Word | Means |
|---|---|
| **B** | The fitted three-centre tensor B<sub>μν</sub><sup>P</sup>, on the host heap |
| **pair column** | One orbital shell-pair function index μν of B, in spherical functions. Not used bare |
| **pair batch** | A run of pair columns, the unit the build of B works in |
| **aux batch** | A run of aux functions P, the unit a J/K pass works in |
| **J/K pass** | One sweep over all of B building J and K for a set of densities |
| **`EriManager`** | `eri`'s `Output`: B, the metric's factor, and per GPU the ERI buffer, its streams and their events. Reached by its readers under its lock |
| **ERI buffer** | One device buffer per GPU within `[eri] mem_per_gpu`, owned by the `EriManager`: the build's ring of pair batches, then a pass's gather tables, a copy of the density, the J and K fragments, one aux batch's work and the pass's ring of aux batches |
| **ring** | The buffers of the batches in flight: two by default (`[eri] ring`), one filling while the other is used. The build has one, of pair batches; a J/K pass has one, of aux batches |
| **staging buffer** | One pinned buffer that B passes through between a GPU and the host: a fitted pair batch on its way down, of at most 64 MiB, or an aux batch of B's rows on its way up, of at most 128 MiB. One per ring buffer |
| **records** | Per-primitive input data a kernel reads: centre, exponent, weighted coefficient |
| **work list** | Indices, one entry per unit of kernel work. `cells` is one of them, by name |

## Three owners

The things one has to decide about a feature belong to three owners. A task
answers only for itself; an edge answers for the data between two tasks; the
orchestrator derives the rest. Keeping them apart is what stops two tasks from
disagreeing about the edge between them, and keeps each task's spec short.

### A task declares

1. **Device.** CPU or GPU, and the two are different sorts of task. A GPU
   task holds device rows and pinned rows and runs on a stream, and reaches
   a pinned row only through a copy it enqueues; a CPU task holds host rows
   alone, pinned and heap, and runs on the CPU pool, so a CPU task holds no
   device row and no task holds all three kinds.
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

In the crates the orchestrator has two halves: the planner,
`fb_core_planner`, which lays a `Dag` out and places it as a `Plan`
(`Dag::plan`), and the executor, `fb_core_thread_pool`, which runs the
`TaskGraph` that plan's `Instance` builds. "The planner" names
`fb_core_planner` alone: the config-driven assembly in `fb_pipeline_config`,
which reads a config and calls the builders, is not the planner.

### A builder adds tasks

A computation is not one task, and the code that wires its tasks together
is a builder: a public function that takes the DAG and adds nodes to it
through `Dag::add_gpu` and `Dag::add_cpu` ("Memory: layouts and arenas"
below). Every builder has one shape, `add_x(dag, worker, inputs…) ->
outputs`:

- **Named `add_*`**, for what it adds: `add_overlap`, `add_scf`.
- **`dag: &mut Dag` first**, after `self` for a method
  (`Diis::add_push`), and **`worker`, when it takes one, second**.
  Then the rows of earlier builders it reads, and the planning input
  (a basis, a molecule, a step's settings).
- **Rows out**: the row ids the next builder reads, which is how one wires
  to the next. A builder that can refuse its input returns a `Result` and
  refuses at planning, before it adds anything, so a refused plan holds no
  half of a builder.

`add_core_hamiltonian` is the pattern whole: it refuses nuclei with no
nucleus, adds one GPU task, hands that task's row to
`add_cartesian_to_spherical`, and returns the row that adds.

It is a convention a source scan holds, not a trait. A trait could not
stop a free function taking a `&mut Dag` beside it, since `Dag::add_gpu`
is public to every task crate, and nothing yet calls builders generically;
each call names its builder. The config's table of which tables each
calculation requires and accepts (`pipeline/config`) is not that caller:
it checks keys, and each calculation's plan still names its builders.
A `Step` trait for the pipeline's steps, never for a recipe's or an
integral's builders, is revisited when the first nested chain rerun
appears (an optimization, a semi-numerical Hessian or an IRC replanning
the chain up to `[scf]` at each geometry), or when two calculations'
plans first build the same product (both calling `add_df_eri`), whichever
comes first. A private function that adds nodes for a builder, and the planner's
own `Dag` methods in `core/`, are not held to the shape.

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
`GpuTask` and `CpuTask`, one trait per sort: the task's value is its
dimensions, so `dims` is `&self`, and `layout` declares through one
`Layout` per memory kind the sort holds, reading an earlier task's output
by the row id that task's layout returned. A GPU task's `layout` is handed
a device and a pinned `Layout`, and its `run` its device rows as a `Bufs`,
its pinned rows as a `CopyOnly` that only its stream's copies can take, and
the stream. A CPU task's `layout` is handed a pinned and a heap `Layout`,
host kinds alone, and its `run` those rows as two `Bufs`; a loop's
`Decision` is handed the same two. The sort decides the worker: `Dag::add_gpu`
puts a task on its GPU worker's stream, `Dag::add_cpu` on the CPU pool.

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
| `Footprint<Device>` and `Footprint<Pinned>` | One arena per memory kind, and the kind stays in the type. |

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
Its output is an offset per buffer and the block's total. Greedy best-fit
placement, the static memory planning a tensor compiler does, run in a few
orders and kept at the smallest total ([`memory.md`](memory.md),
"Lifetimes"), is enough for the DAGs this tree has; the general problem is hard, and
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
| **Process** | The running binary: one config, one logger, the devices (`num_gpus` of them, "The run's devices"), the worker budget | One | `main` to exit |
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
| Pinned host | The GPU worker whose copies it stages | The per-GPU pinned cap, and with the heap block the sum over a node's workers within the host's available memory | One block per plan, allocated portable |
| Heap | The plan | Its own cap, and with the pinned blocks within the host's available memory | One block per plan, from the global allocator |

**Pinned memory is per GPU.** Physically it is the node's, page-locked host
RAM any card can read, and the C ABI's portable flag is set on both backends
so that fact does not depend on the unified address space. But every pinned
buffer exists to feed one GPU's copy engines, and its lifetime is bounded by
a copy on that GPU's stream, so its owner is that GPU worker, as the C++
tree's `DeviceResources` already has it. A GPU worker is then one
self-contained unit, its streams, its device block and its pinned block,
admitted together against the per-GPU caps, and NUMA locality comes free
from allocating the block on the worker's own thread. The node is a cap and
not an owner: the sum of a node's pinned blocks and the plan's heap block
must fit the host's available memory, which is the C++ `check_pinned_node`
check carried over, with the heap beside it. Host
data broadcast to several cards is placed in one consumer's pinned block and
read across, which the portable flag makes legal and the "GPUs on one node"
row of the worker table prices, or staged once per card; both are placement.
Pinning costs milliseconds and locks pages, so once per plan, reused every
run, is where plan-once-run-many pays most.

**Pinned is staging; large flat host data is a heap row.** A pinned row
exists to feed or drain a copy on a stream, and page-locked memory spent on
anything else is locked for no reason: the SCF decision's carried energy
never meets a copy engine, and host data too large to lock, B kept on the
host past the pinned carve, has to live somewhere counted. Such data is a
heap row: a slice of an `Element`, sized at planning, in the plan's one
heap block, which the assigner lays out as it does any table, so its
footprint is the table and there is no second formula. Only a CPU node
holds one; when it must reach the device, a pinned row is the window it
streams through. The heap block is one global-allocator call at
instantiation, freed at the plan's drop: an arena, not an allocator, so a
run allocates nothing on the host either.

**Structured planning input is `Vec`.** The config, a `BasisSet` and the
planner's own tables exist before any table can be sized, and fix its
dimensions, so they cannot be rows: they are `Vec`, not modelled and not
admitted. Making an arena the backend for `Vec` means the unstable allocator
API or a third-party `Vec` across every host type, for no property the model
asks for. What a run hands its caller after it, a `Vec` copied out of its
last row, is the one copy a run makes ([`memory.md`](memory.md), "The three
kinds").

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
are one run, a `Device` spanning its GPU workers' devices.

Stream capture into a replayed graph is the same split one level down, which
is a sign the split is real rather than imposed. A loop is therefore not a
special node: it is a planned sub-DAG run repeatedly, with a host decision
between runs. The decision is a node (convergence, a step accepted, the walk
reaching a minimum) whose output is read on the host.

**The executor replays the body in the run it is in.** In
`fb_core_thread_pool`, a loop is its body, the nodes added after a given
one, and a decision node `TaskGraph::add_loop` adds after the body's sinks and
any node before the body it is given, so after every body node. The decision is a CPU node, so a worker has waited
on the host for the body's sinks before it runs, and every GPU body node's
work precedes theirs on the device, so it waits on no other: the body's
work has completed, and its rows and events are free to be used again, when the
decision returns. Its `Continue` readies the body again, under the run's
one lock, in the same scope and on the same workers, so a replay spawns no
thread and allocates nothing; its `Break` readies what follows the loop,
which depends on the decision and never on a body node. The decision is
handed the iteration's index. The iteration count has a bound, fixed at
planning beside the dimensions: a decision that would start one iteration
past it fails the run with `MaxIterationsExceeded`, and a failed body node
ends the loop and skips what follows it, as any failure skips its
descendants.

**The planner closes a loop the same way.** `fb_core_planner::Dag::add_loop`
takes the position of the body's first task, its bound and a `Decision`: a
CPU node that declares host rows as a task does, reads what the body copied
to the host for it, and returns `Continue` or `Break`. The body's rows are a
table laid out once, in a region of each worker's tables the parent holds
while the loop runs, and the rows it carries between iterations are the
parent's ([`memory.md`](memory.md), "Nested plans"), so a replay is
admitted with the plan and allocates nothing.

**A counted loop has a trip count and a depth, and no decision.** Beside
the decision loop, whose next iteration waits on a host answer, a loop
over the batches of a streamed tensor knows how many iterations it has at
planning, and a host decision per batch would sync the host once a batch
and keep a batch's upload from overlapping the GEMMs of the one before.
`TaskGraph::add_counted` and `Dag::add_counted` close the body over a trip
count `N` and a depth `R`, both fixed at planning beside the dimensions:

- **Ordering.** Iteration `k` waits on iteration `k - R` alone: each body
  node with no predecessor in the body runs after each one with no
  successor there, `R` iterations back. That is the one edge across
  iterations, the ring. Each body node also runs after itself in the
  iteration before, which on a lane is the stream's order anyway and on
  the CPU pool keeps a row the body accumulates into in order. Within an
  iteration the body's edges hold as in any plan.
- **No host wait.** Every edge is satisfied as any edge is, so between
  lanes it is a wait on the device and the GPU worker's thread enqueues
  every iteration back to back; the run's record holds no sync point for
  a GPU body node. A CPU body node waits on the host for what it reads,
  off the device's path.
- **The iteration index.** A body node is handed its iteration's index,
  `k` from 0, by which it picks its batch: the executor's `Lane`, and a
  task's views (`Bufs::iteration`).
- **Memory.** The body's rows are `R` copies of its table, iteration `k`
  in copy `k mod R`, and a row it accumulates into across iterations is
  the parent's ([`memory.md`](memory.md), "Nested plans").
- **Events.** Each GPU body node has an event pair per iteration, so each
  iteration's stream scope is its own, read after the run, and no wait in
  the run finds its event recorded again by a later iteration.

For batched density fitting the body is an upload on a copy lane and the
GEMMs on a compute lane, with `R = 2`: the next batch's upload runs under
this one's GEMMs, ordered against the GEMMs two batches back by the ring.
A node after a counted loop may follow a body node directly, and runs
after its last iteration.

**A counted loop may lie in a decision loop's body.** J and K over a
streamed B run every SCF iteration, so the SCF's body holds a counted
loop over B's batches; that is the one nesting built. A counted loop holds
no loop, and a decision loop no decision loop. The counted loop is closed
before the decision, and the executor unrolls its trips inside the
decision loop's run of nodes, so the decision's `Continue` readies it
whole: every trip again, overlapped as at the top level, with no host
wait between trips. The decision runs after the body's sinks, which
every trip precedes, and before its replay reads the stream scope of each
trip of each GPU body node, its events about to be recorded again. A body
node of the counted loop is handed its trip; the decision loop's
iteration is the timing record's
([`diagnostics.md`](diagnostics.md), "The timing record"). An edge
leaving only the counted loop, to a later node of the decision loop's
body, runs from its last trip; one leaving the decision loop runs from
the decision. The counted loop's region is a row of the decision loop's
body table ([`memory.md`](memory.md), "Nested plans").

**A wait on an event binds to its recording at the time the wait is
enqueued**, on both backends: record an event, enqueue a wait for it on
another stream, record it again, and the wait is released by the first
recording. The host enqueues ahead of the device, so a design that waits
on an event while later recordings of it are being enqueued rests on
this; the counted loop does not today, since each iteration has events of
its own, and a ring of `R` events per node, which would, is open to it.

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
  It is built as a task per worker over its slice, `Partition`'s range,
  with `Dag::broadcast` in and `Dag::gather` or `Dag::reduce` out
  ("Collectives"); the slice is the task's dimension, so the task is still
  written once, for one stream.

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
| GPUs on one node | The collectives library's broadcast, on the devices | Its reduce, or a gather of disjoint pieces |
| Nodes | A network transfer | A network reduce |

The single-GPU case is the degenerate one: broadcast is a pointer, reduce is
an in-place accumulate, and the workers are streams. One DAG description
serves every row, which is the point of wanting all of them.

**Between the GPUs of one node the transport is the vendor's collectives
library**, NCCL on CUDA and RCCL on HIP, through WarpWraps' C ABI and its
safe wrapper, `fb_core_wwr::ccl`. A run is one process holding all its
devices, so its communicators are one clique, made at once over the
devices (`Comm::init_all`), a rank per device, and each rank driven by
the thread of the GPU worker on its device. A broadcast, a reduce, a
gather of equal pieces and a point-to-point send are each one call per
rank, enqueued on that rank's stream like any launch, so the row's bytes
move on the devices with no host copy and no host wait; the library picks
NVLink, PCIe peer access or a host bounce from the topology. The row
across nodes is still open ("Open questions").

**A GPU worker is one card, one thread, one block per kind, and several
streams, its lanes.** A GPU node is placed on a lane of its worker
(`fb_core_thread_pool::LaneId`), and the worker's tables and blocks are its
lanes', so the "streams on one GPU" row costs what it says: the reader
reads the producer's row in place. How an edge is ordered follows from where
its two ends landed:

| Edge | Ordering |
|---|---|
| Same lane | The stream's order; no event |
| Different lanes of one worker | An event the producer's lane records and the consumer's waits on, on the device; no host wait |
| Lane to CPU | A host wait on the producer's event, a sync point |

The worker's one thread enqueues its ready nodes in rank order across all
its lanes and never waits on the device, so lanes overlap on the card as
streams do, and the thread count stays one per worker however many lanes it
has. The C++ `src/integrals/mmd` is the pattern adapted: a stream there
belongs to a piece of data, each primitive-pair class carrying its own, its
readers enqueued on it, and one host thread feeds every stream, since an
enqueue does not block. Here too the code that builds the DAG names the
lane, as mmd names one per class; deriving lanes from the DAG's shape is
not built ("Open questions"). A worker's lanes are the ones its tasks name,
so the count grows as the builder names them and is fixed when the DAG is
planned, before the backend makes one stream per lane.

**Each lane has its own library handles.** Two lanes run at once on the
card, and a BLAS or solver handle's state and workspace are one stream's at
a time, so the `Device` backend makes a BLAS and a dense-solver handle per
lane, bound to its stream ([`gpu_layer.md`](gpu_layer.md), "Layering"),
and a declared BLAS workspace is a worker row per lane. A lane above 0
whose tasks all say they call no library (`GpuTask::uses_libraries`), as
the mmd records' classes and the copy nodes do, has neither.

**Reuse across lanes is a handoff, and every one serializes.** The assigner
packs a worker's table to the fewest bytes against the plan's ranks,
whatever lane a node is on, and every reuse it makes is a handoff the
planner adds as an edge ([`memory.md`](memory.md), "Lifetimes"). Between
two lanes that edge is an event, so a row that takes another lane's bytes
waits for that lane's last user, and two lanes whose rows share bytes do
not overlap there. The rule is the one the C++ mmd makes by hand, whose
classes' sorts share one carve, "which is why the forks chain": bytes win
over concurrency, always, and the planner adds the event edge uniformly,
through the handoffs, rather than exploiting only the paths the DAG
already has. Giving a reuse up for concurrency when the cap leaves room is
admission's search, and not built ([`memory.md`](memory.md), "Open
questions").

### The run's devices

**A run holds `num_gpus` devices, `0..num_gpus`.** The config's top-level
`num_gpus` (`fb_pipeline_config::Config::num_gpus`, default 1) is how many
devices the process holds for the run, devices `0` to `num_gpus - 1` in
that order, its device list. 0 is refused as the file is read, before any
device is touched, and a count above the devices the runtime sees is
refused, naming both numbers, before the plan is made; at 1 the runtime
is not asked, so a config without the key runs as it did before the key
existed. GPU worker `w` is on the list's device `w`.

**Work spreads where a builder names the workers.** Placement is the
builder's: it names each task's worker, so a builder that spreads names
workers `0..num_gpus`, one per device, and one that does not puts every
task on worker 0, as every calculation of the binary does but one: the RHF
over `[eri] storage = "dense"`, whose builder
(`fb_integrals_eri_conventional::add_dense`) cuts the dense ERI's rows
over the workers and gathers each build's J and K onto worker 0. The
instance makes each worker's blocks on its device and the backend its
lanes' streams, handles and fork streams there
(`Plan::instantiate(&devices)`, `Instance::backend`), for the plan's
workers alone, and the clique of communicators when the plan holds a
collective ("Collectives", below); a device past the plan's last worker
holds nothing. So a plan whose tasks are all on worker 0 is, at any
`num_gpus`, the plan of one device: its `plan.txt`, `plan.dot` and
products are unchanged by the key. Two workers on one device are
allowed, and run as two threads with blocks of their own, but not with a
collective, whose clique needs a device per rank.

**What crosses between workers.** A task reads only rows of its own
worker's tables, which are on its device. A row reaches another worker
by a collective ("Collectives"); an ordering reaches one through a CPU
node, which waits on the host for the producer's event, or, in the
executor, through an edge between lanes of two workers, a wait on the
consumer's device for an event recorded on the producer's. Both runtimes
let a stream wait on an event of another device in one process, so that
edge costs no host wait, as between two lanes of one worker. The
planner adds no such edge of its own yet; the collectives need none.

**What an edge across devices costs, the "GPUs on one node" row, is not
priced.** The planner chooses no placement, so it has no price to compare;
the builder that names the workers is choosing with the cost in mind, and
the plan's text names every collective's parts, and its tables the rows
each moves, which is what a price would read ("Open questions").

**One lane is privileged: lane 0 of the worker on the list's first
device.** Its `DeviceStream` says so (`DeviceStream::is_privileged`), set
when the `Device` backend makes the run's streams and false on every other
lane, and on a stream made outside a backend (`DeviceStream::new`), which
is no run's. It is the lane that owns the run's run-wide work, the host
I/O and the reductions every device's partials meet in: a builder that
spreads gathers and reduces onto worker 0, and every collective's part
there runs on that lane. On one device it is the lane every one-lane plan
already runs on. Nothing reads the flag yet.

### Collectives

**A collective is one step of the plan with a part on every GPU worker.**
`Dag::add_collective(name, parts)` adds part `w` as a GPU task on lane 0
of worker `w`, at consecutive positions, each part's layout holding its
own worker's rows; the parts exchange the rows on the devices, each part
calling its rank of the clique (`DeviceStream::comm`) on its lane's
stream. No edge joins the parts: the collective completes on each device
when every rank has issued its part, which the library waits for on the
devices, not on the host. The builders over it are the planner's, as the
copy nodes are:

| Builder | Moves | Result |
|---|---|---|
| `Dag::broadcast` | a root's row to every worker | each worker's row, the root's own the source; exact |
| `Dag::gather` | each worker's piece, rows `split.range(w)` of a `Partition`, onto a root | one row of `split.total()` there; exact, the pieces disjoint |
| `Dag::reduce` | each worker's partial, summed onto a root | one row there; unordered, the library's order |

**Every worker issues the collectives in plan order.** A rank whose part
of one collective is enqueued after its part of the next leaves both
waiting on the devices for good, and the executor enqueues a worker's
ready nodes in rank order but only the ready ones. So the plan orders
each part after its worker's part of the collective before, or after the
decision of the loop that held that one; the parts are on lane 0, one
stream per worker, so the device keeps the order the thread issued.
Every other wait a part can meet is on an event already recorded or a
CPU node that waits on parts already issued, so no cycle forms.

**The clique is every worker's, made with the backend.** A collective
spans every GPU worker of its plan (a contract at `Dag::plan`, which
counts the ranks, `Plan::ranks`), and the backend of a plan that holds
one makes the clique over its workers' devices, a rank per worker, before
any run (`Device::make_clique`), its ranks dropped with the backend. A
plan with none makes none, so the binary's plans pay nothing for it.

**Where a collective may lie.** In a decision loop's body whole or not at
all, the next iteration's parts issued after the decision, which waits
for this iteration's; in no counted loop's, whose trips in flight could
issue two collectives on two workers in two orders. Both are contracts
of `Dag::add_loop` and `Dag::add_counted`.

### Forks inside a task

A lane is the planner's: nodes are placed on it, the executor orders the
edges between lanes, and it costs a stream, library handles and a
workspace row. Some concurrency lives inside one node instead, where the
DAG does not look: the one-electron builds can launch a kernel per
primitive-pair class into one matrix, every class adding atomically, as
the C++ `launch_each` does on a stream per class. A GPU task reaches it
through its stream, `DeviceStream::fork_join(n, |i, fork| ...)`:

- **Fork.** The task's stream records an event, and each fork stream
  waits on it, so each starts after everything already on the task's
  stream: the executor's waits for its inputs and what the task enqueued
  before the call.
- **Items.** The closure runs once per item, on the host and in order,
  each enqueueing on its fork; item `i` goes on fork `i mod k`. The items
  are not ordered among themselves, so they must not conflict: they add
  atomically, as the class kernels do, or write disjoint bytes.
- **Join.** Each fork records an event, and the task's stream waits on
  each, before `fork_join` returns. Every wait is on the device; a fork
  is no host sync, and a task that forks is still one task
  ("Granularity").

**The forks are a worker's own streams, not its lanes.** Each GPU worker
has a pool of fork streams beside its lanes, with no library handles,
made with the backend as its lanes are, never during a run. Its size is
the most any of the worker's tasks declares it forks onto
(`GpuTask::forks`, none by default), and `k` is the smaller of that and
`n`; with fewer than two, the items run on the task's stream itself, so a
count left out costs concurrency, never order. Forking onto another
lane's stream, as the C++ forks onto a stream each class carries, would
order the task behind whatever is queued on that lane, a dependency
the planner cannot see. Two nodes of one worker that fork share its pool, so their forked
items keep the order the worker's thread enqueued them in; that never
deadlocks, since the one thread enqueues all of a worker's work and every
wait it enqueues is on an event already recorded. No node is placed on a
fork and no edge names one, so the plan's text, its tables and its
goldens are those of the same DAG without forks.

**The one-electron builds fork only when asked.** On the gfx1200 the
forks' events cost more than the classes' overlap wins at the sizes
tested, so their forking is an option, off: a builder sets
`ClassRows::fork` on the records it passes in, and with it unset the
builds declare no fork and launch in order on their lane, the order they
had before forks existed. The bench is in
`fb_integrals_one_electron_core_hamiltonian`'s crate doc; the
density-fitted three-center build, with larger classes, is the likelier
payoff.

**The scope is what makes a fork sound.** The forks start after the
task's input waits and are joined before its body returns, so everything
the executor records on the task's lane afterwards covers the forked
work: its completion event, which every consumer on another lane waits
on, and every handoff of its rows' bytes to a later owner. What a row's
`DeviceSlice` promise and the class records' `ClassRows` argument rest
on, that a launch goes on the stream the executor ordered, holds in
effect. The API allows no fork without its join: a fork stream is
reached only as the closure's borrowed `&Stream`, as a scoped thread is
only inside `std::thread::scope`, and a failed item or fork still joins
every fork already made before the error returns, as `launch_each` does.
A worker drains its forks with its lanes as it exits, so a run returns
with nothing outstanding even after a failed join.

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
per GPU worker inside it, whatever its lanes; the scope joins them all when
the run ends. A ready queue of node ids feeds the CPU workers, another per
GPU worker feeds its thread across all its lanes, and a GPU worker's thread
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
host's `available_parallelism` less the GPU workers' threads, floored at
one, and the config overrides it beside the `[memory]` caps. A placement that
needs more CPU workers than the budget is refused in the same admission pass
that refuses an arena that does not fit.

**Fan-out is nodes, not a pool inside a task.** Writing three files when the
RHF step finishes is three CPU nodes with no edge between them, each reading
an edge from the step; the orchestrator sees three ready nodes and runs them
at once. A task that opened a pool of its own to do the same (the host
integrals below are the one exception, and say why) would hide the
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
sleep; and a GPU worker is a thread dedicated to one device, which a
general pool has no notion of. Tokio is for many sockets, and its file I/O is
a blocking thread pool underneath. The `threadpool` crate's `'static` bound on
a job forces every edge buffer behind an `Arc` and defeats the scoped borrow.
The workers live behind one type in one module, so that if a profile ever
asked for stealing on the CPU side a crate could slot in behind the same type.

**The host integrals run on rayon, and nothing else does.** The one-electron
matrices S, T and V and the density-fitting metric (P|Q) are built on the
host, not on a card ("The MainBus and its steps", above): each is a loop over shell pairs, one
rayon task per center writing a disjoint run of rows of a packed triangle,
with nothing in it for the planner to place. They run on rayon's threads,
called from inside one step, a second pool beside the orchestrator's workers
with a thread count of its own. Stealing there is no placement the plan
cares about, for the reason a GPU task's forks are none ("Forks inside a
task"): the tasks start and finish inside one node, and each crate's
`the_thread_count_does_not_change_a_bit` holds the triangle bitwise the same
on one thread as on eight. The crates do not log, so no line comes from a
thread without a task record. The crates that may name rayon are those
four, under `rust/integrals/`: `fb-integrals-s-overlap-host`,
`fb-integrals-t-kinetic-host`, `fb-integrals-v-potential-en-host` and
`fb-integrals-df-metric`. A crate above them reaches rayon only through
them, never by an edge of its own, and no member depends on any other pool
crate but `fb-core-flow` on rustflow (below). Each needs the FMA target feature, which the workspace's
`.cargo/config.toml` sets for every member, so a dependent built from
`rust/` gets it. The pool is one `HostPool` (in `fb-integrals-df-metric`,
since only these crates may name rayon), which `fb-steps` ("The MainBus and
its steps") starts in setup with `[resources] host_threads` threads: by
default the host's `available_parallelism` less the executor's `threads`,
floored at one, as `n_cpu` above, and 0 refused. Every host integral call
from a step runs inside its `install`, so rayon's global pool is never
built, and the pool's threads are joined when the bus drops. A run's
threads are then the executor's and the pool's beside the main thread,
which `steps/tests/steps_threads.rs` counts exactly after a device-free
run and again after the plan drops. Each crate writes its triangle into a
caller's slice (`overlap_into` and its kin), so a host step builds it in its
pinned staging and its `run` allocates nothing of its size
([`memory.md`](memory.md), "The budget").

**The v0.0.4 layer's executor, and its fan-out.** The step DAG runs on
rustflow's `Executor` ("Steps on rustflow"), not on the orchestrator's
workers: one per process, its thread count fixed at init from the config,
a work-stealing scheduler whose stealing moves which thread runs a step,
never where it runs, since a step's GPUs are fixed before budget mode.
rustflow brings `crossbeam-deque`, the second exception to "no pool
crate": `fb-core-flow` is the one member that names rustflow, and it
re-exports `Executor`, so a crate above it reaches the executor only
through it. A sync blocks its worker, so the width covers the GPU steps
that can wait at once plus the CPU work, a handful of threads. An all-GPU
step's inner work across its GPUs is `fb_core_flow::fan_out`: one scoped
thread per GPU for the length of the fan-out, each making its device
current, joined before it returns, run inline for one GPU. It is the one
place the layer spawns, and the host integrals' rayon pool, called from a
step, is the other pool beside the executor.

**What is tested.** Every node runs exactly once, in a topological order, on
no more threads than the plan said. The last is a tier-3 test
([`testing.md`](testing.md), "Tiers: where a check lives") that reads the
`Threads:` line of `/proc/self/status` after a device-free run and compares it
to the plan's worker total plus the main thread, over GPU workers of several
lanes each, and reads the same count before a loop's first iteration and
after its last. It is the one check that
catches what a lint cannot see: a linked host library opening a pool of its
own at load, as a host BLAS does. A HIP build links one, ROCm's host
OpenBLAS, through hipSOLVER: its load-time constructor starts one thread per
core before `main`, and no call of the tree runs on it. So the v0.0.4
layer's setup, before it starts a thread of its own, sets it to one thread
and joins that pool (`fb_core_device::single_thread_host_blas`, which finds
OpenBLAS's controls by name in the loaded objects and does nothing on a
build without one), and `steps/tests/steps_threads.rs` counts against the
main thread alone, which is what notices when that slips.

### Capabilities are enforced by lints

Three things a task may not do are each held by named code: allocating is
the arena's ("Memory: layouts and arenas"), spawning is the worker modules',
and `unsafe` is a listed few crates'. Each is a lint, in the manifest's
`[lints]` table that [`documentation.md`](documentation.md) ("Code:
rustdoc") already uses for `missing_docs` or at a crate's root, so a violation is a build failure in every invocation, the
gates and an editor alike, and not a convention a review has to remember.

**Spawning.** `std::thread` cannot be removed from the standard library, but
clippy's `disallowed_methods` makes calling it a compile error everywhere but
the modules with the `allow`:

```toml
# clippy.toml
disallowed-methods = [
  { path = "std::thread::spawn", reason = "workers belong to the orchestrator, or to an all-GPU step's fan-out" },
  { path = "std::thread::scope", reason = "workers belong to the orchestrator, or to an all-GPU step's fan-out" },
  { path = "std::thread::Builder::spawn", reason = "workers belong to the orchestrator, or to an all-GPU step's fan-out" },
  { path = "std::thread::Builder::spawn_scoped", reason = "workers belong to the orchestrator, or to an all-GPU step's fan-out" },
]
```

The allows are two modules: `core::thread_pool`'s `run`, which opens the
old layer's workers, and `core::flow`'s `fan_out`, an all-GPU step's one
scoped thread per GPU ("CPU workers"). rustflow's executor spawns its own
workers, out of clippy's sight, which is why the manifest holds it. The
manifest does the rest: rayon is a dependency of the four host integral
crates alone, rustflow of `fb-core-flow` alone ("CPU workers"), tokio and
every other pool crate of none, and a pool crate arriving later is a
`Cargo.toml` diff a review sees.

**`unsafe`.** The rebuild's `unsafe` is the FFI into `wwr-sys` and the
kernel libraries, and the code that wraps it. `unsafe_code` is rustc's own
lint, and `forbid` is the level that cannot be allowed back, so it is set
per crate: a crate on the list below leaves it at its default, and every
other crate forbids it with `#![forbid(unsafe_code)]` at its root. That is
an attribute rather than a manifest line because Cargo refuses a member that
inherits `[workspace.lints]` and adds a `[lints.rust]` table of its own. The
list is in two tiers:

- **Core:** `fb-core-wwr`, `fb-core-arena`, `fb-core-log`,
  `fb-core-thread-pool`, `fb-core-device`, `fb-core-budget`.
- **Bindings:** `fb-integrals-sys`, `fb-localization-sys`, `fb-recipes-sys`,
  `fb-recipes-launch`.

The core crates hold the wrappers: `fb-core-wwr` over the `wwr-sys`
bindings (and the `statvfs` of `/dev/shm` a collectives' system error
reports), the arena over the raw allocation and copy bindings, the two
`libc` calls in `core::log`, and the one `unsafe impl Sync` in
`core::thread_pool`, on `run`'s private wrapper of the backend it shares
with its workers, which rests on a run borrowing the backend exclusively
and on `Backend` being sealed ("Plan once, run many"). `fb-core-wwr` also
holds the `unsafe impl Send` of the handles a step carries between
workers: `Stream` and `Event`, `Sync` as well, on the runtime's thread
safety; the BLAS and dense-solver handles, `Send` but never `Sync`, since
the libraries allow a handle on any thread one call at a time. Each impl
states its argument and is asserted at compile time beside it. The CCL
rank stays neither. The binding crates declare a kernel
library's `extern "C"` functions, check its version at load, and wrap its
launches safely beside the bindings (`fb-integrals-sys` its records tasks
too, since the class launchers trust records only those tasks write, and
`fb-localization-sys` its Pipek-Mezey tasks, whose launches trust an
orbital schedule and atom offsets only those tasks stage); the recipes'
safe calls are `fb-recipes-launch`'s, a crate apart from `fb-recipes-sys`'s
bindings because they take `fb-core-wwr`'s types, which the v0.0.4 layer's
`fb-core-device`, calling those bindings itself, must not reach
([`gpu_layer.md`](gpu_layer.md), "The v0.0.4 runtime"). There
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
`fb-recipes-blas`, the C++ recipes' sandwich products over those calls
(and no single call of its own: [`gpu_layer.md`](gpu_layer.md),
"Layering"), forbids `unsafe`
itself, as the arena's `memcpy_*` do for the copies. A kernel library's
launch is wrapped the same way in its binding crate, beside the generated
bindings (the recipes' in `fb-recipes-launch`, over `fb-recipes-sys`'s):
`fb_recipes_launch::inverse_sqrt`, `merge_sort` and
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

All four are on. `missing_panics_doc` holds part of the other half of
the `unsafe` rule: the length an `unsafe` call trusts is checked in every
build, and where that check is a body `assert!` the lint fails a public
function whose doc has no `# Panics` naming it. It does not see a
`#[requires]`, which rustdoc renders in `§ Contracts` instead
([`testing.md`](testing.md), "Contracts are part of the spec").

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
- **Lanes from the DAG's shape.** The builder names each GPU node's lane
  today, as the C++ mmd names a stream per class. Two nodes with no path
  between them could be put on two lanes by the planner instead, but a
  lane is a stream, and a library handle pair and a BLAS workspace row
  when its tasks call a library, so the
  count is a cost, and which nodes are worth splitting is the schedule
  search's question below. Forks inside a task do not answer it: they
  are concurrency within one node, which the DAG does not see into
  ("Forks inside a task").
- **How loose is too loose.** A bound from the dimensions is the preferred
  answer to a data-dependent size. At what ratio of bound to typical size
  does a sizing node pay for its extra host round trip?
- **The schedule search.** Finding an order whose peak fits is a scheduling
  problem that is hard in general. A plan order from the config, improved
  greedily, is probably enough for the DAGs this tree has. When is it not?
- **Transport across nodes.** The collectives library moves the node row
  of the worker table ("Workers are a hierarchy"); the nodes row needs a
  process per node and a clique made from a shared id
  (`wwr_ccl_comm_init_rank_scalable`), or MPI beside it. Nothing in the
  model depends on the choice, and nothing here decides it.
- **Pricing edges across devices.** The worker table's "GPUs on one
  node" row says what a broadcast and a reduce are there, and the plan
  names each collective's bytes, but the planner chooses no placement, so
  it prices none. When placement becomes its search, the price of a
  collective is a measurement per topology (NVLink pairs and PCIe through
  a host bridge differ by an order on the A6000 node) before it is a
  formula.
- **Pinned blocks and NUMA.** On a multi-socket node, pinned pages on the
  GPU's own socket copy faster than pages across the interconnect. A pinned
  block is allocated on its worker's thread, so pinning that thread to the
  GPU's socket would place the block without the planner knowing. Whether
  the gap is worth the pinning is a measurement on such a node.

**The v0.0.4 layer's**, to discuss:

- **Several nodes.** The commitment is to one node's GPUs. Keeping several
  nodes open cheaply is one rule: work on different GPUs communicates only
  through the collectives layer. B on one host's heap is the first thing
  that rule would have to answer.
- **What is cleared between replays, and by whom**: a reset mode per step
  (as built), an explicit `on_replay`, or a node at the loop's entry or
  end.

And to measure or verify:

- **The blocking sync**: which scheduling flag makes a waiting worker
  sleep, on each backend.
- **The executor's width** against the GPU steps that can wait at once, on
  a sync or a lock.
- **An inner graph on rustflow**: none is built. The case to measure is a
  step whose inner work has independent host-heavy parts, which no step of
  the DF-RHF chain has.
- **Rebuilding only the records on a replay**, if the full rebuild costs.
- **Collectives**: whether NCCL and RCCL mind a rank moving between steps'
  streams.
- **Idle workers**: what rustflow's parking costs while a run waits on the
  device.
