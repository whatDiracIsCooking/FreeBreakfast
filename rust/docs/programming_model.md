# Rust rebuild: programming model

We design a feature as a DAG of steps before we write it as code. A node is
a step: one config table, the `Output` it owns, its `mem_budget` and its
`run`. An edge is an `Output` one step produces and another reads. The
executor runs the DAG twice, once to budget and once to compute; the steps
only compute.

The model is not new to this tree. The C++ pipeline is already an instance of
it, in its most restricted form, and the first section says exactly which
restrictions the Rust rebuild lifts. The rest of the document is the
contract: what a step declares, what the MainBus and the `EriManager` hold,
the threads a run has and the capabilities a crate may hold.

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
the device becomes a node's GPUs. A step's GPUs are fixed before budget
mode, so the first job on one GPU is not running steps in parallel but
fitting the budget ([`memory.md`](memory.md), "The budget"); parallelism is
whatever room the DAG leaves.

## Steps on rustflow

The layer is `fb_core_flow`, over `fb_core_budget` and `fb_core_device`,
the design the v0.0.4 prototype was built to try; its streams, locks and
handles are [`gpu_layer.md`](gpu_layer.md)'s, "The v0.0.4 runtime".

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
  `[eri]` the metric and the ERI; `[scf]` the SCF; `[localization]` the
  localization of the SCF's occupied orbitals; `[resources]` the GPUs,
  the executor's width and the caps. A step whose table is absent is not in
  the DAG, and a step that reads one the config leaves out is refused.
  A relative path a key names (`[molecule] xyz`, `[guess] from`, and a
  `[basis] name` or `[eri] aux_basis` that is not a compiled-in set's
  name) is read against the config file's directory, never the working
  directory; a compiled-in name is passed on unchanged.
- **`calculation` picks the steps and the products.** `"overlap"` is the
  overlap alone, `"guess"` the guess alone, `"rhf"` the chain to the SCF,
  `"localization"` that and the localization; the optional tables it reads
  are filled at their defaults as the config is parsed, so the rule above
  still places each step. A table it does not read is refused at the
  table's line, and `"rhf"` or `"localization"` with no `[eri]` at
  `calculation`'s, each naming the file (`InvalidConfig`). The
  `free-breakfast` binary is `fb-steps`' own: a config through `run`, its
  products written by `Run::write`, its `main` the one boundary
  ([`diagnostics.md`](diagnostics.md), "The one boundary").
- **`[guess]` chooses the guess's density, the SAD density when absent.**
  `from` loads a float64 `.npy` (a run's `d.npy`) into the guess's staging
  in place of the SAD stamps, so its budget is the SAD guess's. Setup
  refuses it beside `type` or of a shape other than `n_ao × n_ao`
  (`InvalidConfig`), and with an open shell (`Unsupported`: `d.npy` is
  ROHF's two spin densities summed, which cannot restart it), before
  anything is allocated, naming the key, the file and the line.
- **`[molecule]` sets the charge and the multiplicity, and setup counts
  the occupations.** `charge` (default 0) and `multiplicity` (default 1)
  give `Dims`' `n_alpha` and `n_beta`, `n_alpha - n_beta = multiplicity -
  1`, `n_beta` also the doubly occupied. A multiplicity the
  electron count cannot have is refused `InvalidConfig`, naming the key,
  the file and the line ([`documentation.md`](documentation.md),
  "Configuration: the reference is emitted from the type"); an open
  shell, `n_alpha > n_beta`, runs ROHF in the SCF. A `mem_budget` over
  the AO space reads the occupations only for whether the shell is open
  (the SCF's open-shell scratch, the J/K pass's second density), so its
  budget for a closed-shell ion is the neutral's. A step whose arrays span
  the occupied space, the localization, sizes them by its blocks of C's
  columns, `[0, n_beta)` and `[n_beta, n_alpha)`: an n_ao bound would cost
  it `(n_ao / n_occ)²`, 22 times at gly10, in its population tensor.
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
  the SAD density are built on rayon's threads ("CPU workers"), or a
  loaded guess read from its file, written
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
  their download's token once, after the turn's last enqueue. An open
  shell's iteration is ROHF's: one J/K pass over its alpha and beta
  factors, the spin Fock matrices, and the Roothaan effective Fock matrix
  in place of F, on which the residual, the DIIS step and the Roothaan
  step are the closed shell's; its scratch follows the closed shell's in
  the one layout, and its `Output` adds the effective Fock matrix, the
  occupation classes C's columns `[0, n_beta)` and `[n_beta, n_alpha)`.
  `fb_steps::run`
  is setup, budget mode and execute mode over the run's one executor, and
  `Run::replay` execute mode again in the same buffers: the SCF's orbitals
  carry over in its `Output`, nothing in scratch does.
- **The localization is a gpu0 step under gpu0's lock whose sweep loop
  is plain code in `run`.** It reads the overlap's S and the SCF's C and
  orbital energies, and rotates each block of occupied columns by
  Pipek-Mezey sweeps on the population tensor; each turn reads `L`, the
  turn's orbitals `C U` and the sort key `diag(Uᵀ diag(ε) U)`, each
  column's `⟨i|F|i⟩`, through their downloads' token, the loop's turn
  read. The converged turn's columns are stably sorted by the key into
  the host copy. An open shell's two blocks, the doubly
  and the singly occupied columns, are each localized among itself and
  sorted by its own key, one after the other in the step's scratch, and
  `L` and the sweeps are summed over them.

### The EriManager

The density-fitted ERI is a shared product with state of its own: streams
and events, and a ring of buffers on every GPU. That state lives in one
object, `fb_steps::EriManager`, which is `eri`'s `Output`, and its users
call its methods. `build_jk(shared, held, d, j, k)` is the one that works:
J and K for each density of `d`, on gpu0, into the caller's `j` and `k`.

This section is the model's contract with it. Its mechanics are specified
beside the code, on the items' rustdoc pages, each with its figures
([`documentation.md`](documentation.md), "Diagrams and animations"):

| Page, under the site's `/api/fb_steps/` | Has |
|---|---|
| `struct.EriManager.html` | What the manager holds and where, B's two cuts (columns in the build, rows in a pass), the ERI buffer's tenants, and what `[eri] mem_per_gpu` gives |
| `struct.EriManager.html#method.build_jk` | A J/K pass: the densities' crossing, each GPU's share, the pass's ring, the contractions from D and from C, the reduction |
| `struct.EriStep.html` | B's build: the pair batches, the spherical kernel, the screen's bound, the build's ring and its launch order |

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
- **It takes a slice of densities.** DF-RHF passes one, and ROHF two,
  its alpha and beta factors, in one pass. A response solve, a Davidson
  or a state average would pass many. The buffers are sized for an open
  shell's two, a closed shell's one, from `Dims`, a contract of
  `build_jk`.
- **A density is D, or a factor C of D = 2 C Cᵀ** (`Density::Dense`,
  `Density::Occupied`). K is cheaper from a factor, by `n_occ / n_ao`, so
  a caller that has one passes it: the SCF does on every pass, its first
  the guess's eigen-factor.
- **A pass is whole when `build_jk` returns.** Only the matrix given
  crosses to the other GPUs: downloaded from gpu0 once, and one host end
  uploaded to each of them under one token. The pass joins the manager's
  streams before it returns: the wrapper does not know them.
- **B is built over pairs and contracted over aux functions.** B has one
  copy, on the host heap. `eri` builds it a batch of pair columns at a
  time, each GPU its own contiguous columns of every row; a J/K pass
  uploads it a batch of rows at a time, each GPU its own fixed share, and
  no GPU keeps rows of it between passes. No array in the DF-RHF chain
  has a Cartesian index.
- **The build is Schwarz-screened.** `[eri] schwarz_threshold`, 1e-12 by
  default (the old path's) and at least 0, drops a shell pair whose
  summed record bounds are under it: the pair has no cells and its
  columns of B are zero, and every other column is the unscreened one.
  B and the ring stay at the unscreened bound `Dims` gives
  ([`memory.md`](memory.md), "Batched only"), so the budget does not
  depend on the threshold, and 0 is the unscreened build.
- **Derivatives of B are scratch.** A gradient or a Hessian step builds
  the derivative of B a batch at a time and contracts it at once, inside
  the step that needs it. It reads the metric and L, which is why the
  metric is an `Output` of its own and L is the manager's. No such step
  is built yet.

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

## CPU workers

A run's threads are few and fixed, and that is enforced rather than hoped
for, because the place a thread count gets out of hand is never the one that
was designed. The step DAG runs on rustflow's executor; the host integrals
run on one rayon pool; an all-GPU step's inner work runs on `fan_out`'s
scoped threads. No step opens a pool or a thread of its own.

**The host integrals run on rayon, and nothing else does.** The one-electron
matrices S, T and V and the density-fitting metric (P|Q) are built on the
host, not on a card ("The MainBus and its steps", above): each is a loop over shell pairs, one
rayon task per center writing a disjoint run of rows of a packed triangle.
They run on rayon's threads, called from inside one step, a second pool
beside the executor with a thread count of its own. Stealing there moves
no work between steps: the rayon tasks start and finish inside one step's
`run`, and each crate's
`the_thread_count_does_not_change_a_bit` holds the triangle bitwise the same
on one thread as on eight. The crates that may name rayon are those
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
floored at one, and 0 refused. Every host integral call
from a step runs inside its `install`, so rayon's global pool is never
built, and the pool's threads are joined when the bus drops. A run's
threads are then the executor's and the pool's beside the main thread,
which `steps/tests/steps_threads.rs` counts exactly after a device-free
run and again after the plan drops. Each crate writes its triangle into a
caller's slice (`overlap_into` and its kin), so a host step builds it in its
pinned staging and its `run` allocates nothing of its size
([`memory.md`](memory.md), "The budget").

**The executor, and its fan-out.** The step DAG runs on rustflow's
`Executor` ("Steps on rustflow"): one per process, its thread count fixed at init from the config,
a work-stealing scheduler whose stealing moves which thread runs a step,
never where it runs, since a step's GPUs are fixed before budget mode.
rustflow brings `crossbeam-deque`, the second exception to "no pool crate", after rayon: `fb-core-flow` is the one member that names rustflow, and it
re-exports `Executor`, so a crate above it reaches the executor only
through it. A sync blocks its worker, so the width covers the GPU steps
that can wait at once plus the CPU work, a handful of threads. An all-GPU
step's inner work across its GPUs is `fb_core_flow::fan_out`: one scoped
thread per GPU for the length of the fan-out, each making its device
current, joined before it returns, run inline for one GPU. It is the one
place a step spawns, and the host integrals' rayon pool, called from a
step, is the other pool beside the executor. Outside the layer, the host-only
stamp generator `tools/sad` runs each element's serial atomic SCF on scoped
threads of its own in its `compute`, as many as the CPU affinity allows,
joined before it returns: it has no executor, and no pool crate may serve
it.

**What is tested.** A run's threads are counted, which is the one check that
catches what a lint cannot see: a linked host library opening a pool of its
own at load, as a host BLAS does. A HIP build links one, ROCm's host
OpenBLAS, through hipSOLVER: its load-time constructor starts one thread per
core before `main`, and no call of the tree runs on it. So the steps'
setup, before it starts a thread of its own, sets it to one thread
and joins that pool (`fb_core_device::single_thread_host_blas`, which finds
OpenBLAS's controls by name in the loaded objects and does nothing on a
build without one), and `steps/tests/steps_threads.rs`, which reads the `Threads:` line of
`/proc/self/status` after a device-free run, counts exactly the main
thread, the executor's and the pool's, and the main thread alone after the
plan drops, which is what notices when that slips.

## Capabilities are enforced by lints

Three things a step may not do are each held by named code: allocating is
a `Budget`'s, in `mem_budget` ([`memory.md`](memory.md), "The budget"),
spawning is `fb_core_flow`'s, and `unsafe` is a listed few crates'. Each is
a lint, in the manifest's `[lints]` table that
[`documentation.md`](documentation.md) ("Code: rustdoc") already uses for
`missing_docs` or at a crate's root, so a violation is a build failure in every invocation, the
gates and an editor alike, and not a convention a review has to remember.

**Spawning.** `std::thread` cannot be removed from the standard library, but
clippy's `disallowed_methods` makes calling it a compile error everywhere but
the modules with the `allow`:

```toml
# clippy.toml
disallowed-methods = [
  { path = "std::thread::spawn", reason = "workers belong to rustflow's executor (core::flow), or to an all-GPU step's fan-out (core::flow::fan_out)" },
  { path = "std::thread::scope", reason = "workers belong to rustflow's executor (core::flow), or to an all-GPU step's fan-out (core::flow::fan_out)" },
  { path = "std::thread::Builder::spawn", reason = "workers belong to rustflow's executor (core::flow), or to an all-GPU step's fan-out (core::flow::fan_out)" },
  { path = "std::thread::Builder::spawn_scoped", reason = "workers belong to rustflow's executor (core::flow), or to an all-GPU step's fan-out (core::flow::fan_out)" },
]
```

The allow is one module, `core::flow`'s `fan_out`, an all-GPU step's one
scoped thread per GPU ("CPU workers"), beside the stamp generator
`tools/sad`'s `compute`, which has no executor. rustflow's executor spawns its own
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

- **Core:** `fb-core-log`, `fb-core-device`, `fb-core-budget`.
- **Bindings:** `fb-integrals-sys`, `fb-localization-sys`, `fb-recipes-sys`.

The core crates hold the wrappers: `fb-core-device` over the `wwr-sys`
runtime, BLAS and dense-solver bindings, with the `unsafe impl Send` of the
handles a step carries between workers (`Stream` and `Event`, `Sync` as
well, on the runtime's thread safety; `GpuResources`, `Send` but never
`Sync`, since the libraries allow a handle on any thread one call at a
time); `fb-core-budget` over the raw allocations, with the slices' and
leases' marker impls; and the two `libc` calls in `core::log`. Each impl
states its argument beside it. The binding crates declare a kernel
library's `extern "C"` functions, check its version at load, and wrap its
launches safely beside the bindings, over `fb-core-device`'s stream and
slices, each call uploading the indices its kernel trusts itself; the
recipes' safe calls are `fb-core-device`'s, which calls
`fb-recipes-sys`'s bindings itself ([`gpu_layer.md`](gpu_layer.md), "The
v0.0.4 runtime"). There
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
`wwr-sys`: `fb_core_device::GpuResources`' `gemm`, `dgmm` and `syevd`,
reached through the GPU's `Mutex` guard, take their operands as
`DeviceSlice`s (device memory with its length, which the budget's buffers
and leases implement) and check each extent against them in every build
(syevd's work against the solver's own query), and `Stream::zero` zeroes
one on a stream. A kernel library's launch is wrapped the same way in its
binding crate, beside the generated bindings: `fb-integrals-sys`'s
`mmd_records`, `three_center_spherical_class` and `packed_gather` take
`fb-core-device`'s slices, assert every length, check every index their
kernel follows on the host and upload it themselves, and map the launch
status through `fb_core_device::launch_status`; the recipes' launches are
`GpuResources`' methods, on the same terms. The handle workspace, which
the library keeps beyond any borrow, is handed over by value instead: the
final node allocates it and `GpuResources::bind_workspace` owns it until
the handles are destroyed ([`gpu_layer.md`](gpu_layer.md), "The v0.0.4
runtime"). Every crate above the core and binding crates has nothing left
to call raw and forbids `unsafe`: `fb-steps` and the four host integral
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

The same DAG must give the same answer, to a declared tolerance, on one GPU
and on several. Where every reduce is ordered, that tolerance is zero and
the result is bitwise. Where a reduce is unordered, its declared tolerance
is the bar, and the test tier compares the one-GPU result against a
several-GPU one at that bar
([`testing.md`](testing.md), "Tiers: where a check lives", tiers 4 and 5).
Tolerances are fixed by analysis and never loosened to make a comparison
pass.

## Granularity

A step is the smallest unit with its own budget and no host sync inside its
`run`. A launch is below the line: a launch wrapper allocates nothing and
syncs nothing. A computation that must wait for part of itself on the host
is two steps, the wait the edge between them, which the wrapper's sync after
`run` provides. The exceptions are named, each an `allow` at its call: a
step reading a solver's flag after its last enqueue, the `EriManager`'s
ring waiting on one slot's copy, and a loop deciding on the host reading
its turn's scalars once per turn ([`rules.md`](rules.md), "Design"). A chain
of launches on one stream is not a sync, so it stays one step.

## Open questions

To discuss:

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
