# rust/v0.0.4: the design, and the prototype built from scratch

**The design of the Rust rebuild as it stands at v0.0.4, and what v0.0.4 will
build.** The executor becomes
[rustflow](https://github.com/whatDiracIsCooking/rustflow), and the memory
model, the planner and admission are replaced by one function per step that
counts and allocates. Agreed in discussion and not yet built: the code is the
prototypes `../v0.0.2` and `../v0.0.3`, and nothing under `rust/`. `rust/docs/`
describes the tree as it is, and changes with the implementation. There is no
crate beside this page yet. Written 2026-11-25, after `../v0.0.3`.

**This page is the design document.** It absorbed `docs/rust-redesign.html`
("Rust rebuild: steps, the memory budget and rustflow") on 2026-11-25, and that
file is gone. From here on the latest prototype's page is the latest design:
the next prototype's page carries this text forward with what changed, and an
earlier prototype's page stays as it was written.

Two things were decided on 2026-11-25, and the text below has them folded in:

1. **The card owns a stream, under its lock.** v0.0.3 could not build one
   handle set per card as the design worded it, with the handles rebound to
   the holder's stream. Now each card has a stream of its own, the card's lock
   guards that stream with the handles and the workspace, and nothing is
   rebound. A step owns its own streams; a step that calls the library also
   borrows the card's ([one handle set per
   card](#one-handle-set-per-card-under-one-lock)).
2. **The new layer is written from scratch over the C ABI.** Not over
   `rust/`'s handle wrappers, its safe kernel wrappers or its task layer, which
   were shaped by the memory model this design retires. The kernel libraries
   and the device-free crates stay ([from
   scratch](#built-from-scratch-over-the-c-abi)).

What v0.0.4 builds on top of that is [the DF-RHF chain as
steps](#the-v004-prototype), on `num_gpus` cards, batched. Like its
predecessors it will be throwaway: a crate and a workspace of its own, in no
gate, never merged into `rust/`.

In the design sections, "the experiment" is the executor experiment of
2026-11-19, "the prototype" with no version is `v0.0.2`, and v0.0.3 and v0.0.4
are named.

*decided* agreed in discussion, 2026-11-19 (executor) and 2026-11-20 (memory;
then views, managers and devices), and 2026-11-21 (the step boundary around B,
the target of an output, setup's lengths, step scratch, and library workspace
after `experimental/library_handle_workspace`; then step scratch packed by the
closure of the DAG, a step as one node of the outer graph, and the records and
work lists as scratch with no umbrella name), and 2026-11-23 (no managers:
shared code is a function over a `View` and its caller's scratch; B built in
batches of pairs, kept on the host and contracted in batches of aux functions;
batched only; a step runs on no GPU, gpu0 or all of them, and "target" is
dropped; scratch's contents and addresses over a replay; two thread pools),
and 2026-11-24 (no library stream: one handle set per card under one lock,
held for a step's run, with a fixed 128 MiB BLAS workspace; then scratch
packed by the closure alone and largest first, the lower bound in the budget
table, one layout function per step, small host scratch as a policy, a default
host cap checked at every call, and B's slab kept at the unscreened pair
bound; then, after `experimental/rust/blas_workspace_scaling`, no vendor
`trsm`, `potrs` or `getrs`: the tree's own recipes, as the C++ tree has them;
then the kernels' stack left out of the budget, the two default margins given
values, every cap clamped to what the machine has at init, and the free-memory
check dropped), and 2026-11-25 (the card's stream under its lock, with no
rebind; the new layer from scratch over the C ABI). *proposed* this document's
suggestion, not yet agreed: in [workspace](#blas-and-solver-workspace), a
nightly sentinel test per vendor binding; in [shared
code](#shared-code-a-function-over-a-view-not-a-manager), batches of B kept on
the cards between passes; and the proposals in [carried from
v0.0.3](#carried-from-v003-and-what-stays-open). *carried* v0.0.3's, built
again unchanged. *to discuss* raised in review and set aside for a
fuller discussion; [collected at the end](#open-questions). *run* measured or
checked by an experiment, each section naming its own: the executor experiment
(under the old memory model), the prototypes, the library-workspace probe, the
closure model, the handle measurements, and the workspace scaling probe.
Everything is unbuilt unless marked run. "Verify" marks a claim recalled from
vendor or rustflow documentation and not checked against source.

## The shape

A calculation is a DAG of **steps**, as in the C++ `src/pipeline`. One shared
object, the **`MainBus`**, holds the dimensions, the config, the memory caps
and counters, and every step's `Output`. A step declares which other steps'
`Output`s it reads; the DAG's edges are built from those declarations, and its
`run` is handed a `View` of exactly them. Every step implements
**`MemBudget`**: an associated function `mem_budget` that charges the counters
for what the step will allocate and then allocates it. The same DAG runs in
two modes over one rustflow `Executor`.

1. **Setup.** Parse the config, the geometry and the bases on the host. Fills
   the dimensions (n\_atom, n\_ao, n\_aux, …), with [every
   length](#lengths-counted-in-setup-built-by-the-step) a step's arrays need:
   counted, not built. Not budgeted.
2. **Budget mode.** The DAG, each node calling its step's `mem_budget`: count,
   check, allocate into the `MainBus`. A final node packs step scratch and
   accepts or refuses the whole plan.
3. **Execute mode.** The same DAG, each node calling its step's `run`.
   Allocates nothing. Replays (an optimization, AIMD) write into the same
   buffers.

There is no planner, no table, and no byte reuse between the `Output`s steps
pass on. The peak of a kind is the sum of what the steps' `Output`s hold, plus
step scratch, plus a fixed BLAS-handle workspace. Step scratch is what dies
with its step, every library call's work buffer included; it is packed into
one slab per device and kind, and two steps share bytes only where the DAG
already orders them, by its [transitive closure](#step-scratch). No edge is
added for memory. That is the trade: some memory for a model small enough to
hold in one page, and one in which work stealing cannot interact with memory
at all.

## What changes

|  | Today (Rust tree) | Proposed |
|---|---|---|
| Executor | `fb_core_thread_pool`: scoped threads per run, a ready queue per GPU worker, loops unrolled by hand | rustflow's `Executor`, one for the process, work-stealing *decided* |
| Unit of the DAG | A task (`GpuTask`, `CpuTask`) with a `layout` | A step: one node of the outer graph, with `mem_budget` and `run`; its inner tasks are its own *decided* |
| Memory description | `layout`: rows of the plan's table | `mem_budget`: charges and allocates *decided* |
| Who allocates | `Plan::instantiate`: one block per device and kind, carved by the table | Each step, in budget mode, through a `Budget` handle *decided* |
| Byte reuse | The assigner packs rows by lifetime; every reuse is a handoff edge | None between `Output`s. Step scratch only: packed per device and kind, two steps' regions overlapping only where the DAG's closure orders them; no edge added *decided* |
| Admission | Block + reserve + stack against the cap; the reserve measured per worker | An atomic counter per kind against the cap; no reserve measurement *decided* |
| Adaptive widths | The planner bisects `bytes(width)` against the room left | Every step over an O(N³) tensor is batched, and has no other path: its own limit picks the width, and one batch on one card is the old unbatched case *decided* |
| Products | Rows of the arena, read by row handle | Each step's `Output` in the `MainBus`, behind an `RwLock`, read through the reader's declared `View` *decided* |
| Shared machinery | Task builders (`add_*`) called by each user | A [plain function](#shared-code-a-function-over-a-view-not-a-manager) over a `View` of the product it reads and a slice of its caller's scratch (the J and K build over B). No manager *decided* |
| Devices | A worker per device; tasks placed by the builder | `num_gpus` from the config. A step runs on no GPU, on gpu0, or on all of them; all-GPU work is reduced onto gpu0 *decided* |
| Streams | A lane, named by the builder: a stream and handles per lane | A GPU step's own streams, and one `CardResources` per card: a stream, the library handles made on it and their workspace, behind the card's lock *decided* 2026-11-25 |
| Bindings | Safe wrappers over `wwr-sys`: a handle that owns its stream, kernel launches that take arena rows | Written again over the C ABI: a library call takes the card's guard, a kernel launch takes a stream and slices of step scratch *decided* 2026-11-25 |
| Host sync | Never in a task | At the end of every GPU step, by the shared wrapper *decided* |
| Process-lifetime state | Devices; the backend is per plan | The executor, the devices and their `CardResources` |

## Steps and the MainBus

*decided* The `MainBus` is a deliberate god object: one place that holds every
step's `Output`. What it keeps from the C++ `Pipeline` is the `View`: a step
declares the `Output`s it reads, the builder makes the DAG's edges from that
declaration, and `run` receives a view of exactly those and not the bus. A
read with no edge behind it cannot be written.

*decided* The steps and the `MainBus` live in **one crate**. The tree's many
small crates exist so that a change invalidates little of the build and test
graph; the steps are the top of that graph, and invalidating them together is
fine. One crate also means the bus can name every `Output` type with no
dependency cycle.

| The `MainBus` holds | Written |
|---|---|
| The parsed config: every step's options, and its memory limit | Setup |
| The dimensions: n\_atom, n\_ao, n\_aux, n\_occ, and the bases they come from | Setup |
| The caps per kind, and the counters (below) | Setup; budget mode |
| One `RwLock<Output>` per step | Budget mode puts the allocated `Output` in place; the owning step's `run` fills it |

```text
FIGURE-01 (fig)
```

- **An `Output` holds what outlives its step's run; scratch holds the rest.**
  An `Output` is what other steps read, or what carries over a replay (the
  last geometry's orbitals as the next guess). Everything else the step
  allocates dies with the step and is [step scratch](#step-scratch): the DIIS
  rings, a Davidson subspace, a streamed pass's ring, the records and work
  lists, build intermediates. Revised 2026-11-21: the step's private buffers
  were fields of its `Output`, summed for the whole calculation. *decided*
- **A step is exactly one node of the outer graph.** Something is its own step
  only if another step reads its product, its product carries over a replay,
  or it has its own config table. Everything finer is the step's inner graph,
  built in `run`: a subflow, condition loops inside it, a pipeline, or a plain
  loop on one stream. The SCF iteration is a condition loop inside `ScfStep`'s
  subflow, not in the outer graph. Splitting a step is not free: its
  intermediates become `Output`s, which are summed (B is the case: one step,
  one 7.8 GB slab, where two would hold two). What a finer step would buy,
  overlap between its pieces, is available inside the step. It also gives
  scratch one lifetime, a node's run, so the closure is over nodes. *decided*
  2026-11-21
- **The lock is used plainly.** The owning step takes `write()`, a reader's
  view takes `read()`. No generation stamp, no `try_`, no assertion. It exists
  because Rust will not compile shared mutation across threads without a lock
  or `unsafe`; on a chain it is never contended. A thin wrapper hides the
  standard lock's poisoning `unwrap()`. *decided*
- **A step reads only what it declared.** The declaration is the edge, so a
  missing edge is a missing declaration, and that does not compile: the view
  has no such field. *decided*
- **Only the owning step writes, by visibility.** An `Output`'s mutating
  methods are private to its step's module. That is ordinary privacy.
  *decided*
- **A step is one config table and one slot.** `[gradient]` is a single step
  as far as the TOML goes; what it computes depends on the energy method the
  config names. No step type has two instances, so slots need no instance key.
  *decided*
- **A step the config does not ask for is not in the DAG.** A step that
  declares it reads one is a refusal when the DAG is built, naming both, as
  the C++ planner does. So no declared read finds an empty slot; the types
  hold only that a read is declared, and the slot's emptiness is a runtime
  assert (an `Option` budget mode fills), as the prototype found. *decided*
- **Shared code with per-user state is a type, not a step.** DIIS is the case:
  the SCF's history and a coupled-cluster solver's cannot be shared and their
  vector lengths differ. Each user builds its own `Diis` in `run` over a slice
  of its step scratch (the ring of Fock and error matrices, on the device),
  declared in its own `mem_budget`. The history then lives exactly as long as
  the step's run, and a replay starts with none. Shared code over a shared
  product is the same thing with a `View` added
  ([below](#shared-code-a-function-over-a-view-not-a-manager)). *decided*
  2026-11-23
- **An `Output` can be loaded from a file**, in part or whole, as in
  `src/pipeline`. That is how a step is tested alone: its inputs are loaded
  and not computed. Budget mode still allocates; the load fills. *decided*
- **Replay allocates nothing.** A geometry optimization or an AIMD trajectory
  reruns the chain at the same dimensions, so each step writes again into the
  buffers budget mode made. What must not carry over between replays (a DIIS
  history, a converged flag) has to be cleared by someone; what should (the
  last geometry's orbitals as the next guess) must not be. Three candidates: a
  prologue node at the loop's entry, an epilogue node at its end, or a `reset`
  mode that some steps support. The prototype's reset mode sufficed; one way
  to make what carries over explicit is a trait method the wrapper calls when
  a replay enters (an `on_replay` with a no-op default). Tagged for later,
  2026-11-21. The split above narrows it: what lives in scratch never carries
  over, so only an `Output`'s fields need the rule. A DIIS history is scratch,
  so it is no longer part of the question. *to discuss*
- **`mem_budget` is an associated function.** Sizes depend on step options
  (`nroots`, a Davidson subspace) as well as on the dimensions, so it reads
  both from the `MainBus`.

### Shared code: a function over a View, not a manager

*decided* 2026-11-23: **there are no managers.** The earlier form made
`EriStep`'s `Output` an `EriManager`: an object owning B, the machinery over
it, scratch of its own and a lock around that scratch. Once the scratch moves
to the callers, nothing mutable is left inside, and what remains is an
`Output` and a function.

- **A shared product is an ordinary `Output`.** In density-fitting mode
  `EriOutput` holds the metric (P|Q) and its factor L on every card, and B.
  Steps read it through their declared `View`, like any other. Two steps may
  read it at once: each holds a read view.
- **The J and K build is a function.** It takes a `View` of B, the density,
  the J and K to write (the caller's), a slice of the caller's scratch, and
  the guards of the cards its caller holds ([the
  lock](#one-handle-set-per-card-under-one-lock)): `build_jk(&b, &density,
  &mut j, &mut k, scratch)`. It has no state of its own, so no internal lock
  and no question about concurrent callers. DIIS is the same without the
  `View` ([above](#steps-and-the-mainbus)).
- **The callee gives the size, the caller declares it.** A size function
  (`jk_scratch(dims, n_dens)`), pure in `Dims` and the options, with a row per
  card; each caller adds it to its own declaration in `mem_budget`, and the
  build works in the caller's region, since it runs during the caller's `run`.
  Packing then covers one density for the SCF, `nroots` for a state average
  and many right-hand sides for a response solve with no new rule. The
  [closure model](#what-the-closure-model-showed) checked both forms with two
  unordered callers: declared by the callers, safe in every simulated run;
  owned by the callee, one region used during both callers' runs, a collision
  in every one of 300.
- **It writes only what it is passed.** J and K belong to the caller. More
  generally, global mutable statics are minimized and, ideally, banned.
- **Whoever fans out, joins.** A function that recruits every card owns the
  whole of that: each inner task makes its card current with a
  `ScopedDeviceIndex`, the fragments each card produces (not necessarily
  disjoint) are reduced onto gpu0, and the call returns with that done. The
  caller sees an ordinary call. The caller is still an [all-GPU
  step](#work-stealing-what-the-one-thread-was-providing): the scratch and the streams on every card are its own.
- **One B, built over pairs, contracted over aux functions.** The two loops
  run over different indices, as `experimental/rust/df_batched/single_gpu`
  builds it:
  - *The build* runs over batches of orbital shell pairs (mn). A column mn of
    B depends on every P, so a batch of P cannot be fitted alone; a batch of
    pairs can. Each card takes its own batches, forms (mn|P), fits it against
    its own copy of L, and downloads the batch to the host. No card needs
    anything from another.
  - *B's home is the host heap*, one slab of `EriOutput`. The cards hold the
    O(N²) data and one batch each, which is scratch. So host memory is what B
    is held against, not gpu0's cap.
  - *The J and K build* runs over batches of P. Each card takes its own
    batches from the host, contracts them into fragments of J and K, and the
    fragments are summed onto gpu0.

  A second copy of B is ruled out by size.
- **Derivatives of B are scratch.** Analytic gradients, Hessians and higher
  derivatives build the derivative of B a batch at a time and contract it at
  once, inside the step that needs it. It is never an `Output`; its records
  and work lists are that step's scratch too. Those steps read the metric and
  L, so the two are real `Output`s and not, as an earlier draft had it,
  products only B reads.
- **Batches of B kept on the cards between passes.** Streaming uploads all of
  B on every J and K pass (4.3 s on gly10 in the
  [experiment](#what-the-experiment-showed), B on the heap). A cache over the
  host copy would keep as many batches on each card as the step's limit allows
  and stream the rest. An optimization for after the streamed path works.
  *proposed*

## The memory budget

*decided* The names: `MemBudget` the trait, `mem_budget` its function, *budget
mode* the pass that runs it, `Budget` the handle it allocates through.
`footprint` and `layout` are dropped. The C++ footprint returned a description
that tests held in step with the allocation; here the description *is* the
allocation, so there is still one program.

```text
pub trait MemBudget {
    /// Charge what this step allocates, allocate it, and put the
    /// `Output` in the bus. Budget mode calls it once per step.
    fn mem_budget(bus: &MainBus, budget: &Budget);
}

impl MemBudget for ScfStep {
    fn mem_budget(bus: &MainBus, budget: &Budget) {
        let n = bus.dims.n_ao;
        let dev = bus.device_of::<Self>();
        let output = ScfOutput {
            c:    budget.device(dev, n * n),      // summed: the CASSCF reads it
            eps:  budget.heap(n),                 // a fixed-size host slab
            ..
        };
        // Dies with the run: the Fock matrix, the DIIS ring, the eigensolve's
        // work. One region, packed by the final node, lent to run.
        budget.scratch(dev, Kind::Device,
            n * n + diis_bytes(bus) + syevd_lwork(n) + eri_jk_scratch(bus, 1));
        *bus.scf.write() = Some(output);
    }
}
```

A sketch, not a signature to hold anyone to. The step's streams are not in
it: the wrapper makes them and keeps them beside the `Output` in the slot, so
it can sync them without the step's cooperation (as the prototype found). What is
fixed is what each `Budget` call for step-owned memory does, in order:

1. **Add** the bytes to the kind's atomic counter (per device for device and
   pinned memory), and write them into this step's row of a per-step table.
2. **Check** the counter against the cap. Over it: set a shared `over_budget`
   flag. Do not panic here.
3. **Allocate**, unless the flag is set.

A `budget.scratch` call only records a row (step, device, kind, bytes); it
charges and allocates nothing. **The final node of the budget DAG** then:
takes the closure of the DAG and packs each device and kind's scratch rows
into one slab ([step scratch](#step-scratch)); adds each packed peak and the
fixed BLAS-handle workspace ([below](#blas-and-solver-workspace)) to its
counter and makes the same check; holds the heap and pinned totals against the
host's available memory; and, if the flag is set, prints the whole table with
the totals and exits. Otherwise it allocates the scratch slabs and each card's
handle workspace, binds the latter to that card's BLAS handle, once
([below](#where-the-handles-and-the-workspace-are-made)), records each step's
offset, and logs the table.

```text
FIGURE-02 (fig)
```

- **One pass, not a count pass and an allocate pass.** Two modes in all:
  budget and execute. *decided*
- **The same DAG, in parallel, with atomic counters.** A serial pass would be
  a different execution graph. The total is order-independent, so accept or
  refuse is the same every run; deferring the exit to the final node is what
  makes the *report* the same every run too, complete and not whichever step
  crossed the line. The only waste on a refusal is what was allocated before
  the flag flipped, and the process is exiting. *decided*
- **The device check has two moments**: per step for `Output`s, and once at
  the end for the packed scratch and the handle workspace. A plan can pass
  every step and be refused at the end; the table carries each as its own row
  so that reads plainly. *decided*
- **Running the passes in parallel earns its keep on the host**, where
  touching the pages of a multi-GB slab is the slow part.

### Lengths: counted in setup, built by the step

A kernel cannot walk a `BasisSet`. It reads flat arrays that say, for every
unit of work, where its inputs are and where its result goes; the host builds
them from the geometry and the bases, stages them in pinned memory and uploads
them. They are of two kinds. **Records** are data: one entry per primitive,
with its centre, exponent and weighted coefficient. **Work lists** are
indices: one entry per unit of kernel work.

*decided* 2026-11-21: **no umbrella name.** "Manifest" named them while they
were a budget category of their own, counted in setup, charged in
`mem_budget`, built in `run` and kept for the whole calculation. A step
rebuilds them on every run, so keeping them buys nothing: they are [step
scratch](#step-scratch), pinned for the staging and device for what the
kernels read, and once their step is done their bytes go to later steps. What
is left of the category is one rule, about lengths, and it covers every array
a step allocates, `Output` or scratch. A binding that launches over records
and work lists may name a struct holding both locally; the prototype's
`Tables` collides with the budget table and goes.

The density-fitting step's arrays, from
`experimental/rust/v0.0.2/src/steps/eri.rs`, with their sizes on gly10 in
cc-pVDZ / cc-pVDZ-JKFIT:

| Array | Kind | One entry per | Length, from `Dims` | gly10 |
|---|---|---|---|---|
| `aux_records` | records | aux primitive: x, y, z, α, c(π/α)<sup>3/2</sup> | 5 × `n_aux_prim` f64 | 51 kB |
| `metric_pairs` | work list | aux shell pair P ≤ Q, by class (l<sub>P</sub>, l<sub>Q</sub>) | 6 × `n_aux_shell`(`n_aux_shell`+1)/2 i32 | 19.4 MB |
| `cells` | work list | orbital shell pair × aux shell, by (orbital class, l<sub>P</sub>) | 6 × `n_shell_pair` × `n_aux_shell` i32 | 1.79 GB |
| `gather_at`, `gather_weight` | work list | Cartesian pair column: its μ and ν, and 1 or 2 | 2 × `n_col` i32, `n_col` f64 | 4.3 MB |

```text
FIGURE-03 (fig)
```

*decided* 2026-11-21: **setup counts, the step builds.**

- **Setup fills every length.** It has already read the geometry and both
  bases, and it is outside the DAG and unbudgeted, so it adds the counts the
  arrays need to `Dims`: `n_aux_prim`, `n_aux_shell` and `n_shell_pair` beside
  the `n_col` it already derives from the orbital `RecordTables`. Counting is
  a walk over the shells and builds nothing. A `mem_budget` is then a function
  of `Dims` and the step's options alone, as [the step
  contract](#steps-and-the-mainbus) says.
- **Setup counts every length that scales; it builds none of those arrays.**
  The rule for what setup may do: counts, and small host data the dimensions
  need. Any array whose length scales with a pair count, or a product of shell
  counts, has its length counted into `Dims` by setup, and belongs to a step
  and is budgeted: B's slab as much as the work lists. `cells` is the case:
  1.79 GB on gly10, which would otherwise sit on the plain heap with nothing
  counting it.
- **`mem_budget` charges, `run` builds.** `mem_budget` declares the records
  and work lists at their lengths from `Dims` as scratch, pinned and device.
  It builds nothing. `run` builds them straight into its pinned region (a
  counting pass over the shells gives each class's offset, so no temporary
  `Vec` is needed), uploads them on the step's stream into its device region,
  and launches.
- **A replay builds again, into the same regions.** The records hold
  coordinates, which move in an optimization or a trajectory; the lengths do
  not, and neither do the offsets. So every `run` rebuilds in place and
  allocates nothing. Rebuilding only the records (the work lists depend on the
  bases alone) is to measure.
- **A refused or dry budget builds nothing** and still prints exact lengths,
  since they are counts. This is what the prototype could not do: its
  `Tables::new` ran in `mem_budget` for the lengths and again in `run` for the
  contents, because a refused or dry budget hands back empty buffers it could
  not fill.
- **Not a panic on the first over-cap charge.** That would fix the refused
  case only (a dry run's buffers are empty by design) and would undo the
  decision above: in a parallel budget pass which step crosses the cap first
  is scheduling, so the report would name a different step from run to run
  instead of the whole table.

```text
FIGURE-04 (fig)
```

## What is budgeted, by kind

| Kind | Counter | What `Budget` hands back | Rule |
|---|---|---|---|
| Device | Sum, per device | A fixed-size device buffer | Owned by the step's `Output`: read by other steps, or carried over a replay |
| Pinned | Sum, per device | A fixed-size page-locked slab | As device |
| Heap | Sum, one for the node | A `HeapSlab<T>` over a `Box<[T]>` | Cannot grow; pages touched in `mem_budget` |
| Scratch | Packed, per device and kind (device, pinned) | A lease on the step's region of the slab, for the length of its `run`: build intermediates, solver and DIIS state, records and work lists, and every library call's work buffer | Packed and allocated by the final node; regions overlap only between steps the DAG orders ([below](#step-scratch)) |
| Handle workspace | A fixed constant, per device | Nothing: bound to the card's BLAS handle once, by the final node; the step never sees it | 128 MiB, allocated by the final node ([below](#blas-and-solver-workspace)) |

### Host heap

- **A boxed slice, not a `Vec`.** `Box<[T]>` has no `push`, `reserve` or
  `extend`, so "no growing" is the type's property and needs no lint.
  Everything taking `&[T]` or `&mut [T]` works unchanged. The `HeapSlab`
  newtype is constructible only by `Budget`, which separates budgeted memory
  from an incidental `Vec` in a signature. *decided*
- **Touch the pages in `mem_budget`.** Zeroed memory is handed out lazily on
  Linux, so an untouched slab "succeeds" without existing and fails at first
  touch mid-run, as a bare kill from the OOM killer. *decided*
- **The host is checked at every call, and again by the final node.** Heap and
  pinned bytes add to one host counter, held against the host cap at each
  `Budget` call, before the slab is allocated and its pages touched: a B
  larger than the host sets the flag and is never touched, where a check in
  the final node alone would come after the OOM killer. The final node repeats
  the check with the packed pinned scratch added. The default cap is the
  cgroup-aware reading that exists today (`fb_core_arena::host_available`),
  taken once in setup, less its margin
  ([caps](#caps-and-what-the-limit-covers)). *decided* 2026-11-24; the first
  form checked in the final node only.

### Step scratch

*decided* 2026-11-21. Memory a step needs during its `run` and that nothing
reads afterwards: the build intermediates of B, which were 68% of the
prototype's `eri` device bytes on benzene (237 of 347 MiB); the work buffer of
every library call (`syevd`'s at order n, …;
[below](#blas-and-solver-workspace)); a solver's DIIS ring or Davidson
subspace; and the records and work lists, staged in pinned memory and read on
the device. Without it the only home for them is the step's `Output`, summed
for the whole calculation.

- **Why not allocate and free it inside the step.** Two reasons. Allocation in
  `run` is ruled out ([Enforcement](#enforcement)), and a replay would
  allocate again, which breaks "a replay allocates nothing" and lets a replay
  fail late. And under the sum, a release frees bytes no step can use:
  admission happened up front and counted every step's bytes for the whole
  run, so what `eri` gives back mid-run was promised to nobody. What lowers
  the peak is sharing scratch between steps, not releasing it.
- **Declared per step, packed by the closure.** A step declares
  `budget.scratch(dev, kind, bytes)` in `mem_budget`, for device or pinned
  memory; the call records a row and allocates nothing. The final budget node
  takes the transitive closure of the DAG (a bitset of successors per node, at
  a few dozen nodes) and packs each device and kind's rows into one slab: two
  steps' regions may overlap only if one reaches the other. It packs by first
  fit, each region at the lowest offset clear of every unordered region
  already placed, in a fixed order, so the offsets, and the table, are the
  same every run. Then it allocates one slab per device and kind and records
  each step's offset.
- **The order: largest first.** First fit by size, then step id, and no other
  order. The [closure model](#what-the-closure-model-showed) found it at the
  lower bound on 98.1% of random DAGs and 1.281× the bound at worst; the best
  of four fixed orders reaches 99.2% and 1.174×, and is not taken. A poorer
  packing costs bytes and never safety (next item), real pipelines are mostly
  chains, and the budget table prints the lower bound beside the packed peak,
  so a config that packs badly shows itself. More orders are added when one
  does. *decided* 2026-11-24
- **The closure alone, not the card's lock.** Two steps that both hold a
  card's lock ([below](#one-handle-set-per-card-under-one-lock)) never run at
  once on it either, so they could share scratch there with no path between
  them. The packer does not use that: unordered regions stay apart whoever
  holds the lock. One safety argument and one test, for the bytes of the
  smaller of two unordered library users (in DF-RHF, the orthogonalizer's
  eigensolve work, 8.7 MB on gly10). *decided* 2026-11-24
- **Safe whatever the packer's quality.** Safety is the packer's constraint,
  not its objective: any placement that keeps unordered regions apart is safe,
  the best and the worst alike. A poorer packing costs bytes, and at worst
  refuses a plan a better one would have admitted: a refusal at init, never a
  collision in a run.
- **Why the closure is enough.** A finished node is finished device work (the
  wrapper's step-end sync), so a path in the DAG from A to B is a
  happens-before on the card as well as on the host: B cannot touch the bytes
  before A is done with them, under any schedule the executor picks. Unordered
  steps never share. So, unlike the old assigner, which made reuse possible by
  *adding* handoff edges, nothing here constrains the schedule, and work
  stealing still cannot interact with memory. The model's simulated runs on a
  stealing pool agree, and its negative control shows the sync is what carries
  it ([below](#what-the-closure-model-showed)).
- **The step-end sync now carries memory safety.** It was introduced to order
  the card between dependent steps ([streams](#streams-and-device-resources));
  with scratch packed, it is also what keeps bytes shared along a path from
  being used twice at once. Pinned scratch needs both of its halves: the host
  writes it inside `run`, before the node finishes, and the device reads it
  through copies on the step's streams, done by the time the wrapper's sync
  returns. So a cheaper replacement is not free. Event chaining (each step's
  stream waiting on its predecessors' join events, forwarded by host-only
  nodes) still orders device work along every path, which keeps device scratch
  safe; it does not order a host write in B after a copy A enqueued from the
  same pinned bytes. Any change to the sync is a change to the scratch rule,
  and is decided with it.
- **Computed once.** The graph and every length are fixed for the calculation,
  so the closure and the offsets are made once in budget mode and every replay
  runs in the same regions. A replayed chain's closure is taken over one
  iteration of its body, laid flat into the outer DAG with the steps before
  and after it: a module cannot run concurrently with itself, so one iteration
  ends before the next begins, two steps unordered within the body stay apart,
  as they must, and so does a step outside the loop that no path orders
  against it. The model packs once and runs three iterations unrolled: safe
  statically and in simulated runs.
- **A condition's exclusive branches are packed as unordered.** Two branches
  of which only one runs never use their regions at once, so they could share
  bytes. The closure does not know that, and keeps them apart: safe, and the
  branches' sum rather than their max (in the model, 220 against 120 MiB).
  Packing them as exclusive is also safe, and is not adopted: a choice made in
  the config never enters the DAG, and a runtime condition, such as a
  convergence loop, lives inside a step's subflow, so exclusive branches of
  the outer graph should be rare. *decided*
- **Over a replay the addresses hold and the contents do not.** A step gets
  the same region at the same address on every run. What is in it on entry is
  undefined, on the first run and on every replay: any step the DAG orders
  against it may have written those bytes in between, and nothing clears them.
  A step whose region happens to share with nobody would find its old data
  still there, and must not rely on it. The check: in a test mode the wrapper
  fills each lease with NaN before `run`, so a step that depends on what was
  left fails in the gate. *decided* 2026-11-23
- **One layout function per step.** A step's region holds several tenants (the
  SCF's: the DIIS ring, the J and K build's scratch, the eigensolve's work).
  `mem_budget` sums their sizes and `run` carves the region, so two pieces of
  code must agree. One function from `Dims` and the options to a struct of
  offsets serves both: `mem_budget` declares its total, `run` slices by it.
  *decided* 2026-11-24
- **A lease that cannot escape.** The wrapper hands `run` a lease on its
  regions that borrows for the length of `run`, not `'static`. A step cannot
  store its scratch in its `Output`, so "scratch dies with the step" holds by
  type. The overlapping slices need one `unsafe`, in the arena, with the
  closure invariant written beside it.
- **A step's inner tasks split its region themselves.** The closure is over
  the outer graph's nodes, a step being one node; two unordered tasks of one
  step's subflow share bytes only if the step arranges it. Inner graphs are
  small and written by hand, so the closure is not taken recursively.
- **What it keeps.** Everything allocated in budget mode; nothing allocated in
  execute mode; accept or refuse independent of order; no lock, so a
  scratch-using step never waits on another.
- **What it costs.** Bytes in place of time. The packed peak is at least the
  largest single declaration, and equals it only when every scratch user is
  chained. In DF-RHF, `eri` and the orthogonalizer are unordered, so their
  regions are disjoint; the SCF follows both and may reuse either. Revised
  2026-11-21: the first form was one max per device under a lock, which made
  `eri` and X serialize. Since 2026-11-24 the two serialize anyway, as library
  users under gpu0's lock ([below](#one-handle-set-per-card-under-one-lock));
  the regions stay disjoint, by the item on the closure above. Unordered steps
  that call no library still overlap on the card.
- **The host heap is not packed.** It stays a sum. What a step's `run` needs
  on the heap and drops is small, or it is a pinned staging buffer, which is
  scratch.
- **Small host scratch is a policy, not a kind.** A `Vec` or a `HashMap` that
  lives within a step's `run` is assumed small, and is documented as outside
  the budget; there is no host scratch kind and no suballocator. What holds
  the policy is the allocator record of [Enforcement](#enforcement), which
  flags any single large allocation during `run` in the test tiers. One number
  added to it: the peak heap growth over execute mode, printed as an
  unbudgeted row of the budget table. The rule a step is written to: heap
  scratch in `run` stays within O(n\_ao²), and anything larger is a `HeapSlab`
  on the `MainBus`. Every large object of the methods here has a length that
  is a closed form in the dimensions, so none needs to live outside the bus;
  what the record is for is the accident (`cells` was one). A host
  suballocator with the standard collections banned was raised again on
  2026-11-24 and not taken: allocator-aware collections are not on stable
  Rust, so it would mean rewriting every host routine over slices, to hold a
  rule the record already checks by size. *decided* 2026-11-24
- **A batch bounds the scratch.** Every build over an O(N³) tensor is
  [batched](#sizes-batched-only), so its scratch is one batch on each card it
  runs on, and an all-GPU step declares a row per card.
- **The same machinery would cover `Output`s.** An `Output` only one later
  step reads still holds its bytes for the whole calculation. Packing it too
  is the same closure over a different interval, from its producer to its last
  reader. Held until the measurement ([open questions](#open-questions)) says
  `Output`s, not scratch, are what stacks. *to discuss*

### What the closure model showed

*run* 2026-11-21,
[`experimental/dag_transitive_closure`](../../dag_transitive_closure/)
(its README and page have the detail; `model.py` pins every number below). A
model in plain Python, with no GPU and nothing built: a fixed dummy DAG,
loosely gly10-shaped, with sizes in MiB that are *not measurements*, and
seeded random DAGs. Two kinds of check. Static ones hold the packing against
the closure: no two regions of unordered steps intersect. Those cannot show
the closure is the right rule, so the execution checks simulate runs on a pool
of one to eight workers taking any ready step at random, as stealing would,
and look for two regions that share bytes while both are in use.

```text
FIGURE-05 (fig)
```

```text
FIGURE-06 (fig)
```

| Check | Result |
|---|---|
| The closure | The bitset closure equals reachability by plain depth-first search on 1000 random DAGs of 1 to 60 steps |
| Determinism | Offsets unchanged over 200 shuffles of the dummy DAG's declaration order and every permutation of six of its steps, for largest first and for the best of four |
| The dummy DAG | Packed peak 2049 MiB, equal to the lower bound; plain sum 3318; the lock's max 2040 |
| Random DAGs, 2 to 16 steps | 2994 with scratch, every one safe, and max ≤ bound ≤ packed ≤ sum on each |
| largest first | At the bound on 2938 (98.1%), worst 1.281×, mean 1.0014. On each of the 56 misses some other first-fit order reaches the bound: the shortfall is the order, not the bound |
| best of four orders | At the bound on 2970 (99.2%), worst 1.174×, mean 1.0003; never worse than largest first |
| Exact optimum, 4000 small DAGs | At most 8 regions, small integer sizes: the optimum equals the bound on every one (but see below). Largest first optimal on 3769, the best of four on 3953 |
| Simulated runs | 3500 runs, 1 to 8 workers, random ready steps: no two byte-sharing regions in use at once. Without the step-end sync: 2395 of 3000 runs collide |
| Devices and kinds | 1777 slabs over 500 random DAGs on two cards, device and pinned memory each: all safe. A slab's offsets do not move when another slab's regions change; an all-GPU step declares on every card |
| Scratch of code two steps call (the model's "manager") | Declared by its two unordered callers: safe, peak 500. Owned by the callee and used during its callers' runs: 450, and a collision in every one of 300 runs |
| Exclusive branches | Packed by the closure: 220, safe. Packed as exclusive: 120, also safe; not adopted ([above](#step-scratch)) |
| Replay | Packed once over one iteration; three iterations unrolled: no unsafe pair, and safe in simulated runs |

- **The bound is a floor, not a promise.** It was reached on every small DAG
  solved exactly, but not in principle: packing regions apart along a conflict
  graph is NP-hard even when the lifetimes are intervals, and interval orders
  are step DAGs too, while the heaviest antichain is found in polynomial time.
  So some DAG has its optimum above the bound; at a few dozen steps, the gap
  the model found is the order's and it is small.
- **The budget table shows it.** Beside the packed peak and the plain sum, the
  table prints the lower bound and the steps that make it, so how far packing
  is from the best any packing could do reads plainly. A branch and bound over
  a few dozen rows is instant. *decided* 2026-11-24, with largest first as the
  one order ([step scratch](#step-scratch)): the row is what says when a
  second order is worth adding.
- **What it does not show.** The sizes are dummies, so nothing here says how
  much packing saves on a real calculation: that is still [to
  measure](#open-questions), on the `examples/` configs. And the simulation
  models the sync; it does not test wwr's.

## BLAS and solver workspace

*decided* 2026-11-21, after
[`experimental/library_handle_workspace`](#what-the-probe-showed). Two
different things go by the name, and they are handled apart:

|  | The handle's workspace | A call's work buffer |
|---|---|---|
| What it is | Device memory a handle draws on for any call, set with `SetWorkspace` or managed by the library itself | The `work` argument of a dense-solver call, sized by its `_bufferSize` query (`syevd`, `potrf`, …) |
| Libraries | cuBLAS, hipBLAS over rocBLAS | cuSOLVER; hipSOLVER, which attaches it to its internal rocBLAS handle for the call |
| Size | A fixed constant per device: 128 MiB, with no exception: the calls it would not cover are not made (below) | The query at the call's arguments, declared in `mem_budget` |
| Home | The card's `CardResources`, allocated by the final budget node and bound to its BLAS handle once | [Step scratch](#step-scratch), lent to the step for its `run` |
| Protected by | The card's lock: one holder at a time, released only after the holder's step-end sync ([below](#one-handle-set-per-card-under-one-lock)) | The closure and the step-end sync, as for any scratch |

- **Every BLAS handle is user-owned, never library-managed.** Each card's BLAS
  handle always runs on a fixed workspace of the caller's, so rocBLAS never
  manages the handle's memory. 128 MiB, raised from 32 on 2026-11-24: 1.3% of
  a 10 GB RTX 3080 and 0.8% of a 16 GB RX 9060 XT, paid once per card. On
  cuBLAS the size only picks the algorithm (a 256-byte workspace still ran, by
  another algorithm); on rocBLAS it is a hard need: a call that needs more
  than it was given takes memory of its own, outside the budget, with no error
  ([the measurements](#one-handle-set-per-card-under-one-lock): rocBLAS's own
  `dtrsm` at 4096 × 4096 took 8 MiB). Setting it on hipBLAS also frees
  rocBLAS's own 32 MiB default pool. It is a fixed row of the budget table,
  not an exclusion. *decided*
- **No vendor `trsm`, `potrs` or `getrs`, on either backend.** On rocBLAS
  5.6.0 a `trsm` takes workspace that grows with its right-hand sides, up to a
  copy of B, and hipSOLVER's `potrs` and `getrs` take B's size plus about 42
  MiB by themselves and keep it
  ([below](#what-the-workspace-scaling-probe-showed)): no constant covers
  either, and none of the three takes a work buffer a binding could check. So
  the triangular solve is the tree's own: the C++ `recipes/trsm`, blocked as
  its Cholesky is, each diagonal block solved by a kernel and the rest updated
  by one `gemm`, which needs only the handle's fixed workspace. The solve
  against a Cholesky factor is two of those; the solve after `getrf` is a
  pivot kernel and two. The C++ tree made the change on 2026-11-24, on both
  backends (`recipes/trsm`, `cholesky_solve`, `lu_solve`, and every
  `wwr::trsm` call in `src/`); the Rust tree binds the same recipes and wraps
  none of the three vendor calls. One path on both backends, so CUDA runs the
  recipe as well. What it replaces, a size function for `trsm` and a binding
  that swapped a slice of the step's scratch onto the handle for the call
  (decided earlier the same day), goes, and with it the question of where the
  size would come from. Measured against both libraries (*run* 2026-11-24,
  #1782, [`src/recipes/trsm/README.md`](../../../src/recipes/trsm/README.md)'s
  table, from `bench/trsm_vendor.cpp`): at the density-fitting solve's shape,
  1 GB of B and up, the recipe takes 0.76 to 0.86 of cuBLAS's time on the RTX
  3080 and 0.99 to 1.04 of rocBLAS's on the RX 9060 XT, after its
  diagonal-block kernel was reworked (1.13 to 1.28 of rocBLAS's before); one
  right-hand side goes to the library's `trsv`, which takes no workspace, at
  the library's own time. What is left is small solves a few milliseconds
  long, up to 1.6 times the library's. *decided* 2026-11-24
- **A sentinel test per vendor binding, in the nightly.** What is left of the
  vendor libraries (`gemm` and the other BLAS calls, `syevd`, `potrf`,
  `getrf`) was measured to stay inside the fixed workspace or the size query,
  on one release of each. A test per binding, the workspace scaling probe's
  shape: the call on a sentinel-filled workspace or work buffer at a size
  where growth would show, twice, with the card's free memory read around it.
  A toolkit bump that changes what a call takes then fails there, and it is
  the only check that would see it: the step-end free-memory read is not taken
  ([below](#one-handle-set-per-card-under-one-lock)). *proposed*
- **A call's work buffer is the step's, like any kernel's scratch.** The step
  declares it in `mem_budget` from the size query (`budget.scratch(dev,
  Kind::Device, syevd_lwork(n))`), and passes a slice of its lent scratch to
  the call in `run`. There is no pooled library workspace apart from scratch,
  no packing of its own, and no step-invisible bytes. *decided*
- **The binding checks the slice against the query, before the call.** Our own
  kernels are safe from a short buffer for two reasons: their scratch travels
  with the call as a slice, and the binding checks the slice's length against
  a formula we wrote. A library call has the same two, with the vendor's size
  query as the formula. `fb_core_wwr::solver::syevd` does it today, and the new
  layer's binding is written the same way
  ([from scratch](#built-from-scratch-over-the-c-abi)): it repeats
  `syevd_lwork` at the same arguments and panics, at the caller's line
  (`#[track_caller]`), before anything is enqueued. A call `run` makes that
  `mem_budget` never declared then has no buffer to be handed, or a short one
  that the binding refuses. It is not optional: cuSOLVER does not check
  `lwork` itself ([below](#what-the-probe-showed)). The rule, "a solver
  binding that takes a work buffer checks it against its own size query",
  joins `rust/docs/rules.md` with a `doc_check` test over the solver bindings,
  when the layer lands in `rust/`. *decided*
- **A work buffer is never null.** A query that returns 0 is still handed a
  minimal non-null slice: on HIP, a null `work` drops hipSOLVER's rocBLAS
  handle into library-managed mode, which grows memory mid-run or aborts.
  *decided*
- **The queries need live handles**, so each device's `CardResources` exists
  before budget mode starts. The device-free `--budget-only` cannot ask them,
  and reports each call's work as zero.

### What the probe showed

*run* 2026-11-21, RTX 3080 and RX 9060 XT (gfx1200),
`experimental/library_handle_workspace` (its README has the tables). One
process per case, the card's free memory read around each step, through wwr's
raw names (on HIP, hipSOLVER and hipBLAS over rocSOLVER and rocBLAS), at n = 2048.

|  | cuSOLVER / cuBLAS | hipSOLVER / hipBLAS |
|---|---|---|
| Work buffer half the query | `syevd` **writes its full need (130 of 65 MiB) and returns `SUCCESS`**: an illegal access with a tight buffer, silent corruption otherwise | `ALLOC_FAILED`, nothing written |
| `work = nullptr` | `INVALID_VALUE` or an illegal access | **+34 MiB taken mid-call** (`potrf`, `gesvd`), or a process abort (`syevd`) |
| Exact work buffer, a second call | Nothing taken | Nothing taken |
| `dgemm`, any handle workspace | Nothing taken on a call | Nothing taken on a call; `SetWorkspace` frees the 32 MiB pool |
| A handle, when created | 10–14 MiB | 58–60 MiB, and 2–10 MiB more at the first call (likely code objects; not separated) |

So the only growth after creation is rocBLAS's managed mode, which a set
workspace and a non-null work buffer both rule out; and the only silent
failure is cuSOLVER's unchecked `lwork`, which the binding's check rules out.
Not covered: `getrf` (it takes no length, so nothing can check one) and
hipBLASLt. rocBLAS's `trsm` against a small set workspace, not covered here,
is measured below; the design no longer calls it
([above](#blas-and-solver-workspace)).

### What the workspace scaling probe showed

*run* 2026-11-24, RX 9060 XT (ROCm 10.0.0: rocBLAS 5.6.0, hipBLAS 3.6) and RTX
3080, through `wwr-sys`
([`experimental/rust/blas_workspace_scaling`](../blas_workspace_scaling/),
whose README has every reading). A call on a handle given a large
sentinel-filled workspace, read for how much it wrote; then on a 256-byte one,
read for what the library took by itself. Solver calls are handed a work
buffer of exactly their size query.

| Call, on HIP | What it needs beyond its operands |
|---|---|
| `trsm`, order k not a multiple of 128 | s·(m·n + 128·k) bytes for an element of s bytes: a copy of B. 132 MiB for 128 MiB of B, 4104 MiB for 4100; at 8200 MiB the 16 GiB card returns `ALLOC_FAILED` |
| `trsm`, k a multiple of 128 | s·(m·n/32 + 128·k) bytes: 8 MiB for 128 MiB of B, 260 MiB for 8 GiB. Exact on six shapes |
| `trsm`'s side, transpose and type | Change nothing but s: single, double and complex, left and right, plain and transposed all follow the two forms |
| `dot`, `nrm2`, `asum`; `iamax`, `dotc` | 8 bytes per 1024 elements: 8 MB per billion; 16 for `iamax` and complex `dotc`. With too small a workspace they return `ALLOC_FAILED`, where `trsm` grows silently |
| `trmm`, `trsv`, `gemm_strided_batched`, `syrk`, `gemv`, `geam`, `dgmm`, `axpy`, `scal`, `copy` | Nothing |
| `syevd`, `potrf`, `getrf`, `gesvd`, `heevd`, `heevdx` | Nothing that grows beyond the size query (3.26 n² doubles for `syevd`, with no step at an odd order; a constant 128 for `potrf`; about a copy of A for `getrf`; about n²/2 for `gesvd`; about 2.26 and 3.26 n² complex doubles for `heevd` and `heevdx`). 2 to 24 MiB once, at the first call |
| `potrs`, `getrs` | B's size plus about 42 MiB, taken by the library itself at any order: 170 and 554 MiB for 128 and 512 MiB of right-hand sides |

On CUDA (RTX 3080, the same sweep) nothing grows: `dtrsm` used none of the
handle's workspace up to 8 GiB of B, the other BLAS calls use at most 17 KB
whatever their size and succeed on a 256-byte workspace, every solver call
stays inside its size query, and cuSOLVER's `potrs` and `getrs` take nothing.
So the fixed 128 MiB covers every call measured except `trsm` on HIP, and on
HIP the reductions up to 17 billion elements (8.6 for `iamax` and `dotc`).
Every BLAS and solver call the C++ tree made when the probe ran is measured on
both backends: `trsm`, `potrs` and `getrs` on HIP were the only ones that take
memory by themselves, and none of the three is called any more
([above](#blas-and-solver-workspace)). So the fixed 128 MiB covers every
vendor call that is left. The formulas are read from rocBLAS 5.6.0 and may
change with a release.

## Caps, and what the limit covers

- **Caps come from the user's config**, per kind, as today. *decided*
- **The reserve measurement is dropped**: the per-worker reading, the 256 MiB
  floor, the growth tracking. Its job was to predict a failure that would
  otherwise come late; with everything allocated in budget mode a real
  out-of-memory surfaces at init, where the tree's own abort names it.
  *decided*
- **An explicit device cap is used as written.** The user owns the margin.
  With no cap given, the default is the device's free memory at start less a
  margin: one read per device, after its handles exist, which also sees
  another process already holding part of a shared card. *decided*
- **The device margin is the larger of 512 MiB and 5% of the card's total
  memory.** It is for what the card takes after budget mode
  ([below](#what-a-card-takes-after-budget-mode)): the kernels' stack, and
  modules loading at first launch. Both numbers are judgment and not
  measurement; they scale with the card because the stack does. No scan of the
  kernels and no test ties the constant to them. One that turns out too small
  costs a failed launch that says so
  ([failure](#failure-diagnostics-determinism)), and is raised then; a config
  that wants the memory back sets an explicit cap. *decided* 2026-11-24
- **The host cap has the same default.** With none given, it is the host's
  available memory read once in setup, less a margin, and every heap and
  pinned charge is held against it as it is made ([host
  heap](#what-is-budgeted-by-kind)). With B on the host heap this is the cap a
  large density-fitted run meets first. *decided* 2026-11-24
- **The host margin is the larger of 1 GiB and 5% of the available memory.**
  It tracks nothing. Host scratch is outside the budget by policy ([step
  scratch](#step-scratch)), and only what is on the `MainBus` is counted; the
  margin is room left unpromised, so that a B admitted up to the whole of the
  host cannot leave the uncounted allocations of a `run` with nothing, which
  would end as a kill from the OOM killer mid-run. Judgment again: the policy
  holds a step's host scratch within O(n\_ao²), megabytes at these sizes.
  *decided* 2026-11-24
- **Every cap is clamped to what the machine has at init.** A plan is held
  against the smaller of the cap and the reading made at start: each card's
  free memory once its handles exist, and the host's available memory in
  setup. A defaulted cap is that reading less its margin; an explicit one past
  the machine is clamped to the reading itself, with no margin taken, since
  the user owns it. The budget table prints the cap as written beside the
  value used. Without the clamp an explicit cap the machine cannot meet is not
  a refusal: on a card the tree aborts at whichever allocation fails first,
  with no table and a different step named from run to run, and on the host
  budget mode touches pages until the OOM killer ends the process with no
  message. The reading is a snapshot, so on a shared machine it can refuse a
  plan that would have fitted a moment later; the refusal prints what it read.
  *decided* 2026-11-24

> **The compliance statement**, to be written into the user docs whatever else
> changes *decided*: the memory limits cover what steps allocate through
> `mem_budget`, step scratch (which holds every library call's work buffer), and
> the fixed BLAS-handle workspace. They exclude the device context, the handles'
> own memory and loaded modules, the kernels' stack, and small host allocations
> such as reading the geometry and the basis data. A defaulted cap leaves a
> margin for those; an explicit one is the user's.

### What a card takes after budget mode

*decided* 2026-11-24: none of it is budgeted, measured or checked at run time.
The margin covers it, and the failure it leaves is loud and lands in one
place.

- **The kernels' stack is wholly outside the budget.** CUDA holds a kernel's
  per-thread stack for every thread the card can keep resident (104,448 on the
  RTX 3080), from that kernel's first launch on, by raising the device's stack
  limit and keeping it (`src/core/wwr/README.md`, "Kernel stacks outside the
  pool"). At the largest frame the Rust-only kernels have, 3,856 bytes in
  their low-stack form, that is about 280 MiB on the 3080; at the C++ tree's
  largest, 16,912 bytes, 1.6 GiB. It has no row in the budget table and no
  step declares it.
- **What that risks.** One failure: on CUDA, a plan that nearly fills the card
  fails at the first launch of a kernel whose frame is larger than any
  launched so far, with an allocation error. If the kernel is a late step's,
  the run loses the work before it. Nothing is silent and no result is wrong.
  It is isolable here in a way it was not under the old model, where the
  driver trimmed a pool's cached memory to make the stack and the failure
  surfaced at a later, unrelated allocation (#1259): there is no pool,
  everything budgeted is allocated before the first launch, and so the failure
  can only be the launch itself, in a named step. It needs free memory after
  budget mode below the stack the run will take, which a defaulted cap's
  margin rules out while the margin is the larger.
- **Nothing on HIP.** ROCm sizes a kernel's scratch per dispatch and holds no
  reservation sized by the frame: no growth past the first launch at 1 KiB,
  10,000 or 23,200 bytes of scratch a lane, and a 23,200-byte launch ran on a
  card full but for 256 MiB (gfx1200, ROCm 7.2.4, the same README). What a
  large launch takes while it runs was not sampled, and the measurement
  predates ROCm 10 ([to measure](#open-questions)).
- **Not taken: claiming it.** A row of the budget, counted under the cap and
  sized per step, needs each kernel's frame. That is known when the kernel is
  compiled and can be read from the built objects on both backends (`cuobjdump
  --dump-resource-usage`, as the C++ ctest `core_kernel_stack_bound` does; on
  HIP `.private_segment_fixed_size` in the code object's metadata note, by
  `llvm-readelf --notes` once the object is taken out of `.hip_fatbin`). So it
  would be a generated manifest of frames, a declaration in every step, and a
  device-attribute call wwr's C ABI does not have, for the card's thread
  count: all for one backend, to make an explicit cap honest about one of
  several things the compliance statement (above) already excludes, and to
  save at most the difference between the tree's largest frame and the run's.
- **Not taken: raising the stack limit at init.** One `wwr_device_set_limit`
  to a tree-wide constant, before the reading the default cap is made from,
  would take the stack before anything is budgeted and move the failure to
  init. It needs that constant kept at the largest frame the layer launches,
  by hand or by a scan of the compiled kernels, and the margin covers the case
  without either. It is the known fix, about ten lines, if a late failure ever
  costs a real run.
- **When to look again.** When a kernel with a frame past 8 KiB is ported (the
  Hessian kernels are the ones): the stack then outgrows the margin on a small
  card, and both of the above are worth their cost.
- **Modules and a library's first call are the margin's too.** Kernel modules
  load at first launch, and a library takes a few MiB at its first call (2–10
  MiB at a HIP solver handle's, up to 14 on cuSOLVER, by the
  [probes](#what-the-probe-showed)). No eager loading, no warm-up launch and
  no warm-up call in budget mode. The handles themselves (10–14 MiB each on
  CUDA, 58–60 on HIP) exist before the free-memory reading, so the default cap
  already sees them.

## Enforcement

| Rule | Check |
|---|---|
| Device, pinned and budgeted heap memory are allocated only in `mem_budget` | By visibility: the `wwr` allocation calls and their RAII wrappers are callable only from the crate that defines `Budget`, and `run` is never handed a `Budget`. A workspace-wide clippy ban on the raw `wwr-sys` allocators, with the one `allow` in that crate, backs it up |
| A budgeted host buffer never grows | By type: `Box<[T]>` |
| Nothing large is allocated in `run` on the plain heap | Two, below |
| Only the owning step writes its `Output` | Visibility of the mutating methods, within the steps' one crate |
| A step reads only the `Output`s it declared | By type: `run` receives its `View`, not the bus |
| Scratch dies with its step | By type: the lease borrows for the length of `run`, so it cannot be stored in an `Output` |
| A step relies on nothing its scratch held on entry | In a test mode the wrapper fills each lease with NaN before `run` |
| Two steps share scratch bytes only if the DAG orders them | By construction: the final node packs from the closure; one `unsafe`, in the arena, and a test that packs a DAG and checks every overlapping pair against reachability (the shape of the closure model's `unsafe_pairs`) |
| A library call's work buffer is at least its size query | By the binding: it repeats the query and checks the slice before the call ([above](#blas-and-solver-workspace)); a `doc_check` test holds every solver binding to it |
| No library handle manages its own memory | The final budget node binds the card's fixed workspace to its BLAS handle once, and `SetStream` is never called after the handle is made, so nothing drops it; a work buffer is never null |
| A card's handles are used by one step at a time, and by nothing in flight | By type: the stream, the handles and the workspace are inside the card's `Mutex`, reachable only through its guard, and the wrapper syncs the card's stream before the guard drops ([below](#one-handle-set-per-card-under-one-lock)). A `compile_fail` doctest keeps a handle past the guard |
| No vendor call takes memory by itself | By what is bound: the calls measured to (`trsm`, `potrs`, `getrs`) have no binding, the tree's recipes standing in for them ([above](#blas-and-solver-workspace)). For the rest, a sentinel test per binding in the nightly *proposed*. No check at run time ([below](#one-handle-set-per-card-under-one-lock)) |

The first rule covers device and pinned memory completely, since both go
through `wwr`. The third is the one neither types nor visibility can hold: a
`Vec` in `run` never touches `wwr`. Rust's analogues of C++ `#pragma GCC
poison` cover it from two sides *decided* (the threshold's value is open):

- **By name, at compile time.** clippy's `disallowed_methods`,
  `disallowed_types` and `disallowed_macros`, which `rust/clippy.toml` already
  uses for `wwr_mem_get_info`. Step crates ban `vec!`, `Vec::with_capacity`
  and the like, each exception an `allow` with its reason. Cheap and early,
  and it leaks: it bans by name, not size, and `collect`, `to_vec` and `clone`
  allocate too.
- **By size, at run time.** A `#[global_allocator]` wrapper over the system
  allocator. In execute mode, a single allocation above a threshold (1 MiB,
  say) is recorded, and the step wrapper checks the record after each `run`
  and fails naming the step. One atomic load and a compare per allocation. It
  must record and not panic (unwinding out of a global allocator is undefined
  behaviour), and it sees Rust's heap only, not a C library's `malloc`. Run in
  the test tiers, this is the real check.

## Sizes: batched only

*decided* 2026-11-23: **a step over an O(N³) tensor has a batched path and no
other.** The earlier plan was the full slab or a refusal first, with a batched
path later, per step. But the unbatched path is the batched one with a single
batch on a single card, so there is one path, and it is the one the gates run.

| What a step holds | Rule |
|---|---|
| Everything O(N²): the records, the n\_ao² matrices, the metric and its factor | Resident on the card, whole. It fits or the plan is refused: a dense eigensolve at n\_ao cannot be batched |
| The O(N³) tensors: (mn\|P), B, a derivative of B | Never resident whole. Built or contracted a batch at a time, the batch in step scratch; B itself on the host heap ([shared code](#shared-code-a-function-over-a-view-not-a-manager)) |
| The width of a batch | The widest whose bytes fit the step's own limit; no width fits: refuse, naming the step, its limit and its least bytes |
| The step's own limit | A fixed fraction of the kind's cap by default; the config overrides it, which is also how a test forces several batches on a small molecule |
| What is charged | What is used, never the limit: the limit only picks the width, the cap still admits. Limits need not sum to the cap |

- **Every GPU step is built batched and for every card, or not built.**
  Hartree-Fock, CASSCF, NEVPT2 and what follows: a feature is not added until
  it runs on a single node's `num_gpus` cards, one card being the case of one.
  That is tedious, and it is the commitment. Whether to go further, to several
  nodes or not built at all, is [to discuss](#open-questions). *decided*
  2026-11-23
- **Width stops mattering early.** In
  `experimental/rust/df_batched/single_gpu` the build's time is flat past
  about 20 trips, and width 1 costs 5.6× to 11.6×. So a default limit well
  under the cap loses little.
- **No "what is left".** A width is a function of the step's own limit, not of
  the room other steps leave, so `mem_budget` stays local and order-free, and
  there is no second, ordered phase.
- **A plan with slack may run narrower than it could.** The width is logged at
  `debug`, and at `warn` when it is 1; the user raises the limit.
- **Sizes fixed only by an earlier step's run** (the pruned pair-class counts,
  the orbital count after linear dependencies are dropped) are allocated at
  the bound the dimensions give, and less is used, as today. *decided*
- **Out of scope for now** *decided*: AVAS, the semi-numerical Hessian, and
  semi-numerical gradients.

## What is given up, and what disappears

**Given up: byte reuse between `Output`s.** Every step's `Output` stays
allocated for the run; down a chain, they stack. Reuse *within* a step is
untouched: a step owns its buffers and may use one for two purposes, so where
two things should share bytes, the tool is the step boundary ([the
example](#example-the-df-hf-setup-dag) has the case that matters, B solved in
place). Past that, the remedy when a calculation does not fit is a narrower
batch in the step concerned, not a shared pool. The exception is [step
scratch](#step-scratch): memory dead after a step's `run`, library work
buffers included, packed per device and kind so that steps the DAG orders
share bytes.

**Reuse and batching close different gaps.** There are three numbers per
calculation: the *sum* (every buffer alive at once), the *fold* (the peak with
bytes reused perfectly) and the cap. Reuse narrows sum toward fold and can
never go below it; batching lowers the fold itself. So where a chain's fold
fits and its sum does not, packing is the remedy; where the peak sits inside
one step (DF-HF's is inside `eri`: a batch, its records and work lists, and
its build intermediates, all live at once), only that step's own design helps:
its batch width, or a kernel that derives its work lists. Packing scratch
alone reaches part of the way from sum to fold; how far depends on how much of
each step is scratch.

**Disappears:**

- The planner, the table, per-buffer lifetimes, the assigner and its handoff
  edges. What lifetime there is, a step's run, comes from the DAG as written.
- **"The peak under stealing".** Bytes are shared only between steps the DAG
  orders, so no order of execution can break memory, and the
  bytes-against-concurrency search is moot.
- The adaptive pass and the question of iterating it.
- The reserve, its floor and its growth.
- Nested regions: a replay reuses its own buffers.
- The plan goldens' table totals; what is pinned instead is the budget table.

**What can still be refused** (revised 2026-11-23, with [batched
only](#sizes-batched-only)): a plan whose O(N²) data does not fit its cards,
whose B does not fit the host, or whose `Output`s, each fitting alone, do not
fit summed down a chain. The last is the sum's price. The ratio is worth
knowing: the C++ footprints already give, for each `examples/` config, the sum
of the steps' own peaks against the fold's peak. Split three ways, it says
which remedy applies: each step's `Output` bytes, its scratch bytes, and
whether the peak sits inside one step or across many. It is information, with
no bar attached.

## Work stealing: what the one thread was providing

A GPU worker today is one thread that never waits on the device. Four things
rest on that, and each needs an explicit owner once any rustflow worker may
run any step.

1. **The current device.** Launches and library calls run on the calling
   thread's current device. The step's wrapper makes its device current for
   the body (the C++ rule for a taskflow task: a `ScopedDeviceIndex` in the
   task), and so does every inner task that launches, since a stolen inner
   task runs on a thread whose current device is whatever it last used. Not
   run: the experiment bound one device per worker thread, once.
2. **Exclusive use of the library handles.** Enqueueing on a stream is
   thread-safe on both runtimes; the handle is the question. cuBLAS documents
   calls on a shared handle as thread-safe and warns only about reconfiguring
   it under other users; rocBLAS says not to share a handle between threads
   (verify both). The card's lock, held by one step at a time
   ([below](#one-handle-set-per-card-under-one-lock)), answers it either way.
3. **Order between unordered steps.** It was repeatable and now varies run to
   run. Correctness never rested on it, provided every ordering that matters
   is an edge. *run* with lanes kept and chained: results bitwise equal.
4. **Order on the card between dependent steps.** One lane's stream gave it
   for free. Here the step's own sync gives it: a finished node means finished
   device work.

**Device assignment is fixed before budget mode** *decided*: `mem_budget` has
to know which device's counter to charge, and an all-GPU step charges every
card. The assignment is made when the DAG is built from the config. Stealing
moves which thread runs a step, never where its memory is.

**The rule of assignment** *decided* 2026-11-23: the config gives `num_gpus`,
and a step runs on **no GPU, on gpu0, or on all of them**. Those are the only
three; no step runs on card k alone, or on some of the cards.

- **gpu0 is the privileged card.** A one-GPU step runs on gpu0. A card has to
  be privileged in any case, since dense diagonalization is single-GPU for
  now. The overlap, computed on the host as `experimental/rust/s_overlap_host`
  does, is uploaded to gpu0 and nowhere else.
- **An all-GPU step uses every card and splits its data across them.**
  Anything that builds J, or contracts like it, is one: the batches are shared
  out, each card works on its own, and the result is reduced onto gpu0. A step
  that calls an all-GPU function is an all-GPU step (the SCF, through the J
  and K build), though its eigensolve still runs on gpu0.
- **With one GPU, all is gpu0.** A single card is the degenerate case of the
  same code, not a second path.
- **A device buffer is on gpu0 or on every card.** Which is said by the
  `Budget` call that allocates it, not by the step's kind: the SCF is an
  all-GPU step whose orbitals are on gpu0; the metric and its factor are on
  every card (`experimental/rust/df_metric`'s (P|Q), uploaded to each). The
  one rule joining the two: an all-GPU step's reduced results land on gpu0.
- **So gpu0's cap is the one that binds**: every one-GPU step's memory and
  every reduced result is charged there, and the other cards hold their copies
  of the every-card buffers and their batches.
- **"Target" is dropped.** It named the card a product lives on, so that a
  one-GPU step could run on card k and deliver to gpu0, and a reader on
  another card could be refused. With three kinds of step and two places for a
  buffer, neither case exists: nothing is delivered, and no reader is on the
  wrong card. Revised 2026-11-23; the earlier form (2026-11-21) had "the
  producer delivers to the target, and a reader never pulls".

## Streams and device resources

*decided* A GPU step has streams of its own on each card it spans. Each card
has one stream, one set of library handles made on it and their workspace,
which a step borrows with the card's lock. Changed 2026-11-24: the device's
library stream, which any step could enqueue on under a per-call mutex, and
the fork and join events between it and a step's stream are gone. Changed
2026-11-25: the card has a stream again, but it is the lock holder's alone for
the holder's whole `run`, and the wrapper syncs it
([below](#one-handle-set-per-card-under-one-lock)).

|  | A step's own streams | The card's `CardResources` |
|---|---|---|
| What it is | Streams. Their fork and join events (the C++ `ForkStream`'s `fork_from`, `record_join`, `join_into`) serve only a step's own inner fan-out | A `Mutex` around the card's stream, the BLAS and solver handles made on it, any further library object a step declares, and the BLAS handle's fixed workspace |
| How many | One per GPU step per card it spans by default; more if declared | One per card |
| Made | By the wrapper, in budget mode | In setup, with the handles, before budget mode |
| Where it lives | The step's slot, beside its `Output`; the step never owns it | The `MainBus` |
| Reached through | The step's lane | The card's guard |
| What goes on it | The step's kernels and copies | Library calls; and a holder's kernels and copies, if it wants one order with no event |
| Who may use it | Its step | The step holding the card's lock, for its whole `run` |

A step that calls no library holds no guard and overlaps other steps on the
card.

```text
card's stream:  eri [ factor L | fit 0 | fit 1 | fit 2 ] sync, release; orthogonalizer [ syevd, gemms ]
```

- **`run` enqueues; a shared wrapper syncs.** The DAG builder wraps every GPU
  step's `run` in one generic function that makes the device current, takes
  the guard of every card the step declared the library on, calls `run`, then
  synchronizes every stream the step has and each held card's stream, and only
  then drops the guards. A trait default method would do it too, but a step
  could override that; a step cannot forget or skip the wrapper.
- **The same holds one level down.** A step's inner GPU tasks, and a shared
  function's fan-out across cards, are finished before `run` returns: whoever
  fans out, joins. The wrapper's sync is then the last word and not the only
  one. *decided*
- **A finished node is finished device work**, on every stream the step
  touched. So the DAG's edges order everything between steps: no event between
  steps, no chain edge, and any worker can run any step, because a stream is
  the step's or the card's and not the thread's. Independent GPU steps that
  call no library overlap on the card for free.
- **A step's library calls go on the card's stream.** A simple library user,
  the orthogonalizer or the SCF, puts everything there, kernels and library
  calls alike, in the order it enqueues, with no event. A step that keeps
  kernels on streams of its own orders them against its library calls with
  events, inside the step ([a step with streams of its
  own](#a-step-with-streams-of-its-own-and-library-calls)).
- **This replaces "a task never syncs the host".** That rule exists because a
  GPU worker was one thread that had to keep enqueueing. At step granularity,
  over a pool, a blocked worker is fine. The sync should block so the thread
  sleeps and does not spin; which scheduling flag the streams are made with is
  to check.
- **A waiting worker is not stealing.** The executor's width has to cover the
  GPU steps that can be waiting at once, on a sync or on a card's lock, plus
  the CPU work: for these pipelines, a handful of threads.
- **Two thread pools are allowed.** The host integrals
  (`experimental/rust/df_metric`, `s_overlap_host`) run on rayon's threads,
  called from a step on a rustflow worker. The pools are kept separate, each
  with its thread count from the config; a worker blocked in a rayon call is a
  waiting worker, as above. *decided* 2026-11-23

### One handle set per card, under one lock

*decided* 2026-11-24, and revised 2026-11-25: the card's stream, the handles
and the workspace are one thing under the card's lock, and nothing is rebound.
It replaces the device's one library stream, its per-call mutex and the fork
and join events between it and a step's stream, and the ring of handles
proposed on 2026-11-24. It closes the [open question](#open-questions) on the
step-end sync and the library stream.

**The hole it closes, in DF-RHF.** With the library stream, `eri` and the
`orthogonalizer` are unordered, so their scratch regions on gpu0 are disjoint;
the SCF follows both, and the packer may lay its region over the
orthogonalizer's.

| Step | What it enqueues | On which stream | Scratch on gpu0 |
|---|---|---|---|
| `orthogonalizer` | `syevd` on S, then the `gemm`s of X | The library stream only, as in the prototype | `syevd`'s work at n\_ao |
| `eri` | Class kernels, then `potrf` and `trsm` on the metric | Its own, then the library stream | The build's intermediates |
| `scf` | The J and K batches, then `syevd` on F | Its own, then the library stream | The J and K batch, the DIIS ring and `syevd`'s work, over the orthogonalizer's bytes |

- **The schedule that breaks.** `eri` enqueues its library calls, joins, and
  its wrapper syncs. The orthogonalizer enqueues `syevd`(S) behind them and
  returns; its wrapper syncs the step's own stream, which is empty, so the
  node finishes at once. Both of the SCF's predecessors are finished, it
  starts, and its first J and K batch, on its own stream, writes the bytes
  `syevd`(S) is still using as work. X is wrong, with no error, and only on
  schedules where the orthogonalizer enqueues after `eri`.
- **Why the prototype never showed it.** Every step put everything on the
  library stream, and that one stream ordered all of it ([the
  prototype](#what-the-first-prototype-showed-v002)). The hazard needs a step that uses
  its own stream, which the batched J and K build does.
- **The same gap is inside a step.** In an SCF trip, `syevd` has to wait for
  the kernels that built F, and the density's kernels for `syevd`'s C. With
  two streams that takes the fork as well as the join.

**The design.** *decided* 2026-11-25. It revises the form of 2026-11-24,
which kept no stream in `CardResources` and rebound the handles to the
holder's stream at every acquisition of the lock (`SetStream`, then
`SetWorkspace`).

- **The card owns a stream.** Each card's `CardResources` holds one stream, a
  BLAS handle and a dense-solver handle made on that stream, and the fixed
  128 MiB BLAS workspace. The handles keep that stream for the life of the
  process. Nothing is rebound: `SetStream` is never called after creation, and
  the workspace is set once.
- **The lock guards all of it.** `CardResources` is a `Mutex` around the
  stream, the handles and the workspace, not a `Mutex<()>` beside them. They
  are reachable only through the guard. A further library object a step needs
  on that card (a sparse or FFT handle, a collectives communicator) joins the
  same set under the same lock, so there is never an order between locks to
  get wrong.
- **A step owns its streams.** A GPU step has streams of its own on each card
  it spans: one by default, more if it declares them. The wrapper makes them
  in budget mode and keeps them in the step's slot.
- **A library-using step also borrows the card's.** It declares the library
  per card in `mem_budget`, beside its scratch. For its `run` it holds that
  card's guard, and its library calls go on the card's stream.
- **The wrapper takes the guards, and syncs before it drops them.** Before
  `run` it takes the guard of every card the step declared, in card order, so
  two multi-card steps cannot deadlock, and that order also keeps collectives
  issued in the same order on every rank. After `run` it synchronizes every
  stream the step has on each card, and each held card's stream, and only then
  drops the guards. A step re-run by a condition loop takes the lock each
  time.
- **Safety is the scratch argument, unchanged.** A finished node is finished
  device work on every stream the step touched, the card's included, so the
  closure still orders every byte two steps share. The previous holder of a
  card's handles has synced the card's stream before releasing them, so no
  handle is ever shared with work in flight, and no two steps ever use the
  fixed workspace at once.
- **The BLAS workspace is fixed, 128 MiB per card**, a row of the budget table
  ([workspace](#blas-and-solver-workspace)). Solver work buffers stay per
  call, sized by their query, in step scratch: `syevd`'s grows as n² (130 MiB
  at n = 2048 on cuSOLVER), so no fixed size covers them. rocBLAS's `trsm`,
  the one BLAS call in the same position, is not called: the triangular solve
  is the tree's recipe, which needs only the fixed workspace
  ([workspace](#blas-and-solver-workspace)).
- **A step's inner tasks share its handle set.** An inner task for a card
  takes that card's guard from the step, mutably, so two inner tasks cannot
  call the library at once on one card (rocBLAS says not to share a handle
  between threads; verify). A step that wants concurrent library calls inside
  itself declares further handles of its own, like scratch.

**Why this is not the library stream dropped on 2026-11-24.** Two things made
the hole above, and neither is here:

|  | The library stream, as dropped | The card's stream |
|---|---|---|
| Who may enqueue on it | Any step, under a mutex held per call | The lock's holder alone, for its whole `run` |
| What the wrapper synchronizes | The step's own stream | The step's own streams and the card's, before the lock is released |

**Why not the rebind.** v0.0.3 set out to build the 2026-11-24 form and could
not, on `rust/` as merged (its README, "What the prototype says to change",
item 1):

- `fb_core_wwr::blas::BlasHandle` keeps the `Arc<Stream>` it was made on and
  has no rebind.
- `fb_recipes_sys::trsm` launches its diagonal kernels on that stream
  (`blas.owner()`) and its gemms through the handle. A handle rebound through
  the raw call would run one solve on two streams with nothing ordering them.
- `BlasHandle::bind_workspace` takes ownership of the workspace, where the
  design's workspace is the card's and outlives every binding.

So v0.0.3 made a BLAS handle per library-using step per card: the form
rejected below for its cost on HIP. Those three are properties of `rust/`'s
wrappers and not of the libraries, and a layer written [from
scratch](#built-from-scratch-over-the-c-abi) has none of them: a rebind could
be built there. It is still not taken, for the one reason that is the
libraries' own. The rebind is safe by measurement, on one release of each:
cuBLAS drops the workspace at `SetStream`, hipBLAS keeps it (below), and what
the solver handles do was not measured.

**What it buys.**

- **Safe by construction.** With no rebind there is nothing to measure about
  `SetStream`, and no test of it to keep across toolkit bumps.
- **Exclusive use by type.** The handles are behind the guard, so "no handle
  is shared with work in flight" is the borrow checker's.
- **A recipe cannot be handed the wrong stream.** The library bindings and the
  recipes take the card's guard: the stream and the handles together, in place
  of a handle and a stream passed apart. `trsm`'s diagonal kernels and its
  gemms land on one stream because the guard has only one, and a step with
  streams of its own has no way to pass one of them.
- **The default cap sees the handles.** They are made in setup, so "one read
  per device, after its handles exist" holds as
  [caps](#caps-and-what-the-limit-covers) words it, and their memory does not
  grow with the number of library-using steps.

**What it gives up: overlap between library-using steps the DAG leaves
unordered on one card.** The lock is held through a step's non-library phases
too. In DF-RHF's setup that is `eri` against the `orthogonalizer`: the
orthogonalizer's one `syevd` at n\_ao, a few milliseconds, waits for `eri`. So
library-using steps are serial on a card, whatever the DAG says; steps that
call no library still overlap. The packer takes nothing from this: two lock
holders the DAG leaves unordered keep disjoint scratch ([step
scratch](#step-scratch)). Heavy steps fill the card on their own, and two of
them on one card time-slice rather than overlap: this tree's test batches
measured a CASSCF case at 2.3 s alone and 158 s beside the exchange build.

Given up with it since 2026-11-25: narrowing the lock to a step's library
phases later. A holder's library calls are on the card's stream, so two
library-using steps cannot interleave on a card without a second handle set.
The lock was already held through a step's whole `run`, and the ring of handle
sets already declined, so nothing decided is lost.

**Not taken.**

- **A rebind**, as above: a handle moved to the holder's stream at every
  acquisition, with the recipes reading the stream from the current binding.
  It keeps every step's library calls on the step's own stream, at the price
  of resting on the measured `SetStream` behaviour.
- **Every recipe takes its stream explicitly.** It does not remove the rebind;
  it only moves the stream out of the handle.
- **Rebinding inside a step, per call, to each of its streams.** Two library
  calls in flight on two streams would share the one fixed workspace.
- **A ring of handle sets**, assigned by the closure so that two steps share a
  set only if one reaches the other. It keeps that overlap, at a handle set
  per unit of the DAG's width (about 20 MiB on CUDA and 52 on HIP, measured
  below) and an assignment in the final budget node, for overlap the paragraph
  above says is worth little.
- **A handle set per step.** The same with no sharing, at a set per
  library-using step: ten such steps cost 520 MiB on HIP against 52. It is
  what v0.0.3 built, for want of a rebind.
- **The lock per call.** It waits less, but two unordered steps would then
  have calls in flight on one handle at once, and whether cuSOLVER or rocBLAS
  keep internal device state that those calls would race on is unverified.
- **The join made by the access**: the old library stream kept, a step
  reaching the handles only through a call that forks from its stream and
  joins back. It closes the hole, but every simple library user then pays a
  fork and a join per call, where the lock's holder pays none.

**No free-memory check at run time** *decided* 2026-11-24. It was proposed
here: in execute mode nothing allocates, so a card's free memory should hold
its value from the end of budget mode, and one memory-info read by the wrapper
after each step's sync would catch what a vendor library takes by itself,
which the compile-time rule ([Enforcement](#enforcement)) cannot see. Dropped,
in every form:

- **What it was for is ruled out by construction.** Every case it was written
  for was a vendor call taking memory by itself: rocBLAS's `trsm`, and `potrs`
  and `getrs` on HIP, which are no longer called
  ([workspace](#blas-and-solver-workspace)); a workspace too small on rocBLAS,
  which the fixed 128 MiB, bound once, covers; a null work
  buffer, which no binding passes. What is left is a toolkit bump changing
  what a call takes, or a call nobody measured.
- **The reading moves for reasons that are not the tree's.** The kernels'
  stack at a first launch on CUDA, modules loading, a library's first call
  ([what a card takes after budget
  mode](#what-a-card-takes-after-budget-mode)), and any other process on the
  card. As an error it needs an allowance for each, or a warm-up launch per
  kernel family and a warm-up call per library in budget mode; as a warning it
  is mostly noise.
- **It is weakest where it would be enforced.** The gates run small molecules,
  where growth that scales with a call's operands is the size of the
  allocator's noise; it would show on production runs, where it could only
  warn.
- **A replay test is not taken either**: the card's free memory equal across
  replays 2 to n, in the gates. It needs no allowance, since the stack and the
  modules are loaded by then, but anything else on the card moves it, so it is
  a flaky gate or one with a tolerance; and what it would catch, a vendor call
  that leaks per call, is the sentinel test's
  ([workspace](#blas-and-solver-workspace), proposed), at the binding.
- **What is accepted.** A vendor call that starts taking memory after a
  toolkit bump, on a shape no test runs, surfaces as an out-of-memory failure
  on a large run. It is loud, and it is at a library call in a named step.

**What the measurements showed** *run* 2026-11-24, RTX 3080 and RX 9060 XT,
through `wwr-sys` (`experimental/rust/handle_ring`, whose README has every
reading):

1. **A further handle is not nearly free.** Each BLAS handle after the first
   takes 8 MiB on cuBLAS and 58 on hipBLAS, each solver handle 12 on cuSOLVER
   and 58 on hipSOLVER; on HIP each gives back rocBLAS's 32 MiB pool once it
   has a caller's workspace. So a handle set costs about 20 MiB on CUDA and 52
   on HIP, and the first-call growth (10 on HIP, 0 to 14 on cuSOLVER) is once
   per process. A handle per step costs the pipeline's length times that; one
   set per card is the floor.
2. **The move is safe in the proposal's order.** The design no longer moves a
   handle (2026-11-25); these readings stand as what a rebind would rest on.
   On cuBLAS a call made after
   `SetStream` then `SetWorkspace` used the workspace it was given; with the
   two reversed, `SetStream` silently put the handle back on cuBLAS's own
   workspace, outside the budget, with no error (read from the result of a
   split-K `gemm` whose algorithm follows the workspace length). On hipBLAS
   `SetStream` keeps the workspace, so the order does not matter there (read
   from a `dtrsm` at the DF fit's shape, which writes 8 MiB of the workspace
   in every order). Nothing is taken by a move on either backend, and an idle
   switch costs 0.02 µs on CUDA and 0.15 on HIP; the first `SetWorkspace` of a
   hipBLAS handle costs about 150 µs, the pool's free, once per handle at
   budget time. The two calls belong in one wrapper function that nothing else
   reorders.
3. **On HIP a workspace too small grows.** Given 256 B, that `dtrsm` took 8
   MiB of its own at the call, outside the budget, with no error; cuBLAS only
   changes algorithm. So on HIP the card's workspace must cover the largest
   BLAS call any step makes: hence 128 MiB. Measured since ([the workspace
   scaling probe](#what-the-workspace-scaling-probe-showed)): it covers every
   call but `trsm`, whose need grows with its right-hand sides, and which is
   no longer called ([workspace](#blas-and-solver-workspace)).

### A step with streams of its own and library calls

*decided* 2026-11-25. `experimental/rust/df_batched/multi_gpu_spherical` is
the case: each card builds its share of B over a ring of two batches in
flight, and every trip has kernels, a library solve and a download. v0.0.4's
`eri` is built that way, so the rule above is run in the form that needs
events.

Per card, with a ring of depth 2:

- **The ring's slots are the step's own streams**, one per slot. A slot's
  stream carries its batch's table upload, its three-centre kernels and its
  download.
- **The fit goes on the card's stream.** It waits on the event the slot
  recorded after its kernels, and the slot's download waits on the event
  recorded after the fit.
- **The host waits on a slot's last event**, not on a whole stream, before it
  scatters that batch into B and writes the slot's pinned bytes again.
- **The metric's factor** is on the card's stream before the loop, and
  overlaps the first batch's kernels.

The events are a step's own inner fan-out. Whoever fans out, joins: every
event is waited on before `run` returns, and the wrapper's sync is the last
word.

- **What serializes.** The two slots' fits run one after the other on the one
  card's stream. The ring overlaps a batch's copies and kernels with the
  previous batch's fit; it does not overlap two fits. Expected to cost little,
  since two solves on one card time-slice. Not measured.
- **Further handles are not built.** A step that needs concurrent library
  calls inside itself declares handles of its own, each with a workspace it
  budgets. No step here needs it.
- **The ring is part of the layout.** A slot's device and pinned bytes are
  tenants of `eri`'s scratch, so the layout holds two of each, and the width
  is the widest whose ring fits the step's limit, as that experiment sizes it.
- **A simple library user needs none of this.** The orthogonalizer and the SCF
  put everything on the card's stream, in the order they enqueue, with no
  event.

### Where the handles and the workspace are made

- **Setup** makes each card's stream and its two handles. *decided*
  2026-11-25
- **Setup then reads the card's free memory** for the default cap
  ([caps](#caps-and-what-the-limit-covers)). *carried*
- **The final budget node allocates the 128 MiB workspace** and binds it to
  the card's BLAS handle, once. It stays a row of the budget table, charged
  against the cap. Binding it in setup would put it before the reading, and
  the row would then count it twice. *carried*
- **A device-free `--budget-only`** makes no handles and reports each solver
  call's work buffer as zero. *carried*

### Resources

The C++ `src/core/resources` comes back as a crate, without a pool.

|  | C++ | Rust |
|---|---|---|
| `NodeResources` | Executor, the `CardResources`, the host queries | The rustflow `Executor`, the device list, the host queries, one `CardResources` per device |
| `CardResources` | One per device: stream, handles, pool, workspaces, pinned memory | One per device, a `Mutex` around: the card's stream, the BLAS and solver handles made on it, and the fixed 128 MiB BLAS workspace the final budget node allocates. No pool; memory is the steps', and solver work is their scratch |
| Lifetime | The run | The process, behind an `Arc` |

- **The `'static` bound is answered.** rustflow's task bodies are `FnMut() +
  Send + 'static`
  ([DESIGN.md](https://github.com/whatDiracIsCooking/rustflow/blob/main/DESIGN.md)).
  A node's body captures an `Arc<MainBus>`, which owns everything a body
  reaches, so the lifetime shim the experiment needed goes away.
- **The thread count is the executor's**, fixed at init from the config: a
  property of the process, no longer of a plan.

## The step DAG on rustflow

One graph is written, over steps, and built twice from the same description:
once with each node's body calling `mem_budget`, once calling `run`. Names are
rustflow's ([README](https://github.com/whatDiracIsCooking/rustflow),
[DESIGN.md](https://github.com/whatDiracIsCooking/rustflow/blob/main/DESIGN.md)).
rustflow is pinned at `f184bcab` (its head on 2026-11-24), in
`rust/Cargo.toml`'s `[workspace.dependencies]`; see [the
dependency](#open-questions).

| In the calculation | In the `Taskflow` |
|---|---|
| A step | One node of the outer graph: a static task (`emplace`) whose body is the wrapper around `mem_budget` or `run`, or a subflow task when `run` builds an inner graph |
| A step reads another's `Output` | A dependency (`precede`), made by the builder from the step's declared `View`, and nothing else: the producer synced before its node finished |
| A GPU step in execute mode | The wrapper: device current, the guards of the cards it declared, `run`, sync the step's streams and each held card's, drop the guards |
| The end of budget mode | One final task every step precedes: the closure, scratch packing and the handle workspace, host check, accept or refuse |
| A step's inner work (integral batches, the SCF's iteration) | The step's own, inside its node: a subflow (`emplace_subflow`), condition loops built in it (`Subflow` implements `FlowBuilder`, so `emplace_condition` is there; read in source at `67a07b9` and `f184bcab`), or a `Pipeline` for a streamed pass. A subflow joins before its node finishes, so the wrapper's sync stays the last word |
| A replayed chain: geometry optimization, AIMD | The chain as a module task (`composed_of`) inside an outer condition loop; the scratch closure is over the loop's body, computed once. Where the per-replay reset goes is [to discuss](#steps-and-the-mainbus) |

- **A loop needs no entry node.** A condition picks one successor by index,
  but a multi-condition (`emplace_multi_condition`, a body returning
  `Vec<i32>`) names as many as it likes, so the back edge names each source of
  a body of several, and the SCF's empty entry node goes. Each trip runs the
  sources once and their join after them. An iteration limit is the
  condition's own check. *run* (`experimental/rust/v0.0.2`'s
  `tests/rustflow_pin.rs`, at `f184bcab`)
- **A condition loop inside a subflow runs**, and joins before its node
  finishes: the node after the step sees the loop done. rustflow's own tests
  still run only a subflow inside an outer loop (`tests/subflows.rs`); the
  case is the prototype's `tests/rustflow_pin.rs`, which passes at `f184bcab`
  and at the prototype's old pin `b77e5474` alike. *run*
- **A module cannot be run by two tasks at once** (DESIGN.md: upstream's rule
  is that a target must not be run by two threads concurrently). So concurrent
  replays would be copies, each with buffers of its own. None is in scope: the
  semi-numerical Hessian is deferred.
- **A pipeline's back-pressure is host-side.** A line is free when its last
  pipe returns, that is when the batch is enqueued, not when the card has run
  it. A streamed pass still needs device events. *run*
- **Budget mode needs no order**, but it runs the same edges anyway; the cost
  is nothing and the graph is one.
- **A serial mode**, if ever wanted for debugging, is a one-worker executor:
  `Executor::new(1)` (zero workers panics). Both cases of
  `tests/rustflow_pin.rs` run on one worker and on four. *run*

## Example: the DF-HF setup DAG

Everything that must exist before the first SCF iteration of a density-fitted
Hartree-Fock, and what each piece depends on. Nothing here is iterative: every
node runs once. Edges point down, from what is needed to what needs it. Each
node carries the phase it belongs to and what its `mem_budget` would charge;
those are illustrative, to show the rules at work, and not sizes the tree
commits to.

```text
FIGURE-GRAPH (the graph: pan, zoom, and click a node to trace what it needs)
```

### Nodes

```text
FIGURE-NODES (the node table: one row per node, filled by the page's script)
```

### How it maps onto this design

> **The sizes below predate 2026-11-23.** They take B as one 7.80 GB slab on
> device 0, built whole, with the whole `cells` work list beside it. As now
> decided ([batched only](#sizes-batched-only), [shared
> code](#shared-code-a-function-over-a-view-not-a-manager)), B's slab is on the
> host heap, the cards hold one batch of pairs each as scratch, and the work
> lists are a batch's. The graph, the phases and the O(N²) charges stand; the
> device total, the two budget figures and the node table's gly10 column for the
> three-centre step are to be redone at a batch width.

- **The top three levels are the setup phase.** The reads, the electron count,
  the nuclear repulsion and the two basis sets are small host work,
  unbudgeted, and they are what fills n\_atom, n\_ao, n\_aux and n\_occ.
  Budget mode starts below them.
- **Everything below is a step**, each with a `mem_budget` that is a function
  of those dimensions alone.
- **The integrals level is where step-owned streams pay.** S, T, V, (μν|P) and
  (P|Q) have no edges between them. Each enqueues on a stream of its own, so
  those that call no library overlap on the card with no lane named and no
  event between them.
- **Three nodes use the library**: X (an eigensolve), H (a sum) and B (a
  triangular solve). Each holds gpu0's lock for its `run` ([one handle set per
  card](#one-handle-set-per-card-under-one-lock)). They are on separate
  chains, so the lock may be contended here: X, or H, waits for B's step;
  nothing is wrong when it does.
- **The scratch slab is packed**: X's eigensolve work, and B's build
  intermediates with its records and work lists, are unordered, so their
  regions are disjoint, though X waits on gpu0's lock while the three-centre
  step holds it; the SCF, which follows both, may reuse either. The slab is
  allocated once, by the final budget node, beside the fixed 128 MiB
  BLAS-handle workspace.
- **n\_pair is budgeted at its bound.** The screening map decides how many
  pairs (μν|P) and B hold, it is a GPU computation and not a parse, and it
  changes from one geometry to the next, while the unscreened count does not.
  So both are budgeted at the unscreened count, which also keeps a replay
  inside its buffers. That wastes memory on a large sparse molecule. Kept on
  2026-11-24, with B on the host heap: batching bounds what a card holds, not
  B's slab, so the waste is host memory and there is no remedy for it short of
  a tighter bound, which is not taken; a replay can then never outgrow its
  buffers. The bound is the count in the tree's own pair layout and not the
  bare n\_ao(n\_ao+1)/2: gly10 below holds 270,497 pair columns where that
  formula gives 269,745. *decided*
- **The step boundary around B decides the peak.** As two steps, (μν|P) and B
  are two slabs of n\_pair × n\_aux, both alive for the run. The solve is in
  place, so as one step ("three-centre integrals, then the solve") the
  `Output` is one slab. At the experiment's gly10 in cc-pVDZ / cc-pVDZ-JKFIT
  (n\_ao 734, n\_aux 3606, 270,497 pair columns) one slab is 7.8 GB, so the
  choice is 7.8 or 15.6 GB. The metric and its factor are 104 MB each, and
  every n\_ao² matrix 4.3 MB. *decided* 2026-11-21: one step. (μν|P) is built
  and solved within it; built into B's own slab, the in-place solve needs no
  second one, and the build's other intermediates go in [step
  scratch](#step-scratch). The batched and multi-GPU paths will change these
  sizes.
- **The whole example's budget**, on that molecule, is then 9.86 GB of device
  memory: the one slab (7.80), the records and work lists (1.81, nearly all of
  it the three-centre step's `cells`), the metric and its factor (0.21), and
  0.04 for everything else, the eigensolve's scratch included; plus the 128
  MiB handle workspace, and the records and work lists' 1.81 GB pinned. An
  earlier draft said "the slab plus about 0.25 GB": it left the work lists
  out. Packing does not lower this: the peak is inside the three-centre step,
  where its slab, its work lists and its build intermediates are all live at
  once. Packing frees the 1.81 GB, on each side, only for the steps after it.
  The work lists are the second-largest charge, and `cells` is a dense cross
  product a kernel could derive from two short lists ([to
  measure](#open-questions)). The metric and its factor were named here as the
  other case, `Output`s only B reads, which packing `Output`s would cover; the
  derivative steps read them too ([shared
  code](#shared-code-a-function-over-a-view-not-a-manager)), so they stay for
  the whole calculation by right.

```text
FIGURE-12 (budgetfig)
```

```text
FIGURE-13 (fig)
```

### What the graph says

- **Three independent roots.** The geometry, the orbital basis and the
  auxiliary basis are read with no ordering between them.
- **Each basis has its own side.** The orbital set feeds its transform and the
  pair screening map; the auxiliary set feeds its own transform. The two sides
  meet only at (μν|P).
- **The widest level is the integrals.** S, T, V, (μν|P) and (P|Q) depend on
  nothing in their own level, so all five can run at once. That level is also
  where nearly all the cost and memory are: (μν|P) dominates both.
- **Two chains hang off it.** S → X on the orbital side, and (P|Q) → factor →
  B on the fitting side. X and the factorization are independent of each other
  and of T and V.
- **The initial guess is off to one side.** It is a table read and an assembly
  that needs only the orbital basis set: no integral, no S, no H, no X. It can
  run beside everything after the basis set is built.
- **B is the only node that joins the two bases' heavy products.** It is the
  last to finish, and (μν|P) and the factor are not needed once it has.
- **The geometry reaches the SCF five ways:** through the shell centres,
  through which elements get an atomic density (by way of the basis set),
  through V's point charges, through the nuclear repulsion, and through the
  electron count.

### Choices the graph leaves open

- **How (P|Q) is "orthogonalized".** A Cholesky factor J = LLᵀ with a
  triangular solve, or a symmetric J<sup>−1/2</sup> from an
  eigendecomposition, which can also drop near-dependent fitting functions.
  Either gives a B with (μν|λσ) ≈ Σ<sub>P</sub> B<sub>μν</sub><sup>P</sup>
  B<sub>λσ</sub><sup>P</sup>.
- **How X is built.** Symmetric S<sup>−1/2</sup>, or canonical
  orthogonalization that drops small eigenvalues of S and so may give fewer
  orbitals than basis functions.
- **Where the spherical transform is applied.** Inside each integral batch, as
  drawn, or once on a finished Cartesian matrix. Applied inside, no
  Cartesian-sized (μν|P) is ever stored.
- **Which initial guess.** Drawn as a superposition of atomic densities read
  from a table computed ahead of time, one entry per element and basis.
  Computing them in the run instead adds a small SCF per element ahead of this
  node. A core-Hamiltonian guess would instead depend on H, X and the electron
  count.
- **What the screening map bounds.** Drawn as the Schwarz bound over orbital
  shell pairs. An overlap or distance criterion gives the same node with a
  cheaper body; a bound that also uses √(P|P) would add an edge from the
  auxiliary set.
- **Whether B is stored at all.** It can be held whole, held in μν-packed
  form, or rebuilt in slices each iteration from (μν|P) and the factor; the
  graph up to the factor is the same.

### Left out on purpose

- **Symmetry, frozen orbitals, effective core potentials.** None changes the
  shape; an ECP adds one more term beside V.

## What the experiment showed

2026-11-19, RTX 3080 (CUDA) only, rustflow at `b77e5474`. The code is a patch
against main at `a69b5c49`, kept with its logs, plans, dumps and findings
outside the tree (`FINDINGS.md`, `rustflow-df.patch`). It ran on the **old**
memory model (the planner and the arena), so it is evidence about the executor
and not about the budget.

**What ran.** The streamed DF J/K pass on gly10 in cc-pVDZ / cc-pVDZ-JKFIT:
n\_ao 734, n\_aux 3606, n\_occ 237, B 7.8 GB on the heap, width 64, 57 trips.
Four ways, in one process, the card held, four runs each:

|  | Pass wall (s) | J, K |
|---|---|---|
| The plan on `fb_core_thread_pool::run` | 4.27 to 4.37 | reference |
| The plan on rustflow, GPU instances chained per worker | 4.24 to 4.51 | bitwise equal |
| The plan on rustflow, chained per lane | 4.24 to 4.36 | bitwise equal |
| No planner: hand allocation, the loop a rustflow `Pipeline` | 4.22 to 4.34 | bitwise equal |

On water the same four match PySCF's DF J and K to 1e-14 of their maximum.

**What that is evidence of.**

- rustflow runs this tree's real GPU work correctly, and its cost is below
  what a 4.3 s GPU-bound pass resolves (about 1%). Dispatch cost on small or
  many-task graphs was not measured.
- **The fourth row is this design's shape**: no planner, buffers allocated by
  hand up front, the loop a `Pipeline`. It was as fast as the planned paths
  and bitwise equal.
- Chain edges give a lane one user at a time with nothing blocking at run time
  and no event added. This design no longer needs them: a step has a stream of
  its own.

**What it did not run.** One card only; every worker made device 0 current
once in `WorkerInterface`'s prologue, with no per-node device selection. No
condition loop, so no SCF. No module, subflow or semaphore, no collectives, no
forks, no HIP. Failure was a flag per instance that skips descendants, built
and never exercised. And nothing of budget mode, the `MainBus` or step
scratch.

**What it found.**

- **A borrowed plan needs a lifetime shim** under `'static` bodies; shared
  state behind an `Arc` removes it, which the hand-written variant confirmed.
- **The executor cannot be swapped from outside the crate**:
  `fb_core_thread_pool`'s graph and run internals are private. With this
  design that crate is replaced, not adapted.
- **Kernel bindings that take arena rows** (`PackedTables::pack`, `gather`)
  cannot be called without the planner; the BLAS wrappers, which take a device
  slice, can. Those bindings change with this design.
- **rustflow's prelude exports `Graph`, `Node`, `Work` and `Task`**, which
  collide with this tree's names; import by name.

## What the first prototype showed (v0.0.2)

2026-11-20, RTX 3080 (CUDA) only. `experimental/rust/v0.0.2`: this design with
the new memory model, DF-RHF on one card, every step in its crudest form
(dense B, no batching, no enforcement lints). Its README has the tests and
figures; this is what bears on the text above. Not run: more than one card,
HIP, contention, inner tasks, batched paths, the enforcement checks, the sum
against the fold.

**Held** *run*: water, benzene and ethylene within 1.1e-13 Eh of the old Rust
path; a replay allocates nothing; twenty runs at eight workers agree to
2.3e-13; an over-cap plan is refused with the same whole table every time; the
declaration is the edge with no macro (an undeclared read is a compile error);
the SCF is a condition loop of the graph, entry node needed; a failed step
fails its readers only; `'static` bodies need no shim. Execute mode costs
nothing that shows against the old path at these sizes.

| The text above says | The prototype found | What follows |
|---|---|---|
| [A library call against its declaration](#blas-and-solver-workspace) | The wrapper's repeated size query refuses an undeclared `syevd`, naming the step, before anything is enqueued *run* | Taken, in the binding, which already does it; the work buffer moves to step scratch. On HIP hipSOLVER refuses a short buffer itself ([the probe](#what-the-probe-showed)) *decided* |
| [Replay reset](#steps-and-the-mainbus): three candidates *to discuss* | Reset mode in the step sufficed, with no new node. The loop's host state then sits in the `Output`, and what carries over is implicit in the step's code | Reset mode, with what carries over stated per step. Tagged for later, 2026-11-21 |
| [One step around B](#example-the-df-hf-setup-dag) *decided* | Works as `eri`. But 68% of its device bytes on benzene (237 of 347 MiB) are build scratch, dead once B exists, and an `Output` cannot free them | Not a second lifetime inside a step: re-taking it on a replay allocates in execute mode, and under the sum a release frees bytes no admitted step can use. Instead [step scratch](#step-scratch), packed by the DAG's closure *decided* |
| `mem_budget` is a function of the dimensions and options | Sizes come from host tables (record counts, kernel table lengths). `eri` builds them twice, since a refused or dry budget hands back empty buffers | Setup counts every length into `Dims`; `run` builds the [records and work lists](#lengths-counted-in-setup-built-by-the-step) into its pinned scratch, once per run *decided* 2026-11-21 |
| [The library stream carries every BLAS and solver call](#streams-and-device-resources) | And every kernel between them. `orthogonalizer` and `scf` never use their own stream; `eri` only for its class kernels. Each holds the mutex for its whole enqueueing | No library stream any step may enqueue on: [one handle set per card](#one-handle-set-per-card-under-one-lock), on the card's own stream, the lock holder's alone *decided* 2026-11-24, revised 2026-11-25 |
| A manager's internal lock | Needed, as predicted, but held only while enqueueing. The one library stream is what keeps two callers' scratch apart on the card | Gone with managers, 2026-11-23: the J and K build works in its caller's scratch and holds no state ([shared code](#shared-code-a-function-over-a-view-not-a-manager)) *decided* |
| Step-end sync cost: to measure | Under 25 µs over 50 SCF trips: a step that reads a value back has already synced. Where a step only enqueues, its device time lands there *run* | Closed for the SCF |
| The `ForkStream` is a field of the `Output` | The wrapper makes it, keeps it beside the `Output` in the slot, and syncs it without the step's cooperation | The lane is the wrapper's; taken into [Streams](#streams-and-device-resources) |
| No slot is ever read empty | A runtime assert: the slot is an `Option` filled by budget mode | Said so in [Steps](#steps-and-the-mainbus); the types hold only that a read is declared |
| The budget table, device-free | A dry budget counts every step-owned byte exactly and the library calls' work as zero: the size queries need live handles | Accept, and print it as such |
| [Nothing large on the plain heap in `run`](#enforcement) | Host libraries that return a `Vec` break it (`fb_sad::sad_density`) | The host layer writes into slices |
| The port ([order of work](#what-is-retired-and-the-order-of-work)) | wwr's `Stream` and handles were `!Send`, and the prototype carried two `unsafe impl`s for them. All of `fb-integrals-sys`'s task layer is rewritten; `RecordTables` and `fb-recipes-sys`'s slice-taking functions carry over. A step touches four places, and `Reads` stops at eight | Settled in `fb-core-wwr` itself (#1784): `Stream` and `Event` are `Send + Sync`, `BlasHandle` and `SolverHandle` are `Send` and not `Sync`, and a library handle takes an `Arc<Stream>`; the CCL rank stays `!Send`. `v0.0.2` carries no shim. *decided* Since 2026-11-25 the new layer does not use those wrappers: it is written [from scratch](#built-from-scratch-over-the-c-abi), with the same `Send` and `Sync` answers |

## What the second prototype showed (v0.0.3)

2026-11-25, RTX 3080 (CUDA) only. `experimental/rust/v0.0.3`: items 1 and 2 of
[the order of work](#the-order-of-work), the device-free core with scratch
packed by the closure, and two batched steps (`eri` and `jk`) on `num_gpus`
cards. Its README has the tests, the tables and fourteen things for the design
to change; this is what bears on the text above. Not run: the DF-RHF chain, a
real second card, HIP, the enforcement checks.

**Held** *run*: B within 1.4e-13 of the old Rust path's on water and benzene,
J and K within 1.0e-13 of a host contraction; the same at forced batches (206
for benzene's `eri`) and on two logical cards, NaN-filled; a replay allocates
nothing; an over-cap plan is refused with the whole table, a cap past the card
is clamped, and a limit that holds no batch is refused naming the step. On no
card: the closure, the packing, the lower bound, the offsets' independence of
declaration order, the lease that cannot escape (a `compile_fail` doctest),
and the NaN fill, each as [the closure
model](#what-the-closure-model-showed) and [step scratch](#step-scratch) word
them.

**What it changed here.**

- **One handle set per card could not be built on `rust/`'s wrappers**, and
  v0.0.3 held a BLAS handle per library-using step per card instead. That is
  the origin of both decisions of 2026-11-25: [the card's stream under its
  lock](#one-handle-set-per-card-under-one-lock), and [from
  scratch](#built-from-scratch-over-the-c-abi).
- **The other thirteen items** are answered one by one in [carried from
  v0.0.3](#carried-from-v003-and-what-stays-open): some built by v0.0.4, some
  proposed, the rest carried.

## Failure, diagnostics, determinism

- **A plan over budget is a refusal, not a panic**: the final budget node
  prints the table and exits with an error. An allocation the counter admitted
  and the device then refused aborts at the failing call, by the tree's create
  policy, still at init. A cap the machine cannot meet is not that case: every
  cap is clamped to what the machine has at init
  ([caps](#caps-and-what-the-limit-covers)), so it is a refusal with the
  table.
- **A launch that fails for memory reports what it can read.** By execute mode
  everything budgeted is allocated, so a kernel launch that returns an
  allocation error is about memory outside the budget ([what a card takes
  after budget mode](#what-a-card-takes-after-budget-mode)). The error names
  the step and the kernel, prints the card's free memory and, on CUDA, the
  device's stack limit, and lists what can cause it on that backend: on CUDA
  the kernel's stack, when its frame is larger than any launched so far;
  modules loading at a first launch; another process on the card. It suggests
  a lower device cap, or looking for the other process. It does not guess
  which, and cannot print the kernel's own frame, which would need the
  manifest that is not kept. *decided* 2026-11-24
- **A config whose steps do not close is refused when the DAG is built**: a
  step that declares it reads another the config does not ask for, named with
  what it needs.
- **A step's failure in execute mode is a `Result`**. A failed step fails its
  successors and nothing else, so the wrapper holds a failed flag per node and
  skips a node with a failed predecessor. rustflow catches a panic per task
  and surfaces it from `wait` as `RunError::Panicked`
  ([README](https://github.com/whatDiracIsCooking/rustflow)); that replaces
  the unwinding guard `programming_model.md` describes. Built in the
  experiment, not tested.
- **The budget table is the memory diagnostic**: one row per step and kind,
  each step's scratch and the handle workspace their own rows, the packed
  scratch peak beside the plain sum of the declarations (so packing's gain
  reads plainly) and the lower bound with the steps that make it (so its
  distance from the best reads plainly), the totals against the caps, each cap
  as written beside the value used when it was clamped. Logged on every run,
  printed in full on a refusal. rustflow's `ChromeObserver` and `dump()`
  (Graphviz DOT) sit beside the timing record.
- **Determinism.** Results are those of the edges. Accept or refuse, the
  scratch offsets, and the table, are the same every run. The order steps were
  budgeted or timed in is not.

## What is retired, and the order of work

**Retired:** `fb_core_planner`, `fb_core_arena`'s tables and blocks (its
`host_available` reading stays), `fb_core_thread_pool`'s executor, the
`layout` contract, the `plan_*.txt` goldens, and the kernel bindings'
arena-row arguments. Every existing `add_*` builder is written against those,
so this is a port of the Rust tree's task layer and not an edit to it. Since
2026-11-25 the wrappers under the task layer go the same way.

### Built from scratch over the C ABI

*decided* 2026-11-25. The new layer is designed as if nothing in `rust/` above
the C ABI existed. v0.0.3 was written over `rust/`'s wrappers, and the one
thing it could not build as designed, it could not build because of them
([why not the rebind](#one-handle-set-per-card-under-one-lock)): they were
shaped by lanes and arena rows, the model this design retires. Working around
them inside each prototype, and proposing changes to them afterwards, costs
more than writing the few calls a prototype makes.

| In `rust/` | In the new layer | Why |
|---|---|---|
| The kernel libraries and the C ABI: what `rust/CMakeLists.txt` builds, `wwr-sys`, and the `extern` declarations in each binding crate's `bindings.rs` | Kept, linked as they are | The GPU substance. Nothing here changes a kernel or a call |
| The device-free crates: `fb-basis`, `fb-molecule`, `fb-io-*`, `fb-sad`, `fb-integrals-mmd`, `fb-kernel-abi`, `fb-core-error-handling` | Kept, as dependencies | They hold no stream, no handle and no memory model |
| The handle wrappers: `fb-core-wwr`'s streams, events, copies, and its BLAS and solver handles | Written again | A handle there owns the `Arc<Stream>` it was made on and takes ownership of its workspace; here both live inside the card's guard |
| The safe kernel and recipe wrappers: `fb-integrals-sys`'s launch functions, `fb-recipes-sys`'s `trsm` | Written again, over the same `extern` declarations | The launch functions take `fb_core_arena`'s `Bufs`, `RowId` and `Layout`, the arena-row arguments retired above; `trsm` reads its stream from `blas.owner()` |
| The task layer: the planner, the arena, the thread pool, every `add_*` builder, `rust/pipeline` | Retired, as above | A port, not an edit |

- **The rule a prototype is held to.** Its code names the C ABI, the
  device-free crates and plain data, and nothing else of `rust/`: no type that
  owns or names a stream, a library handle or an arena row. The binding crates
  are still linked, for their `extern` declarations and build scripts.
  Anything from the wrapper or task layers appears only under
  `[dev-dependencies]`, as the old path the tests hold the new one against.
- **What it frees.** A library binding takes the card's guard, so a recipe
  cannot be handed the wrong stream. A kernel launch takes a stream and slices
  of the step's lease, and checks their lengths against its own formula, with
  no row handle between. A solver call's work buffer is a slice checked against
  the size query. The stream and event types are made for a step's slots and
  the card's stream, not adapted from a lane. And what setup counts, and what
  `Dims` carries, are written with the layouts that read them.
- **What it does not free.** The design questions: what a step's limit covers,
  the default limit, B written by several cards, the reduction through the
  host ([carried from v0.0.3](#carried-from-v003-and-what-stays-open)). Those
  are about the memory model, not about `rust/`.
- **What it costs.** v0.0.3 is about 4,700 lines over about 15,000 of wrappers
  (`fb-core-wwr` 6,900, `fb-integrals-sys` 6,200, `fb-recipes-sys` 2,300). A
  prototype writes only the calls its steps make: streams, events and copies,
  the BLAS and solver calls, the recipes it calls (the blocked `trsm`, the
  metric's factor) and its kernel launches.
- **The old path stays the oracle.** The energies, and the planner's block the
  budget is measured against, come from `rust/` as merged, run in the same
  process by the tests.
- **What happens to `rust/`'s wrappers** is settled when the layer lands
  there, not before: the new crates beside the old ones, and the old retired
  with the task layer once the gates pass on the new one alone. This takes the
  place of changing `fb-core-wwr` and `fb-recipes-sys` in step with each
  prototype. *proposed*

### The order of work

Item 1 is prototyped twice (`v0.0.2`, then `v0.0.3` with scratch packed by the
closure), and item 2 by `v0.0.3`, on one card and on two logical cards. Item 3
ran in `v0.0.2` in its crudest form, unbatched on one card, and item 4's sync
cost is measured there ([the
prototype](#what-the-first-prototype-showed-v002)); the sum against the fold
is not. **v0.0.4 is item 3 as the design words it**, and the first written
from scratch ([below](#the-v004-prototype)). All of it is throwaway.

1. **The core, device-free.** `MainBus`, the declared `View` and the builder
   that makes edges from it, `MemBudget`, `Budget` with its counters, flag,
   table and final node, the heap slab, and the closure and the scratch
   packing, and the two-mode DAG on rustflow. Tests: accept and refuse are
   deterministic; the table is complete on a refusal; a slab cannot grow; a
   config that does not close is refused; the closure equals reachability by a
   plain search; every pair of overlapping scratch regions is ordered by the
   DAG, and the offsets do not depend on declaration order (each as the
   [closure model](#what-the-closure-model-showed) checks it). (A condition
   loop inside a subflow, at the pinned rustflow, is checked already:
   `tests/rustflow_pin.rs` in `experimental/rust/v0.0.2`.)
2. **One step, batched, on `num_gpus` cards, beside the old path.** The DF J/K
   pass over batches of P is the candidate: the experiment's fourth variant is
   most of it, on one card. The step's streams and the wrapper's sync, each
   card's `CardResources` and its lock taken in card order, its scratch region
   with the solver's work in it, the fixed handle workspace, per-node device
   selection.
3. **The DF-RHF chain as steps**: setup, the integrals, X, the guess, the SCF
   with its condition loop inside its subflow. Held against the old path's
   energies, which the tree already holds against PySCF.
4. **Measure**: the budget's total against the planner's block for the same
   runs, and what the step-end sync costs on the SCF iteration.
5. Then **the enforcement checks** (the clippy bans, the allocator record, the
   NaN fill of scratch) and **HIP**. Batching and every card are not on this
   list: each step has them from its first form ([batched
   only](#sizes-batched-only)).

**Development on the old task layer stops now** *decided*. The new layer is
built from the ground up, and crates of the old one may be disabled along the
way where that helps; the retired crates are deleted once the gates pass on
the new one alone. `rust/docs/memory.md` and `programming_model.md` change
with the implementation, not before, so they never describe what does not
exist.

## The v0.0.4 prototype

**A design, not yet built.** The third prototype: item 3 of the order of work,
the DF-RHF chain as steps on `num_gpus` cards, batched. It is the first to run
[the card's stream under its lock](#one-handle-set-per-card-under-one-lock)
and [a step with streams of its own and library
calls](#a-step-with-streams-of-its-own-and-library-calls), and the first
written [from scratch over the C ABI](#built-from-scratch-over-the-c-abi).
What it takes from v0.0.3 is that prototype's own code and not `rust/`'s: the
device-free core (the bus, the flow layer, the budget and the packer), copied
and changed where the tables below say.

Three marks are used in this part. **Decided** was agreed in discussion on
2026-11-25. **Carried** is v0.0.3's or the design's, unchanged. **Proposed**
is this page's suggestion and not agreed.

### The DF-RHF chain

v0.0.2's graph, over v0.0.3's core. What changes against v0.0.2: `eri` is
v0.0.3's batched build with the ring of two, there is no manager, the SCF
calls `build_jk` itself, and every step runs on `num_gpus` cards or on gpu0.

```text
harmonics ─┬─> overlap ──> orthogonalizer ─┐
records  ──┼─> core_hamiltonian ───────────┤
           └─> eri ────────────────────────┼─> scf
guess ─────────────────────────────────────┘
```

| Step | Runs on | Library | Its `Output` | Its scratch |
|---|---|---|---|---|
| `harmonics` | gpu0 | no | U, the Cartesian-to-spherical transform | The pinned staging |
| `records` | all GPUs | no | The orbital records, on every card | The pinned staging |
| `guess` | gpu0 | no | The SAD density | The pinned staging |
| `overlap` | gpu0 | yes | S | The Cartesian matrix |
| `core_hamiltonian` | gpu0 | yes | H = T + V | The Cartesian matrix, the nuclei |
| `orthogonalizer` | gpu0 | yes | X = S^(-1/2) | `syevd`'s work at `n_ao`, the eigenvectors |
| `eri` | all GPUs | yes, every card | B, on the host heap | The metric, its factor, and a ring of two batches per card |
| `scf` | all GPUs | yes, every card | C, the orbital energies, D, the energy | `build_jk`'s per card; on gpu0 also F, the DIIS ring and `syevd`'s work |

- **The SCF's loop is inside its node.** A condition loop in `ScfStep`'s
  subflow, as [steps](#steps-and-the-mainbus) has it; v0.0.2 had it in the
  outer graph. One trip is `build_jk` over the carried D, F, the energy, the
  residual, the DIIS step and the Roothaan step. *Carried.*
- **`build_jk` is v0.0.3's algorithm**: batches of aux functions, each card
  contracting its share, reduced onto gpu0 through the host, a blocking sync
  per batch. The SCF declares `JkLayout`'s bytes per card in its own layout
  and passes its lease. *Carried*, over the new bindings.
- **The SCF's solver calls are on gpu0's guard.** `syevd`'s work buffer is a
  tenant of the SCF's scratch, sized by the query in `mem_budget`, and the
  binding checks the slice against the query before the call. *Carried.*
- **Five steps hold gpu0's lock**, and the DAG leaves several pairs of them
  unordered: `overlap`, `core_hamiltonian` and `eri` among themselves, and
  the orthogonalizer against the last two. This is the first DAG in which a
  step waits for a card's lock.
- **Every binding is the prototype's own.** A module of the crate holds the
  stream, the event, the copies, the guard with its BLAS and solver calls, the
  `trsm` and Cholesky recipes over the guard, and one launch function per
  kernel family the eight steps use, each over the `extern` declaration and
  slices. *Decided*, by the rule [from
  scratch](#built-from-scratch-over-the-c-abi) states.
- **Crudest forms stay**: K from D with no occupied factor, no screening, a
  neutral closed-shell molecule, built-in bases.

### What it is held against

One run each, in one process per molecule, the bars fixed by analysis and
not bitwise. "The old Rust path" is `rust/` as merged, a dev-dependency of the
tests and of nothing else:

| What | Against | Bar |
|---|---|---|
| The energy of water, benzene and ethylene (#865's geometry), cc-pVDZ over cc-pVDZ-JKFIT | The old Rust path in the same process; PySCF's DF-RHF as v0.0.2's README records it | 1e-9 Eh, the old test's |
| The same at forced batches (`eri` and the SCF's `build_jk`) | The default run | 1e-9 Eh |
| The same on two logical cards (`devices = [0, 0]`), NaN-filled | One card | 1e-9 Eh |
| B from the ring of two | v0.0.3's B, one batch in flight | 1e-10 of the largest element |
| A replay | The run before it | The same energy to 1e-9 Eh; the ledger's allocation count unchanged |

### What it runs that no prototype has

| The design says | Held by |
|---|---|
| A library-using step's work is finished on every stream it touched before its lock is released | The orthogonalizer schedule that broke the library stream: the SCF's scratch laid over the orthogonalizer's, twenty runs on eight workers, NaN-filled, X held against a host `S^(-1/2)` |
| The handles are reachable only through the guard | A `compile_fail` doctest: a `run` that keeps the handle past the guard |
| A recipe cannot be handed a step's stream | A `compile_fail` doctest: `trsm` called with a slot's stream in place of the guard |
| The new layer names nothing of `rust/`'s wrapper and task layers | A test over the crate's `src/`: no path into `fb_core_wwr`, `fb_core_arena`, `fb_core_planner` or `fb_core_thread_pool`, and from the binding crates only `extern` functions and plain data; `Cargo.toml` lists those four under `[dev-dependencies]` or not at all |
| Locks are taken in card order and released after the sync | The acquisitions the bus records, as in v0.0.3, now with waits: a step that found gpu0's lock held is recorded as having waited |
| Unordered lock holders keep disjoint scratch | The budget table's offsets for `overlap`, `core_hamiltonian`, `eri` and the orthogonalizer on gpu0 |
| A step's own streams and the card's are ordered by events inside the step | `eri` at ring depth 2 against depth 1, to the bar above; and a negative control with the event before the fit removed, expected to fail under the NaN fill |
| A solver call's work buffer is step scratch, checked against the size query | The SCF and the orthogonalizer run on it; an undeclared `syevd` is refused naming the step |
| A kernel launch's slices are checked against its formula | A launch handed a short slice is refused before anything is enqueued, naming the kernel |
| The default cap is read after the handles exist | The table's `cap used` against a reading taken by hand after setup |
| The SCF is a condition loop inside its step's subflow | The node after `scf` sees the loop done; the trip counts against v0.0.2's |

One measurement beside them, with no bar: each molecule's budget total
against the old path's planner block for the same run (item 4 of the order of
work).

One-off times are printed as v0.0.3's were: benzene's `eri` at 206 forced
batches took 347 ms there with a blocking sync per batch, and the ring is
what should move it. Not a benchmark; the real question is at gly10's size,
which this prototype does not run.

### Carried from v0.0.3, and what stays open

v0.0.3's README lists fourteen things for the design to change. Item 1 is
settled by [the card's stream](#one-handle-set-per-card-under-one-lock), and
item 2 with it. The rest:

| v0.0.3's item | Here |
|---|---|
| 3. What a step's limit covers | **Proposed:** the limit bounds the batch (here, the ring) and not the O(N²) residents, which the cap admits or refuses; a refusal names both parts. Built as v0.0.3 has it until agreed |
| 4. `mem_budget` depends on the cap through the default limit | **Proposed:** the default limit is an absolute constant, so a dry budget is exact. Built as v0.0.3 has it until agreed |
| 5. Pinned scratch forces a sync per batch, or a ring | The ring is built for `eri`. `build_jk` keeps the blocking sync |
| 6. Setup builds a pair-scaling array to count from | **Proposed:** a counting walk over the shells that builds no array. Built here, in the prototype's setup; it belongs in `fb-integrals-mmd` if it holds |
| 7. The layout function called by the wrapper, once | **Proposed:** the layout as an associated type of `Step`. Built here, since the ring's slots make a double carve easy to write |
| 8. An all-GPU step's layout takes the card | Carried |
| 9. Largest-first against the bound | Carried: the table's lower-bound row, on a wider DAG than v0.0.3's |
| 10. `run` takes its lane by `&mut` | Changed by the guard: a step's inner task for a card takes that card's guard and the step's streams on it. The lane is the prototype's own type |
| 11. B written by several cards needs an `unsafe` | Carried, and listed as the memory model's second `unsafe` |
| 12. The reduction and the broadcast go through the host | Carried |
| 13. `Dims` carries per-class counts | Carried; `Dims` is written again beside the layouts that read it |
| 14. Logical cards as a test facility | Carried |

### Not in it

- **Any change under `rust/`.** The prototype links `rust/`'s libraries and
  changes no crate there; what replaces the wrappers is settled when the layer
  lands ([from scratch](#built-from-scratch-over-the-c-abi)).
- **Real multi-card.** Two logical cards on device 0, as v0.0.3. A launch on
  the wrong card's context still cannot fail.
- **HIP.** CUDA only. With no rebind, what was untested there is smaller: the
  workspace set once on a hipBLAS handle, and rocBLAS's hard need for it.
- **A step with further handles of its own**, and two fits in flight on one
  card.
- **A ring in `build_jk`**, and batches of B kept on the cards between
  passes: every SCF trip uploads all of B, a batch at a time.
- **gly10, or any size where B does not fit a card.**
- **The enforcement checks**: the clippy bans, the allocator record, the
  unbudgeted-heap row.
- **Screening, the default limit's value, packing `Output`s, several
  nodes.**
- **The replay reset.** v0.0.2's reset mode is kept; what carries over is
  the SCF's orbitals and nothing in scratch.

## Open questions

### Set aside for a fuller discussion

- **Several nodes.** The commitment is to a single node's cards ([batched
  only](#sizes-batched-only)). The further one was raised on 2026-11-23: a
  step runs on several nodes with several GPUs each, or is not built. Against
  making it a gate now: nothing here can test it, and a rule with no check
  drifts. What would keep it open cheaply is one rule, that work on different
  cards communicates only through the collectives layer; B on one host's heap
  is the first thing that rule would have to answer. *to discuss*
- **The step-end sync and the library stream.** Closed 2026-11-24: the library
  stream any step could enqueue on is gone, under one handle set per card and
  that card's lock. Revised 2026-11-25: a step's library calls go on the
  card's own stream, which its lock holder has alone and the wrapper syncs
  ([one handle set per card](#one-handle-set-per-card-under-one-lock)).
  *decided*
- **Packing `Output`s too**, by the same closure, held until the measurement
  says `Output`s are what stacks. [Step scratch](#step-scratch). *to discuss*
- **What is cleared between replays, and by whom.** A prologue node, an
  epilogue node, or a `reset` mode on the steps that need one (an explicit
  `on_replay` suggested). Tagged for later. [Steps](#steps-and-the-mainbus).
  *to discuss*

### To measure, verify or pick a value for

- **The blocking sync.** Which scheduling flag makes a worker waiting on its
  step's stream sleep, on each backend. If what it costs on the SCF iteration
  argues for a cheaper replacement, the replacement has to keep pinned scratch
  safe as well as device scratch ([step scratch](#step-scratch)).
- **The executor's width** against the GPU steps that can be waiting at once.
- **What the sum costs.** The ratio of summed step peaks to the fold's peak,
  per `examples/` config, split per step into `Output` bytes and scratch
  bytes, and whether the peak sits inside one step or across many. That says
  whether packing scratch is enough, whether `Output`s should be packed too
  ([the same closure](#step-scratch)), or whether the remedy is a step's
  batched path. Not measured; information only, no bar.
- **Packing quality.** The packed peak against the lower bound the closure
  gives (the heaviest set of mutually unordered declarations), on the same
  configs. On the model's dummy and random DAGs it is answered
  ([there](#what-the-closure-model-showed)): at the bound on 98.1% largest
  first, the order taken; real configs remain, and the table's lower-bound row
  answers it on each.
- **The rest of the workspace sweep.** Not measured: upper solves, rectangular
  `syrk` and `gemv`, the batched solver calls, hipBLASLt. What the sentinel
  tests would run, if taken ([workspace](#blas-and-solver-workspace),
  proposed).
- **The host and device margins' values** are picked
  ([caps](#caps-and-what-the-limit-covers)): judgment, to be raised when a
  launch fails for memory or a run is killed on the host. The kernels' stack
  is not read per card.
- **Scratch on HIP while a kernel runs.** The measurement that ROCm reserves
  nothing for a kernel's scratch read the card after each launch, on ROCm
  7.2.4 ([what a card takes after budget
  mode](#what-a-card-takes-after-budget-mode)). To rerun on ROCm 10 with the
  card's memory sampled during a large launch of a large-frame kernel: what it
  takes then is the HIP share of the device margin.
- **The default limit fraction** that picks a step's batch width.
- **Task granularity inside a step.** A step is one node of the outer graph
  ([decided](#steps-and-the-mainbus)); how much of its inner graph is
  rustflow's (a subflow, a pipeline) and how much stays a plain loop on one
  stream. A chain of small launches belongs in one task.
- **The `cells` work list on the device.** It is a dense cross product of
  orbital shell pairs and aux shells, so a kernel could derive each entry from
  two short lists (58,653 and 1,272 entries on gly10) and drop 1.79 GB pinned
  and 1.79 GB device. A kernel change, in FelizJueves; what it costs in the
  kernel is to measure. Unlike packing, it lowers DF-HF's peak: the work lists
  are live inside the step that sets it.
  [Lengths](#lengths-counted-in-setup-built-by-the-step).
- **Rebuilding the records and work lists on a replay.** Every `run` rebuilds
  them in place, since the records hold coordinates. Whether rebuilding only
  the records is worth a second path depends on what the full rebuild costs
  per geometry step.
- **Collectives.** An all-GPU step holds a stream per card, so one rank's
  calls within a step stay on one stream. Whether NCCL and RCCL mind a rank
  moving between steps' streams is to verify.
- **Idle workers.** rustflow's workers give up a search after a fixed number
  of attempts and park through a notifier (DESIGN.md, from a summary; verify
  in `src/executor/notifier.rs`). What that costs while a run waits on the
  device needs reading.
- **Submitting from outside the pool.** "A graph submitted from outside the
  pool reaches it only by stealing", about 100 ns a task
  ([README](https://github.com/whatDiracIsCooking/rustflow)). At a few dozen
  steps it should not matter.
- **Pinned pages and NUMA.** With no thread per device, a pinned slab is
  allocated and touched by whichever worker runs that step's `mem_budget`.
- **The large-allocation check and C libraries.** The allocator wrapper does
  not see a library's own `malloc`.
- **The dependency.** Settled: rustflow is unpublished, so it is a git
  dependency pinned once, by its full commit (`f184bcab…`, its head on
  2026-11-24), in `rust/Cargo.toml`'s `[workspace.dependencies]`; a member
  inherits it with `workspace = true`, and a prototype under
  `experimental/rust/` names the same `rev` (`v0.0.2` builds at it). The rule
  is in `rust/docs/testing.md`, "What runs when", and
  `doc_check/tests/git_dependencies.rs` checks it. Cargo locks a workspace
  dependency only once a member uses it, so `rust/Cargo.lock` has no rustflow
  entry yet. *to discuss*: rustflow depends on `crossbeam-deque`, which
  `doc_check`'s thread-pool rule (`rust/docs/programming_model.md`, "CPU
  workers") forbids any member to reach. The first member of `rust/` on
  rustflow, the executor's port or a prototype if one joins the workspace,
  changes that rule with it; a prototype that is a workspace of its own, as
  `v0.0.2` and `v0.0.3` are and v0.0.4 will be, needs no change and locks the
  pin's `rev`.
- **Deferred by decision**: AVAS, the semi-numerical Hessian, semi-numerical
  gradients.

## This page as HTML

`v0.0.4.html` beside this file is this README rendered by
`rust/docs/md2html.py`, with the figures of `README.figs` spliced in: each
`text` block above that names a figure, and the two text diagrams, is replaced
by one. `README.figs` also holds the figures' style sheet and the script that
draws the example's graph, its node table and its budget chart. The `.md` and
the `.figs` are the source; after editing either, from this directory:

```sh
uv run --no-project --with markdown-it-py python ../../../rust/docs/md2html.py README.md v0.0.4.html README.figs
```

The figures of the design sections are `docs/rust-redesign.html`'s, as drawn
there; the script is spliced in after this paragraph, at the page's end.
