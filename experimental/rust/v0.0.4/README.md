# rust/v0.0.4: the card's stream under its lock, and the DF-RHF chain

**A design, not yet built.** This page says what the third prototype of the
redesign in `docs/rust-redesign.html` ("Rust rebuild: steps, the memory budget
and rustflow") will build, and how it will be held. There is no crate beside
it yet. Written 2026-11-25, after `../v0.0.3`.

It covers two things:

1. **One handle set per card, revised.** v0.0.3 could not build it as the
   design words it. Decided 2026-11-25: the card owns a stream, and the card's
   lock guards that stream with the handles. A step owns its own streams; a
   step that calls the library also borrows the card's.
2. **The DF-RHF chain as steps** (item 3 of the design's order of work), on
   `num_gpus` cards, batched, over v0.0.3's core: the one-electron steps, the
   orthogonalizer, the guess, `eri`, and an SCF that calls `build_jk` on every
   trip.

Like its two predecessors it will be throwaway: a crate and a workspace of its
own, in no gate, never merged into `rust/`.

Three marks are used below. **Decided** was agreed in discussion on
2026-11-25. **Carried** is v0.0.3's or the design's, unchanged. **Proposed**
is this page's suggestion and not agreed.

## What v0.0.3 could not build

The design has one BLAS handle per card, rebound to the holder's stream at
every acquisition of the card's lock (`SetStream`, then `SetWorkspace`).
v0.0.3 found three things in the way (its README, "What the prototype says to
change", item 1):

- `fb_core_wwr::blas::BlasHandle` keeps the `Arc<Stream>` it was made on and
  has no rebind.
- `fb_recipes_sys::trsm` launches its diagonal kernels on that stream
  (`blas.owner()`) and its gemms through the handle. A handle rebound through
  the raw call would run one solve on two streams with nothing ordering them.
- `BlasHandle::bind_workspace` takes ownership of the workspace, where the
  design's workspace is the card's and outlives every binding.

So v0.0.3 made a BLAS handle per library-using step per card: the form the
design had rejected for its cost on HIP (58 MiB a handle).

## The card's library: a stream, the handles and the workspace, under one lock

**Decided.** It replaces the rebind in the design's "One handle set per card,
under one lock".

- **The card owns a stream.** Each card's `CardResources` holds one stream, a
  BLAS handle and a dense-solver handle made on that stream, and the fixed
  128 MiB BLAS workspace. The handles keep that stream for the life of the
  process. Nothing is rebound: `SetStream` is never called after creation, and
  the workspace is set once.
- **The lock guards all of it.** `CardResources` is a `Mutex` around the
  stream, the handles and the workspace, not a `Mutex<()>` beside them. They
  are reachable only through the guard.
- **A step owns its streams.** A GPU step has streams of its own on each card
  it spans: one by default, more if it declares them. The wrapper makes them
  in budget mode and keeps them in the step's slot.
- **A library-using step also borrows the card's.** It declares the library
  per card in `mem_budget`, as in v0.0.3. For its `run` it holds that card's
  guard, and its library calls go on the card's stream.
- **The wrapper syncs all of them before release.** Before `run` it takes the
  guard of every card the step declared, in card order. After `run` it
  synchronizes every stream the step has on each card, and each held card's
  stream, and only then drops the guards.

| | A step's own streams | The card's stream |
|---|---|---|
| How many | One per GPU step per card by default; more if declared | One per card |
| Made | By the wrapper, in budget mode | In setup, with the handles |
| Reached through | The step's lane | The card's guard |
| What goes on it | The step's kernels and copies | Library calls; and a holder's kernels and copies, if it wants one order with no event |
| Who may use it | Its step | The step holding the card's lock, for its whole `run` |

A step that calls no library holds no guard and overlaps other steps on the
card as before.

```text
card's stream:  eri [ factor L | fit 0 | fit 1 | fit 2 ] sync, release; orthogonalizer [ syevd, gemms ]
```

**Why this is not the old library stream.** The design dropped the device's
library stream on 2026-11-24 for a hole: the orthogonalizer enqueued
`syevd(S)` on it, its wrapper synchronized the step's own, empty stream, and
the SCF then wrote over `syevd`'s work. Two things made the hole, and neither
is here:

| | The library stream, as dropped | The card's stream |
|---|---|---|
| Who may enqueue on it | Any step, under a mutex held per call | The lock's holder alone, for its whole `run` |
| What the wrapper synchronizes | The step's own stream | The step's own streams and the card's, before the lock is released |

So a finished node is still finished device work, on every stream the step
touched, and the scratch argument of the design is unchanged.

**What it buys.**

- **Safe by construction.** The rebind was safe by measurement, on one release
  of each library: cuBLAS drops the workspace at `SetStream`, hipBLAS keeps
  it, and what the solver handles do was not measured. With no rebind there
  is nothing to measure and no sentinel test to keep.
- **Exclusive use by type.** The handles are behind the guard, so "no handle
  is shared with work in flight" is the borrow checker's.
- **The recipes are right as written.** `trsm`'s kernels and its gemms land on
  one stream because the handle has only ever had one.
- **The default cap sees the handles.** They are made in setup, so "one read
  per device, after its handles exist" holds as the design words it, and
  their memory no longer grows with the number of library-using steps.

**What it gives up.** Narrowing the lock to a step's library phases later. A
holder's library calls are on the card's stream, so two library-using steps
cannot interleave on a card without a second handle set. The design already
holds the lock through a step's whole `run` and declined the ring of handle
sets, so nothing decided is lost.

**Not taken.**

- **(a) A rebind.** `BlasHandle::rebind(&mut self, Arc<Stream>, workspace)`,
  with the recipes reading the stream from the current binding. It keeps
  "every step's library calls on its own stream", at the price of resting on
  the measured `SetStream` behaviour above.
- **(b) Every recipe takes its stream explicitly.** It does not remove the
  rebind; it only moves the stream out of the handle.
- **Rebinding inside a step, per call, to each of its streams.** Two library
  calls in flight on two streams would share the one fixed workspace.

## A step with streams of its own and library calls

`experimental/rust/df_batched/multi_gpu_spherical` is the case: each card
builds its share of B over a ring of two batches in flight, and every trip
has kernels, a library solve and a download. v0.0.4's `eri` is built that
way, so the rule above is run in the form that needs events.

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

The events are those the design keeps `ForkStream` for: a step's own inner
fan-out. Whoever fans out, joins: every event is waited on before `run`
returns, and the wrapper's sync is the last word.

**What serializes.** The two slots' fits run one after the other on the one
card's stream. The ring overlaps a batch's copies and kernels with the
previous batch's fit; it does not overlap two fits. Expected to cost little,
since two solves on one card time-slice. Not measured.

**Further handles are not built.** A step that needs concurrent library calls
inside itself declares handles of its own, each with a workspace it budgets,
as the design already says. No step here needs it.

**The ring is part of the layout.** A slot's device and pinned bytes are
tenants of `eri`'s scratch, so the layout holds two of each, and the width is
the widest whose ring fits the step's limit, as that experiment sizes it.

A simple library user needs none of this. The orthogonalizer and the SCF put
everything on the card's stream, in the order they enqueue, with no event.

## Where the handles and the workspace are made

- **Setup** makes each card's stream and its two handles. *Decided.*
- **Setup then reads the card's free memory** for the default cap. *Carried.*
- **The final budget node allocates the 128 MiB workspace** and binds it to
  the card's BLAS handle, once. It stays a row of the budget table, charged
  against the cap. *Carried.* Binding it in setup would put it before the
  reading, and the row would then count it twice.
- **A device-free `--budget-only`** makes no handles and reports each solver
  call's work buffer as zero. *Carried.*

## The DF-RHF chain

v0.0.2's graph, over v0.0.3's core. What changes against v0.0.2: `eri` is
v0.0.3's batched build with the ring above, there is no manager, the SCF
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
  subflow, as the design has it; v0.0.2 had it in the outer graph. One trip
  is `build_jk` over the carried D, F, the energy, the residual, the DIIS
  step and the Roothaan step. *Carried.*
- **`build_jk` is v0.0.3's**, unchanged: batches of aux functions, each card
  contracting its share, reduced onto gpu0 through the host, a blocking sync
  per batch. The SCF declares `JkLayout`'s bytes per card in its own layout
  and passes its lease. *Carried.*
- **The SCF's solver calls are on gpu0's guard.** `syevd`'s work buffer is a
  tenant of the SCF's scratch, sized by the query in `mem_budget`, and the
  binding checks the slice against the query before the call. *Carried.*
- **Five steps hold gpu0's lock**, and the DAG leaves several pairs of them
  unordered: `overlap`, `core_hamiltonian` and `eri` among themselves, and
  the orthogonalizer against the last two. This is the first DAG in which a
  step waits for a card's lock.
- **Crudest forms stay**: K from D with no occupied factor, no screening, a
  neutral closed-shell molecule, built-in bases.

## What it is held against

One run each, in one process per molecule, the bars fixed by analysis and
not bitwise:

| What | Against | Bar |
|---|---|---|
| The energy of water, benzene and ethylene (#865's geometry), cc-pVDZ over cc-pVDZ-JKFIT | The old Rust path in the same process; PySCF's DF-RHF as v0.0.2's README records it | 1e-9 Eh, the old test's |
| The same at forced batches (`eri` and the SCF's `build_jk`) | The default run | 1e-9 Eh |
| The same on two logical cards (`devices = [0, 0]`), NaN-filled | One card | 1e-9 Eh |
| B from the ring of two | v0.0.3's B, one batch in flight | 1e-10 of the largest element |
| A replay | The run before it | The same energy to 1e-9 Eh; the ledger's allocation count unchanged |

## What it runs that no prototype has

| The design says, or this page does | Held by |
|---|---|
| A library-using step's work is finished on every stream it touched before its lock is released | The orthogonalizer schedule that broke the library stream: the SCF's scratch laid over the orthogonalizer's, twenty runs on eight workers, NaN-filled, X held against a host `S^(-1/2)` |
| The handles are reachable only through the guard | A `compile_fail` doctest: a `run` that keeps the handle past the guard |
| Locks are taken in card order and released after the sync | The acquisitions the bus records, as in v0.0.3, now with waits: a step that found gpu0's lock held is recorded as having waited |
| Unordered lock holders keep disjoint scratch | The budget table's offsets for `overlap`, `core_hamiltonian`, `eri` and the orthogonalizer on gpu0 |
| A step's own streams and the card's are ordered by events inside the step | `eri` at ring depth 2 against depth 1, to the bar above; and a negative control with the event before the fit removed, expected to fail under the NaN fill |
| A solver call's work buffer is step scratch, checked against the size query | The SCF and the orthogonalizer run on it; an undeclared `syevd` is refused naming the step |
| The default cap is read after the handles exist | The table's `cap used` against a reading taken by hand after setup |
| The SCF is a condition loop inside its step's subflow | The node after `scf` sees the loop done; the trip counts against v0.0.2's |

One measurement beside them, with no bar: each molecule's budget total
against the old path's planner block for the same run (item 4 of the
design's order of work).

One-off times are printed as v0.0.3's were: benzene's `eri` at 206 forced
batches took 347 ms there with a blocking sync per batch, and the ring is
what should move it. Not a benchmark; the real question is at gly10's size,
which this prototype does not run.

## Carried from v0.0.3, and what stays open

v0.0.3's README lists fourteen things for the design to change. Item 1 is
settled above, and item 2 with it. The rest:

| v0.0.3's item | Here |
|---|---|
| 3. What a step's limit covers | **Proposed:** the limit bounds the batch (here, the ring) and not the O(N²) residents, which the cap admits or refuses; a refusal names both parts. Built as v0.0.3 has it until agreed |
| 4. `mem_budget` depends on the cap through the default limit | **Proposed:** the default limit is an absolute constant, so a dry budget is exact. Built as v0.0.3 has it until agreed |
| 5. Pinned scratch forces a sync per batch, or a ring | The ring is built for `eri`. `build_jk` keeps the blocking sync |
| 6. Setup builds a pair-scaling array to count from | **Proposed:** a counting walk in `fb-integrals-mmd`. Not built here |
| 7. The layout function called by the wrapper, once | **Proposed:** the layout as an associated type of `Step`. Built here, since the ring's slots make a double carve easy to write |
| 8. An all-GPU step's layout takes the card | Carried |
| 9. Largest-first against the bound | Carried: the table's lower-bound row, on a wider DAG than v0.0.3's |
| 10. `run` takes its lane by `&mut` | Changed by the guard: a step's inner task for a card takes that card's guard and the step's streams on it |
| 11. B written by several cards needs an `unsafe` | Carried, and listed as the memory model's second `unsafe` |
| 12. The reduction and the broadcast go through the host | Carried |
| 13. `Dims` carries per-class counts | Carried |
| 14. Logical cards as a test facility | Carried |

**Proposed, about `rust/`.** The library bindings and the recipes take the
card's guard, stream and handles together, in place of a handle and a stream
passed apart. A step with streams of its own can then not hand a recipe the
wrong one, and `blas.owner()` goes. v0.0.4 wraps the calls it makes that way
inside the prototype; the change to `fb-core-wwr` and `fb-recipes-sys`
follows if it holds.

## Not in it

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

## This page as HTML

`v0.0.4.html` beside this file is this README rendered by
`rust/docs/md2html.py`, with the hand-drawn figures of `README.figs` spliced
in (they replace the two text diagrams above). The `.md` and the `.figs` are
the source; after editing either, from this directory:

```sh
uv run --no-project --with markdown-it-py python ../../../rust/docs/md2html.py README.md v0.0.4.html README.figs
```
