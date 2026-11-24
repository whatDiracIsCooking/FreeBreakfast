# rust/v0.0.2: the rustflow redesign, prototyped

A prototype of the redesign proposed on 2026-11-20 ("Rust rebuild: steps, the
memory budget and rustflow", a design note that is not in the tree): a
calculation as a DAG of **steps** over one shared **MainBus**, each step one
function that counts and allocates (`mem_budget`) and one that runs, the same
DAG run in two modes over one [rustflow](https://github.com/whatDiracIsCooking/rustflow)
`Executor`. No planner, no arena, no table of lifetimes.

It carries **DF-RHF of a closed-shell molecule on one card**, every step in its
crudest form (the full slab, no batching, no enforcement lints), and is held
against the old Rust path's energy in the same process. Throwaway: a crate and
a workspace of its own, in no gate, never to be merged into `rust/`.

## Result

cc-pVDZ over cc-pVDZ-JKFIT, RTX 3080 (CUDA), 2026-11-20, one run each.
`|ΔE|` in hartree; the bar is the old test's, 1e-9.

| Molecule | AOs / aux | E (new) | \|new − old\| | \|new − PySCF\| | Iterations new / old |
|---|---|---|---|---|---|
| water | 24 / 116 | −76.026751140543 | 4.3e-14 | 2.8e-13 | 11 / 11 |
| benzene | 114 / 558 | −230.721729270921 | 1.1e-13 | 1.2e-12 | 11 / 11 |
| ethylene (#865's geometry) | 48 / 232 | −77.867545824336 | 1.4e-14 | 3.4e-13 | 50 / 62 |

Water and benzene follow the old loop iteration for iteration. Ethylene is the
case whose DIIS subspace goes singular: both loops restart, the new one by
emptying the subspace and the old one keeping its newest pair, so the counts
differ and the energy does not.

Beside the energies, on the same runs:

- **A replay allocates nothing and lands on the same energy** (to 1.3e-13):
  `Plan::execute` called a second time writes into the buffers budget mode
  made; the ledger's allocation count is unchanged.
- **Twenty whole runs at eight workers agree** to 2.3e-13, any worker running
  any step; `eri` and `core_hamiltonian`, which have no edge between them,
  held the card at the same time in all twenty.
- **A device cap of 4 MiB refuses the plan before anything runs**, with the
  whole table: 4.69 MiB asked.
- **`[scf]` without `[eri]` is refused when the DAG is built**: `step 'scf'
  reads step 'eri', which the config does not ask for`.

Warm wall time of execute mode (the replay) against a second run of the old
path, release build, one run each on a card that also drives a display: water
19.0 ms against 24.8 ms; ethylene 131 ms (50 iterations) against 176 ms (62);
benzene 257 ms against 224 ms. The new K is built from D with no occupied
factor, `n³ n_aux` where the old one is `n² n_occ n_aux`, which is benzene's
gap. These say the new layer adds no cost that shows at this size; they are not
a benchmark.

## Run it

```sh
git submodule update --init --recursive
(cd rust && cmake --preset cuda && cmake --build --preset cuda -j8)   # the libraries the crates link
cd experimental/rust/v0.0.2
cargo test                                                # 11 device-free tests, 2 doctests
cargo test -- --include-ignored --nocapture --test-threads=1   # and the 8 GPU cases
cargo run --release --example run -- examples/h2o.toml    # setup, budget mode, execute mode
cargo run --release --example run -- --budget-only examples/h2o.toml   # the DAG and the table, no device
```

The GPU cases use device 0. On this box hold both of the card's locks around
them (`flock ~/.sccache/gpu-locks/cuda.0.lock flock …/cuda.1.lock cargo test …`).

## What is here

| File | What | The design's section |
|---|---|---|
| `src/flow.rs` | `Step`, `Slot`, the declared `View`, `Dag` and `Plan`: one step graph built into a rustflow `Taskflow` once per mode. Generic over a `Bus`, so it names no device | Steps and the MainBus; The step DAG on rustflow |
| `src/budget.rs` | `Ledger` (caps, atomic counters, the `over` flag, the per-step table, the final node), `Budget`, `DeviceBuf`, `PinnedBuf`, `HeapSlab` | The memory budget; What is budgeted, by kind |
| `src/device.rs` | `ForkStream`, `DeviceResources` behind its mutex, `syevd` held against its declaration | Streams and device resources; BLAS and solver workspace |
| `src/bus.rs` | The TOML config, setup, `Shared`, `MainBus` with one slot per step | Steps and the MainBus; Caps |
| `src/steps/` | `harmonics`, `records`, `overlap`, `core_hamiltonian`, `orthogonalizer`, `eri` (the `EriManager`), `guess`, `scf` | Example: the DF-HF setup DAG |
| `tests/core.rs` | The core with fake steps and no device | "The core, device-free" |
| `tests/rustflow_pin.rs` | At the pinned rustflow: a condition loop inside a subflow, and a multi-condition looping a body of two sources with no entry node, on one worker and four | The step DAG on rustflow |
| `tests/water.rs` | The energies above, the replay, the stress loop, the refusals | "The DF-RHF chain as steps" |

The step DAG (`--budget-only` prints it as DOT):

```text
harmonics ─┬─> overlap ──> orthogonalizer ─┐
records  ──┼─> core_hamiltonian ───────────┤
           └─> eri ────────────────────────┼─> scf (a condition loop)
guess ─────────────────────────────────────┘
```

Water's budget, as the final node logs it:

```text
step                          device 0      pinned 0          heap
core_hamiltonian             14.16 KiB          96 B             -
eri                           3.68 MiB    223.81 KiB             -
guess                         4.50 KiB      4.50 KiB             -
harmonics                     4.69 KiB      4.69 KiB             -
orthogonalizer               13.69 KiB           4 B             -
overlap                      14.07 KiB             -             -
records                     193.11 KiB      2.21 KiB             -
scf                         117.77 KiB      9.21 KiB             -
workspace (max)             669.65 KiB             -             -
  orthogonalizer        asks 669.65 KiB of device 0's workspace
  scf                   asks 669.65 KiB of device 0's workspace
total                         4.69 MiB    244.51 KiB           0 B
cap                           7.51 GiB      1.00 GiB     57.02 GiB
```

## What held

Each of these is a claim of the design that the prototype ran, with the test
that holds it.

- **The declaration is the edge, with no macro.** A step names what it reads as
  a tuple of steps, `type Reads = (OverlapStep, EriStep)`. The builder makes
  the edges from that tuple, and `view.read(EriStep)` compiles only for a step
  in it. A read with no declaration is error E0277, "this step reads a step its
  `Reads` does not name", with the caret on the argument (a `compile_fail`
  doctest in `flow.rs`).
- **Budget mode in parallel is deterministic.** Twenty refusals on four
  workers print the same table, every step in it, counting past the cap
  (`a_plan_over_its_cap_is_refused_with_the_whole_table_every_time`). A plan
  whose steps all pass can still be refused by the workspace at the final node,
  and the table says who asked (`a_workspace_can_refuse_a_plan_every_step_passed`).
- **One description, two modes.** `Dag::add(step)` makes both bodies; neither
  mode has a graph of its own.
- **The SCF is a condition loop of the graph.** A step that sets `LOOPS` and
  returns `Next::Again` becomes entry, body, condition and exit nodes; each
  trip goes through the wrapper (device current, the slot's lock, the sync).
  The entry node is needed, as the design said: a condition's edge is weak, so
  a body with no other strong predecessor would never start.
- **A finished node is finished device work.** No event between steps and no
  chain edge anywhere; the only events are the two inside `ForkStream::library`.
- **A failed step fails its readers and nothing else**
  (`a_failed_step_fails_its_readers_and_nothing_else`); a task's panic comes
  back from `wait` as the run's error.
- **A library call is held against its declaration.** `DeviceResources::syevd`
  repeats the size query and compares it with the workspace the final node
  allocated; an eigensolve the step did not declare is refused naming the step,
  before anything is enqueued (`an_undeclared_eigensolve_is_refused_naming_the_step`).
  On CUDA this is natural, because the wwr wrapper takes the work buffer.
- **`'static` bodies need no lifetime shim.** Every body captures an
  `Arc<MainBus>`.

## What the prototype says to change

Ordered by how much each bears on the design document.

1. **BLAS-heavy steps barely use their own stream.** `orthogonalizer` and `scf`
   enqueue everything on the device's library stream, and `eri` everything but
   its class kernels, each holding the device's mutex for the whole of it. The
   kernels that sit between BLAS calls (`inverse_sqrt`, `symmetrize`, the
   Cholesky tile, the DIIS solve, the packed gather) went on the library stream
   too, since putting them on the step's stream would cost a fork and a join
   each. So "the library stream carries every BLAS and solver call" became
   "and every kernel between them", and two BLAS-heavy steps on one card would
   serialize both on the host and on the device. This is the open question
   "does a manager own its own streams and handles", and it applies to steps as
   much as managers. The split the code suggests: a BLAS handle per step, on the
   step's own stream, and only the solver, which is what has a workspace, on a
   shared one. Not measured: nothing here contends.
2. **Build-only scratch has nowhere to go.** 68% of `eri`'s device bytes on
   benzene (237 of 347 MiB) are intermediates of the build that nothing reads
   once B exists; B and the J/K scratch are the other 111 MiB. An `Output`
   lives for the run, and a replay needs the scratch again, so the step cannot
   free it. "Reuse within a step is the step's own" covers two buffers sharing
   bytes at once, not a buffer that is dead after `run`. A second lifetime
   inside a step (allocated in `mem_budget`, counted in the peak, released at
   the end of `run`, re-taken on a replay) would recover it and is the smallest
   thing that would.
3. **The wwr wrappers are `!Send`.** `Stream`, `BlasHandle` and `SolverHandle`
   hold `Rc`s, which fit one thread per device. With any worker running any
   step, `ForkStream` and `DeviceResources` each needed an `unsafe impl Send`
   (and `Sync` for the first), argued from the slot's lock and the mutex.
   `fb-core-wwr` wants `Send` handles before the real port.
4. **Every safe kernel binding takes arena rows.** The prototype calls seven raw
   `extern` launchers directly (19 `unsafe` blocks, 7 of them launches in the
   steps) and re-derives the host staging the old tasks keep private: the
   records' device layout, the metric's pair table, the packed cells. The
   design said the bindings change; the measure is that all of
   `fb-integrals-sys`'s task layer is rewritten, not adapted, while its host
   tables (`RecordTables`) and `fb-recipes-sys`'s slice-taking functions carry
   over unchanged.
5. **`mem_budget` is a function of more than the dimensions.** Sizes come from
   the per-class record counts and the kernels' table lengths, which are host
   tables. `Shared` therefore carries `RecordTables`, and `eri` builds its own
   tables twice: once in `mem_budget` for their lengths and once in `run` for
   their contents, because a refused or dry budget hands back empty buffers and
   so `mem_budget` cannot fill what it allocates. Either the host tables are
   setup's products, or `mem_budget` may fill a buffer it was really given.
6. **A dry budget cannot count the workspace.** The size queries need live
   handles, so the device-free budget (`Backing::Dry`, `--budget-only`) reports
   every step-owned byte exactly and the workspace as zero
   (`the_budget_is_the_same_dry_and_real_but_for_the_workspace`).
7. **A step that reads a value back syncs itself.** The solver's flag, the
   Cholesky's, and the SCF's decision are each read on the host inside `run`,
   after the step's own `sync`. The wrapper's sync then finds nothing to wait
   for (under 25 µs summed over 50 trips). Where a step only enqueues, the
   wrapper's sync is where its device time lands (0.9 ms for benzene's
   `core_hamiltonian`). So the step-end sync costs the SCF nothing beyond the
   one wait per iteration its decision needs anyway.
8. **The manager's lock does not protect its scratch on the device.**
   `EriManager::build_jk(&self, …)` takes an internal mutex, as the design
   predicted, but holds it only while enqueueing. What keeps two callers'
   builds apart on the card is that both enqueue on the one library stream.
   If managers get streams of their own (item 1), the scratch needs an order on
   the device as well, an event or a scratch per caller.
9. **The replay reset fell to the step.** `ScfStep` resets its counters and its
   DIIS cursor when it is entered after having converged: the "reset mode"
   candidate, and the only one that needed no new node. The cost is that the
   loop's host state (iteration count, last energy, the cursor) lives in the
   `Output` beside the buffers, and what carries over between replays is
   implicit in the step's code.
10. **The lane is the wrapper's, not a field of the `Output`.** The design put
    the step's `ForkStream` in its `Output`. Here the wrapper makes it in
    budget mode, keeps it beside the `Output` in the slot, and hands `run` a
    reference, so it can sync without the step's cooperation.
11. **"No slot is read empty" is a runtime assert.** A slot is
    `RwLock<Option<(Output, Lane)>>`, filled by budget mode. The types say a
    read is declared; that it is filled is checked when the guard is taken.
12. **Host libraries that return a `Vec` break the third enforcement rule.**
    `fb_sad::sad_density` allocates the guess inside `run`. Rule 3 ("nothing
    large is allocated in `run` on the plain heap") needs the host layer to
    write into slices, as the old tasks' pinned rows made it do.
13. **Adding a step touches four places**: its module, its slot in `MainBus`,
    its `slot()` accessor, and `plan()`. The view's diagnostic cannot name the
    undeclared step (the compiler knows it only as `_` there), and `Reads`
    stops at eight steps.

## What it did not test

- **More than one card, and HIP.** `Placement::AllGpus` and a lane per device
  are written and never run; with one card, making the device current per node
  cannot fail. No collectives.
- **Inner tasks.** Every step is one rustflow task that enqueues in sequence:
  no subflow, no pipeline, no module. The step-against-task granularity
  question is untouched.
- **Contention.** No two steps here want the library mutex at once for long,
  so its cost under load, and what an idle or blocked worker costs, are
  unmeasured. Whether `sync` sleeps or spins was not looked at.
- **Batched paths and per-step limits.** `eri` refuses past the dense B's
  32-bit index and has no narrower mode. B is the dense square, twice the
  packed layout the old tree streams: gly10 would need 15.5 GB for B alone.
- **The enforcement checks**: the clippy bans and the allocator record.
- **The host heap kind** outside the fake steps: no real step budgets a
  `HeapSlab`.
- **The default margin.** `DEVICE_MARGIN` is 768 MiB by fiat.
- **The sum against the fold.** The planner's peak for the same runs was not
  read, so the ratio the design asks for is still unmeasured.

## Crudest form, spelled out

- `eri`: B dense, `n_ao² × n_aux`; no Schwarz screen; every build intermediate
  kept; the fit by the old tree's blocked Cholesky and a trsm.
- `scf`: closed shell only; K from D with no occupied factor; DIIS without the
  old loop's kept-pair restart; no ROHF, no conventional integrals, no ion
  rescale of the guess (a charged molecule is refused at setup).
- `records`: no screening bounds.
- Setup: one molecule, built-in bases, `num_gpus = 1` in effect.

## This page as HTML

`v0.0.2.html` beside this file is this README rendered by `rust/docs/md2html.py`,
with the hand-drawn figures of `README.figs` spliced in (the step DAG there
replaces the text one above). The `.md` and the `.figs` are the source; after
editing either, from this directory:

```sh
uv run --no-project --with markdown-it-py python ../../../rust/docs/md2html.py README.md v0.0.2.html README.figs
```
