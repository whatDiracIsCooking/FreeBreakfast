# rust/v0.0.3: packed step scratch, and batched steps on num_gpus cards

The second prototype of the redesign in `experimental/rust/v0.0.4/v0.0.4.html` ("Rust
rebuild: steps, the memory budget and rustflow"), built against the design as
it stood on 2026-11-24. It covers items 1 and 2 of that document's order of
work and nothing more:

1. **The core, device-free.** The MainBus's flow layer, the declared view, the
   budget with its counters, flag, table and final node, the heap slab, and
   what is new since v0.0.2: step scratch declared per step and packed by the
   DAG's transitive closure, the lower bound in the budget table, a lease that
   borrows for the length of `run`, one layout function per step, the NaN fill,
   and caps clamped to what the machine has.
2. **Batched steps on `num_gpus` cards.** No library stream: a stream per GPU
   step per card, one lock and one fixed 128 MiB BLAS workspace per card.
   `eri` builds B in batches of shell pairs and keeps it on the host heap;
   `jk` builds J and K of the SAD density over batches of aux functions, each
   card contracting its share, reduced onto gpu0.

The SCF and the rest of the DF-RHF chain (item 3) are not here. Throwaway: a
crate and a workspace of its own, in no gate, never to be merged into `rust/`.

**This box has one CUDA card.** Everything is written for `num_gpus` cards,
and the multi-card path is run with two *logical* cards aliased onto device 0
(`[resources] devices = [0, 0]`), each with its own lock, workspace, streams
and scratch slabs. That exercises the lock order, the per-card rows and the
reduction. It is not a multi-GPU run: real multi-card was not run, and no
number here says anything about scaling.

## Result

cc-pVDZ over cc-pVDZ-JKFIT, RTX 3080 (CUDA), 2026-11-25, one run each, in one
process per molecule (`tests/gpu.rs`). Each cell is the largest absolute
difference; the bars are `1e-10` of the largest element for B and `1e-11` for
J and K, by analysis, not bitwise.

| | water | benzene |
|---|---|---|
| AOs / aux functions / pair columns of B | 24 / 116 / 322 | 114 / 558 / 6669 |
| B against the old Rust path's B (largest \|B\| 1.33, 1.07) | 1.1e-14 | 1.4e-13 |
| J against a host contraction of the same B and D (largest \|J\| 17.0, 19.7) | 4.4e-15 | 2.8e-14 |
| K against the host (largest \|K\| 9.79, 7.21) | 4.4e-15 | 1.0e-13 |
| Batches at the default limits: `eri` pairs / `jk` aux | 6 / 1 | 6 / 1 |
| Batches with the limits forced down | 12 / 4 | 206 / 4 |
| Forced batches against the default run: B, J, K | 1.3e-15, 3.6e-15, 1.8e-15 | 1.3e-14, 1.4e-14, 1.2e-14 |
| Two logical cards, forced batches, NaN fill, against one card: B, J, K | 1.3e-15, 3.6e-15, 8.9e-16 | 1.3e-14, 1.4e-14, 9.3e-15 |
| The replay of that run: allocations before and after | 40 and 40 | 40 and 40 |

The old path is `add_packed_b` under the planner and the arena, with every
pair kept; it runs the same kernels, the same Cholesky and the same
`fb_recipes_sys::trsm`, so this row holds the batching and the layout, not the
kernels. The replay's B, J and K differ from the run before it by nothing the
bars see.

Beside those, on the same runs:

- **An over-cap plan is refused with the whole table.** Water needs
  129.72 MiB of device memory (128 MiB of it the handle workspace); with
  `device_cap_mib = 129` the final node refuses it and prints every step's
  row, the packed scratch, the lower bound and the workspace.
- **A cap the card cannot meet is clamped, not an abort.**
  `device_cap_mib = 1000000` runs; the table prints `cap as written
  976.56 GiB` beside `cap used 8.82 GiB`, the card's free memory at setup.
- **A limit too small for one batch is refused naming the step, its limit and
  its least bytes**: `step 'eri': its limit of 10.24 KiB holds no batch: the
  narrowest, of width 36, takes 529.50 KiB of device scratch per card`, and
  the same for `jk` (width 1, 16.81 KiB).

One-off times, debug build at `opt-level = 1`, a card that also drives a
display. They say what a batch costs here and are not a benchmark. Benzene's
`eri` takes 52 ms in 6 batches and 347 ms in 206: about 1.5 ms a batch, the
price of one blocking sync and one launch set per batch. On two logical cards
the same 206 batches take 191 ms. Both cards are device 0, so that is two
streams overlapping each other's waits on one card, not two cards.

## Run it

```sh
git submodule update --init --recursive
(cd rust && cmake --preset cuda && cmake --build --preset cuda -j12)   # the libraries the crates link
cd experimental/rust/v0.0.3
cargo test                                                       # 17 device-free tests, 2 doctests
cargo test --test gpu -- --include-ignored --nocapture --test-threads=1   # the 4 GPU cases
cargo run --release --example run -- examples/h2o.toml                # setup, budget mode, execute mode
cargo run --release --example run -- examples/benzene_two_cards.toml  # two logical cards on device 0
cargo run --release --example run -- --budget-only examples/h2o.toml  # the edges and the table, no device
```

The GPU cases use device 0. On this box hold both of the card's locks around
them (`flock ~/.sccache/gpu-locks/cuda.0.lock flock …/cuda.1.lock cargo test …`).

## What is here

| File | What | The design's section |
|---|---|---|
| `src/pack.rs` | The closure as bitsets, first fit largest-first then step name, the lower bound (heaviest antichain) by branch and bound. Pure functions | Step scratch; What the closure model showed |
| `src/budget.rs` | `Ledger` (caps as written and as used, atomic counters, the `over` flag, the table, the final node that packs and allocates), `Budget` (`device`, `heap`, `scratch`, `library`), `HeapSlab`, `Layout` and `Span`, the `Lease` and its one `unsafe` | The memory budget; What is budgeted, by kind; Caps |
| `src/flow.rs` | `Step`, `Slot`, the declared `View`, `Dag` and `Plan`: one step graph built into a rustflow `Taskflow` once per mode; the wrapper that acquires, runs, syncs and releases. Generic over a `Bus`, so it names no device | Steps and the MainBus; The step DAG on rustflow |
| `src/device.rs` | `CardLane` (a step's stream on one card, and its BLAS handle), the bind (`SetStream` then `SetWorkspace`), `CardResources` (the card's lock) | Streams and device resources; One handle set per card, under one lock |
| `src/bus.rs` | The TOML config, setup and `Dims`, the clamped caps, `MainBus` with one slot per step, the lock order | Steps and the MainBus; Lengths; Caps |
| `src/steps/` | `records` and `guess` (carried from v0.0.2, their tables now scratch), `eri` (B in batches of pairs), `jk` (`build_jk`, the shared function, and its one caller) | Shared code: a function over a View; Sizes: batched only |
| `tests/core.rs` | The core with five fake steps on the plain heap, and the packer on seeded random DAGs | "The core, device-free" |
| `tests/gpu.rs` | The table above, the replay, the refusals | "One step, batched, on num_gpus cards" |
| `tests/rustflow_pin.rs` | v0.0.2's two cases at the pinned rustflow, unchanged | The step DAG on rustflow |

The step DAG (`--budget-only` prints the edges):

```text
records ──> eri ──┬─> jk
guess ────────────┘
```

Benzene's budget on two logical cards with the limits forced down, as the
final node logged it in the run of the table above:

```text
memory budget: accepted
step                       device 0     pinned 0     device 1     pinned 1         heap
eri                        4.75 MiB            -     4.75 MiB            -    28.39 MiB
guess                    101.53 KiB            -            -            -            -
jk                       203.06 KiB            -            -            -   203.06 KiB
records                    4.28 MiB            -     4.28 MiB            -            -
outputs (sum)              9.33 MiB          0 B     9.03 MiB          0 B    28.59 MiB
scratch (packed)          42.22 MiB     9.00 MiB    42.51 MiB     8.90 MiB            -
  plain sum               51.71 MiB    12.61 MiB    52.01 MiB    12.41 MiB            -
  lower bound             42.22 MiB     9.00 MiB    42.51 MiB     8.90 MiB            -
blas workspace           128.00 MiB            -   128.00 MiB            -            -
total                    179.54 MiB     9.00 MiB   179.54 MiB     8.90 MiB    28.59 MiB
cap used                   3.88 GiB            -     3.88 GiB            -            -
cap as written              default            -      default            -            -
host: heap + pinned 46.49 MiB against 106.33 GiB used (default as written)
scratch device 0: packed 42.22 MiB, plain sum 51.71 MiB, lower bound 42.22 MiB (jk)
  eri                      9.47 MiB at offset 0 B
  jk                      42.22 MiB at offset 0 B
  records                 23.75 KiB at offset 0 B
scratch pinned 0: packed 9.00 MiB, plain sum 12.61 MiB, lower bound 9.00 MiB (jk)
  eri                      3.49 MiB at offset 0 B
  guess                  101.75 KiB at offset 3.49 MiB
  jk                       9.00 MiB at offset 0 B
  records                 23.75 KiB at offset 0 B
scratch device 1: packed 42.51 MiB, plain sum 52.01 MiB, lower bound 42.51 MiB (jk)
  eri                      9.47 MiB at offset 0 B
  jk                      42.51 MiB at offset 0 B
  records                 23.75 KiB at offset 0 B
scratch pinned 1: packed 8.90 MiB, plain sum 12.41 MiB, lower bound 8.90 MiB (jk)
  eri                      3.49 MiB at offset 0 B
  jk                       8.90 MiB at offset 0 B
  records                 23.75 KiB at offset 0 B
library users (hold the card's lock): eri on card 0, eri on card 1, jk on card 0, jk on card 1
```

The packed scratch of that run, card 0. `records`, `eri` and `jk` lie on one
path, so all three start at offset 0. `guess` is ordered against `jk` alone,
so in the pinned slab it sits above `eri` and under `jk`'s region. Each slab
is at its lower bound, which here is its largest single declaration, `jk`'s.

Each device cap is half of the card's free memory less half its margin: two
logical cards on one device share its reading. B is the heap column: 28.39 MiB
on the host, on no card.

The locks of that run. `eri` and `jk` each declared the library on both
cards, so the wrapper took card 0's lock and then card 1's, bound each card's
handle, ran, synchronized both streams and only then released.

## What held

Each of these is a claim of the design that the prototype ran, with the test
that holds it. `core` is `tests/core.rs`, on no card; `gpu` is `tests/gpu.rs`.

| The design says | Held by |
|---|---|
| Accept and refuse are deterministic, and the table is complete on a refusal | `core`: a plan at its cap is accepted and one byte under it refused, on 1, 2, 4 and 8 workers, six times, with one identical table that has every step's row. `gpu`: the 129 MiB case |
| A plan can pass every step and be refused at the end | `core`: the same case; every `Output` fits, the packed scratch and the workspace do not |
| A heap slab cannot grow; the host is checked before the pages are touched | `core`: the slab is a boxed slice; a heap past the host cap is counted, flagged and never allocated |
| A config that does not close is refused when the DAG is built | `core`: `step 'd' reads step 'c', which the config does not ask for` |
| The closure equals reachability by a plain search | `core`: 1000 seeded random DAGs of 1 to 60 steps, every pair |
| Every pair of overlapping scratch regions is ordered by the DAG | `core`: 2973 random DAGs with scratch, every pair of regions; and largest single ≤ bound ≤ packed ≤ sum on each |
| The lower bound is the heaviest antichain | `core`: against brute force over every subset, 500 small DAGs |
| Offsets do not depend on declaration order | `core`: 200 shuffles of the regions given to the packer; and the five steps added to the DAG in five orders, budgeted on four workers, give the same regions and the same table text |
| Unordered steps share no scratch bytes while both run; no edge is added | `core`: each fake step fills its region, sleeps, and checks it; 20 runs on eight workers, with two steps inside `run` at once in at least one. `gpu`: the two-card runs |
| The lease borrows for `run` and cannot be stored in an `Output` | A `compile_fail` doctest on `Lease` (E0521), beside the same code with the borrow's own lifetime, which compiles |
| One layout function serves `mem_budget` and `run` | Every step is written that way; `core`: a `run` that carves 64 bytes past its declaration is stopped, naming both sizes |
| Over a replay the addresses hold and the contents do not; the NaN fill shows a step that relies on them | `core`: three runs find the same address and NaN on entry each time; without the fill a step that shares with nobody finds its old value. `gpu`: the two-card runs and their replay run NaN-filled |
| A replay allocates nothing | `core` and `gpu`: the ledger's allocation count is unchanged |
| Caps are clamped to the reading; an explicit cap keeps no margin | A unit test of `Cap::resolve` and the two margins; `gpu`: the 976 GiB case |
| A refused or dry budget builds nothing and still prints exact lengths | `--budget-only` on water touches no device and prints the same rows, offsets and 129.72 MiB total as the run's table at the default limits; `core`: a dry budget counts 140 MiB and allocates nothing |
| A failed step fails its readers only | `core`: `b` fails, `d` is skipped, `a`, `c` and `e` run |
| B is built in batches of pairs, kept on the host heap, never whole on a card | `gpu`: B is a `HeapSlab` of `eri`; a card holds one batch, 36 Cartesian columns wide in the forced runs |
| The width is the step's own limit's; too small a limit is refused naming the step | `gpu`: the two 10.24 KiB cases; and limits found on a dry budget force 12 and 206 batches |
| The J and K build is a plain function over a view of B and its caller's scratch | `build_jk` holds no state; `JkStep` declares `JkLayout`'s bytes per card and passes its lease |
| A step declares per card that it calls the library; the locks are taken in card order and released after the sync | `gpu`: the acquisitions the bus recorded: card 0 then card 1 for each step, `eri`'s release before `jk`'s first lock |
| All-GPU work is reduced onto gpu0; one card is the case of one | `gpu`: one card and two logical cards run the same code and agree to the bars |
| No vendor trsm, potrs or getrs | `eri` calls `fb_recipes_sys::trsm` for the Cholesky panel and for the fit, and no other triangular solve |

## What the prototype says to change

Ordered by how much each bears on the design document.

1. **One handle set per card cannot be built on `fb-core-wwr` as merged, and
   the prototype did not build it.** `BlasHandle` keeps the `Arc<Stream>` it
   was made on and has no rebind. `fb_recipes_sys::trsm` takes its kernel
   stream from that owner (`blas.owner()`) and its gemms from the handle, so a
   handle rebound through the raw `SetStream` would run one solve on two
   streams with nothing ordering them. `BlasHandle::bind_workspace` also takes
   ownership of the workspace and synchronizes the owner's stream, where the
   design's workspace is the card's and outlives every binding. So the
   prototype holds **one BLAS handle per library-using step per card**, made in
   budget mode on that step's own stream: two per card here. What is per card,
   as designed, is the lock and the one 128 MiB workspace, which every handle
   on the card is pointed at when its step takes the lock (raw `SetStream` to
   its own stream, then raw `SetWorkspace`, every time). That is the "handle
   set per step" the design rejected for its cost on HIP. Three ways out, for
   the document to choose between: (a) `BlasHandle::rebind(&mut self,
   Arc<Stream>, workspace)`, one function, with the recipes reading the stream
   from the current binding; (b) every recipe takes its stream explicitly, as
   `cholesky_diagonal` and `symmetrize` already do; (c) no rebind at all: the
   card's handle keeps its creation stream, and a library-using step runs on
   *that* stream while it holds the lock. Under (c) the safety argument is
   unchanged (everything the holder enqueues is on one stream, synchronized
   before the lock is released), `SetStream` is never called, and the cuBLAS
   ordering hazard goes away. It gives up a step's own stream object for
   library users, which already never overlap on a card.
2. **"The default cap is read after the handles exist" depends on item 1.**
   Here the handles are made in budget mode, after the reading, so their
   memory (8 MiB each on cuBLAS by the design's handle measurements; not measured here) comes out of
   the margin, and it grows with the number of library-using steps. With one
   handle set per card made in setup, the sentence holds as written.
3. **A step's limit has to say what it covers.** `eri`'s scratch is a batch
   plus O(N²) residents that no width removes: the Cartesian metric, its half
   transform, U_aux, the records. On water the narrowest batch needs
   529.50 KiB, of which the batch itself (its two tensors and its cells, by the layout's arithmetic) is about 93 KiB. "The widest whose bytes
   fit the step's own limit" was implemented over the whole region, so a
   limit below the residents is a refusal however narrow the batch. The
   document should say whether the limit bounds the batch or the step's
   scratch, and the refusal's "least bytes" should name the two parts.
4. **`mem_budget` is not a function of the dimensions and the options alone:
   the cap enters.** The default limit is a fraction of the card's cap, so the
   width, and with it every scratch row, depends on the reading. A dry budget
   has no card to read, so its defaulted cap is none and its table shows one
   batch per class, not what the run will use. Either the device-free budget
   takes a cap, or the document says the table is exact only under an explicit
   cap or limit.
5. **Pinned scratch forces a sync per batch, or a ring.** A batch's cells are
   written into pinned scratch and its result is downloaded into pinned
   scratch; the next batch may not touch either until the card has finished
   with them. With one tenant of each, that is a blocking sync per batch:
   about 1.5 ms a batch on benzene, 6.7 times the 6-batch build at 206
   batches. Overlapping batches needs a ring of pinned tenants and device
   events inside the step, and the ring's depth is then part of the step's
   layout and of what its limit holds. The design covers this in one clause
   ("a streamed pass still needs device events"); it deserves a paragraph in
   "Sizes: batched only", since every batched step meets it.
6. **Setup builds a pair-scaling array to count from.** The lengths `eri` and
   `records` need come from `fb_integrals_mmd::record_tables`, which builds
   the bra, ket and record-prefix arrays of every shell pair on the plain
   heap. That is what "setup counts every length that scales; it builds none
   of those arrays" rules out. It is small (three integers a pair), but the
   rule as written needs either a counting walk in `fb-integrals-mmd` or the
   tables as a step's `HeapSlab`.
7. **The layout function should be called by the wrapper, once.** A `Span`
   is consumed when it is carved, so the tenants of one layout cannot be
   handed out twice; two calls of the layout function can. Nothing in the
   types stops a `run` from calling it twice and holding two slices of the
   same pinned bytes. Making the layout an associated type of `Step`, built
   once by the wrapper and handed to both `mem_budget` and `run`, closes that
   and removes the one place the two could disagree. What catches a
   disagreement today is a panic when `run` carves past the declaration.
8. **An all-GPU step's layout is per card, and gpu0's differs.** `jk` needs D
   and fragments of J and K in scratch on every card but gpu0, which reads and
   writes `Output`s. The width is then held to the largest card's layout (169
   aux on two cards, 170 on one, at the same limit). "A row per card" is
   right; the size function takes the card.
9. **Largest-first is at the bound less often than the model says, on other
   sizes.** On this prototype's random DAGs (2 to 16 steps, an edge with
   probability 0.25, sizes uniform over 1 to 40 units where the model's are
   log-uniform over 1 to 2048) first fit largest-first is at the lower bound
   on 2467 of 2973 (83.0%), 1.453 times the bound at worst, against the
   model's 98.1% and 1.281. Safe on every one. The lower-bound row is what
   shows it, and it argues for keeping the door open to the best of four
   orders. On the real graphs here it says nothing: they are near-chains, and
   each slab is at its bound, the largest single declaration.
10. **`run` needs its lane mutably, and its inner tasks need a way to split
    it.** `BlasHandle` is `Send` and not `Sync`, so a step that fans out over
    cards hands each inner task one card's lane by `&mut`. The view therefore
    no longer carries the lane; `run` takes `(out, view, lane, scratch)`. The
    fan-out here is scoped threads, one per card, and not a rustflow subflow.
11. **B written by several cards needs an `unsafe`, or a staging pass.** B is
    pair-fastest, so two cards' batches interleave in memory and no slice cut
    parts them. The prototype scatters each batch through a raw pointer, with
    "a batch owns its columns" written beside it. This is a second `unsafe`
    of the memory model that the design's enforcement table does not list.
12. **The reduction and the broadcast went through the host.** D reaches the
    other cards, and their fragments reach gpu0, by a download, a host sum
    and an upload, with no collectives. With one card nothing is copied. The
    document says where the result lands, not how it gets there.
13. **`Dims` carries per-class counts.** A batch never spans two classes, so
    the batch list, and `eri`'s cells bound, need each class's pair and record
    counts. `Dims` is a small vector longer and no longer `Copy`.
14. **Logical cards are worth keeping as a test facility.** `devices = [0, 0]`
    ran the lock order, the per-card slabs and the reduction on a one-card
    box for the cost of a list in the config. It cannot show contention
    between real cards, a wrong current device (both are device 0), or any
    scaling.

## Not run

- **Real multi-card.** One physical card. Two logical cards share its
  context, so a launch on the wrong card's context cannot fail here, and the
  per-inner-task device selection is exercised only as code that runs.
- **HIP.** CUDA only. The `SetStream` then `SetWorkspace` order, rocBLAS's
  hard need for the workspace and the hipBLAS pool are untested here.
- **One handle set per card**, as item 1 says, and any solver handle: no
  step here calls `syevd`, so a solver work buffer in step scratch and the
  binding's size check are not exercised.
- **The SCF and the DF-RHF chain** (item 3 of the order of work): `jk` runs
  once, over the SAD density. No condition loop, no energy, no comparison
  with PySCF beyond what the old path's B carries.
- **Contention on a card's lock.** `eri` and `jk` are ordered, so neither
  ever waited for the other. No two unordered library users exist in this DAG.
- **The NaN fill's power on the device.** The device regions are filled with
  `0xFF` before `run` and the results hold, which says no step here reads
  stale device scratch. No step was written to fail under it on a card; the
  failing case is `core`'s, on the heap.
- **The enforcement checks**: the clippy bans, the allocator record, the
  unbudgeted-heap row of the table.
- **The default limit fraction.** `DEFAULT_LIMIT_FRACTION` is 1/8 of the
  card's cap, a placeholder: the design leaves the value open, and nothing
  here measures it. At these sizes it never binds.
- **Screening.** Every shell pair is kept. B is budgeted at the unscreened
  bound, which is then also what is used.
- **A size where B does not fit a card.** Benzene's B is 28 MiB. The path is
  the one a larger molecule would take, and it was not timed at one.
- **Batches kept on the cards between passes**, the sentinel tests, packing
  `Output`s, several nodes: proposed or set aside in the design, not built.

## Crudest form, spelled out

- `eri`: one batch in flight per card; the transpose into B's rows is the
  host's scatter; every card computes its own metric and factor; no screen.
- `jk`: K from D with no occupied factor; every pass uploads all of B; a
  host copy of J and K is kept in the `Output` for the tests.
- `records`: no screening bounds; built on every card, one card after
  another.
- Setup: a neutral closed-shell molecule, built-in bases.
- `Budget` has no pinned `Output` kind: every pinned byte here is scratch.

## This page as HTML

`v0.0.3.html` beside this file is this README rendered by `rust/docs/md2html.py`,
with the hand-drawn figures of `README.figs` spliced in (the step DAG there
replaces the text one above). The `.md` and the `.figs` are the source; after
editing either, from this directory:

```sh
uv run --no-project --with markdown-it-py python ../../../rust/docs/md2html.py README.md v0.0.3.html README.figs
```
