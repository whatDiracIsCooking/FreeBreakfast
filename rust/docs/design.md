# Rust rebuild: design

A feature's design is a set of answers. This document is the questions:
what a feature computes, where it runs, what it holds, whether it splits,
what it depends on, what already exists for it, what it says about itself,
what it writes, what it documents, and how its spec is communicated. Each
section has the same three parts: what the designer **answers**, where the
answer is **recorded**, and what **checks** it. The mechanics behind each
answer live in six documents of their own, cited rather than restated:

| Document | Owns |
|---|---|
| [`programming_model.md`](programming_model.md) | Steps, the MainBus and the step DAG on rustflow, the `EriManager`, the vocabulary, threads and capabilities |
| [`memory.md`](memory.md) | The budget, caps and margins, step scratch, the final node, batching |
| [`gpu_layer.md`](gpu_layer.md) | The C ABI between Rust and the kernels: the launch wrapper, the workspace query, statuses, shared types, the build; the v0.0.4 runtime: streams, events, copies and their tokens, a GPU's handles under its lock |
| [`testing.md`](testing.md) | Contracts, tiers and levels, what runs when, selection, redundancy, regression, fuzzing, sanitizers |
| [`diagnostics.md`](diagnostics.md) | Timing, logging, errors, panics |
| [`documentation.md`](documentation.md) | rustdoc, the C header's comments, equations, references, the config reference, the book, diagrams, publishing |
| [`rules.md`](rules.md) | The register: every rule these documents state, where it is stated, and what checks it, or that nothing does |

A new feature is a step, or part of one
([`programming_model.md`](programming_model.md), "Steps on rustflow"), and
"A step's design" below is each question as a step answers it. The
sections after it give each question at length. Porting a C++ module is a
feature whose answers the C++ counterpart mostly fixes; the last section
says what differs.

## Where a design lives

**Before the code, in the issue.** The issue that scopes the feature
answers every question below, in prose, with the steps it adds and the
`Output`s they read drawn as a DAG. A question the issue cannot answer is a
design question to settle before the PR, not during it.

**After the code, in the code.** Each answer lands in one artefact: a
type, a declaration, a contract, a doc comment, a golden. There is no
design document beside a module, because rustdoc renders the doc comment and
a second copy would drift ([`documentation.md`](documentation.md), "Code:
rustdoc"). The PR's description says which answers changed from the issue
and why.

## A step's design

A step is one config table, one slot, `mem_budget` and `run`
([`programming_model.md`](programming_model.md), "Steps on rustflow", "The
MainBus and its steps"). The first question is whether the feature is a
step at all: it is one only if another step reads its `Output`, its
`Output` carries over a replay, or it has a config table of its own.
Otherwise it is code inside the step that needs it, in that step's
module's crate, and the questions below are asked of that step.

| Question | A step answers | Recorded | Checked by |
|---|---|---|---|
| What it computes | Its `Output`, the `Output`s it reads, and nothing else it touches; the reference computing the same approximation, and the bar, fixed by analysis | `Step::Reads`, the tuple the DAG's edges are made from, and the `Output` type | `View::read` builds only for a declared read; one run against the reference at the bar ([`testing.md`](testing.md), "The steps' tests", "Correctness") |
| Where it runs | The host (it builds on rayon's threads and uploads, taking no lock), gpu0 alone, or every GPU; whether it calls a library, and so takes a GPU's lock; how many streams of its own it has per GPU | `Step::placement`, `Step::STREAMS`, and `Budget::locks` in `mem_budget` | A declared lock the step does not run on is refused in budget mode; `run` reaches a handle only through a guard the wrapper took |
| What it holds | Per kind, device, pinned and heap: what its `Output` owns, and its scratch, each a function of `Dims`, its options and the caps. A solver's work buffer is scratch, sized by its query. A length setup must count is added to `Dims` | `mem_budget`, which allocates the `Output` through the `Budget` and declares scratch as one `Layout`; `run` carves the same layout from its `Lease` ([`memory.md`](memory.md), "The budget", "Step scratch") | Only a `Budget` allocates; a carve past the declaration panics; the allocation record fails a `run` that takes 1 MiB from the plain heap; a dry budget of gly10 pins the lengths |
| Whether it splits | Every step over an O(N³) tensor is batched and spans every GPU, with no other path: the batch axis, the step limit that picks the width, the least bytes that hold one batch, and how the GPUs' fragments are reduced onto gpu0 ([`memory.md`](memory.md), "Batched only") | The step's limit, a config key with a constant default; the width in `mem_budget` | The limit a byte under the least refused; forced batches, a ring of one, and two GPUs against one, each at the bar |
| What it depends on, and what depends on it | Which `Output`s it reads and which steps will read its own, which decides `Output` against scratch. A product with state of its own that several steps call into is a manager, with a lock its callers declare ([`programming_model.md`](programming_model.md), "The EriManager"). In the workspace: the method code stays in its module's crate, and `fb-steps` holds the step | `Step::Reads`; the lock's `LockId`; `Cargo.toml` | `Dag::build` refuses a read of a step the config left out, a step added twice and a cycle; the workspace-graph test |
| What already exists | As below, "What already exists": a library call bound on `GpuResources`, a recipe that takes the guard, a kernel's binding in its `-sys` crate, a host integral's `_into` form | `Cargo.toml` | Review |
| What it says about itself | Usually nothing: the wrapper times it, records its locks, and names it on any error ([`diagnostics.md`](diagnostics.md), "What a step says"). A width of one, slow convergence, at `warn`; its result at `info` | The log lines in `mem_budget` and `run`, each naming the step | A test that expects a failure asserts on the `Error` value |
| What it writes | Its products' files, written on the host after execute mode from its `Output`'s host copies, never in `run`: a step's `run` allocates nothing and waits on no disk for them. The run's record, `budget.txt` and `timing.json`, is written beside them ([`diagnostics.md`](diagnostics.md), "What a step says") | One private function of its `Output` in `steps/src/write.rs`, called from `Run::write` when the config has the step's table | `steps/tests/write_device.rs` reads the products back bitwise; `steps/tests/write_host.rs` holds the dry record to the budget table |
| What it documents | As below, "What it documents": the equations, the config keys of its table, each with default and unit | The step's module doc and its options type | `missing_docs`, the doctests |

**A step's contract with the wrapper** is the part no table row holds.
`run` enqueues and the wrapper syncs, so a step makes no host sync of its
own but at the three sites [`gpu_layer.md`](gpu_layer.md) ("The v0.0.4
runtime") names: a solver's flag after its last enqueue, a ring's wait on
one slot's copy, and a loop's read of its turn's scalars. A step that fans
out across GPUs, or has streams of its own, joins them before `run`
returns. It relies on nothing its scratch held on entry, which the NaN
fill tests, and on nothing a replay did not carry in an `Output`.

## What it computes

**Answers.** The `Output`s it reads, its own `Output`, and nothing else it
touches: its boundary. The reference it will be compared against, computing
the same approximation, and the bar, fixed by analysis before the code
exists and never loosened to pass.

**Recorded.** The boundary is the step's `Reads` and `Output`
([`programming_model.md`](programming_model.md), "Steps on rustflow"); the
reference, its version and the command that produced it are committed with
the golden ([`testing.md`](testing.md), "Reference data").

**Checks.** A unit test of the method code from a fixture, never from the
steps upstream of it; the chain at the integration level; the whole run as
an example config `cargo test` runs ([`testing.md`](testing.md), "Levels:
how much of the DAG a test covers"). The comparison is one run at the bar
([`testing.md`](testing.md), "Correctness").

## Where it runs

**Answers.** The host, gpu0 alone, or every GPU, and which library
surfaces it reaches (BLAS, a solver) and which of the tree's own kernels it
launches, since that fixes the `-sys` crates it links
([`gpu_layer.md`](gpu_layer.md), "Layering", "The v0.0.4 runtime").

**Recorded.** `Step::placement` and the locks `mem_budget` declares; the
`-sys` dependencies in `Cargo.toml`.

**Checks.** The tier the test lands in: code that touches no device is
tier 3 and runs on every PR; a device case is tier 4 and runs when its
crate is affected ([`testing.md`](testing.md), "Tiers: where a check
lives", "What runs when").

## What it holds

**Answers.** Every buffer, as a function of `Dims`, its options and the
caps, per kind: device, pinned and heap, an `Output`'s or scratch
([`memory.md`](memory.md), "The budget", "Step scratch"). A pageable `Vec`
is for structured input alone (the config, the basis) and small scratch
within O(n_ao²). A kernel's share is its `*_workspace_bytes(dims)` query
and a solver's its size query ([`gpu_layer.md`](gpu_layer.md), "Scratch:
the workspace query"). A size the dimensions do not fix is allocated at a
bound setup gives ([`memory.md`](memory.md), "Batched only"). A batch width
is chosen in `mem_budget` from the step's own limit, never at run time
from free memory. Threads are a resource too: no step opens a pool or a
thread of its own, the host integrals on rayon's threads and an all-GPU
step's fan-out aside ([`programming_model.md`](programming_model.md), "CPU
workers").

**Recorded.** `mem_budget`, which allocates the `Output` through the
`Budget` and declares scratch as one `Layout`; `run` carves the same
layout from its `Lease`. Nobody writes a footprint: the budget table is
what was allocated.

**Checks.** Only a `Budget` allocates; a carve past the declaration
panics; a dry budget of gly10 pins the lengths. The thread count after a
device-free run is counted exactly
([`programming_model.md`](programming_model.md), "CPU workers"). Reading
free memory is clippy's `disallowed_methods` outside the sites that say
why.

## Whether it splits

**Answers.** Every step over an O(N³) tensor is batched and spans every
GPU: along which axis, recombined onto gpu0 by which reduce, in which
order ([`memory.md`](memory.md), "Batched only"). Either kind of reduce is
held, one GPU against many, to a tolerance the acceptance rule fixes by
analysis, and bitwise across GPU counts is not promised.

**Recorded.** The batch axis and width in `mem_budget`; the reduce in
`run`.

**Checks.** One GPU against many, at a tolerance fixed by analysis
([`testing.md`](testing.md), "Determinism and placements").

## What it depends on, and what depends on it

**Answers.** Two graphs. In the step DAG: the `Output`s it reads, with
their producers, and the steps that will read its own, which is what
decides whether a buffer is scratch or an `Output`. A computation that
must wait on the host for part of itself splits there
([`programming_model.md`](programming_model.md), "Granularity"). In
the workspace: which crates the feature uses and which will use it. Every
crate edge costs selection on every PR that touches its target, there is no
umbrella crate, and a crate that only logs must not depend on the handle
types ([`testing.md`](testing.md), "Selection: the workspace graph is the
selection graph"). So every direct dependency is one the crate's sources
name, in code or an intra-doc link: a normal one in `src/`, a build one in
`build.rs`, a dev one in `src/` or the test targets. A normal dependency
only the tests name is a dev-dependency, since every reverse dependent
inherits a normal edge. An edge no source can show, a `links` crate kept so
its link line reaches the binary or a feature switched on in another crate,
is listed with its reason in the check.

**Recorded.** `Step::Reads`; the crate edges in `Cargo.toml`.

**Checks.** `Dag::build` refuses a read of a step the config left out, a
step added twice and a cycle. The workspace graph is read by a tier-3 test
that fails when a crate's reverse dependents pass a bar, and when a crate
declares a dependency its sources do not name.

## What already exists

**Answers.** What the feature can use rather than write, looked for before
the design is settled:

- **A WarpWraps library call**, reached straight through `wwr-sys`: BLAS,
  the solvers, the collectives ([`gpu_layer.md`](gpu_layer.md), "Layering").
- **A ported crate**, and the tree's rule of looking in `core` before writing
  a helper or a constant.
- **A forked kernel library** under `rust/<module>/kernels/`, whose
  wrapper may already be there, behind its C header and `-sys` crate.
- **A C++ bridge and its kernels to fork**, hand-written or
  FelizJueves-emitted: copied into `rust/`, the launcher rewritten as an
  `extern "C"` wrapper and the emitted units emitted again
  ([`gpu_layer.md`](gpu_layer.md), "The layer already exists", "What the
  port changes"). Forking hand-written device code adds to what is kept in
  step by hand, so a feature forks what it uses and no more.
- **The C++ implementation**, as the reference design of the algorithm and
  the source of its footprint formula, which becomes `mem_budget`.

A new kernel is a FelizJueves emit or a `.cu` under `rust/`, never Rust
([`gpu_layer.md`](gpu_layer.md), "Launch shape and dispatch"); a kernel the
C++ tree lacks too is added to `rust/` alone.

**Recorded.** The dependencies the answer leaves: `Cargo.toml`, the
`-sys` crate's header, the forked files under `rust/<module>/kernels/` and
`rust/device_headers/`.

**Checks.** Nothing beyond the dependencies' own. A duplicate is caught in
review, which is why the issue names what it reuses.

## What it says about itself

**Answers.** Usually nothing: the wrapper times every step, records its
locks, and names it on any error ([`diagnostics.md`](diagnostics.md), "What
a step says"). The design says what the step logs and at which level, by
the table in [`diagnostics.md`](diagnostics.md), "Levels": a step starting
and a result at `info`, something a user should look at, such as slow
convergence or a batch width of 1, at `warn`, the wrapper's own lines at
`debug`.

**Recorded.** The log lines in `mem_budget` and `run`. Every fallible path
returns `Result`; nothing between the detection site and `main` logs the
error ([`diagnostics.md`](diagnostics.md), "Values, never panics", "The one
boundary").

**Checks.** A test that expects a failure asserts on the `Error` value;
one that did not fails with its `Display`, the line `main` would have
written.

## What it writes

**Answers.** The products the feature leaves on disk. They are written on
the host after execute mode, from the `Output`'s host copies, never in
`run` ([`diagnostics.md`](diagnostics.md), "What a step says").

**Recorded.** One private function of its `Output` in
`steps/src/write.rs`, called from `Run::write`.

**Checks.** The example config that exercises the feature, which runs
through the binary and reads what it wrote.

## What it documents

**Answers.** Which equations the code implements and where they come from,
which configuration keys it adds, and what a reader of the reference needs
that the contracts do not say.

**Recorded.** The doc comment is the spec, and every example in it compiles:
plain for host code, `no_run` for anything that touches a device,
`compile_fail` with its error code for a type-level rule
([`documentation.md`](documentation.md), "Code: rustdoc"). A published
equation is cited by number and bibliography key, one of the tree's own by
label ([`documentation.md`](documentation.md), "Equations", "References"). A
config key is a documented field on the config type, with its default and
unit, and an example config sets it
([`documentation.md`](documentation.md), "Configuration: the reference is
emitted from the type"). A derivation too long for a doc comment is a book
page ([`documentation.md`](documentation.md), "Narrative: mdBook beside the
rustdoc").

**Checks.** `missing_docs` and broken links denied; the doctests; the
bibliography keys matched against citations; the config reference
regenerated and diffed ([`documentation.md`](documentation.md),
"Publishing").

## The spec is the contracts

Every answer above is communicated to the next reader as a check the code
carries, at the cheapest tier that can express it
([`testing.md`](testing.md), "Contracts are part of the spec", "Tiers: where
a check lives"):

| Answer | Becomes |
|---|---|
| A dimension, a buffer's memory kind, a shell's angular momentum | A type: a newtype, a distinct buffer type, a `const` generic. Misuse does not build, and the `compile_fail` doctest is the test |
| What may cross between workers | `Send` and `Sync` |
| A contract: a precondition (shapes agree, an index is in range, `la <= lb`, a carve is in the declared layout), a postcondition on host-side state the function builds (a staging's cells tile its records), an invariant | `#[debug_requires]`, `#[debug_ensures]`, `#[debug_invariant]`; `#[contract_trait]` on a trait every implementation must honour. One memory safety rests on, a condition on the arguments, is the always-on `#[requires]`, its operands' values printed by `fb_core_log`'s value checks (`le!`, `lt!`, `eq!`, `fits_i32!`) |
| What a step holds | The budget table of a dry run, its lengths pinned by a test |
| The boundary and the bar | The reference comparison, one run |
| The batch axis and its reduce | One GPU against many |
| What a launch wrapper requires | The C header's `///` comment, carried into the `-sys` crate by `devtools/rust-bindings.py`. In the safe wrapper, every length or extent the launcher trusts is checked in every build, a `#[requires]` written with the value checks or, when the check is on state the body produces or its recomputation would be long, an `assert!` named in `# Panics`, and only the rules a launch's correctness alone rests on (`la <= lb`, a momentum within the table, the stream's device current) are `debug_requires` ([`gpu_layer.md`](gpu_layer.md), "The safe wrapper") |

Three rules hold for all of them. A contract is debug-only, so a release
build pays nothing and the PR gate, a debug build, has every one on, unless
soundness rests on it; then it is always on, a `#[requires]` when it is a
condition on the arguments, whose value checks carry its operands into the
panic's line, and a body `assert!` only for state the body produces, a
recomputation too long for an attribute, or a helper called once per operand,
naming it ([`testing.md`](testing.md), "Contracts are part of the spec"). The
outcome of a GPU call that can fail is not a contract; the call returns a
status or a `Result`, a failure is an expected result, and making it a
contract would turn it into a panic. A
contract's reach is every caller, but its strength is the inputs those
callers happen to pass, so it complements a dedicated test and never
replaces one. rustdoc renders each contract as its item's `§ Contracts`
section, so the doc comment never restates one
([`documentation.md`](documentation.md), "Code: rustdoc").

The capabilities a step may not have are contracts the manifest holds:
`unsafe_code` forbidden outside the core wrappers and the `-sys` bindings
with their safe launchers, spawning
disallowed outside the modules that say why
([`programming_model.md`](programming_model.md), "Capabilities are enforced
by lints"). A feature that needs more than the lints allow has found a
design question, not an exception.

## Porting a module

A module port answers the same questions, and the C++ counterpart fixes most
of them: what it computes, its reference, its dimensions, and the footprint
formula that becomes `mem_budget`. What differs:

- **The port line is the bridge header.** Everything above it, the host
  preparation, the copies, the allocation, the `.cppm` and `.cpp`, is
  rewritten as the Rust crate of the same name; the launchers and kernels
  below it are forked into `rust/<module>/kernels/`, the bridge as a C
  header, and built into the module's shared object by `rust/`'s own CMake
  project ([`gpu_layer.md`](gpu_layer.md), "What the port changes", "From
  bridge to C ABI", "Build"). The fork and the crate can be one PR or two,
  both under `rust/`; the cargo half of the gates builds the library and
  runs its tier 4, which is where a broken device link shows.
- **The doc comment carries a C++-to-Rust table** while the port lasts, with
  three kinds of row: ported, under the Rust name; not ported, because the
  language does it, naming the rule that replaces the code (`NonCopyable`,
  the move constructors, the dangling-view checks); and more than the C++,
  because Rust asks for it (a raw handle operation is `unsafe fn` and each
  owner has the safe method). The crate's `core` shows each.
- **The C++ module stays, and the trees are two.** Each calls its own copy
  of the kernels and is tested by its own suite, and the two are compared at
  the regression bars, not by sharing a binary, until the Rust pipeline runs
  every config under `examples/` that the C++ one does and matches it at
  those bars. Nothing crosses from Rust back into C++. Until then the
  hand-written device code a port forks exists twice, kept in step by hand
  ([`gpu_layer.md`](gpu_layer.md), "What the port changes"). "Done" for a
  module PR is its crate passing its tiers.

A module is ported bottom-up from `core`, one module per crate
([`gpu_layer.md`](gpu_layer.md), "What the port changes").
