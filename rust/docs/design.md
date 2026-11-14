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
| [`programming_model.md`](programming_model.md) | What a task and an edge declare, what the orchestrator derives, layouts and arenas, workers, plan-once-run-many |
| [`memory.md`](memory.md) | The arena's table and its rows' owners, lifetimes and aliasing, admission, adaptive widths, nested plans, the device and pinned blocks |
| [`gpu_layer.md`](gpu_layer.md) | The C ABI between Rust and the kernels: the launch wrapper, the workspace query, statuses, shared types, the build |
| [`testing.md`](testing.md) | Contracts, tiers and levels, what runs when, selection, redundancy, regression, fuzzing, sanitizers |
| [`diagnostics.md`](diagnostics.md) | Timing, logging, errors, panics |
| [`documentation.md`](documentation.md) | rustdoc, the C header's comments, equations, references, the config reference, the book, diagrams, publishing |

A feature is a DAG of tasks ([`programming_model.md`](programming_model.md),
"Three owners"), so most questions are asked per task and a few per DAG.
Porting a C++ module is a feature whose answers the C++ counterpart mostly
fixes; the last section says what differs.

## Where a design lives

**Before the code, in the issue.** The issue that scopes the feature
answers every question below, in prose, with the DAG drawn in the shape of
the Coulomb example ([`programming_model.md`](programming_model.md),
"Example: the Coulomb build under two placements"). A question the issue
cannot answer is a design question to settle before the PR, not during it.

**After the code, in the code.** Each answer lands in one artefact: a
layout, a declaration, a contract, a doc comment, a golden. There is no
design document beside a module, because rustdoc renders the doc comment and
a second copy would drift ([`documentation.md`](documentation.md), "Code:
rustdoc"). The PR's description says which answers changed from the issue
and why.

## What it computes

**Answers.** The inputs, the outputs, and what the task promises not to
touch: its boundary. The reference it will be compared against, computing
the same approximation, and the bar, fixed by analysis before the code
exists and never loosened to pass.

**Recorded.** The boundary is the task's declaration
([`programming_model.md`](programming_model.md), "A task declares"); the
reference, its version and the command that produced it are committed with
the golden ([`testing.md`](testing.md), "Reference data").

**Checks.** A unit test of the node from a fixture, never from the sub-DAG
upstream of it; the wiring at the integration level; the whole run as an
example config `cargo test` runs ([`testing.md`](testing.md), "Levels: how
much of the DAG a test covers"). The comparison is one run at the bar
([`testing.md`](testing.md), "Correctness").

## Where it runs

**Answers.** GPU or host-only. A host-only node is a CPU worker's: a file
write, a decision between runs of a sub-DAG, a small assembly
([`programming_model.md`](programming_model.md), "CPU workers"). A GPU task
says which library surfaces it reaches through `wwr-sys` (BLAS, a solver, a
collective) and which of the tree's own kernels it launches, since that
fixes the `-sys` features and the shared objects it links
([`gpu_layer.md`](gpu_layer.md), "Layering").

**Recorded.** The device is the task's declaration. The library surfaces
are the crate's `wwr-sys` features and its `-sys` dependencies in
`Cargo.toml`.

**Checks.** The tier the test lands in: a task that touches no device is
tier 3 and runs on every PR with fake workers; a GPU task is tier 4 and
runs when its crate is affected ([`testing.md`](testing.md), "Tiers: where a
check lives", "What runs when").

## What it holds

**Answers.** Every buffer, as a function of the dimensions, per memory kind
([`programming_model.md`](programming_model.md), "Memory kinds"):

- **Device**, carved from the worker's block, scratch or output.
- **Pinned host**, owned by the GPU worker whose copies it stages, in the
  layout with its lifetime extended to the last copy or write that reads it.
- **Pageable host**, `Vec`, not modelled and not admitted. Anything large
  goes pinned instead, where it is counted.

A kernel's share is its `*_workspace_bytes(dims)` query and a solver's its
`lwork` query, called at planning ([`gpu_layer.md`](gpu_layer.md), "Scratch:
the workspace query"). A size the dimensions do not fix declares its bound
from them, or justifies a sizing node that fixes a sub-DAG's dimensions
before it is planned; the bound is preferred
([`programming_model.md`](programming_model.md), "Memory: layouts and
arenas"). An adaptive width is chosen at planning from the room the schedule
leaves, never at run time from free memory. Threads are a resource too: host
work that would fan out is CPU nodes the planner can see, never a pool inside
a task, and the plan's thread count is a budget admitted beside memory.

**Recorded.** `layout(dims)`, the task's first function; `run(bufs, stream)`
reaches a buffer only through it. Nobody writes a footprint. An adaptive
part is `bytes(width)` and `width_max(dims)` beside the layout, bisected by
the planner ([`memory.md`](memory.md), "Adaptive widths: one planner
routine"); the rows the layout becomes, and who owns each, are
[`memory.md`](memory.md), "The table".

**Checks.** A task may ask the arena only for a buffer its layout named, a
`debug_requires` on the arena. The plan's arena size is a tier-3 golden,
committed and compared bitwise, so a layout change is a diff in the PR
([`testing.md`](testing.md), "Performance"). The thread count after a
device-free run is compared with the plan's worker total
([`programming_model.md`](programming_model.md), "CPU workers").

## Whether it splits

**Answers.** Every task is placeable on any worker of any topology, so the
question is not whether several GPUs are allowed but whether this task's own
work can be sliced: along which axis, recombined by which reduce, in which
order. Ordered gives a bitwise result at every worker count; unordered
declares a tolerance the acceptance rule fixes by analysis. Most tasks
declare no partition, and parallelism across independent sub-DAGs, the
displacements of a Hessian, the roots of a state average, the fragments of
an expansion, costs the task nothing and is the orchestrator's by placement
([`programming_model.md`](programming_model.md), "Two kinds of
distribution").

**Recorded.** The partition, if any, in the task's declaration, and the
reduce's kind and order on the edge
([`programming_model.md`](programming_model.md), "An edge declares").

**Checks.** One worker against many: streams on one card at tier 4, cards
at tier 5, bitwise for an ordered reduce and at the declared tolerance for an
unordered one ([`testing.md`](testing.md), "Determinism and placements").

## What it depends on, and what depends on it

**Answers.** Two graphs. In the DAG: the edges in, with their producers, and
the edges out, with their consumers, which is what decides whether a buffer
is scratch or an output that outlives the task, and how long each edge
lives. A computation that must wait for part of itself is two tasks and an
edge ([`programming_model.md`](programming_model.md), "Granularity"); a
replayed loop is a planned sub-DAG with a host decision node between runs
([`programming_model.md`](programming_model.md), "Plan once, run many"). In
the workspace: which crates the feature uses and which will use it. Every
crate edge costs selection on every PR that touches its target, there is no
umbrella crate, and a crate that only logs must not depend on the handle
types ([`testing.md`](testing.md), "Selection: the workspace graph is the
selection graph").

**Recorded.** The edges, each with its size, kind, order and lifetime, in
the DAG; the crate edges in `Cargo.toml`.

**Checks.** The planner's invariants over the DAG, fuzzed as a pure
function: every edge's sync precedes its first read, no two live buffers
alias ([`testing.md`](testing.md), "Fuzzing"). The workspace graph is read
by a tier-3 test that fails when a crate's reverse dependents pass a bar.

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
  the source of its footprint formula, which becomes the layout.

A new kernel is a FelizJueves emit or a `.cu` under `rust/`, never Rust
([`gpu_layer.md`](gpu_layer.md), "Launch shape and dispatch"); a kernel the
C++ tree lacks too is added to `rust/` alone.

**Recorded.** The dependencies the answer leaves: `Cargo.toml`, the
`-sys` crate's header, the forked files under `rust/<module>/kernels/` and
`rust/device_headers/`.

**Checks.** Nothing beyond the dependencies' own. A duplicate is caught in
review, which is why the issue names what it reuses.

## What it says about itself

**Answers.** Usually nothing: the orchestrator times every task on its
stream, sets the task record every log line and error carries, and
attributes an asynchronous fault to the tasks since the last sync
([`diagnostics.md`](diagnostics.md), "What the orchestrator times on its
own", "The task context", "Context accrues at task boundaries"). The design
says what the task logs and at which level, by the table in
[`diagnostics.md`](diagnostics.md), "Levels": a step starting and a result
at `info`, something a user should look at, such as slow convergence or a
width planned at 1, at `warn`, orchestration at `debug`. It says whether a
`scope!` inside the task is worth having for finer timing.

**Recorded.** The log lines and scopes in `run`. Every fallible path
returns `Result`; nothing between the detection site and `main` logs the
error ([`diagnostics.md`](diagnostics.md), "Values, never panics", "The one
boundary").

**Checks.** A test that expects a failure asserts on the `Error` value;
one that did not fails with its `Display`, the line `main` would have
written.

## What it writes

**Answers.** The products the feature leaves on disk, and when. Each write
is a CPU node that depends on a device-to-pinned copy node and feeds
nothing, so the pinned buffer is in the layout and the next iteration's GPU
work proceeds while the write runs
([`programming_model.md`](programming_model.md), "CPU workers"). A dump
switched on is a different plan, not a branch inside
a task. The timing record beside the products is the orchestrator's
([`diagnostics.md`](diagnostics.md), "The timing record").

**Recorded.** The write nodes and their edges in the DAG; the pinned buffer
in the producer's layout.

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
regenerated and diffed; the plan's diagram as a tier-3 golden
([`documentation.md`](documentation.md), "Publishing").

## The spec is the contracts

Every answer above is communicated to the next reader as a check the code
carries, at the cheapest tier that can express it
([`testing.md`](testing.md), "Contracts are part of the spec", "Tiers: where
a check lives"):

| Answer | Becomes |
|---|---|
| A dimension, a buffer's memory kind, a shell's angular momentum | A type: a newtype, a distinct buffer type, a `const` generic. Misuse does not build, and the `compile_fail` doctest is the test |
| What may cross between workers | `Send` and `Sync` |
| A precondition: shapes agree, an index is in range, `la <= lb`, a buffer is in the layout | `#[debug_requires]`, `#[debug_ensures]`, `#[debug_invariant]`; `#[contract_trait]` on a trait every implementation must honour |
| The layout | The arena size, a committed tier-3 golden |
| The DAG's shape | The emitted diagram, a committed tier-3 golden |
| The boundary and the bar | The reference comparison, one run |
| The partition and its reduce order | One worker against many |
| What a launch wrapper requires | The C header's `///` comment, carried into the `-sys` crate by `devtools/rust-bindings.py`, and the safe wrapper's `debug_requires` ([`gpu_layer.md`](gpu_layer.md), "The safe wrapper") |

Three rules hold for all of them. Contracts are debug-only, so a release
build pays nothing and the PR gate, a debug build, has every one on; a
check that soundness rests on is therefore an always-on `assert!`, not a
contract ([`testing.md`](testing.md), "Contracts are part of the spec"). The
outcome of a GPU call that can fail is not a contract; the call returns a
status or a `Result`, a failure is an expected result, and making it a
contract would turn it into a panic. A
contract's reach is every caller, but its strength is the inputs those
callers happen to pass, so it complements a dedicated test and never
replaces one. The doc comment restates each contract in a line of its own
until the `contracts` crate is known to render them
([`documentation.md`](documentation.md), "Open questions").

The capabilities a task may not have are contracts the manifest holds:
`unsafe_code` forbidden outside the `-sys` wrappers and the arena, spawning
disallowed outside the worker module
([`programming_model.md`](programming_model.md), "Capabilities are enforced
by lints"). A feature that needs more than the lints allow has found a
design question, not an exception.

## Porting a module

A module port answers the same questions, and the C++ counterpart fixes most
of them: what it computes, its reference, its dimensions, and the footprint
formula that becomes the layout. What differs:

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

The crate's README has the order the modules are taken in
([`README.md`](../README.md), "Order of work").
