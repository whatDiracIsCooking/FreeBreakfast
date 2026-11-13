# Rust rebuild: testing

Testing absorbs contracts. A contract is a check the code carries with it, and
every test that reaches the code runs that check. The rest of this document is
where a check lives (tiers), how much of a DAG a test covers (levels), what
runs when and how a run holds a card, how a PR's run is chosen from the diff,
what keeps the tests few, and the checks no unit test provides: regression,
fuzzing, the C++ boundary's parity checks and sanitizers.

The design has one goal throughout: **the most defects caught for the least
test resource**, where the resource is lines of test code, runtime on a card,
and maintenance. Every rule below is justified by that trade, and most of
them come from a cost the C++ tree's `test/` pays today ("What the C++ tree
teaches", below).

## Contracts are part of the spec

A function's, type's or module's contract is part of its spec. It is written
with the [`contracts`](https://crates.io/crates/contracts) crate, as the crate
already does: `#[debug_requires]`, `#[debug_ensures]`, `#[debug_invariant]`, and
`#[contract_trait]` on a trait, so that every implementation is held to the
trait's contract. The attributes keep the tedious checks (shapes agree, an index
is in range, a matrix is square) out of the function body. The body then says
what the function does, and the attributes say what it assumes and promises.

The rules already in the crate carry over:

- **Contracts are debug-only.** A release build pays nothing for them.
  `cargo test --release` skips the tests that exist only to trip a contract.
- **The outcome of a GPU call is not a contract.** If a GPU call can fail, it
  returns a status or a `Result`, and failure is an expected result, not a
  broken promise. Making it a contract would turn it into a panic.

A contract's reach is every caller, but its strength is only as good as the
inputs those callers happen to pass. It complements a dedicated test; it does
not replace one.

## Tiers: where a check lives

Push each check to the cheapest tier that can express it:

| Tier | Catches | Examples |
|---|---|---|
| **1. Compile time** | Misuse that should not build | A view borrows its handle, so it cannot outlive it. A host buffer and a device buffer are different types. Dimensions are newtypes (`Nbf`, `Naux`), so they cannot be swapped. `Send`/`Sync` decide which data may cross between workers (the programming model's sharing question, enforced). `const` generics fix angular momentum. |
| **2. Contract** | Preconditions and invariants the type system cannot express | Shapes agree, an index is in range, a task asks the arena only for a buffer its layout named |
| **3. CPU only** | Logic that touches no device | The DAG planner, placement, memory admission, layouts and the arena's offsets, config parsing, error codes. Multi-worker orchestration runs with fake workers, as the fake `HandleKind` already does for handles. |
| **4. Single GPU** | One task's numbers on one device | One J build against a reference. Several workers as streams on one card. |
| **5. Multi GPU** | What only real separate devices show | Peer transfers, broadcast and reduce across cards, per-device placement |

Each tier down costs more and runs less often. A check that can move up a tier
should.

Moving down a tier does not mean a host copy of every kernel. The tree is
GPU-first. Tier 3 is for code that never needed a device in the first place,
mainly the orchestrator, not for CPU twins of the GPU tasks.

Tier 3 is also where the expensive tools pay: fuzzing, mutation testing and
Miri all run thousands of times per case, and only a case that never touches a
card can afford that. The more of the orchestrator stays device-free, the more
of it those tools can reach. The arena, the thread pool and the planner build
with no wwr library for this (their `device` feature off,
[`README.md`](../README.md), "Build and test"), so a tier-3 run needs no CMake
step and nothing in it calls into a C library, which Miri cannot follow.
Miri itself has not been run on them yet: it needs a nightly toolchain the
image does not carry.

## Levels: how much of the DAG a test covers

| Level | Covers | DAG terms |
|---|---|---|
| **Unit** | One function, type or task | One node |
| **Integration** | Tasks wired together | A sub-DAG: the edges, the fork and the sync |
| **End to end** | A whole calculation from its config | Traversing the whole DAG |

A feature is tested in that order: its nodes first, then their wiring, then the
whole run. The tiers and levels are independent axes. A unit test of the
planner is tier 3. A unit test of the J kernel is tier 4.

**The examples are the end-to-end level.** `cargo test` compiles everything
under `examples/`, and one case runs each example config through the binary,
as `test/examples/` does for the C++ tree. An example that stops running is
then a failed test, not a stale document, and the end-to-end level needs no
configs of its own.

Two rules keep the levels apart. They answer where the C++ tree's tests spend
most of their card time:

- **A unit test of a node takes its inputs from a fixture, never by running
  the sub-DAG upstream of it.** A test of the Fock build is handed a density,
  not an SCF. Small inputs are committed goldens; large ones are computed once
  (below) and checked against a committed hash. Running the upstream nodes is
  what the integration level is for, and a unit test that does it retests
  them for free, which is to say for a card's time.
- **One cache per test process serves the fixtures that must be computed.** A
  molecule, its basis and its integrals are built once per binary and shared.
  In the C++ tree every fixture builds its own, and the static caches that
  patch this hold device memory across suites, which is how a case that
  passes alone OOMs in the nightly (#1248). The cache here is one, explicit,
  and its footprint is a number the binary can print.

## What runs when

| Run | Tiers | Runs |
|---|---|---|
| **PR** (`test.py push`, `test.py pr`) | 1, 2, 3 in full; 4 for the affected crates | `cargo test` of every crate in debug, so every contract is on. The tier-4 representatives of the crates the diff affects ("Selection", below). |
| **Nightly** | 4 in full; performance | Every GPU case. A release build, so the timings it records are real ("Performance", below). |
| **Weekly** | sanitizers, fuzz, mutants, 5 | compute-sanitizer over tier 4. Miri and ASan over tier 3. The fuzzers and property tests with their large case counts. `cargo-mutants` over tier 3. Tier 5 on a multi-GPU VM. |

Three rules go with the table:

- **The PR gates keep the C++ rule: under five minutes on a busy card.**
  Tiers 1 to 3 are seconds, so the budget is spent on tier 4, and selection
  decides how much of it a PR spends.
- **Debug and release are two runs, not one redundant one.** The PR gate
  wants every contract on, so it builds debug. The timing records want the
  code that ships, so the nightly builds release. The same case in both
  profiles answers two different questions.
- **A skip is visible, never a silent pass.** Whatever mechanism selects the
  GPU tiers in cargo ("Open questions"), a case that does not run says so, in
  a list the gate prints, as `FBF_REQUIRE_GPU()` and the `[gpu]` lines do
  today.
- **One case gets a bounded share of the gate.** The C++ rule is a case past
  about 1.5 s on an idle RTX 3080 runs in the nightly only, unless it is its
  family's representative (`test/support/slow_test.h`). The same bar holds
  here, read from the recorded times, and the nightly flags every case that
  breaks it. A five-minute budget with no per-case bar is spent by the first
  case that grows.
- **A run is one backend.** A build is CUDA or HIP, as today, and a gate runs
  on one card. The HIP cover is the cross-backend compile check on every PR,
  and a HIP nightly when that card is up. Production numbers come from the
  CUDA run, since a bare HIP run has no stall watchdog.
- **Format and lint are the commit hook's, as for C++ and Python.**
  `rustfmt --check` and `clippy` with warnings denied run beside
  clang-format, cmake-lint and ruff at commit and in CI's `lint` job.
  rustfmt's defaults are the style; `docs/style.md` gains a Rust section the
  first time a rule is needed that rustfmt does not hold.
- **The cargo half of the gates starts with tiers 1 to 3, not with the first
  tier-4 test.** Those tiers are seconds and need no card, so `test.py push`
  and `test.py pr` run `cargo test` and `cargo doc` for a diff under the
  crate, through the diff dispatch they already have for `.py` and C++.
  Today `rust/` is outside both gates (`NOT_GATED` in `devtools/test.py`), so
  until that lands the Rust tree is checked by whoever runs `cargo test` by hand, and a PR says so
  ([`README.md`](../README.md), "Order of work").

## Running on a card: the process model

`cargo test` runs a binary's cases on parallel threads by default. That is
right for tier 3 and wrong for tier 4: several GPU cases in one process would
share the card, and the C++ tree measured what sharing costs. The CASSCF case
takes 2.3 s alone and 158 s beside the exchange suite, and two runs on one
card OOMed it. So:

- **Each crate's tier 4 is one test binary, run with one test thread.** The
  cases run serially, in one process, on one card that the run holds whole.
  That is the C++ gates' model since PR #499: one gtest process per GPU
  binary, because a process per case paid the device's context init on every
  case, and that was most of the batch's wall clock.
- **Workers within a case are the case's own.** A tier-4 test of several
  workers as streams starts them itself, and a run's threads are scoped to
  the run and joined before it returns
  ([`programming_model.md`](programming_model.md), "CPU workers"), so a case
  leaks no thread into the next. The harness never runs two cases
  concurrently on a device.
- **`cargo-nextest` is used for its filtersets, timeouts and skip list, not
  for its process-per-test default.** Its per-test timeout is the stall
  watchdog, and a timed-out case is named, as the `[gpu] hung in <case>` line
  does today. Its retries stay off: a case that passes on the second try is
  a flaky case, and a flaky case is a blocker with an issue filed at once,
  which is the tree's rule for every test failure.
- **A failure reproduces alone, in one command.** Every failure prints what
  recreates it: the seed, the shapes, the task and the worker. The output
  names the one command that runs that case by itself. A failure that needs
  the whole batch to reproduce is a shared-state bug, and the one cache per
  process ("Levels") is the only shared state a binary may hold.

## Selection: the workspace graph is the selection graph

The gates run only what the diff can have broken. The C++ tree tried this and
could not: **#988** replayed 50 merged PRs against a dependency graph built
from the module scanner's `.ddi` files, the header deps and the CMake File
API, and the median PR still selected 73% of the fast GPU batch. The one
representative per family that the gate runs instead costs 53% for every PR.
The causes were structural, not the tool's:

- **Adding a file touched a `CMakeLists.txt`, which ran everything.** 17 of
  the 23 run-everything PRs were there only for that.
- **Umbrella imports flattened the graph.** `free_breakfast.core` is imported
  by 154 of the 157 GPU test files, and `pipeline.config` by 48.

Rust removes the first cause and makes the second a design decision, so the
decision is written down here:

- **One crate per module, mirroring `src/`, and a task's GPU tests live in
  the crate that owns the task.** Cargo's unit of build and test is the crate,
  so the crate graph is the only graph a selector can read, and "the affected
  crates' tier-4 tests" is then exactly the set to run. A module declaration
  lives in a `.rs`, so adding a file touches no build file. Adding a
  dependency touches one `Cargo.toml`. A `Cargo.lock` change affects
  everything, which is honest. The tree is that workspace: `core` split as
  the next rule says, `core/wwr/` the crate `fb-core-wwr` and so on, one per
  module ([`README.md`](../README.md), "What is here").
- **No umbrella crate.** No prelude, no crate that re-exports the tree. `core`
  is split so that a crate which only logs does not depend on the handle
  types. An edge that exists for convenience costs every PR that touches its
  target, forever.
- **The graph is tested.** A tier-3 test reads the package graph with
  [`guppy`](https://crates.io/crates/guppy) and fails when a crate's reverse
  dependents pass a bar, except the few crates the test names as allowed to
  be global. A flattening then shows up in the PR that causes it, not in a
  replay a year later.
- **The selector is [`determinator`](https://docs.rs/determinator/) over
  `guppy`**, which marks the packages a git diff affects. Two of its rules
  matter here. A changed file that lies in no package marks everything
  changed unless a path rule covers it, and it does not read a build script's
  `rerun-if-changed`, so the C++ library that `build.rs` links needs a path
  rule mapping its sources to the crate that wraps them. Those rules are the
  one hand-written part of the graph, and the graph test above checks them
  against the C++ tree's layout.
- **Representatives are the fallback, not the gate.** Where selection cannot
  place a change (a tool, a shared fixture, the C++ library's build), the run
  is every crate's representatives, as today. Selection runs more, never
  fewer.
- **The claim is replayed before it is believed.** Per-case times are recorded
  from the first commit, to the metrics branch as the C++ gates record theirs,
  so that #988's replay can be repeated over this tree's merged PRs with the
  same bar: the median PR's selection costs at most half the fast GPU batch.
  Until that replay passes, the gate is the representatives.

## Redundancy

A test earns its place by catching a defect no other test catches. The rules:

- **A test's doc comment names the defect it alone catches.** If that sentence
  cannot be written, the test is folded into the one that already catches it.
- **Doc examples are tests** (`cargo test` runs them), so a `tests/` case that
  only shows an API call again is deleted.
- **No host and GPU twins.** The C++ tree builds a host and a GPU binary per
  directory and places each case by hand (#1037 is the lint for the ones
  placed wrong). Here a case's tier is declared once, on the case.
- **Redundancy is measured, not argued.**
  [`cargo-mutants`](https://mutants.rs/) mutates the code and records which
  tests kill each mutant. A test whose killed set is a subset of another's is
  redundant by construction, and a surviving mutant is either a test to add or
  code that nothing needs. It runs weekly over tier 3; a mutant round over a
  GPU tier would take days, which is one more reason the orchestrator stays
  device-free.
- **Coverage is a guide, never a bar.** `cargo-llvm-cov` says which lines no
  test reaches, which is where to point the mutants first. A coverage
  percentage as a gate buys tests that reach lines without checking them.

### What the C++ tree teaches

The numbers at the commit this document was written against (a66eeffc,
November 2026):

| | Lines | Cases |
|---|---|---|
| Hand-written `src/`, generated kernels excluded | 110,725 | |
| `test/` | 119,459 | 1,594 gtest, 369 pytest |
| Cases behind `FBF_SKIP_IF_FAST_TESTS()` | | 273, in 82 files |
| Cost of one representative per family | | 53% of the fast GPU batch |

Test code outnumbers the source it tests. That alone proves nothing, but the
habits behind it cost card time on every run: fixtures that rebuild their
inputs, step tests that rerun the steps before them, host and GPU binaries
that each need their own placement, and a selection that cannot beat 53%
because the graph is flat. Each rule above is one of those habits, refused.

## Regression

### Correctness

Results are compared against references (PySCF, using the same approximation,
such as density fitting over the same auxiliary set) at fixed tolerances
derived by analysis. Each comparison is one run, and the tolerance is never
loosened to make a test pass. A reference value is committed with the
reference's version and the command that produced it, so it can be regenerated
and the diff reviewed.

#### Reference data

- **Goldens live beside the crate that reads them**, small enough to commit.
  A large reference is generated by a script under `tools/`, and the test
  holds its hash and the command, so it is one run away.
- **The reference's version is pinned.** PySCF comes from `uv.lock`, like
  every Python dependency, so a golden regenerated on another machine is the
  same golden.
- **The reference and the tree compute the same thing.** PySCF's built-in
  basis sets differ from the tree's BSE JSON in contraction, so a comparison
  loads the tree's JSON into PySCF rather than naming the set. The
  contraction norms are not renormalized, so an AO-basis comparison goes
  through the overlap (D S D, not D). The constants are the tree's
  (CODATA 2018). Each of those cost the C++ tree a round of debugging a
  "wrong" result that was a different calculation.

### Determinism and placements

The programming model promises that a planned DAG gives the same answer
under every placement: bitwise where every reduce is ordered, and within a
declared tolerance where one is not. That promise is a test here:

- **One worker against many.** At tier 4 the many are streams on one card;
  at tier 5 they are cards. An ordered reduce compares bitwise, and an
  unordered one at its declared tolerance, and nowhere else is a tolerance
  used to absorb a placement.
- **Seeds are fixed and printed.** A test that draws random shapes or data
  seeds its generator from a constant, or from a value it prints on failure.
- **Atomics do not loosen a bar.** A result that varies run to run because a
  kernel reduces with atomics meets the bar its analysis fixed, or the kernel
  gets an ordered reduce. The bar is never raised to make the variation fit.

### Performance

- **Time.** The orchestrator opens a stream scope around every task
  ([`diagnostics.md`](diagnostics.md), "Timing and profiling"), so each task's device
  time is recorded per run with no code in the task. The nightly, built in
  release, writes them to the metrics branch keyed by host, card, task and
  dimensions. A task is flagged when it passes a fixed fraction over its
  rolling median on that key; the nightly fails on it, and a PR run never
  times anything, since it is a debug build on a shared card.
- **Memory is a tier-3 golden.** A run allocates nothing, so a plan's arena
  is its peak, and the arena's size is a deterministic function of the DAG
  and its dimensions. It is therefore a committed number per plan, compared
  bitwise, with no card involved. A planning change that grows the footprint
  is a diff in the PR, read in review, rather than a measurement taken later.

## Fuzzing

The best fuzz target in the tree is the planner. It is pure, CPU-only and
has invariants that are short to state:

- no two buffers live at once alias in the arena;
- every buffer's offset honours its alignment, and a sub-DAG's offsets lie
  inside the region its parent reserved for it;
- every edge's sync precedes its first read;
- the plan is deterministic in its inputs;
- the same DAG plans to the same footprint for one worker and for many,
  up to the transport buffers the placement adds;
- every task's placement satisfies its declared device, and the plan's
  worker total is within the budget it was admitted against.

These are [`proptest`](https://crates.io/crates/proptest) properties over
random DAGs with fake workers: a few hundred cases per PR, and many more
weekly. Property tests run on stable and cost no card, so they are a tier-3
unit test with a generator in place of a hand-picked input.

[`cargo-fuzz`](https://rust-fuzz.github.io/book/) (libFuzzer, nightly only)
fits the parsers: the TOML config, the basis JSON, the status `TryFrom`, and
the compatibility layer's types as they arrive from C. It runs weekly, with
the corpus committed so a crash reproduces as a plain test.

Kernels do not fuzz. A kernel run is milliseconds on a card, so a campaign of
thousands is not a fuzz, and the invariants of an integral are its reference
values. The equivalent at tier 4 is a property test over a handful of random
shapes against the reference, with the shapes' seed printed on failure.

## The C++ boundary

The layer between Rust and the kernels ([`gpu_layer.md`](gpu_layer.md)) is
tested at the cheapest tier that can see each of its promises:

- **Shared types match, at tier 1.** Every `#[repr(C)]` mirror of a header
  type carries `const` assertions of its `size_of`, `align_of` and each
  field's `offset_of`, and the C++ side has the matching `static_assert`s.
  A layout that drifts does not build on either side.
- **The status lists match, at tier 3.** The C enum of launch statuses and
  Rust's `TryFrom` are compared member by member, so neither side can add a
  code the other lacks. A value Rust does not know is an error, and that path
  has a test that sends one.
- **Bindings do not drift.** The bindings are committed
  ([`gpu_layer.md`](gpu_layer.md), "Build"), so a reviewer sees a changed
  signature in the diff, and a tier-3 test regenerates them and fails on a
  difference.
- **Each wrapper's workspace query is checked against its kernel.** A
  `*_workspace_bytes(dims)` that under-reports is an out-of-bounds write that
  compute-sanitizer finds weekly; a tier-4 test that hands a wrapper exactly
  the bytes it asked for, under `memcheck`, finds it the week it is written.
- **The wrappers' own refusals are tested from Rust.** An angular momentum the
  kernels do not cover must come back as `ErrorKind::Unsupported` through the
  whole stack, not as a launch failure.

## Sanitizers

The C++ tree has both sanitizers in `docker/compose.yaml` and neither on a
schedule. Here they are weekly from the start.

- **compute-sanitizer is a launcher**, so it wraps a cargo test binary as it
  wraps a ctest one. `CARGO_TARGET_<triple>_RUNNER` is the counterpart of
  `CMAKE_TEST_LAUNCHER`, and two of the C++ tree's lessons carry over:
  `--error-exitcode=1` is load-bearing, since the tool exits 0 by default
  with errors reported, and a flag the runner silently ignores is a run that
  checked nothing. `memcheck` is the weekly run over tier 4. `racecheck`
  sees only shared-memory races and is far slower, so it and `initcheck` and
  `synccheck` run on demand over a named case. HIP has no counterpart; the
  cross-backend compile check stays its cover, as today.
- **ASan on Rust is not yet stable.** `-Zsanitizer=address` is nightly only;
  a Tier 2 target that lets a stable compiler use it has merged, and the
  [stabilization](https://github.com/rust-lang/rust/pull/123617) is open.
  Its reach also shrinks by design: the C++ launch wrappers never allocate
  ([`gpu_layer.md`](gpu_layer.md)), so host-side memory errors can
  only be in their argument handling, and device-side errors are
  compute-sanitizer's. `ASAN_OPTIONS=protect_shadow_gap=0` carries over,
  because the shadow gap's collision with the driver's address reservation
  is the driver's, not the language's.
- **Miri is the sanitizer for unsafe Rust.** It runs tier-3 tests under an
  interpreter that catches undefined behaviour in `unsafe` code. It cannot
  cross FFI, so every wwr call is out of its reach, and that is the right
  division: Miri for the orchestrator's unsafe code, ASan for the C++ side,
  compute-sanitizer for the device.

## Open questions

- **Selecting GPU tiers in cargo.** How does a test declare that it needs tier
  4 or tier 5? Options are a cargo feature, `#[ignore]` with a runner flag, or
  a runtime skip with a reason, like `FBF_REQUIRE_GPU()`. A skip must be
  visible, never a silent pass. `cargo-nextest`'s filtersets, or a custom test
  harness, are the two routes that give the gate one list of what skipped.
- **Tier 5 on this box.** The two cards here run different backends, so no
  single build sees two devices. Can several workers on one card stand in for
  most of tier 5, with true multi-GPU tested only on a multi-GPU VM?
- **Contracts in release.** Should any contract always be checked? A cheap
  `requires` at an API boundary might be worth keeping in release. An `ensures`
  that has to read back GPU output would force a sync, so it should stay
  debug-only.
- **Determinator's path rules.** The rules that map the C++ library's sources
  to the crate that wraps them are hand-written. Who regenerates them when a
  module is added, and does the graph test catch a stale one?
- **ASan on stable.** Adopt the Tier 2 target now, or wait for the
  stabilization and run ASan on nightly Rust until then?
- **The mutant bar.** What kill rate does a crate need, and is a surviving
  mutant a test to add or code to delete? The first mutant rounds decide.
