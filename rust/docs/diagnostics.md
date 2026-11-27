# Rust rebuild: diagnostics

Diagnostics is everything a run says about itself besides its results: how
long each part took, what it logged, how it failed, and what the process did
when a bug surfaced. Three of the four are ports. The C++ `core` has `log`,
`error_handling` and the `diagnostics.h` macros over them; the Rust core
carries the first two (`fb-core-log`, `fb-core-error-handling`), each with
its C++-to-Rust table in its doc comment. The
fourth, timing, is new: the C++ tree has no timing infrastructure beyond two
files that pair `steady_clock` reads by hand.

The unit is a *step*, and the one function around its `run` is the
*wrapper* ([`programming_model.md`](programming_model.md), "Steps on
rustflow"). "What a step says", next, is the layer's timing, memory,
logging and failure; the sections after it are the rules under them.

## What a step says

A step says nothing about itself by default: the wrapper and the plan do
it.

**Time.** `Plan::execute` returns one `StepTime` per step: the time inside
`run`, the time inside the wrapper's sync and release, when the node
began and ended from the execute's start, the locks the wrapper took in
the order it took them, whether it found one held and so waited, and when
it held them all and when it began to release them. Two holders of one
lock therefore show disjoint held spans, and a step that waited on a GPU
says so. `fb_steps::Run` keeps them beside the wall time of setup, of
budget mode and of execute mode. These are host times: a step's `run`
enqueues, so its `run` time is its enqueueing and whatever host waits it
makes, and the device's share shows in the wrapper's sync. A step whose
loop decides on the host carries its own meter in its `Output` (the SCF's
`LoopMeter`: the loop's wall time, its host syncs and the time blocked in
them, read from `Stream::sync_meter`). `Run::write` puts them on disk
after the run as `timing.json`, beside the products: the profile that
built the binary, setup's, budget mode's and execute mode's times, and
each step's `StepTime`, every time integer nanoseconds and each lock by
its rank's name; its rustdoc is the schema. There is no stream scope in
this layer yet.

**Memory.** The budget table is the memory diagnostic
([`memory.md`](memory.md), "The final node"): logged at `info` on every
accepted plan, and the whole of a refusal's message. After an execute it
carries the plain heap's peak growth as its unbudgeted row. `Run::write`
writes the table as budget mode left it, before any execute, as
`budget.txt`: the text `--budget-only` prints.

**Logging.** A step's line names the step (`step 'eri': ...`,
`scf: ...`), by hand:
no step context is set on the thread, since a step's fan-out and the
host integrals' pool run on threads the wrapper does not own. The levels
are the table's below: a result and each SCF iteration at `info`, a batch
width of one at `warn`, a skipped step at `debug`.

**Failure.**

- **A plan over budget is a refusal, not a panic.** The final node
  returns an `Allocation` error whose message is the table, and nothing
  more is allocated ("Allocation is a planning-time error", below). Bytes
  the budget accepted and the machine refused abort at the failing call,
  as every create does.
- **A config whose steps do not close is refused when the DAG is built**,
  naming the step and the one it reads, before anything is budgeted.
- **A step's failure is a `Result`**, with `step '<name>'` pushed on its
  context (`step '<name>', mem_budget` in budget mode). The wrapper keeps
  a failed flag per node: a step that reads a failed one is skipped and
  flagged in turn, and nothing else is. The plan returns the first failure
  in its topological order, any other on that error's context, so the
  report is the same at every executor width ("One worker fails", below).
- **A step that failed is still synced.** The wrapper synchronizes every
  stream the step has and each held GPU's stream whether `run` returned
  an error or not, then releases the locks, so a failed step leaves no
  device work behind its guard.
- **A panic inside a step is the run's error** where the build unwinds:
  rustflow reports it from the wait, and the plan returns it as an error
  naming the panic's message. A release build aborts at the panic
  ("Panics and profiles", below).
- **Determinism.** Results are those of the edges. Accept or refuse, the
  scratch offsets and the table are the same every run; the order the
  steps were budgeted in is not.

## Timing and profiling

### Scopes

Timing uses RAII guards. A guard starts its clock when it is created and records
when it drops, so an early return or a `?` still closes the scope. There are two
clocks, because the host and the device disagree about when work happened:

- **Host scope.** Wall time between creation and drop. For GPU work this
  measures enqueueing, not execution, unless the scope ends in a sync.
- **Stream scope.** An event pair recorded on a stream, resolved later, when
  some sync happens anyway. It measures device time without adding a sync of
  its own.

Profiler ranges are a compile-time option. A cargo feature (`tx`) turns on a
range guard over wwr's `tx` layer (NVTX under CUDA, rocTX under HIP), the Rust
counterpart of `wwr::extension::ScopedRange`. `wwr-sys` already exposes that
layer as its own `tx` feature, which the crate leaves off today by pinning its
features to `runtime_api` alone, so the crate's feature is a forwarder to
`wwr-sys/tx` and nothing more. With the feature off, the guard is a zero-sized
no-op and compiles away.

One macro opens a scope that times and, when `tx` is on, also pushes a range:

```rust
let _scope = scope!("j build");   // bound to a name: lives to the end of the block
```

The macro exists for two reasons:

- **The binding pitfall.** `let _ = Guard::new()` drops the guard at once and
  times nothing. The macro always binds the guard to a name.
- **Thread safety.** NVTX and rocTX ranges are per-thread, so the guard is
  `!Send`. A range opened on one thread then cannot be popped on another, and
  that rule is enforced at compile time.

### The timing record

The numbers go to one place: `timing.json`, which `Run::write` puts beside
the products after the run, every time integer nanoseconds ("What a step
says", above). It says which profile built the binary, because a debug
build's timings are not comparable with a release build's, and it is a
file of its own rather than a section of a larger report, so a nightly
diff of the times is not a diff of the config too.

The record has three readers, each written down elsewhere:

- **The nightly**, which is to write the step times to the metrics
  branch, keyed by host, card, step and molecule, and flag a step that
  drifts over its rolling median ([`testing.md`](testing.md),
  "Performance").
- **The documentation site**, which is to show a step timeline drawn from
  a run's record ([`documentation.md`](documentation.md), "Diagrams and
  animations").
- **Someone reading a slow run**, who wants the step, not the kernel, that
  took the time, and whether it was the step's enqueueing or the wrapper's
  sync that waited.

## Logging

### Levels

`core::log` already provides the levels. A call site logs through `log`'s
macros and `fb_core_log::critical!`, imported from those two crates, and no
other crate re-exports them: a crate that only logs then depends on the
logging crates alone, not on whatever crate would re-export them
([`testing.md`](testing.md), "Selection: the workspace graph is the
selection graph").

The rule for each level:

| Level | For | Examples |
|---|---|---|
| `trace` | Inner-loop detail, off by default | Per batch, per kernel launch |
| `debug` | The plan's workings, for whoever is debugging the run | A step skipped, a lock waited on |
| `info` | What someone running a calculation wants to see | A step starts, SCF iterations and energies, final results |
| `warn` | A result is produced, but someone should look at it | Near-linear dependence, slow convergence, a batch width of one |
| `error` | An error that reached the place that handles it | Logged once, there |
| `critical` | The process is about to abort | `check_or_abort`, a broken contract in release, a panic in release |

The rules that go with the table:

- **Log an error once, where it is handled, not at every layer it passes
  through.** Propagation is silent, and the error carries its origin with it.
  Where it is handled is named below ("The one boundary").
- **A step's line names the step**, by hand ("What a step says", above),
  so interleaved lines from concurrent steps can be pulled apart.
- **Formatting a message costs nothing when its level is off.** `log`'s macros
  check the level before they format.
- **A warning counts when it is written.** The counter is bumped in the
  backend, so a warning filtered out by the level does not count, as the C++
  `FBF_LOG_WARN` behaves. The count is printed at the boundary, so a run that
  completed with warnings says so on its last line.

## Errors

### Values, never panics

**Expected failures are values.** Every fallible path returns `Result<T>`, and
`?` propagates it. A panic means a bug: a broken contract or an `unreachable!`.
Panics are never caught for control flow.

### GPU codes

**GPU errors bubble up.** Every wwr call's status goes through `gpu_status`
into an `Error`. The crate does this today, with the message built from the
backend's code name and description. The design keeps the backend's code itself
too, not just its text:

- **The code's name is the vendor's own** (`cudaErrorMemoryAllocation`,
  `hipErrorOutOfMemory`), so a message can be searched for in the vendor docs.
- **The numeric code and the backend are kept**, so a caller can branch on them.
- **Codes are mapped to `ErrorKind`.** A runtime code that fails becomes
  `GpuRuntime`, out-of-memory from either vendor included, in
  fb-core-device's statuses. Memory is counted in budget mode, and the
  final node's refusal is the one `Allocation` value ("Allocation is a
  planning-time error", below); a create that fails aborts, and an
  out-of-memory code met after the final node is a runtime fault like any
  other. At a kernel launch its message also names the kernel stack, which
  the budget does not count, and what to lower
  ([`gpu_layer.md`](gpu_layer.md), "Statuses").
  An unsupported configuration the backend refuses is to become
  `Unsupported`; no wrapper maps one yet. The kinds are what the exit code and
  the boundary's message are read from, so a reader learns what went wrong
  without the vendor's table.
- **Exit codes are the C++ ones**, which `ErrorKind::exit_code` already pins.

### Allocation is a planning-time error

A run allocates nothing: a step's `mem_budget` allocates through a
`Budget`, and `run` is handed none ([`memory.md`](memory.md), "The
budget"). So no `Allocation` error is met while a plan executes, and there
is no retry to attempt there; the C++ tree's rule that a failed create or
allocation aborts (`CreatePolicy`) carries over unchanged, and the
allocations the final node makes are under it: a card another process has
emptied below the accepted budget aborts there. An `Allocation` error is a
value in one place only: the final node refusing a plan over a cap, before
anything is launched. It keeps the kind, and exit code 2, because it is the
same failure reported earlier.

### Sticky faults

The vendors distinguish two kinds of runtime error, and the design does too:

- **A sticky fault** (an illegal address, a misaligned access, a trap, a
  launch that timed out) poisons the device context. Every later call on that
  device returns the same code until the process exits.
- **Everything else** (an invalid value, an unsupported configuration, an
  out-of-memory) reports once and the context stays usable.

Only the first error after a sticky fault means anything. The plan is
therefore to stop launching on a poisoned device as soon as it sees one,
and to report the fault rather than the echoes that follow it, the `Error`
recording that the fault was sticky, so the boundary can say the device
was lost rather than listing the same code once per step.

### Asynchronous errors surface at sync points

A fault in a kernel is reported at the next sync on its stream, not at the
launch that caused it. Its origin is therefore the sync point. The wrapper
syncs every stream a step has after its `run`, so a fault names the step
that enqueued it, unless the step itself synced first (a loop deciding on
the host), where it names the step and that sync.

For debugging, a switch that syncs after every launch would make a fault
name exactly one kernel, at the price of the overlap inside a step. It is
this tree's `CUDA_LAUNCH_BLOCKING`, and the compute-sanitizer runs would
turn it on, since a sanitizer report is only as good as the launch it
points at.

### Context accrues at step boundaries

`Error::origin` stays the detection site. When an error leaves a step, the
wrapper pushes `step '<name>'` (`step '<name>', mem_budget` in budget mode)
onto the error's *context*: one list the error carries beside its kind,
message and origin. `Display` prints the message and then the context,
innermost first. The reader then gets the line where it happened and the
step that was running, in one log line.

### One worker fails

A failed step fails its successors and nothing else
([`programming_model.md`](programming_model.md), "Steps on rustflow").
Concretely:

- **Nothing new is launched** downstream of the failure: a step that reads
  a failed one is skipped and flagged in turn.
- **Siblings already launched run to their sync.** Enqueued device work cannot
  be recalled, and a sibling's result may still be worth having.
- **The run returns one error: the first in topological order.** Topological,
  not first in time, so the same failure reports the same way at every
  executor width. A genuinely distinct sibling failure is pushed onto that error's context, so the boundary still logs
  once and loses nothing.

### `Drop` cannot fail

A failed destroy or free is logged at `error` and counted: the logger's
error counter, bumped in the backend as the warning counter is, counts
every error line written, and the boundary prints it. This is the one
place an error does not bubble, the same as the C++ `DestroyPolicy`: a destroy
often runs during teardown after an earlier failure, where aborting would bury
the root cause under the knock-on one.

### The one boundary

The place an error is handled is the binary's `main`, as in C++: it logs the
error once, at the error's origin and with its context, prints
`completed: N errors, N warnings` with the error and warning counters, so
a run whose teardown failed a free does not report 0 errors, and returns
`ErrorKind::exit_code`. Nothing between the detection site and `main` logs the
error. The test harness is the other boundary of the same shape: a test that
expects a failure asserts on the `Error` value it got, and a test that did not
expect one fails with the error's `Display`, which is the same line `main`
would have written.

## Panics and profiles

**Release aborts; test unwinds.** The release profile sets `panic = "abort"`,
and the dev and test profiles keep the default unwinding. The two halves have
one reason each:

- **Release aborts** because a panic is a bug, and the C++ `critical` fails the
  process at once for the same reason. Unwinding would run every `Drop` on the
  way out, which during a GPU fault is exactly the knock-on failure the
  log-on-destroy rule exists to keep out of the way of the root cause; a
  panic cannot cross the C ABI into a launch wrapper in any case; and in a
  multi-worker
  run an unwind takes down one worker's thread and leaves the others running
  against a process that is half dead. An abort loses no backtrace: the panic
  hook prints it before the process ends.
- **Test unwinds** because the contract-tripping tests
  ([`testing.md`](testing.md), "Contracts are part of the spec") are
  `#[should_panic]`, and that attribute needs the panic to be caught.

**The panic hook logs at `critical`.** It routes the panic's message and
location through `critical!`, so they reach the file sink and are flushed
before the abort, which runs no exit handlers. A panic in a test binary goes
through the same hook, so a tripped contract reads the same in a log as in the
test's output.

**A tripped contract's line carries its values.** When the contract's
condition was written with `fb_core_log`'s value checks (`le!`, `lt!`, `eq!`,
`fits_i32!`), the check that failed stored each operand's source text and
`Debug` value on its thread, and the hook takes them, clearing the slot, and
ends its line, critical or stderr, with ` [text = value, ...]`:

```text
Pre-condition of inverse_sqrt violated: count is within data: le! (count, data.len()) [count = 5, data.len() = 4]
```

A panic with no failed check behind it ends at its message. The slot is
cleared each time, so a panic carries only its own values, and a check is
written only as a whole `&&` term of a contract's condition so that no false
check is left for a later panic (`doc_check/tests/contracts.rs`).

**A panic's message is never lost.** `set_level(Off)` silences `critical`, as
in the C++ tree, and that would leave an aborting release build printing a
bare backtrace with nothing saying what panicked. So when the logger would
drop the hook's line (the level is off, or the panicking thread holds the
sinks, which it cannot lock again), the hook writes it to stderr instead, as
one line, `thread '<name>' panicked at <file>:<line>:<col>: <message>`, with
the failed check's ` [text = value, ...]` after it when there is one, before
the backtrace. At every level where the critical line is written there is no
stderr copy of it; the only other copy is std's own hook, which follows ours
in an unwinding build.

The hook is `fb-core-log`'s, installed with the logger: by `init`, which each
binary's `main` calls first, and by every other function of the crate, so a
test binary has it from its first call into the logger. Under abort it always
prints the backtrace, which is the last chance to; under unwinding, where the
panic may be caught, std's hook follows it, so a test harness still shows a
failing test's panic in its report and prints the backtrace when
`RUST_BACKTRACE` asks. Cargo builds test targets unwinding even under `--release`, so `cargo
test --release` keeps `#[should_panic]`. `doc_check/tests/panic_profile.rs`
checks the profile line and `core/log/tests/panic_hook.rs` the hook, at
level off as at info.

**A `debug_*` contract is debug-only**, as the crate already has them, so a
release build pays nothing for a check and a debug build fails loudly on
one. A check soundness rests on is always on, a `#[requires]` or a body
`assert!` ([`testing.md`](testing.md), "Contracts are part of the spec"),
and under abort its panic ends a release run through the hook like any
other.

## Open questions

- **Compiling `trace` out.** `log` has `release_max_level_*` features that
  remove a level's call sites from a release build entirely, which is cheaper
  than checking the level in a hot loop. Whether the trace lines in inner loops
  are worth that, or whether the level check is already free enough, is a
  measurement.
- **How the sync-every-launch switch is set.** An environment variable (as
  `CUDA_LAUNCH_BLOCKING` is), a config key, or a debug-build default.
