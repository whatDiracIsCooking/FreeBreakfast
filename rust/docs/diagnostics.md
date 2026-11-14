# Rust rebuild: diagnostics

Diagnostics is everything a run says about itself besides its results: how
long each part took, what it logged, how it failed, and what the process did
when a bug surfaced. Three of the four are ports. The C++ `core` has `log`,
`error_handling` and the `diagnostics.h` macros over them, and the crate
already carries all three (`core::log`, `core::error_handling`,
`core::diagnostics`), each with its C++-to-Rust table in its doc comment. The
fourth, timing, is new: the C++ tree has no timing infrastructure beyond two
files that pair `steady_clock` reads by hand.

The design leans on the programming model's terms
([`programming_model.md`](programming_model.md)): a *task*, the unit the
orchestrator admits, times, tests and retries; a *worker*, which is a thread, a
stream, a GPU or a node; and the DAG's *sync points*, which the orchestrator
derives.

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

### What the orchestrator times on its own

The orchestrator opens a stream scope around every task, on the stream the
task was given, and a host scope around the whole run and around each sync
point. The task-level timing that performance regression reads therefore
needs no code in the task. Scopes inside a task are for finer detail, and are
recorded under the task that opened them.

The orchestrator also resolves the stream scopes. It owns every sync point in
the run, so each pending event pair is read at the first sync point after its
closing event, and no scope ever adds a sync of its own. A task that has no
internal host sync (the programming model's rule) can therefore be timed on
the device without being slowed down.

What `fb_core_thread_pool` does today is a part of this. It opens a stream
scope around every GPU node: an event recorded on the node's stream after the
waits its edges need and before it runs, closed by the event recorded after
it returns, which is also the event its consumers wait on, so the closing
half costs nothing new. It opens a host scope around every node's callable,
which for a CPU node is its compute and for a GPU node its enqueueing, and
one around the whole run. Every pair is read after the run, when each stream
worker has synced its stream on the way out: that is a sync point after
every closing event, if not always the first. It also opens a host scope
around every sync point: each CPU node's wait on each GPU predecessor, and
each stream worker's sync as it exits. A loop's body runs its GPU nodes
on the same events every iteration, so the loop's decision, after the host
waits that make the whole body complete, reads that iteration's stream
scopes before it readies the body again: a sync point the loop has anyway,
the first after each closing event. `scope!` inside a task is not here
yet. A failed run still returns the times of the tasks that ran; a scope
that cannot be read fails a run that otherwise succeeded, and is dropped
from one that already failed, since it would only echo that failure
(below, "Sticky faults").

### The timing record

The numbers go to one place: a timing record the run writes beside its
products, as JSON like `hessian.json` and the rest. It holds, per task, the
task's name, its worker, its dimensions, its host enqueue time and its device
time; the run's sync points and what each waited for; and the plan's arena
size, so a run's footprint and its time are read from the same file. It also
says which profile built the binary, because a debug build's timings are not
comparable with a release build's.

**It is a file of its own,** `timing.json` beside the products, not a
section of a larger run report that would also carry the plan. The plan's
text already has a home that is compared rather than measured (the
planner's golden), it changes only when the code does, and a record that
carried it would make every nightly diff a diff of the DAG too. The record
names what it timed instead: each task by its name and worker, and the
plan by its arena size and profile, which is what a reader of the times
needs from it. The second option's gain, the plan and its measurement read
together, comes from the task names, which are the plan's own.

In memory, the record is what a run returns: `fb_core_thread_pool::Record`,
from `thread_pool::run` and the planner's `Instance::run`. Today it holds
the run's result; the profile that built it (`debug` or `release`, from
`cfg!(debug_assertions)`); its wall time; per task its node's rank, its
name, its worker, its dimensions (a task's `Task::dims`, empty by default),
its host time and its device time, and for a loop's body node or decision
the iteration it ran in, one entry per iteration; every sync point, with
the worker that waited, the node it waited for, the iteration of the node
that waited and the host time of the wait; each loop's iteration count, by
its decision node; and, from
`Instance::run`, the plan's block sizes per worker and kind. `Record::to_json`
is the `timing.json` text, and the type's rustdoc is its schema: times are
integer nanoseconds, nodes their ranks, and an error keeps its kind,
message and context but not its origin, a source location in the binary
that wrote the file. Writing the file waits for a binary that writes
products, and so do the tasks' dimensions: no task names its own yet.

The record has three readers, and each is already written down elsewhere:

- **The nightly** writes the task times to the metrics branch, keyed by host,
  card, task and dimensions, and flags a task that drifts over its rolling
  median ([`testing.md`](testing.md), "Performance").
- **The documentation site** shows a task timeline the nightly draws from its
  own run's record and publishes with the task times
  ([`documentation.md`](documentation.md), "Diagrams and animations").
- **Someone reading a slow run**, who wants the task, not the kernel, that
  took the time, and whether the device or the enqueueing host was the one
  that waited.

## Logging

### Levels

`core::log` already provides the levels. A call site logs through `log`'s
macros and `fb_core_log::critical!`, imported from those two crates, and no
other crate re-exports them: `core::diagnostics` holds only `gpu_must`,
which reads a wwr status, so it depends on the handle types, and a crate
that only logs must not ([`testing.md`](testing.md), "Selection: the
workspace graph is the selection graph").

The rule for each level:

| Level | For | Examples |
|---|---|---|
| `trace` | Inner-loop detail, off by default | Per batch, per kernel launch |
| `debug` | Orchestration, for whoever is debugging the run | A task starts or ends, its placement, its layout's share of the arena, sync points |
| `info` | What someone running a calculation wants to see | A step starts, SCF iterations and energies, final results |
| `warn` | A result is produced, but someone should look at it | Near-linear dependence, slow convergence, an adaptive task planned at width 1 |
| `error` | An error that reached the place that handles it | Logged once, there |
| `critical` | The process is about to abort | `check_or_abort`, a broken contract in release, a panic in release |

The rules that go with the table:

- **Log an error once, where it is handled, not at every layer it passes
  through.** Propagation is silent, and the error carries its origin with it.
  Where it is handled is named below ("The one boundary").
- **A log line from a task names the task and its worker**, so interleaved lines
  from concurrent workers can be pulled apart. The mechanism is below.
- **Formatting a message costs nothing when its level is off.** `log`'s macros
  check the level before they format.
- **A warning counts when it is written.** The counter is bumped in the
  backend, so a warning filtered out by the level does not count, as the C++
  `FBF_LOG_WARN` behaves. The count is printed at the boundary, so a run that
  completed with warnings says so on its last line.

### The task context

The line format is the C++ one,
`[time] [tid] [level] [file:line] message`, and the thread id in it is not
enough to tell tasks apart: a worker's thread runs every task placed on that
worker over the run, the id is a number a reader would have to map to a
worker by hand, and at tier 3 the fake workers share one thread anyway.

So the orchestrator sets a thread-local *current task* record, the task's name
and its worker, for exactly the span of the task's run, and the backend reads
it when it writes a line and appends it as `[task@worker]`. A line written
outside any task carries nothing extra. The task does nothing for this, in
the same way it does nothing for its own timing: both are the orchestrator's,
because both are properties of the task as a node in the DAG rather than of
the code inside it.

The same record is what error context reads (below), so a log line and an
error from the same task name it the same way.

The record is thread-local, and that is enough because one thread runs one
node at a time: a worker is a thread the orchestrator owns, and host work
that would fan out is written as nodes the DAG can see rather than as a pool
inside a task ([`programming_model.md`](programming_model.md), "CPU
workers"). No record is ever forwarded into a pool, because no task has one.

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
- **Codes are mapped to `ErrorKind`.** Out-of-memory from either vendor becomes
  `Allocation`, not `GpuRuntime`, and an unsupported configuration the
  backend refuses becomes `Unsupported`. The kinds are what the exit code and
  the boundary's message are read from, so a reader learns what went wrong
  without the vendor's table.
- **Exit codes are the C++ ones**, which `ErrorKind::exit_code` already pins.

### Allocation is a planning-time error

The programming model makes a run allocate nothing: the arena is laid out at
planning, and admission is where a plan can be refused. So the orchestrator
never meets an `Allocation` error while a DAG runs, and it has no retry to
attempt there; the C++ tree's rule that a failed create or allocation aborts
(`CreatePolicy`) carries over unchanged, and the one allocation a plan makes,
its blocks at instantiation ([`programming_model.md`](programming_model.md),
"Blocks, plans and the process"), is under it: a card that another process
has emptied below the admitted budget aborts there. An `Allocation` error is a value in
one place only: admission refusing a plan that does not fit a worker's
budget, before anything is launched. It keeps the kind, and exit code 2,
because it is the same failure reported earlier.

### Sticky faults

The vendors distinguish two kinds of runtime error, and the design does too:

- **A sticky fault** (an illegal address, a misaligned access, a trap, a
  launch that timed out) poisons the device context. Every later call on that
  device returns the same code until the process exits.
- **Everything else** (an invalid value, an unsupported configuration, an
  out-of-memory) reports once and the context stays usable.

Only the first error after a sticky fault means anything. The orchestrator
therefore stops launching on a poisoned device as soon as it sees one, and
reports the fault rather than the echoes that follow it. The `Error` records
that the fault was sticky, so the boundary can say the device was lost rather
than listing the same code once per task.

### Asynchronous errors surface at sync points

A fault in a kernel is reported at the next sync on its stream, not at the
launch that caused it. Its origin is therefore the sync point. The orchestrator
attributes it to the tasks that ran on that stream since the previous sync,
and that attribution is where the DAG's sync points pay for themselves in
diagnostics. In release it is as narrow as the DAG makes it.

For debugging there is a switch that makes the orchestrator sync after every
task, so a fault names exactly one task, at the price of every bit of
concurrency the DAG had. It is this tree's `CUDA_LAUNCH_BLOCKING`, and the
compute-sanitizer runs turn it on, since a sanitizer report is only as good as
the task it points at.

### Context accrues at task boundaries

`Error::origin` stays the detection site. When an error crosses a task's
boundary, the orchestrator pushes the task's name and worker, read from the
same current-task record the logger uses, onto the error's *context*: one
list the error carries beside its kind, message and origin. `Display` prints
the message and then the context, innermost first. The reader then gets the
line where it happened, the task in the DAG that was running, and the worker
it ran on, in one log line.

### One worker fails

A failed node fails its successors and nothing else (the programming model's
rule). Concretely:

- **Nothing new is launched** downstream of the failure, or anywhere on a
  poisoned device.
- **Siblings already launched run to their sync.** Enqueued device work cannot
  be recalled, and a sibling's result may still be worth having.
- **The run returns one error: the first in topological order.** Topological,
  not first in time, so the same failure reports the same way at every worker
  count. A genuinely distinct sibling failure (two configs refused on two
  workers) is pushed onto that error's context, so the boundary still logs
  once and loses nothing.

### `Drop` cannot fail

A failed destroy or free is logged at `error` and counted. This is the one
place an error does not bubble, the same as the C++ `DestroyPolicy`: a destroy
often runs during teardown after an earlier failure, where aborting would bury
the root cause under the knock-on one.

### The one boundary

The place an error is handled is the binary's `main`, as in C++: it logs the
error once, at the error's origin and with its context, prints
`completed: N errors, N warnings` with the warning counter, and returns
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

**A panic's message is never lost.** `set_level(Off)` silences `critical`, as
in the C++ tree, and that would leave an aborting release build printing a
bare backtrace with nothing saying what panicked. So when the logger would
drop the hook's line (the level is off, or the panicking thread holds the
sinks, which it cannot lock again), the hook writes it to stderr instead, as
one line, `thread '<name>' panicked at <file>:<line>:<col>: <message>`, before
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

**Contracts are debug-only**, as the crate already has them, so a release
build pays nothing for a check and a debug build fails loudly on one.

## Open questions

- **Compiling `trace` out.** `log` has `release_max_level_*` features that
  remove a level's call sites from a release build entirely, which is cheaper
  than checking the level in a hot loop. Whether the trace lines in inner loops
  are worth that, or whether the level check is already free enough, is a
  measurement.
- **How the sync-every-task switch is set.** An environment variable (as
  `CUDA_LAUNCH_BLOCKING` is), a config key, or a debug-build default.
