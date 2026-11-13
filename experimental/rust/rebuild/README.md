# rust/rebuild — free_breakfast's core, rebuilt in Rust

Can free_breakfast's C++ layers be rebuilt in Rust, one at a time, over
WarpWraps' C ABI (`wwr-sys`), keeping each one's contract? This crate is
that rebuild. Its module tree mirrors `src/`: `core::wwr` is `src/core/wwr`.
It stands alone: nothing here uses the other crates under `experimental/rust/`.

## What is here

| Module | Ports | What |
|---|---|---|
| `core::wwr::common` | wwr's `src/extension/common`, and the `src/extension/error_handling` primitives it re-exports | `ErrorCode` (`success_code` / `error_name` / `error_string`), `wwr_status` as an error code, `ScopedDeviceIndex` |
| `core::wwr::handle` | wwr's `src/extension/handle` | `BaseHandle` over a `HandleKind` (CRTP's `create`/`destroy` hooks), `DeviceBoundHandle`, `StreamBoundHandle` over a `StreamBoundKind`, each with a fallible `try_new`; the three views; the `DeviceHandle` / `DeviceHandleStream` traits |
| `core::wwr::runtime` | wwr's `src/extension/runtime` | `Stream`, `Event`, `MemPool` (with `MemPoolProps`), `Graph`, `GraphExec`, each a newtype over the handle it is built on, whose constructors abort on failure; stream capture through to a launched graph |
| `core::wwr` (`gpu_status`) | `src/core/wwr`'s `:status`, and the rule its `:policy` encodes | `gpu_status` / `gpu_status_at`, and `gpu_log_error_at`, what a `Drop` does with a failed destroy |
| `core::log` | `src/core/log` | The one backend behind the [`log`](https://crates.io/crates/log) facade: spdlog's line format (local time, OS thread id, `file:line`), the seven levels, `set_level` / `should_log` / `flush`, one file sink, the warning counter, and `log_at` for a location already held |
| `core::error_handling` | `src/core/error_handling` | `ErrorKind` (with the pinned exit codes as `exit_code`), `Error` and its `#[track_caller]` factories, `Result<T>` / `Status`, `check_or_abort` |
| `core::diagnostics` | `src/core/diagnostics` (`diagnostics.h`) | `gpu_must` (`FBF_GPU_MUST`), `critical!` (`FBF_LOG_CRITICAL`), and `log`'s `trace!` ... `error!` re-exported (`FBF_LOG_TRACE` ... `FBF_LOG_ERROR`); `FBF_TRY` is `?` |
| `core::thread_pool` | nothing: the C++ pipeline runs on one thread | The run's workers (`docs/programming_model.md`, "CPU workers"): `Plan` (nodes added in topological order, each placed on the CPU pool or one stream worker), `run` (one scope, `n_cpu` CPU workers and a thread per stream worker, readiness counts, failure skipping descendants, the lowest-ranked error returned, a panic re-raised rather than hung), and the `Backend` the edges leaving a stream wait through: `Device` (a stream per worker, an event per GPU node) or `HostOnly` for device-free tests. `core::log::TaskScope` tags each node's lines `[task@worker]`, and `Error::context` carries it out. `clippy.toml` makes it the one place that spawns |

Each module's doc comment has the full C++-to-Rust table. Some C++ has no
Rust counterpart, because the language already does what that code exists
to force:

- `NonCopyable`. A Rust type is move-only unless it opts in to `Clone`.
- The error policies (`error_policy`, `policy_map`, `policy_slot`,
  `gpu_check`, the C++ tree's `CreatePolicy` / `DestroyPolicy`) and wwr's
  `kit`. A C++ constructor cannot return a failure, so what it does with
  one is injected as a type parameter. A Rust constructor returns `Result`,
  and this tree's answer is fixed: a wrapper's constructor aborts on an
  `Err` (`check_or_abort`), and a `Drop` logs a failed destroy
  (`gpu_log_error_at`). `core::wwr::common`'s doc has the whole reasoning.
- The move constructors, `release()`, the moved-from state and the
  null-handle sentinel. A Rust move leaves nothing behind to drop, and a
  failed create yields an `Err`, not an empty handle.
- `device_handle_pool` and `kit::DeviceHandle`, the rung a buffer allocates
  from a stream-ordered pool through. The arena allocates one plain block
  per plan ([`memory.md`](docs/memory.md), "The two kinds").
- Dangling views. A view borrows the handle it came from.

- The `FBF_LOG_*` macros, and the rule that formatting goes only through
  them. `log`'s macros check the level before they format.
- `LocatedFormat` and the `err_*` helpers. A `#[track_caller]` factory
  captures its caller even with a format string, so
  `Err(Error::io(format!(...)))` does what `err_io(...)` does.
- `FBF_TRY`. It is `?`, which moves the error's origin through unchanged.

Two things go the other way, where Rust asks for more than the C++ does:

- wwr's borrow-safe operations (`sync`, `record`, `launch`, ...) are free
  functions on a raw handle. A destroyed handle there is a use-after-free in
  the runtime, so the raw forms are `unsafe fn`, and each owner has methods
  that call them safely.
- `wwrMemPoolProps` reaches the C ABI as a bare `void*`. `MemPoolProps`
  mirrors the vendors' layout, which CUDA 13 and ROCm share (88 bytes,
  identical offsets; checked with a C program against both headers).
  `wwrDeviceProp`, whose layout differs between the backends, is not
  mirrored.

## Contracts

One precondition is checked with the [`contracts`](https://crates.io/crates/contracts)
crate, in debug builds only: a device-bound view's `dev_idx` is -1
(unknown) or more.

Outcomes of GPU calls are deliberately not contracts. A call that can fail
returns a status or a `Result`, and a failure is an expected result, not a
broken promise; a contract would turn it into a panic.

## Build and test

```bash
cmake --preset experimental            # or hip-experimental
cmake --build build-experimental --target wwr.capi.runtime_api
cd experimental/rust/rebuild
cargo test                             # debug: the contract tests run too
cargo test --release                   # the contract tests are skipped
```

`build.rs` finds `libwwr_capi_runtime_api.so` under
`build-experimental/deps/WarpWraps/src/capi`. Set `FB_RUST_BUILD_DIR` to use
another CMake build directory, for example
`FB_RUST_BUILD_DIR=../../../build-hip-experimental`. Only the `runtime_api`
surface of `wwr-sys` is linked.

The GPU tests use device 0 and need room on it for a context. The host-only
tests (the fake handle kind, error codes and contracts) need no GPU but still
link the library. Use a separate `CARGO_TARGET_DIR` per backend: `build.rs`
bakes the library's path into the rpath.

Allocating from a memory pool is tested in `tests/pool_allocation.rs`, a
binary of its own. Under ROCm 7.2.4 on the gfx1200,
`hipMallocFromPoolAsync` / `hipFreeAsync` on one thread beside other
threads' stream work intermittently deadlocked inside `libamdhip64` (about
one run in two of the unit tests); that file has the evidence. Cargo runs
each test binary as its own process, so they never overlap.

The logger is process-global, so `tests/log.rs` is a binary of its own too,
and its in-process checks are one test. Its abort and exit cases re-run
that binary as a child and read the child's stdout: what they check (the
critical line flushed before `abort`, the buffered tail flushed at exit) is
what survives the process ending.

Tests run in parallel, so a capture test must not use global capture mode.
A global capture forbids other threads' unsafe calls, and on the 3080 it
made a neighbouring pool's create abort or its destroy segfault.

Nothing in ctest or either gate runs cargo. This directory is checked by hand only.

## Design

`docs/` holds the design. [`design.md`](docs/design.md) is the questions a
feature's design answers and where each answer lands; the six beside it
hold the mechanics: the [programming model](docs/programming_model.md),
[memory](docs/memory.md), the [GPU layer](docs/gpu_layer.md),
[testing](docs/testing.md), [diagnostics](docs/diagnostics.md) and
[documentation](docs/documentation.md).
They describe the finished shape; the section below is the order the tree
gets there in.

## Order of work

The crate today is `core` alone, in one package, and nothing runs cargo in
the gates. The first steps, in order, each one PR:

1. **The section-citation check**, as the first tier-3 test
   ([`documentation.md`](docs/documentation.md), "Narrative: mdBook beside
   the rustdoc"). It is a few lines, needs no card, and the documents have
   drifted once already.
2. **The workspace.** One crate per module and `core` split
   ([`testing.md`](docs/testing.md), "Selection: the workspace graph is the
   selection graph"), done while the crate is four thousand lines rather than
   forty.
3. **The cargo half of the gates**, tiers 1 to 3 for a diff under the crate
   ([`testing.md`](docs/testing.md), "What runs when").
4. **One task through the whole model**, before any module is ported: the
   overlap, which `experimental/rust/s_overlap` already launches, or the
   Coulomb build. Declared with a layout, laid out in an arena, planned once
   and run on one stream, with its arena size committed as a golden and its
   stream scope timed. Everything in the documents leans on the
   orchestrator, and this is where the model meets its hard cases, an
   adaptive width and a data-dependent size, on one task rather than twenty.
5. **The per-module shared objects**, a C++-tree change
   ([`gpu_layer.md`](docs/gpu_layer.md), "Build").

Then the port, bottom-up from `core`, one module per crate, with the C++
tree green throughout ([`gpu_layer.md`](docs/gpu_layer.md), "What the port
changes").

## Verified

2026-11-12, on both cards: on the RTX 3080 (CUDA) and on the gfx1200 (HIP),
33 unit tests and the 2 pool-allocation tests pass in a debug build, and 30
and 2 in release. The debug unit-test binary passed 20 runs in a row on
each card, and the pool binary 20 on each. Clippy, rustdoc and `cargo fmt
--check` are clean.

2026-11-12, `core::log`, `core::error_handling`, `core::diagnostics` and
`core::wwr`'s status and policies, on the RTX 3080 (CUDA) only: 39 unit
tests, the 5 log tests and the 2 pool-allocation tests pass in a debug
build, and 36, 5 and 2 in release. The log binary passed 20 runs in a row.
Clippy, rustdoc (`-D warnings`) and `cargo fmt --check` are clean. Nothing
added calls a backend-specific path, but it has not been run on the
gfx1200.

2026-11-13, the error policies, `kit` and the pool rung removed, on the RTX
3080 (CUDA) only: 44 unit tests, the 5 log tests, the pool-allocation test
and the thread-count test pass in a debug build, and 42, 5, 1 and 1 in
release. Clippy, rustdoc (`-D warnings`) and `cargo fmt --check` are clean.
Not run on the gfx1200.
