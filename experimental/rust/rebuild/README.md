# rust/rebuild — free_breakfast's core, rebuilt in Rust

Can free_breakfast's C++ layers be rebuilt in Rust, one at a time, over
WarpWraps' C ABI (`wwr-sys`), keeping each one's contract? This crate is
that rebuild. Its module tree mirrors `src/`: `core::wwr` is `src/core/wwr`.
It stands alone: nothing here uses the other crates under `experimental/rust/`.

## What is here

| Module | Ports | What |
|---|---|---|
| `core::wwr::common` | wwr's `src/extension/common`, and the `src/extension/error_handling` primitives it re-exports | `ErrorCode` (`success_code` / `error_name` / `error_string`), `ErrorPolicy` (`error_policy` and `typed_error_policy` in one trait), `PolicyMap`, `gpu_check` / `gpu_check_at`, `wwr_status` as an error code, `ScopedDeviceIndex`, `kit::AbortPolicy` |
| `core::wwr::handle` | wwr's `src/extension/handle` | `BaseHandle` over a `HandleKind` (CRTP's `create`/`destroy` hooks), `DeviceBoundHandle`, `StreamBoundHandle` over a `StreamBoundKind`, the three views, and the `DeviceHandle` / `DeviceHandleStream` / `DeviceHandlePool` ladder |
| `core::wwr::runtime` | wwr's `src/extension/runtime` | `Stream`, `Event`, `MemPool` (with `MemPoolProps`), `Graph`, `GraphExec`, and `kit::DeviceHandle` (a stream and a pool on one device), each a newtype over the handle it is built on; stream capture through to a launched graph |
| `core::wwr` (`gpu_status`, the policies) | `src/core/wwr`'s `:status` and `:policy` | `gpu_status` / `gpu_status_at`; the tree's `AbortPolicy` / `LogPolicy`, `CreatePolicy` / `DestroyPolicy` and `PolicyMap`, which log through `core::log` (not wwr's stderr `kit::AbortPolicy`) |
| `core::log` | `src/core/log` | The one backend behind the [`log`](https://crates.io/crates/log) facade: spdlog's line format (local time, OS thread id, `file:line`), the seven levels, `set_level` / `should_log` / `flush`, one file sink, the warning counter, and `log_at` for a location already held |
| `core::error_handling` | `src/core/error_handling` | `ErrorKind` (with the pinned exit codes as `exit_code`), `Error` and its `#[track_caller]` factories, `Result<T>` / `Status`, `check_or_abort` |
| `core::diagnostics` | `src/core/diagnostics` (`diagnostics.h`) | `gpu_must` (`FBF_GPU_MUST`), `critical!` (`FBF_LOG_CRITICAL`), and `log`'s `trace!` ... `error!` re-exported (`FBF_LOG_TRACE` ... `FBF_LOG_ERROR`); `FBF_TRY` is `?` |

Each module's doc comment has the full C++-to-Rust table. Some C++ has no
Rust counterpart, because the language already does what that code exists
to force:

- `NonCopyable`. A Rust type is move-only unless it opts in to `Clone`.
- `policy_slot`. A zero-sized field takes no space, however many share a type.
- `nothrow_error_policy`. Rust has no `noexcept`, and a panic in `Drop`
  during unwinding aborts, which is the C++ outcome too.
- The move constructors, `release()` and the moved-from state. A Rust move
  leaves nothing behind to drop.
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
  `kit::DeviceHandle::props()` is left out: `wwrDeviceProp`'s layout differs
  between the backends.

## Contracts

Two preconditions are checked with the [`contracts`](https://crates.io/crates/contracts)
crate, in debug builds only:

- `ErrorPolicy::handle_error` is never handed `SUCCESS`. The trait is a
  `#[contract_trait]`, so every implementation needs the attribute or does
  not compile, and every policy is checked. `AbortPolicy` therefore does not
  re-check the code, as wwr's does: in a release build, a direct call with
  `SUCCESS` aborts.
- A device-bound view's `dev_idx` is -1 (unknown) or more.

Postconditions on GPU outcomes are deliberately not contracts. Whether
"the handle landed on the device asked for" holds depends on the policy: a
policy that records and returns makes failure an expected outcome, and a
contract would turn it into a panic.

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
