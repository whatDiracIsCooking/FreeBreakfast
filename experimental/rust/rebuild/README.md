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
link the library.

Nothing in ctest or either gate runs cargo. This directory is checked by hand only.

## Verified

2026-11-12, on the gfx1200 (HIP): 18 tests pass in a debug build and 15 in
release, with clippy and rustdoc clean in both. Not run on CUDA: the 3080 was
full at the time.
