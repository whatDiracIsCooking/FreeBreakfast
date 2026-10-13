# Error Handling

Central error type and result aliases for the project. Uses C++23
`std::expected` to propagate typed errors with automatic source location
capture. It also turns WarpWraps' GPU error codes into those errors, and holds
the policy for what a failed GPU call inside a wwr RAII wrapper does.

## Module

`free_breakfast.core.error_handling` — target `free_breakfast::core::error_handling`,
also re-exported by `free_breakfast.core`. Three partitions, each building on
the one before:

| Partition | Exports |
|-----------|---------|
| `:error` | `Error`, `ErrorKind`, `Result<T>`, `Status`, `LocatedFormat`, the `err_*` helpers, `to_exit_code` |
| `:gpu_status` | `gpu_status`, `check_or_abort` |
| `:gpu_policy` | `AbortPolicy` / `LogPolicy`, `CreatePolicy` / `DestroyPolicy` / `PolicyMap` |

## Usage

```cpp
import free_breakfast.core.error_handling;
import free_breakfast.core.types; // int32_t, size_t

namespace free_breakfast {

Result<int32_t> parse_count(int32_t raw) {
  if (raw <= 0) {
    return err_config("count must be positive");
  }
  return raw;
}

// Source location is auto-captured at the call site
Status allocate_buffer(size_t n) {
  if (n == 0) {
    return std::unexpected(Error::allocation("zero-size allocation"));
  }
  return {};
}

} // namespace free_breakfast
```

The `err_*` helpers take a `std::format` string and its arguments directly,
so there is no `std::format(...)` to wrap around the message:

```cpp
return free_breakfast::err_io("cannot open '{}'", path.string());
```

The format string is checked at compile time, as `std::format`'s is, and a
plain message with no arguments still works. The call-site location rides on
the format string's own type (`LocatedFormat`), because a defaulted
`std::source_location` cannot follow a parameter pack. One consequence: a
caller that must forward somebody else's location (as `gpu_status` does)
cannot use `err_*` and writes `err(Error::gpu(msg, loc))` instead.

`err_*` factories construct; they do not log. Log a value-tier error once, at
the boundary that turns it into an exit code — see `../diagnostics/`.

## ErrorKind

`ErrorKind` is deliberately flat and small: a dispatch tag for `to_exit_code`
and for callers that branch on failure category, not a deep hierarchy. Every
enumerator has a matching `Error::<name>` factory, a matching `err_<name>`
helper that returns it pre-wrapped in `std::unexpected`, and a distinct
`to_exit_code` arm.

| Kind | Factory | Exit code | Meaning |
|------|---------|-----------|---------|
| `InvalidConfig` | `config` | 1 | Well-formed input that is semantically wrong (bad shape, out-of-range value, missing required key) |
| `Allocation` | `allocation` | 2 | Host or device memory could not be obtained |
| `GpuRuntime` | `gpu` | 3 | A GPU runtime or library call failed (launch, driver, handle), under either backend — not a property of the *answer*, see `Numerical` |
| `IO` | `io` | 4 | A file, stream, or parse operation failed |
| `Divergence` | `divergence` | 5 | An iterative process blew up — the iterate itself went bad, see `MaxIterationsExceeded` |
| `MaxIterationsExceeded` | `max_iterations_exceeded` | 6 | Still converging normally, ran out of iteration budget — recoverable by raising the cap |
| `Unsupported` | `unsupported` | 7 | Well-formed input requesting functionality that is deliberately not implemented — distinct from `InvalidConfig`, where the input itself is wrong |
| `Numerical` | `numerical` | 8 | A computation completed but produced a mathematically invalid result (e.g. a singular matrix) — distinct from `GpuRuntime` |

**The exit codes are pinned.** A shell script branching on `$?` is an API
consumer, so a new kind extends the mapping rather than renumbering it. The
host test suite asserts every number.

### Parse errors stay folded into IO

There is no `Parse` kind. The exit-code contract is about recovery action, not
failure stage: a malformed file and an unreadable file both mean "the run
cannot proceed with this file as given", and splitting them would add a case
every `switch` over `ErrorKind` has to handle without changing what a caller
does. The distinguishing detail (bad magic, syntax error at line/column, an
unopenable path) belongs in `Error::message`. If a caller someday needs to
branch on parse-vs-open programmatically, that is the trigger to revisit this.

## gpu_status and check_or_abort

- `gpu_status(code)` — converts any WarpWraps `error_type` (`wwrError_t`, which
  is `cudaError_t` or `hipError_t` depending on the build, or a library status
  code) to a `Status`. A failure becomes `ErrorKind::GpuRuntime` carrying the
  code's name and description, located at the caller. Not named `gpu_check`,
  which is `wwr::extension::gpu_check` (policy-driven, returns `bool`).
- `check_or_abort(result_or_status)` — on error, logs at `critical` with the
  error's original location and calls `std::abort()`; otherwise unwraps.

`check_or_abort` is the abort tier, and the only place an error is logged at
the point of detection — there, detection and the process boundary are the
same call. The macros over both (`FBF_TRY`, `FBF_GPU_MUST`) are in
`../diagnostics/diagnostics.h`.

## GPU error policy

What a failed GPU call inside a wwr RAII wrapper does. WarpWraps' extension
layer forces no error policy: every wrapper takes its policies as template
arguments (`wwr.extension.common`'s `error_policy` concepts). `:gpu_policy`
supplies ours:

| Slot | Alias | Policy | On failure |
| --- | --- | --- | --- |
| Create / alloc / device-select | `CreatePolicy<E>` | `AbortPolicy` | Log at critical with the failing call's location, `std::abort()` |
| Destroy / free | `DestroyPolicy<E>` | `LogPolicy` | Log at error, carry on |

`PolicyMap` binds the two to wwr's `alloc` / `free` slots for the buffer suites.
Name the aliases rather than a concrete policy — or, in practice, neither:
`free_breakfast.core.gpu` has the wrappers already bound (`Device`, `Buffers`,
`BlasHandle`, …).

Because a failed creation aborts at the call that failed, a wrapper that exists
is a valid one: there is no empty-wrapper state to check for, and no error
value to propagate.

```cpp
import free_breakfast.core;
namespace fb = free_breakfast;

auto dev = std::make_shared<fb::Device>(0);
fb::Buffers::device<float> buf(n, dev);   // aborts here if the allocation fails
```

### Why it is shaped like this

- **Abort, not recover.** For this code, the answer to a failed device
  allocation or handle creation is "rerun with a smaller problem or on another
  device", not an in-process fallback. Nothing in the tree recovers, so a
  recoverable path would be machinery with no caller. If one is ever needed,
  add it for that site — wrap that one construction with a policy that
  reports instead of aborting — rather than as a tree-wide mode.
- **Destroy/free logs instead.** It runs in a destructor, often during
  teardown after an earlier failure, where aborting would bury the root cause
  under the knock-on one.
- **Both policies are empty**, so they fit every wwr slot and collapse to zero
  size (`policy_slot`), and `kit::DeviceHandle` can default-construct its
  device policy.

## Files

| File | Role |
|------|------|
| `error_handling.cppm` | Primary module interface; re-exports the three partitions |
| `error.cppm` | `:error` — `Error`, `ErrorKind`, `Result<T>`, `Status`, the `err_*` helpers and `to_exit_code` |
| `gpu_status.cppm` | `:gpu_status` — `gpu_status`, `check_or_abort` |
| `gpu_policy.cppm` | `:gpu_policy` — `AbortPolicy`, `LogPolicy` and the aliases over them |
| `CMakeLists.txt` | The module library |

## Dependencies

- `free_breakfast::core::log` — `log::critical()` / `log::error()` from the
  policies and `check_or_abort`
- `wwr::extension::common` — `error_type`, `success_code`, `error_name`,
  `error_string` and the policy concepts. PUBLIC, like `log`, because
  `gpu_status` and the policies are templates instantiated in the consumer's TU.
