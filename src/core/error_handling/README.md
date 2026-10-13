# Error Handling

Central error type and result aliases for the project. Uses C++23
`std::expected` to propagate typed errors with automatic source location
capture. It also turns WarpWraps' GPU error codes into those errors, and holds
the tree-wide policy for what a failed GPU call inside a wwr RAII wrapper does.

## Module

`free_breakfast.core.error_handling` — target `free_breakfast::core::error_handling`,
also re-exported by `free_breakfast.core`. Three partitions, each building on
the one before:

| Partition | Exports |
|-----------|---------|
| `:error` | `Error`, `ErrorKind`, `Result<T>`, `Status`, `LocatedFormat`, the `err_*` helpers, `to_exit_code` |
| `:gpu_status` | `gpu_status`, `check_or_abort` |
| `:gpu_policy` | `AbortPolicy` / `RecordPolicy` / `LogPolicy`, `CreatePolicy` / `DestroyPolicy` / `PolicyMap`, `kAbortsOnError`, `capture`, `take_error` |

## Usage

```cpp
import free_breakfast.core.error_handling;

free_breakfast::Result<int> parse_count(int raw) {
  if (raw <= 0)
    return free_breakfast::err_config("count must be positive");
  return raw;
}

// Source location is auto-captured at the call site
free_breakfast::Status allocate_buffer(std::size_t n) {
  if (n == 0)
    return std::unexpected(free_breakfast::Error::allocation("zero-size allocation"));
  return {};
}
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

What a failed GPU call inside a wwr RAII wrapper does — decided once, at
configure time, for the whole tree.

WarpWraps' extension layer forces no error policy: every wrapper takes its
policies as template arguments (`wwr.extension.common`'s `error_policy`
concepts). `:gpu_policy` supplies ours, and a switch between two behaviours:

| `FREE_BREAKFAST_GPU_ERROR_POLICY` | Create / alloc / device-select slots | Destroy / free slots |
| --- | --- | --- |
| `ABORT` | `AbortPolicy` — log critical, `std::abort()` at the failing call | `AbortPolicy` |
| `RECORD` | `RecordPolicy` — record for the enclosing `capture` | `LogPolicy` — log at error |
| *(empty, default)* | `ABORT` for `CMAKE_BUILD_TYPE=Debug`, `RECORD` otherwise | |

The resolved mode is printed at configure (`-- GPU error policy: …`) and is
readable in code as `kAbortsOnError`.

### Using it

Name the switched aliases — `CreatePolicy<E>`, `DestroyPolicy<E>`,
`PolicyMap` — rather than a concrete policy. In practice you rarely do even
that: `free_breakfast.core.gpu` has the wrappers already bound (`Device`,
`Buffers`, `BlasHandle`, …).

Construct inside `capture`, which turns a recorded failure into a `Result`:

```cpp
import free_breakfast.core;
namespace fb = free_breakfast;

fb::Result<fb::Buffers::device<float>> make_buffer(std::size_t n) {
  auto dev = FBF_TRY(fb::capture([] { return std::make_shared<fb::Device>(0); }));
  return fb::capture([&] { return fb::Buffers::device<float>(n, dev); });
}
```

Under `ABORT` the same code compiles and never sees an error value — the abort
happens first, at the call that failed — so callers are written once.

### Behaviour worth knowing

- **Write the construction inside the lambda.** A wrapper's defaulted
  `std::source_location` is then your line, and that is what the returned
  `Error::origin` points at.
- **First error wins** within a capture: later failures are usually knock-on
  effects of the first.
- **Captures nest.** An inner capture neither reports nor wipes an error its
  outer one has pending — also when the inner lambda throws.
- **A `RECORD` failure outside any capture is logged at error**, since nobody
  is there to return it to. It is also kept for `take_error()`.
- **Destroy/free failures are logged, not returned** in `RECORD`: they happen in
  a destructor, after any capture that could have returned them has finished.
- A failed create leaves the wrapper empty — `valid()` is false for a handle,
  `num_elements()` is 0 for a buffer — which is wwr's record-and-continue path.

### Why it is shaped like this

- **Per-thread recording, not a stateful policy.** The policies are empty so
  they fit every wwr slot and collapse to zero size (`policy_slot`), and because
  `kit::DeviceHandle` default-constructs its device policy — a policy holding a
  pointer to "its" error could not be threaded through there. The error belongs
  to whichever capture is running on the thread that made the call.
- **The switch is a compile definition on this target only**, read once by
  `:gpu_policy` into
  the `constexpr` `kAbortsOnError`. Every importer gets that value through
  the BMI, so no TU can be built against the other mode.
- **Not `NDEBUG`.** That is `assert`'s switch, invisible in presets and the
  configure log; this one is a cache variable, so a Release build can opt into
  `ABORT` to chase a failure.

## Files

| File | Role |
|------|------|
| `error_handling.cppm` | Primary module interface; re-exports the three partitions |
| `error.cppm` | `:error` — `Error`, `ErrorKind`, `Result<T>`, `Status`, the `err_*` helpers and `to_exit_code` |
| `gpu_status.cppm` | `:gpu_status` — `gpu_status`, `check_or_abort` |
| `gpu_policy.cppm` | `:gpu_policy` — the policies, the ABORT/RECORD switch, `capture` |
| `gpu_policy.cpp` | The per-thread recording slot behind `RecordPolicy` and `capture` |
| `CMakeLists.txt` | The module library, and the ABORT/RECORD compile definition |

## Dependencies

- `free_breakfast::core::log` — `log::critical()` / `log::error()` from the
  policies and `check_or_abort`
- `wwr::extension::common` — `error_type`, `success_code`, `error_name`,
  `error_string` and the policy concepts. PUBLIC, like `log`, because
  `gpu_status` and the policies are templates instantiated in the consumer's TU.
