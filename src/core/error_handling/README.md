# Error Handling

Central error type and result aliases for the project. Uses C++23
`std::expected` to propagate typed errors with automatic source location
capture, and the abort tier over them (`check_or_abort`). It does not name
WarpWraps: the bridge from wwr's GPU error codes into `Error` (`gpu_status`)
and the error policies for the wwr RAII wrappers live in
[`free_breakfast.core.wwr`](../wwr/README.md), which builds on this module.

## Module

`free_breakfast.core.error_handling` — target `free_breakfast::core::error_handling`,
also re-exported by `free_breakfast.core`. Two partitions, the second building
on the first:

| Partition | Exports |
|-----------|---------|
| `:error` | `Error`, `ErrorKind`, `Result<T>`, `Status`, `LocatedFormat`, the `err_*` helpers, `to_exit_code` |
| `:abort` | `check_or_abort` |

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
caller that must forward somebody else's location (as `core.wwr`'s
`gpu_status` does) cannot use `err_*` and writes `err(Error::gpu(msg, loc))`
instead.

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

## check_or_abort

`check_or_abort(result_or_status)`: on error, logs at `critical` with the
error's original location and calls `std::abort()`; otherwise unwraps.

`check_or_abort` is the abort tier, and the only place an error is logged at
the point of detection — there, detection and the process boundary are the
same call. The macros over it (`FBF_TRY`, and `FBF_GPU_MUST` with `core.wwr`'s
`gpu_status`) are in `../diagnostics/diagnostics.h`. The GPU error policy that
aborts a failed wwr wrapper creation is in `../wwr/README.md`.

## Files

| File | Role |
|------|------|
| `error_handling.cppm` | Primary module interface; re-exports the two partitions |
| `error.cppm` | `:error` — `Error`, `ErrorKind`, `Result<T>`, `Status`, the `err_*` helpers and `to_exit_code` |
| `abort.cppm` | `:abort` — `check_or_abort` |
| `CMakeLists.txt` | The module library |

## Dependencies

- `free_breakfast::core::types`: the fixed-width integers in `Error`
- `free_breakfast::core::log`: `log::critical()` from `check_or_abort`. PUBLIC,
  because `check_or_abort<T>` is a template instantiated in the consumer's TU.
