# Error Handling

Central error type and result aliases for the project. Uses C++23
`std::expected` to propagate typed errors with automatic source location
capture.

## Module

`free_breakfast.core.error_handling` — target `free_breakfast::core::error_handling`,
also re-exported by `free_breakfast.core`.

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

## Files

| File | Role |
|------|------|
| `error_handling.cppm` | Module interface exporting `Error`, `ErrorKind`, `Result<T>`, `Status`, the `err_*` helpers and `to_exit_code` |
| `CMakeLists.txt` | Build definition for the module |
