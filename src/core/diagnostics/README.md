# Diagnostics

Bridges `error_handling`, `log` and WarpWraps' GPU error codes. Two targets: a
C++23 module (`gpu_status`, `check_or_abort`) and a plain header of macros that
build on it.

Value-tier errors are logged exactly **once**, at the top-level boundary that
turns an `Error` into a process exit code (via `to_exit_code`) — not at every
propagation hop in between. `err_*` factories stay pure (they construct, they
do not log); only `check_or_abort`'s abort tier logs at the point of detection,
because there the point of detection and the process boundary are the same
call.

## Module

`free_breakfast.core.diagnostics` — target `free_breakfast::core::diagnostics`,
also re-exported by `free_breakfast.core`.

- `gpu_status(code)` — converts any WarpWraps `error_type` (`wwrError_t`, which
  is `cudaError_t` or `hipError_t` depending on the build, or a library status
  code) to a `Status`. A failure becomes `ErrorKind::GpuRuntime` carrying the
  code's name and description, located at the caller. Not named `gpu_check`,
  which is `wwr::extension::gpu_check` (policy-driven, returns `bool`).
- `check_or_abort(result_or_status)` — on error, logs at `critical` with the
  error's original location and calls `std::abort()`; otherwise unwraps.

## Header (macros)

`diagnostics.h`, target `free_breakfast::core::diagnostics_macros`. Macros cannot
cross a module boundary, so they live in a plain header. Include it **after**
the imports it expands against — like `test/gtest/gpu_requirement.h`, the macros
need `free_breakfast::*` and `std` visible in the including TU.

```cpp
#include <gtest/gtest.h>   // any textual headers first

import std;
import free_breakfast.core;
import wwr.runtime_api;

#include "diagnostics.h"

free_breakfast::Result<Config> load(std::string_view path) {
  auto bytes = FBF_TRY(read_file(path));   // auto x = FBF_TRY(result_expr)
  FBF_TRY(validate(bytes));                // bare FBF_TRY(status_expr)
  return parse(bytes);
}

FBF_GPU_MUST(wwr::wwrSetDevice(0));        // check_or_abort(gpu_status(...))

FBF_LOG_DEBUG("iteration {} residual {:.3e}", iter, residual); // formats only if debug is on
```

- `FBF_TRY(expr)` — value-tier propagation via a clang statement-expression
  (this repo builds clang-only). On failure, returns
  `std::unexpected(std::move(error))` from the *enclosing* function, which must
  itself return a `Result<U>`/`Status`. The `Error`'s `source_location` moves
  through unchanged — it still points at the original `err_*` call.
- `FBF_GPU_MUST(call)` — abort tier: `check_or_abort(gpu_status(call))`.
- `FBF_LOG_TRACE` / `DEBUG` / `INFO` / `WARN` / `ERROR` / `CRITICAL` — one
  lazily-guarded macro per `LogLevel` (except `off`), each checking
  `log::should_log` before formatting.

## Files

| File | Role |
|------|------|
| `diagnostics.cppm` | Primary module interface; exports `gpu_status` and `check_or_abort` |
| `diagnostics.h` | Plain header: `FBF_TRY`, `FBF_GPU_MUST`, `FBF_LOG_*` |
| `CMakeLists.txt` | The module library and the `diagnostics_macros` interface library for the header |

## Dependencies

- `free_breakfast::core::error_handling` — `Error`, `Result<T>`, `Status`
- `free_breakfast::core::log` — `log::critical()`, and (for the header) `log::should_log()`
- `wwr::extension::common` — `error_type`, `success_code`, `error_name`,
  `error_string`. PUBLIC, because `gpu_status` is a template instantiated in
  the consumer's TU.
