# Diagnostics

The macro header over `error_handling` and `log`: `FBF_TRY`, `FBF_GPU_MUST`
and the lazily-formatted `FBF_LOG_*`. The functions they expand to
live in `free_breakfast.core.error_handling` (`check_or_abort`) and
`free_breakfast.core.wwr` (`gpu_status`) — see `../error_handling/README.md`
and `../wwr/README.md`.

Value-tier errors are logged exactly **once**, at the top-level boundary that
turns an `Error` into a process exit code (via `to_exit_code`) — not at every
propagation hop in between. `err_*` factories stay pure (they construct, they
do not log); only `check_or_abort`'s abort tier logs at the point of detection,
because there the point of detection and the process boundary are the same
call.

## Header (macros)

`diagnostics.h`, target `free_breakfast::core::diagnostics_macros`. Macros cannot
cross a module boundary, so they live in a plain header. Include it **after**
the imports it expands against — like `test/support/gpu_requirement.h`, the macros
need `free_breakfast::*` and `std` visible in the including TU.

```cpp
#include <gtest/gtest.h>   // any textual headers first

import std;
import free_breakfast.core;   // re-exports core.wwr: gpu_status, wwr::wwr*

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
| `diagnostics.h` | Plain header: `FBF_TRY`, `FBF_GPU_MUST`, `FBF_LOG_*` |
| `CMakeLists.txt` | The `diagnostics_macros` interface library for the header |

## Dependencies

- `free_breakfast::core::error_handling` — `check_or_abort`, and (through it)
  `free_breakfast::core::log` for `log::should_log()`

Not `free_breakfast::core::wwr`, though `FBF_GPU_MUST` expands to its
`gpu_status`: `io/` and `basis/` link this target for `FBF_TRY` and
`FBF_LOG_*` and have no GPU code. A TU that calls `FBF_GPU_MUST` already holds
a wwr error code, so it already imports `free_breakfast.core.wwr` and links its
target.
