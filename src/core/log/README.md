# Log

Lightweight logging with automatic source location capture, backed by spdlog.

## Module

`free_breakfast.core.log` — target `free_breakfast::core::log`, also re-exported
by `free_breakfast.core`. spdlog is linked PRIVATE through the
`free_breakfast.third_party.spdlog` shim in `deps/spdlog/`.

## Usage

```cpp
import free_breakfast.core.log;

free_breakfast::log::set_level(free_breakfast::log::LogLevel::Info);
free_breakfast::log::info("run started");

// Optionally add a file sink (one allowed; subsequent calls with a different path are ignored)
free_breakfast::log::add_file_sink("/tmp/run.log");
free_breakfast::log::debug("this won't print — level is info");
free_breakfast::log::warn("residual growing");

// Anything that formats goes through the FBF_LOG_* macros in
// ../diagnostics/diagnostics.h, so a filtered-out log pays no std::format cost:
FBF_LOG_DEBUG("iteration {} residual {:.3e}", iter, residual);

// log::warn bumps a process-global counter -- a top-level handler can print a
// one-line health summary at exit without grepping the log.
free_breakfast::log::warning_count();       // N, incremented by every warn() so far
free_breakfast::log::reset_warning_count(); // test-only: zero it back out
```

Output goes to **stdout**, formatted
`[YYYY-MM-DD HH:MM:SS.mmm] [level] [file:line] message`. Messages at `error` and
above flush immediately; lower levels stay buffered until `log::flush()`.

## Level semantics

Each level has one job, so a reader can set the level to the question they are
asking instead of grepping one undifferentiated stream:

| Level | Job |
|---|---|
| `trace` | Per-iteration / per-kernel, dev-only. Off by default even in a debug build. |
| `debug` | Algorithmic detail a developer diagnosing a wrong answer or slow convergence would want. |
| `info` | User-facing progress: what run is this, how is it configured, how did it go. |
| `warn` | Recoverable degradation: the run continues, but something was not as requested. |
| `error` | Value-tier failure at a boundary: logged **once**, where a `Result`/`Status` becomes an exit code — never at each propagation hop. |
| `critical` | Abort tier: logged immediately before `std::abort()` (`check_or_abort` / `FBF_GPU_MUST`). |

**Rule: any call site that formats goes through the `FBF_LOG_*` macros.**
`log::trace`/`debug`/... take `std::string_view`, so
`log::debug(std::format(...))` formats eagerly even when debug is filtered out.
`log::<level>(std::format(...))` should appear only inside `diagnostics.h`; call
`log::info`/`log::warn`/... directly only for a plain string with nothing to
format.

Note that `FBF_LOG_WARN` skips the call entirely when warn is filtered, so it
does not count toward `warning_count()` then; a direct `log::warn` always does.

## Files

| File | Role |
|------|------|
| `log.cppm` | Module interface exporting the free functions in `free_breakfast::log` |
| `log_level.cppm` | `LogLevel` enum, plus the spdlog logger itself: the `stdout_color_sink_mt`, the `[%s:%#]` pattern and the `flush_on(err)` policy |
| `log.cpp` | Implementation: the level wrappers, `set_level`, `should_log`, `flush`, `add_file_sink`'s optional `basic_file_sink_mt`, and the atomic warning counter |
| `CMakeLists.txt` | Build configuration |
