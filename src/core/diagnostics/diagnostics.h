/// @file
/// @brief Two-tier error idioms and lazy-logging macros
///
/// Macros cannot cross a C++ module boundary, so this is a PLAIN header
/// rather than a module partition. Include it AFTER
/// `import free_breakfast.core;` (or at least
/// `free_breakfast.core.error_handling` and `.log`, plus `.wwr` for
/// FBF_GPU_MUST):
/// like `test/gtest/gpu_requirement.h`, these macros are expanded at the
/// point of use and need `free_breakfast::check_or_abort`,
/// `free_breakfast::gpu_status` and `free_breakfast::log::should_log` visible
/// there, not here. `std` facilities
/// (`std::move`, `std::format`, `std::forward`) come from the including TU's
/// `import std;` for the same reason.
///
/// All macros here carry an `FBF_` prefix to keep this header's names out of
/// the way of anything the including TU (or a third-party header it drags
/// in) might itself call `TRY` or `MUST`.
///
/// ## FBF_TRY(expr) -- value-tier propagation
///
/// `expr` must be a `free_breakfast::Result<T>` or a `free_breakfast::Status`
/// (anything satisfying the `std::expected<T, Error>` shape). On failure it
/// returns `std::unexpected(std::move(error))` from the ENCLOSING function --
/// so that function's return type has to itself be a `Result<U>` or `Status`
/// for some `U`. On success it evaluates to the unwrapped value (or, for a
/// `Status`, to nothing usable -- write `FBF_TRY(...)` as a bare statement in
/// that case, not `auto x = FBF_TRY(...)`).
///
/// The existing `Error` factories (`Error::io`, `err_config`, ...) already
/// capture their `source_location` at the point the error was CREATED, and
/// `FBF_TRY` only moves that `Error` through `std::unexpected` -- it never
/// constructs a new one -- so the location a caller eventually logs is still
/// the original detection site, not the `FBF_TRY` call site that re-threw it.
///
/// ```cpp
/// free_breakfast::Result<Config> load(std::string_view path) {
///   auto bytes = FBF_TRY(read_file(path));   // auto x = FBF_TRY(result_expr)
///   FBF_TRY(validate(bytes));                // bare FBF_TRY(status_expr)
///   return parse(bytes);
/// }
/// ```
///
/// A clang statement-expression (`({ ... })`) is what makes both forms work
/// from the same macro body -- this repo builds clang-only, libc++, so the
/// extension is available everywhere the macro is used.
///
/// ## FBF_GPU_MUST(call) -- abort-tier
///
/// For GPU calls where there is no reasonable recovery and propagating a
/// `Status` up through the caller would just relocate the abort, not avoid
/// it. Expands to `check_or_abort(gpu_status(call))`: convert the error code
/// (`wwrError_t` or any other WarpWraps `error_type`, under either backend) to
/// a `Status` and, on failure, log at critical and `std::abort()`.
///
/// ## FBF_LOG_TRACE / FBF_LOG_DEBUG / FBF_LOG_INFO / FBF_LOG_WARN /
/// FBF_LOG_ERROR / FBF_LOG_CRITICAL -- lazily-formatted logs
///
/// `log::trace` / `log::debug` / ... all take `std::string_view`, so
/// `log::debug(std::format(...))` formats EAGERLY even when that level is
/// filtered out -- exactly the cost a guard is supposed to avoid. These
/// macros check `log::should_log` first and skip the `std::format` call
/// entirely when the level would be discarded anyway. One macro per
/// `LogLevel` enumerator except `off`, which is never a level to log AT.
///
/// ```cpp
/// FBF_LOG_TRACE("block ({}, {}) skipped", i, j);
/// FBF_LOG_DEBUG("iteration {} residual {:.3e}", iter, residual);
/// FBF_LOG_INFO("converged in {} iterations", iter);
/// FBF_LOG_WARN("cond(A) = {:.3e} exceeds threshold", cond);
/// FBF_LOG_ERROR("extrapolation failed, falling back to damping");
/// FBF_LOG_CRITICAL("input {} is empty", name);
/// ```

#pragma once

/// @brief Value-tier propagation: unwrap a Result/Status or return its error.
///
/// See the file-level doc comment for the two call shapes and what it expands
/// to. `_fb_try_result` is scoped to the statement-expression, so
/// nested/repeated uses in the same statement do not collide.
#define FBF_TRY(expr)                                                                              \
  ({                                                                                               \
    auto &&_fb_try_result = (expr);                                                                \
    if (!_fb_try_result) {                                                                         \
      return ::std::unexpected(::std::move(_fb_try_result).error());                               \
    }                                                                                              \
    ::std::move(_fb_try_result);                                                                   \
  }).value()

/// @brief Abort-tier: check a GPU call's return code and abort on failure,
/// logging at critical first. A no-op (aside from the check itself) on
/// success.
#define FBF_GPU_MUST(call) ::free_breakfast::check_or_abort(::free_breakfast::gpu_status(call))

/// @brief Lazily-formatted trace-level log: `std::format`s only when the
/// trace level would actually be emitted.
#define FBF_LOG_TRACE(...)                                                                         \
  do {                                                                                             \
    if (::free_breakfast::log::should_log(::free_breakfast::log::LogLevel::Trace)) {               \
      ::free_breakfast::log::trace(::std::format(__VA_ARGS__));                                    \
    }                                                                                              \
  } while (0)

/// @brief Lazily-formatted debug-level log: `std::format`s only when the
/// debug level would actually be emitted.
#define FBF_LOG_DEBUG(...)                                                                         \
  do {                                                                                             \
    if (::free_breakfast::log::should_log(::free_breakfast::log::LogLevel::Debug)) {               \
      ::free_breakfast::log::debug(::std::format(__VA_ARGS__));                                    \
    }                                                                                              \
  } while (0)

/// @brief Lazily-formatted info-level log: `std::format`s only when the info
/// level would actually be emitted.
#define FBF_LOG_INFO(...)                                                                          \
  do {                                                                                             \
    if (::free_breakfast::log::should_log(::free_breakfast::log::LogLevel::Info)) {                \
      ::free_breakfast::log::info(::std::format(__VA_ARGS__));                                     \
    }                                                                                              \
  } while (0)

/// @brief Lazily-formatted warn-level log: `std::format`s only when the warn
/// level would actually be emitted.
#define FBF_LOG_WARN(...)                                                                          \
  do {                                                                                             \
    if (::free_breakfast::log::should_log(::free_breakfast::log::LogLevel::Warn)) {                \
      ::free_breakfast::log::warn(::std::format(__VA_ARGS__));                                     \
    }                                                                                              \
  } while (0)

/// @brief Lazily-formatted error-level log: `std::format`s only when the
/// error level would actually be emitted.
#define FBF_LOG_ERROR(...)                                                                         \
  do {                                                                                             \
    if (::free_breakfast::log::should_log(::free_breakfast::log::LogLevel::Error)) {               \
      ::free_breakfast::log::error(::std::format(__VA_ARGS__));                                    \
    }                                                                                              \
  } while (0)

/// @brief Lazily-formatted critical-level log: `std::format`s only when the
/// critical level would actually be emitted.
#define FBF_LOG_CRITICAL(...)                                                                      \
  do {                                                                                             \
    if (::free_breakfast::log::should_log(::free_breakfast::log::LogLevel::Critical)) {            \
      ::free_breakfast::log::critical(::std::format(__VA_ARGS__));                                 \
    }                                                                                              \
  } while (0)
