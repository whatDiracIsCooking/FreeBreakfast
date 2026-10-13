/// @file
/// @brief Logging facade with source-location tracking

export module free_breakfast.core.log;

export import :log_level;
import std;
import free_breakfast.core.types;

export namespace free_breakfast::log {

/// @brief Set the global minimum log level
void set_level(LogLevel lv);

/// @brief Would a message at this level actually be emitted right now?
///
/// Wraps spdlog's own `should_log`, so it reflects whatever `set_level` last
/// set rather than duplicating that state. Intended as a guard BEFORE
/// expensive message formatting -- see FBF_LOG_DEBUG/FBF_LOG_TRACE in
/// core/diagnostics/diagnostics.h, which are the reason this exists: the
/// facade below takes `std::string_view`, so `log::debug(std::format(...))`
/// formats unconditionally even when debug is filtered out.
bool should_log(LogLevel lv);

/// @brief Flush every sink now
///
/// spdlog buffers, so a message is not necessarily on disk when the call
/// returns. Errors and above flush themselves (see get_logger), which covers
/// the case that matters -- losing the tail of the log to a crash. This is for
/// everything else: call it before a checkpoint you intend to read from
/// another process, or at the end of a batch run.
void flush();

/// @brief Add a file output sink alongside the default console sink
void add_file_sink(std::filesystem::path const &path,
                   std::source_location loc = std::source_location::current());

/// @brief Log at trace level
void trace(std::string_view msg, std::source_location loc = std::source_location::current());
/// @brief Log at debug level
void debug(std::string_view msg, std::source_location loc = std::source_location::current());
/// @brief Log at info level
void info(std::string_view msg, std::source_location loc = std::source_location::current());
/// @brief Log at warn level
///
/// Also bumps the process-global warning counter (see @ref warning_count) --
/// this is the only place that counter is incremented, so every `log::warn`
/// call site counts toward it for free.
void warn(std::string_view msg, std::source_location loc = std::source_location::current());
/// @brief Log at error level
void error(std::string_view msg, std::source_location loc = std::source_location::current());
/// @brief Log at critical level
void critical(std::string_view msg, std::source_location loc = std::source_location::current());

/// @brief Number of `log::warn` calls so far this process
///
/// A single atomic counter, not per-thread and not reset by anything except
/// @ref reset_warning_count -- the point is a cheap, always-on health signal
/// a top-level handler can print at exit ("completed: 0 errors, N warnings")
/// without grepping the log. Relaxed ordering: this is a diagnostic count,
/// not a synchronization point.
uint64_t warning_count();

/// @brief Reset the warning counter to zero
///
/// Exists for test isolation -- the counter is function-local static state
/// shared by the whole binary, so a test asserting on it must be able to
/// zero it first rather than read whatever earlier tests left behind.
void reset_warning_count();

} // namespace free_breakfast::log
