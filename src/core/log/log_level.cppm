/// @file
/// @brief Log level enum and spdlog integration

export module free_breakfast.core.log:log_level;

import std;
import free_breakfast.third_party.spdlog;

namespace free_breakfast::log {

/// @brief Severity levels for log filtering, matching spdlog's enum values
export enum class LogLevel : std::int32_t {
  Trace = 0,    ///< Fine-grained diagnostic events for development
  Debug = 1,    ///< Detailed information useful during debugging
  Info = 2,     ///< General informational messages about program progress
  Warn = 3,     ///< Potentially harmful situations that merit attention
  Error = 4,    ///< Error events that may still allow continued execution
  Critical = 5, ///< Severe errors likely followed by program abort
  Off = 6       ///< Disable all logging output
};

inline spdlog::logger &get_logger() {
  static auto instance = [] {
    auto sink = std::make_shared<spdlog::stdout_color_sink_mt>();
    auto lg = spdlog::logger("free_breakfast", sink);
    lg.set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v");
    // Flush anything at error or above immediately. Without this the logger
    // buffers everything, and the messages lost to a crash are precisely the
    // ones written just before it -- the ones explaining the crash. Trace
    // through warn stay buffered, because flushing every line of a hot loop is
    // what makes people turn logging off entirely; log::flush() is there for
    // when a caller wants the rest out at a chosen moment.
    lg.flush_on(spdlog::level_enum::err);
    return lg;
  }();
  return instance;
}

inline void log_impl(LogLevel lv, std::string_view msg, std::source_location loc) {
  get_logger().log(spdlog::source_loc{loc.file_name(), static_cast<std::int32_t>(loc.line()),
                                      loc.function_name()},
                   static_cast<spdlog::level_enum>(lv), "{}", msg);
}

} // namespace free_breakfast::log
