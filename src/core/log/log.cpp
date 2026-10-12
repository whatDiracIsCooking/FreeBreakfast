module free_breakfast.core.log;

import free_breakfast.third_party.spdlog;

namespace {

// Process-global, function-local static so its initialization is safe under
// concurrent first use (like get_logger() in log_level.cppm). Relaxed
// ordering throughout: this counts warnings for a health summary, it does not
// guard access to anything else.
std::atomic<std::uint64_t> &warning_counter() {
  static std::atomic<std::uint64_t> counter{0};
  return counter;
}

} // namespace

namespace free_breakfast::log {

void trace(std::string_view msg, std::source_location loc) {
  log_impl(LogLevel::trace, msg, loc);
}
void debug(std::string_view msg, std::source_location loc) {
  log_impl(LogLevel::debug, msg, loc);
}
void info(std::string_view msg, std::source_location loc) {
  log_impl(LogLevel::info, msg, loc);
}
void warn(std::string_view msg, std::source_location loc) {
  warning_counter().fetch_add(1, std::memory_order_relaxed);
  log_impl(LogLevel::warn, msg, loc);
}
void error(std::string_view msg, std::source_location loc) {
  log_impl(LogLevel::error, msg, loc);
}
void critical(std::string_view msg, std::source_location loc) {
  log_impl(LogLevel::critical, msg, loc);
}

std::uint64_t warning_count() { return warning_counter().load(std::memory_order_relaxed); }

void reset_warning_count() { warning_counter().store(0, std::memory_order_relaxed); }

void set_level(LogLevel lv) {
  get_logger().set_level(static_cast<spdlog::level_enum>(lv));
}

bool should_log(LogLevel lv) {
  return get_logger().should_log(static_cast<spdlog::level_enum>(lv));
}

void flush() {
  get_logger().flush();
}

// Only one file sink is allowed per process — subsequent calls with a
// different path warn and are ignored. Same-path calls are silently idempotent.
void add_file_sink(std::filesystem::path const &path, std::source_location loc) {
  static std::optional<std::filesystem::path> registered;
  if (registered) {
    if (*registered != path) {
      log_impl(LogLevel::warn,
               std::format("file sink already set to '{}', ignoring '{}'", registered->string(),
                           path.string()),
               loc);
    }
    return;
  }
  registered = path;
  auto sink = std::make_shared<spdlog::basic_file_sink_mt>(path.string());
  get_logger().sinks().push_back(std::move(sink));
}

} // namespace free_breakfast::log
