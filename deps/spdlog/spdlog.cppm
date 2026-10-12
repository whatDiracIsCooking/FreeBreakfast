module;

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

export module free_breakfast.third_party.spdlog;

export namespace free_breakfast::spdlog {

using ::spdlog::logger;
using ::spdlog::sink_ptr;
using ::spdlog::source_loc;

using ::spdlog::level::level_enum;

using ::spdlog::sinks::basic_file_sink_mt;
using ::spdlog::sinks::stdout_color_sink_mt;

} // namespace free_breakfast::spdlog
