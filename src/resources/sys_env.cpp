/// @file
/// @brief Implementation of the :sys_env host queries.

module free_breakfast.resources;

import :sys_env;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr; // wwrGetDeviceCount

#include "diagnostics.h"

namespace free_breakfast {

// Parses the integer after the ':' of a /proc/cpuinfo "key : value" line.
// from_chars rather than stoi: a malformed line is skipped, not thrown out of.
[[nodiscard]] std::optional<int32_t> cpuinfo_field_value(std::string_view line) {
  const auto colon = line.find(':');
  if (colon == std::string_view::npos) {
    return std::nullopt;
  }
  auto value = line.substr(colon + 1);
  value.remove_prefix(std::min(value.find_first_not_of(" \t"), value.size()));
  int32_t parsed = 0;
  const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (ec != std::errc{}) {
    return std::nullopt;
  }
  return parsed;
}

// Counts physical (non-hyperthread) cores by deduplicating (physical_id, core_id)
// pairs from /proc/cpuinfo. Each logical CPU gets its own entry, but hyperthreads
// share the same pair.
//
// Most aarch64 kernels write neither field, and an inventory that fails outright
// on such a box is worse than one that reports the logical count there, so a file
// with no pairs falls back to logical_cpu_cores(). Only an unreadable file is an
// error.
[[nodiscard]] Result<int32_t> read_physical_cores() {
  std::ifstream f("/proc/cpuinfo");
  if (!f) {
    return err_io("cannot open '/proc/cpuinfo'");
  }

  std::set<std::pair<int32_t, int32_t>> cores;
  std::optional<int32_t> physical_id;
  std::optional<int32_t> core_id;
  std::string line;

  while (std::getline(f, line)) {
    if (line.starts_with("physical id")) {
      physical_id = cpuinfo_field_value(line);
    } else if (line.starts_with("core id")) {
      core_id = cpuinfo_field_value(line);
    } else if (line.empty() && physical_id && core_id) {
      cores.emplace(*physical_id, *core_id);
      physical_id.reset();
      core_id.reset();
    }
  }
  // Handle last entry if file doesn't end with a blank line.
  if (physical_id && core_id) {
    cores.emplace(*physical_id, *core_id);
  }

  if (cores.empty()) {
    FBF_LOG_DEBUG("no physical id / core id in '/proc/cpuinfo'; "
                  "reporting logical cores as physical");
    return logical_cpu_cores();
  }
  return static_cast<int32_t>(cores.size());
}

// Reads total physical RAM from /proc/meminfo (returns bytes).
[[nodiscard]] Result<size_t> read_host_ram() {
  std::ifstream f("/proc/meminfo");
  if (!f) {
    return err_io("cannot open '/proc/meminfo'");
  }

  constexpr std::string_view kMemTotal = "MemTotal:";
  std::string line;
  while (std::getline(f, line)) {
    if (line.starts_with(kMemTotal)) {
      size_t kb = 0;
      std::istringstream iss(line.substr(kMemTotal.size()));
      iss >> kb;
      if (kb == 0) {
        return err_io("failed to parse MemTotal from '/proc/meminfo'");
      }
      return kb * 1024;
    }
  }
  return err_io("MemTotal not found in '/proc/meminfo'");
}

int32_t available_gpus() {
  int32_t count = 0;
  if (wwr::wwrGetDeviceCount(&count) != wwr::wwrSuccess) {
    return 0;
  }
  return count;
}

int32_t logical_cpu_cores() {
  const auto n = static_cast<int32_t>(std::thread::hardware_concurrency());
  return (n > 0) ? n : 1;
}

} // namespace free_breakfast
