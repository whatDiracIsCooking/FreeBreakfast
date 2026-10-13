/// @file
/// @brief Implementation of the system environment query.

module free_breakfast.core.sys_env;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import wwr.runtime_api;

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

[[nodiscard]] Result<size_t> read_disk_space() {
  std::error_code ec;
  const auto info = std::filesystem::space("/", ec);
  if (ec) {
    return err_io("failed to query disk space: {}", ec.message());
  }
  return static_cast<size_t>(info.available);
}

constexpr double bytes_to_gib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

// Enumerates every device the runtime can see, as plain data -- no Device, no
// stream. A runtime with no usable device yields an empty vector, not an error.
// Only a device that counts yet fails to report its properties is a genuine
// error.
[[nodiscard]] Result<std::vector<GpuInfo>> read_gpu_info() {
  const int32_t count = available_gpus();

  std::vector<GpuInfo> gpus;
  gpus.reserve(static_cast<size_t>(count));
  for (int32_t i = 0; i < count; ++i) {
    wwr::wwrDeviceProp prop{};
    if (auto status = gpu_status(wwr::wwrGetDeviceProperties(&prop, i)); !status) {
      return err_gpu("cannot read properties of device {}: {}", i, status.error().message);
    }
    gpus.push_back(GpuInfo{i, std::string(static_cast<const char *>(prop.name)),
                           prop.totalGlobalMem, prop.major, prop.minor});
  }
  return gpus;
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

[[nodiscard]] Result<SysEnv> query_sys_env() {
  SysEnv env{};

  env.cpu_logical_cores = logical_cpu_cores();
  env.cpu_physical_cores = FBF_TRY(read_physical_cores());
  env.host_ram_bytes = FBF_TRY(read_host_ram());
  env.disk_available_bytes = FBF_TRY(read_disk_space());
  env.gpus = FBF_TRY(read_gpu_info());
  env.gpu_count = static_cast<int32_t>(env.gpus.size());

  return env;
}

void log_sys_env(const SysEnv &env) {
  log::info("--- system environment ---");
  FBF_LOG_INFO("CPU logical cores:  {}", env.cpu_logical_cores);
  FBF_LOG_INFO("CPU physical cores: {}", env.cpu_physical_cores);
  FBF_LOG_INFO("Host RAM:           {:.2f} GiB", bytes_to_gib(env.host_ram_bytes));
  FBF_LOG_INFO("Disk available:     {:.2f} GiB", bytes_to_gib(env.disk_available_bytes));
  FBF_LOG_INFO("GPU count:          {}", env.gpu_count);
  for (const auto &g : env.gpus) {
    FBF_LOG_INFO("  GPU {}: {} | {:.2f} GiB | compute {}.{}", g.index, g.name,
                 bytes_to_gib(g.total_memory_bytes), g.compute_capability_major,
                 g.compute_capability_minor);
  }
  log::info("--------------------------");
}

} // namespace free_breakfast
