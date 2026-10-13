/// @file
/// @brief System environment query -- CPU, GPU, RAM, and disk discovery

export module free_breakfast.core.sys_env;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;

export namespace free_breakfast {

/// @brief Plain-data report of one physical GPU, for logging/inventory only
///
/// Deliberately not a `Device` (`free_breakfast.core.wwr`): a `Device` owns a
/// stream and a memory pool, and this roster exists purely to report what is on
/// the box at startup -- it must not create a stream and a pool per device just
/// for that.
struct GpuInfo {
  int32_t index;
  std::string name;
  size_t total_memory_bytes;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
};

/// @brief Snapshot of system hardware at startup
struct SysEnv {
  int32_t cpu_logical_cores;
  int32_t cpu_physical_cores;
  size_t host_ram_bytes;
  size_t disk_available_bytes;
  int32_t gpu_count;
  std::vector<GpuInfo> gpus;
};

/// @brief Query all available system hardware
///
/// A runtime with no usable GPU yields an empty roster, not an error; only a
/// device that is counted yet fails to report its properties is one.
[[nodiscard]] Result<SysEnv> query_sys_env();

/// @brief Return the number of GPUs visible to this build's runtime (0 if none)
[[nodiscard]] int32_t available_gpus();

/// @brief Return the number of logical CPU cores (hyperthreads included)
[[nodiscard]] int32_t logical_cpu_cores();

/// @brief Log all fields of a SysEnv at info level
void log_sys_env(const SysEnv &env);

} // namespace free_breakfast
