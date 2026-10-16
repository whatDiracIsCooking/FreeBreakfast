/// @file
/// @brief Host hardware queries NodeResources is built from: CPU cores, host
/// RAM, and the number of visible GPUs

// An internal partition: nothing here is exported. What a caller needs of it
// reaches them through NodeResources' accessors.
module free_breakfast.core.resources:sys_env;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;

namespace free_breakfast {

/// @brief The number of GPUs visible to this build's runtime (0 if none).
[[nodiscard]] int32_t available_gpus();

/// @brief The number of logical CPU cores (hyperthreads included), at least 1.
[[nodiscard]] int32_t logical_cpu_cores();

/// @brief The number of physical CPU cores, from /proc/cpuinfo.
///
/// Falls back to logical_cpu_cores() where the file has no physical id /
/// core id lines (most aarch64 kernels). Only an unreadable file is an error.
[[nodiscard]] Result<int32_t> read_physical_cores();

/// @brief Total host RAM in bytes, from /proc/meminfo.
[[nodiscard]] Result<size_t> read_host_ram();

} // namespace free_breakfast
