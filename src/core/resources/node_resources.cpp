/// @file
/// @brief Implementation of NodeResources' constructor and log().

module free_breakfast.core.resources;

import :sys_env;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;

#include "diagnostics.h"

namespace free_breakfast {

// The executor's worker count: the config's, or the physical cores less the
// reserve. A configured count below 1 is a config error rather than a silent
// clamp; the default is clamped, so a two-core box still gets a worker.
[[nodiscard]] Result<size_t> node_worker_count(const NodeResourcesConfig &cfg,
                                               int32_t physical_cores) {
  if (cfg.num_threads) {
    if (*cfg.num_threads < 1) {
      return err_config("NodeResources needs at least 1 worker thread, got {}", *cfg.num_threads);
    }
    return static_cast<size_t>(*cfg.num_threads);
  }
  return static_cast<size_t>(std::max(1, physical_cores - NodeResources::kReservedCores));
}

// One DeviceResources per configured device. Creating a device's handles
// leaves that device current on this thread, so the guard opened on the first
// one restores the caller's device after the last.
[[nodiscard]] std::vector<DeviceResources> make_node_devices(const NodeResourcesConfig &cfg) {
  std::vector<int32_t> indices;
  if (cfg.devices) {
    indices = *cfg.devices;
  } else {
    indices.resize(static_cast<size_t>(available_gpus()));
    std::iota(indices.begin(), indices.end(), 0);
  }

  std::vector<DeviceResources> devices;
  if (indices.empty()) {
    // No guard either: on a device-less runtime wwrGetDevice itself fails.
    return devices;
  }
  devices.reserve(indices.size());
  const ScopedDeviceIndex restore{indices.front()};
  for (const int32_t idx : indices) {
    devices.emplace_back(idx, cfg.pool_release_threshold);
  }
  return devices;
}

constexpr double bytes_to_gib(size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

NodeResources::NodeResources(const NodeResourcesConfig &cfg)
    : logical_cores_(logical_cpu_cores()), physical_cores_(check_or_abort(read_physical_cores())),
      host_ram_bytes_(check_or_abort(read_host_ram())), devices_(make_node_devices(cfg)),
      executor_(check_or_abort(node_worker_count(cfg, physical_cores_))) {}

void NodeResources::log() const {
  log::info("--- node resources ---");
  FBF_LOG_INFO("CPU logical cores:  {}", logical_cores_);
  FBF_LOG_INFO("CPU physical cores: {}", physical_cores_);
  FBF_LOG_INFO("Host RAM:           {:.2f} GiB", bytes_to_gib(host_ram_bytes_));
  FBF_LOG_INFO("Worker threads:     {}", num_threads());
  FBF_LOG_INFO("Devices:            {}", devices_.size());
  for (const auto &res : devices_) {
    const auto &props = res.device()->props();
    FBF_LOG_INFO("  GPU {}: {} | {:.2f} GiB | compute {}.{}", res.dev_idx(),
                 static_cast<const char *>(props.name), bytes_to_gib(props.totalGlobalMem),
                 props.major, props.minor);
  }
  log::info("----------------------");
}

} // namespace free_breakfast
