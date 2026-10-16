/// @file
/// @brief NodeResources: every device's DeviceResources plus a taskflow
/// executor for host-side work

export module free_breakfast.core.resources:node_resources;

import std;
import :device_resources;
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.wwr;

// Re-exported: executor() hands out a tf::Executor, and running work on it
// takes tf::Taskflow, so a caller needs taskflow wherever it has a node.
export import free_breakfast.third_party.taskflow;

export namespace free_breakfast {

/// @brief What a NodeResources takes from the node. Every field defaults to
/// "all of it, sensibly".
struct NodeResourcesConfig {
  /// @brief Devices to build a DeviceResources on, by runtime index, in order.
  /// Empty optional: every device the runtime can see. An empty vector: none.
  std::optional<std::vector<int32_t>> devices;
  /// @brief Executor worker threads. Empty: the physical core count less
  /// NodeResources::kReservedCores, and at least 1.
  std::optional<int32_t> num_threads;
  /// @brief Release threshold of every device's memory pool.
  uint64_t pool_release_threshold = DeviceResources::kDefaultPoolReleaseThreshold;
};

/// @brief One node's working set: a DeviceResources per device, and a taskflow
/// executor whose workers drive them.
///
/// A worker may last have run work for any device, and BLAS and solver calls
/// and kernel launches use the calling thread's current device, so a task
/// reaches a device through on_device(), which selects it for the call.
///
/// Neither copyable nor movable: the executor's workers hold its address. Hold
/// one per process and pass it by reference.
class NodeResources : private NonTransferable {
public:
  /// @brief Physical cores left out of the default worker count, for the
  /// thread that submits work and for the driver's own threads.
  static constexpr int32_t kReservedCores = 2;

  /// @brief Build a DeviceResources on each device `cfg` names and an executor
  /// with its worker count. The calling thread's current device is unchanged.
  /// A failed device creation, an unreadable /proc, or a worker count below 1
  /// aborts.
  explicit NodeResources(const NodeResourcesConfig &cfg = {});

  /// @brief The per-device working sets, in the order the config named them.
  std::span<DeviceResources> devices() noexcept { return devices_; }
  /// @copydoc devices()
  std::span<const DeviceResources> devices() const noexcept { return devices_; }
  /// @brief How many devices this node holds.
  size_t device_count() const noexcept { return devices_.size(); }
  /// @brief The working set at position `i` of devices() (not runtime index `i`).
  DeviceResources &device(size_t i) { return devices_.at(i); }
  /// @copydoc device()
  const DeviceResources &device(size_t i) const { return devices_.at(i); }

  /// @brief Run `f(device(i))` with that device current on this thread, then
  /// restore the device that was current before. Safe on any executor worker.
  template <typename F>
  decltype(auto) on_device(size_t i, F &&f) {
    DeviceResources &res = devices_.at(i);
    const ScopedDeviceIndex on{res.dev_idx()};
    return std::invoke(std::forward<F>(f), res);
  }

  /// @brief The executor for host-side task graphs.
  tf::Executor &executor() noexcept { return executor_; }
  /// @brief The executor's worker count.
  int32_t num_threads() const noexcept { return static_cast<int32_t>(executor_.num_workers()); }

  /// @brief Logical CPU cores on this node (hyperthreads included).
  int32_t logical_cores() const noexcept { return logical_cores_; }
  /// @brief Physical CPU cores on this node.
  int32_t physical_cores() const noexcept { return physical_cores_; }
  /// @brief Total host RAM, in bytes.
  size_t host_ram_bytes() const noexcept { return host_ram_bytes_; }

  /// @brief Log the node's cores, RAM, worker count and devices at info level.
  void log() const;

private:
  int32_t logical_cores_;
  int32_t physical_cores_;
  size_t host_ram_bytes_;
  std::vector<DeviceResources> devices_;
  // Declared last, so destroyed first: ~Executor waits for running tasks, and
  // they may still be using devices_.
  tf::Executor executor_;
};

} // namespace free_breakfast
