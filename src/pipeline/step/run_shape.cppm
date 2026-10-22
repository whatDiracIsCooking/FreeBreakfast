/// @file
/// @brief `:run_shape`: where a step runs, as its `run` signature says.
///
/// A partition of its own because :pipeline_step's unexported signature
/// helpers name it, and so it must be declared before them.

export module free_breakfast.pipeline.step:run_shape;

import free_breakfast.core.types;

export namespace free_breakfast {

/// @brief Where a step runs and what it holds, read off its `run` signature.
///
/// The resource parameters are the declaration: a step not handed a
/// `DeviceResources&` cannot reach the device, so there is no separate
/// constant that could disagree with what `run` takes.
enum class RunShape : uint8_t {
  Host,          ///< `run(View)`: free host work
  HostSubflow,   ///< `run(View, tf::Subflow&)`: free host work, parallel inside
  Device,        ///< `run(View, DeviceResources&)`: on that device's chain
  DeviceSubflow, ///< `run(View, DeviceResources&, tf::Subflow&)`: on the chain,
                 ///< host-parallel inside
  Node,          ///< `run(View, NodeResources&)`: holds every device's chain
};

} // namespace free_breakfast
