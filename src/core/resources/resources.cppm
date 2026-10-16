/// @file
/// @brief The node's compute resources: NodeResources, and the DeviceResources
/// it holds per device

export module free_breakfast.core.resources;

// One device's working set: a shared Device plus the BLAS and solver handles
// bound to its stream.
export import :device_resources;

// The whole node: a DeviceResources per device and a taskflow executor. The
// host queries it is built from (:sys_env) are an internal partition.
export import :node_resources;
