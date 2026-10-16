/// @file
/// @brief The machine's compute resources: DeviceResources, and the startup
/// hardware inventory

export module free_breakfast.resources;

// One device's working set: a shared Device plus the BLAS and solver handles
// bound to its stream.
export import :device_resources;

// Startup hardware inventory: CPU cores, host RAM, disk, and a plain-data GPU
// roster that deliberately creates no Device.
export import :sys_env;
