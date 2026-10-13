/// @file
/// @brief Errors, the GPU-code bridge into them, and the tree's GPU error policy

export module free_breakfast.core.error_handling;

// Partitions rather than separate modules because each later one needs the one
// before it, and diagnostics.h needs all three: Error and Result (:error);
// gpu_status and check_or_abort over them (:gpu_status); and the wwr error
// policies, which report through both (:gpu_policy).
export import :error;
export import :gpu_status;
export import :gpu_policy;
