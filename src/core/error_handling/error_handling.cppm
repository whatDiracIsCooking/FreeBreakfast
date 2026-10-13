/// @file
/// @brief Errors, and the abort tier over them

export module free_breakfast.core.error_handling;

// Partitions rather than separate modules because :abort needs :error, and
// diagnostics.h needs both: Error and Result (:error), and check_or_abort over
// them (:abort). The GPU-code bridge into these (gpu_status) and the wwr error
// policies live in free_breakfast.core.wwr, so this module never names wwr.
export import :error;
export import :abort;
