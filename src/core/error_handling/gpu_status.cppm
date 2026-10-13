/// @file
/// @brief GPU error codes to Status, and the abort tier over Result/Status

export module free_breakfast.core.error_handling:gpu_status;

import :error;
import free_breakfast.core.log;
import wwr.extension.common; // error_type, success_code, error_name, error_string
import std;

export namespace free_breakfast {

/**
 * @brief Convert a GPU error code to a Status
 *
 * Backend-neutral: accepts anything WarpWraps registers as an error type --
 * `wwrError_t` (which is `cudaError_t` or `hipError_t` depending on the build)
 * and the library status codes its extension modules specialize. A failure
 * becomes `ErrorKind::GpuRuntime` carrying the code's name and description,
 * located at the caller.
 *
 * Named `gpu_status` rather than `gpu_check` so it does not collide with
 * `wwr::extension::gpu_check`, which hands failures to an error policy and
 * returns `bool`.
 */
template <wwr::extension::error_type T>
Status gpu_status(T code, std::source_location loc = std::source_location::current()) {
  if (code == wwr::extension::success_code<T>()) {
    return {};
  }
  return err_gpu(
      std::format("{} ({})", wwr::extension::error_name(code), wwr::extension::error_string(code)),
      loc);
}

/**
 * @brief Log and abort if the Result holds an error, otherwise unwrap the value
 *
 * @tparam T Success value type
 * @return The unwrapped success value
 */
template <typename T>
T check_or_abort(Result<T> result) {
  if (!result) {
    log::critical(result.error().message, result.error().origin);
    std::abort();
  }
  return std::move(*result);
}

/// @brief Log and abort if the Status holds an error
inline void check_or_abort(Status status) {
  if (!status) {
    log::critical(status.error().message, status.error().origin);
    std::abort();
  }
}

} // namespace free_breakfast
