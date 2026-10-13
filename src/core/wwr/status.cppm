/// @file
/// @brief GPU error codes to Status

export module free_breakfast.core.wwr:status;

import std;
import free_breakfast.core.error_handling;
import wwr.extension.common; // error_type, success_code, error_name, error_string

export namespace free_breakfast {

/// @brief Convert a GPU error code to a Status
///
/// Backend-neutral: accepts anything WarpWraps registers as an error type --
/// `wwrError_t` (which is `cudaError_t` or `hipError_t` depending on the build)
/// and the library status codes its extension modules specialize. A failure
/// becomes `ErrorKind::GpuRuntime` carrying the code's name and description,
/// located at the caller.
///
/// Named `gpu_status` rather than `gpu_check` so it does not collide with
/// `wwr::extension::gpu_check`, which hands failures to an error policy and
/// returns `bool`.
template <wwr::extension::error_type T>
Status gpu_status(T code, std::source_location loc = std::source_location::current()) {
  if (code == wwr::extension::success_code<T>()) {
    return {};
  }
  // Error::gpu, not err_gpu: the location is the caller's, forwarded in `loc`,
  // which err_gpu's call-site-capturing format string has no way to take.
  return err(Error::gpu(
      std::format("{} ({})", wwr::extension::error_name(code), wwr::extension::error_string(code)),
      loc));
}

} // namespace free_breakfast
