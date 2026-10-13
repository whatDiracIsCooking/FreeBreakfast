/// @file
/// @brief The abort tier over Result/Status

export module free_breakfast.core.error_handling:abort;

import std;
import :error;
import free_breakfast.core.log;

export namespace free_breakfast {

/// @brief Log and abort if the Result holds an error, otherwise unwrap the value
///
/// @tparam T Success value type
/// @return The unwrapped success value
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
