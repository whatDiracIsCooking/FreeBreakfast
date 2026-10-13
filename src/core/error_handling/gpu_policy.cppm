/// @file
/// @brief The tree-wide error policies for the wwr RAII wrappers

export module free_breakfast.core.error_handling:gpu_policy;

import :error;
import :gpu_status; // gpu_status, check_or_abort
import free_breakfast.core.log;
import wwr.extension.common; // error_type, success_code, policy concepts
import std;

export namespace free_breakfast {

// Both policies are empty and noexcept, so each fits every wwr slot --
// including the destruction slots (nothrow_error_policy) -- and the wrappers'
// policy_slot collapses repeated ones to nothing. Empty is also what
// kit::DeviceHandle needs: it default-constructs its device policy rather than
// taking one.

/// @brief Log at critical with the failing call's location, then abort.
template <wwr::extension::error_type E>
struct AbortPolicy {
  using error_type = E;
  void handle_error(E code, std::source_location loc) noexcept {
    check_or_abort(gpu_status(code, loc));
  }
};

/// @brief Log at error level and carry on.
///
/// For the destruction slots: a failed destroy/free runs in a destructor, often
/// during teardown after an earlier failure, where aborting would bury the
/// root cause under the knock-on one.
template <wwr::extension::error_type E>
struct LogPolicy {
  using error_type = E;
  void handle_error(E code, std::source_location loc) noexcept {
    try {
      if (auto status = gpu_status(code, loc); !status) {
        log::error(status.error().message, loc);
      }
    } catch (...) {
    }
  }
};

/// @brief The policy for creation, allocation and device-select slots.
template <wwr::extension::error_type E>
using CreatePolicy = AbortPolicy<E>;

/// @brief The policy for destruction and free slots.
template <wwr::extension::error_type E>
using DestroyPolicy = LogPolicy<E>;

/// @brief wwr policy map binding CreatePolicy to alloc and DestroyPolicy to free,
/// for kit::buffer_suite / kit::device_buffer_suite.
struct PolicyMap {
  template <typename E>
  using alloc = CreatePolicy<E>;
  template <typename E>
  using free = DestroyPolicy<E>;
};

} // namespace free_breakfast
