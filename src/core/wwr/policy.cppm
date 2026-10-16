/// @file
/// @brief The tree-wide error policies for the wwr RAII wrappers

export module free_breakfast.core.wwr:policy;

import std;
import :status;                            // gpu_status
import free_breakfast.core.error_handling; // check_or_abort
import free_breakfast.core.log;
import wwr.extension.common; // error_type, success_code, policy concepts

export namespace free_breakfast {

// Both policies are empty and noexcept, so each fits every wwr slot --
// including the destruction slots (nothrow_error_policy) -- and the wrappers'
// policy_slot collapses repeated ones to nothing.

/// @brief Log at critical with the failing call's location, then abort.
template <wwr::extension::error_type E>
struct AbortPolicy {
  /// @brief The error code type this policy handles, as wwr's concepts require.
  using error_type = E;
  /// @brief If `code` is a failure, log it at critical against `loc` and abort.
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
  /// @brief The error code type this policy handles, as wwr's concepts require.
  using error_type = E;
  /// @brief If `code` is a failure, log it at error against `loc`; never throws.
  void handle_error(E code, std::source_location loc) noexcept {
    // Formatting and logging allocate, so either can throw, and a throw out of
    // this noexcept function would terminate mid-teardown. Losing one log line
    // is the cheaper failure.
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
  /// @brief The policy for the alloc slot.
  template <typename E>
  using alloc = CreatePolicy<E>;
  /// @brief The policy for the free slot.
  template <typename E>
  using free = DestroyPolicy<E>;
};

} // namespace free_breakfast
