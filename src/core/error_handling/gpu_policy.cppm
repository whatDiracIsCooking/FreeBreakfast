/// @file
/// @brief Tree-wide error policies for the wwr RAII wrappers, chosen at build time

export module free_breakfast.core.error_handling:gpu_policy;

import :error;
import :gpu_status; // gpu_status, check_or_abort
import free_breakfast.core.log;
import wwr.extension.common; // error_type, success_code, policy concepts
import std;

// Set on this target only (src/core/error_handling/CMakeLists.txt). It is read once,
// here, into a constexpr that lands in this module's BMI -- so every importer sees
// the same value and no consumer TU can be compiled against the other one.
#ifndef FREE_BREAKFAST_GPU_POLICY_ABORTS
#error "FREE_BREAKFAST_GPU_POLICY_ABORTS is set by src/core/error_handling/CMakeLists.txt"
#endif

export namespace free_breakfast {

/// @brief True when this build's wrappers abort on a GPU failure (ABORT), false
/// when they record it for capture (RECORD). Follows
/// FREE_BREAKFAST_GPU_ERROR_POLICY, which defaults by build type.
inline constexpr bool aborts_on_error = FREE_BREAKFAST_GPU_POLICY_ABORTS != 0;

// ============================================================================
// The recording slot behind RecordPolicy and capture
// ============================================================================

/// @brief Hand an error to the innermost active capture on this thread.
///
/// First error wins: a later failure in the same capture is usually a
/// consequence of the first (a stream that was never created cannot bind a
/// handle), so the root cause is the one kept. With no capture active there is
/// nobody to return the error to, so it is also logged at error level -- a
/// RECORD-mode failure outside a capture is never silent.
void record_error(Error error) noexcept;

/// @brief Take (and clear) the error recorded on this thread, if any.
std::optional<Error> take_error() noexcept;

namespace detail {
/// Brackets one capture: stashes any error already pending for an outer
/// capture, and puts it back on the way out -- so an inner capture neither
/// reports its caller's error nor wipes it.
class CaptureScope {
public:
  CaptureScope() noexcept;
  ~CaptureScope();
  CaptureScope(const CaptureScope &) = delete;
  CaptureScope &operator=(const CaptureScope &) = delete;

private:
  std::optional<Error> outer_;
};
} // namespace detail

// ============================================================================
// Policies
// ============================================================================
// All three are empty and noexcept, so each fits every wwr slot -- including
// the destruction slots (nothrow_error_policy) -- and the wrappers' policy_slot
// collapses repeated ones to nothing. Empty is also what kit::DeviceHandle
// needs: it default-constructs its device policy rather than taking one.

/// @brief Log at critical with the failing call's location, then abort.
template <wwr::extension::error_type E>
struct AbortPolicy {
  using error_type = E;
  void handle_error(E code, std::source_location loc) noexcept {
    check_or_abort(gpu_status(code, loc));
  }
};

/// @brief Record the failure for the enclosing capture and carry on.
///
/// The wrapper is left empty (a handle's valid() is false, a buffer has
/// size 0), which is wwr's record-and-continue path.
template <wwr::extension::error_type E>
struct RecordPolicy {
  using error_type = E;
  void handle_error(E code, std::source_location loc) noexcept {
    try {
      if (auto status = gpu_status(code, loc); !status) {
        record_error(std::move(status).error());
      }
    } catch (...) {
      // Formatting the message ran out of memory. Nothing better to do from a
      // noexcept policy; the wrapper is still left empty.
    }
  }
};

/// @brief Log at error level and carry on.
///
/// For the destruction slots in RECORD mode: a failed destroy/free runs in a
/// destructor, after any capture that could have returned it has finished.
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

/// @brief This build's policy for creation, allocation and device-select slots.
template <wwr::extension::error_type E>
using CreatePolicy = std::conditional_t<aborts_on_error, AbortPolicy<E>, RecordPolicy<E>>;

/// @brief This build's policy for destruction and free slots.
template <wwr::extension::error_type E>
using DestroyPolicy = std::conditional_t<aborts_on_error, AbortPolicy<E>, LogPolicy<E>>;

/// @brief wwr policy map binding CreatePolicy to alloc and DestroyPolicy to free,
/// for kit::buffer_suite / kit::device_buffer_suite.
struct PolicyMap {
  template <typename E>
  using alloc = CreatePolicy<E>;
  template <typename E>
  using free = DestroyPolicy<E>;
};

// ============================================================================
// capture
// ============================================================================

/**
 * @brief Run `f`, returning its result -- or the first GPU error a RECORD-mode
 * policy saw while it ran -- as a Result.
 *
 * Write the wrapper constructions and calls inside the lambda: a wrapper's
 * defaulted std::source_location is then the lambda's line, so the Error's
 * origin points at your code rather than at this function.
 *
 * ```cpp
 * auto dev = capture([] { return std::make_shared<Device>(0); });
 * if (!dev) return std::unexpected(std::move(dev).error());
 * ```
 *
 * Under ABORT nothing is ever recorded -- a failure aborts inside `f` -- so the
 * result is always a value, and the same caller code builds in both modes.
 * Nesting is safe; see detail::CaptureScope.
 *
 * @return Result<R> for a lambda returning R; Status for one returning void.
 */
template <std::invocable F>
Result<std::invoke_result_t<F>> capture(F &&f) {
  using R = std::invoke_result_t<F>;
  detail::CaptureScope scope;
  if constexpr (std::is_void_v<R>) {
    std::invoke(std::forward<F>(f));
    if (auto e = take_error()) {
      return std::unexpected(std::move(*e));
    }
    return {};
  } else {
    R value = std::invoke(std::forward<F>(f));
    if (auto e = take_error()) {
      return std::unexpected(std::move(*e));
    }
    return value;
  }
}

} // namespace free_breakfast
