/// @file
/// @brief Core error handling types used throughout free-breakfast

export module free_breakfast.core.error_handling:error;

import std;

export namespace free_breakfast {

/// @brief Classification of errors that can occur in the system
///
/// Kept flat and small on purpose -- this is a dispatch tag for `to_exit_code`
/// and for callers that branch on failure category, not a deep taxonomy.
///
/// A few pairs are easy to confuse:
///
/// - `Divergence` vs `MaxIterationsExceeded`: `Divergence` means the iterate
///   itself went bad (blew up, went non-finite, stagnated) -- raising the
///   iteration cap cannot fix it. `MaxIterationsExceeded` means the process was
///   still converging normally and ran out of budget -- a knob, not a failure
///   of the method.
/// - `Unsupported` vs `InvalidConfig`: `Unsupported` is well-formed input
///   requesting functionality that is deliberately not implemented;
///   `InvalidConfig` is input that is itself wrong.
/// - `Numerical` vs `GpuRuntime`: `Numerical` is a computation that completed
///   but produced a mathematically invalid result (e.g. a solver reporting a
///   singular matrix); `GpuRuntime` is the GPU runtime/library call itself
///   failing (launch, driver, handle), under either backend.
///
/// Parse-vs-IO is deliberately NOT split: parse failures stay folded into
/// `IO`. See this module's README for the reasoning.
enum class ErrorKind {
  Allocation,
  IO,
  GpuRuntime,
  InvalidConfig,
  Divergence,
  MaxIterationsExceeded,
  Unsupported,
  Numerical
};

/**
 * @brief Structured error value carrying a kind, message, and source location
 *
 * Errors are constructed via named factory functions rather than constructors.
 * The source location is captured automatically at the call site.
 */
struct Error {
  ErrorKind kind;
  std::string message;
  std::source_location origin;

  /// @brief Create an Allocation error
  static Error allocation(std::string msg,
                          std::source_location loc = std::source_location::current()) {
    return {ErrorKind::Allocation, std::move(msg), loc};
  }
  /// @brief Create an IO error
  static Error io(std::string msg, std::source_location loc = std::source_location::current()) {
    return {ErrorKind::IO, std::move(msg), loc};
  }
  /// @brief Create a GpuRuntime error
  static Error gpu(std::string msg, std::source_location loc = std::source_location::current()) {
    return {ErrorKind::GpuRuntime, std::move(msg), loc};
  }
  /// @brief Create an InvalidConfig error
  static Error config(std::string msg, std::source_location loc = std::source_location::current()) {
    return {ErrorKind::InvalidConfig, std::move(msg), loc};
  }
  /// @brief Create a Divergence error -- the iterate blew up (not merely ran
  /// out of iterations; see @ref max_iterations_exceeded for that)
  static Error divergence(std::string msg,
                          std::source_location loc = std::source_location::current()) {
    return {ErrorKind::Divergence, std::move(msg), loc};
  }
  /// @brief Create a MaxIterationsExceeded error -- still converging normally,
  /// ran out of iteration budget (not a numerical blow-up; see @ref divergence
  /// for that)
  static Error max_iterations_exceeded(std::string msg,
                                       std::source_location loc = std::source_location::current()) {
    return {ErrorKind::MaxIterationsExceeded, std::move(msg), loc};
  }
  /// @brief Create an Unsupported error -- well-formed input requesting
  /// functionality that is deliberately not implemented
  static Error unsupported(std::string msg,
                           std::source_location loc = std::source_location::current()) {
    return {ErrorKind::Unsupported, std::move(msg), loc};
  }
  /// @brief Create a Numerical error -- a computation produced a
  /// mathematically invalid result (e.g. a singular matrix)
  static Error numerical(std::string msg,
                         std::source_location loc = std::source_location::current()) {
    return {ErrorKind::Numerical, std::move(msg), loc};
  }
};

/// @brief Alias for std::expected<T, Error> used as a failable return type
template <typename T>
using Result = std::expected<T, Error>;

/// @brief Alias for Result<void>, used when no value is returned on success
using Status = std::expected<void, Error>;

/// @brief Wrap an already-constructed Error as std::unexpected for direct return
inline std::unexpected<Error> err(Error e) {
  return std::unexpected(std::move(e));
}

/// @brief Construct an unexpected Allocation error for direct return from a Result/Status function
inline std::unexpected<Error>
err_allocation(std::string msg, std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::allocation(std::move(msg), loc));
}
/// @brief Construct an unexpected IO error for direct return from a Result/Status function
inline std::unexpected<Error> err_io(std::string msg,
                                     std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::io(std::move(msg), loc));
}
/// @brief Construct an unexpected GpuRuntime error for direct return from a Result/Status function
inline std::unexpected<Error> err_gpu(std::string msg,
                                      std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::gpu(std::move(msg), loc));
}
/// @brief Construct an unexpected InvalidConfig error for direct return from a Result/Status
/// function
inline std::unexpected<Error>
err_config(std::string msg, std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::config(std::move(msg), loc));
}
/// @brief Construct an unexpected Divergence error for direct return from a Result/Status function
inline std::unexpected<Error>
err_divergence(std::string msg, std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::divergence(std::move(msg), loc));
}
/// @brief Construct an unexpected MaxIterationsExceeded error for direct return from a
/// Result/Status function
inline std::unexpected<Error>
err_max_iterations_exceeded(std::string msg,
                            std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::max_iterations_exceeded(std::move(msg), loc));
}
/// @brief Construct an unexpected Unsupported error for direct return from a Result/Status
/// function
inline std::unexpected<Error>
err_unsupported(std::string msg, std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::unsupported(std::move(msg), loc));
}
/// @brief Construct an unexpected Numerical error for direct return from a Result/Status function
inline std::unexpected<Error>
err_numerical(std::string msg, std::source_location loc = std::source_location::current()) {
  return std::unexpected(Error::numerical(std::move(msg), loc));
}

/// @brief Map an ErrorKind to a process exit code
///
/// The numbers are pinned: a shell script branching on `$?` is an API
/// consumer, so a new kind extends this mapping rather than renumbering it.
inline std::int32_t to_exit_code(ErrorKind kind) {
  switch (kind) {
  case ErrorKind::Allocation:
    return 2;
  case ErrorKind::IO:
    return 4;
  case ErrorKind::GpuRuntime:
    return 3;
  case ErrorKind::InvalidConfig:
    return 1;
  case ErrorKind::Divergence:
    return 5;
  case ErrorKind::MaxIterationsExceeded:
    return 6;
  case ErrorKind::Unsupported:
    return 7;
  case ErrorKind::Numerical:
    return 8;
  }
  return 127;
}

} // namespace free_breakfast
