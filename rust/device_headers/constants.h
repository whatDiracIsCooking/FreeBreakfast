/// @file
/// @brief The mathematical constants rust/'s launch wrappers use, for device
/// code.
///
/// Forked from src/core/constants/constants.h and trimmed to what they use:
/// kPi, its literal unchanged. The other constants, the typed literals and
/// their complex specializations come back with the first wrapper that names
/// them; the host side's are Rust's own (fb_core_constants). Nothing here
/// crosses the C ABI, so it stays C++ (rust/docs/gpu_layer.md, "Shared
/// types").

#pragma once

namespace free_breakfast {

/// @brief Mathematical constant pi (3.141592653589793)
template <typename T>
constexpr T kPi = static_cast<T>(3.141592653589793);

} // namespace free_breakfast
