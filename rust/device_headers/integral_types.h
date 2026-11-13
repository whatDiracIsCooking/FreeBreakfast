/// @file
/// @brief Fixed-width integer types and size_t for host and device code,
/// declared into namespace free_breakfast.
///
/// Forked from src/core/types/integral_types.h, unchanged: rust/'s launch
/// wrappers include this copy and never src/'s (rust/README.md, "What is
/// here").
#pragma once

#include <cstddef>
#include <cstdint>

namespace free_breakfast {

using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::int8_t;
using std::size_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::uint8_t;

} // namespace free_breakfast
