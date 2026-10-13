/// @file
/// @brief `free_breakfast.core.types` partition re-exporting the fixed-width
/// integer types and `size_t` from namespace free_breakfast.
///
/// The declarations live in integral_types.h so a device translation unit can
/// `#include` it directly (a named module cannot be imported into a .cu); this
/// partition GMF-includes that header and exports its names, giving host module
/// code the same `free_breakfast::int32_t` / `free_breakfast::size_t` through
/// `import free_breakfast.core.types;`.
module;

#include "integral_types.h"

export module free_breakfast.core.types:integral_types;

export namespace free_breakfast {

using free_breakfast::int16_t;
using free_breakfast::int32_t;
using free_breakfast::int64_t;
using free_breakfast::int8_t;
using free_breakfast::size_t;
using free_breakfast::uint16_t;
using free_breakfast::uint32_t;
using free_breakfast::uint64_t;
using free_breakfast::uint8_t;

} // namespace free_breakfast
