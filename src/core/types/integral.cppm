/// @file
/// @brief `free_breakfast.core.types` partition re-exporting fixed-width
/// integer types and `size_t`.
///
/// The definitions live in integral.h so a device translation unit can `#include`
/// it directly (a named module cannot be imported into a .cu); this partition
/// GMF-includes that header and re-exports its names into the global namespace,
/// giving host module code the same types through
/// `import free_breakfast.core.types;`.
module;

#include "integral.h"

export module free_breakfast.core.types:integral;

export using ::int8_t;
export using ::int16_t;
export using ::int32_t;
export using ::int64_t;
export using ::uint8_t;
export using ::uint16_t;
export using ::uint32_t;
export using ::uint64_t;
export using ::size_t;
