/// @file
/// @brief Module wrapper re-exporting the GPU vector-type mapping and the
/// fixed-width integer types.
///
/// The vector-type mapping lives in vec.h so a device translation unit can
/// `#include` it directly -- a C++23 named module cannot be imported into a
/// .cu. This interface GMF-includes that header and re-exports its names, giving
/// host module code `import free_breakfast.core.types;` over the same
/// definitions. The fixed-width integer types and `size_t` come from the
/// `:integral` partition (integral.cppm over integral.h), re-exported here so a
/// single import covers both.

module;

#include "vec.h"

export module free_breakfast.core.types;

export import :integral;

namespace free_breakfast {

/// @brief Trait mapping a scalar type and an order N (1..4) to the matching
/// builtin vector type, exposed as the nested @c type.
export using free_breakfast::vector_type;

/// @brief The builtin vector type with scalar @c T and @c N components,
/// e.g. @c Vec<double,4> is @c double4 and @c Vec<float,3> is @c float3.
export using free_breakfast::Vec;

/// @brief Host/device factory that builds a @c Vec from its components,
/// generalising @c make_double4 / @c make_float3.
export using free_breakfast::make_vec;

} // namespace free_breakfast
