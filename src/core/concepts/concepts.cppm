/**
 * @file concepts.cppm
 * @brief The scalar type-trait concepts, re-exported from WarpWraps into
 * namespace free_breakfast.
 *
 * real_fp and int_type are defined once, in wwr.wrappers.common, and this
 * module only re-exports them -- it does not redefine them. A second
 * definition here would be a different concept that happens to share the name,
 * so a template constrained on free_breakfast::int_type and one constrained on
 * wwr::int_type could disagree about the same type.
 *
 * int_type is exactly {int, int64_t}, with no convertible_to arm: that arm
 * admitted double, float and bool, so a std::size_t caller casts explicitly.
 * wwr's int_types.cppm has the full reasoning.
 *
 * The GPU-bound concepts that build on real_fp (complex_fp, half_fp, usual_fp,
 * ComplexToRealType, ...) are not re-exported here; import wwr.wrappers.common
 * for those.
 *
 * Usage:
 *   import free_breakfast.core.concepts;
 *   using namespace free_breakfast;
 */

export module free_breakfast.core.concepts;

import std;
import wwr.wrappers.common;

export namespace free_breakfast {

// ========================================================================
// Re-export standard integer types used alongside int_type
// ========================================================================
using std::int64_t;
using std::size_t;

// ========================================================================
// Concepts
// ========================================================================

/// @brief T is float or double.
using wwr::real_fp;

/// @brief T is exactly int or int64_t.
using wwr::int_type;

}  // namespace free_breakfast
