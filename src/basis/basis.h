/// @file
/// @brief The center-independent pieces of the Gaussian basis description: the
/// Cartesian and spherical function counts and the primitive.
///
/// These carry no center, so they live in a plain header that both CUDA
/// `#include`rs and the `basis` module share. @c Shell and @c BasisSet embed a
/// @c Vec<double, 3> center and stay in the module; this header is included in that
/// module's global module fragment and its names re-exported from within the
/// namespace, so importers and CUDA code see one entity, not two.
///
/// Everything is in atomic units: primitive exponents are in inverse bohr
/// squared. See @ref free_breakfast::Shell for the contraction and the Cartesian
/// ordering convention.
///
/// It includes @c <cstdint> rather than relying on `import std`, so it works
/// unchanged in a module's global module fragment and in CUDA / HIP device code.

#pragma once

#include "integral_types.h"

namespace free_breakfast {

/// @brief Number of Cartesian Gaussian functions in a shell of momentum @p l.
///
/// A Cartesian shell of angular momentum L spans x^a y^b z^c with a+b+c = L,
/// and there are (L+1)(L+2)/2 such triples: 1 (s), 3 (p), 6 (d), 10 (f), ...
///
/// This is where a signed angular momentum becomes an unsigned count; callers
/// keep @p l signed for its arithmetic and the count unsigned for indexing.
///
/// @pre @p l >= 0 (@ref Shell and @ref ShellStamp abort on a negative one).
[[nodiscard]] constexpr uint32_t function_count_cart(int32_t l) {
  const auto n = static_cast<uint32_t>(l);
  return ((n + 1) * (n + 2)) / 2;
}

/// @brief Number of real solid harmonic (spherical) functions in a shell of
/// momentum @p l: 2l+1.
///
/// Every shell in this tree is spherical: this is the count a shell presents
/// to the outside world, while the GPU kernels evaluate its @ref
/// function_count_cart Cartesian functions and @c BasisSet's
/// @c spherical_transform_matrix combines them (see @c src/basis/harmonics.cppm).
///
/// @pre @p l >= 0, as for @ref function_count_cart.
[[nodiscard]] constexpr uint32_t function_count_sphe(int32_t l) {
  const auto n = static_cast<uint32_t>(l);
  return (2 * n) + 1;
}

/// @brief One primitive Gaussian in a contraction: an exponent and the
/// contraction coefficient that multiplies it.
///
/// A primitive belongs to exactly one @ref Shell and shares that shell's
/// angular momentum and center, so neither is repeated here -- storing them
/// per primitive would be state that can disagree with its own shell. Pairing
/// the two numbers that ARE per-primitive is what keeps them from drifting out
/// of step: every consumer reads them together, and there is no longer a length
/// invariant to enforce.
///
/// Plain aggregate, brace-initialised (`Primitive{alpha, c}`). The exponent
/// is in inverse bohr^2; the coefficient is stored exactly as supplied, with no
/// normalization applied (see @ref Shell).
struct Primitive {
  double exponent{0.0};
  double coefficient{0.0};
};

} // namespace free_breakfast
