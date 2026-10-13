/// @file
/// @brief The center-independent pieces of the Gaussian basis description: the
/// Cartesian function count and the primitive.
///
/// These carry no @c wwr::double3, so they live in a plain header that
/// both CUDA `#include`rs and the `basis` module share. @c Shell and @c BasisSet
/// embed a center and stay in @c basis.cppm, which reaches @c wwr::double3 by
/// importing @c free_breakfast.molecule; this header is included in that
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
[[nodiscard]] constexpr uint32_t cartesian_function_count(int32_t l) {
  const auto n = static_cast<uint32_t>(l);
  return ((n + 1) * (n + 2)) / 2;
}

/// @brief Number of real solid harmonic (spherical) functions in a shell of
/// momentum @p l: 2l+1.
[[nodiscard]] constexpr uint32_t spherical_function_count(int32_t l) {
  const auto n = static_cast<uint32_t>(l);
  return (2 * n) + 1;
}

/// @brief Which linear combination of a shell's @c cartesian_function_count(l)
/// Cartesian Gaussians a basis-set file means: the (L+1)(L+2)/2 Cartesian
/// monomials themselves, or the 2l+1 real solid harmonics embedded in them.
///
/// Center-independent, so it lives alongside @ref cartesian_function_count in
/// the same CUDA-shareable header. The GPU integral engines only ever
/// evaluate Cartesian shells; a @c Spherical shell is Cartesian at the kernel
/// level too, with the harmonic combination formed afterward via a
/// Cartesian-to-spherical transform matrix (@c BasisSet's
/// @c cartesian_to_spherical) -- see @c src/basis/harmonics.cppm. For l <= 1
/// the two conventions coincide exactly (3 Cartesian p functions ARE the 3
/// spherical p functions), so @c Cartesian is the only meaningful tag there.
enum class Harmonic { Cartesian, Spherical };

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
