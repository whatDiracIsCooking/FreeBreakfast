/// @file
/// @brief One contracted Gaussian shell (@ref free_breakfast::Shell) and the
/// center-independent basis primitives it is built from.
///
/// The center-independent pieces -- @ref free_breakfast::kMaxAngularMomentum (in
/// @c angular_momentum.h), and @ref free_breakfast::cartesian_function_count and
/// @ref free_breakfast::Primitive (in @c basis.h) -- are included in this
/// partition's global module fragment and re-exported from within the namespace,
/// so importers and CUDA `#include`rs share one entity. @ref free_breakfast::Shell
/// embeds a center, a @c Vec<double, 3> from @c free_breakfast.core.types.
///
/// Everything is in atomic units: shell centers are in bohr (a @c
/// Vec<double, 3>), and primitive exponents are in inverse
/// bohr squared. A shell is a fixed contraction of primitive Gaussians that all
/// share one center and one angular momentum L; each primitive is an
/// (exponent, coefficient) pair (@ref free_breakfast::Primitive).
///
/// **Function convention: Cartesian.** A shell of angular momentum L spans the
/// Cartesian Gaussians x^a y^b z^c with a+b+c = L, of which there are
/// (L+1)(L+2)/2 -- so 1 for L=0 (s), 3 for L=1 (p), 6 for L=2 (d). The working
/// target is up to d, i.e. L in {0, 1, 2}. The Cartesian ordering within a
/// shell is the canonical lexicographic-on-(a,b,c)-with-a-highest order used by
/// most integral libraries; for p it is (x, y, z), and for d it is
/// (xx, xy, xz, yy, yz, zz).
module;

#include "angular_momentum.h"
#include "basis.h"

export module free_breakfast.basis:shell;

import free_breakfast.molecule;
import free_breakfast.core.error_handling;
import std;
import free_breakfast.core.types;

#include "diagnostics.h"

export namespace free_breakfast {

// Re-export the center-independent names defined in basis.h (global module
// fragment). The using-declarations must sit inside `namespace free_breakfast`
// so importers can form declarations with them; at global scope the names would
// be visible but unusable.
using free_breakfast::cartesian_function_count;
using free_breakfast::Harmonic;
using free_breakfast::kMaxAngularMomentum;
using free_breakfast::Primitive;
using free_breakfast::spherical_function_count;

/// @brief One contracted Gaussian shell: an angular momentum, a set of
/// primitives (@ref Primitive) and a center.
///
/// A shell is the atomic unit of a basis set. The center is a position in bohr,
/// typically the coordinate of the atom the shell sits on.
///
/// **Coefficients are stored as supplied.** No primitive or contraction
/// normalization is applied here; the integral engines take coefficients as given,
/// so a caller that wants normalized primitives bakes them into the basis with
/// @c BasisStamp::primitive_normalized (the fused-Fock path, and what the GPU
/// one-electron/quadrupole engines are fed).
class Shell {
public:
  /// @param l           Angular momentum L (0 = s, 1 = p, 2 = d). No upper cap:
  ///                    the host reference engines evaluate arbitrary L; the GPU
  ///                    engines separately reject L > @ref kMaxAngularMomentum.
  /// @param primitives  The contracted primitives, in contraction order.
  /// @param center      Shell center, in bohr.
  /// @param harmonic    Which functions this shell means: @c Cartesian (the
  ///                    default) or @c Spherical. The GPU kernels evaluate
  ///                    this shell's Cartesian functions either way; a
  ///                    @c Spherical tag says a @ref BasisSet-level transform
  ///                    (@c cartesian_to_spherical) should combine them into
  ///                    2l+1 real solid harmonics afterward.
  ///
  /// Aborts if @p l is negative or there are no primitives -- a caller-side
  /// invariant (the one production caller, the BSE basis-set reader, already
  /// screens both conditions before constructing), not a
  /// value a caller could plausibly recover from.
  Shell(int32_t l, std::vector<Primitive> primitives, Vec<double, 3> center,
        Harmonic harmonic = Harmonic::Cartesian)
      : l_(l), primitives_(std::move(primitives)), center_(center), harmonic_(harmonic) {
    if (l_ < 0) {
      check_or_abort(err_config("Shell: angular momentum must be non-negative"));
    }
    if (primitives_.empty()) {
      check_or_abort(err_config("Shell: a shell needs at least one primitive"));
    }
  }

  /// @brief Convenience constructor zipping two parallel arrays.
  ///
  /// Basis-set files give exponents and coefficients as separate lists (the BSE
  /// JSON does), so the zip belongs at that boundary and nowhere else. This is
  /// the ONE place the two lists must agree in length, and the only place that
  /// check now lives.
  ///
  /// @param l             Angular momentum L (0 = s, 1 = p, 2 = d).
  /// @param exponents     Primitive Gaussian exponents, in inverse bohr^2.
  /// @param coefficients  Contraction coefficients, one per exponent.
  /// @param center        Shell center, in bohr.
  /// @param harmonic      See the other constructor.
  ///
  /// Aborts on the conditions above, or if the exponent and coefficient
  /// counts differ (see the other constructor).
  Shell(int32_t l, const std::vector<double> &exponents, const std::vector<double> &coefficients,
        Vec<double, 3> center, Harmonic harmonic = Harmonic::Cartesian)
      : Shell(l, zip(exponents, coefficients), center, harmonic) {}

  /// @brief Angular momentum L of this shell (0 = s, 1 = p, 2 = d).
  [[nodiscard]] int32_t angular_momentum() const { return l_; }

  /// @brief The contracted primitives, in contraction order.
  [[nodiscard]] const std::vector<Primitive> &primitives() const { return primitives_; }

  /// @brief Number of primitive Gaussians contracted into this shell.
  [[nodiscard]] uint32_t nprimitives() const { return static_cast<uint32_t>(primitives_.size()); }

  /// @brief Shell center, in bohr.
  [[nodiscard]] const Vec<double, 3> &center() const { return center_; }

  /// @brief Which functions this shell means: @ref Harmonic::Cartesian or
  /// @ref Harmonic::Spherical.
  [[nodiscard]] Harmonic harmonic() const { return harmonic_; }

  /// @brief Number of Cartesian AO functions this shell contributes.
  ///
  /// (L+1)(L+2)/2: 1 for s, 3 for p, 6 for d. This is the count the GPU
  /// kernels and @ref BasisSet's Cartesian AO layout always use, regardless
  /// of @ref harmonic -- see @ref presented_function_count for the count a
  /// @c Spherical shell actually presents to the outside world.
  [[nodiscard]] uint32_t function_count() const { return cartesian_function_count(l_); }

  /// @brief Number of functions this shell actually presents: @ref
  /// function_count for a @c Cartesian shell, or 2l+1 (@ref
  /// spherical_function_count) for a @c Spherical one.
  [[nodiscard]] uint32_t presented_function_count() const {
    return harmonic_ == Harmonic::Spherical ? spherical_function_count(l_) : function_count();
  }

private:
  /// @brief Pair two equal-length arrays into primitives, or abort.
  [[nodiscard]] static std::vector<Primitive> zip(const std::vector<double> &exponents,
                                                  const std::vector<double> &coefficients) {
    if (exponents.size() != coefficients.size()) {
      check_or_abort(err_config("Shell: exponent and coefficient counts must match"));
    }
    std::vector<Primitive> out;
    out.reserve(exponents.size());
    for (size_t p = 0; p < exponents.size(); ++p) {
      out.push_back(Primitive{exponents[p], coefficients[p]});
    }
    return out;
  }

  int32_t l_{0};
  std::vector<Primitive> primitives_{};
  Vec<double, 3> center_{};
  Harmonic harmonic_{Harmonic::Cartesian};
};

} // namespace free_breakfast
