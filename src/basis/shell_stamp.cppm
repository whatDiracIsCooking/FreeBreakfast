/// @file
/// @brief The geometry-free shell *definition* -- @ref free_breakfast::ShellStamp,
/// a @ref free_breakfast::Shell with the center removed.
///
/// A @ref Shell embeds a center, so it describes a shell *over one geometry*.
/// The information a basis-set file actually carries per shell is
/// center-independent: an angular momentum and its primitives. @ref ShellStamp
/// is that centerless shell; @ref place stamps it onto a real position to
/// produce a @ref Shell.
///
/// Everything stays in atomic units: primitive exponents in inverse bohr^2.
module;

#include "basis.h"

export module free_breakfast.basis:shell_stamp;

import std;
import free_breakfast.core.types;
import free_breakfast.core.primitives;
import free_breakfast.molecule;
import free_breakfast.core.error_handling;
import :shell;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief One contracted Gaussian shell without a center: an angular momentum L
/// and its primitives (@ref Primitive), the part of a @ref Shell that a basis-set
/// file specifies per element rather than per atom.
///
/// A @ref ShellStamp is a @ref Shell with the center removed. @ref place turns
/// it into a @ref Shell by supplying a center; @ref primitive_normalized folds the
/// per-primitive normalization into the coefficients (a center-independent
/// operation, so it belongs here and is computed once per stamp rather than
/// once per geometry).
class ShellStamp {
public:
  /// @param l           Angular momentum L (0 = s, 1 = p, 2 = d). No upper cap;
  ///                    the GPU engines separately reject L > @ref
  ///                    kMaxAngularMomentum.
  /// @param primitives  The contracted primitives, in contraction order.
  ///
  /// Aborts if @p l is negative or there are no primitives -- the same shape
  /// contract as @ref Shell, and the same reasoning: the one
  /// production caller, the BSE basis-set reader, already screens both
  /// conditions before constructing.
  ShellStamp(int32_t l, std::vector<Primitive> primitives)
      : l_(l), primitives_(std::move(primitives)) {
    if (l_ < 0) {
      check_or_abort(err_config("ShellStamp: angular momentum must be non-negative"));
    }
    if (primitives_.empty()) {
      check_or_abort(err_config("ShellStamp: a shell needs at least one primitive"));
    }
  }

  /// @brief Convenience constructor zipping parallel exponent/coefficient arrays,
  /// as basis-set files give them. Mirrors the @ref Shell zipping constructor.
  ///
  /// Aborts on the conditions above, or if the exponent and coefficient
  /// counts differ.
  ShellStamp(int32_t l, const std::vector<double> &exponents,
             const std::vector<double> &coefficients)
      : ShellStamp(l, zip(exponents, coefficients)) {}

  /// @brief Angular momentum L of this shell (0 = s, 1 = p, 2 = d).
  [[nodiscard]] int32_t angular_momentum() const { return l_; }

  /// @brief The contracted primitives, in contraction order.
  [[nodiscard]] const std::vector<Primitive> &primitives() const { return primitives_; }

  /// @brief Number of primitive Gaussians contracted into this shell.
  [[nodiscard]] uint32_t nprimitives() const { return static_cast<uint32_t>(primitives_.size()); }

  /// @brief Number of Cartesian AO functions this shell contributes:
  /// (L+1)(L+2)/2, i.e. 1 (s), 3 (p), 6 (d).
  [[nodiscard]] uint32_t function_count_cart() const {
    return free_breakfast::function_count_cart(l_);
  }

  /// @brief Number of real solid harmonic functions this shell presents to a
  /// basis: 2l+1.
  [[nodiscard]] uint32_t function_count_sphe() const {
    return free_breakfast::function_count_sphe(l_);
  }

  /// @brief A copy with the per-primitive normalization folded into the
  /// contraction coefficients.
  ///
  /// Standard basis-set coefficients are given for primitive-normalized
  /// primitives, and the integral engines take coefficients as given, so this bakes
  /// the normalization into the basis once -- the fused build_fock path, and what
  /// the GPU one-electron/quadrupole engines are fed. The scale is
  ///   N(alpha, l) = (2 alpha / pi)^{3/4} (4 alpha)^{l/2} / sqrt((2l-1)!!),
  /// the (l,0,0) primitive normalization. It depends only on the exponent and L,
  /// never the center, so it is a property of the *stamp*: computed once here
  /// rather than once per placed geometry. Every Cartesian component of a
  /// shell shares the same radial contraction, so the harmonic combinations
  /// @c spherical_transform_matrix forms inherit this same per-primitive scale
  /// (see @c src/basis/harmonics.cppm).
  [[nodiscard]] ShellStamp primitive_normalized() const {
    const double dfac = math::double_factorial_odd((2 * l_) - 1);
    std::vector<Primitive> normalized = primitives_;
    for (Primitive &prim : normalized) {
      const double a = prim.exponent;
      const double norm = std::pow(2.0 * a / std::numbers::pi, 0.75) *
                          std::pow(4.0 * a, static_cast<double>(l_) / 2.0) / std::sqrt(dfac);
      prim.coefficient *= norm;
    }
    return ShellStamp{l_, std::move(normalized)};
  }

private:
  /// @brief Pair two equal-length arrays into primitives, or abort. Mirrors the
  /// @ref Shell zipping constructor's check -- the one place the two lists must
  /// agree in length.
  [[nodiscard]] static std::vector<Primitive> zip(const std::vector<double> &exponents,
                                                  const std::vector<double> &coefficients) {
    if (exponents.size() != coefficients.size()) {
      check_or_abort(err_config("ShellStamp: exponent and coefficient counts must match"));
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
};

/// @brief @p stamp centered at @p center: a @ref Shell carrying the same L and
/// primitives.
[[nodiscard]] Shell place(const ShellStamp &stamp, const Vec<double, 3> &center) {
  return Shell{stamp.angular_momentum(), stamp.primitives(), center};
}

} // namespace free_breakfast
