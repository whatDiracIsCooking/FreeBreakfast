/// @file
/// @brief The geometry-free shell *definition* -- @ref free_breakfast::ShellStamp,
/// a @ref free_breakfast::Shell with the center removed.
///
/// A @ref Shell embeds a center, so it describes a shell *over one geometry*.
/// The information a basis-set file actually carries per shell is
/// center-independent: an angular momentum and its primitives. @ref ShellStamp
/// is that centerless shell; @ref ShellStamp::at stamps it onto a real position
/// to produce a @ref Shell.
///
/// Everything stays in atomic units: primitive exponents in inverse bohr^2.
module;

#include "basis.h"

export module free_breakfast.basis:shell_stamp;

import std;
import free_breakfast.molecule;
import free_breakfast.core.error_handling;
import :shell;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief One contracted Gaussian shell without a center: an angular momentum L
/// and its primitives (@ref Primitive), the part of a @ref Shell that a basis-set
/// file specifies per element rather than per atom.
///
/// A @ref ShellStamp is a @ref Shell with the center removed. @ref at turns it
/// into a @ref Shell by supplying a center; @ref primitive_normalized folds the
/// per-primitive normalization into the coefficients (a center-independent
/// operation, so it belongs here and is computed once per stamp rather than
/// once per geometry).
class ShellStamp {
public:
  /// @param l           Angular momentum L (0 = s, 1 = p, 2 = d). No upper cap;
  ///                    the GPU engines separately reject L > @ref
  ///                    kMaxAngularMomentum.
  /// @param primitives  The contracted primitives, in contraction order.
  /// @param harmonic    Which functions this shell means: @c Cartesian (the
  ///                    default -- every existing caller gets today's
  ///                    behavior unchanged) or @c Spherical. For @p l <= 1 the
  ///                    two coincide, so the tag only matters from d upward.
  ///
  /// Aborts if @p l is negative or there are no primitives -- the same shape
  /// contract as @ref Shell, and the same reasoning: the one
  /// production caller, the BSE basis-set reader, already screens both
  /// conditions before constructing.
  ShellStamp(int l, std::vector<Primitive> primitives, Harmonic harmonic = Harmonic::Cartesian)
      : l_(l), primitives_(std::move(primitives)), harmonic_(harmonic) {
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
  ShellStamp(int l, const std::vector<double> &exponents, const std::vector<double> &coefficients,
             Harmonic harmonic = Harmonic::Cartesian)
      : ShellStamp(l, zip(exponents, coefficients), harmonic) {}

  /// @brief Angular momentum L of this shell (0 = s, 1 = p, 2 = d).
  [[nodiscard]] int angular_momentum() const { return l_; }

  /// @brief The contracted primitives, in contraction order.
  [[nodiscard]] const std::vector<Primitive> &primitives() const { return primitives_; }

  /// @brief Number of primitive Gaussians contracted into this shell.
  [[nodiscard]] std::uint32_t nprimitives() const {
    return static_cast<std::uint32_t>(primitives_.size());
  }

  /// @brief Which functions this shell means: @ref Harmonic::Cartesian or
  /// @ref Harmonic::Spherical.
  [[nodiscard]] Harmonic harmonic() const { return harmonic_; }

  /// @brief Number of Cartesian AO functions this shell contributes:
  /// (L+1)(L+2)/2, i.e. 1 (s), 3 (p), 6 (d).
  [[nodiscard]] std::uint32_t function_count() const { return cartesian_function_count(l_); }

  /// @brief Number of functions this shell actually presents to a basis:
  /// @ref function_count for a @c Cartesian shell, or @ref
  /// spherical_function_count (2l+1) for a @c Spherical one.
  [[nodiscard]] std::uint32_t presented_function_count() const {
    return harmonic_ == Harmonic::Spherical ? spherical_function_count(l_) : function_count();
  }

  /// @brief This shell centered at @p center: a @ref Shell carrying the same
  /// L, primitives and harmonic tag.
  [[nodiscard]] Shell at(const wwr::double3 &center) const {
    return Shell{l_, primitives_, center, harmonic_};
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
  /// rather than once per placed geometry. Applies identically regardless of
  /// @ref harmonic: every Cartesian component of a shell shares the same
  /// radial contraction, so a @c Spherical shell's harmonic combinations
  /// inherit this same per-primitive scale (see @c src/basis/harmonics.cppm).
  [[nodiscard]] ShellStamp primitive_normalized() const {
    double dfac = 1.0; // (2l-1)!!
    for (int m = (2 * l_) - 1; m > 1; m -= 2) {
      dfac *= static_cast<double>(m);
    }
    std::vector<Primitive> normalized = primitives_;
    for (Primitive &prim : normalized) {
      const double a = prim.exponent;
      const double norm = std::pow(2.0 * a / std::numbers::pi, 0.75) *
                          std::pow(4.0 * a, static_cast<double>(l_) / 2.0) / std::sqrt(dfac);
      prim.coefficient *= norm;
    }
    return ShellStamp{l_, std::move(normalized), harmonic_};
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
    for (std::size_t p = 0; p < exponents.size(); ++p) {
      out.push_back(Primitive{exponents[p], coefficients[p]});
    }
    return out;
  }

  int l_{0};
  std::vector<Primitive> primitives_{};
  Harmonic harmonic_{Harmonic::Cartesian};
};

} // namespace free_breakfast
