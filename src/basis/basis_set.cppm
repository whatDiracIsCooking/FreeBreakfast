/// @file
/// @brief An ordered collection of shells with atomic-orbital indexing
/// (@ref free_breakfast::BasisSet), and the Cartesian-to-spherical transform
/// (@ref free_breakfast::spherical_transform_matrix) built from it.
///
/// @ref free_breakfast::BasisSet is built over @ref free_breakfast::Shell,
/// imported from the @c :shell partition, and its @ref free_breakfast::AO list
/// over the @ref free_breakfast::cartesian_powers ordering, imported from
/// @c :ao. Its @c basis.h include in the global module fragment is only for the
/// global @c std::uint32_t used by the AO-count accessors; the center-independent
/// basis names are re-exported by @c :shell.
///
/// **Two parallel AO layouts.** Every shell is spherical, but the GPU kernels
/// evaluate its @c cartesian_function_count(l) Cartesian functions, so each
/// shell has a slot in the Cartesian layout (@ref BasisSet::aos, @ref
/// BasisSet::ao_offset) sized (l+1)(l+2)/2 AND one in the spherical layout
/// (@ref BasisSet::nao_sphe, @ref BasisSet::spherical_ao_offset) sized
/// 2l+1. @ref spherical_transform_matrix bridges the two: a block-diagonal matrix
/// @c U (Cartesian rows x spherical columns) holding each shell's
/// (normalization-reconciled) real solid harmonic embedding (@c :harmonics) --
/// so @c U^T M U converts any Cartesian-basis matrix @c M into the same
/// quantity over the spherical basis.
module;

#include "basis.h"

export module free_breakfast.basis:basis_set;

import std;
import free_breakfast.core.types;
import free_breakfast.core.primitives;
import :shell;
import :ao;
import :harmonics;
import free_breakfast.core.error_handling;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief An ordered collection of shells with atomic-orbital indexing.
///
/// The shells are held in the order supplied, and the atomic orbitals (AOs) are
/// laid out by concatenating each shell's Cartesian functions in that same
/// order: shell 0 owns AOs [0, n0), shell 1 owns [n0, n0 + n1), and so on,
/// where n_i is shell i's @ref Shell::function_count. @ref ao_offset returns
/// the first AO index of a shell and @ref nao_cart the total across all shells;
/// @ref aos enumerates the same layout one @ref AO (shell plus Cartesian
/// powers) at a time.
///
/// A basis made by @ref place also records which atom owns which shells
/// (@ref natoms, @ref atom_shell_offset, @ref atom_shell_count), so per-atom
/// work -- the SAD guess, population analysis -- reads the grouping instead of
/// re-deriving it from shell centers.
class BasisSet {
public:
  BasisSet() = default;

  /// @param shells The shells, in the order that fixes the AO layout. The basis
  ///               carries no per-atom shell map (@ref natoms is 0).
  explicit BasisSet(std::vector<Shell> shells) : shells_(std::move(shells)) { build_layouts(); }

  /// @param shells             The shells, in the order that fixes the AO layout.
  /// @param atom_shell_offsets One entry per atom plus a trailing total: atom
  ///                           @c a owns shells [offsets[a], offsets[a + 1]). It
  ///                           must start at 0, never decrease and end at
  ///                           @c shells.size(); anything else aborts, since the
  ///                           one producer (@ref place) builds it from the same
  ///                           loop that builds @p shells.
  BasisSet(std::vector<Shell> shells, std::vector<uint32_t> atom_shell_offsets)
      : shells_(std::move(shells)), atom_shell_offsets_(std::move(atom_shell_offsets)) {
    const bool valid = !atom_shell_offsets_.empty() && atom_shell_offsets_.front() == 0 &&
                       atom_shell_offsets_.back() == shells_.size() &&
                       std::ranges::is_sorted(atom_shell_offsets_);
    if (!valid) {
      check_or_abort(err_config("BasisSet: atom shell offsets do not partition the shells"));
    }
    build_layouts();
  }

  /// @brief Number of atoms the shells are grouped by: the molecule's atom count
  /// for a basis made by @ref place, 0 for one built from a bare shell list (or
  /// moved from).
  [[nodiscard]] uint32_t natoms() const {
    return atom_shell_offsets_.empty() ? 0 : static_cast<uint32_t>(atom_shell_offsets_.size() - 1);
  }

  /// @brief First shell owned by atom @p atom; the atom owns shells
  /// [atom_shell_offset(atom), atom_shell_offset(atom) + atom_shell_count(atom)).
  ///
  /// Aborts if @p atom is not below @ref natoms (see @ref ao_offset).
  [[nodiscard]] uint32_t atom_shell_offset(uint32_t atom) const {
    if (atom >= natoms()) {
      check_or_abort(err_config("BasisSet::atom_shell_offset: atom index out of range"));
    }
    return atom_shell_offsets_[atom];
  }

  /// @brief Number of shells owned by atom @p atom. Aborts if @p atom is not
  /// below @ref natoms.
  [[nodiscard]] uint32_t atom_shell_count(uint32_t atom) const {
    if (atom >= natoms()) {
      check_or_abort(err_config("BasisSet::atom_shell_count: atom index out of range"));
    }
    return atom_shell_offsets_[atom + 1] - atom_shell_offsets_[atom];
  }

  /// @brief The shells, in the order supplied.
  [[nodiscard]] const std::vector<Shell> &shells() const { return shells_; }

  /// @brief The atomic orbitals, in global AO order (length @ref nao_cart).
  ///
  /// Entry @c mu is the AO at global index @c mu: its @c shell_index names the
  /// owning shell and its (a, b, c) its Cartesian powers. Concatenates each
  /// shell's @ref cartesian_powers in shell order, so it matches the @ref
  /// ao_offset layout exactly.
  [[nodiscard]] const std::vector<AO> &aos() const { return aos_; }

  /// @brief Number of shells in the basis.
  [[nodiscard]] uint32_t nshells() const { return static_cast<uint32_t>(shells_.size()); }

  /// @brief Total number of atomic orbitals, Sum over shells of their
  /// function counts.
  ///
  /// Zero for a moved-from basis: moving takes the offset vector, trailing
  /// total included, so the empty check keeps this from reading past it.
  [[nodiscard]] uint32_t nao_cart() const { return ao_offsets_.empty() ? 0 : ao_offsets_.back(); }

  /// @brief First AO index owned by shell @p shell_index.
  ///
  /// Shell i owns AOs [ao_offset(i), ao_offset(i) + shells()[i].function_count()).
  /// Aborts if @p shell_index is not a valid shell -- every call site in this
  /// tree loops over @c 0..nshells() itself (never a raw index from outside
  /// the basis), so out-of-range here is a caller bug, the same posture as
  /// @c std::vector::at.
  [[nodiscard]] uint32_t ao_offset(uint32_t shell_index) const {
    if (shell_index >= shells_.size()) {
      check_or_abort(err_config("BasisSet::ao_offset: shell index out of range"));
    }
    return ao_offsets_[shell_index];
  }

  /// @brief Total number of AOs in the SPHERICAL layout: sum over shells of
  /// @ref Shell::spherical_function_count (2l+1). Equals @ref nao_cart exactly when
  /// no shell is above p. Zero for a moved-from basis, as @ref nao_cart is.
  [[nodiscard]] uint32_t nao_sphe() const {
    return spherical_ao_offsets_.empty() ? 0 : spherical_ao_offsets_.back();
  }

  /// @brief First SPHERICAL-layout AO index owned by shell @p shell_index; the
  /// spherical-layout counterpart of @ref ao_offset.
  ///
  /// Aborts if @p shell_index is not a valid shell (see @ref ao_offset).
  [[nodiscard]] uint32_t spherical_ao_offset(uint32_t shell_index) const {
    if (shell_index >= shells_.size()) {
      check_or_abort(err_config("BasisSet::spherical_ao_offset: shell index out of range"));
    }
    return spherical_ao_offsets_[shell_index];
  }

private:
  // Fills the Cartesian and spherical AO layouts from shells_.
  void build_layouts() {
    // Rebuild from scratch: the default member initializer seeds a single
    // trailing 0, which is right for an empty basis but must not remain in
    // front of the offsets computed here.
    ao_offsets_.clear();
    ao_offsets_.reserve(shells_.size() + 1);
    spherical_ao_offsets_.clear();
    spherical_ao_offsets_.reserve(shells_.size() + 1);
    aos_.clear();
    const uint32_t nsh = static_cast<uint32_t>(shells_.size());
    uint32_t offset = 0;
    uint32_t spherical_offset = 0;
    for (uint32_t s = 0; s < nsh; ++s) {
      const Shell &shell = shells_[s];
      ao_offsets_.push_back(offset);
      spherical_ao_offsets_.push_back(spherical_offset);
      // One AO per Cartesian function, in the canonical order (:ao owns it), so
      // aos_ and the ao_offsets_ slices describe the identical layout. The GPU
      // kernels evaluate the shell's Cartesian functions; see the file doc
      // comment.
      for (const std::array<int32_t, 3> &p : cartesian_powers(shell.angular_momentum())) {
        aos_.push_back(AO{s, static_cast<uint8_t>(p[0]), static_cast<uint8_t>(p[1]),
                          static_cast<uint8_t>(p[2])});
      }
      offset += shell.function_count();
      spherical_offset += shell.spherical_function_count();
    }
    ao_offsets_.push_back(offset);
    spherical_ao_offsets_.push_back(spherical_offset);
  }

  std::vector<Shell> shells_{};
  // One offset per shell plus a trailing total, so nao_cart() is the last element.
  // A default-constructed (empty) basis still needs the trailing 0 so nao_cart()
  // reads it -- hence a single-element {0}, not an empty vector. A moved-from
  // basis is the one state where it IS empty; nao_cart() checks for that.
  std::vector<uint32_t> ao_offsets_ = std::vector<uint32_t>(1, 0);
  // The same layout, sized by spherical_function_count() instead of
  // function_count() -- see nao_sphe()/spherical_ao_offset().
  std::vector<uint32_t> spherical_ao_offsets_ = std::vector<uint32_t>(1, 0);
  // One AO per Cartesian function, global AO order; empty for an empty basis.
  std::vector<AO> aos_{};
  // One offset per atom plus a trailing total; empty when the basis carries no
  // per-atom map. See natoms().
  std::vector<uint32_t> atom_shell_offsets_{};
};

/// @brief The Cartesian-to-spherical transform @c U for @p basis: a
/// @c basis.nao_cart() x @c basis.nao_sphe() column-major matrix (element
/// (i, j) at @c i + j*nao_cart(), so it goes straight into a BLAS call) such that
/// @c U^T M U converts a Cartesian-AO-basis matrix @p M (an overlap, a Fock
/// matrix, ...) into the same quantity over @p basis's spherical AO layout.
///
/// Block-diagonal, one block per shell at (@ref BasisSet::ao_offset, @ref
/// BasisSet::spherical_ao_offset): for a shell of momentum @c l, the
/// closed-form real solid harmonic table @c solid_harmonic_block(l)
/// (@c :harmonics) -- ncart(l) x (2l+1), the purely angular change of basis
/// from Cartesian monomials to the 2l+1 real solid harmonics (@c m = -l..+l,
/// PySCF/libcint convention) -- SCALED by @c sqrt((2l-1)!!).
///
/// **Why that scale, and why nothing else.** Every Cartesian component of a
/// shell shares the identical radial contraction (@ref Shell stores one
/// primitive list for the whole shell), so the harmonic embedding combines
/// them correctly regardless of how that shared radial part is normalized --
/// no per-Cartesian-component correction is needed. What DOES need
/// reconciling is a single per-shell scale: @c solid_harmonic_block's columns
/// are orthonormal under the RAW (unnormalized) monomial Gaussian-moment inner
/// product, while this repo's actual Cartesian AOs carry @ref
/// ShellStamp::primitive_normalized's scale, which normalizes the (l,0,0)
/// Cartesian component (self-overlap exactly 1 for a single normalized
/// primitive) and applies that SAME scale uniformly to every other Cartesian
/// component of the shell (so e.g. a d shell's xy component has self-overlap
/// 1/3, not 1, in this repo's convention -- (2l-1)!! is exactly the ratio
/// between the (l,0,0) and raw-monomial normalizations). Multiplying the
/// harmonic block by @c sqrt((2l-1)!!) exactly cancels that gap: for a
/// single-primitive shell, @c U^T S_cart U reproduces the identity on that
/// shell's block to machine precision; for a general contraction (which this
/// repo does not force to be unit-self-normalized -- coefficients are taken
/// as given), it reproduces a uniform scalar times identity, with all
/// off-diagonal (cross-m) entries exactly zero either way. Pinned
/// numerically in @c test/gtest/basis_host.cpp.
///
/// For l <= 1 a shell's block is an identity (the harmonic embedding of s/p
/// IS the Cartesian set, and @c sqrt((2*0-1)!!) = @c sqrt((2*1-1)!!) = 1).
[[nodiscard]] inline std::vector<double> spherical_transform_matrix(const BasisSet &basis) {
  const size_t nrow = basis.nao_cart();
  const size_t ncol = basis.nao_sphe();
  std::vector<double> u(nrow * ncol, 0.0);

  const uint32_t nsh = basis.nshells();
  for (uint32_t s = 0; s < nsh; ++s) {
    const Shell &shell = basis.shells()[s];
    const int32_t l = shell.angular_momentum();
    const uint32_t row0 = basis.ao_offset(s);
    const uint32_t col0 = basis.spherical_ao_offset(s);
    const uint32_t ncart = shell.function_count();
    const uint32_t nsph = shell.spherical_function_count();

    const std::vector<double> block = solid_harmonic_block(l);
    const double scale = std::sqrt(math::double_factorial_odd((2 * l) - 1));
    for (uint32_t i = 0; i < ncart; ++i) {
      for (uint32_t j = 0; j < nsph; ++j) {
        u[row0 + i + ((col0 + j) * nrow)] = block[i + (static_cast<size_t>(j) * ncart)] * scale;
      }
    }
  }
  return u;
}

} // namespace free_breakfast
