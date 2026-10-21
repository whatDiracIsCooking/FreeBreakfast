/// @file
/// @brief The molecular SAD guess from atomic SCFs: @ref
/// free_breakfast::atomic_scf_density.
export module free_breakfast.experimental.atomic_scf:density;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import :scf;

#include "diagnostics.h"

namespace free_breakfast {

// Two atoms share an SCF when their shells match exactly -- momentum,
// exponents and coefficients -- whatever their centers.
bool same_shells(std::span<const Shell> a, std::span<const Shell> b) {
  return std::ranges::equal(a, b, [](const Shell &x, const Shell &y) {
    return x.angular_momentum() == y.angular_momentum() &&
           std::ranges::equal(x.primitives(), y.primitives(),
                              [](const Primitive &p, const Primitive &q) {
                                return p.exponent == q.exponent && p.coefficient == q.coefficient;
                              });
  });
}

} // namespace free_breakfast

export namespace free_breakfast {

/// @brief The SAD initial guess from atomic SCFs: each atom's @ref
/// AtomicScf::density on its block of @p basis's spherical layout, zeros
/// between atoms.
///
/// One SCF per distinct (Z, shells) pair, so a molecule pays for its elements,
/// not its atoms: @ref place gives every atom of an element the same shells.
///
/// @param basis A basis made by @ref place over @p atoms, so it carries the
///              per-atom shell grouping (@ref BasisSet::natoms).
/// @param atoms The atoms, in the order @p basis was placed over.
/// @return nao_sphe x nao_sphe, column-major. ErrorKind::InvalidConfig if
///   @c basis.natoms() differs from the atom count; otherwise the first
///   atom's @ref atomic_scf failure.
[[nodiscard]] Result<std::vector<double>> atomic_scf_density(const BasisSet &basis,
                                                             std::span<const Atom> atoms) {
  if (basis.natoms() != atoms.size()) {
    return err_config("atomic_scf_density: the basis groups its shells over {} atoms but {} were "
                      "given; the basis was not placed over these atoms",
                      basis.natoms(), atoms.size());
  }
  const size_t n = basis.nao_sphe();
  std::vector<double> density(n * n, 0.0);

  struct Solved {
    int32_t z;
    std::span<const Shell> shells;
    std::vector<double> density;
  };
  std::vector<Solved> solved;

  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    const uint32_t count = basis.atom_shell_count(a);
    if (count == 0) {
      continue;
    }
    const uint32_t first = basis.atom_shell_offset(a);
    const std::span<const Shell> shells(basis.shells().data() + first, count);
    const int32_t z = nuclear_charge(atoms[a]);
    auto it = std::ranges::find_if(
        solved, [&](const Solved &s) { return s.z == z && same_shells(s.shells, shells); });
    if (it == solved.end()) {
      AtomicScf scf = FBF_TRY(atomic_scf(z, shells));
      solved.push_back(Solved{z, shells, std::move(scf.density)});
      it = std::prev(solved.end());
    }
    const size_t offset = basis.ao_offset_sphe(first);
    const size_t block = basis.ao_offset_sphe(first + count - 1) +
                         basis.shells()[first + count - 1].function_count_sphe() - offset;
    for (size_t j = 0; j < block; ++j) {
      for (size_t i = 0; i < block; ++i) {
        density[(offset + i) + (n * (offset + j))] = it->density[i + (block * j)];
      }
    }
  }
  return density;
}

} // namespace free_breakfast
