/// @file
/// @brief Implementation of the SAD initial guess: one @c AtomicDensityStamp
/// per distinct (Z, shells), each placed on its atoms' diagonal blocks.

module free_breakfast.sad;

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

// One atom's converged SCF density over its own spherical AOs, column-major.
// Every integral of the atomic SCF is one-center, so the block is
// geometry-free: every atom with this Z and these shells wears the same one.
struct AtomicDensityStamp {
  int32_t z;
  std::span<const Shell> shells;
  std::vector<double> density;
};

// Write @p stamp's block onto the diagonal of the n x n column-major
// @p density, starting at AO @p offset.
void place(const AtomicDensityStamp &stamp, size_t offset, size_t n, std::span<double> density) {
  size_t block = 0;
  for (const Shell &shell : stamp.shells) {
    block += shell.function_count_sphe();
  }
  for (size_t j = 0; j < block; ++j) {
    for (size_t i = 0; i < block; ++i) {
      density[(offset + i) + (n * (offset + j))] = stamp.density[i + (block * j)];
    }
  }
}

Result<std::vector<double>> sad_density(const BasisSet &basis, std::span<const Atom> atoms) {
  if (basis.natoms() != atoms.size()) {
    return err_config("sad_density: the basis groups its shells over {} atoms but {} were given; "
                      "the basis was not placed over these atoms",
                      basis.natoms(), atoms.size());
  }
  const size_t n = basis.nao_sphe();
  std::vector<double> density(n * n, 0.0);
  std::vector<AtomicDensityStamp> stamps;

  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    const uint32_t count = basis.atom_shell_count(a);
    if (count == 0) {
      continue;
    }
    const uint32_t first = basis.atom_shell_offset(a);
    const std::span<const Shell> shells(basis.shells().data() + first, count);
    const int32_t z = nuclear_charge(atoms[a]);
    auto it = std::ranges::find_if(stamps, [&](const AtomicDensityStamp &s) {
      return s.z == z && same_shells(s.shells, shells);
    });
    if (it == stamps.end()) {
      AtomicScf scf = FBF_TRY(atomic_scf(z, shells));
      stamps.push_back(AtomicDensityStamp{z, shells, std::move(scf.density)});
      it = std::prev(stamps.end());
    }
    place(*it, basis.ao_offset_sphe(first), n, density);
  }
  return density;
}

} // namespace free_breakfast
