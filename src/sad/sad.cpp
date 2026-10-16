/// @file
/// @brief Implementation of the SAD initial guess.

module free_breakfast.sad;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;

#include "diagnostics.h"

namespace free_breakfast {

std::vector<int32_t> subshell_occupation(int32_t z) {
  if (z < 0 || z > 118) {
    check_or_abort(err_config("subshell_occupation: Z={} is outside [0, 118]", z));
  }
  std::vector<int32_t> occupation;
  int32_t remaining = z;
  for (const SubshellFill &subshell : kAufbauOrder) {
    if (remaining == 0) {
      break;
    }
    const int32_t here = std::min(remaining, subshell.capacity);
    occupation.push_back(here);
    remaining -= here;
  }
  return occupation;
}

std::vector<double> atomic_diagonal(int32_t z, std::span<const Shell> shells) {
  // shells_of_l[l] lists the indices of this atom's shells of momentum l, in
  // order; the k-th subshell of momentum l goes on shells_of_l[l][k].
  std::vector<std::vector<size_t>> shells_of_l;
  std::vector<size_t> ao_offset(shells.size(), 0);
  size_t nao_sphe = 0;
  for (size_t s = 0; s < shells.size(); ++s) {
    const auto l = static_cast<size_t>(shells[s].angular_momentum());
    if (l >= shells_of_l.size()) {
      shells_of_l.resize(l + 1);
    }
    shells_of_l[l].push_back(s);
    ao_offset[s] = nao_sphe;
    nao_sphe += shells[s].spherical_function_count();
  }

  std::vector<double> diagonal(nao_sphe, 0.0);
  std::vector<size_t> next_of_l(shells_of_l.size(), 0);
  const std::vector<int32_t> occupation = subshell_occupation(z);
  for (size_t k = 0; k < occupation.size(); ++k) {
    const auto l = static_cast<size_t>(kAufbauOrder[k].l);
    if (l >= shells_of_l.size() || next_of_l[l] >= shells_of_l[l].size()) {
      continue; // No shell of this momentum left: these electrons are dropped.
    }
    const size_t s = shells_of_l[l][next_of_l[l]++];
    const uint32_t nfun = shells[s].spherical_function_count();
    const double per_ao = static_cast<double>(occupation[k]) / static_cast<double>(nfun);
    std::fill_n(diagonal.begin() + static_cast<std::ptrdiff_t>(ao_offset[s]), nfun, per_ao);
  }
  return diagonal;
}

Result<std::vector<double>> sad_diagonal(const BasisSet &basis, const Molecule &molecule) {
  if (basis.natoms() != molecule.natoms()) {
    return err_config("sad_diagonal: the basis groups its shells over {} atoms but the molecule "
                      "has {}; the basis was not placed over this molecule",
                      basis.natoms(), molecule.natoms());
  }
  std::vector<double> diagonal(basis.nao_sphe(), 0.0);
  const std::vector<Atom> &atoms = molecule.atoms();
  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    const uint32_t count = basis.atom_shell_count(a);
    if (count == 0) {
      continue;
    }
    const uint32_t first = basis.atom_shell_offset(a);
    const std::span<const Shell> shells(basis.shells().data() + first, count);
    const std::vector<double> block = atomic_diagonal(nuclear_charge(atoms[a]), shells);
    const auto offset = static_cast<std::ptrdiff_t>(basis.spherical_ao_offset(first));
    std::ranges::copy(block, diagonal.begin() + offset);
  }
  return diagonal;
}

Result<std::vector<double>> sad_density(const BasisSet &basis, const Molecule &molecule) {
  const std::vector<double> diagonal = FBF_TRY(sad_diagonal(basis, molecule));
  const size_t n = diagonal.size();
  std::vector<double> density(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    density[i + (i * n)] = diagonal[i];
  }
  return density;
}

} // namespace free_breakfast
