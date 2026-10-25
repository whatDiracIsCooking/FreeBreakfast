/// @file
/// @brief orbital_weights, orbital_populations and partial_charges: A^T S C on
/// the device, its squares summed per atom on the host.

module free_breakfast.analysis.population;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm

#include "diagnostics.h"

namespace free_breakfast {

// The atom owning each spherical AO of @p minimal, which holds each atom's
// shells together, in atom order.
[[nodiscard]] std::vector<uint32_t> population_ao_atoms(const BasisSet &minimal) {
  std::vector<uint32_t> owner(minimal.nao_sphe());
  for (uint32_t atom = 0; atom < minimal.natoms(); ++atom) {
    const uint32_t first = minimal.atom_shell_offset(atom);
    for (uint32_t shell = first; shell < first + minimal.atom_shell_count(atom); ++shell) {
      const uint32_t ao = minimal.ao_offset_sphe(shell);
      const auto width =
          static_cast<uint32_t>((2 * minimal.shells()[shell].angular_momentum()) + 1);
      std::fill_n(owner.begin() + ao, width, atom);
    }
  }
  return owner;
}

Result<std::vector<double>> orbital_weights(DeviceResources &res, const BasisSet &basis,
                                            const BasisSet &minimal, const int32_t n_orb,
                                            const double *s, const double *c, const double *iao) {
  if (basis.natoms() == 0) {
    return err_config("orbital_weights: the basis has no atoms; place it over a molecule");
  }
  if (minimal.natoms() != basis.natoms()) {
    return err_config("orbital_weights: the minimal basis is over {} atoms, the basis over {}",
                      minimal.natoms(), basis.natoms());
  }
  if (n_orb < 0) {
    return err_config("orbital_weights: n_orb {} is negative", n_orb);
  }
  if (s == nullptr || c == nullptr || iao == nullptr) {
    return err_config("orbital_weights: s, c and iao must not be null");
  }
  const auto k = static_cast<size_t>(n_orb);
  const auto n = static_cast<int32_t>(basis.nao_sphe());
  const auto m = static_cast<int32_t>(minimal.nao_sphe());
  std::vector<double> weights_host(static_cast<size_t>(m) * k);
  if (k == 0) {
    return weights_host;
  }
  const ScopedDeviceIndex on{res.dev_idx()};

  // The orbitals over the minimal basis, A^T S C, m x n_orb.
  Buffers::device<double> sc(static_cast<size_t>(n) * k, res.device());
  Buffers::device<double> weights(static_cast<size_t>(m) * k, res.device());
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const double one = kOne<double>;
  const double zero = kZero<double>;
  FBF_TRY(gpu_status(gemm(blas, n, n_orb, n, &one, s, c, &zero, sc.data())));
  FBF_TRY(gpu_status(
      gemm<wwr::WWRBLAS_OP_T>(blas, m, n_orb, n, &one, iao, sc.data(), &zero, weights.data())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(weights_host.data(), weights.data(),
                                         weights_host.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  for (double &w : weights_host) {
    w *= w;
  }
  return weights_host;
}

Result<std::vector<double>> orbital_populations(const BasisSet &minimal, const int32_t n_orb,
                                                const std::span<const double> weights) {
  if (minimal.natoms() == 0) {
    return err_config("orbital_populations: the minimal basis has no atoms");
  }
  const size_t m = minimal.nao_sphe();
  if (n_orb < 0 || weights.size() != m * static_cast<size_t>(n_orb)) {
    return err_config("orbital_populations: {} weights are not {} x {}", weights.size(), m, n_orb);
  }
  const uint32_t n_atoms = minimal.natoms();
  const auto k = static_cast<size_t>(n_orb);
  std::vector<double> populations(static_cast<size_t>(n_atoms) * k, 0.0);
  const std::vector<uint32_t> owner = population_ao_atoms(minimal);
  for (size_t i = 0; i < k; ++i) {
    for (size_t r = 0; r < m; ++r) {
      populations[owner[r] + (n_atoms * i)] += weights[r + (m * i)];
    }
  }
  return populations;
}

Result<std::vector<double>> orbital_populations(DeviceResources &res, const BasisSet &basis,
                                                const BasisSet &minimal, const int32_t n_orb,
                                                const double *s, const double *c,
                                                const double *iao) {
  const std::vector<double> weights =
      FBF_TRY(orbital_weights(res, basis, minimal, n_orb, s, c, iao));
  return orbital_populations(minimal, n_orb, weights);
}

Result<std::vector<double>> partial_charges(const std::span<const Atom> atoms,
                                            const std::span<const double> populations,
                                            const double occupation) {
  const size_t n_atoms = atoms.size();
  if (n_atoms == 0 ? !populations.empty() : populations.size() % n_atoms != 0) {
    return err_config("partial_charges: {} populations are not whole columns of {} atoms",
                      populations.size(), n_atoms);
  }
  std::vector<double> charges(n_atoms);
  for (size_t a = 0; a < n_atoms; ++a) {
    charges[a] = static_cast<double>(static_cast<int32_t>(atoms[a].element));
  }
  for (size_t p = 0; p < populations.size(); ++p) {
    charges[p % n_atoms] -= occupation * populations[p];
  }
  return charges;
}

} // namespace free_breakfast
