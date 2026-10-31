/// @file
/// @brief df_rhf: the full DF-RHF nuclear Hessian, its explicit part (the
/// one-electron second derivatives at D and -W, V_nn's, and
/// df_rhf_two_electron) beside df_rhf_response, summed and symmetrized.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kZero, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.hessian;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast::hessian {

Result<RhfTerms> df_rhf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                        const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                        const Buffers::device<double> &b, const double *l,
                        const Buffers::device<double> &c, const std::span<const double> mo_energies,
                        const int32_t nocc, const double tolerance, const int32_t max_iterations) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const size_t nmo = mo_energies.size();
  // df_rhf_response's own check, made before the one-electron part reads C,
  // so a bad input is refused before anything runs.
  if (molecule.natoms() != basis.natoms() || nocc < 1 || static_cast<size_t>(nocc) >= nmo ||
      c.num_elements() != nao * nmo || b.num_elements() != naux * nao * nao) {
    return err_config("hessian::df_rhf: needs a molecule of the basis's {} atoms, 1 <= nocc < "
                      "nmo, C nao nmo and B naux nao^2 long ({} atoms, nocc {}, nmo {}, nao {}, "
                      "naux {}, C {} doubles, B {} doubles)",
                      basis.natoms(), molecule.natoms(), nocc, nmo, nao, naux, c.num_elements(),
                      b.num_elements());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<double> one_electron;
  {
    // D = 2 C_o C_o^T and -W = C_o diag(-2 eps_o) C_o^T, as gradient::df_rhf
    // makes them. The host diagonal outlives its copy: overlap_hessian waits
    // on the stream before this scope ends.
    const auto n = static_cast<int32_t>(nao);
    SphericalMatrixAO d(res, basis);
    SphericalMatrixAO minus_w(res, basis);
    std::vector<double> diagonal(static_cast<size_t>(nocc));
    std::ranges::transform(mo_energies.first(diagonal.size()), diagonal.begin(),
                           [](const double e) { return -2.0 * e; });
    Buffers::device<double> d_diagonal(diagonal.size(), res.device());
    Buffers::device<double> work(nao * diagonal.size(), res.device());
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), n, n, nocc, &kTwo<double>, c.data(), c.data(), &kZero<double>, d.data())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_diagonal.data(), diagonal.data(),
                                           sizeof(double) * diagonal.size(),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    FBF_TRY(gpu_status(gmmwich<wwr::WWRBLAS_OP_T>(res.blas(), n, nocc, c.data(), d_diagonal.data(),
                                                  work.data(), minus_w.data())));
    one_electron = FBF_TRY(overlap_hessian(res, basis, classes, minus_w));
    const std::vector<double> kinetic = FBF_TRY(kinetic_hessian(res, basis, classes, d));
    const std::vector<double> potential =
        FBF_TRY(potential_hessian(res, basis, classes, molecule, d));
    for (size_t i = 0; i < one_electron.size(); ++i) {
      one_electron[i] += kinetic[i] + potential[i];
    }
  }
  RhfTwoElectron two_electron =
      FBF_TRY(df_rhf_two_electron(res, basis, classes, aux, b, l, c, nocc));
  RhfTerms out{.hessian = {},
               .asymmetry = 0.0,
               .one_electron = std::move(one_electron),
               .nuclear_repulsion = nuclear_repulsion_hessian(molecule),
               .two_electron = std::move(two_electron),
               .response = FBF_TRY(df_rhf_response(res, molecule, basis, classes, aux, b, l, c,
                                                   mo_energies, nocc, tolerance, max_iterations))};

  // Every term is (3 n_atom)^2, symmetric to rounding: summed, then the sum
  // symmetrized, its antisymmetric part measured as it is discarded.
  const size_t n_x = 3 * static_cast<size_t>(basis.natoms());
  std::vector<double> sum(n_x * n_x);
  for (size_t i = 0; i < sum.size(); ++i) {
    sum[i] = out.one_electron[i] + out.nuclear_repulsion[i] + out.two_electron.coulomb[i] +
             out.two_electron.exchange[i] + out.response.hessian[i];
  }
  out.hessian.resize(sum.size());
  for (size_t y = 0; y < n_x; ++y) {
    for (size_t x = 0; x < n_x; ++x) {
      const double upper = sum[x + (n_x * y)];
      const double lower = sum[y + (n_x * x)];
      out.asymmetry = std::max(out.asymmetry, std::abs(upper - lower));
      out.hessian[x + (n_x * y)] = 0.5 * (upper + lower);
    }
  }
  return out;
}

} // namespace free_breakfast::hessian
