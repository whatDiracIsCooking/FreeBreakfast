/// @file
/// @brief CasciStep: the orbital order, and run's path from the Scf's orbitals
/// through the active-space Hamiltonian and the CI solve to the 1-RDMs.

module free_breakfast.pipeline.casci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.solver;
import free_breakfast.ci.rdm;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<std::vector<int32_t>> CasciStep::orbital_order(int32_t n_electrons,
                                                                    int32_t nmo) const {
  if (ncas < 1 || nelecas < 0 || nelecas > 2 * ncas || nelecas > n_electrons) {
    return err_config("casci: needs ncas >= 1 and nelecas in [0, min(2 ncas, {})] (ncas {}, "
                      "nelecas {})",
                      n_electrons, ncas, nelecas);
  }
  if ((n_electrons - nelecas) % 2 != 0) {
    return err_config("casci: {} electrons less nelecas {} leaves an odd core electron count",
                      n_electrons, nelecas);
  }
  const int32_t ncore = (n_electrons - nelecas) / 2;
  if (ncore + ncas > nmo) {
    return err_config("casci: ncore {} + ncas {} exceeds the {} orbitals", ncore, ncas, nmo);
  }
  std::vector<int32_t> chosen = active;
  if (chosen.empty()) {
    chosen.resize(static_cast<size_t>(ncas));
    std::iota(chosen.begin(), chosen.end(), ncore);
  }
  if (std::cmp_not_equal(chosen.size(), ncas)) {
    return err_config("casci: active names {} orbitals, not ncas {}", chosen.size(), ncas);
  }
  std::vector<bool> is_active(static_cast<size_t>(nmo), false);
  for (const int32_t i : chosen) {
    if (i < 0 || i >= nmo) {
      return err_config("casci: active orbital {} is not in [0, {})", i, nmo);
    }
    if (is_active[static_cast<size_t>(i)]) {
      return err_config("casci: active orbital {} is named twice", i);
    }
    is_active[static_cast<size_t>(i)] = true;
  }
  // The core is the lowest ncore orbitals not active, then the active ones,
  // then the rest: PySCF's sort_mo.
  std::vector<int32_t> order;
  order.reserve(static_cast<size_t>(nmo));
  for (int32_t i = 0; i < nmo && std::cmp_less(order.size(), ncore); ++i) {
    if (!is_active[static_cast<size_t>(i)]) {
      order.push_back(i);
    }
  }
  const auto first_rest = order.empty() ? 0 : order.back() + 1;
  order.insert(order.end(), chosen.begin(), chosen.end());
  for (int32_t i = first_rest; i < nmo; ++i) {
    if (!is_active[static_cast<size_t>(i)]) {
      order.push_back(i);
    }
  }
  return order;
}

[[nodiscard]] Result<Casci> CasciStep::run(
    View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses, CoreHamiltonian, Scf> in,
    DeviceResources &res) const {
  const Molecule &molecule = in.get<Molecule>();
  const BasisSet &basis = in.get<BasisSet>();
  const Scf &scf = in.get<Scf>();
  const auto nmo = static_cast<int32_t>(scf.mo_energies.size());
  const std::vector<int32_t> order = FBF_TRY(orbital_order(molecule.electron_count(), nmo));
  const int32_t ncore = (molecule.electron_count() - nelecas) / 2;

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // C's columns in the order: each column is one contiguous run of nao.
  const size_t nao = basis.nao_sphe();
  SphericalMatrixAO c(res, basis);
  for (size_t col = 0; col < order.size(); ++col) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(
        c.data() + (col * nao), scf.c.data() + (static_cast<size_t>(order[col]) * nao),
        nao * sizeof(double), wwr::wwrMemcpyDeviceToDevice, stream)));
  }

  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClasses>().classes;
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  const ActiveSpaceHamiltonian ham = FBF_TRY(active_space_hamiltonian(
      res, basis, molecule, in.get<CoreHamiltonian>().h, in.get<SphericalTransform>().u, j_classes,
      k_classes, c, ncore, ncas));

  // The sigma takes host arrays, row-major: h_eff transposed, and (tu|vw)
  // from the Hamiltonian's t-fastest order to w-fastest.
  const auto n = static_cast<size_t>(ncas);
  std::vector<double> h_col(n * n);
  std::vector<double> eri_col(n * n * n * n);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(h_col.data(), ham.h_eff.data(), h_col.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eri_col.data(), ham.eri.data(),
                                         eri_col.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<double> h_eff(n * n);
  std::vector<double> eri(n * n * n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      h_eff[(t * n) + u] = h_col[t + (n * u)];
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          eri[(((((t * n) + u) * n) + v) * n) + w] = eri_col[t + (n * (u + (n * (v + (n * w)))))];
        }
      }
    }
  }

  const int32_t n_beta = nelecas / 2;
  const StringSet alpha = FBF_TRY(StringSet::create(ncas, nelecas - n_beta));
  const StringSet beta = FBF_TRY(StringSet::create(ncas, n_beta));
  CiSigma sigma = FBF_TRY(CiSigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ham.e_core, h_eff, eri));
  const auto dim = static_cast<size_t>(sigma.size());
  Buffers::device<double> diagonal_device(dim, res.device());
  FBF_TRY(sigma.diagonal(res, diagonal_device.data()));
  std::vector<double> diagonal(dim);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(diagonal.data(), diagonal_device.data(), dim * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  CiRoots roots = FBF_TRY(ci_solve(res, sigma, alpha, beta, diagonal, nroots));

  const auto count = static_cast<size_t>(nroots);
  std::vector<double> gamma(count * n * n);
  for (size_t r = 0; r < count; ++r) {
    FBF_TRY(rdm1(alpha, beta, std::span<const double>{roots.vectors.data() + (r * dim), dim},
                 std::span<double>{gamma.data() + (r * n * n), n * n}));
  }
  return Casci{
      .ncore = ncore,
      .ncas = ncas,
      .nelecas = nelecas,
      .c = std::move(c),
      .energies = std::move(roots.energies),
      .spin_squares = std::move(roots.spin_squares),
      .vectors = std::move(roots.vectors),
      .rdm1 = std::move(gamma),
      .davidson = roots.davidson,
  };
}

} // namespace free_breakfast
