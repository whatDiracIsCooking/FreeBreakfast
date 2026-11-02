/// @file
/// @brief NacStep: the refusals, the CasscfOutput back into the
/// mcscf::Solution mcscf::df_nac takes, one df_nac per pair, and nac.json.

module free_breakfast.pipeline.nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.mcscf.nac;

#include "diagnostics.h"

namespace free_breakfast {

// @p v's n_atom x 3 row-major entries as n_atom rows of three, the layout of
// test/gtest/data/casscf_nac/casscf_nac.json's couplings.
[[nodiscard]] json nac_rows(const std::vector<double> &v) {
  json rows = json::array();
  for (size_t a = 0; a + 2 < v.size(); a += 3) {
    rows.push_back(json::array({v[a], v[a + 1], v[a + 2]}));
  }
  return rows;
}

Result<NacOutput> NacStep::run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput,
                                    EriOutput, CoreHamiltonianOutput, CasscfOutput>
                                   in,
                               DeviceResources &res) const {
  // The Casscf step runs mcscf::DensityFitting::Full exactly when the
  // EriOutput is fitted, so this one check is also the one on its mode.
  const auto *const fit = std::get_if<DensityFitted>(&in.get<EriOutput>().eri);
  if (fit == nullptr) {
    return err_unsupported("[{}] runs over density fitting only (the DF-SA-CASSCF's couplings), "
                           "and [eri] mode is 'integral_direct': set [eri] mode = "
                           "'density_fitting'",
                           kName);
  }
  const CasscfOutput &cas = in.get<CasscfOutput>();
  const auto nroots = static_cast<int32_t>(cas.energies.size());
  if (nroots < 2 || cas.weights.size() != cas.energies.size()) {
    return err_config("[{}] couples roots of a state average, and the CASSCF has {} root(s) and "
                      "{} weight(s): set [casscf] nroots to 2 or more",
                      kName, nroots, cas.weights.size());
  }
  std::vector<std::array<int32_t, 2>> chosen = pairs;
  if (chosen.empty()) {
    for (int32_t i = 0; i < nroots; ++i) {
      for (int32_t j = i + 1; j < nroots; ++j) {
        chosen.push_back({i, j});
      }
    }
  }
  for (const auto &[bra, ket] : chosen) {
    if (bra < 0 || bra >= nroots || ket < 0 || ket >= nroots || bra == ket) {
      return err_config("[{}] pair ({}, {}) is not two distinct roots of the CASSCF's {}", kName,
                        bra, ket, nroots);
    }
  }
  const auto n2 = static_cast<size_t>(cas.ncas) * static_cast<size_t>(cas.ncas);
  if (cas.rdm1.size() != cas.weights.size() * n2) {
    return err_config("[{}] the CASSCF holds {} 1-RDM elements, not {} roots of {}", kName,
                      cas.rdm1.size(), cas.weights.size(), n2);
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const auto &classes = in.get<PrimitivePairClassesOutput>().classes;
  const SphericalMatrixAO &h = in.get<CoreHamiltonianOutput>().h;
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  // The product back into the mcscf::Solution df_nac takes, as
  // [casscf_gradient] builds it for df_gradient: the orbitals and the CI
  // vectors on the device, gamma the weighted average of the roots'. The
  // core and virtuals' canonicalization moves no coupling.
  std::vector<double> gamma(n2, 0.0);
  for (size_t r = 0; r < cas.weights.size(); ++r) {
    for (size_t i = 0; i < n2; ++i) {
      gamma[i] += cas.weights[r] * cas.rdm1[(r * n2) + i];
    }
  }
  mcscf::Solution solution{
      .c = SphericalMatrixAO(res, basis),
      .roots =
          {
              .energies = cas.energies,
              .spin_squares = cas.spin_squares,
              .vectors = Buffers::device<double>(cas.vectors.size(), res.device()),
              .davidson = {},
              .fp32_iterations = 0,
          },
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations,
  };
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                         cas.vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                         cas.c.num_elements() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));

  NacOutput out{.pairs = chosen, .couplings = {}};
  json couplings = json::array();
  for (const auto &[bra, ket] : chosen) {
    mcscf::Nac d =
        FBF_TRY(mcscf::df_nac(res, molecule, basis, classes, fit->aux, h, fit->b, fit->l.data(),
                              solution, cas.ncore, cas.ncas, bra, ket, options));
    const double gap =
        cas.energies[static_cast<size_t>(ket)] - cas.energies[static_cast<size_t>(bra)];
    const double largest = std::ranges::fold_left(
        d.total, 0.0, [](double m, double x) { return std::max(m, std::abs(x)); });
    FBF_LOG_INFO("nac: <{}|d{}>, E_J - E_I = {:.10f}, max |d| {:.6e}, Z-vector in {} products "
                 "to {:.2e}",
                 bra, ket, gap, largest, d.iterations, d.residual);
    couplings.push_back(json{
        {"bra", bra},
        {"ket", ket},
        {"energy_gap", gap},
        {"h", nac_rows(d.numerator)},
        {"csf", nac_rows(d.csf)},
        {"total", nac_rows(d.total)},
        {"zvector", {{"iterations", d.iterations}, {"residual", d.residual}}},
    });
    out.couplings.push_back(std::move(d));
  }
  if (!json_path.empty()) {
    FBF_TRY(write_json(json_path, json{
                                      {"mult_ediff", options.mult_ediff},
                                      {"use_etfs", options.use_etfs},
                                      {"energies", cas.energies},
                                      {"couplings", std::move(couplings)},
                                  }));
  }
  return out;
}

} // namespace free_breakfast
