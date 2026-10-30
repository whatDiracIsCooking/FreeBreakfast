/// @file
/// @brief GradientStep: the ROHF refusal, gradient::df_rhf or gradient::rhf
/// over the pipeline's products, and gradient.json.

module free_breakfast.pipeline.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.gradient;

#include "diagnostics.h"

namespace free_breakfast {

// @p g's n_atom x 3 row-major entries as n_atom rows of three, the layout of
// test/gtest/data/gradient/gradient.json.
[[nodiscard]] json gradient_rows(const std::vector<double> &g) {
  json rows = json::array();
  for (size_t a = 0; a + 2 < g.size(); a += 3) {
    rows.push_back(json::array({g[a], g[a + 1], g[a + 2]}));
  }
  return rows;
}

Result<GradientOutput> GradientStep::run(
    View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput, ScfOutput> in,
    DeviceResources &res) const {
  const ScfOutput &scf = in.get<ScfOutput>();
  if (scf.n_alpha != scf.n_beta) {
    return err_unsupported("[{}] is the closed-shell RHF gradient only, and the SCF is ROHF ({} "
                           "alpha, {} beta electrons)",
                           kName, scf.n_alpha, scf.n_beta);
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const auto &classes = in.get<PrimitivePairClassesOutput>().classes;
  const auto &eri = in.get<EriOutput>().eri;
  // gradient.json's layout is the same for both ERI kinds: RhfTerms is.
  GradientOutput out;
  if (const auto *const fit = std::get_if<DensityFitted>(&eri)) {
    out.gradient = FBF_TRY(gradient::df_rhf(res, molecule, basis, classes, fit->aux, fit->b.data(),
                                            fit->l.data(), scf.c, scf.mo_energies, scf.n_alpha));
  } else {
    out.gradient = FBF_TRY(gradient::rhf(res, molecule, basis, classes,
                                         std::get<IntegralDirect>(eri).k_classes, scf.c,
                                         scf.mo_energies, scf.n_alpha));
  }
  if (!json_path.empty()) {
    const gradient::RhfTerms &g = out.gradient;
    FBF_TRY(
        write_json(json_path, json{
                                  {"energy", scf.energy},
                                  {"total", gradient_rows(g.total)},
                                  {"terms",
                                   {
                                       {"kinetic", gradient_rows(g.kinetic)},
                                       {"potential", gradient_rows(g.potential)},
                                       {"overlap", gradient_rows(g.overlap)},
                                       {"coulomb", gradient_rows(g.coulomb)},
                                       {"exchange", gradient_rows(g.exchange)},
                                       {"nuclear_repulsion", gradient_rows(g.nuclear_repulsion)},
                                   }},
                              }));
  }
  return out;
}

} // namespace free_breakfast
