/// @file
/// @brief CasscfGradientStep: the integral-direct refusal, the CasscfOutput
/// back into the mcscf::Solution mcscf::df_gradient takes, and
/// casscf_gradient.json.

module free_breakfast.pipeline.casscf_gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.primitives; // math::binomial
import free_breakfast.io.json;
import free_breakfast.mcscf.gradient;

#include "diagnostics.h"

namespace free_breakfast {

// @p g's n_atom x 3 row-major entries as n_atom rows of three, the layout of
// test/gtest/data/casscf_gradient/casscf_gradient.json's `total`.
[[nodiscard]] json casscf_gradient_rows(const std::vector<double> &g) {
  json rows = json::array();
  for (size_t a = 0; a + 2 < g.size(); a += 3) {
    rows.push_back(json::array({g[a], g[a + 1], g[a + 2]}));
  }
  return rows;
}

// The CASSCF a step reading its product is sized over, at @p dims: its
// active space, core and multiplicity as the Casscf step takes them, its
// roots, and the orbitals and CI vectors run puts back on the device.
struct CasscfReadShape {
  int32_t ncore = 0;
  int32_t ncas = 0;
  int32_t nelecas = 0;
  int32_t multiplicity = 1;
  int32_t nroots = 0;
  MemoryFootprint upload;
};

CasscfReadShape casscf_read_shape(const FootprintDimensions &dims) {
  CasscfReadShape out;
  out.ncas = static_cast<int32_t>(dims.ncas);
  out.nelecas = static_cast<int32_t>(dims.nelecas);
  out.ncore = (static_cast<int32_t>(dims.n_alpha + dims.n_beta) - out.nelecas) / 2;
  const auto two_s = static_cast<int32_t>(dims.n_alpha - dims.n_beta);
  out.multiplicity = two_s + 1;
  out.nroots = static_cast<int32_t>(dims.casscf_nroots);
  const int32_t n_beta = (out.nelecas - two_s) / 2;
  const auto strings = [&](const int32_t electrons) {
    return static_cast<size_t>(std::round(math::binomial(out.ncas, electrons)));
  };
  const size_t dim = strings(out.nelecas - n_beta) * strings(n_beta);
  out.upload.device.persistent =
      sizeof(double) * ((dims.nao * dims.nao) + (dim * dims.casscf_nroots));
  return out;
}

Result<CasscfGradientOutput>
CasscfGradientStep::run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
                             CoreHamiltonianOutput, CasscfOutput>
                            in,
                        DeviceResources &res) const {
  // The Casscf step runs mcscf::DensityFitting::Full exactly when the
  // EriOutput is fitted, so this one check is also the one on its mode.
  const auto *const fit = std::get_if<DensityFitted>(&in.get<EriOutput>().eri);
  if (fit == nullptr) {
    return err_unsupported("[{}] runs over density fitting only (the DF-CASSCF's gradient), "
                           "and [eri] mode is 'integral_direct': set [eri] mode = "
                           "'density_fitting'",
                           kName);
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const BasisSet &basis = in.get<BasisSetOutput>().basis;
  const auto &classes = in.get<PrimitivePairClassesOutput>().classes;
  const SphericalMatrixAO &h = in.get<CoreHamiltonianOutput>().h;
  const CasscfOutput &cas = in.get<CasscfOutput>();
  const auto n2 = static_cast<size_t>(cas.ncas) * static_cast<size_t>(cas.ncas);
  if (cas.rdm1.size() != cas.weights.size() * n2) {
    return err_config("[{}] the CASSCF holds {} 1-RDM elements, not {} roots of {}", kName,
                      cas.rdm1.size(), cas.weights.size(), n2);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();

  // df_gradient takes the solve's Solution, and the product is that
  // Solution with its core and virtuals canonicalized, which moves neither D
  // nor W: the orbitals and the CI vectors back on the device, and the
  // weighted gamma, which the product holds per root.
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
      .weights = cas.weights,
  };
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                         cas.vectors.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                         cas.c.num_elements() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));

  CasscfGradientOutput out{
      .gradient = FBF_TRY(mcscf::df_gradient(res, molecule, basis, classes, fit->aux, h, fit->b,
                                             fit->l.data(), solution, cas.ncore, cas.ncas, root)),
  };
  if (!json_path.empty()) {
    const mcscf::GradientTerms &g = out.gradient;
    FBF_TRY(write_json(json_path,
                       json{
                           {"root", root},
                           {"energy", cas.energies[static_cast<size_t>(root)]},
                           {"total", casscf_gradient_rows(g.total)},
                           {"terms",
                            {
                                {"kinetic", casscf_gradient_rows(g.kinetic)},
                                {"potential", casscf_gradient_rows(g.potential)},
                                {"overlap", casscf_gradient_rows(g.overlap)},
                                {"two_electron", casscf_gradient_rows(g.two_electron)},
                                {"nuclear_repulsion", casscf_gradient_rows(g.nuclear_repulsion)},
                            }},
                       }));
  }
  return out;
}

MemoryFootprint CasscfGradientStep::footprint(const FootprintDimensions &dims) const {
  if (dims.basis == nullptr || dims.aux == nullptr || dims.casscf_nroots == 0) {
    return {};
  }
  const CasscfReadShape cas = casscf_read_shape(dims);
  return drop(then(cas.upload, mcscf::df_gradient_footprint(
                                   *dims.basis, *dims.aux, cas.ncore, cas.ncas, cas.nelecas,
                                   cas.multiplicity, cas.nroots, dims.casscf_unequal_weights)));
}

} // namespace free_breakfast
