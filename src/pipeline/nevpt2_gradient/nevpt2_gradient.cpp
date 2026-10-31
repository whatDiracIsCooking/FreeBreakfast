/// @file
/// @brief Nevpt2GradientStep: the refusals, the CasscfOutput back into the
/// mcscf::Solution nevpt2::sc_gradient and nevpt2::pc_gradient take, and
/// nevpt2_gradient.json.

module free_breakfast.pipeline.nevpt2_gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.nevpt2.gradient;

#include "diagnostics.h"

namespace free_breakfast {

// @p g's n_atom x 3 row-major entries as n_atom rows of three, the layout of
// test/gtest/data/nevpt2_gradient/nevpt2_gradient.json's gradients.
[[nodiscard]] json nevpt2_gradient_rows(const std::vector<double> &g) {
  json rows = json::array();
  for (size_t a = 0; a + 2 < g.size(); a += 3) {
    rows.push_back(json::array({g[a], g[a + 1], g[a + 2]}));
  }
  return rows;
}

Result<Nevpt2GradientOutput>
Nevpt2GradientStep::run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
                             CoreHamiltonianOutput, ScfOutput, CasscfOutput>
                            in,
                        DeviceResources &res) const {
  // The Nevpt2 step's refusals, for the same reasons: the strings are
  // nelecas / 2 a spin, and only the density-fitted NEVPT2 is ported.
  if (const ScfOutput &scf = in.get<ScfOutput>(); scf.n_alpha != scf.n_beta) {
    return err_unsupported("[{}] needs a closed-shell reference, and the ScfOutput has {} alpha "
                           "and {} beta electrons",
                           kName, scf.n_alpha, scf.n_beta);
  }
  if (const int32_t multiplicity = in.get<MoleculeOutput>().molecule.multiplicity();
      multiplicity != 1) {
    return err_unsupported("[{}] is implemented for a singlet only, and the molecule's "
                           "multiplicity is {}",
                           kName, multiplicity);
  }
  const auto *const fit = std::get_if<DensityFitted>(&in.get<EriOutput>().eri);
  if (fit == nullptr) {
    return err_unsupported("[{}] runs over density fitting only, and [eri] mode is "
                           "'integral_direct': set [eri] mode = 'density_fitting'",
                           kName);
  }
  const CasscfOutput &cas = in.get<CasscfOutput>();
  // sc_gradient and pc_gradient refuse a state average themselves, but only after its
  // Solution is built: this says so in the config's words.
  if (cas.energies.size() != 1) {
    return err_unsupported("[{}] differentiates a single-state CASSCF, and [casscf] has {} roots: "
                           "set [casscf] nroots = 1",
                           kName, cas.energies.size());
  }
  if (root != 0) {
    return err_config("[{}] root is {}, and the single-state CASSCF has only root 0", kName, root);
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

  // The CasscfGradient step's Solution: the product is the solve's with its
  // core and virtuals canonicalized, which both gradients need (its orbital
  // energies are f's diagonal there), and one root's gamma is its rdm1.
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
      .gamma = cas.rdm1,
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

  // Batch 0: each class's whole extent in one slab, as nevpt2_gradient_gpu.cpp
  // runs it (src/pipeline/nevpt2_gradient/README.md, "Memory"). A PC
  // refusal (no gap at the cut, a non-positive denominator) is the energy's
  // error, returned as it is.
  Nevpt2GradientOutput out =
      variant == Nevpt2Variant::Pc
          ? Nevpt2GradientOutput{.gradient = FBF_TRY(nevpt2::pc_gradient(
                                     res, molecule, basis, classes, fit->aux, h, fit->b,
                                     fit->l.data(), solution, cas.ncore, cas.ncas, 0))}
          : Nevpt2GradientOutput{.gradient = FBF_TRY(nevpt2::sc_gradient(
                                     res, molecule, basis, classes, fit->aux, h, fit->b,
                                     fit->l.data(), solution, cas.ncore, cas.ncas, 0))};
  if (!json_path.empty()) {
    const double e_cas = cas.energies[static_cast<size_t>(root)];
    FBF_TRY(write_json(
        json_path,
        std::visit(
            [&]<class G>(const G &g) {
              constexpr bool kPc = std::same_as<G, nevpt2::PcGradient>;
              json doc{
                  {"root", root},
                  {"variant", kPc ? "pc" : "sc"},
                  {"energy", e_cas + g.energies.total},
                  {"e_casscf", e_cas},
                  {"e_corr", g.energies.total},
                  {"total", nevpt2_gradient_rows(g.terms.total)},
                  {"terms",
                   {
                       {"kinetic", nevpt2_gradient_rows(g.terms.kinetic)},
                       {"potential", nevpt2_gradient_rows(g.terms.potential)},
                       {"overlap", nevpt2_gradient_rows(g.terms.overlap)},
                       {"two_electron", nevpt2_gradient_rows(g.terms.two_electron)},
                       {"nuclear_repulsion", nevpt2_gradient_rows(g.terms.nuclear_repulsion)},
                   }},
                  {"max_zc", g.zc.max_z},
                  // null when neither inactive space has two orbitals (infinity).
                  {"min_inactive_gap",
                   std::isfinite(g.zc.min_gap) ? json(g.zc.min_gap) : json(nullptr)},
                  {"pcg_iterations", g.multipliers.iterations},
                  {"pcg_residual", g.multipliers.residual},
                  {"asymmetry",
                   {
                       {"core_core", g.asymmetry[0]},
                       {"active_active", g.asymmetry[1]},
                       {"virtual_virtual", g.asymmetry[2]},
                       {"active_core", g.asymmetry[3]},
                       {"virtual_core", g.asymmetry[4]},
                       {"virtual_active", g.asymmetry[5]},
                   }},
              };
              // PC's one addition: the metric modes each class cut as null, in
              // PySCF's class order.
              if constexpr (kPc) {
                doc["dropped"] = g.energies.dropped;
              }
              return doc;
            },
            out.gradient)));
  }
  return out;
}

} // namespace free_breakfast
