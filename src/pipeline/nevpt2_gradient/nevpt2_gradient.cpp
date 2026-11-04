/// @file
/// @brief Nevpt2GradientStep: the refusals, the CasscfOutput back into the
/// mcscf::Solution nevpt2::sc_gradient, pc_gradient, qd_sc_gradient and
/// qd_pc_gradient take, and nevpt2_gradient.json.

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

// @p terms' five terms (an mcscf::GradientTerms), each as n_atom rows of
// three.
template <class Terms>
[[nodiscard]] json nevpt2_gradient_terms(const Terms &terms) {
  return {
      {"kinetic", nevpt2_gradient_rows(terms.kinetic)},
      {"potential", nevpt2_gradient_rows(terms.potential)},
      {"overlap", nevpt2_gradient_rows(terms.overlap)},
      {"two_electron", nevpt2_gradient_rows(terms.two_electron)},
      {"nuclear_repulsion", nevpt2_gradient_rows(terms.nuclear_repulsion)},
  };
}

// X^L's antisymmetric residue per block, by name.
[[nodiscard]] json nevpt2_gradient_asymmetry(const std::array<double, 6> &a) {
  return {
      {"core_core", a[0]},   {"active_active", a[1]}, {"virtual_virtual", a[2]},
      {"active_core", a[3]}, {"virtual_core", a[4]},  {"virtual_active", a[5]},
  };
}

// nevpt2_gradient.json for @p g, root @p root of @p cas (state-specific) or
// QD state @p state.
template <class G>
[[nodiscard]] json nevpt2_gradient_json(const G &g, const CasscfOutput &cas, int32_t root,
                                        int32_t state) {
  constexpr bool kQd =
      std::same_as<G, nevpt2::QdScGradient> || std::same_as<G, nevpt2::QdPcGradient>;
  constexpr bool kPc = std::same_as<G, nevpt2::PcGradient> || std::same_as<G, nevpt2::QdPcGradient>;
  if constexpr (kQd) {
    const nevpt2::QdScLagrangian &l = g.lagrangian;
    return json{
        {"quasi_degenerate", true},
        {"state", state},
        {"variant", kPc ? "pc" : "sc"},
        {"energy", l.eigenvalues[static_cast<size_t>(state)]},
        {"qd_energies", l.eigenvalues},
        {"e_casscf", cas.energies},
        {"rho", l.rho},
        {"total", nevpt2_gradient_rows(g.total)},
        {"terms", nevpt2_gradient_terms(g.terms)},
        {"transition", nevpt2_gradient_rows(g.transition.total)},
        {"max_zc", l.zc.max_z},
        // null when neither inactive space has two orbitals (infinity).
        {"min_inactive_gap", std::isfinite(l.zc.min_gap) ? json(l.zc.min_gap) : json(nullptr)},
        {"max_zeta", g.max_zeta},
        {"root_gap", g.root_gap},
        {"heff_gap", g.heff_gap},
        {"pcg_iterations", l.multipliers.iterations},
        {"pcg_residual", l.multipliers.residual},
        {"asymmetry", nevpt2_gradient_asymmetry(l.asymmetry)},
    };
  } else {
    const double e_cas = cas.energies[static_cast<size_t>(root)];
    json doc{
        {"root", root},
        {"variant", kPc ? "pc" : "sc"},
        {"energy", e_cas + g.energies.total},
        {"e_casscf", e_cas},
        {"e_corr", g.energies.total},
        {"total", nevpt2_gradient_rows(g.terms.total)},
        {"terms", nevpt2_gradient_terms(g.terms)},
        {"max_zc", g.zc.max_z},
        // null when neither inactive space has two orbitals (infinity).
        {"min_inactive_gap", std::isfinite(g.zc.min_gap) ? json(g.zc.min_gap) : json(nullptr)},
        {"pcg_iterations", g.multipliers.iterations},
        {"pcg_residual", g.multipliers.residual},
        {"asymmetry", nevpt2_gradient_asymmetry(g.asymmetry)},
    };
    // PC's one addition: the metric modes each class cut as null, in
    // PySCF's class order.
    if constexpr (kPc) {
      doc["dropped"] = g.energies.dropped;
    }
    return doc;
  }
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
  const size_t nroots = cas.energies.size();
  if (quasi_degenerate) {
    if (nroots < 2) {
      return err_config("[{}] quasi_degenerate couples the roots of a state average, and the "
                        "CASSCF has {} root",
                        kName, nroots);
    }
    if (state < 0 || std::cmp_greater_equal(state, nroots)) {
      return err_config("[{}] state is {}, and H_eff over the CASSCF's {} roots has states 0 to {}",
                        kName, state, nroots, nroots - 1);
    }
  } else {
    if (state != 0) {
      return err_config("[{}] state is {}, and state names a QD-NEVPT2 state: it needs "
                        "quasi_degenerate = true",
                        kName, state);
    }
    // sc_gradient and pc_gradient refuse a state average themselves, but only after its
    // Solution is built: this says so in the config's words.
    if (nroots != 1) {
      return err_unsupported("[{}] differentiates a single-state CASSCF, and [casscf] has {} "
                             "roots: set [casscf] nroots = 1",
                             kName, nroots);
    }
    if (root != 0) {
      return err_config("[{}] root is {}, and the single-state CASSCF has only root 0", kName,
                        root);
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

  // The CasscfGradient step's Solution: the product is the solve's with its
  // core and virtuals canonicalized, which every gradient here needs (its
  // orbital energies are f's diagonal there), gamma the weighted average of
  // the roots' 1-RDMs (one root's own rdm1), and rdm2 already weighted.
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

  // Batch 0: each class's whole extent in one slab, as nevpt2_gradient_gpu.cpp
  // runs it (src/pipeline/nevpt2_gradient/README.md, "Memory"). A PC
  // refusal (no positive metric eigenvalue, a non-positive denominator) is the
  // energy's error, returned as it is.
  const bool pc = variant == Nevpt2Variant::Pc;
  using Gradient = decltype(Nevpt2GradientOutput::gradient);
  auto gradient = [&]() -> Result<Gradient> {
    if (quasi_degenerate && pc) {
      return Gradient{
          FBF_TRY(nevpt2::qd_pc_gradient(res, molecule, basis, classes, fit->aux, h, fit->b,
                                         fit->l.data(), solution, cas.ncore, cas.ncas, state, 0))};
    }
    if (quasi_degenerate) {
      return Gradient{
          FBF_TRY(nevpt2::qd_sc_gradient(res, molecule, basis, classes, fit->aux, h, fit->b,
                                         fit->l.data(), solution, cas.ncore, cas.ncas, state, 0))};
    }
    if (pc) {
      return Gradient{
          FBF_TRY(nevpt2::pc_gradient(res, molecule, basis, classes, fit->aux, h, fit->b,
                                      fit->l.data(), solution, cas.ncore, cas.ncas, 0))};
    }
    return Gradient{FBF_TRY(nevpt2::sc_gradient(res, molecule, basis, classes, fit->aux, h, fit->b,
                                                fit->l.data(), solution, cas.ncore, cas.ncas, 0))};
  }();
  Nevpt2GradientOutput out{.gradient = FBF_TRY(std::move(gradient))};
  if (!json_path.empty()) {
    FBF_TRY(write_json(
        json_path,
        std::visit([&]<class G>(const G &g) { return nevpt2_gradient_json(g, cas, root, state); },
                   out.gradient)));
  }
  return out;
}

} // namespace free_breakfast
