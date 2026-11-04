/// @file
/// @brief Nevpt2NacStep: the refusals, the CasscfOutput back into the
/// mcscf::Solution nevpt2::qd_sc_nac and qd_pc_nac take, H_eff and R, one
/// coupling per pair, and nevpt2_nac.json.

module free_breakfast.pipeline.nevpt2_nac;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.recipes.diagonalize;
import free_breakfast.mcscf.nac;
import free_breakfast.nevpt2.density_fitting; // MoBlocks
import free_breakfast.nevpt2.energy;          // sc_ and pc_effective_hamiltonian
import free_breakfast.nevpt2.gradient;        // qd_inputs
import free_breakfast.nevpt2.nac;
import free_breakfast.pipeline.nevpt2_gradient; // Nevpt2Variant

#include "diagnostics.h"

namespace free_breakfast {

// @p v's n_atom x 3 row-major entries as n_atom rows of three, nac.json's
// layout.
[[nodiscard]] json nevpt2_nac_rows(const std::vector<double> &v) {
  json rows = json::array();
  for (size_t a = 0; a + 2 < v.size(); a += 3) {
    rows.push_back(json::array({v[a], v[a + 1], v[a + 2]}));
  }
  return rows;
}

// H_eff = diag(E_r) + 1/2 (C + C^T) of the @p pc or SC correlation elements
// over @p solution, as the interstate coupling forms it, diagonalized: its
// eigenvalues ascending into @p energies, and R, state K's column at
// [K n, (K + 1) n) signed with its largest-magnitude component positive,
// into @p vectors. Every refusal of the effective Hamiltonian, PC's among
// them, is returned unchanged.
[[nodiscard]] Status nevpt2_nac_heff(DeviceResources &res, const Molecule &molecule,
                                     const BasisSet &basis, const BasisSet &aux,
                                     const SphericalMatrixAO &h, const Buffers::device<double> &b,
                                     const mcscf::Solution &solution, int32_t ncore, int32_t ncas,
                                     bool pc, std::vector<double> &energies,
                                     std::vector<double> &vectors) {
  std::vector<double> totals;
  FBF_TRY(nevpt2::qd_inputs(
      res, molecule, basis, aux, h, b, solution, ncore, ncas,
      [&](const nevpt2::MoBlocks &blocks,
          const std::span<const nevpt2::EnergyInputs> pairs) -> Status {
        if (pc) {
          const auto elements = FBF_TRY(nevpt2::pc_effective_hamiltonian(res, blocks, pairs, 0));
          std::ranges::transform(elements, std::back_inserter(totals),
                                 &nevpt2::PcClassEnergies::total);
        } else {
          const auto elements = FBF_TRY(nevpt2::sc_effective_hamiltonian(res, blocks, pairs, 0));
          std::ranges::transform(elements, std::back_inserter(totals),
                                 &nevpt2::ClassEnergies::total);
        }
        return {};
      }));
  const std::vector<double> &e = solution.roots.energies;
  const size_t n = e.size();
  vectors.assign(n * n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      vectors[i + (n * j)] =
          (i == j ? e[i] : 0.0) + (0.5 * (totals[(i * n) + j] + totals[(j * n) + i]));
    }
  }
  energies.assign(n, 0.0);
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  Buffers::device<double> a(n * n, res.device());
  Buffers::device<double> w(n, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a.data(), vectors.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(diagonalize<double>(res, static_cast<int32_t>(n), a.data(), w.data()));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors.data(), a.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(energies.data(), w.data(), n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  for (size_t k = 0; k < n; ++k) {
    const std::span<double> v{vectors.data() + (k * n), n};
    if (*std::ranges::max_element(v, {}, [](double x) { return std::abs(x); }) < 0.0) {
      std::ranges::transform(v, v.begin(), std::negate<>{});
    }
  }
  return {};
}

Result<Nevpt2NacOutput>
Nevpt2NacStep::run(View<MoleculeOutput, BasisSetOutput, PrimitivePairClassesOutput, EriOutput,
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
  // The Casscf step runs mcscf::DensityFitting::Full exactly when the
  // EriOutput is fitted, so this one check is also the one on its mode.
  const auto *const fit = std::get_if<DensityFitted>(&in.get<EriOutput>().eri);
  if (fit == nullptr) {
    return err_unsupported("[{}] runs over density fitting only (the DF-SA-CASSCF's QD-NEVPT2), "
                           "and [eri] mode is 'integral_direct': set [eri] mode = "
                           "'density_fitting'",
                           kName);
  }
  const CasscfOutput &cas = in.get<CasscfOutput>();
  const auto nroots = static_cast<int32_t>(cas.energies.size());
  if (nroots < 2 || cas.weights.size() != cas.energies.size()) {
    return err_config("[{}] couples QD states of a state average, and the CASSCF has {} root(s) "
                      "and {} weight(s): set [casscf] nroots to 2 or more",
                      kName, nroots, cas.weights.size());
  }
  // H_eff has as many states as the CASSCF has roots.
  std::vector<std::array<int32_t, 2>> chosen = pairs;
  if (chosen.empty()) {
    for (int32_t p = 0; p < nroots; ++p) {
      for (int32_t q = p + 1; q < nroots; ++q) {
        chosen.push_back({p, q});
      }
    }
  }
  for (const auto &[bra, ket] : chosen) {
    if (bra < 0 || bra >= nroots || ket < 0 || ket >= nroots || bra == ket) {
      return err_config("[{}] pair ({}, {}) is not two distinct QD states of H_eff over the "
                        "CASSCF's {} roots",
                        kName, bra, ket, nroots);
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

  // The Nevpt2Gradient step's Solution: the product's orbitals (core and
  // virtuals canonical over the SA Fock) and CI vectors on the device, gamma
  // the weighted average of the roots' 1-RDMs, rdm2 already weighted.
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

  // H_eff and R first, the coupling's own formation of them: the states the
  // pairs name, and a PC refusal before any Z-vector is solved.
  const bool pc = variant == Nevpt2Variant::Pc;
  const std::string_view scheme = pc ? "pc" : "sc";
  Nevpt2NacOutput out{.energies = {}, .vectors = {}, .pairs = chosen, .couplings = {}};
  FBF_TRY(nevpt2_nac_heff(res, molecule, basis, fit->aux, h, fit->b, solution, cas.ncore, cas.ncas,
                          pc, out.energies, out.vectors));

  // Batch 0: each class's whole extent in one slab, as the Nevpt2Gradient
  // step runs its QD gradients.
  json couplings = json::array();
  for (const auto &[bra, ket] : chosen) {
    Result<nevpt2::QdNac> coupling =
        pc ? nevpt2::qd_pc_nac(res, molecule, basis, classes, fit->aux, h, fit->b, fit->l.data(),
                               solution, cas.ncore, cas.ncas, bra, ket, 0, options)
           : nevpt2::qd_sc_nac(res, molecule, basis, classes, fit->aux, h, fit->b, fit->l.data(),
                               solution, cas.ncore, cas.ncas, bra, ket, 0, options);
    nevpt2::QdNac d = FBF_TRY(std::move(coupling));
    const double gap =
        out.energies[static_cast<size_t>(ket)] - out.energies[static_cast<size_t>(bra)];
    const double largest = std::ranges::fold_left(
        d.total, 0.0, [](double m, double x) { return std::max(m, std::abs(x)); });
    FBF_LOG_INFO("nevpt2_nac: {}-QD <{}|d{}>, E_Q - E_P = {:.10f}, max |d| {:.6e}, Z-vectors in "
                 "{} products",
                 scheme, bra, ket, gap, largest, d.iterations);
    couplings.push_back(json{
        {"bra", bra},
        {"ket", ket},
        {"energy_gap", gap},
        {"h", nevpt2_nac_rows(d.numerator)},
        {"cas", nevpt2_nac_rows(d.cas)},
        {"pt2", nevpt2_nac_rows(d.pt2)},
        {"csf", nevpt2_nac_rows(d.csf)},
        {"total", nevpt2_nac_rows(d.total)},
        {"heff_gap", d.heff_gap},
        // null when no T^PQ_MN couples two roots (infinity).
        {"root_gap", std::isfinite(d.root_gap) ? json(d.root_gap) : json(nullptr)},
        {"max_zeta", d.max_zeta},
        {"zvector", {{"iterations", d.iterations}, {"residual", d.residual}}},
    });
    out.couplings.push_back(std::move(d));
  }
  if (!json_path.empty()) {
    json r = json::array();
    const size_t n = out.energies.size();
    for (size_t k = 0; k < n; ++k) {
      r.push_back(
          std::vector<double>(out.vectors.begin() + static_cast<std::ptrdiff_t>(k * n),
                              out.vectors.begin() + static_cast<std::ptrdiff_t>((k + 1) * n)));
    }
    FBF_TRY(write_json(json_path, json{
                                      {"variant", scheme},
                                      {"mult_ediff", options.mult_ediff},
                                      {"use_etfs", options.use_etfs},
                                      {"e_casscf", cas.energies},
                                      {"qd_energies", out.energies},
                                      {"r", std::move(r)},
                                      {"couplings", std::move(couplings)},
                                  }));
  }
  return out;
}

} // namespace free_breakfast
