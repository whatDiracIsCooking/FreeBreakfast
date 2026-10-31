/// @file
/// @brief OptimizeStep: geometry::optimize over a nested Pipeline per
/// evaluation, the RHF chain or a CASSCF chain (to its gradient or its
/// SC-NEVPT2's) with its orbitals carried from one evaluation to the next, and the trajectory, final.xyz and history
/// it writes.

module free_breakfast.pipeline.optimize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kAngstromPerBohr, kOne, kZero
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.atom; // element_symbol
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.pipeline;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.orthogonalize;

#include "diagnostics.h"

namespace free_breakfast {

// The chain's root at one geometry: the molecule from atoms in memory, where
// MoleculeStep reads a file. Not MoleculeStep itself, whose load reads this
// step's product: a registry holding it must hold OptimizeStep too.
struct TrialGeometryStep {
  static constexpr std::string_view kName = "molecule";

  std::vector<Atom> atoms;
  int32_t charge{0};
  int32_t multiplicity{1};

  [[nodiscard]] Result<MoleculeOutput> run(View<> /*in*/, DeviceResources &res) const {
    return MoleculeOutput{Molecule(res, atoms, charge, multiplicity)};
  }
};

// The CASSCF chain's ScfOutput at every geometry after the first: the last
// evaluation's CASSCF orbitals, core, active, virtual, Loewdin-orthonormalized
// in this geometry's overlap, C (C^T S C)^-1/2, the orthonormal set nearest
// them, so the active space the CasscfStep starts from is the one it
// converged last time and not a fresh pick that could swap orbitals. Only
// the CasscfStep reads it, by index over its first columns: the energy is
// NaN, no SCF ran, and d is left zero.
struct CarriedOrbitalsStep {
  static constexpr std::string_view kName = "scf";

  // AO by MO, column-major, on the host: the last CasscfOutput's c.
  std::vector<double> c;
  // One per column: the last CasscfOutput's mo_energies.
  std::vector<double> mo_energies;
  // The first geometry's ScfOutput's.
  int32_t n_alpha{0};
  int32_t n_beta{0};

  [[nodiscard]] Result<ScfOutput> run(View<BasisSetOutput, OverlapOutput> in,
                                      DeviceResources &res) const {
    const BasisSet &basis = in.get<BasisSetOutput>().basis;
    const SphericalMatrixAO &s = in.get<OverlapOutput>().s;
    if (c.size() != s.num_elements() || mo_energies.size() != s.n()) {
      return err_config("[{}] carries {} orbital coefficients and {} energies into a basis of {} "
                        "functions",
                        OptimizeStep::kName, c.size(), mo_energies.size(), s.n());
    }
    const ScopedDeviceIndex on{res.dev_idx()};
    const auto n = static_cast<int32_t>(s.n());
    const size_t n2 = s.num_elements();
    // C, M = C^T S C and M^-1/2: held across orthogonalize, which takes the
    // workspace, so each is its own buffer.
    SphericalMatrixAO previous(res, basis);
    Buffers::device<double> m(n2, res.device());
    Buffers::device<double> m_inv_sqrt(n2, res.device());
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(previous.data(), c.data(), n2 * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    {
      const auto [work] = res.device_regions<double>(n2);
      FBF_TRY(gpu_status(gemmwich(res.blas(), n, previous.data(), s.data(), work, m.data())));
    }
    FBF_TRY(orthogonalize<double>(res, n, m.data(), m_inv_sqrt.data()));
    ScfOutput out{
        .energy = std::numeric_limits<double>::quiet_NaN(),
        .iterations = 0,
        .n_alpha = n_alpha,
        .n_beta = n_beta,
        .mo_energies = mo_energies,
        .c = SphericalMatrixAO(res, basis),
        .d = SphericalMatrixAO(res, basis),
    };
    FBF_TRY(gpu_status(gemm(res.blas(), n, &kOne<double>, previous.data(), m_inv_sqrt.data(),
                            &kZero<double>, out.c.data())));
    FBF_TRY(res.sync());
    return out;
  }
};

// @p atoms with their positions replaced by @p x (bohr, 3 per atom).
[[nodiscard]] std::vector<Atom> placed(std::vector<Atom> atoms, std::span<const double> x) {
  for (size_t a = 0; a < atoms.size(); ++a) {
    atoms[a].position = {x[(3 * a) + 0], x[(3 * a) + 1], x[(3 * a) + 2]};
  }
  return atoms;
}

// One XYZ frame of @p atoms (Angstrom), @p comment its second line.
[[nodiscard]] std::string xyz_frame(const std::vector<Atom> &atoms, std::string_view comment) {
  std::string out = std::format("{}\n{}\n", atoms.size(), comment);
  for (const Atom &atom : atoms) {
    out += std::format("{:<2} {:18.12f} {:18.12f} {:18.12f}\n", element_symbol(atom.element),
                       atom.position.x * kAngstromPerBohr<double>,
                       atom.position.y * kAngstromPerBohr<double>,
                       atom.position.z * kAngstromPerBohr<double>);
  }
  return out;
}

Result<OptimizeOutput> OptimizeStep::run(View<> /*in*/, NodeResources &node) const {
  if (node.device_count() == 0) {
    return err_gpu("[{}] needs a GPU device, and none is visible", kName);
  }
  if (casscf_gradient && nevpt2_gradient) {
    return err_config("[{}] optimizes the energy of [{}] or of [{}], and both are set: keep the "
                      "one to optimize",
                      kName, CasscfGradientStep::kName, Nevpt2GradientStep::kName);
  }
  // The two CASSCF gradients' own refusal, made before the first SCF and
  // CASSCF.
  if ((casscf_gradient || nevpt2_gradient) && eri.mode != EriMode::DensityFitting) {
    return err_unsupported("[{}] reruns [{}], which runs over density fitting only, and [{}] mode "
                           "is 'integral_direct': set [{}] mode = 'density_fitting'",
                           kName,
                           casscf_gradient ? CasscfGradientStep::kName : Nevpt2GradientStep::kName,
                           EriStep::kName, EriStep::kName);
  }
  DeviceResources &res = node.device(0);
  const Molecule start =
      FBF_TRY(read_xyz(res, molecule.xyz, molecule.charge, molecule.multiplicity));
  std::vector<double> x0;
  for (const Atom &atom : start.atoms()) {
    x0.insert(x0.end(), {atom.position.x, atom.position.y, atom.position.z});
  }

  // The chain writes nothing: its steps' outputs are the outer pipeline's.
  ScfStep chain_scf = scf;
  chain_scf.molden.clear();
  CasscfStep chain_casscf = casscf;
  chain_casscf.molden.clear();
  CasscfGradientStep chain_casscf_gradient = casscf_gradient.value_or(CasscfGradientStep{});
  chain_casscf_gradient.json_path.clear();
  Nevpt2GradientStep chain_nevpt2_gradient = nevpt2_gradient.value_or(Nevpt2GradientStep{});
  chain_nevpt2_gradient.json_path.clear();
  // After the first CASSCF: its orbitals for the next evaluation's start, and
  // the CasscfStep that takes them, by index, as they come, core, active,
  // virtual.
  std::optional<CarriedOrbitalsStep> carried;
  CasscfStep carried_casscf = chain_casscf;
  // A CASSCF chain's energy and gradient, by @p last, the chain's gradient
  // step, and the orbitals it carries on.
  const auto casscf_result = [&]<class G>(const auto &chain, const G & /*last*/, int32_t n_alpha,
                                          int32_t n_beta,
                                          std::span<double> gradient) -> Result<double> {
    const CasscfOutput &cas = *chain.template get<CasscfOutput>();
    CarriedOrbitalsStep next{.c = std::vector<double>(cas.c.num_elements()),
                             .mo_energies = cas.mo_energies,
                             .n_alpha = n_alpha,
                             .n_beta = n_beta};
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(next.c.data(), cas.c.data(), next.c.size() * sizeof(double),
                                       wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
    carried = std::move(next);
    carried_casscf.active.clear();
    carried_casscf.avas.clear();
    carried_casscf.ncas = cas.ncas;
    carried_casscf.nelecas = cas.nelecas;
    // The gradient ran, so the root is one of the CASSCF's.
    if constexpr (std::same_as<G, CasscfGradientStep>) {
      std::ranges::copy(chain.template get<CasscfGradientOutput>()->gradient.total,
                        gradient.begin());
      return cas.energies[static_cast<size_t>(chain_casscf_gradient.root)];
    } else {
      const nevpt2::ScGradient &g = chain.template get<Nevpt2GradientOutput>()->gradient;
      std::ranges::copy(g.terms.total, gradient.begin());
      return cas.energies[static_cast<size_t>(chain_nevpt2_gradient.root)] + g.energies.total;
    }
  };
  // The CASSCF chain ending in @p last at @p trial: at the first geometry the
  // SCF, and the active space picked from it; at every later one no SCF, the
  // last orbitals carried over.
  const auto casscf_chain = [&]<class G>(const TrialGeometryStep &trial, const G &last,
                                         std::span<double> gradient) -> Result<double> {
    if (!carried) {
      auto chain =
          FBF_TRY((Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep,
                            PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                            CoreHamiltonianStep, GuessStep, ScfStep, CasscfStep, G>::create({
              trial,
              basis_set,
              SphericalTransformStep{},
              PrimitivePairClassesStep{},
              eri,
              OverlapStep{},
              orthogonalizer,
              CoreHamiltonianStep{},
              guess,
              chain_scf,
              chain_casscf,
              last,
          })));
      FBF_TRY(chain.run(node));
      const ScfOutput &rhf = *chain.template get<ScfOutput>();
      return casscf_result(chain, last, rhf.n_alpha, rhf.n_beta, gradient);
    }
    auto chain =
        FBF_TRY((Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep,
                          PrimitivePairClassesStep, EriStep, OverlapStep, CoreHamiltonianStep,
                          CarriedOrbitalsStep, CasscfStep, G>::create({
            trial,
            basis_set,
            SphericalTransformStep{},
            PrimitivePairClassesStep{},
            eri,
            OverlapStep{},
            CoreHamiltonianStep{},
            *carried,
            carried_casscf,
            last,
        })));
    FBF_TRY(chain.run(node));
    return casscf_result(chain, last, carried->n_alpha, carried->n_beta, gradient);
  };
  // Every evaluated geometry, in order, for the trajectory.
  std::vector<std::vector<Atom>> frames;
  const auto energy_and_gradient = [&](std::span<const double> x,
                                       std::span<double> gradient) -> Result<double> {
    frames.push_back(placed(start.atoms(), x));
    const TrialGeometryStep trial{frames.back(), start.charge(), start.multiplicity()};
    double energy = 0.0;
    if (casscf_gradient) {
      energy = FBF_TRY(casscf_chain(trial, chain_casscf_gradient, gradient));
    } else if (nevpt2_gradient) {
      energy = FBF_TRY(casscf_chain(trial, chain_nevpt2_gradient, gradient));
    } else {
      auto chain =
          FBF_TRY((Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep,
                            PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                            CoreHamiltonianStep, GuessStep, ScfStep, GradientStep>::create({
              trial,
              basis_set,
              SphericalTransformStep{},
              PrimitivePairClassesStep{},
              eri,
              OverlapStep{},
              orthogonalizer,
              CoreHamiltonianStep{},
              guess,
              chain_scf,
              GradientStep{},
          })));
      FBF_TRY(chain.run(node));
      std::ranges::copy(chain.get<GradientOutput>()->gradient.total, gradient.begin());
      energy = chain.get<ScfOutput>()->energy;
    }
    FBF_LOG_INFO("optimize: evaluation {:>3} E {:.10f} max |g| {:.3e}", frames.size() - 1, energy,
                 std::ranges::max(gradient, {}, [](double v) { return std::abs(v); }));
    return energy;
  };
  const geometry::Solution solution = FBF_TRY(geometry::optimize(x0, energy_and_gradient, options));

  const std::vector<Atom> optimum = placed(start.atoms(), solution.x);
  if (!trajectory.empty()) {
    std::string text;
    for (size_t i = 0; i < frames.size() && i < solution.iterations.size(); ++i) {
      const geometry::Iteration &it = solution.iterations[i];
      text += xyz_frame(frames[i], std::format("evaluation {} E = {:.12f}{}", i, it.energy,
                                               it.accepted ? "" : " (rejected)"));
    }
    FBF_TRY(write_file(trajectory, text));
  }
  if (!final_xyz.empty()) {
    FBF_TRY(write_file(final_xyz,
                       xyz_frame(optimum, std::format("optimized E = {:.12f}", solution.energy))));
  }
  if (!history.empty()) {
    json iterations = json::array();
    for (const geometry::Iteration &it : solution.iterations) {
      iterations.push_back({
          {"energy", it.energy},
          {"max_force", it.max_force},
          {"rms_force", it.rms_force},
          {"max_displacement", it.max_displacement},
          {"rms_displacement", it.rms_displacement},
          {"trust_radius", it.trust_radius},
          {"accepted", it.accepted},
      });
    }
    FBF_TRY(write_json(history, json{{"energy", solution.energy}, {"iterations", iterations}}));
  }
  return OptimizeOutput{.molecule = Molecule(res, optimum, start.charge(), start.multiplicity()),
                        .energy = solution.energy};
}

} // namespace free_breakfast
