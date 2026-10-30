/// @file
/// @brief OptimizeStep: geometry::optimize over a nested Pipeline per
/// evaluation, and the trajectory, final.xyz and history it writes.

module free_breakfast.pipeline.optimize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kAngstromPerBohr
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.atom; // element_symbol
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.pipeline;

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
  // Every evaluated geometry, in order, for the trajectory.
  std::vector<std::vector<Atom>> frames;
  const auto energy_and_gradient = [&](std::span<const double> x,
                                       std::span<double> gradient) -> Result<double> {
    frames.push_back(placed(start.atoms(), x));
    auto chain =
        FBF_TRY((Pipeline<TrialGeometryStep, BasisSetStep, SphericalTransformStep,
                          PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                          CoreHamiltonianStep, GuessStep, ScfStep, GradientStep>::create({
            TrialGeometryStep{frames.back(), start.charge(), start.multiplicity()},
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
    const std::vector<double> &g = chain.get<GradientOutput>()->gradient.total;
    std::ranges::copy(g, gradient.begin());
    const double energy = chain.get<ScfOutput>()->energy;
    FBF_LOG_INFO("optimize: evaluation {:>3} E {:.10f} max |g| {:.3e}", frames.size() - 1, energy,
                 std::ranges::max(g, {}, [](double v) { return std::abs(v); }));
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
