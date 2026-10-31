/// @file
/// @brief HessianStep: the 6N displaced gradients, each a nested Pipeline
/// run of the chain to GradientStep, their central differences, the
/// symmetrized Hessian, its two checks, and hessian.json.

module free_breakfast.pipeline.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.pipeline.pipeline;

#include "diagnostics.h"

namespace free_breakfast {

// The chain's root at one displaced geometry: the molecule from atoms in
// memory, where MoleculeStep reads a file.
struct DisplacedMoleculeStep {
  static constexpr std::string_view kName = "molecule";

  std::vector<Atom> atoms;
  int32_t charge{0};
  int32_t multiplicity{1};

  [[nodiscard]] Result<MoleculeOutput> run(View<> /*in*/, DeviceResources &res) const {
    return MoleculeOutput{Molecule(res, atoms, charge, multiplicity)};
  }
};

Result<HessianOutput> HessianStep::run(View<MoleculeOutput> in, NodeResources &node) const {
  if (node.device_count() == 0) {
    return err_gpu("[{}] needs a GPU device, and none is visible", kName);
  }
  const Molecule &molecule = in.get<MoleculeOutput>().molecule;
  const std::vector<Atom> &atoms = molecule.atoms();
  const size_t n = 3 * atoms.size();

  // The chain writes nothing: its steps' outputs are the outer pipeline's.
  ScfStep chain_scf = scf;
  chain_scf.molden.clear();
  // dE/dR at the geometry with coordinate x (3 A + k) moved by @p shift.
  const auto gradient_at = [&](size_t x, double shift) -> Result<std::vector<double>> {
    DisplacedMoleculeStep displaced{atoms, molecule.charge(), molecule.multiplicity()};
    Vec<double, 3> &position = displaced.atoms[x / 3].position;
    (x % 3 == 0 ? position.x : x % 3 == 1 ? position.y : position.z) += shift;
    auto chain =
        FBF_TRY((Pipeline<DisplacedMoleculeStep, BasisSetStep, SphericalTransformStep,
                          PrimitivePairClassesStep, EriStep, OverlapStep, OrthogonalizerStep,
                          CoreHamiltonianStep, GuessStep, ScfStep, GradientStep>::create({
            std::move(displaced),
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
    return chain.get<GradientOutput>()->gradient.total;
  };

  // Row x the central difference along x, before symmetrizing.
  std::vector<double> raw(n * n);
  for (size_t x = 0; x < n; ++x) {
    const std::vector<double> plus = FBF_TRY(gradient_at(x, displacement));
    const std::vector<double> minus = FBF_TRY(gradient_at(x, -displacement));
    for (size_t y = 0; y < n; ++y) {
      raw[(x * n) + y] = (plus[y] - minus[y]) / (2.0 * displacement);
    }
    FBF_LOG_INFO("hessian: coordinate {:>3} of {}", x + 1, n);
  }
  HessianOutput out{.hessian = std::vector<double>(n * n)};
  for (size_t x = 0; x < n; ++x) {
    for (size_t y = 0; y < n; ++y) {
      out.asymmetry = std::max(out.asymmetry, std::abs(raw[(x * n) + y] - raw[(y * n) + x]));
      out.hessian[(x * n) + y] = 0.5 * (raw[(x * n) + y] + raw[(y * n) + x]);
    }
  }
  for (size_t x = 0; x < n; ++x) {
    for (size_t l = 0; l < 3; ++l) {
      double sum = 0.0;
      for (size_t b = l; b < n; b += 3) {
        sum += out.hessian[(x * n) + b];
      }
      out.translational_sum = std::max(out.translational_sum, std::abs(sum));
    }
  }
  FBF_LOG_INFO("hessian: asymmetry {:.3e}, translational sum {:.3e}", out.asymmetry,
               out.translational_sum);

  if (!json_path.empty()) {
    json rows = json::array();
    for (size_t x = 0; x < n; ++x) {
      rows.push_back(
          std::vector<double>(out.hessian.begin() + static_cast<std::ptrdiff_t>(x * n),
                              out.hessian.begin() + static_cast<std::ptrdiff_t>((x + 1) * n)));
    }
    FBF_TRY(write_json(json_path, json{
                                      {"method", "numerical"},
                                      {"displacement", displacement},
                                      {"hessian", rows},
                                      {"asymmetry", out.asymmetry},
                                      {"translational_sum", out.translational_sum},
                                  }));
  }
  return out;
}

} // namespace free_breakfast
