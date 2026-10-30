/// @file
/// @brief The `Molecule` step: read an XYZ geometry into a `MoleculeOutput`,
///        the pipeline's root product.
///
/// A thin wrapper over io.xyz's read_xyz. A root step, so it reads nothing
/// (`View<>`) and its options are its inputs: the file, and the charge and
/// multiplicity XYZ does not carry. Device-shaped because a Molecule uploads
/// its nuclei at construction. See README.md for the one device contract it
/// does not keep.
///
/// It also declares `OptimizeOutput`, the product of pipeline.optimize's
/// `OptimizeStep`, because the step's `load` reads it: after an `[optimize]`,
/// the molecule is the geometry that step wrote. pipeline.optimize imports
/// every step it reruns, each of which imports this module, so the product
/// cannot be declared there.

export module free_breakfast.pipeline.molecule;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.step;
export import free_breakfast.molecule;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The geometry read, its nuclei on the device the step ran on.
struct MoleculeOutput {
  Molecule molecule;
};

/// @brief A geometry optimization's result: the product of `OptimizeStep`
/// (pipeline.optimize), declared here for MoleculeStep::load.
struct OptimizeOutput {
  /// @brief The optimized geometry, on the device the step ran on.
  Molecule molecule;
  /// @brief The energy there (hartree): the SCF's at the last accepted
  /// evaluation.
  double energy;
};

/// @brief Reads @ref xyz into a @ref Molecule on @p res's device.
struct MoleculeStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "molecule";

  /// @brief The XYZ geometry file (Angstrom). Resolved by the front end, so
  /// a relative path here is relative to the working directory.
  std::filesystem::path xyz{};
  /// @brief Net molecular charge (XYZ carries none).
  int32_t charge{0};
  /// @brief Spin multiplicity 2S+1 (XYZ carries none).
  int32_t multiplicity{1};

  /// @brief Read @ref xyz. A missing or malformed file is ErrorKind::IO.
  [[nodiscard]] Result<MoleculeOutput> run(View<> /*in*/, DeviceResources &res) const {
    return MoleculeOutput{FBF_TRY(read_xyz(res, xyz, charge, multiplicity))};
  }

  /// @brief The geometry an `[optimize]` wrote to @p from (its final.xyz), in
  /// place of @ref run, so every step reading the molecule runs at the
  /// optimum. The charge and multiplicity are the optimized molecule's, which
  /// XYZ does not carry; the file must hold its atoms, element by element, or
  /// it is ErrorKind::InvalidConfig. A missing or malformed file is
  /// ErrorKind::IO.
  [[nodiscard]] static Result<MoleculeOutput> load(const std::filesystem::path &from,
                                                   View<OptimizeOutput> in, DeviceResources &res) {
    const Molecule &optimized = in.get<OptimizeOutput>().molecule;
    Molecule molecule = FBF_TRY(read_xyz(res, from, optimized.charge(), optimized.multiplicity()));
    if (!std::ranges::equal(molecule.atoms(), optimized.atoms(),
                            [](const Atom &a, const Atom &b) { return a.element == b.element; })) {
      return err_config("[{}] {} does not hold the {} atoms [optimize] optimized", kName,
                        from.string(), optimized.natoms());
    }
    return MoleculeOutput{std::move(molecule)};
  }
};

} // namespace free_breakfast
