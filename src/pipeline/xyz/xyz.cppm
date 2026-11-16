/// @file
/// @brief The `Xyz` step: read an XYZ geometry's atoms into a
///        `GeometryOutput`, the pipeline's root product, on the host.
///
/// A thin wrapper over io.xyz's read_xyz. A root step, so it reads nothing
/// (`View<>`) and its one option, the file, is its input. Host-shaped: the
/// atoms are host data, and the device copy of the nuclei is MoleculeStep's.
/// It shares the `[molecule]` table with MoleculeStep, the table's head, so
/// `[molecule] xyz = ...` is this step's key and charge and multiplicity
/// are MoleculeStep's.
///
/// It also declares `OptimizeOutput`, the product of pipeline.optimize's
/// `OptimizeStep`, because the step's `load` reads it: after an `[optimize]`,
/// the geometry is the one that step wrote. pipeline.optimize imports every
/// step it reruns, each of which imports this module, so the product cannot
/// be declared there.

export module free_breakfast.pipeline.xyz;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.footprint;
export import free_breakfast.atom;
export import free_breakfast.molecule;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief The geometry read: its atoms, positions in bohr, in file order.
struct GeometryOutput {
  std::vector<Atom> atoms;
};

/// @brief A geometry optimization's result: the product of `OptimizeStep`
/// (pipeline.optimize), declared here for XyzStep::load.
struct OptimizeOutput {
  /// @brief The optimized geometry, on the device the step ran on.
  Molecule molecule;
  /// @brief The energy there (hartree): the SCF's at the last accepted
  /// evaluation, under a CASSCF chain its CASSCF root's, under a NEVPT2
  /// chain its NEVPT2 (or QD state's) energy, and under `meci` the mean
  /// (E_I + E_J) / 2 of the two crossing roots.
  double energy;
};

/// @brief Reads @ref xyz into its atoms, on the host.
struct XyzStep {
  /// @brief The step's name: its log lines and the registry.
  static constexpr std::string_view kName = "xyz";
  /// @brief The TOML table its option is read from, MoleculeStep's.
  static constexpr std::string_view kTable = "molecule";
  /// @brief The keys of @ref kTable it reads.
  static constexpr std::array<std::string_view, 1> kKeys{"xyz"};

  /// @brief The XYZ geometry file (Angstrom). Resolved by the front end, so
  /// a relative path here is relative to the working directory.
  std::filesystem::path xyz{};

  /// @brief Read @ref xyz. A missing or malformed file is ErrorKind::IO.
  [[nodiscard]] Result<GeometryOutput> run(View<> /*in*/) const {
    return GeometryOutput{FBF_TRY(read_xyz(xyz))};
  }

  /// @brief Nothing: host-shaped, the atoms are host data.
  [[nodiscard]] MemoryFootprint footprint(const FootprintDimensions & /*dims*/) const { return {}; }

  /// @brief The geometry an `[optimize]` wrote to @p from (its final.xyz), in
  /// place of @ref run, so every step reading the geometry runs at the
  /// optimum. The file must hold the optimized molecule's atoms, element by
  /// element, or it is ErrorKind::InvalidConfig. A missing or malformed file
  /// is ErrorKind::IO.
  [[nodiscard]] static Result<GeometryOutput> load(const std::filesystem::path &from,
                                                   View<OptimizeOutput> in) {
    const Molecule &optimized = in.get<OptimizeOutput>().molecule;
    std::vector<Atom> atoms = FBF_TRY(read_xyz(from));
    if (!std::ranges::equal(atoms, optimized.atoms(),
                            [](const Atom &a, const Atom &b) { return a.element == b.element; })) {
      return err_config("[{}] {} does not hold the {} atoms [optimize] optimized", kTable,
                        from.string(), optimized.natoms());
    }
    return GeometryOutput{std::move(atoms)};
  }
};

} // namespace free_breakfast
