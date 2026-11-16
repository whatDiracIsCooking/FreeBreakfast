/// @file
/// @brief The `Molecule` step: the `GeometryOutput`'s atoms, with a charge
///        and a multiplicity, into a `MoleculeOutput` on the device.
///
/// The head of the `[molecule]` table, whose `xyz` key is XyzStep's: the file
/// is read on the host by that step, and this one only builds the Molecule,
/// adding the charge and multiplicity XYZ does not carry. Device-shaped
/// because a Molecule uploads its nuclei at construction. See README.md for
/// the one device contract it does not keep.

export module free_breakfast.pipeline.molecule;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.footprint;
export import free_breakfast.molecule;
export import free_breakfast.pipeline.xyz;

export namespace free_breakfast {

/// @brief The geometry read, its nuclei on the device the step ran on.
struct MoleculeOutput {
  Molecule molecule;
};

/// @brief Builds a @ref Molecule of the GeometryOutput's atoms on @p res's
/// device.
struct MoleculeStep {
  /// @brief The step's name, and the TOML table it heads.
  static constexpr std::string_view kName = "molecule";
  /// @brief The keys of the table it reads; `xyz` is XyzStep's.
  static constexpr std::array<std::string_view, 2> kKeys{"charge", "multiplicity"};

  /// @brief Net molecular charge (XYZ carries none).
  int32_t charge{0};
  /// @brief Spin multiplicity 2S+1 (XYZ carries none).
  int32_t multiplicity{1};

  /// @brief The Molecule of the geometry's atoms, at @ref charge and
  /// @ref multiplicity.
  [[nodiscard]] Result<MoleculeOutput> run(View<GeometryOutput> in, DeviceResources &res) const {
    return MoleculeOutput{Molecule(res, in.get<GeometryOutput>().atoms, charge, multiplicity)};
  }

  /// @brief The Molecule's nuclei, Z and the positions, persistent; their
  /// staging, the pinned workspace (@ref molecule_footprint).
  [[nodiscard]] MemoryFootprint footprint(const FootprintDimensions &dims) const {
    return molecule_footprint(dims.natoms);
  }
};

} // namespace free_breakfast
