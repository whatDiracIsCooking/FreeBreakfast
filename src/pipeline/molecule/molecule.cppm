/// @file
/// @brief The `Molecule` step: read an XYZ geometry into a `Molecule`, the
///        pipeline's root product.
///
/// A thin wrapper over io.xyz's read_xyz. A root step, so it reads nothing
/// (`View<>`) and its options are its inputs: the file, and the charge and
/// multiplicity XYZ does not carry. Device-shaped because a Molecule uploads
/// its nuclei at construction. See README.md for the one device contract it
/// does not keep.

export module free_breakfast.pipeline.molecule;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.step;
export import free_breakfast.molecule;

export namespace free_breakfast {

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
  [[nodiscard]] Result<Molecule> run(View<> /*in*/, DeviceResources &res) const {
    return read_xyz(res, xyz, charge, multiplicity);
  }
};

} // namespace free_breakfast
