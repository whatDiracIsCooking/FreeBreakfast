/// @file
/// @brief The `BasisSet` step: read a BSE JSON basis and place it over the
///        `Molecule`.
///
/// A thin wrapper over io.basis's read_basis and basis's place, with the
/// per-primitive normalization folded in between. Host-shaped: place reads
/// only the molecule's atoms, never its device nuclei.

export module free_breakfast.pipeline.basis_set;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.molecule;
import free_breakfast.io.basis;
import free_breakfast.pipeline.step;
export import free_breakfast.basis;

#include "diagnostics.h"

export namespace free_breakfast {

/// @brief Reads @ref basis and places it over the @ref Molecule's atoms.
struct BasisSetStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "basis_set";

  /// @brief The BSE JSON basis file. Resolved by the front end, so a relative
  /// path here is relative to the working directory.
  std::filesystem::path basis{};

  /// @brief Read @ref basis, normalize its primitives and place it. A missing
  /// or syntactically invalid file is ErrorKind::IO; a malformed basis, or one
  /// lacking an element the molecule uses, is ErrorKind::InvalidConfig.
  ///
  /// BSE coefficients are for normalized primitives, and the integral engines
  /// take coefficients as given, so the normalization is folded in here, once.
  [[nodiscard]] Result<BasisSet> run(View<Molecule> in) const {
    const BasisStamp stamp = FBF_TRY(read_basis(basis));
    return place(stamp.primitive_normalized(), in.get<Molecule>());
  }
};

} // namespace free_breakfast
