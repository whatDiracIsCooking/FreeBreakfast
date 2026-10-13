/// @file
/// @brief The Config type itself, shared by the reader (and any future writer).

export module free_breakfast.io.config:shared;

import std;

export namespace free_breakfast {

/// @brief A single-point calculation's configuration, as read from TOML.
///
/// Field defaults are the conventional ones for a closed-shell single point:
/// neutral, singlet, a tight-ish SCF and a generous iteration cap. A file need
/// only override what it cares about; absent keys keep these values.
struct Config {
  /// @brief Electronic-structure method name, e.g. "RHF" (verbatim, uppercased
  /// nowhere -- interpretation is the caller's job).
  std::string method{};
  /// @brief Basis-set name, e.g. "sto-3g".
  std::string basis{};
  /// @brief Auxiliary (density-fitting) basis-set name, e.g. "cc-pVDZ-RI" or
  /// "def2-universal-JKFIT".
  ///
  /// Empty by default and consulted only by the density-fitting path: an SCF or
  /// post-HF method that does not fit its integrals ignores it entirely. When a
  /// method that needs it finds it empty, that is the method's error to report,
  /// not the reader's -- an absent key is not a parse error here, exactly as for
  /// every other @ref Config field.
  std::string aux_basis{};
  /// @brief Net molecular charge in units of the elementary charge.
  std::int32_t charge{0};
  /// @brief Spin multiplicity 2S+1 (1 = singlet).
  std::int32_t multiplicity{1};
  /// @brief SCF energy-convergence threshold (hartree).
  double scf_energy_tolerance{1e-8};
  /// @brief SCF density/commutator-convergence threshold.
  double scf_density_tolerance{1e-6};
  /// @brief Maximum number of SCF iterations before giving up.
  std::int32_t max_iterations{100};
};

} // namespace free_breakfast
