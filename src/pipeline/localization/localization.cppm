/// @file
/// @brief The `Localization` step: the SCF's occupied orbitals localized, by
/// the method its @ref LocalizationMode names.
///
/// A thin wrapper over the localization modules: Pipek-Mezey, and the
/// intrinsic bond orbitals over MINAO.
/// Device-shaped. See README.md.

export module free_breakfast.pipeline.localization;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.basis_set;
import free_breakfast.pipeline.step;
export import free_breakfast.core.wwr;         // Buffers, in the product
export import free_breakfast.localization.ibo; // and pipek_mezey
export import free_breakfast.analysis.population;
export import free_breakfast.analysis.bond;
export import free_breakfast.pipeline.scf;

export namespace free_breakfast {

/// @brief Which localization @ref LocalizationStep runs.
enum class LocalizationMode : uint8_t {
  PipekMezey, ///< Mulliken populations (@ref pipek_mezey_localize)
  Ibo,        ///< Intrinsic-atomic-orbital populations over MINAO (@ref ibo_localize)
};

/// @brief The localized occupied orbitals of a closed-shell RHF wavefunction,
/// on the device the step ran on.
struct LocalizationOutput {
  /// @brief The occupied orbitals localized: nao_sphe x @ref n_occ,
  /// column-major, C_local = C_occ U.
  Buffers::device<double> c;
  /// @brief The n_occ x n_occ orthogonal U, column-major.
  Buffers::device<double> u;
  /// @brief The occupied count, electron_count() / 2: the columns of @ref c.
  int32_t n_occ;
  /// @brief The sweeps taken and the functional before and after.
  PipekMezeyInfo info;
  /// @brief IBO only, empty for Pipek-Mezey: each localized orbital's IAO
  /// population on each atom, natoms x @ref n_occ column-major, orbital i's
  /// on atom A at [A + natoms * i] (@ref orbital_populations).
  std::vector<double> populations{};
  /// @brief IBO only, empty for Pipek-Mezey: the IAO partial charge of each
  /// atom, Knizia's eq 3 (@ref partial_charges).
  std::vector<double> charges{};
  /// @brief IBO only, empty for Pipek-Mezey: each localized orbital's centers
  /// and their angular character, at bond_orbitals' default threshold
  /// (@ref bond_orbitals).
  std::vector<BondOrbital> bonds{};
};

/// @brief Localizes the @ref ScfOutput's occupied orbitals on @p res's device.
struct LocalizationStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "localization";

  /// @brief The method.
  LocalizationMode mode = LocalizationMode::PipekMezey;
  /// @brief The sweeps, tolerances and exponent, for either mode: IBO is
  /// Pipek-Mezey over other populations. Knizia's IBO exponent is four, which
  /// pipeline/config sets for mode "ibo" unless the table names another.
  PipekMezeyOptions pipek_mezey;
  /// @brief Where the localized orbitals are written as a Molden file, if not
  /// empty: the n_occ of them at occupation 2, each with its Fock expectation
  /// value sum_k U_ki^2 eps_k as its energy. Arrives resolved
  /// (pipeline/config anchors a relative one to the config file).
  std::filesystem::path molden{};

  /// @brief The localized orbitals. The Scf step has already refused an open
  /// shell, so the occupied count is half the electrons. For IBO, MINAO is
  /// resolved and placed over the Molecule first, and an element it does not
  /// carry is place's error; the IAOs it localized over then give the IBOs'
  /// populations and bond centers and the atoms' charges, the charges and
  /// centers logged. Errors are otherwise the method's;
  /// for both, InvalidConfig for an option out of range,
  /// MaxIterationsExceeded when the sweeps do not converge, GpuRuntime for a
  /// failed launch, copy or BLAS call. A @ref molden file that cannot be
  /// written is @c write_molden's error.
  [[nodiscard]] Result<LocalizationOutput>
  run(View<MoleculeOutput, BasisSetOutput, OverlapOutput, ScfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
