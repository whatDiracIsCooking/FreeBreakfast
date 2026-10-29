/// @file
/// @brief The `Localization` step: the SCF's occupied orbitals localized, by
/// the method its @ref LocalizationMode names.
///
/// A thin wrapper over the localization modules: Pipek-Mezey, and the
/// intrinsic bond orbitals over MINAO; and, in either mode, the localized
/// intrinsic valence virtual orbitals (LIVVOs).
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
  PipekMezey, ///< Mulliken populations (@ref localization::pm_localize)
  Ibo,        ///< Intrinsic-atomic-orbital populations over MINAO (@ref localization::ibo_localize)
};

/// @brief The localized occupied orbitals of a closed-shell RHF wavefunction,
/// on the device the step ran on.
struct LocalizationOutput {
  /// @brief The occupied orbitals localized: nao_sphe x @ref n_occ,
  /// column-major, C_local = C_occ U, in ascending order of their Fock
  /// expectation value sum_k U_ki^2 eps_k (stably, so ties keep the
  /// localizer's order).
  Buffers::device<double> c;
  /// @brief The n_occ x n_occ orthogonal U, column-major.
  Buffers::device<double> u;
  /// @brief The occupied count, ScfOutput::n_alpha: the columns of @ref c.
  int32_t n_occ;
  /// @brief The sweeps taken and the functional before and after.
  localization::Info info;
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
  /// @brief LIVVO only, empty otherwise: the localized valence virtual
  /// orbitals, nao_sphe x @ref n_vvo, column-major (@ref localization::livvo_localize).
  Buffers::device<double> livvo;
  /// @brief LIVVO only, zero otherwise: the LIVVO count, MINAO's nao_sphe
  /// less @ref n_occ.
  int32_t n_vvo = 0;
  /// @brief LIVVO only: the LIVVO sweeps taken and the functional before and
  /// after.
  localization::Info livvo_info{};
  /// @brief LIVVO only, empty otherwise: each LIVVO's centers, angular
  /// character and sigma/pi kind, over its IAO weights (@ref bond_orbitals).
  std::vector<BondOrbital> livvo_bonds{};
  /// @brief LIVVO only, empty otherwise: the valence character t_val of each
  /// canonical virtual, in the SCF's order; they sum to @ref n_vvo
  /// (@ref localization::valence_character).
  std::vector<double> valence_character{};
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
  localization::PmOptions pipek_mezey;
  /// @brief Where the localized orbitals are written as a Molden file, if not
  /// empty: the n_occ of them at occupation 2, each with its Fock expectation
  /// value sum_k U_ki^2 eps_k as its energy, then the SCF's canonical
  /// virtuals at occupation 0 with their own energies. Arrives resolved
  /// (pipeline/config sets it to localization.molden in the run's output
  /// directory).
  std::filesystem::path molden{};
  /// @brief Also build the LIVVOs over MINAO, their bonds and the canonical
  /// virtuals' valence character, in either mode. Their sweep takes
  /// @ref pipek_mezey's options at Knizia's exponent four, whichever it names.
  bool livvo = false;
  /// @brief Where the LIVVOs are written as a Molden file, if not empty: the
  /// n_vvo of them at occupation 0, each with its Fock expectation value over
  /// the canonical virtuals, sum_a (C_vir^T S L)_al^2 eps_a, as its energy.
  /// Needs @ref livvo. Arrives resolved (pipeline/config sets it to
  /// localization_livvo.molden in the run's output directory when @ref livvo
  /// is set).
  std::filesystem::path livvo_molden{};

  /// @brief The localized orbitals, of the ScfOutput's n_alpha occupied
  /// orbitals; an open shell, n_alpha != n_beta, is Unsupported before
  /// anything is allocated. They are sorted by
  /// energy before anything reads them, so every product and file is in that
  /// order. For IBO, MINAO is
  /// resolved and placed over the Molecule first, and an element it does not
  /// carry is place's error; the IAOs it localized over then give the IBOs'
  /// populations and bond centers and the atoms' charges, the charges and
  /// centers logged. With @ref livvo, MINAO is placed in either mode, the
  /// IAOs are built once (localization::ibo_localize's, or
  /// localization::intrinsic_atomic_orbitals' for Pipek-Mezey), and the
  /// LIVVOs' bonds and the virtuals' valence character
  /// are logged; their errors are livvo's. InvalidConfig for a
  /// @ref livvo_molden without @ref livvo. Errors are otherwise the method's;
  /// for both, InvalidConfig for an option out of range,
  /// MaxIterationsExceeded when the sweeps do not converge, GpuRuntime for a
  /// failed launch, copy or BLAS call. A @ref molden or @ref livvo_molden
  /// file that cannot be written is @c write_molden's error.
  [[nodiscard]] Result<LocalizationOutput>
  run(View<MoleculeOutput, BasisSetOutput, OverlapOutput, ScfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
