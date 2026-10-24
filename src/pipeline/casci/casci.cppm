/// @file
/// @brief The `Casci` step: the lowest singlets of an active space of the
/// @ref ScfOutput's orbitals, named by index or picked by AVAS, its Hamiltonian
/// built integral-direct and solved over the GPU sigma, with each root's
/// 1-RDM.
///
/// Device-shaped. The selection is `pipeline.active_space`'s, which
/// `CasciStep` derives from. A wrapper over `integrals.active_space`,
/// `ci.solver` over `ci.sigma`'s `CiSigma`, which also gives the 1-RDMs, all
/// private to `run`, as is the Molden writer. See README.md.

export module free_breakfast.pipeline.casci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.davidson; // DavidsonInfo, in the product
export import free_breakfast.pipeline.scf;
export import free_breakfast.pipeline.active_space; // CasciStep's base

export namespace free_breakfast {

/// @brief The lowest roots of a complete active space over a set of RHF
/// orbitals, ascending in energy.
struct CasciOutput {
  /// @brief The doubly occupied core orbitals: columns [0, ncore) of @ref c.
  int32_t ncore;
  /// @brief The active orbitals: columns [ncore, ncore + ncas) of @ref c.
  int32_t ncas;
  /// @brief The active electrons, nelecas / 2 of each spin.
  int32_t nelecas;
  /// @brief The orbitals, AO by MO: core, active, the rest. The selection's
  /// @ref ActiveSpace::c.
  SphericalMatrixAO c;
  /// @brief The total energy per root, core and nuclear repulsion included
  /// (hartree).
  std::vector<double> energies;
  /// @brief <S^2> per root.
  std::vector<double> spin_squares;
  /// @brief The unit CI vectors, root r at [r dim, (r + 1) dim), dim =
  /// C(ncas, nelecas / 2)^2, in `ci.sigma`'s layout: determinant
  /// (I_alpha, I_beta) at I_alpha C(ncas, nelecas / 2) + I_beta, the strings
  /// in PySCF's `cistring` order.
  std::vector<double> vectors;
  /// @brief The spin-summed 1-RDM per root over the active orbitals, root r
  /// at [r ncas^2, (r + 1) ncas^2), ncas x ncas, row-major.
  std::vector<double> rdm1;
  /// @brief What the Davidson solve did.
  DavidsonInfo davidson;
};

/// @brief CASCI over the @ref ScfOutput's orbitals, on @p res's device. The
/// active space is the @ref ActiveSpaceSelection's: `ncas`, `nelecas`,
/// `active`, or `avas` with `avas_threshold`.
struct CasciStep : ActiveSpaceSelection {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "casci";

  /// @brief The singlet roots wanted, at least 1.
  int32_t nroots = 1;
  /// @brief Where the lowest root's natural orbitals are written as a Molden
  /// file, if not empty: the core at occupation 2, the active space's
  /// natural orbitals by descending occupation, the rest at 0, each with its
  /// energy in the ScfOutput's Fock matrix.
  std::filesystem::path molden{};

  /// @brief The @ref nroots lowest singlets. The active-space Hamiltonian is
  /// `active_space_hamiltonian`'s over the orbitals
  /// @ref ActiveSpaceSelection::select gives, at its own default screens; the roots are `ci_solve`'s over
  /// `CiSigma`, at its default spin penalty and tolerance; each 1-RDM is
  /// `CiSigma::rdm1`'s. Synchronizes res.stream(): the integrals and every Davidson
  /// iteration are read back to the host.
  ///
  /// @return The roots; @ref ActiveSpaceSelection::select's error; or the
  ///         error of the first module call that failed:
  ///         `active_space_hamiltonian`'s, `CiSigma`'s, `ci_solve`'s
  ///         (InvalidConfig for @ref nroots outside [1, dim],
  ///         MaxIterationsExceeded, Numerical for a root of the wrong spin),
  ///         `diagonalize`'s or `write_molden`'s, and GpuRuntime for a failed
  ///         copy or BLAS call.
  [[nodiscard]] Result<CasciOutput>
  run(View<MoleculeOutput, BasisSetOutput, SphericalTransformOutput, PrimitivePairClassesOutput,
           CoreHamiltonianOutput, ScfOutput> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
