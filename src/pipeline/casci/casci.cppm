/// @file
/// @brief The `Casci` step: the lowest singlets of an active space of the
/// @ref Scf's orbitals, named by index or picked by AVAS, its Hamiltonian
/// built integral-direct and solved over the GPU sigma, with each root's
/// 1-RDM.
///
/// Device-shaped. A wrapper over `integrals.active_space`, `ci.solver` over
/// `ci.sigma`'s `CiSigma`, and `ci.rdm`, all private to `run`, as are AVAS's
/// MINAO overlaps (`overlap_mixed_spherical`) and the Molden writer. See README.md.

export module free_breakfast.pipeline.casci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.pipeline.step;
export import free_breakfast.recipes.davidson; // DavidsonInfo, in the product
export import free_breakfast.pipeline.scf;

export namespace free_breakfast {

/// @brief The lowest roots of a complete active space over a set of RHF
/// orbitals, ascending in energy.
struct Casci {
  /// @brief The doubly occupied core orbitals: columns [0, ncore) of @ref c.
  int32_t ncore;
  /// @brief The active orbitals: columns [ncore, ncore + ncas) of @ref c.
  int32_t ncas;
  /// @brief The active electrons, nelecas / 2 of each spin.
  int32_t nelecas;
  /// @brief The orbitals, AO by MO: core, active, the rest. The Scf's columns
  /// in the order @ref CasciStep::orbital_order gives; or, under
  /// @ref CasciStep::avas, AVAS's rotation of them, each of the three spaces
  /// canonical (diagonal in the Scf's Fock matrix).
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

/// @brief CASCI over the @ref Scf's orbitals, on @p res's device.
struct CasciStep {
  /// @brief The step's TOML table name.
  static constexpr std::string_view kName = "casci";

  /// @brief The active orbitals. Required in a config without @ref avas; at
  /// least 1. Left at 0 under @ref avas, which picks it.
  int32_t ncas = 0;
  /// @brief The active electrons, in [0, 2 ncas]. Required in a config
  /// without @ref avas. The rest, N - nelecas, fill the core, so it is even.
  /// Left at 0 under @ref avas, which picks it.
  int32_t nelecas = 0;
  /// @brief The singlet roots wanted, at least 1.
  int32_t nroots = 1;
  /// @brief The active orbitals as 0-based indices into the Scf's orbitals
  /// (ascending in energy), ncas of them, distinct, in the order they become
  /// columns. Empty: the frontier orbitals, the ncas above the core. Left
  /// empty under @ref avas.
  std::vector<int32_t> active;
  /// @brief AVAS's MINAO AO labels, each `<symbol> <n><l>` (`"O 2p"`,
  /// `"Fe 3d"`): every component of that shell of MINAO on every atom of
  /// that element. Not empty: AVAS (Sayfutyarova et al., JCTC 13, 4063
  /// (2017)), as PySCF's `mcscf.avas`, picks the active space instead of
  /// @ref ncas, @ref nelecas and @ref active. The occupied and the virtual
  /// orbitals are each rotated to the eigenvectors of their projector onto
  /// the labelled AOs; those of weight at least @ref avas_threshold are
  /// active, and core, active and the rest are each made canonical.
  std::vector<std::string> avas;
  /// @brief The projection weight an orbital needs to be active under
  /// @ref avas, in (0, 1): PySCF's default.
  double avas_threshold = 0.2;
  /// @brief Where the lowest root's natural orbitals are written as a Molden
  /// file, if not empty: the core at occupation 2, the active space's
  /// natural orbitals by descending occupation, the rest at 0, each with its
  /// energy in the Scf's Fock matrix.
  std::filesystem::path molden{};

  /// @brief The Scf's orbital at each column of @ref Casci::c for a molecule
  /// of @p n_electrons over @p nmo orbitals: the core, the (N - nelecas) / 2
  /// lowest orbitals not active, ascending; then the active ones; then the
  /// rest, ascending.
  ///
  /// @return The nmo indices; InvalidConfig when ncas is below 1, nelecas is
  ///         negative or above 2 ncas or N, N - nelecas is odd, ncore + ncas
  ///         exceeds nmo, or @ref active is not ncas distinct indices in
  ///         [0, nmo).
  [[nodiscard]] Result<std::vector<int32_t>> orbital_order(int32_t n_electrons, int32_t nmo) const;

  /// @brief The @ref nroots lowest singlets. The active-space Hamiltonian is
  /// `active_space_hamiltonian`'s over the orbitals @ref orbital_order puts
  /// in place, or @ref avas picks, at its own default screens; the roots are `ci_solve`'s over
  /// `CiSigma`, at its default spin penalty and tolerance; each 1-RDM is
  /// `rdm1`'s. Synchronizes res.stream(): the integrals and every Davidson
  /// iteration are read back to the host.
  ///
  /// @return The roots; @ref orbital_order's InvalidConfig; under @ref avas,
  ///         InvalidConfig for a malformed label or one naming no MINAO AO of
  ///         the molecule, @ref avas_threshold outside (0, 1), @ref ncas,
  ///         @ref nelecas or @ref active set beside it, or no orbital
  ///         selected; or the error of the first module call that failed:
  ///         `overlap_mixed_spherical`'s, `diagonalize`'s,
  ///         `active_space_hamiltonian`'s,
  ///         `CiSigma`'s, `ci_solve`'s (InvalidConfig for @ref nroots
  ///         outside [1, dim], MaxIterationsExceeded, Numerical for a root of
  ///         the wrong spin), or `write_molden`'s, and GpuRuntime for a failed
  ///         copy or BLAS call.
  [[nodiscard]] Result<Casci>
  run(View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses, CoreHamiltonian, Scf> in,
      DeviceResources &res) const;
};

} // namespace free_breakfast
