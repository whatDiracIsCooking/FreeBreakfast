/// @file
/// @brief The active-space selection a step over the @ref ScfOutput's orbitals takes:
/// by index, or by AVAS over MINAO, and the rotation of the ScfOutput's orbitals it
/// gives. Not a step: @ref ActiveSpaceSelection is what a step's config derives
/// from (`CasciStep`), and @ref ActiveSpaceSelection::select what its run calls.
///
/// Device-shaped. AVAS's MINAO overlaps (`overlap_mixed_spherical`) and its
/// eigenproblems (`diagonalize`) are private to `select`. See README.md.

export module free_breakfast.pipeline.active_space;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.scf; // Molecule, BasisSet, ScfOutput, SphericalMatrixAO

export namespace free_breakfast {

/// @brief An active space as a rotation of the @ref ScfOutput's orbitals.
struct ActiveSpace {
  /// @brief The doubly occupied core orbitals: columns [0, ncore) of @ref c.
  int32_t ncore;
  /// @brief The active orbitals, an open shell's singly occupied ones among
  /// them by index: columns [ncore, ncore + ncas) of @ref c.
  int32_t ncas;
  /// @brief The active electrons, N - 2 ncore.
  int32_t nelecas;
  /// @brief R, nmo x nmo, column-major, orthogonal: column j of @ref c is the
  /// ScfOutput's orbitals combined by column j of R. By index, a permutation.
  std::vector<double> r;
  /// @brief C R, AO by MO: core, active, the rest. The ScfOutput's columns in the
  /// order @ref ActiveSpaceSelection::orbital_order gives, exactly; or, under
  /// @ref ActiveSpaceSelection::avas, AVAS's rotation of them, each of the
  /// three spaces canonical (diagonal in the ScfOutput's Fock matrix).
  SphericalMatrixAO c;
};

/// @brief The options that pick an active space: by index (@ref ncas,
/// @ref nelecas, @ref active) or by AVAS (@ref avas, @ref avas_threshold).
struct ActiveSpaceSelection {
  /// @brief The active orbitals. Required in a config without @ref avas; at
  /// least 1. Left at 0 under @ref avas, which picks it.
  int32_t ncas = 0;
  /// @brief The active electrons, in [0, 2 ncas]. Required in a config
  /// without @ref avas. The rest, N - nelecas, fill the core, so it is even.
  /// Left at 0 under @ref avas, which picks it.
  int32_t nelecas = 0;
  /// @brief The active orbitals as 0-based indices into the ScfOutput's orbitals
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
  /// active, and core, active and the rest are each made canonical. An open
  /// shell's occupied orbitals are its first n_alpha, the singly occupied
  /// ones projected with the doubly occupied (PySCF's `openshell_option` 2).
  std::vector<std::string> avas;
  /// @brief The projection weight an orbital needs to be active under
  /// @ref avas, in (0, 1): PySCF's default.
  double avas_threshold = 0.2;

  /// @brief The ScfOutput's orbital at each column of @ref ActiveSpace::c for a
  /// molecule of @p n_electrons over @p nmo orbitals: the core, the
  /// (N - nelecas) / 2 lowest orbitals not active, ascending; then the active
  /// ones; then the rest, ascending.
  ///
  /// @return The nmo indices; InvalidConfig when ncas is below 1, nelecas is
  ///         negative or above 2 ncas or N, N - nelecas is odd, ncore + ncas
  ///         exceeds nmo, or @ref active is not ncas distinct indices in
  ///         [0, nmo).
  [[nodiscard]] Result<std::vector<int32_t>> orbital_order(int32_t n_electrons, int32_t nmo) const;

  /// @brief The active space over @p scf's orbitals, on @p res's device: by
  /// @ref orbital_order's permutation over its n_alpha + n_beta electrons, or,
  /// under @ref avas, AVAS's rotation with its first n_alpha orbitals occupied.
  /// For an open shell (ROHF, n_alpha > n_beta) by index, the core is of the
  /// first n_beta orbitals, the doubly occupied ones, and the singly occupied
  /// ones, [n_beta, n_alpha), are all active. Synchronizes res.stream(): under
  /// @ref avas, once for the overlaps and once per eigenproblem.
  ///
  /// @return The space; @ref orbital_order's InvalidConfig; for an open shell
  ///         by index, InvalidConfig for a core orbital not doubly occupied
  ///         or a singly occupied one not active; under @ref avas,
  ///         InvalidConfig for a malformed label or one naming no MINAO AO of
  ///         the molecule, @ref avas_threshold outside (0, 1), @ref ncas,
  ///         @ref nelecas or @ref active set beside it, or no orbital
  ///         selected, Numerical for a labelled MINAO overlap not positive
  ///         definite, or `overlap_mixed_spherical`'s or `diagonalize`'s
  ///         error; and GpuRuntime for a failed copy or BLAS call.
  [[nodiscard]] Result<ActiveSpace> select(DeviceResources &res, const Molecule &molecule,
                                           const BasisSet &basis, const ScfOutput &scf) const;
};

} // namespace free_breakfast
