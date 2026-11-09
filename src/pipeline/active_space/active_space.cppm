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
import free_breakfast.pipeline.footprint;
export import free_breakfast.ci.solver;    // SolverOptions, in active_space_ci_footprint
export import free_breakfast.pipeline.scf; // Molecule, BasisSet, ScfOutput, SphericalMatrixAO

export namespace free_breakfast {

/// @brief Which of MINAO's spherical AOs the AVAS @p labels name (select's rule, for its bound).
[[nodiscard]] Result<std::vector<bool>> avas_minao_aos(std::span<const Atom> atoms,
                                                       const BasisSet &minao,
                                                       std::span<const std::string> labels);

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
/// @ref nelecas, @ref active) or by AVAS (@ref avas, @ref avas_threshold,
/// @ref avas_max_ncas).
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
  /// @brief The most active orbitals @ref avas may pick, if above 0: a pick
  /// of more is refused, never truncated, since AVAS is defined by
  /// @ref avas_threshold and a cut could split a degenerate set. 0: no cap.
  int32_t avas_max_ncas = 0;
  /// @brief A float64 .npy of orbitals to start from in place of the
  /// ScfOutput's, if not empty: nao x nmo (AO by MO, this tree's spherical AO
  /// order), as a Casscf step's `orbitals_npy` writes them, of this basis at
  /// this or another geometry. Orthonormalized in this geometry's overlap
  /// S, the core and active ones A by Löwdin, A (A^T S A)^-1/2, then the
  /// rest projected off them and Löwdin'd, they stand in for the ScfOutput's:
  /// @ref active and the core index them, so the frontier default is a
  /// converged CASSCF's own core and active columns. Not with @ref avas.
  std::filesystem::path orbitals{};

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

  /// @brief The active space over @p scf's orbitals, or over @ref orbitals'
  /// when set, on @p res's device: by
  /// @ref orbital_order's permutation over its n_alpha + n_beta electrons, or,
  /// under @ref avas, AVAS's rotation with its first n_alpha orbitals occupied.
  /// For an open shell (ROHF, n_alpha > n_beta) by index, the core is of the
  /// first n_beta orbitals, the doubly occupied ones, and the singly occupied
  /// ones, [n_beta, n_alpha), are all active. Under @ref orbitals the rotation
  /// @ref ActiveSpace::r is of @p scf's orbitals still, C_scf^T S C', and the
  /// largest |A^T S A - I| element the file's core and active orbitals have
  /// here is logged at info. Synchronizes res.stream(): under @ref avas, once for the overlaps
  /// and once per eigenproblem; under @ref orbitals, a few times.
  ///
  /// @return The space; @ref orbital_order's InvalidConfig; for an open shell
  ///         by index, InvalidConfig for a core orbital not doubly occupied
  ///         or a singly occupied one not active; under @ref avas,
  ///         InvalidConfig for a malformed label or one naming no MINAO AO of
  ///         the molecule, @ref avas_threshold outside (0, 1), @ref ncas,
  ///         @ref nelecas or @ref active set beside it, or no orbital
  ///         selected, or more selected than @ref avas_max_ncas, Numerical for a labelled MINAO overlap not positive
  ///         definite, or `overlap_mixed_spherical`'s or `diagonalize`'s
  ///         error; under @ref orbitals, InvalidConfig beside @ref avas or
  ///         for a file not nao x nmo (naming both shapes), `read_npy`'s IO
  ///         error, or `overlap_mixed_spherical`'s or `orthogonalize`'s; and
  ///         GpuRuntime for a failed copy or BLAS call.
  [[nodiscard]] Result<ActiveSpace> select(DeviceResources &res, const Molecule &molecule,
                                           const BasisSet &basis, const ScfOutput &scf) const;

  /// @brief What @ref select holds at @p dims' active space, ncas and
  /// nelecas: ActiveSpace::c, persistent; R's upload, or under @ref orbitals
  /// S, the work, M and X and their overlap and orthogonalizations, or under
  /// @ref avas MINAO's overlaps and the eigenproblems, scratch; and their
  /// workspaces. Under @ref avas it is exact at the space AVAS picks, which
  /// a caller that knows it puts in @p dims (plan_shape's is the largest it
  /// may pick).
  [[nodiscard]] MemoryFootprint select_footprint(const FootprintDimensions &dims) const;
};

/// @brief What a CI over an active space of @p ncas orbitals and
/// @p n_alpha + @p n_beta electrons holds, as CASCI and FCI run it: the
/// DeviceStringSets (one for both spins when N_alpha = N_beta), the
/// ci::Sigma, diag(H), ci::solve's roots at @p options (its scratch under
/// set_scratch_limit @p sigma_scratch_limit; @ref ci::solve_footprint) and
/// the @p nroots spin-summed 1-RDMs, each @ref ci::sigma_footprint's rdm1.
/// Everything is persistent: all of it is held to the step's end.
/// @p syevd_lwork is FootprintDimensions::syevd_lwork.
[[nodiscard]] MemoryFootprint
active_space_ci_footprint(int32_t ncas, int32_t n_alpha, int32_t n_beta, int32_t nroots,
                          const ci::SolverOptions &options, std::span<const size_t> syevd_lwork,
                          size_t sigma_scratch_limit = std::numeric_limits<size_t>::max());

} // namespace free_breakfast
