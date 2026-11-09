/// @file
/// @brief The plan's shape: a planned Calculation's bases placed and its
///        FootprintDimensions filled, before the first step runs.
///
/// Not a step, and not in pipeline.config: it needs the parsed Calculation,
/// and pipeline.footprint, which every step module imports, must not import
/// the registry. See README.md.

export module free_breakfast.pipeline.shape;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.pipeline.footprint;
export import free_breakfast.pipeline.config;

export namespace free_breakfast {

/// @brief The bases a plan places and the dimensions that point into them.
///
/// The bases are held by pointer so that @ref dims, whose @c basis and
/// @c aux point at them, stays valid when the shape is moved.
struct PlanShape {
  /// @brief The orbital basis; null for a plan with no basis step.
  std::unique_ptr<const BasisSet> basis;
  /// @brief The density-fitting aux basis; null without density fitting.
  std::unique_ptr<const BasisSet> aux;
  /// @brief MINAO over the molecule; null unless a planned active space is
  ///        AVAS's.
  std::unique_ptr<const BasisSet> minao;
  /// @brief The plan's dimensions, @c basis, @c aux and @c minao pointing at
  ///        the three above.
  FootprintDimensions dims;
};

/// @brief Place @p calc's bases and fill its @ref FootprintDimensions, on
///        @p res's device, before @p calc runs.
///
/// The molecule is the Molecule step's, or under an `[optimize]` the start
/// its chain walks from (the atoms its loaded final.xyz must hold, which are
/// all the dimensions read). The bases are placed as the BasisSet and Eri
/// steps place them, from the top-level steps or, failing those, an
/// `[optimize]`'s or `[hessian]`'s chain, which the config copies from the
/// same tables. The counts:
///
/// - nao, naux, nocc = n_beta, n_alpha, n_beta, natoms and the atoms, from
///   the bases and the molecule's electrons at its multiplicity;
/// - mmd_records, the sum over shell pairs i <= j of nprim_i nprim_j per
///   class (mmd screens nothing, so this is every class's record count);
/// - syevd_lwork at every order through the plan's largest eigensolve, nao
///   or a planned CI's subspace (16 nroots, and for a CASSCF
///   ah_max_subspace + 1), queried through res.solver() (host computations);
/// - MINAO, placed, when a planned active space is AVAS's;
/// - ncas and nelecas, the largest active space a planned step may take:
///   an explicit ncas and nelecas; under AVAS, avas_max_ncas when set, else
///   twice the labelled MINAO AOs (each of the occupied and the virtual
///   projectors has at most that rank), nelecas then at half filling, ncas,
///   the CI's largest space; FCI's nao and every electron. Each is held to
///   nao and the electron count. Of several, the one of most orbitals, then
///   of nelecas nearest half filling.
///
/// Nothing is run: the Molecule's nuclei and the tables are freed on return.
///
/// @return The shape (empty dims for a plan with no molecule); the Molecule
///         or BasisSet step's error, or the aux basis's as the Eri step
///         reports it; InvalidConfig for electrons that cannot be at the
///         molecule's multiplicity, or a malformed AVAS label or one naming no
///         MINAO AO; or the GpuRuntime error of the size query.
[[nodiscard]] Result<PlanShape> plan_shape(const Calculation &calc, DeviceResources &res);

} // namespace free_breakfast
