/// @file
/// @brief The plan's shape: a planned Calculation's bases placed and its
///        FootprintDimensions filled, before the first step runs; and
///        pre-flight, which holds the plan's footprint at that shape to the
///        `[memory]` caps.
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
/// - casscf_nroots and casscf_unequal_weights, the CASSCF step's roots and
///   whether two of its weights differ;
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

/// @brief Pre-flight: hold @p calc's plan to its `[memory]` caps on @p node,
///        and set the node up to run it under them, before the first step.
///
/// With neither cap set it does nothing: an uncapped plan is not checked.
/// Otherwise, in order:
///
/// - a `max_device_mem_per_gpu` that leaves no pool past the reserve outside
///   it (DeviceResources::kDeviceReserve) and the fragmentation headroom
///   (kPoolHeadroomBytes) is refused;
/// - under a pinned cap, the node's devices times it must fit MemAvailable
///   (check_pinned_node);
/// - each of @p node's DeviceResources is rebuilt with the pool capped at
///   `max_device_mem_per_gpu` less the reserve and the pinned total at
///   `max_pinned_mem_per_gpu`;
/// - the plan is shaped on device 0 (plan_shape) and its footprints folded
///   and checked (check_plan), the device peak against the pool's cap less
///   the headroom and, on a pool that rounds a reused block, an eighth of
///   the peak. A step with no footprint yet is refused under a cap. When the
///   only active spaces planned are AVAS's and their worst case does not
///   fit, each AVAS step's `avas_max_ncas` is narrowed to the most orbitals
///   that do, so a larger pick is refused at the step;
/// - the device and pinned workspaces are grown to the plan's maximum, so a
///   footprinted step never grows them;
/// - under a device cap, the pool's cap less the headroom is allocated and
///   freed once, so memory another process holds is reported now, not at a
///   later step; the pool keeps it, its release threshold being the cap.
///
/// A node with no device is left to Pipeline::run to refuse.
///
/// @return InvalidConfig for a cap leaving no pool, the machine check, or
///         check_plan's refusal; Allocation when the device cannot supply
///         the pool's cap, naming its free memory and the cap; plan_shape's
///         error, or a pinned workspace's; GpuRuntime for a failed sync.
[[nodiscard]] Status preflight(Calculation &calc, NodeResources &node);

} // namespace free_breakfast
