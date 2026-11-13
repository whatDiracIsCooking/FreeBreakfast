/// @file
/// @brief mcscf::root_rotation ported to a Routine<..., Alloc::Yes>: one In,
/// check, footprint and invoke, its own five device buffers keyed; and
/// RootRotationStep, the root that mints the key and calls it as a step
/// would.

export module free_breakfast.experimental.routine:root_rotation;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.footprint;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.derivative;
import free_breakfast.mcscf.root_rotation; // the result type, mcscf::RootRotation
import :key;
import :base;

export namespace free_breakfast::routine {

/// @brief The root-rotation term of an equal-weight state-averaged DF-CASSCF
/// Hessian: mcscf::root_rotation's computation, call for call, as a routine.
///
/// Named for what it is (the routine that produces a RootRotation) because
/// mcscf::RootRotation, the result, already holds the bare noun; the object
/// that is called keeps the function's name, `root_rotation`, so a call site
/// changes its argument list and nothing else. README.md "Naming".
struct RootRotationRoutine : Routine<RootRotationRoutine, Alloc::Yes> {
  /// @brief mcscf::root_rotation's arguments, in its order, by name.
  struct In {
    DeviceResources &res;                  ///< queue, BLAS and pool
    const Molecule &molecule;              ///< as df_derivatives takes it
    const BasisSet &basis;                 ///< the orbital basis
    const SphericalMatrixAO &h;            ///< the core Hamiltonian, nao x nao
    const Buffers::device<double> &b;      ///< density_fit's B, naux nao^2
    const SphericalMatrixAO &c;            ///< the orbitals, nao x nao
    int32_t ncore;                         ///< core orbitals
    int32_t ncas;                          ///< active orbitals
    int32_t nalpha;                        ///< active alpha electrons
    int32_t nbeta;                         ///< active beta electrons
    const double *roots;                   ///< the roots on the device, dim x nroots
    std::span<const double> weights;       ///< the average's, all equal
    std::span<const double> energies;      ///< one per weight
    const mcscf::Derivatives &derivatives; ///< df_derivatives' at these roots
    const double *kappa;                   ///< lambda^x's orbital part, n_pairs x n
    double max_rotation = 10.0;            ///< bohr^-1, the refusal's threshold
  };

  /// @brief mcscf::root_rotation's InvalidConfig refusals as aborts: a
  /// broken shape is the caller's bug (#1226), so it logs the condition and
  /// aborts rather than returning. The degeneracy refusals are numerical
  /// outcomes a caller can branch on, and stay in invoke as Numerical.
  static void check(const In &in);

  /// @brief What invoke holds, all freed on return:
  /// mcscf::root_rotation_footprint's sum, over the same In (naux read off
  /// B's length, nroots off the weights).
  [[nodiscard]] static MemoryFootprint footprint(const In &in);

  /// @brief The computation, after check: the rotations and the term, or
  /// Numerical near a degeneracy, or a callee's first error.
  [[nodiscard]] static Result<mcscf::RootRotation> invoke(const AllocKey &key, const In &in);
};

/// @brief The routine, called as `root_rotation(key, {.res = ..., ...})`.
inline constexpr RootRotationRoutine root_rotation{};

/// @brief The root: mints the key, as the pipeline's planner would, and
/// calls the port the way a step's run does.
struct RootRotationStep {
  /// @brief root_rotation over @p in, with a key minted here.
  [[nodiscard]] static Result<mcscf::RootRotation> run(const RootRotationRoutine::In &in);
};

} // namespace free_breakfast::routine
