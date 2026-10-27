/// @file
/// @brief Pipek-Mezey orbital localization on the GPU: the occupied orbitals
/// rotated, within their own span, to maximize the sum of powers of their
/// Mulliken atomic populations.
///
/// The canonical orbitals an SCF produces are eigenvectors of the Fock matrix
/// and spread over the molecule. Pipek-Mezey trades that for orbitals sitting
/// on as few atoms as possible -- cores, lone pairs, bonds -- while the
/// density, the energy and every observable stay the same. It needs only S and
/// C, which the SCF already has. The README has the algorithm;
/// pipek_mezey_kernels.cu has the arithmetic.

export module free_breakfast.localization.pipek_mezey;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;

export namespace free_breakfast::localization {

/// @brief The power p of L_p = sum_i sum_A (Q^A_ii)^p.
///
/// Two is Pipek and Mezey's functional, with an exact rotation angle per pair.
/// Its rotation Hessian can have an exact zero mode for a symmetric pi system
/// (benzene's six equivalent pi orbitals), leaving a continuum of equally
/// localized answers. Four is Knizia's IBO exponent, which concentrates the
/// populations harder and breaks that degeneracy; its angle is approximate,
/// but a rotation is accepted only where it raises L_4 exactly.
enum class PmExponent : int32_t {
  Two = 2,  ///< L_2, Pipek and Mezey (1989)
  Four = 4, ///< L_4, Knizia (2013)
};

/// @brief The knobs of one pm_localize call.
struct PmOptions {
  /// @brief Sweeps before giving up; each visits every orbital pair once. At
  /// least 1.
  int32_t max_sweeps = 64;
  /// @brief Converged when a whole sweep moves L by less than this.
  double functional_tolerance = 1e-10;
  /// @brief A pair whose gain does not exceed this is left alone, rather than
  /// spending a rotation, and its rounding, on nothing.
  double rotation_gain_threshold = 1e-14;
  /// @brief The functional's power.
  PmExponent exponent = PmExponent::Two;
};

/// @brief What a converged pm_localize did.
struct Info {
  /// @brief Sweeps taken; 0 for fewer than two orbitals, which have no pair.
  int32_t sweeps = 0;
  /// @brief L of the orbitals as given.
  double initial_functional = 0.0;
  /// @brief L of the localized orbitals; never below initial_functional.
  double functional = 0.0;
};

/// @brief Localize @p n_occ orbitals by Pipek-Mezey, over the spherical AOs of
/// @p basis.
///
/// @p s is the n_ao x n_ao overlap and @p c the n_ao x n_occ orbitals, where
/// n_ao is basis.nao_sphe(); both column-major and packed, so the first n_occ
/// columns of a square C serve as @p c. @p c_local receives C U, the localized
/// orbitals, and @p u (if not null) the n_occ x n_occ orthogonal U. @p c_local
/// must not alias @p c, since C U is a gemm.
///
/// @p c is expected to be S-orthonormal, C^T S C = I, which is what makes an
/// orbital's populations sum to one over the atoms. That is the caller's to
/// keep; it is not checked.
///
/// Each atom owns its shells' AOs, which @p basis knows when it was placed
/// over a molecule. A basis built from a bare shell list has no atoms, and is
/// rejected.
///
/// Queued on res.stream(), which is synchronized once per sweep to read L
/// back. Every scratch buffer is carved from res.device_workspace(), so no
/// argument may point into it.
///
/// @return The sweeps and the functional, or an error: InvalidConfig for a
///         basis with no atoms, a bad pointer, @p n_occ outside 0..n_ao (or
///         past 65535, the grid's limit) or max_sweeps below 1;
///         MaxIterationsExceeded when max_sweeps pass without converging, with
///         @p c_local and @p u holding the last sweep's orbitals; GpuRuntime
///         for a failed launch, copy or BLAS call.
[[nodiscard]] Result<Info> pm_localize(DeviceResources &res, const BasisSet &basis, int32_t n_occ,
                                       const double *s, const double *c, double *c_local, double *u,
                                       const PmOptions &options = {});

} // namespace free_breakfast::localization
