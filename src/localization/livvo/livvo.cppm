/// @file
/// @brief Valence virtual orbitals on the GPU (Derricotte & Evangelista,
/// J. Chem. Theory Comput. 13, 5984 (2017)): the part of the intrinsic atomic
/// orbitals' span orthogonal to the occupied orbitals.
///
/// Three gemms, one m x m syevd and one more gemm over the IAOs, which are an
/// input: the caller builds them (intrinsic_atomic_orbitals, or ibo_localize's
/// iao out-parameter) once for both. livvo_localize localizes them over the
/// IAO populations (the paper's LIVVOs), and valence_character measures how
/// much of an orbital lies in their span. The README has the formulas.

export module free_breakfast.localization.livvo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
export import free_breakfast.localization.pipek_mezey;

export namespace free_breakfast::localization {

/// @brief The n x (m - @p n_occ) valence virtual orbitals: an S-orthonormal
/// basis of the IAOs' span orthogonal to the @p n_occ occupied orbitals.
///
/// @p s is the n x n overlap of @p basis, @p c its n x n_occ occupied
/// orbitals, S-orthonormal, and @p iao the n x m orthonormal IAOs of those
/// orbitals, as intrinsic_atomic_orbitals returns them, n = basis.nao_sphe()
/// and m = minimal.nao_sphe(); every matrix column-major and packed. @p basis
/// and @p minimal give only the sizes: nothing is integrated here. That @p iao
/// spans @p c is the caller's to keep. @p vvo receives V, with V^T S V = I and
/// C^T S V = 0; it is fixed up to a rotation among its own columns.
///
/// Queued on res.stream(), which diagonalize synchronizes once and which is
/// synchronized again before returning. Scratch is a pool buffer on
/// res.device() and res.device_workspace(); no argument may point into the
/// latter.
///
/// @return Success, or InvalidConfig for @p minimal over a different atom
///         count than @p basis, @p n_occ outside 0..m, or a null pointer;
///         Numerical if syevd does not converge; GpuRuntime for a failed
///         BLAS, solver or copy call.
[[nodiscard]] Status valence_virtual_orbitals(DeviceResources &res, const BasisSet &basis,
                                              const BasisSet &minimal, int32_t n_occ,
                                              const double *s, const double *c, const double *iao,
                                              double *vvo);

/// @brief Localize the m - @p n_occ valence virtual orbitals into LIVVOs:
/// Pipek-Mezey over the populations of the intrinsic atomic orbitals.
///
/// @p s, @p iao and @p vvo are valence_virtual_orbitals' (@p vvo its output,
/// n x (m - n_occ)), over @p basis and @p minimal as there; IAO r belongs to
/// the atom that owns @p minimal's AO r. @p livvo receives V U, the n x
/// (m - n_occ) LIVVOs, and @p u (if not null) the (m - n_occ) x (m - n_occ)
/// orthogonal U; @p livvo must not alias @p vvo. The default exponent is
/// Knizia's four, as for ibo_localize.
///
/// Queued on res.stream() and synchronized before returning; the sweep
/// synchronizes once per sweep. Scratch is pool buffers on res.device() and
/// res.device_workspace(); no argument may point into the latter.
///
/// @return The sweeps and L over the IAO populations, or InvalidConfig for
///         @p minimal over a different atom count than @p basis, @p n_occ
///         outside 0..m, a null pointer or @p livvo aliasing @p vvo; else
///         pm_localize's errors. On MaxIterationsExceeded @p livvo
///         and @p u hold the last sweep's orbitals.
[[nodiscard]] Result<Info>
livvo_localize(DeviceResources &res, const BasisSet &basis, const BasisSet &minimal, int32_t n_occ,
               const double *s, const double *iao, const double *vvo, double *livvo,
               double *u = nullptr, const PmOptions &options = {.exponent = PmExponent::Four});

/// @brief The valence character t_val of each of @p n_orb orbitals: its
/// squared overlap summed over the @p n_vvo LIVVOs (the paper's eqs 11-12).
///
/// @p s is the n x n overlap of @p basis, @p c the n x n_orb orbitals (the
/// canonical virtuals, say) and @p livvo the n x n_vvo LIVVOs, or the VVOs:
/// t_val depends only on their span. Every matrix column-major and packed. For
/// S-orthonormal @p c each t_val is in [0, 1], and over a complete set of
/// virtuals they sum to @p n_vvo.
///
/// Queued on res.stream() and synchronized once, to download L^T S C. Scratch
/// is res.device_workspace(); no argument may point into it.
///
/// @return One t_val per orbital, or InvalidConfig for a negative @p n_orb,
///         @p n_vvo outside 0..n, or a null pointer; GpuRuntime for a failed
///         BLAS call or copy.
[[nodiscard]] Result<std::vector<double>> valence_character(DeviceResources &res,
                                                            const BasisSet &basis, int32_t n_orb,
                                                            int32_t n_vvo, const double *s,
                                                            const double *c, const double *livvo);

} // namespace free_breakfast::localization
