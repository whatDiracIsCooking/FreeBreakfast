/// @file
/// @brief Valence virtual orbitals on the GPU (Derricotte & Evangelista,
/// J. Chem. Theory Comput. 13, 5984 (2017)): the part of the intrinsic atomic
/// orbitals' span orthogonal to the occupied orbitals.
///
/// Three gemms, one m x m syevd and one more gemm over the IAOs, which are an
/// input: the caller builds them (intrinsic_atomic_orbitals, or ibo_localize's
/// iao out-parameter) once for both. The README has the formulas.

export module free_breakfast.localization.livvo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;

export namespace free_breakfast {

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
/// Queued on res.stream(), which diagonalize synchronizes once. Scratch is a
/// pool buffer on res.device() and res.device_workspace(); no argument may
/// point into the latter.
///
/// @return Success, or InvalidConfig for @p minimal over a different atom
///         count than @p basis, @p n_occ outside 0..m, or a null pointer;
///         Numerical if syevd does not converge; GpuRuntime for a failed
///         BLAS, solver or copy call.
[[nodiscard]] Status valence_virtual_orbitals(DeviceResources &res, const BasisSet &basis,
                                              const BasisSet &minimal, int32_t n_occ,
                                              const double *s, const double *c, const double *iao,
                                              double *vvo);

} // namespace free_breakfast
