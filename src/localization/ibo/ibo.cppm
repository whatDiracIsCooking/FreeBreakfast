/// @file
/// @brief Intrinsic atomic and bond orbitals on the GPU (Knizia, J. Chem.
/// Theory Comput. 9, 4834 (2013)): the IAOs, a minimal basis polarized to
/// span the occupied orbitals exactly, and the IBOs, the occupied orbitals
/// localized over the IAOs' atomic populations.
///
/// The IAOs are gemms, Cholesky solves and two Loewdin orthogonalizations
/// over the overlaps of the working basis and a free-atom minimal basis
/// (MINAO); the IBOs are pipek_mezey_localize run over the occupied orbitals
/// written in the orthonormal IAOs, where the Mulliken populations are the
/// IAO ones. The README has the formulas.

export module free_breakfast.localization.ibo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
export import free_breakfast.localization.pipek_mezey;

export namespace free_breakfast {

/// @brief The orthonormal intrinsic atomic orbitals of @p n_occ occupied
/// orbitals, over the spherical AOs of @p basis, one per spherical AO of
/// @p minimal.
///
/// @p minimal is the free-atom minimal basis (B2, MINAO in the paper) placed
/// over the same atoms as @p basis, in the same order; IAO r belongs to the
/// atom that owns @p minimal's AO r. Its functions are normalized here, so a
/// contraction that is not unit-normalized makes no difference.
///
/// @p s is the n x n overlap of @p basis and @p c its n x n_occ orbitals,
/// n = basis.nao_sphe(), column-major and packed; @p c is expected
/// S-orthonormal (C^T S C = I), which is the caller's to keep. @p iao receives
/// the n x m IAO coefficients, m = minimal.nao_sphe(), with A^T S A = I and
/// the occupied orbitals in their span: C = A A^T S C.
///
/// Queued on res.stream(), which is synchronized several times (each Cholesky
/// flag, each orthogonalization). Scratch is pool buffers on res.device(),
/// apart from the solvers' work, which is res.device_workspace(); no argument
/// may point into it.
///
/// @return Success, or InvalidConfig for a basis with no atoms, @p minimal
///         over a different atom count, m above n, @p n_occ outside 0..m, or
///         a null pointer; Numerical when either overlap is not positive
///         definite; GpuRuntime for a failed integral, launch, copy, BLAS or
///         solver call.
[[nodiscard]] Status intrinsic_atomic_orbitals(DeviceResources &res, const BasisSet &basis,
                                               const BasisSet &minimal, int32_t n_occ,
                                               const double *s, const double *c, double *iao);

/// @brief Localize @p n_occ orbitals into intrinsic bond orbitals: Pipek-Mezey
/// over the populations of the intrinsic atomic orbitals.
///
/// The arguments are pipek_mezey_localize's, plus @p minimal as for
/// intrinsic_atomic_orbitals. @p c_local receives C U and @p u (if not null)
/// the n_occ x n_occ orthogonal U; @p c_local must not alias @p c. @p iao
/// (if not null) receives the n x m IAOs the localization ran over, as
/// intrinsic_atomic_orbitals would, for a population analysis without
/// building them twice. The default exponent is Knizia's four.
///
/// @return The sweeps and L over the IAO populations, or
///         intrinsic_atomic_orbitals' and pipek_mezey_localize's errors. On
///         MaxIterationsExceeded @p c_local and @p u hold the last sweep's
///         orbitals.
[[nodiscard]] Result<PipekMezeyInfo>
ibo_localize(DeviceResources &res, const BasisSet &basis, const BasisSet &minimal, int32_t n_occ,
             const double *s, const double *c, double *c_local, double *u, double *iao = nullptr,
             const PipekMezeyOptions &options = {.exponent = PipekMezeyExponent::Four});

} // namespace free_breakfast
