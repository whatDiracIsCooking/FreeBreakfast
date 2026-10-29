/// @file
/// @brief Intrinsic atomic and bond orbitals on the GPU (Knizia, J. Chem.
/// Theory Comput. 9, 4834 (2013)): the IAOs, a minimal basis polarized to
/// span the occupied orbitals exactly, and the IBOs, the occupied orbitals
/// localized over the IAOs' atomic populations.
///
/// The IAOs are gemms, Cholesky solves (S^-1/2 for a near-singular basis
/// overlap) and two Loewdin orthogonalizations
/// over the overlaps of the working basis and a free-atom minimal basis
/// (MINAO); the IBOs are pm_localize run over the occupied orbitals
/// written in the orthonormal IAOs, where the Mulliken populations are the
/// IAO ones. The README has the formulas.

export module free_breakfast.localization.ibo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
export import free_breakfast.localization.pipek_mezey;

export namespace free_breakfast::localization {

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
/// S1^-1, @p s's inverse, is a Cholesky solve, or X X^T over X = S1^-1/2
/// (orthogonalize, floored) when @p s is too near singular for one: a pivot
/// at or below kOrthogonalizeEigenvalueFloor, or none positive.
///
/// Queued on res.stream(), which is synchronized several times (each Cholesky
/// flag, each orthogonalization). Scratch is pool buffers on res.device(),
/// apart from the solvers' work, which is res.device_workspace(); no argument
/// may point into it.
///
/// @return Success, or InvalidConfig for a basis with no atoms, @p minimal
///         over a different atom count, m above n, @p n_occ outside 0..m, or
///         a null pointer; Numerical when the minimal basis's overlap is not
///         positive definite, or orthogonalize's syevd does not converge;
///         GpuRuntime for a failed integral, launch, copy, BLAS or solver
///         call.
[[nodiscard]] Status intrinsic_atomic_orbitals(DeviceResources &res, const BasisSet &basis,
                                               const BasisSet &minimal, int32_t n_occ,
                                               const double *s, const double *c, double *iao);

/// @brief Localize @p n_occ orbitals into intrinsic bond orbitals: Pipek-Mezey
/// over the populations of the intrinsic atomic orbitals.
///
/// The arguments are pm_localize's, plus @p minimal as for
/// intrinsic_atomic_orbitals. @p c_local receives C U and @p u (if not null)
/// the n_occ x n_occ orthogonal U; @p c_local must not alias @p c. @p iao
/// (if not null) receives the n x m IAOs the localization ran over, as
/// intrinsic_atomic_orbitals would, for a population analysis without
/// building them twice. The default exponent is Knizia's four.
///
/// @return The sweeps and L over the IAO populations, or
///         intrinsic_atomic_orbitals' and pm_localize's errors. On
///         MaxIterationsExceeded @p c_local and @p u hold the last sweep's
///         orbitals.
[[nodiscard]] Result<Info> ibo_localize(DeviceResources &res, const BasisSet &basis,
                                        const BasisSet &minimal, int32_t n_occ, const double *s,
                                        const double *c, double *c_local, double *u,
                                        double *iao = nullptr,
                                        const PmOptions &options = {.exponent = PmExponent::Four});

/// @brief Localize @p n_orb orbitals in the span of IAOs the caller built:
/// ibo_localize's sweep, without building them.
///
/// For a subspace of the occupied orbitals the IAOs were built from, each
/// localized on its own over the same IAOs: an open shell's doubly and singly
/// occupied orbitals, whose IAOs are those of both together. @p iao is the
/// n x m output of intrinsic_atomic_orbitals, and that it spans @p c, the
/// n x n_orb orbitals, is the caller's to keep; @p s, @p c_local, @p u and
/// @p options are ibo_localize's, and @p c_local must not alias @p c.
///
/// @return The sweeps and L over the IAO populations, or InvalidConfig for
///         @p minimal over a different atom count than @p basis, @p n_orb
///         outside 0..m, a null pointer or @p c_local aliasing @p c; else
///         pm_localize's errors. On MaxIterationsExceeded @p c_local and
///         @p u hold the last sweep's orbitals.
[[nodiscard]] Result<Info> iao_localize(DeviceResources &res, const BasisSet &basis,
                                        const BasisSet &minimal, int32_t n_orb, const double *s,
                                        const double *iao, const double *c, double *c_local,
                                        double *u,
                                        const PmOptions &options = {.exponent = PmExponent::Four});

} // namespace free_breakfast::localization
