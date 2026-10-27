/// @file
/// @brief Atomic populations and partial charges over an orthonormal,
/// atom-owned minimal basis, such as the intrinsic atomic orbitals (Knizia,
/// J. Chem. Theory Comput. 9, 4834 (2013), eq 3).
///
/// An orbital's population on atom A is its squared weight on A's functions
/// of the minimal basis; a partial charge is Z_A less the occupied orbitals'
/// populations on A. Neither cares how the minimal basis was built: the IAOs
/// are what localization/ibo makes, and the caller hands them in. Nor which
/// rotation of the occupied orbitals is given: the charges of the canonical
/// orbitals and of the IBOs are the same, and the IBOs' populations are their
/// bonding picture. The README has the formulas.

export module free_breakfast.analysis.population;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.basis;

export namespace free_breakfast {

/// @brief The weight of each of @p n_orb orbitals on each function of the
/// S-orthonormal minimal basis @p iao: (A^T S C)_ri^2.
///
/// @p minimal says which atom owns which column of @p iao: column r belongs
/// to the atom that owns @p minimal's spherical AO r, as for
/// localization::intrinsic_atomic_orbitals. @p basis is the one @p s, @p c and @p iao are
/// over, n = basis.nao_sphe(): @p s the n x n overlap, @p c the n x n_orb
/// orbitals and @p iao the n x m minimal basis, m = minimal.nao_sphe(), all
/// column-major and packed. @p iao is expected S-orthonormal (A^T S A = I)
/// and to span @p c, which is the caller's to keep; then each orbital's
/// weights sum to its norm, one for S-orthonormal orbitals.
///
/// Queued on res.stream() and synchronized once, to download A^T S C. Scratch
/// is pool buffers on res.device().
///
/// @return The m x n_orb weights, column-major on the host, orbital i's on
///         function r at [r + m * i]. Or InvalidConfig for a basis with no
///         atoms, @p minimal over a different atom count, a negative
///         @p n_orb or a null pointer; GpuRuntime for a failed BLAS call or
///         copy.
[[nodiscard]] Result<std::vector<double>>
orbital_weights(DeviceResources &res, const BasisSet &basis, const BasisSet &minimal, int32_t n_orb,
                const double *s, const double *c, const double *iao);

/// @brief The populations of @p n_orb orbitals on each atom: their
/// orbital_weights summed over each atom's functions of @p minimal.
///
/// @return The natoms x n_orb populations, column-major on the host, orbital
///         i's on atom A at [A + natoms * i]; or InvalidConfig when
///         @p weights is not m x @p n_orb, or @p minimal has no atoms.
[[nodiscard]] Result<std::vector<double>>
orbital_populations(const BasisSet &minimal, int32_t n_orb, std::span<const double> weights);

/// @brief orbital_weights, then the populations from them: the arguments,
/// result and errors are those two's.
[[nodiscard]] Result<std::vector<double>>
orbital_populations(DeviceResources &res, const BasisSet &basis, const BasisSet &minimal,
                    int32_t n_orb, const double *s, const double *c, const double *iao);

/// @brief The partial charges q_A = Z_A - @p occupation * sum_i
/// populations[A + natoms * i], over the natoms x n_orb @p populations of
/// orbital_populations.
///
/// With the occupied orbitals of a closed shell at @p occupation two, this is
/// eq 3 of Knizia's paper over the IAOs; the charges sum to the molecule's.
///
/// @return One charge per atom of @p atoms, or InvalidConfig when
///         @p populations is not a whole number of columns of natoms.
[[nodiscard]] Result<std::vector<double>> partial_charges(std::span<const Atom> atoms,
                                                          std::span<const double> populations,
                                                          double occupation = 2.0);

} // namespace free_breakfast
