/// @file
/// @brief The root-rotation term of an equal-weight state-averaged DF-CASSCF
/// Hessian (casscf_hessian.tex eqs. theta and rot): the orbital-relaxed
/// couplings between the averaged roots, the roots' rotations among
/// themselves, and the term they add to each root's Hessian, refused near a
/// degeneracy. See README.md for the layouts.

export module free_breakfast.mcscf.root_rotation;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.mcscf.derivative;

export namespace free_breakfast::mcscf {

/// @brief The rotations among the roots of an equal-weight state average,
/// along every nuclear coordinate x = 3 A + k (x last), on the host.
/// README.md has each layout and sign.
struct RootRotation {
  /// @brief H^x_rs = <c_r|H~^x + H'(kappa^x)|c_s>, the orbital-relaxed
  /// coupling (the derivative-coupling numerator), symmetric, its diagonal
  /// zero: element (r, s, x) at r + nroots (s + nroots x).
  std::vector<double> coupling;
  /// @brief theta^x_sr = H^x_rs / (E_r - E_s), the roots' first-order
  /// rotation (dc_r gains sum_s theta^x_sr c_s), antisymmetric: element
  /// (s, r, x) at s + nroots (r + nroots x).
  std::vector<double> theta;
  /// @brief Each root's term of eq. rot,
  /// 2 sum_{s != I} H^x_Is H^y_Is / (E_I - E_s): n x n per root, element
  /// (x, y, I) at x + n (y + n I), n = 3 n_atom.
  std::vector<double> hessian;
};

/// @brief The root-rotation term of an equal-weight state-averaged DF-CASSCF
/// Hessian, eq. rot, and the rotations theta^x of eq. theta it comes from.
///
/// At equal weights the rotations among the averaged roots are not SA
/// parameters, so lambda^x does not carry them, and each root's Hessian gains
/// 2 sum_{s != I} H^x_Is H^y_Is / (E_I - E_s), the factor 2 being the two
/// orders of x and y (one copy misses by half the term). H^x_rs is
/// @p derivatives' hamiltonian (H~^x at fixed parameters) plus
/// <c_r|H'(kappa^x)|c_s>, the active-space Hamiltonian's first derivative
/// along the orbital part of lambda^x: the packed gradient of the roots'
/// symmetrized transition RDMs (from c_r +- c_s) against @p kappa. The CI
/// part of lambda^x does not enter, being off every root.
///
/// The term diverges as E_s -> E_I (at an intersection E_I has no Hessian),
/// so a pair of roots closer than 1e-8 hartree, or one whose largest
/// |H^x_Is| / |E_I - E_s| over x passes @p max_rotation (bohr^-1), is
/// refused. Past it the quadratic model of E_I holds only within about
/// 1 / (2 max_rotation) bohr of the geometry.
///
/// @p molecule, @p basis, @p h, @p b (density_fit's, naux nao^2), @p c,
/// @p ncore, @p ncas, @p nalpha, @p nbeta and @p roots are as
/// @c df_derivatives takes them, and @p derivatives is its result at those
/// orbitals and roots (its hamiltonian is the same at any weights).
/// @p weights are the average's (all equal), @p energies each root's energy.
/// @p kappa is lambda^x's orbital part on the device, size x n column-major
/// over the packed pairs of @c response::pack_orbital_pairs (x_pq = K_pq at
/// p > q, C -> C e^K): @c zvectors' orbital block. One root gives zeros.
///
/// Queued on res.stream() with res.blas() (host pointer mode); it
/// synchronizes. The device res.dev_idx() names is made current for the call
/// and the previous one restored.
///
/// @return The rotations and the term; InvalidConfig if @p c or @p h is not
///         nao x nao, ncore is negative, ncas below 1 or ncore + ncas past
///         nao, @p b is not a whole number of nao^2 slabs, @p weights is empty
///         or not all equal, @p energies is not one per weight,
///         @p derivatives is not nroots^2 x 3 n_atom, or @p max_rotation is
///         not positive; Numerical, naming the two roots and their gap, near a
///         degeneracy as above; or the first error of the strings,
///         @c ci::Rdm, @c casscf_integrals, @c response::fock, a BLAS call or
///         a copy.
[[nodiscard]] Result<RootRotation>
root_rotation(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              const SphericalMatrixAO &h, const Buffers::device<double> &b,
              const SphericalMatrixAO &c, int32_t ncore, int32_t ncas, int32_t nalpha,
              int32_t nbeta, const double *roots, std::span<const double> weights,
              std::span<const double> energies, const Derivatives &derivatives, const double *kappa,
              double max_rotation = 10.0);

} // namespace free_breakfast::mcscf
