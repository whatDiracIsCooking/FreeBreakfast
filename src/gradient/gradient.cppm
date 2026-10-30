/// @file
/// @brief Analytic nuclear gradients on the GPU: the closed-shell DF-RHF one,
/// term by term,
///
///     dE/dR = Sum D dT + Sum D dV - Sum W dS + dE_J + dE_K + dV_nn,
///
/// over the derivative integrals of integrals.one_electron.gradient and
/// integrals.eri.density_fitting.gradient. The README has each term's
/// derivation and the memory it holds.

export module free_breakfast.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.integral_direct.exchange.pairs;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast::gradient {

/// @brief An RHF gradient and the terms it sums, each flat and row-major as
/// @c (n_atom, 3), as @c nuclear_repulsion_gradient: element 3 A + k is
/// dE/dR_{A,k} (hartree / bohr), atoms in the molecule's order.
struct RhfTerms {
  /// @brief Sum_mn D_mn dT_mn/dR.
  std::vector<double> kinetic;
  /// @brief Sum_mn D_mn dV_mn/dR, the operator's (Hellmann-Feynman) part
  /// included.
  std::vector<double> potential;
  /// @brief -Sum_mn W_mn dS_mn/dR, W = 2 C_occ diag(eps_occ) C_occ^T.
  std::vector<double> overlap;
  /// @brief dE_J/dR, E_J = 1/2 Sum D_mn (mn|ls) D_ls.
  std::vector<double> coulomb;
  /// @brief dE_x/dR, E_x = -1/4 Sum D_ml (mn|ls) D_ns: RHF's exchange
  /// energy.
  std::vector<double> exchange;
  /// @brief dV_nn/dR.
  std::vector<double> nuclear_repulsion;
  /// @brief The sum of the six.
  std::vector<double> total;
};

/// @brief The closed-shell DF-RHF nuclear gradient at converged orbitals,
/// every term of it, the two-electron ones over the fit B = L^-1 (mn|P).
///
/// The one-electron terms take D = 2 C_occ C_occ^T and W = 2 C_occ
/// diag(eps_occ) C_occ^T. The Coulomb term takes the fitted density
/// d = L^-T (B vec D) = M^-1 (P|mn) D_mn, as Gam = D (x) d against d(mn|P) and
/// -1/2 d d^T against d(P|Q). The exchange term takes the fitted occupied
/// pairs c_P,ij = (L^-T B_ij)_P (@c b_mo over the occupied columns, then a
/// trsm), as -2 Sum_ij C_mi C_nj c_P,ij against d(mn|P) and Sum_ij c_P,ij
/// c_Q,ij against d(P|Q). Every contraction is a GPU gemm; only the n_atom x 3
/// gradients come back to the host.
///
/// @p molecule is the molecule both bases were placed on (by @c place), and
/// @p classes are @c primitive_pair_classes of @p basis. @p b is B as
/// @c density_fit leaves it, naux x nao^2 with element (Q, n + nao m) at
/// Q + naux (n + nao m), and @p l is its L, the naux x naux lower triangle
/// (column-major, packed; the strict upper triangle is not read), naux =
/// aux.nao_sphe(). @p c is the RHF orbitals, nao x nao over @p basis, with
/// @p mo_energies one per column and the first @p nocc columns occupied. The
/// formula is the gradient only at a converged SCF: W stands in for the
/// orbitals' response.
///
/// **Gam is held in full**, naux nao^2 doubles beside B (and the derivative
/// call's Cartesian copy of it); with the fitted pairs and their half
/// back-transform, the call's own device memory is about
/// 8 naux (nao^2 + nocc^2 + nocc nao + naux) bytes. Past the device's free
/// memory it is Unsupported, naming the bytes, before anything is allocated.
///
/// Queued on res.stream(); each derivative call synchronizes it to return its
/// rows, so the call returns with everything done. Scratch for @c b_mo and the
/// derivative calls comes from res.device_workspace(), so none of @p b, @p l
/// and @p c may point into it. The device res.dev_idx() names is made current
/// for the call and the previous one restored.
///
/// @return The gradient and its terms; InvalidConfig if @p c is not
///         basis.nao_sphe() square or not on @p res's device, if
///         @p mo_energies does not hold one energy per column, if @p nocc is
///         outside [0, nao], or for any input the derivative calls refuse;
///         Unsupported for memory past the device's free bytes or naux nao^2
///         past int32_t; or the GpuRuntime error of the first copy, BLAS call
///         or launch that failed.
[[nodiscard]] Result<RhfTerms> df_rhf(DeviceResources &res, const Molecule &molecule,
                                      const BasisSet &basis,
                                      std::span<const PrimitivePairClass> classes,
                                      const BasisSet &aux, const double *b, const double *l,
                                      const SphericalMatrixAO &c,
                                      std::span<const double> mo_energies, int32_t nocc);

/// @brief The closed-shell integral-direct RHF nuclear gradient at converged
/// orbitals, every term of it, the two-electron ones over the integrals
/// themselves.
///
/// The one-electron, overlap and nuclear-repulsion terms are @ref df_rhf's.
/// The Coulomb term is @c coulomb_gradient on D = 2 C_occ C_occ^T, unscreened;
/// the exchange term is -1/2 @c exchange_gradient on D, unscreened too, so
/// the call is O(N^4) in the primitive pairs.
///
/// @p molecule is the molecule @p basis was placed on (by @c place), with its
/// per-atom shell map; @p classes are @c primitive_pair_classes of @p basis,
/// and @p k_classes @c ordered_pair_classes of it, the integral-direct ERI
/// step's K classes (only read, but @c exchange_gradient takes them
/// mutable). @p c, @p mo_energies and @p nocc are as @ref df_rhf takes them,
/// and the formula is likewise the gradient only at a converged SCF.
///
/// Queued on res.stream(); each derivative call synchronizes it to return its
/// rows, so the call returns with everything done. The device res.dev_idx()
/// names is made current for the call and the previous one restored.
///
/// @return The gradient and its terms; InvalidConfig if @p c is not
///         basis.nao_sphe() square or not on @p res's device, if
///         @p mo_energies does not hold one energy per column, if @p nocc is
///         outside [0, nao], or for any input the derivative calls refuse; or
///         the GpuRuntime error of the first copy, BLAS call or launch that
///         failed.
[[nodiscard]] Result<RhfTerms>
rhf(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, std::span<const double> mo_energies, int32_t nocc);

} // namespace free_breakfast::gradient

namespace free_breakfast::gradient {

// The terms every closed-shell RHF gradient shares, whatever its ERIs:
// kinetic, potential, overlap and nuclear_repulsion of @p g, from D and W of
// @p c's first @p nocc columns, through integrals.one_electron.gradient and
// nuclear_repulsion_gradient. Checks @p c, @p mo_energies and @p nocc as
// df_rhf documents, and returns D, which the two-electron terms take too.
// Unexported: the module's own entry points (df_rhf, and the integral-direct
// one beside it) are its callers.
[[nodiscard]] Result<SphericalMatrixAO>
one_electron_terms(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
                   std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &c,
                   std::span<const double> mo_energies, int32_t nocc, RhfTerms &g);

// a + b, element by element: defined in df_rhf.cpp, and rhf's too.
std::vector<double> sum_of(std::vector<double> a, const std::vector<double> &b);

} // namespace free_breakfast::gradient
