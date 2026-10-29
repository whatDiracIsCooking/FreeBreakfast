/// @file
/// @brief J[D] - K[D]/2 and K[D] of a non-symmetric AO density D (a CIS
/// transition density), integral-direct: one coulomb of D's symmetric part
/// and two exchange sweeps, its symmetric part's and its antisymmetric part's.

export module free_breakfast.integrals.eri.integral_direct.nonsymmetric_jk;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

export namespace free_breakfast {

/// @brief The singlet and triplet AO responses of a non-symmetric Cartesian
/// density @p density, D: @p g = J[D] - K[D]/2 and @p k = K[D], with
/// J_mn = sum_ls (mn|ls) D_ls and K_ab = sum_cd D_cd (ac|bd).
///
/// D = S + A, S = (D + D^T)/2 and A = (D - D^T)/2, each one geam. J[A] = 0,
/// since (mn|ls) is symmetric in l and s, so J[D] = @c coulomb of S; and
/// K[D] = @c exchange of S as DensitySymmetry::Symmetric plus @c exchange of
/// A as DensitySymmetry::Antisymmetric, the same kb work as one sweep over
/// every block. A symmetric or antisymmetric D is a valid input too; its
/// other part's sweep is then of zero.
///
/// @p j_classes and @p k_classes are J's and K's classes of the basis
/// @p density was made for (@c coulomb_pair_classes, @c ordered_pair_classes,
/// once per basis; pipeline.eri's IntegralDirect holds both), under
/// @c coulomb's and @c exchange's contracts: J's classes pruned at a tau that
/// assumes |S_cd| <= 2, and K's the whole set in its order. Each call writes
/// their scratch, so one call at a time per set of classes. @p tau_j and
/// @p tau_k are the two builds' screens, at their defaults. J's 1e-10 is
/// coarse for a dense D: on H2O/cc-pVDZ with a random one, G misses libcint
/// by 9e-12 of max|G| there, and meets 1e-12 at tau_j = 3e-12 (README.md).
///
/// Its own scratch is one nao_cart^2 block of res.device_workspace(), which
/// neither @c coulomb nor @c exchange takes: it carries K[A], then K[D],
/// across the builds, since each build zeroes its own output and reads its
/// density from a matrix of its own type, and those are @p g and @p k.
///
/// Queued on res.stream(), with no host synchronization; the device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig, before anything is queued, if @p tau_j or
///         @p tau_k is negative or NaN, if @p density, @p g or @p k is not on
///         @p res's device or the three differ in order, or if @p g and @p k
///         are one matrix or either is @p density; else @c coulomb's or
///         @c exchange's own refusals, or the GpuRuntime error of the first
///         BLAS call, copy or build that failed, either of which leaves @p g
///         and @p k unspecified.
[[nodiscard]] Status nonsymmetric_jk(DeviceResources &res, std::span<CoulombPairClass> j_classes,
                                     std::span<OrderedPairClass> k_classes,
                                     const CartesianMatrixAO &density, CartesianMatrixAO &g,
                                     CartesianMatrixAO &k, double tau_j = 1e-10,
                                     double tau_k = 1e-13);

} // namespace free_breakfast
