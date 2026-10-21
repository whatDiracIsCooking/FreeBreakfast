/// @file
/// @brief The Cartesian exchange matrix K on the GPU, by McMurchie-Davidson,
/// over the k kernels FelizJueves generates, screened by the record bound mmd
/// forms (PrimitivePairClass::bound).
///
/// The records are free_breakfast.integrals.exchange.pairs's ordered classes,
/// re-exported here, so one import gives both the classes and the build: every
/// ordered class is both a bra class (a, c) and a ket class (b, d).

export module free_breakfast.integrals.exchange;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.linalg.matrix_ao;
export import free_breakfast.integrals.exchange.pairs;

export namespace free_breakfast {

/// @brief The Cartesian exchange matrix K_ab = sum_cd D_cd (ac|bd) of the AO
/// density @p density into @p k, screened at @p tau.
///
/// @p classes are @c ordered_pair_classes(res, basis, primitive_pair_classes)
/// of the basis @p density and @p k were made for, built once per basis.
///
/// Screened by kernel_k's Schwarz bound: a primitive quartet is skipped when
/// `Ip * Iq * max|D| < tau`, Ip and Iq the bra's and the ket's record bounds
/// (mmd's bound field) and max|D| the whole density's (the density straddles
/// bra and ket, so no per-ket factor exists). @p tau = 0 skips nothing. The
/// default, 1e-10, is @c coulomb's.
///
/// Zeroes @p k and, on res.stream(), computes max|D|; then, per class on that
/// class's own stream, its momentum's rank-to-AO table (a class (l, l)) and,
/// after a join, the (la, lb) gather of the density into Dcd; then, after a
/// join, each class as a bra on its own stream: for every ket class and every
/// first-shell group b of it, one k<la, lc, lb, ld> launch of all of the bra's
/// records against the group's kets into the (bra class, b) block, then the
/// bra's blocks scattered into @p k. The ordered sweep computes K_ab and K_ba
/// each directly: nothing is mirrored, and every quartet is visited twice (the
/// bra-ket swap is not done). @p density must be symmetric; it is not checked.
///
/// Queued on res.stream(), each phase's classes forked from it and joined back
/// (@c launch_each), with no host synchronization; the scratch (per bra class
/// n_rec x n_x(la) x n_ao doubles of blocks, n_rec more doubles and (l_max + 1)
/// n_rec int32s; one Dcd per class; one rank table per momentum) is drawn from
/// res.device()'s pool and returned to it in res.stream()'s order. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p tau is negative or NaN, if @p res is
///         not on @p k's, @p density's or @p classes' device, if @p density and
///         @p k differ in order, if a class is above d, if a class's shell
///         reaches past @p k's order, or if k.n() is too large for the kernels'
///         32-bit index (k.n()^2 above INT32_MAX); or the GpuRuntime error of
///         the first memset, fork, launch or join that failed, which leaves
///         @p k unspecified.
[[nodiscard]] Status exchange(DeviceResources &res, std::span<const OrderedPairClass> classes,
                              const CartesianMatrixAO &density, CartesianMatrixAO &k,
                              double tau = 1e-10);

} // namespace free_breakfast
