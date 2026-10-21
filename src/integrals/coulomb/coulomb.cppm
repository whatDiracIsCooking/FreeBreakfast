/// @file
/// @brief The Cartesian Coulomb matrix J on the GPU, by McMurchie-Davidson,
/// over the dq, e_hermite and jab kernels FelizJueves generates, screened by
/// the record bound mmd forms (PrimitivePairClass::bound).
///
/// The records come from free_breakfast.integrals.mmd: every class is both a
/// ket class (its density pre-contracted, its records sorted by the screen's
/// bound) and a bra class (its J_p contracted against every ket class, then
/// scattered into J), or from coulomb_pair_classes' copy of them with the
/// primitive pairs J cannot need pruned.

export module free_breakfast.integrals.coulomb;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.linalg.matrix_ao;

export namespace free_breakfast {

/// @brief The Cartesian Coulomb matrix J_mn = sum_ls (mn|ls) D_ls of the AO
/// density @p density into @p j, screened at @p tau.
///
/// Screened by e_hermite's Schwarz bound: a primitive quartet's [p|q] D_q is
/// skipped when `Dmax * Ip * Iq < tau`, Ip and Iq being the bra's and the
/// ket's record bounds (mmd's bound field) and Dmax the ket's max_q |D_q|. That bounds
/// each term of the bra's J_p, not of J: what jab then scatters is weighted by
/// the bra's E^ab_p. @p tau = 0 skips nothing. The default, 1e-10, is the J
/// threshold of EXESS's J engine (Palethorpe and Barca), whose [p|q] kernel and
/// screen these are.
///
/// Zeroes @p j, then, per class on that class's own stream, computes the
/// bounds, pre-contracts the density into the class's Hermite basis (dq) and
/// sorts the class's kets descending by Iq * Dmax (SortedKets: the screen
/// breaks at a lane's first ket under @p tau, so that is the order it needs);
/// then contracts each bra class against every ket class (e_hermite, J_p added
/// atomically) and scatters J_p into @p j (jab, added at (m, n) and (n, m)).
/// @p density must be symmetric; it is not checked. The classes must come from
/// the basis @p density and @p j were made for: a record's AO offsets are not
/// checked against j.n().
///
/// Queued on res.stream(), each phase's classes forked from it and joined back
/// (@c launch_each), with no host synchronization; the scratch (per class
/// n_rec x n_hermite(la + lb) doubles three times, and 13 n_rec more doubles
/// or uint32s) is drawn from res.device()'s pool and returned to it in
/// res.stream()'s order. The device res.dev_idx() names is made current for
/// the call and the previous one restored.
///
/// @return Success; InvalidConfig if @p tau is negative or NaN, if @p res is
///         not on @p j's, @p density's or @p classes' device, if @p density and
///         @p j differ in order, if a class is above d (or not la <= lb), or if
///         j.n() is too large for the kernels' 32-bit index (j.n()^2 above
///         INT32_MAX); or the GpuRuntime error of the first memset, fork,
///         launch or join that failed, which leaves @p j unspecified.
[[nodiscard]] Status coulomb(DeviceResources &res, std::span<const PrimitivePairClass> classes,
                             const CartesianMatrixAO &density, CartesianMatrixAO &j,
                             double tau = 1e-10);

/// @brief The records of @p classes (primitive_pair_classes of one basis)
/// that J can need at @p tau: a copy of each class with its negligible
/// primitive pairs pruned, for @ref coulomb to read in place of @p classes.
/// Once per basis; mmd's own records (which S, T and V read) are untouched.
///
/// J's term of a bra record (a, b) and a ket record (c, d) is
/// `sum_p s E^ab_p sum_q [p|q] D_q`, D_q = w sum_cd D_cd s E^cd_q (w = 2 for
/// two shells, 1 for a shell against itself). With |[p|q]| <= I_p I_q (Schwarz)
/// and **|D_cd| <= Dc = 2, the density this assumes and @ref coulomb does not
/// check**. It is an assumption, not a theorem: a closed-shell density has
/// |D_cd| <= sqrt(D_cc D_dd) and D_cc <= 2 (S^-1)_cc, which is 2 for a
/// normalized AO that overlaps nothing and grows with the overlap, without
/// bound toward linear dependence; the H10 chain's RHF density has max|D|
/// 0.33. Under it, every such term of an element
/// of J is at most `B_ab * Dc * K_cd`, where B is a record's dressed bound
/// (ordered_pair_bound over mmd's records: max_{i in a} sum_{k in b}
/// sum_p |s E^{ik}_p| I_p, which bounds every single (i, k) as a bra) and
/// K = w ncart(la) B (which bounds the sum over every (c, d) as a ket). A
/// record is pruned when it is negligible both ways: `B_r Dc max K < tau` and
/// `K_r Dc max B < tau`, the maxima over every record of @p classes. So each
/// term a pruned record would have given, as bra or as ket, is under @p tau.
///
/// What it saves: mmd's bound, J's Ip, is pair-factor free, so J's screen
/// never drops a far bra, and each of its warps runs every ket a near bra
/// would. On long chains that is most of J's time (README.md, "Pruning", has
/// the measurement).
///
/// The default, 1e-13, is three orders under @ref coulomb's screen: each
/// pruned term is under it, but an element sums many, and on the H10 chain
/// pruning at 1e-10 moves J by 6.6e-12 max|J|, past the 1e-12 libcint bar; at
/// 1e-13, by 7.9e-15. The record count barely depends on it (the pair factor
/// falls as exp(-ab/z |AB|^2)). @p tau = 0 keeps every record.
///
/// Each kept record is copied in its class's order, mmd's descending zeta, with
/// every field; an emptied class is left out, the others in @p classes' order.
/// Each class is on a new @c DeviceStream of res.device(), its copy to the
/// device queued there. The device res.dev_idx() names is made current for the
/// call and the previous one restored.
///
/// @return The pruned classes (none for no @p classes); InvalidConfig if
///         @p tau is negative or NaN, or a class is above d (it has no bound
///         kernel, and @ref coulomb refuses it); or the GpuRuntime error of the
///         first synchronization, launch or copy that failed.
[[nodiscard]] Result<std::vector<PrimitivePairClass>>
coulomb_pair_classes(DeviceResources &res, std::span<const PrimitivePairClass> classes,
                     double tau = 1e-13);

} // namespace free_breakfast
