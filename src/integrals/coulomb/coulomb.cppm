/// @file
/// @brief The Cartesian Coulomb matrix J on the GPU, by McMurchie-Davidson,
/// over the dq, e_hermite and jab kernels FelizJueves generates, screened by
/// the record bound mmd forms (PrimitivePairClass::bound).
///
/// The records come from free_breakfast.integrals.mmd: every class is both a
/// ket class (its density pre-contracted, its records sorted by the screen's
/// bound) and a bra class (its J_p contracted against every ket class, then
/// scattered into J).

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

} // namespace free_breakfast
