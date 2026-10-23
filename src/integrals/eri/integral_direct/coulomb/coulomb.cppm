/// @file
/// @brief The Cartesian Coulomb matrix J on the GPU, by McMurchie-Davidson,
/// over the dq, e_hermite and jab kernels FelizJueves generates, screened by
/// the record bound mmd forms (PrimitivePairClass::bound).
///
/// The records are J's own, CoulombPairClass (coulomb.pair_class, re-exported
/// here). Every class is both a ket class (its density pre-contracted, dq, its
/// records sorted by the screen's bound, then contracted into every bra
/// class's J_p, e_hermite) and a bra class (its J_p scattered into J,
/// j_coulomb).

export module free_breakfast.integrals.eri.integral_direct.coulomb;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
export import free_breakfast.integrals.eri.integral_direct.coulomb.pair_class;
import free_breakfast.recipes.matrix_ao;

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
/// Ket-major, in two phases. Zeroes @p j and every class's J_p; then, per
/// class on that class's own stream, pre-contracts the density into the
/// class's Hermite basis (dq), sorts the class as a ket descending by
/// Iq * Dmax (the screen breaks at a lane's first ket under @p tau, so that is
/// the order it needs), and contracts every bra class against it (e_hermite,
/// added atomically into the bra's J_p, which every ket class's stream adds
/// to). Once every class's stream is joined, each class's J_p is complete and
/// it scatters it into @p j on its own stream (jab, added at (m, n) and
/// (n, m)). A class's sorted ket copies are read only on its own stream; the
/// bra side reads the class's records in record order, its Ip the class's own
/// bound, so jab's rows stay the identity. @p density must be symmetric; it is
/// not checked. The classes must come from @ref coulomb_pair_classes over the
/// basis @p density and @p j were made for (at tau = 0 for every record): a
/// record's AO offsets are not checked against j.n().
///
/// Queued on res.stream(), each phase's classes forked from it and joined back
/// (@c launch_each), with no host synchronization, and every class's stream
/// forked from it once more at the end, succeeded or not, so nothing queued
/// there later (a class's own frees) runs ahead of another class's stream
/// still reading it. Writes the classes' scratch (@ref CoulombPairClass), so
/// one call at a time per set of classes; it allocates only a max(n_rec) row
/// map, from res.device()'s pool in res.stream()'s order. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p tau is negative or NaN, if @p res is
///         not on @p j's, @p density's or @p classes' device, if @p density and
///         @p j differ in order, if a class is above d (or not la <= lb), or if
///         j.n() is too large for the kernels' 32-bit index (j.n()^2 above
///         INT32_MAX); or the GpuRuntime error of the first memset, fork,
///         launch or join that failed, which leaves @p j unspecified.
[[nodiscard]] Status coulomb(DeviceResources &res, std::span<CoulombPairClass> classes,
                             const CartesianMatrixAO &density, CartesianMatrixAO &j,
                             double tau = 1e-10);

} // namespace free_breakfast
