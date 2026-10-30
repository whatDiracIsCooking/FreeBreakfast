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
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
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
/// one call at a time per set of classes; it allocates nothing. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success; InvalidConfig if @p tau is negative or NaN, if @p res is
///         not on @p j's, @p density's or @p classes' device, if @p density and
///         @p j differ in order, if a class is above f (or not la <= lb), or if
///         j.n() is too large for the kernels' 32-bit index (j.n()^2 above
///         INT32_MAX); or the GpuRuntime error of the first memset, fork,
///         launch or join that failed, which leaves @p j unspecified.
[[nodiscard]] Status coulomb(DeviceResources &res, std::span<CoulombPairClass> classes,
                             const CartesianMatrixAO &density, CartesianMatrixAO &j,
                             double tau = 1e-10);

/// @brief G[A, k] = dE_J/dR_{A,k}, the nuclear gradient of the Coulomb energy
/// E_J = 1/2 sum_abcd D_ab D_cd (ab|cd) of @p density, for every atom A of
/// @p basis and axis k.
///
/// FelizJueves's J gradient chain (its docs/implementation/gradient.md, "The
/// RHF gradient, assembled"): the ket derivative of E_J equals its bra
/// derivative, so dE_J/dR_{A,k} = sum_ab D_ab sum_t (dA_k E^ab)_t J_t, J_t the
/// bra record's J_p one Hermite order up, the half of E_J and the bra-plus-ket
/// two cancelling. Every class as a ket, on its own stream, as @ref coulomb
/// runs it (dq, the ket sort), then contracted against every bra class raised
/// by one (e_hermite at lp = la + lb + 1) into that bra's raised J_p; then
/// every class as a bra, gjab reading its J_p where it lies, into one zeroed
/// gradient. Each atom's derivative takes both of its pair's centres,
/// differentiated, so the per-axis sum over atoms vanishes to rounding rather
/// than by construction. The signs and halves of an SCF energy are the
/// caller's: RHF adds this to its gradient as is.
///
/// @p density is a spherical AO matrix of @p basis, symmetric (not checked),
/// carried onto the Cartesian layout internally (U D U^T, as the Fock build
/// carries it). @p classes must be @c primitive_pair_classes of @p basis, which
/// must carry its per-atom shell map (made by @c place): the gradient's rows
/// are its atoms. J's classes are made from them over every record
/// (@ref coulomb_pair_classes at 0): the prune bounds an energy term, not its
/// derivative.
///
/// @p tau is the raised [p|q]'s screen, e_hermite's `Dmax * Ip * Iq < tau`
/// with the energy's bounds: they bound the integral, not its derivative, so
/// the default, 0, screens nothing (README.md, "The gradient's screen", has
/// what a nonzero one costs).
///
/// Queued on res.stream(), each phase's classes forked from it and joined
/// back; the call waits for the stream, to return the gradient on the host. Its
/// scratch (the raised J_p, the records' atoms and the device gradient) is the
/// workspace's. The device res.dev_idx() names is made current for the call
/// and the previous one restored.
///
/// @return The gradient, flat and row-major as @c (basis.natoms(), 3); or
///         InvalidConfig if @p density is not basis.nao_sphe() square or not
///         on @p res's device, if @p basis has no per-atom map, if @p tau is
///         negative or NaN, if a class is above f (or not la <= lb), or for any
///         input @c check_one_electron_build refuses; or the GpuRuntime error
///         of the first copy, launch, join or wait that failed.
[[nodiscard]] Result<std::vector<double>>
coulomb_gradient(DeviceResources &res, const BasisSet &basis,
                 std::span<const PrimitivePairClass> classes, const SphericalMatrixAO &density,
                 double tau = 0.0);

} // namespace free_breakfast
