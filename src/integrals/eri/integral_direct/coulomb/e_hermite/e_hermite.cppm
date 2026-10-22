/// @file
/// @brief The J build's [p|q] contraction, over the e_hermite<lp, lq> kernels
/// FelizJueves generates: one ket class against every bra class, screened by
/// the record bound.

export module free_breakfast.integrals.coulomb.e_hermite;

import std;
import free_breakfast.core.wwr;
import free_breakfast.integrals.coulomb.pair_class;

export namespace free_breakfast {

/// @brief Contract ket class @p ket against every class of @p bras, on the
/// ket's own stream: J_p += sum_q [p|q] D_q into each bra's jp, one launch per
/// bra class, one warp per bra record, added atomically.
///
/// Reads @p ket as the last build sorted it (its ket_ fields: zeta, P, Iq,
/// Dmax and D_q, descending by Iq * Dmax), and each bra's records in record
/// order, its Ip the bra's own bound. A lane stops at its first ket with
/// `Dmax * Ip * Iq < tau`; @p tau = 0 skips nothing. Each bra's jp must be
/// zeroed before the first ket class and holds the sum over every one once
/// they all have run; any number of ket classes may add at once, from their
/// own streams. A step of @ref coulomb, which checks its inputs. Queued only;
/// nothing is synchronized.
///
/// @return The first launch status that failed (wwrErrorInvalidValue for a
///         class pair with no kernel). Launch errors only.
[[nodiscard]] wwr::wwrError_t e_hermite(const CoulombPairClass &ket,
                                        std::span<CoulombPairClass> bras, double tau);

} // namespace free_breakfast
