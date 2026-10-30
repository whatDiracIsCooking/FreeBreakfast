/// @file
/// @brief The J build's bra back-transform, over the jab<la, lb> kernels
/// FelizJueves generates: one bra class's J_p scattered into J.

export module free_breakfast.integrals.eri.integral_direct.coulomb.j_coulomb;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;
import free_breakfast.integrals.eri.integral_direct.coulomb.pair_class;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief Add bra class @p bra's back-transform J_ab += sum_p s E^ab_p J_p into
/// @p j, on the class's own stream: one thread per record, each value added
/// atomically at (m, n) and (n, m).
///
/// The class's jp must be complete (every ket class contracted into it,
/// @ref e_hermite); its J_p rows are read through its row map,
/// CoulombPairClass::row. @p j must be zeroed before the first class and is
/// symmetric after the last. A step of @ref coulomb, which checks its inputs.
/// Queued only; nothing is synchronized.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no
///         kernel). Launch errors only.
[[nodiscard]] wwr::wwrError_t j_coulomb(const CoulombPairClass &bra, CartesianMatrixAO &j);

} // namespace free_breakfast
