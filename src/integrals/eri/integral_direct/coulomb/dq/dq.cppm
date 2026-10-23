/// @file
/// @brief The J build's ket pre-contraction, over the dq<la, lb> kernels
/// FelizJueves generates: one class's weights, D_q and Dmax.

export module free_breakfast.integrals.eri.integral_direct.coulomb.dq;

import free_breakfast.core.wwr;
import free_breakfast.integrals.eri.integral_direct.coulomb.pair_class;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief Pre-contract @p density into @p ket's Hermite basis, on the class's
/// own stream: its scratch w (each record's weight, 2 for two shells, 1 for a
/// shell against itself), dq (D_q = w sum_cd D_cd s E^cd_q, q-major) and dmax
/// (max_q |D_q|), each written whole, in record order.
///
/// A step of @ref coulomb, which checks its inputs: @p density must be
/// symmetric, on the class's device, of the basis the class was made for, and
/// small enough for the kernels' 32-bit index (n^2 <= INT32_MAX), and the class
/// at most d with la <= lb. Queued only; nothing is synchronized.
///
/// @return The launch status (wwrErrorInvalidValue for a class with no
///         kernel). Launch errors only.
[[nodiscard]] wwr::wwrError_t dq(CoulombPairClass &ket, const CartesianMatrixAO &density);

} // namespace free_breakfast
