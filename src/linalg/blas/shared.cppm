/// @file
/// @brief The partitions' shared, unexported helpers: the handle unwrap, the
/// op concepts and the adjoint op
///
/// A private interface partition: it exports nothing, but the exported
/// templates that use these are instantiated in the importer's TU, so they must
/// be reachable from there (docs/style.md, "Module-private code").

export module free_breakfast.linalg.blas:shared;

import std;
import free_breakfast.core.wwr; // blas_handle, usual_fp, complex_fp, wwrblasOperation_t

namespace free_breakfast {

// wwr's own unwrapping (ScopedPointerMode::raw_handle) is private, so the two
// arms of blas_handle are resolved here.
template <wwr::extension::blas_handle H>
wwr::wwrblasHandle_t raw_blas_handle(const H &handle) noexcept {
  if constexpr (std::same_as<H, wwr::wwrblasHandle_t>) {
    return handle;
  } else {
    return handle.get();
  }
}

// The ops gemm takes: _C is the conjugate transpose, the same as _T on a real T.
template <wwr::wwrblasOperation_t Op>
concept BlasOp = Op == wwr::WWRBLAS_OP_N || Op == wwr::WWRBLAS_OP_T || Op == wwr::WWRBLAS_OP_C;

// T's adjoint as an op: the transpose on a real T, the conjugate transpose on a
// complex one.
template <wwr::usual_fp T>
constexpr wwr::wwrblasOperation_t kAdjointOp =
    wwr::complex_fp<T> ? wwr::WWRBLAS_OP_C : wwr::WWRBLAS_OP_T;

// The ops the sandwiches take: N, or T's adjoint, in the one spelling that is
// the adjoint for T. A real T leaves out _C, which on it is _T under another
// name; a complex T leaves out _T, because the sandwich's other factor would
// then be (X^T)^H = conj(X), which no BLAS op spells.
template <wwr::wwrblasOperation_t Op, typename T>
concept SandwichOp = Op == wwr::WWRBLAS_OP_N || Op == kAdjointOp<T>;

} // namespace free_breakfast
