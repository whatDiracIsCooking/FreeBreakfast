/// @file
/// @brief The partitions' shared, unexported helpers: the handle unwrap and the
/// op concepts
///
/// A private interface partition: it exports nothing, but the exported
/// templates that use these are instantiated in the importer's TU, so they must
/// be reachable from there (docs/style.md, "Module-private code").

export module free_breakfast.linalg.blas:shared;

import std;
import free_breakfast.core.wwr; // blas_handle, wwrblasHandle_t, wwrblasOperation_t

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

// The ops the real sandwiches take: _C is left out, since on a real T it is _T
// under another name.
template <wwr::wwrblasOperation_t Op>
concept RealOp = BlasOp<Op> && Op != wwr::WWRBLAS_OP_C;

} // namespace free_breakfast
