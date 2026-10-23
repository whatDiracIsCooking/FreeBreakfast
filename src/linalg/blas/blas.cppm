/// @file
/// @brief BLAS conveniences over wwr's typed dispatch: gemm for square operands,
/// and the sandwich products gemmwich and gmmwich
///
/// The general calls stay wwr's (`wwr::gemm`, `wwr::dgmm`, ...), re-exported by
/// free_breakfast.core.wwr.

export module free_breakfast.linalg.blas;

// :shared exports nothing: it holds the partitions' unexported helpers
// (raw_blas_handle, BlasOp, RealOp), and is re-exported only because every
// interface partition must be.
export import :shared;

// gemm (:gemm): wwr::gemm for n x n operands with every leading dimension n,
// its ops fixed at compile time. A convenience over wwr's call, not a
// replacement: any other shape calls wwr::gemm directly.
export import :gemm;

// gemmwich (:gemmwich): the sandwich product op(X) F op(X)^T -- X^T F X by
// default -- as two square gemms through a caller-supplied work matrix.
export import :gemmwich;

// gmmwich (:gmmwich): the same sandwich with a diagonal middle, op(X)^T diag(d)
// op(X), as a dgmm and a gemm, so the diagonal is never formed.
export import :gmmwich;
