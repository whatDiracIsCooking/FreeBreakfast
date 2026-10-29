/// @file
/// @brief fp64 GEMM composed from vendor `Sgemm` calls on a hi/lo fp32 split:
/// how much of a consumer card's fp32 throughput can be spent buying fp64
/// accuracy, and what accuracy survives.
///
/// The question behind it: `nevpt2::rdms`' digest GEMMs run at ~92% of the
/// RTX 3080's fp64 ceiling and at the RX 9060 XT's, so no library or algorithm
/// change can help -- the only lever left is to stop using the fp64 units.
/// CUDA 13 and hipBLASLt both do that with int8 matrix cores, but the AMD one
/// is MI350X/MI355X-only, so neither reaches this tree's cards.
///
/// This is the same idea with fp32 instead of int8, and -- unlike the demo's
/// hand-written int8 WMMA digest, which lost to rocBLAS on kernel quality --
/// it writes no GEMM: the products are the vendor's own `Sgemm`.
///
/// **The catch, and what the k_chunk knob is for.** `Sgemm` accumulates in
/// fp32, so the split fixes the inputs' representation and does nothing for
/// the sum. Over the digest's K that is a ~1e-5 relative error, which no
/// splitting removes. Chunking K and folding each chunk's fp32 product into
/// fp64 is what recovers accuracy, and it costs a pass over C per chunk. The
/// tradeoff is the measurement.

export module free_breakfast.experimental.split_gemm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief The fp32 scratch @ref gemm_split_fp32 needs, caller-owned
///
/// `a_hi`/`a_lo` and `b_hi`/`b_lo` hold the operands' splits -- two fp32 planes
/// are exactly as many bytes as the fp64 operand they came from, so this adds
/// no memory over the inputs themselves. `c32` is one chunk's fp32 product.
struct SplitScratch {
  float *a_hi;
  float *a_lo;
  float *b_hi;
  float *b_lo;
  float *c32;
};

/// @brief The floats @ref SplitScratch needs for an m x n x k product:
/// 2(mk + kn) for the splits and mn for the chunk product
[[nodiscard]] int64_t split_scratch_floats(int64_t m, int64_t n, int64_t k) noexcept;

/// @brief C += op(A) op(B) in fp64, as `terms` fp32 products per K chunk
///
/// Column-major, `wwr::gemm`'s argument order and ops. **The operands must be
/// packed** -- `lda` the leading extent of A as stored (k under `OP_T`, m under
/// `OP_N`), `ldb` likewise -- because the K chunking slices them in place;
/// `Unsupported` otherwise. C is accumulated into, never overwritten.
///
/// `terms` picks how much of the split product is formed:
/// - 1: A_hi B_hi alone -- a plain fp32 GEMM, the speed ceiling of this route
///      and about 1e-7 relative at best;
/// - 2: + A_hi B_lo;
/// - 3: + A_lo B_hi -- the double-single product, ~2^-48 in the inputs. The
///      dropped A_lo B_lo is 2^-48 relative, below what fp32 accumulation
///      leaves anyway.
///
/// `k_chunk` is the K extent of one fp32 accumulation; `k` or more means one
/// chunk, which is fastest and least accurate. Each chunk runs `terms` Sgemm
/// calls into `scratch.c32` and one `accumulate_f32` pass into C.
///
/// Queued on res.stream() through res.blas(); nothing synchronizes.
[[nodiscard]] Status gemm_split_fp32(DeviceResources &res, wwr::wwrblasOperation_t op_a,
                                     wwr::wwrblasOperation_t op_b, int64_t m, int64_t n, int64_t k,
                                     const double *a, int64_t lda, const double *b, int64_t ldb,
                                     double *c, int64_t ldc, int64_t k_chunk, int32_t terms,
                                     const SplitScratch &scratch);

} // namespace free_breakfast
