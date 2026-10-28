/// @file
/// @brief The einsum plan and its contraction launcher, declared once for the
/// module that builds the plan and the device TU that reads it.
///
/// Included by einsum.cpp in its GLOBAL MODULE FRAGMENT, and by einsum.cu
/// directly, for the reason src/integrals/one_electron/overlap/overlap_bridge.h
/// gives: a declaration in a module's purview gets module linkage and can never
/// bind to a definition compiled in a plain TU.
///
/// The plan crosses into the .cu BY VALUE, and the .cu's host half is compiled
/// by nvcc's host compiler, which need not share libc++'s ABI with the module
/// units. So it is a POD of fixed-width integers and C arrays, nothing with a
/// library-defined layout; the static_assert below holds it to that.

#pragma once

// Standard (via integral_types.h) before runtime.h: under HIP a libc++ header
// must not follow the vendor runtime header (see primitives.h).
#include "integral_types.h"

#include <type_traits>

// wwr/runtime_api.h #errors in a device pass, and runtime.h's host part lacks
// wwrError_t: see src/integrals/one_electron/overlap/overlap_bridge.h.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)
#include "runtime.h"
#else
#include "wwr/runtime_api.h"
#endif

namespace free_breakfast::device {

/// @brief The most operands one einsum takes.
///
/// This and the three maxima below are the demo's measured maxima over every
/// call site SC-NEVPT2 has (nevpt2-gpu-demo, src/einsum/einsum_plan.h), not a
/// round guess: the plan is sized by them, and einsum refuses a call past any.
inline constexpr int32_t kEinsumMaxOperands = 3;
/// @brief The highest rank of an operand or of the output.
inline constexpr int32_t kEinsumMaxRank = 6;
/// @brief The most summed (contracted) labels in one call.
inline constexpr int32_t kEinsumMaxContracted = 6;
/// @brief The most distinct labels in one call, output and summed together.
inline constexpr int32_t kEinsumMaxLabels = 8;

/// @brief One contraction, as the kernel reads it.
///
/// Labels are numbered as slots: the output's, in output order, then the
/// summed ones. A thread's flat index is (output element, summed tuple), each
/// half decoded row-major over its extents into slot values, and operand o's
/// element is at sum over its axes a of slot[axis_label[o][a]] *
/// axis_stride[o][a]. A label repeated within an operand (a diagonal) is two
/// axes reading the same slot.
struct EinsumPlan {
  /// @brief How many operands are read (1..kEinsumMaxOperands).
  int32_t operand_count = 0;
  /// @brief Each operand's rank.
  int32_t operand_rank[kEinsumMaxOperands] = {};
  /// @brief The slot each operand axis reads.
  int32_t axis_label[kEinsumMaxOperands][kEinsumMaxRank] = {};
  /// @brief Each operand axis's stride, in elements.
  int64_t axis_stride[kEinsumMaxOperands][kEinsumMaxRank] = {};
  /// @brief The output's rank; its slots are 0..out_rank-1.
  int32_t out_rank = 0;
  /// @brief The output's extents; the output is packed row-major over them.
  int64_t out_dims[kEinsumMaxRank] = {};
  /// @brief How many labels are summed; their slots follow the output's.
  int32_t contracted_count = 0;
  /// @brief The summed labels' extents.
  int64_t contracted_dims[kEinsumMaxContracted] = {};
  /// @brief The product of out_dims (1 for a scalar output).
  int64_t out_size = 1;
  /// @brief The product of contracted_dims (1 when nothing is summed).
  int64_t contracted_size = 1;
};

static_assert(std::is_trivially_copyable_v<EinsumPlan> && std::is_standard_layout_v<EinsumPlan>,
              "EinsumPlan crosses einsum_bridge.h by value: keep it a POD");

/// @brief out[e] += alpha * prod over operands of op[...], for every output
/// element e and summed tuple @p plan describes, on @p stream.
///
/// One thread per (output element, summed tuple), each adding its one term
/// with atomicAdd. @p b and @p c are read only when @p plan names that many
/// operands, and may be null otherwise. A plan with no terms launches nothing.
///
/// @return The launch status (wwrErrorInvalidValue when the terms need more
///         blocks than a grid holds). Launch errors only: the kernel's own
///         faults surface at the next synchronization.
[[nodiscard]] wwr::wwrError_t einsum_contract(wwr::wwrStream_t stream, const EinsumPlan &plan,
                                              const double *a, const double *b, const double *c,
                                              double *out, double alpha);

} // namespace free_breakfast::device
