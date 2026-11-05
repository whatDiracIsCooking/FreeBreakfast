// The device half of free_breakfast.recipes.einsum: the generic contraction.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include <limits>

#include "block_params.h"
#include "primitives.h"

#include "einsum_bridge.h"

// The vendor cooperative-groups header, resolved per backend by WarpWraps.
#include <cooperative_groups.h>

namespace free_breakfast::device {

namespace {

namespace cg = cooperative_groups;

// One thread per (output element, summed tuple) rather than one per output
// element with a loop over the sum: SC-NEVPT2 has calls with a tiny output
// summed over a huge tuple (Sr's "ipqr,pqrabc,iabc->i" has an output of nvirt
// and six summed active labels), where one thread per output element would
// leave all but a handful of the card idle. Spreading the sum over threads
// keeps it busy whichever side dominates; the sum is reassociated, so results
// agree with a host loop to a tolerance, not bit for bit.
//
// The summed tuple runs fastest in t, so one element's terms are consecutive
// lanes, and a warp folds each run of equal elements before the run's first
// lane adds it: one atomicAdd per (element, warp) rather than per term. Without
// that, gfx12 crawls: it has no global fp64 atomic add, so atomicAdd there is a
// compare-and-swap loop, and a warp's lanes all on one address retry it one
// success at a time (README, "Kernel"). A call with nothing summed has one
// term per element, so it skips the fold and stays exact.

// Thread t's term: alpha times its operands' elements at its slots.
__device__ double contraction_term(const EinsumPlan &plan, const double *const a,
                                   const double *const b, const double *const c, const int64_t t,
                                   const int64_t element, const double alpha) {
  int64_t slot[kEinsumMaxLabels];
  int64_t rest = element;
  for (int32_t i = plan.out_rank - 1; i >= 0; --i) {
    slot[i] = rest % plan.out_dims[i];
    rest /= plan.out_dims[i];
  }
  rest = t % plan.contracted_size;
  for (int32_t i = plan.contracted_count - 1; i >= 0; --i) {
    slot[plan.out_rank + i] = rest % plan.contracted_dims[i];
    rest /= plan.contracted_dims[i];
  }

  const double *const operands[kEinsumMaxOperands] = {a, b, c};
  double term = alpha;
  for (int32_t o = 0; o < plan.operand_count; ++o) {
    int64_t at = 0;
    for (int32_t x = 0; x < plan.operand_rank[o]; ++x) {
      at += slot[plan.axis_label[o][x]] * plan.axis_stride[o][x];
    }
    term *= operands[o][at];
  }
  return term;
}

__global__ void einsum_contract_kernel(const EinsumPlan plan, const double *__restrict__ a,
                                       const double *__restrict__ b, const double *__restrict__ c,
                                       double *__restrict__ out, const double alpha) {
  const int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  const bool active = t < plan.out_size * plan.contracted_size;
  if (plan.contracted_size == 1) {
    if (active) {
      atomicAdd(&out[t], contraction_term(plan, a, b, c, t, t, alpha));
    }
    return;
  }
  // Past the end a lane holds element -1 and a zero term, and stays for the
  // shuffles, which are warp-wide collectives.
  const int64_t element = active ? t / plan.contracted_size : -1;
  double term = active ? contraction_term(plan, a, b, c, t, element, alpha) : 0.0;
  // A segmented suffix sum: before each step a lane holds its run's terms
  // over [lane, lane + offset), and appends its neighbour's when that lane is
  // in the warp and the same run (runs are contiguous, so equal elements at
  // both ends means one run between). The run's first lane ends with its sum.
  auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const uint32_t lane = warp.thread_rank();
  for (uint32_t offset = 1; offset < kWarpSize; offset <<= 1) {
    const double next = warp.shfl_down(term, offset);
    const int64_t next_element = warp.shfl_down(element, offset);
    if (lane + offset < kWarpSize && next_element == element) {
      term += next;
    }
  }
  const int64_t previous = warp.shfl_up(element, 1);
  if (active && (lane == 0 || previous != element)) {
    atomicAdd(&out[element], term);
  }
}

} // namespace

wwr::wwrError_t einsum_contract(const wwr::wwrStream_t stream, const EinsumPlan &plan,
                                const double *a, const double *b, const double *c, double *out,
                                const double alpha) {
  const int64_t terms = plan.out_size * plan.contracted_size;
  if (terms < 1) {
    return wwr::wwrSuccess;
  }
  const int64_t blocks = math::idivup(terms, int64_t{kFourWarps});
  // A grid's x extent is at most 2^31 - 1 blocks on both backends.
  if (blocks > int64_t{std::numeric_limits<int32_t>::max()}) {
    return wwr::wwrErrorInvalidValue;
  }
  einsum_contract_kernel<<<static_cast<uint32_t>(blocks), kFourWarps, 0, stream>>>(plan, a, b, c,
                                                                                   out, alpha);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
