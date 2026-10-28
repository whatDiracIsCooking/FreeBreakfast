// The device half of free_breakfast.recipes.einsum: the generic contraction.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include <limits>

#include "block_params.h"
#include "primitives.h"

#include "einsum_bridge.h"

namespace free_breakfast::device {

namespace {

// One thread per (output element, summed tuple) rather than one per output
// element with a loop over the sum: SC-NEVPT2 has calls with a tiny output
// summed over a huge tuple (Sr's "ipqr,pqrabc,iabc->i" has an output of nvirt
// and six summed active labels), where one thread per output element would
// leave all but a handful of the card idle. Spreading the sum over threads
// keeps it busy whichever side dominates, at an atomicAdd per term; the sum is
// reassociated, so results agree with a host loop to a tolerance, not bit for
// bit.
__global__ void einsum_contract_kernel(const EinsumPlan plan, const double *__restrict__ a,
                                       const double *__restrict__ b, const double *__restrict__ c,
                                       double *__restrict__ out, const double alpha) {
  const int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (t >= plan.out_size * plan.contracted_size) {
    return;
  }
  const int64_t element = t / plan.contracted_size;

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
  atomicAdd(&out[element], term);
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
