// The device half of free_breakfast.integrals.coulomb, before its launcher:
// what the generated J-build kernels (kernels/) promise a launcher, checked at
// compile time. One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.
//
// Each family's header declares its kernel template, every specialization and
// its table; the units under kernels/ define them. This file names no class:
// it asserts only that the tables were emitted for kMaxAngularMomentum and that
// e_hermite's fixed shared memory was sized for the block this build launches,
// so a stale emission or a mismatched warp width fails here, by name. A table
// entry no unit defines is a link error, which only a launcher's executable can
// show.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). angular_momentum.h and
// block_params.h include theirs first; the kernels' headers bring the runtime
// in through felizjueves/runtime.h.
#include <type_traits>

#include "angular_momentum.h"
#include "block_params.h"

#include "kernels/dq/dq.h"
#include "kernels/e_hermite/e_hermite.h"
#include "kernels/jab/jab.h"

namespace free_breakfast::device {

// One pair class per la <= lb <= kMaxAngularMomentum, and one Hermite class per
// lp, lq <= 2 kMaxAngularMomentum: a d-shell pair's E reaches |t| = 4.
constexpr auto kPairExtent = static_cast<size_t>(kMaxAngularMomentum + 1);
constexpr auto kHermiteExtent = static_cast<size_t>(2 * kMaxAngularMomentum + 1);

static_assert(std::extent_v<decltype(dq_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(dq_kernels), 1> == kPairExtent,
              "kernels/dq was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");
static_assert(std::extent_v<decltype(jab_kernels), 0> == kPairExtent &&
                  std::extent_v<decltype(jab_kernels), 1> == kPairExtent,
              "kernels/jab was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");
static_assert(std::extent_v<decltype(e_hermite_kernels), 0> == kHermiteExtent &&
                  std::extent_v<decltype(e_hermite_kernels), 1> == kHermiteExtent,
              "kernels/e_hermite was not emitted for kMaxAngularMomentum: rerun "
              "tools/emit_coulomb_kernels.py");

// One warp per bra record: blockDim = (1, warp width), the block e_hermite's
// Jsh[n_p * FJ_WAVEFRONT] was sized for. FJ_WAVEFRONT comes from WWR_WARP_SIZE
// (deps/CMakeLists.txt), so this holds unless the two have been split.
static_assert(e_hermite_block_dim[0] == 1 && e_hermite_block_dim[1] == kWarpSize,
              "e_hermite's block is not (1, kWarpSize): FJ_WAVEFRONT and "
              "WWR_WARP_SIZE disagree");

} // namespace free_breakfast::device
