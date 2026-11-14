// The DIIS coefficient solve: the bordered system built and solved in one
// block's shared memory. One source for both backends -- nvcc under CUDA,
// clang -x hip under HIP.
//
// Forked from src/recipes/diis/diis_solve.cu. The kernel is unchanged. The
// launcher is rewritten as diis_solve.h's extern "C", enqueue-only launch
// wrapper (rust/docs/gpu_layer.md, "From bridge to C ABI"): the bounds the C++
// doc comment stated (m in 1..ld, ld below the shared array's order, newest
// in 0..m-1) and null pointers are refused before anything is enqueued, since
// an ld past the array would write past the kernel's shared memory, and the
// status is the C ABI's wwr_get_last_error(), so no vendor code crosses.

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why).
#include "block_params.h"
#include "primitives.h"

#include "diis/diis_solve.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

// The device-pass gate: #errors outside a CUDA or HIP device compile.
#include <device_guard.h>

#include <runtime.h>

namespace free_breakfast::device {

namespace {

// The bordered system's largest order, the header's constant: the shared
// arrays are sized by it.
constexpr int32_t kDiisSolveMaxDim = FB_DIIS_SOLVE_MAX_DIM;
static_assert((kDiisSolveMaxDim + 1) * kDiisSolveMaxDim * sizeof(double) <= 48 * 1024,
              "the system within CUDA's 48 KiB of static shared memory");

// Below this pivot magnitude the subspace is treated as singular.
constexpr double kSingularPivot = 1e-14;

// One block: build the bordered system in shared memory, then solve it.
//
// Shared memory holds the dim x dim bordered matrix b (row-major, dim = m + 1)
// and the length-dim right-hand side, sized statically for the largest dim
// (kDiisSolveMaxDim); b is packed at stride dim, not the bound. The block builds
// the system cooperatively, striding by blockDim.x rather than the warp size,
// so a 64-wide wavefront and a 32-wide warp take the same path. Thread 0 then
// runs the elimination serially: dim is the history depth plus one, far too
// small to be worth parallelizing.
__global__ void diis_solve_kernel(const int32_t m, const int32_t ld, const int32_t newest,
                                  const double *__restrict__ gram, double *__restrict__ coeff,
                                  int32_t *__restrict__ singular_out) {
  __shared__ double b[kDiisSolveMaxDim * kDiisSolveMaxDim];
  __shared__ double rhs[kDiisSolveMaxDim];
  const int32_t dim = m + 1;
  const auto tid = static_cast<int32_t>(threadIdx.x);
  const auto stride = static_cast<int32_t>(blockDim.x);

  //   b[i][j] = <e_i, e_j>   (i, j < m; read from the upper triangle)
  //   b[i][m] = b[m][i] = -1 (the Lagrange border)
  //   b[m][m] = 0
  //   rhs[i]  = 0 (i < m), rhs[m] = -1
  for (int32_t idx = tid; idx < dim * dim; idx += stride) {
    const int32_t i = idx / dim;
    const int32_t j = idx % dim;
    double v = -1.0;
    if (i < m && j < m) {
      const int32_t a = i <= j ? i : j;
      const int32_t c = i <= j ? j : i;
      v = gram[(a * ld) + c];
    } else if (i == m && j == m) {
      v = 0.0;
    }
    b[idx] = v;
  }
  for (int32_t i = tid; i < dim; i += stride) {
    rhs[i] = i == m ? -1.0 : 0.0;
  }
  __syncthreads();

  if (tid != 0) {
    return;
  }

  // Gauss-Jordan with partial pivoting: every other row is eliminated, so the
  // solution is rhs[i] / b[i][i] with no back substitution.
  bool singular = false;
  for (int32_t col = 0; col < dim; ++col) {
    int32_t pivot = col;
    double best = fabs(b[(col * dim) + col]);
    for (int32_t r = col + 1; r < dim; ++r) {
      const double candidate = fabs(b[(r * dim) + col]);
      if (candidate > best) {
        best = candidate;
        pivot = r;
      }
    }
    if (best < kSingularPivot) {
      singular = true;
      break;
    }
    if (pivot != col) {
      for (int32_t j = 0; j < dim; ++j) {
        const double t = b[(col * dim) + j];
        b[(col * dim) + j] = b[(pivot * dim) + j];
        b[(pivot * dim) + j] = t;
      }
      const double t = rhs[col];
      rhs[col] = rhs[pivot];
      rhs[pivot] = t;
    }
    const double diag = b[(col * dim) + col];
    for (int32_t r = 0; r < dim; ++r) {
      if (r == col) {
        continue;
      }
      const double factor = b[(r * dim) + col] / diag;
      if (factor == 0.0) {
        continue;
      }
      for (int32_t j = col; j < dim; ++j) {
        b[(r * dim) + j] -= factor * b[(col * dim) + j];
      }
      rhs[r] -= factor * rhs[col];
    }
  }

  *singular_out = singular ? 1 : 0;
  for (int32_t i = 0; i < m; ++i) {
    // A singular subspace selects the newest iterate: the plain,
    // unextrapolated one. Under circular storage the newest is at physical
    // index `newest`, not m - 1.
    coeff[i] = singular ? (i == newest ? 1.0 : 0.0) : rhs[i] / b[(i * dim) + i];
  }
}

} // namespace

} // namespace free_breakfast::device

extern "C" fb_launch_status fb_diis_solve(const wwr_stream stream, const int32_t m,
                                          const int32_t ld, const int32_t newest,
                                          const double *gram, double *coeff, int32_t *singular) {
  using namespace free_breakfast;
  using namespace free_breakfast::device;
  if (m < 1 || ld < m || ld >= kDiisSolveMaxDim || newest < 0 || newest >= m || gram == nullptr ||
      coeff == nullptr || singular == nullptr) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // The C ABI's stream is the runtime's, behind a void *. One block: the
  // system is in its shared memory.
  diis_solve_kernel<<<1, kWarpSize, 0, static_cast<wwr::wwrStream_t>(stream)>>>(m, ld, newest, gram,
                                                                                coeff, singular);
  // The launch just enqueued, as a status: the C ABI's own wwr_get_last_error(),
  // whose wwr_status values are fb_launch_status's positive range.
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
