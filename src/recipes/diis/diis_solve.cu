// The device half of free_breakfast.recipes.diis: the bordered coefficient
// solve. One source for both backends -- nvcc under CUDA, clang -x hip under
// HIP.

// Standard and FBF headers before anything that reaches the vendor runtime:
// under HIP a libc++ header must not follow it (see primitives.h).
#include "block_params.h"
#include "primitives.h"

#include "diis_solve_bridge.h"

namespace free_breakfast::device {

namespace {

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
    double best = math::fabs(b[(col * dim) + col]);
    for (int32_t r = col + 1; r < dim; ++r) {
      const double candidate = math::fabs(b[(r * dim) + col]);
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
    // A singular subspace selects the newest Fock: the plain, unextrapolated
    // one. Under circular storage the newest is at physical index `newest`,
    // not m - 1.
    coeff[i] = singular ? (i == newest ? 1.0 : 0.0) : rhs[i] / b[(i * dim) + i];
  }
}

} // namespace

wwr::wwrError_t diis_solve(const wwr::wwrStream_t stream, const int32_t m, const int32_t ld,
                           const int32_t newest, const double *gram, double *coeff,
                           int32_t *singular_out) {
  diis_solve_kernel<<<1, kWarpSize, 0, stream>>>(m, ld, newest, gram, coeff, singular_out);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
