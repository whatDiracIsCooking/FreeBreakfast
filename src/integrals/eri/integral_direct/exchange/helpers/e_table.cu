// The rolled K kernel's E tables (exchange_helpers_bridge.h): every record's
// undressed Hermite expansion, its nonzero entries, record-major. One source
// for both backends -- nvcc under CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). exchange_helpers.cuh
// includes its own first.
#include <type_traits>

#include "exchange_helpers.cuh"

#include "exchange_helpers_bridge.h"

namespace {

namespace fb = free_breakfast;

using fb::int32_t;
using fb::int64_t;

// A class's records as the table reads them, passed by value into a kernel.
struct ETableRecords {
  const double *zeta;
  const double *pa[3];
  const double *pb[3];
};

// One step of the recurrence (FelizJueves's mmd.hermite._raise): from the n
// entries of prev, the n + 1 of out, raised by x (X_PA or X_PB).
__device__ void e_raise(const double *prev, const int32_t n, const double x,
                        const double half_inv_zeta, double *out) {
  for (int32_t t = 0; t <= n; ++t) {
    double v = 0.0;
    if (t >= 1) {
      v += half_inv_zeta * prev[t - 1];
    }
    if (t < n) {
      v += x * prev[t];
    }
    if (t + 1 < n) {
      v += static_cast<double>(t + 1) * prev[t + 1];
    }
    out[t] = v;
  }
}

// One thread per record: each axis's E^{ij}_t for every i <= La, j <= Lb, as
// FelizJueves's e_axis forms E^{ij} (i raises by X_PA, then j by X_PB), then
// the record's nonzero E^{ab}_t, in exchange_e_table's order.
template <int32_t La, int32_t Lb>
__global__ void __launch_bounds__(fb::kFourWarps)
    e_table_kernel(const int32_t n_rec, const ETableRecords rec, const bool ket,
                   const int32_t width, double *e) {
  const auto r = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= n_rec) {
    return;
  }
  constexpr int32_t kL = La + Lb;
  const double zeta = rec.zeta[r];
  const double half_inv_zeta = 0.5 / zeta;
  double axis[3][La + 1][Lb + 1][kL + 1];
  for (int32_t d = 0; d < 3; ++d) {
    const double pa = rec.pa[d][r];
    const double pb = rec.pb[d][r];
    axis[d][0][0][0] = fb::math::exp(zeta * pa * pb);
    for (int32_t i = 1; i <= La; ++i) {
      e_raise(axis[d][i - 1][0], i, pa, half_inv_zeta, axis[d][i][0]);
    }
    for (int32_t i = 0; i <= La; ++i) {
      for (int32_t j = 1; j <= Lb; ++j) {
        e_raise(axis[d][i][j - 1], i + j, pb, half_inv_zeta, axis[d][i][j]);
      }
    }
  }
  double *out = e + static_cast<int64_t>(r) * width;
  int32_t k = 0;
  for (int32_t ax = La; ax >= 0; --ax) {
    for (int32_t ay = La - ax; ay >= 0; --ay) {
      const int32_t az = La - ax - ay;
      for (int32_t bx = Lb; bx >= 0; --bx) {
        for (int32_t by = Lb - bx; by >= 0; --by) {
          const int32_t bz = Lb - bx - by;
          for (int32_t n = 0; n <= kL; ++n) {
            for (int32_t tx = n; tx >= 0; --tx) {
              for (int32_t ty = n - tx; ty >= 0; --ty) {
                const int32_t tz = n - tx - ty;
                if (tx > ax + bx || ty > ay + by || tz > az + bz) {
                  continue;
                }
                const double v = axis[0][ax][bx][tx] * axis[1][ay][by][ty] * axis[2][az][bz][tz];
                out[k++] = ket && n % 2 != 0 ? -v : v;
              }
            }
          }
        }
      }
    }
  }
}

// [la][lb], through f. Its extent is the launcher's limit.
constexpr decltype(&e_table_kernel<0, 0>) kETableKernels[4][4] = {
    {&e_table_kernel<0, 0>, &e_table_kernel<0, 1>, &e_table_kernel<0, 2>, &e_table_kernel<0, 3>},
    {&e_table_kernel<1, 0>, &e_table_kernel<1, 1>, &e_table_kernel<1, 2>, &e_table_kernel<1, 3>},
    {&e_table_kernel<2, 0>, &e_table_kernel<2, 1>, &e_table_kernel<2, 2>, &e_table_kernel<2, 3>},
    {&e_table_kernel<3, 0>, &e_table_kernel<3, 1>, &e_table_kernel<3, 2>, &e_table_kernel<3, 3>},
};

constexpr int32_t kETableMaxMomentum =
    static_cast<int32_t>(std::extent_v<decltype(kETableKernels), 0>) - 1;

static_assert(std::extent_v<decltype(kETableKernels), 1> ==
                  std::extent_v<decltype(kETableKernels), 0>,
              "the E table kernels are square in (la, lb)");

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t exchange_e_table(const wwr::wwrStream_t stream, const int32_t la, const int32_t lb,
                                 const int32_t n_rec, const double *zeta, const double *const pa[3],
                                 const double *const pb[3], const bool ket, double *e) {
  if (la < 0 || la > kETableMaxMomentum || lb < 0 || lb > kETableMaxMomentum || n_rec < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const int32_t width = exchange_e_table_width(la, lb);
  if (static_cast<int64_t>(n_rec) * width > exchange::kMaxElements) {
    return wwr::wwrErrorInvalidValue;
  }
  const ETableRecords rec{zeta, {pa[0], pa[1], pa[2]}, {pb[0], pb[1], pb[2]}};
  kETableKernels[la][lb]<<<exchange::element_blocks(n_rec), kFourWarps, 0, stream>>>(n_rec, rec,
                                                                                     ket, width, e);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
