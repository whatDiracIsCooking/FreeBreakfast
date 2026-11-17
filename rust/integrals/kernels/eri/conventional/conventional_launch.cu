// The conventional ERI's launch wrappers over the generated eri<la, lb, lc,
// ld> and eri_kperm kernels, and the packing and unpacking kernels around
// the GEMVs. One source for both backends -- nvcc under CUDA, clang -x hip
// under HIP.
//
// rust/'s own, with no C++ launcher to fork (conventional.h has why), in the
// shape of the forked ones (rust/docs/gpu_layer.md, "Layering",
// "Statuses"): extern "C", enqueue-only, a refusal split into
// FB_LAUNCH_INVALID_ARGUMENT and FB_LAUNCH_UNSUPPORTED, and the 32-bit
// extent the kernels index by refused rather than assumed. The records are
// read in place from each class's record rows (fb_primitive_pair_class_view).
// The generated kernels are sources of this library of their own
// (kernels/): kernels/eri.h declares the template, every specialization and
// eri_kernels, their [la][lb][lc][ld] table, nullptr where a class is not
// emitted, so nothing here names a class; eri_kperm is one plain kernel,
// declared below with the parameters its unit spells. Launching a kernel
// another TU defines wants relocatable device code under nvcc
// (CMakeLists.txt).

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why). The kernels' header brings the
// runtime in through felizjueves/runtime.h.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "eri/conventional/conventional.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

#include "eri/conventional/kernels/eri.h"

static_assert(std::extent_v<decltype(eri_kernels), 0> == FB_CONVENTIONAL_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(eri_kernels), 1> == FB_CONVENTIONAL_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(eri_kernels), 2> == FB_CONVENTIONAL_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(eri_kernels), 3> == FB_CONVENTIONAL_MAX_MOMENTUM + 1,
              "kernels/'s [la][lb][lc][ld] extents and FB_CONVENTIONAL_MAX_MOMENTUM disagree: "
              "rerun tools/emit/conventional_kernels.py");

// kernels/eri_kperm.cu's kernel, as it spells it.
extern "C" __global__ void eri_kperm(int32_t n_ao, int32_t n_pp, const int32_t *pm,
                                     const int32_t *pn, const int32_t *pidx, const double *eri,
                                     double *X);

namespace {

// n_ao (n_ao + 1) / 2, the packed pair count, for an n_ao the wrappers
// accept: at most FB_CONVENTIONAL_MAX_AO, so n_pp^2 is far inside an int32.
__host__ __device__ int32_t packed(const int32_t n_ao) {
  return n_ao * (n_ao + 1) / 2;
}

// Whether an n_ao is one the wrappers accept.
bool accepted(const int32_t n_ao) {
  return n_ao >= 1 && n_ao <= FB_CONVENTIONAL_MAX_AO;
}

// One thread per packed pair i = pidx(l, s), l >= s: l the largest with
// l (l + 1) / 2 <= i, found by stepping from the floating-point root, so
// exact. D is read at d[l n + s] and d[s n + l]: either layout of D gives
// the same sum.
__global__ void conventional_pack_kernel(const int32_t n_ao, const double *d, double *dp) {
  const int32_t i = static_cast<int32_t>(threadIdx.x + (blockIdx.x * blockDim.x));
  if (i >= packed(n_ao)) {
    return;
  }
  int32_t l = static_cast<int32_t>((sqrt(8.0 * i + 1.0) - 1.0) / 2.0);
  while (l * (l + 1) / 2 > i) {
    --l;
  }
  while ((l + 1) * (l + 2) / 2 <= i) {
    ++l;
  }
  const int32_t s = i - (l * (l + 1) / 2);
  dp[i] = l == s ? d[(l * n_ao) + s] : d[(l * n_ao) + s] + d[(s * n_ao) + l];
}

// One thread per element (m, n) of v: the packed value of its pair, so
// (m, n) and (n, m) read one element.
__global__ void conventional_unpack_kernel(const int32_t n_ao, const double *vp, double *v) {
  const int32_t i = static_cast<int32_t>(threadIdx.x + (blockIdx.x * blockDim.x));
  if (i >= n_ao * n_ao) {
    return;
  }
  const int32_t a = i / n_ao;
  const int32_t b = i % n_ao;
  const int32_t m = a > b ? a : b;
  const int32_t n = a > b ? b : a;
  v[i] = vp[(m * (m + 1) / 2) + n];
}

// One thread per element, kFourWarps to a block.
uint32_t blocks_for(const int32_t threads) {
  return free_breakfast::math::idivup(static_cast<uint32_t>(threads), free_breakfast::kFourWarps);
}

} // namespace

extern "C" fb_launch_status
fb_conventional_eri_class(const wwr_stream stream, const fb_primitive_pair_class_view *bra,
                          const fb_primitive_pair_class_view *ket, const int32_t n_bpair,
                          const int32_t *bra_pairs, const int32_t n_kpair, const int32_t *ket_pairs,
                          const int32_t n_task, const int32_t *tasks, const int32_t n_ao,
                          const int32_t *pidx, double *eri) {
  if (bra == nullptr || ket == nullptr || bra_pairs == nullptr || ket_pairs == nullptr ||
      tasks == nullptr || pidx == nullptr || eri == nullptr || n_bpair < 1 || n_kpair < 1 ||
      n_task < 1 || !accepted(n_ao)) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // Each side's fields the kernels read, in their parameter order.
  const double *const b[] = {bra->zeta, bra->p_x,  bra->p_y,  bra->p_z,  bra->pa_x, bra->pa_y,
                             bra->pa_z, bra->pb_x, bra->pb_y, bra->pb_z, bra->s};
  const double *const k[] = {ket->zeta, ket->p_x,  ket->p_y,  ket->p_z,  ket->pa_x, ket->pa_y,
                             ket->pa_z, ket->pb_x, ket->pb_y, ket->pb_z, ket->s};
  for (int32_t f = 0; f < 11; ++f) {
    if (b[f] == nullptr || k[f] == nullptr) {
      return FB_LAUNCH_INVALID_ARGUMENT;
    }
  }
  if (bra->n_rec < 1 || ket->n_rec < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const int32_t la = bra->la;
  const int32_t lb = bra->lb;
  const int32_t lc = ket->la;
  const int32_t ld = ket->lb;
  if (la < 0 || lb < la || lb > FB_CONVENTIONAL_MAX_MOMENTUM || lc < 0 || ld < lc ||
      ld > FB_CONVENTIONAL_MAX_MOMENTUM) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto kernel = eri_kernels[la][lb][lc][ld];
  if (kernel == nullptr) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto bp = [&](const int32_t field) {
    return bra_pairs + (static_cast<size_t>(field) * static_cast<size_t>(n_bpair));
  };
  const auto kp = [&](const int32_t field) {
    return ket_pairs + (static_cast<size_t>(field) * static_cast<size_t>(n_kpair));
  };
  const auto t = [&](const int32_t field) {
    return tasks + (static_cast<size_t>(field) * static_cast<size_t>(n_task));
  };
  // The C ABI's stream is the runtime's, behind a void *.
  kernel<<<blocks_for(n_task), free_breakfast::kFourWarps, 0,
           static_cast<wwr::wwrStream_t>(stream)>>>(
      n_task, bra->n_rec, ket->n_rec, n_bpair, n_kpair, n_ao, packed(n_ao), b[0], b[1], b[2], b[3],
      b[4], b[5], b[6], b[7], b[8], b[9], b[10], k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
      k[8], k[9], k[10], bp(0), bp(1), bp(2), bp(3), kp(0), kp(1), kp(2), kp(3), t(0), t(1), pidx,
      eri);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

extern "C" fb_launch_status fb_conventional_exchange(const wwr_stream stream, const int32_t n_ao,
                                                     const int32_t *pm, const int32_t *pn,
                                                     const int32_t *pidx, const double *eri,
                                                     double *x) {
  if (pm == nullptr || pn == nullptr || pidx == nullptr || eri == nullptr || x == nullptr ||
      !accepted(n_ao)) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const int32_t n_pp = packed(n_ao);
  // The two ranges as addresses, which compare across allocations where the
  // pointers themselves need not.
  const auto e_begin = reinterpret_cast<uintptr_t>(eri);
  const auto x_begin = reinterpret_cast<uintptr_t>(x);
  const auto bytes = static_cast<uintptr_t>(n_pp) * static_cast<uintptr_t>(n_pp) * sizeof(double);
  if (e_begin < x_begin + bytes && x_begin < e_begin + bytes) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  eri_kperm<<<blocks_for(n_pp), free_breakfast::kFourWarps, 0,
              static_cast<wwr::wwrStream_t>(stream)>>>(n_ao, n_pp, pm, pn, pidx, eri, x);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

extern "C" fb_launch_status fb_conventional_pack(const wwr_stream stream, const int32_t n_ao,
                                                 const double *d, double *dp) {
  if (d == nullptr || dp == nullptr || !accepted(n_ao)) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  conventional_pack_kernel<<<blocks_for(packed(n_ao)), free_breakfast::kFourWarps, 0,
                             static_cast<wwr::wwrStream_t>(stream)>>>(n_ao, d, dp);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

extern "C" fb_launch_status fb_conventional_unpack(const wwr_stream stream, const int32_t n_ao,
                                                   const double *vp, double *v) {
  if (vp == nullptr || v == nullptr || !accepted(n_ao)) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  conventional_unpack_kernel<<<blocks_for(n_ao * n_ao), free_breakfast::kFourWarps, 0,
                               static_cast<wwr::wwrStream_t>(stream)>>>(n_ao, vp, v);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}
