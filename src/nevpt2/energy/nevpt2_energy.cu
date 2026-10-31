// The device half of free_breakfast.nevpt2.energy: the three diagonal-slice
// accumulations the NumPy reference writes as slice assignments rather than
// einsums, and the Sijrs class's reduction. One source for both backends --
// nvcc under CUDA, clang -x hip under HIP.

// Standard headers first, then block_reduce.cuh, which orders its own ahead
// of the vendor runtime header (under HIP that must not precede libc++). It
// brings block_params.h's kFourWarps and primitives.h's math:: with it.
#include <algorithm> // std::min

#include "block_reduce.cuh"

#include "nevpt2_energy_bridge.h"

namespace free_breakfast::nevpt2::device {

// block_reduce lives in the flat free_breakfast::device; inside this namespace
// a bare `device::` names this module's launchers instead.
using free_breakfast::device::block_reduce;

namespace {

namespace cg = cooperative_groups;

// The Sijrs reduction's block, in warps: block_reduce takes it as a template
// argument.
constexpr uint32_t kSijrsWarps = kFourWarps / kWarpSize;

// The Sijrs reduction is a grid-stride loop, so its grid is clamped rather
// than sized to cover: past this many blocks only the atomics grow.
constexpr uint32_t kSijrsMaxBlocks = 1024;

struct SijrsAdd {
  __device__ double operator()(const double a, const double b) const { return a + b; }
};

// The three slices each cover n⁵ (four kept indices and the diagonal i), one
// thread per tuple, i fastest. Every array is row-major, so (x0, ..., x5) of
// an n⁶ is at ((((x0 n + x1) n + x2) n + x3) n + x4) n + x5.

// a16[p, i, r, a, b, i] += fdm2[p, r, a, b].
__global__ void diag_a16_kernel(const double *__restrict__ fdm2, double *__restrict__ a16,
                                const int64_t n, const int64_t count) {
  const int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (t >= count) {
    return;
  }
  const int64_t i = t % n;
  const int64_t prab = t / n;
  const int64_t b = prab % n;
  const int64_t a = (prab / n) % n;
  const int64_t r = (prab / (n * n)) % n;
  const int64_t p = prab / (n * n * n);
  atomicAdd(&a16[(((((((((p * n) + i) * n) + r) * n) + a) * n) + b) * n) + i], fdm2[prab]);
}

// a22[I, i, k, i, b, c] -= fdm2[I, k, b, c].
__global__ void diag_a22a_kernel(const double *__restrict__ fdm2, double *__restrict__ a22,
                                 const int64_t n, const int64_t count) {
  const int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (t >= count) {
    return;
  }
  const int64_t i = t % n;
  const int64_t ikbc = t / n;
  const int64_t c = ikbc % n;
  const int64_t b = (ikbc / n) % n;
  const int64_t k = (ikbc / (n * n)) % n;
  const int64_t ii = ikbc / (n * n * n);
  atomicAdd(&a22[(((((((((ii * n) + i) * n) + k) * n) + i) * n) + b) * n) + c], -fdm2[ikbc]);
}

// a22[I, i, k, a, i, c] += 2 fdm2[I, k, a, c].
__global__ void diag_a22b_kernel(const double *__restrict__ fdm2, double *__restrict__ a22,
                                 const int64_t n, const int64_t count) {
  const int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (t >= count) {
    return;
  }
  const int64_t i = t % n;
  const int64_t ikac = t / n;
  const int64_t c = ikac % n;
  const int64_t a = (ikac / n) % n;
  const int64_t k = (ikac / (n * n)) % n;
  const int64_t ii = ikac / (n * n * n);
  atomicAdd(&a22[(((((((((ii * n) + i) * n) + k) * n) + a) * n) + i) * n) + c], 2.0 * fdm2[ikac]);
}

// g is (il, a, j, b) column-major, so element t decodes il fastest, and the
// partner g[il, b, j, a] is in the same slab. One partials buffer per
// quantity, so the two folds need no barrier between them.
__global__ void __launch_bounds__(kFourWarps)
    sijrs_kernel(const double *__restrict__ g, const int64_t ni, const int64_t i0,
                 const int64_t ncore, const int64_t nvirt, const double *__restrict__ e_core,
                 const double *__restrict__ e_virt, double *__restrict__ acc) {
  __shared__ double norm_partials[kSijrsWarps];
  __shared__ double energy_partials[kSijrsWarps];
  const int64_t count = ni * nvirt * ncore * nvirt;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  double norm = 0.0;
  double energy = 0.0;
  for (int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x; t < count;
       t += stride) {
    int64_t rest = t;
    const int64_t il = rest % ni;
    rest /= ni;
    const int64_t a = rest % nvirt;
    rest /= nvirt;
    const int64_t j = rest % ncore;
    const int64_t b = rest / ncore;
    const double g_iajb = g[t];
    const double g_ibja = g[il + (ni * (b + (nvirt * (j + (ncore * a)))))];
    const double theta = (2.0 * g_iajb) - g_ibja;
    const double denominator = (e_core[j] - e_virt[a]) + (e_core[i0 + il] - e_virt[b]);
    norm += g_iajb * theta;
    energy += g_iajb / denominator * theta;
  }
  const cg::thread_block block = cg::this_thread_block();
  norm = block_reduce<kSijrsWarps>(block, norm_partials, norm, SijrsAdd{});
  energy = block_reduce<kSijrsWarps>(block, energy_partials, energy, SijrsAdd{});
  if (threadIdx.x == 0) {
    atomicAdd(&acc[0], norm);
    atomicAdd(&acc[1], energy);
  }
}

// n⁵, the tuples each slice covers.
// One thread per slab element, decoded as sijrs_kernel does; each writes its
// own g_bar element and adds its share to the four orbital energies.
__global__ void sijrs_adjoint_kernel(const double *__restrict__ g, const int64_t ni,
                                     const int64_t i0, const int64_t ncore, const int64_t nvirt,
                                     const double *__restrict__ e_core,
                                     const double *__restrict__ e_virt, double *__restrict__ g_bar,
                                     double *__restrict__ e_core_bar,
                                     double *__restrict__ e_virt_bar, const int64_t count) {
  const int64_t t = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (t >= count) {
    return;
  }
  int64_t rest = t;
  const int64_t il = rest % ni;
  rest /= ni;
  const int64_t a = rest % nvirt;
  rest /= nvirt;
  const int64_t j = rest % ncore;
  const int64_t b = rest / ncore;
  const double g_iajb = g[t];
  const double g_ibja = g[il + (ni * (b + (nvirt * (j + (ncore * a)))))];
  const double theta = (2.0 * g_iajb) - g_ibja;
  const double denominator = (e_core[j] - e_virt[a]) + (e_core[i0 + il] - e_virt[b]);
  g_bar[t] += 2.0 * theta / denominator;
  const double w = g_iajb * theta / (denominator * denominator);
  if (e_core_bar != nullptr) {
    atomicAdd(&e_core_bar[i0 + il], -w);
    atomicAdd(&e_core_bar[j], -w);
  }
  if (e_virt_bar != nullptr) {
    atomicAdd(&e_virt_bar[a], w);
    atomicAdd(&e_virt_bar[b], w);
  }
}

int64_t diag_count(const int64_t n) {
  return n * n * n * n * n;
}

uint32_t diag_blocks(const int64_t count) {
  return static_cast<uint32_t>(math::idivup(count, int64_t{kFourWarps}));
}

} // namespace

wwr::wwrError_t diag_a16(const wwr::wwrStream_t stream, const double *const fdm2, double *const a16,
                         const int32_t n) {
  const int64_t count = diag_count(n);
  if (count == 0) {
    return wwr::wwrSuccess;
  }
  diag_a16_kernel<<<diag_blocks(count), kFourWarps, 0, stream>>>(fdm2, a16, n, count);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t diag_a22a(const wwr::wwrStream_t stream, const double *const fdm2,
                          double *const a22, const int32_t n) {
  const int64_t count = diag_count(n);
  if (count == 0) {
    return wwr::wwrSuccess;
  }
  diag_a22a_kernel<<<diag_blocks(count), kFourWarps, 0, stream>>>(fdm2, a22, n, count);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t diag_a22b(const wwr::wwrStream_t stream, const double *const fdm2,
                          double *const a22, const int32_t n) {
  const int64_t count = diag_count(n);
  if (count == 0) {
    return wwr::wwrSuccess;
  }
  diag_a22b_kernel<<<diag_blocks(count), kFourWarps, 0, stream>>>(fdm2, a22, n, count);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t sijrs(const wwr::wwrStream_t stream, const double *const g, const int32_t ni,
                      const int32_t i0, const int32_t ncore, const int32_t nvirt,
                      const double *const e_core, const double *const e_virt, double *const acc) {
  const int64_t count = int64_t{ni} * nvirt * ncore * nvirt;
  if (count == 0) {
    return wwr::wwrSuccess;
  }
  const auto blocks = static_cast<uint32_t>(
      std::min(math::idivup(count, int64_t{kFourWarps}), int64_t{kSijrsMaxBlocks}));
  sijrs_kernel<<<blocks, kFourWarps, 0, stream>>>(g, ni, i0, ncore, nvirt, e_core, e_virt, acc);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t sijrs_adjoint(const wwr::wwrStream_t stream, const double *const g,
                              const int32_t ni, const int32_t i0, const int32_t ncore,
                              const int32_t nvirt, const double *const e_core,
                              const double *const e_virt, double *const g_bar,
                              double *const e_core_bar, double *const e_virt_bar) {
  const int64_t count = int64_t{ni} * nvirt * ncore * nvirt;
  if (count == 0) {
    return wwr::wwrSuccess;
  }
  sijrs_adjoint_kernel<<<static_cast<uint32_t>(math::idivup(count, int64_t{kFourWarps})),
                         kFourWarps, 0, stream>>>(g, ni, i0, ncore, nvirt, e_core, e_virt, g_bar,
                                                  e_core_bar, e_virt_bar, count);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::nevpt2::device
