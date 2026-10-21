// The device half of free_breakfast.integrals.coulomb.sort_bounds: each ket
// class's sort key, the sort (merge_sort.cu), and the gather of everything
// e_hermite reads of a ket into that order. One source for both backends --
// nvcc under CUDA, clang -x hip under HIP.

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridges do integral_types.h.
#include "block_params.h"
#include "primitives.h"

#include "merge_sort_bridge.h"
#include "sort_bounds_bridge.h"

namespace {

// key[k] = -(Iq * Dmax) and perm[k] = k: the ascending sort of key is the
// descending order of the screen's bound, carrying each record's index.
__global__ void sort_bounds_key_kernel(const int32_t n_rec, const double *bound, const double *dmax,
                                       double *key, uint32_t *perm) {
  const auto k = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (k < n_rec) {
    key[k] = -(bound[k] * dmax[k]);
    perm[k] = static_cast<uint32_t>(k);
  }
}

// One thread per element of the sorted D_q, row k = record perm[k], so the
// stores coalesce; the row's first thread also moves the record's scalars.
__global__ void sort_bounds_gather_kernel(const int32_t n_rec, const int32_t n_hermite,
                                          const uint32_t *perm, const double *zeta,
                                          const double *p_x, const double *p_y, const double *p_z,
                                          const double *bound, const double *dmax, const double *dq,
                                          double *zeta_sorted, double *p_x_sorted,
                                          double *p_y_sorted, double *p_z_sorted,
                                          double *bound_sorted, double *dmax_sorted,
                                          double *dq_sorted) {
  const auto i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<int64_t>(n_rec) * n_hermite) {
    return;
  }
  const auto k = static_cast<int32_t>(i / n_hermite);
  const auto q = static_cast<int32_t>(i % n_hermite);
  const auto from = static_cast<int64_t>(perm[k]);
  dq_sorted[i] = dq[from * n_hermite + q];
  if (q == 0) {
    zeta_sorted[k] = zeta[from];
    p_x_sorted[k] = p_x[from];
    p_y_sorted[k] = p_y[from];
    p_z_sorted[k] = p_z[from];
    bound_sorted[k] = bound[from];
    dmax_sorted[k] = dmax[from];
  }
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t sort_bounds_kets(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                 const int32_t n_hermite, const double *bound, const double *dmax,
                                 const double *dq, double *key, double *key_alt, uint32_t *perm,
                                 uint32_t *perm_alt, double *zeta_sorted, double *p_x_sorted,
                                 double *p_y_sorted, double *p_z_sorted, double *bound_sorted,
                                 double *dmax_sorted, double *dq_sorted) {
  if (cls.n_rec < 1 || n_hermite < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const auto n_rec = static_cast<uint32_t>(cls.n_rec);
  sort_bounds_key_kernel<<<math::idivup(n_rec, kFourWarps), kFourWarps, 0, stream>>>(
      cls.n_rec, bound, dmax, key, perm);
  if (const wwr::wwrError_t code = wwr::wwrGetLastError(); code != wwr::wwrSuccess) {
    return code;
  }
  if (const auto code = static_cast<wwr::wwrError_t>(
          merge_sort_passes(stream, n_rec, key, key_alt, perm, perm_alt));
      code != wwr::wwrSuccess) {
    return code;
  }
  const uint64_t elements = uint64_t{n_rec} * static_cast<uint64_t>(n_hermite);
  const auto blocks = static_cast<uint32_t>(math::idivup(elements, uint64_t{kFourWarps}));
  sort_bounds_gather_kernel<<<blocks, kFourWarps, 0, stream>>>(
      cls.n_rec, n_hermite, perm, cls.zeta, cls.p_x, cls.p_y, cls.p_z, bound, dmax, dq, zeta_sorted,
      p_x_sorted, p_y_sorted, p_z_sorted, bound_sorted, dmax_sorted, dq_sorted);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
