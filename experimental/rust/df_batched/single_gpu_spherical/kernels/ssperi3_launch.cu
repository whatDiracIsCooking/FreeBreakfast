// The launch wrapper over the fully spherical packed ssperi3<la, lb, lP>
// kernels (emit.py, FelizJueves#745): rust/'s fb_three_center_packed_class
// (rust/integrals/kernels/eri/density_fitting/three_center/), the same
// arguments and refusals, launching ssperi3_kernels in place of
// speri3_kernels. CUDA only, like this experiment: the stream is the
// wwr_stream rust/'s planner hands a task (a cudaStream_t under CUDA), and
// the status is cudaGetLastError()'s, 0 on success, or one of the two
// refusals below.
//
// Output: class (la, lb, lp) of (mu nu|Q), all three indices real solid
// harmonics, element (mu, nu, q) of a cell at
// dst + n_aux (mu (2 lb + 1) + nu) + q, so n_col counts spherical bra
// columns and dst = (the pair's first spherical column) n_aux + (the aux
// shell's first spherical function). A diagonal cell stores its full square
// spherical block. Every cell zeroes its own elements first, so the output
// needs no zeroing and a run is bit-reproducible.

// Standard headers, and rust/'s device headers (standard headers only),
// before the runtime: the order rust/'s launchers keep (primitives.h has
// why).
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "angular_momentum.h"
#include "block_params.h"
#include "fb_launch.h"
#include "primitive_pair_class_view.h"
#include "primitives.h"

#include <runtime.h>

#include "ssperi3.h"

namespace {

/// The aux limit of the ssperi3 table, i: above the orbital one on purpose,
/// as rust/'s FB_THREE_CENTER_AUX_MAX_MOMENTUM.
constexpr int32_t kAuxMaxMomentum = 6;

} // namespace

static_assert(std::extent_v<decltype(ssperi3_kernels), 0> ==
                      free_breakfast::kMaxAngularMomentum + 1 &&
                  std::extent_v<decltype(ssperi3_kernels), 1> ==
                      free_breakfast::kMaxAngularMomentum + 1 &&
                  std::extent_v<decltype(ssperi3_kernels), 2> == kAuxMaxMomentum + 1,
              "ssperi3.h's [la][lb][lP] extents are not f, f, i: rerun emit.py");

extern "C" fb_launch_status fbx_ssperi3_packed_class(void *stream,
                                                     const fb_primitive_pair_class_view *cls,
                                                     const int32_t lp, const int32_t n_aux_rec,
                                                     const double *aux, const int32_t n_cell,
                                                     const int32_t *cells, const int32_t n_col,
                                                     const int32_t n_aux, double *eri3) {
  if (cls == nullptr || aux == nullptr || cells == nullptr || eri3 == nullptr || n_aux_rec < 1 ||
      n_cell < 1 || n_col < 1 || n_aux < 1 || int64_t{n_col} * n_aux > INT32_MAX) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const double *const bra[] = {cls->zeta, cls->p_x,  cls->p_y,  cls->p_z,  cls->pa_x, cls->pa_y,
                               cls->pa_z, cls->pb_x, cls->pb_y, cls->pb_z, cls->s};
  for (const double *field : bra) {
    if (field == nullptr) {
      return FB_LAUNCH_INVALID_ARGUMENT;
    }
  }
  if (cls->n_rec < 1) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  const int32_t la = cls->la;
  const int32_t lb = cls->lb;
  if (la < 0 || lb < la || lb > free_breakfast::kMaxAngularMomentum || lp < 0 ||
      lp > kAuxMaxMomentum) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto a = [&](const int32_t field) {
    return aux + (static_cast<size_t>(field) * static_cast<size_t>(n_aux_rec));
  };
  const auto c = [&](const int32_t field) {
    return cells + (static_cast<size_t>(field) * static_cast<size_t>(n_cell));
  };
  const auto kernel = ssperi3_kernels[la][lb][lp];
  // One thread per cell, kFourWarps to a block.
  const uint32_t blocks =
      free_breakfast::math::idivup(static_cast<uint32_t>(n_cell), free_breakfast::kFourWarps);
  kernel<<<blocks, free_breakfast::kFourWarps, 0, static_cast<cudaStream_t>(stream)>>>(
      n_cell, cls->n_rec, n_aux_rec, n_col, n_aux, bra[0], bra[1], bra[2], bra[3], bra[4], bra[5],
      bra[6], bra[7], bra[8], bra[9], bra[10], a(0), a(1), a(2), a(3), a(4), c(0), c(1), c(2), c(3),
      c(4), c(5), eri3);
  return static_cast<fb_launch_status>(cudaGetLastError());
}
