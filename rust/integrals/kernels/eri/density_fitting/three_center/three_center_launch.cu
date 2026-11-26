// The DF three-center integrals' launch wrappers over the packed
// spherical-aux speri3<la, lb, lP> kernels and the fully spherical
// ssperi3<la, lb, lP> (FelizJueves#745): rust/'s own, which the C++ tree does
// not have (tools/emit/packed_three_center_kernels.py). One source for both
// backends -- nvcc under CUDA, clang -x hip under HIP. The two take the same
// arguments and refuse the same ones, so both are one function over the
// table each indexes.
//
// Its shape is forked from
// src/integrals/eri/density_fitting/three_center/three_center_launch.cu's
// square launcher, rewritten as three_center.h's extern "C", enqueue-only
// launch wrapper (rust/docs/gpu_layer.md, "Layering"), its refusals split
// into FB_LAUNCH_INVALID_ARGUMENT and FB_LAUNCH_UNSUPPORTED ("Statuses"), and
// the 32-bit extent the kernels index by refused rather than assumed. The
// bra records are read in place, from the class's own record rows
// (fb_primitive_pair_class_view), where the C++ launcher takes them packed
// into one staging buffer. The square kernels, eri3<la, lb, lP>, their
// launcher and the symmetrization are not carried (#1725). The generated
// kernels are sources of this library of their own (packed/kernels/ and
// spherical/kernels/), each one explicit specialization of the kernel
// template `speri3` or `ssperi3`. packed/kernels/speri3.h declares the
// template, every specialization and speri3_kernels, their [la][lb][lP]
// table, and spherical/kernels/ssperi3.h the same of ssperi3, nullptr where
// la > lb (only
// la <= lb is emitted), so nothing here names a class: a class the table
// names but no unit defines fails to link. Launching a specialization another
// TU defines wants relocatable device code under nvcc (CMakeLists.txt).

// Standard headers, and ours (which include only standard headers), before the
// wwr headers below: under HIP the vendor runtime header they pull in must not
// precede libc++ (primitives.h has why). The kernels' header brings the
// runtime in through felizjueves/runtime.h.
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "angular_momentum.h"
#include "block_params.h"
#include "primitives.h"

#include "eri/density_fitting/three_center/three_center.h"

// wwr_get_last_error(): the launch status, already mapped onto wwr_status.
#include "runtime_api/wwr_capi_runtime_api.h"

#include <runtime.h>

#include "eri/density_fitting/three_center/packed/kernels/speri3.h"
#include "eri/density_fitting/three_center/spherical/kernels/ssperi3.h"

static_assert(std::extent_v<decltype(speri3_kernels), 0> == FB_THREE_CENTER_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(speri3_kernels), 1> == FB_THREE_CENTER_MAX_MOMENTUM + 1 &&
                  std::extent_v<decltype(speri3_kernels), 2> ==
                      FB_THREE_CENTER_AUX_MAX_MOMENTUM + 1,
              "packed/kernels/'s [la][lb][lP] extents and the three-center limits disagree: "
              "rerun tools/emit/packed_three_center_kernels.py");
static_assert(std::is_same_v<decltype(ssperi3_kernels), decltype(speri3_kernels)>,
              "spherical/kernels/'s table and packed/kernels/'s differ in shape or parameters: "
              "rerun tools/emit/packed_three_center_kernels.py");
static_assert(FB_THREE_CENTER_MAX_MOMENTUM == free_breakfast::kMaxAngularMomentum,
              "(mn|P)'s orbital limit and the orbital basis's kMaxAngularMomentum disagree");

namespace {

/// The launch both wrappers share, over the [la][lb][lP] table `kernels`
/// indexes: speri3_kernels or ssperi3_kernels, one type.
fb_launch_status launch_class(decltype(speri3_kernels) &kernels, const wwr_stream stream,
                              const fb_primitive_pair_class_view *cls, const int32_t lp,
                              const int32_t n_aux_rec, const double *aux, const int32_t n_cell,
                              const int32_t *cells, const int32_t n_col, const int32_t n_aux,
                              double *eri3) {
  if (cls == nullptr || aux == nullptr || cells == nullptr || eri3 == nullptr || n_aux_rec < 1 ||
      n_cell < 1 || n_col < 1 || n_aux < 1 || int64_t{n_col} * n_aux > INT32_MAX) {
    return FB_LAUNCH_INVALID_ARGUMENT;
  }
  // The bra fields the kernels read, in their parameter order.
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
  if (la < 0 || lb < la || lb > FB_THREE_CENTER_MAX_MOMENTUM || lp < 0 ||
      lp > FB_THREE_CENTER_AUX_MAX_MOMENTUM) {
    return FB_LAUNCH_UNSUPPORTED;
  }
  const auto a = [&](const int32_t field) {
    return aux + (static_cast<size_t>(field) * static_cast<size_t>(n_aux_rec));
  };
  const auto c = [&](const int32_t field) {
    return cells + (static_cast<size_t>(field) * static_cast<size_t>(n_cell));
  };
  const auto kernel = kernels[la][lb][lp];
  // One thread per cell, kFourWarps to a block.
  const uint32_t blocks =
      free_breakfast::math::idivup(static_cast<uint32_t>(n_cell), free_breakfast::kFourWarps);
  kernel<<<blocks, free_breakfast::kFourWarps, 0, static_cast<wwr::wwrStream_t>(stream)>>>(
      n_cell, cls->n_rec, n_aux_rec, n_col, n_aux, bra[0], bra[1], bra[2], bra[3], bra[4], bra[5],
      bra[6], bra[7], bra[8], bra[9], bra[10], a(0), a(1), a(2), a(3), a(4), c(0), c(1), c(2), c(3),
      c(4), c(5), eri3);
  return static_cast<fb_launch_status>(wwr_get_last_error());
}

} // namespace

extern "C" fb_launch_status fb_three_center_packed_class(const wwr_stream stream,
                                                         const fb_primitive_pair_class_view *cls,
                                                         const int32_t lp, const int32_t n_aux_rec,
                                                         const double *aux, const int32_t n_cell,
                                                         const int32_t *cells, const int32_t n_col,
                                                         const int32_t n_aux, double *eri3) {
  return launch_class(speri3_kernels, stream, cls, lp, n_aux_rec, aux, n_cell, cells, n_col, n_aux,
                      eri3);
}

extern "C" fb_launch_status
fb_three_center_spherical_class(const wwr_stream stream, const fb_primitive_pair_class_view *cls,
                                const int32_t lp, const int32_t n_aux_rec, const double *aux,
                                const int32_t n_cell, const int32_t *cells, const int32_t n_col,
                                const int32_t n_aux, double *eri3) {
  return launch_class(ssperi3_kernels, stream, cls, lp, n_aux_rec, aux, n_cell, cells, n_col, n_aux,
                      eri3);
}
