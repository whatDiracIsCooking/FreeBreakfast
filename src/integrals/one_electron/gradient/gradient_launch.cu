// The device half of free_breakfast.integrals.one_electron.gradient: the
// launchers over the generated overlap_gradient<la, lb>,
// kinetic_gradient<la, lb> and potential_gradient<la, lb> kernels, and the
// gather of each record's two atoms they read; and over the derivative-matrix
// kernels overlap_ip1<la, lb>, kinetic_ip1<la, lb>, potential_ip1<la, lb> and
// potential_rinv<la, lb>, with the assembly of each atom's matrices from them.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of its family's kernel template.
// kernels/<family>.h declares the template, every specialization and
// <family>_kernels, their [la][lb] table, so nothing here names a class: a
// class a table names but no unit defines fails to link. Launching a
// specialization another TU defines wants relocatable device code under nvcc
// (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' headers bring the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "gradient_bridge.h"

#include "kernels/kinetic_gradient.h"
#include "kernels/kinetic_ip1.h"
#include "kernels/overlap_gradient.h"
#include "kernels/overlap_ip1.h"
#include "kernels/potential_gradient.h"
#include "kernels/potential_ip1.h"
#include "kernels/potential_rinv.h"

namespace {

using free_breakfast::PrimitivePairClassView;

// Record r's bra atom into atom_a[r] and its ket atom into atom_b[r], each the
// atom of the shell's first Cartesian AO.
__global__ void record_atoms(const int32_t n_rec, const int32_t *off_a, const int32_t *off_b,
                             const int32_t *ao_atom, int32_t *atom_a, int32_t *atom_b) {
  const auto r = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (r < n_rec) {
    atom_a[r] = ao_atom[off_a[r]];
    atom_b[r] = ao_atom[off_b[r]];
  }
}

// out[A, k, m, n] += [ao_atom[m] == A] y[k, m, n] + [ao_atom[n] == A] y[k, n, m],
// one thread per (m, n), blockIdx.y the slab A k.
__global__ void centre_derivatives(const int32_t n, const int32_t *ao_atom, const double *y,
                                   double *out) {
  const auto nn = static_cast<int64_t>(n) * n;
  const auto e =
      static_cast<int64_t>((blockIdx.x * static_cast<int64_t>(blockDim.x)) + threadIdx.x);
  if (e >= nn) {
    return;
  }
  const auto atom = static_cast<int32_t>(blockIdx.y / 3);
  const int64_t k = blockIdx.y % 3;
  const int64_t m = e / n;
  const int64_t c = e % n;
  double v = 0.0;
  if (ao_atom[m] == atom) {
    v += y[(k * nn) + (m * n) + c];
  }
  if (ao_atom[c] == atom) {
    v += y[(k * nn) + (c * n) + m];
  }
  out[(blockIdx.y * nn) + e] += v;
}

// The largest momentum each family's table covers, its own limit rather than
// the tree's kMaxAngularMomentum, so a family moves to a higher l when its
// kernels are emitted for one.
constexpr int32_t kOverlapMaxL =
    static_cast<int32_t>(std::extent_v<decltype(overlap_gradient_kernels), 0>) - 1;
constexpr int32_t kKineticMaxL =
    static_cast<int32_t>(std::extent_v<decltype(kinetic_gradient_kernels), 0>) - 1;
constexpr int32_t kPotentialMaxL =
    static_cast<int32_t>(std::extent_v<decltype(potential_gradient_kernels), 0>) - 1;

constexpr int32_t kOverlapIp1MaxL =
    static_cast<int32_t>(std::extent_v<decltype(overlap_ip1_kernels), 0>) - 1;
constexpr int32_t kKineticIp1MaxL =
    static_cast<int32_t>(std::extent_v<decltype(kinetic_ip1_kernels), 0>) - 1;
constexpr int32_t kPotentialIp1MaxL =
    static_cast<int32_t>(std::extent_v<decltype(potential_ip1_kernels), 0>) - 1;
constexpr int32_t kPotentialRinvMaxL =
    static_cast<int32_t>(std::extent_v<decltype(potential_rinv_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(overlap_gradient_kernels), 1> == kOverlapMaxL + 1 &&
                  std::extent_v<decltype(kinetic_gradient_kernels), 1> == kKineticMaxL + 1 &&
                  std::extent_v<decltype(potential_gradient_kernels), 1> == kPotentialMaxL + 1 &&
                  std::extent_v<decltype(overlap_ip1_kernels), 1> == kOverlapIp1MaxL + 1 &&
                  std::extent_v<decltype(kinetic_ip1_kernels), 1> == kKineticIp1MaxL + 1 &&
                  std::extent_v<decltype(potential_ip1_kernels), 1> == kPotentialIp1MaxL + 1 &&
                  std::extent_v<decltype(potential_rinv_kernels), 1> == kPotentialRinvMaxL + 1,
              "a kernels/ [la][lb] table is not square: rerun "
              "tools/emit/one_electron_gradient_kernels.py");

// One thread per record, kFourWarps to a block; a whole number of warps keeps
// every block's lanes in step.
uint32_t record_blocks(const PrimitivePairClassView &cls) {
  return free_breakfast::math::idivup(static_cast<uint32_t>(cls.n_rec), free_breakfast::kFourWarps);
}

// Whether cls has a kernel in a table through max_l, and the launch has
// records and atoms.
bool launchable(const PrimitivePairClassView &cls, const int32_t max_l, const int32_t n_atom) {
  return cls.la >= 0 && cls.la <= cls.lb && cls.lb <= max_l && cls.n_rec >= 1 && n_atom >= 1;
}

// record_atoms over cls into atoms' two halves.
wwr::wwrError_t gather_atoms(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                             const int32_t *ao_atom, int32_t *atoms) {
  record_atoms<<<record_blocks(cls), free_breakfast::kFourWarps, 0, stream>>>(
      cls.n_rec, cls.off_a, cls.off_b, ao_atom, atoms, atoms + cls.n_rec);
  return wwr::wwrGetLastError();
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t overlap_gradient_class(const wwr::wwrStream_t stream,
                                       const PrimitivePairClassView &cls, const int32_t *ao_atom,
                                       int32_t *atoms, const int32_t n_ao, const int32_t n_atom,
                                       const double *m, double *g) {
  if (!launchable(cls, kOverlapMaxL, n_atom)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (const wwr::wwrError_t err = gather_atoms(stream, cls, ao_atom, atoms);
      err != wwr::wwrSuccess) {
    return err;
  }
  overlap_gradient_kernels[cls.la][cls.lb]<<<record_blocks(cls), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, n_atom, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y, cls.pb_z,
      cls.s, cls.ea, cls.eb, cls.off_a, cls.off_b, atoms, atoms + cls.n_rec, m, g);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t kinetic_gradient_class(const wwr::wwrStream_t stream,
                                       const PrimitivePairClassView &cls, const int32_t *ao_atom,
                                       int32_t *atoms, const int32_t n_ao, const int32_t n_atom,
                                       const double *m, double *g) {
  if (!launchable(cls, kKineticMaxL, n_atom)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (const wwr::wwrError_t err = gather_atoms(stream, cls, ao_atom, atoms);
      err != wwr::wwrSuccess) {
    return err;
  }
  kinetic_gradient_kernels[cls.la][cls.lb]<<<record_blocks(cls), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, n_atom, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y, cls.pb_z,
      cls.s, cls.ea, cls.eb, cls.off_a, cls.off_b, atoms, atoms + cls.n_rec, m, g);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t potential_gradient_class(const wwr::wwrStream_t stream,
                                         const PrimitivePairClassView &cls, const int32_t *ao_atom,
                                         int32_t *atoms, const NucleiView &nuclei,
                                         const int32_t n_ao, const double *m, double *g) {
  if (!launchable(cls, kPotentialMaxL, nuclei.n_nuc)) {
    return wwr::wwrErrorInvalidValue;
  }
  if (const wwr::wwrError_t err = gather_atoms(stream, cls, ao_atom, atoms);
      err != wwr::wwrSuccess) {
    return err;
  }
  potential_gradient_kernels[cls.la][cls.lb]<<<record_blocks(cls), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, nuclei.n_nuc, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y,
      cls.pb_z, cls.p_x, cls.p_y, cls.p_z, cls.s, cls.ea, cls.eb, cls.off_a, cls.off_b, atoms,
      atoms + cls.n_rec, m, nuclei.z, nuclei.c, g);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t overlap_ip1_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                  const int32_t n_ao, double *x1) {
  if (!launchable(cls, kOverlapIp1MaxL, 1)) {
    return wwr::wwrErrorInvalidValue;
  }
  overlap_ip1_kernels[cls.la][cls.lb]<<<record_blocks(cls), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y, cls.pb_z, cls.s,
      cls.ea, cls.eb, cls.off_a, cls.off_b, x1);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t kinetic_ip1_class(const wwr::wwrStream_t stream, const PrimitivePairClassView &cls,
                                  const int32_t n_ao, double *x1) {
  if (!launchable(cls, kKineticIp1MaxL, 1)) {
    return wwr::wwrErrorInvalidValue;
  }
  kinetic_ip1_kernels[cls.la][cls.lb]<<<record_blocks(cls), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y, cls.pb_z, cls.s,
      cls.ea, cls.eb, cls.off_a, cls.off_b, x1);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t potential_ip1_class(const wwr::wwrStream_t stream,
                                    const PrimitivePairClassView &cls, const NucleiView &nuclei,
                                    const int32_t n_ao, double *x1) {
  if (!launchable(cls, kPotentialIp1MaxL, nuclei.n_nuc)) {
    return wwr::wwrErrorInvalidValue;
  }
  potential_ip1_kernels[cls.la][cls.lb]<<<record_blocks(cls), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, nuclei.n_nuc, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y,
      cls.pb_z, cls.p_x, cls.p_y, cls.p_z, cls.s, cls.ea, cls.eb, cls.off_a, cls.off_b, nuclei.z,
      nuclei.c, x1);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t potential_rinv_class(const wwr::wwrStream_t stream,
                                     const PrimitivePairClassView &cls, const NucleiView &nuclei,
                                     const int32_t nuc, const int32_t n_ao, double *xc) {
  if (!launchable(cls, kPotentialRinvMaxL, nuclei.n_nuc) || nuc < 0 || nuc >= nuclei.n_nuc) {
    return wwr::wwrErrorInvalidValue;
  }
  potential_rinv_kernels[cls.la][cls.lb]<<<record_blocks(cls), kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, nuclei.n_nuc, nuc, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x,
      cls.pb_y, cls.pb_z, cls.p_x, cls.p_y, cls.p_z, cls.s, cls.off_a, cls.off_b, nuclei.z,
      nuclei.c, xc);
  return wwr::wwrGetLastError();
}

wwr::wwrError_t add_centre_derivatives(const wwr::wwrStream_t stream, const int32_t n_atom,
                                       const int32_t n, const int32_t *ao_atom, const double *y,
                                       double *out) {
  if (n_atom < 1 || n < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const dim3 grid(math::idivup(static_cast<uint32_t>(n) * static_cast<uint32_t>(n), kFourWarps),
                  static_cast<uint32_t>(3 * n_atom));
  centre_derivatives<<<grid, kFourWarps, 0, stream>>>(n, ao_atom, y, out);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
