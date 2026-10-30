// The device half of coulomb_gradient: the launcher over the generated
// gjab<la, lb> kernels, and the gather of each record's two atoms they read.
// One source for both backends -- nvcc under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template `gjab`.
// kernels/gjab.h declares the template, every specialization and gjab_kernels,
// their [la][lb] table, so nothing here names a class: a class the table names
// but no unit defines fails to link. Launching a specialization another TU
// defines wants relocatable device code under nvcc (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h and
// primitives.h include theirs first, as the bridge does integral_types.h; the
// kernels' header brings the runtime in through felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "primitives.h"

#include "coulomb_gradient_bridge.h"
#include "pair_class_bridge.h"

#include "kernels/gjab.h"

namespace {

// The largest momentum the emitted table covers: J's limit.
constexpr int32_t kGjabMaxL = static_cast<int32_t>(std::extent_v<decltype(gjab_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(gjab_kernels), 1> == kGjabMaxL + 1,
              "kernels/'s [la][lb] table is not square: rerun "
              "tools/emit/coulomb_kernels.py");

static_assert(kGjabMaxL == free_breakfast::device::kCoulombMaxMomentum,
              "kernels/'s table and kCoulombMaxMomentum disagree: rerun "
              "tools/emit/coulomb_kernels.py");

// Record r's bra atom into atom_a[r] and its ket atom into atom_b[r], each the
// atom of the shell's first Cartesian AO.
__global__ void coulomb_record_atoms(const int32_t n_rec, const int32_t *off_a,
                                     const int32_t *off_b, const int32_t *ao_atom, int32_t *atom_a,
                                     int32_t *atom_b) {
  const auto r = static_cast<int32_t>((blockIdx.x * blockDim.x) + threadIdx.x);
  if (r < n_rec) {
    atom_a[r] = ao_atom[off_a[r]];
    atom_b[r] = ao_atom[off_b[r]];
  }
}

// One thread per record, kFourWarps to a block; a whole number of warps keeps
// every block's lanes in step.
uint32_t coulomb_record_blocks(const int32_t n_rec) {
  return free_breakfast::math::idivup(static_cast<uint32_t>(n_rec), free_breakfast::kFourWarps);
}

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t coulomb_gradient_class(const wwr::wwrStream_t stream,
                                       const PrimitivePairClassView &cls, const int32_t *rows,
                                       const int32_t *ao_atom, int32_t *atoms, const int32_t n_ao,
                                       const int32_t n_atom, const double *jp, const double *m,
                                       double *g) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kGjabMaxL || cls.n_rec < 1 || n_atom < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  const uint32_t blocks = coulomb_record_blocks(cls.n_rec);
  coulomb_record_atoms<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, cls.off_a, cls.off_b, ao_atom,
                                                          atoms, atoms + cls.n_rec);
  if (const wwr::wwrError_t err = wwr::wwrGetLastError(); err != wwr::wwrSuccess) {
    return err;
  }
  gjab_kernels[cls.la][cls.lb]<<<blocks, kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, n_atom, cls.n_rec, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x,
      cls.pb_y, cls.pb_z, cls.s, cls.ea, cls.eb, cls.off_a, cls.off_b, rows, atoms,
      atoms + cls.n_rec, jp, m, g);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
