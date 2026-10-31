// The device half of free_breakfast.integrals.one_electron.dipole_gradient: the
// launcher over the generated dipole_gradient<la, lb> kernels, and the gather
// of each record's two atoms they read. One source for both backends -- nvcc
// under CUDA, clang -x hip under HIP.
//
// The generated kernels are sources of this library of their own (kernels/),
// each one explicit specialization of the kernel template. kernels/
// dipole_gradient.h declares the template, every specialization and
// dipole_gradient_kernels, their [la][lb] table, so nothing here names a
// class: a class the table names but no unit defines fails to link. Launching
// a specialization another TU defines wants relocatable device code under nvcc
// (CMakeLists.txt).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). block_params.h,
// constants.h and primitives.h include theirs first, as the bridge does
// integral_types.h; the kernels' header brings the runtime in through
// felizjueves/runtime.h.
#include <type_traits>

#include "block_params.h"
#include "constants.h"
#include "primitives.h"

#include "dipole_gradient_bridge.h"

#include "kernels/dipole_gradient.h"

namespace {

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

// The largest momentum the table covers, its own limit rather than the tree's
// kMaxAngularMomentum, so it moves when the kernels are emitted for a higher l.
constexpr int32_t kMaxL =
    static_cast<int32_t>(std::extent_v<decltype(dipole_gradient_kernels), 0>) - 1;

static_assert(std::extent_v<decltype(dipole_gradient_kernels), 1> == kMaxL + 1,
              "kernels/'s [la][lb] table is not square: rerun "
              "tools/emit/dipole_gradient_kernels.py");

} // namespace

namespace free_breakfast::device {

wwr::wwrError_t dipole_gradient_class(const wwr::wwrStream_t stream,
                                      const PrimitivePairClassView &cls, const int32_t *ao_atom,
                                      int32_t *atoms, const int32_t n_ao, const int32_t n_atom,
                                      const double *m, double *g) {
  if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxL || cls.n_rec < 1 || n_atom < 1) {
    return wwr::wwrErrorInvalidValue;
  }
  // One thread per record, kFourWarps to a block; a whole number of warps keeps
  // every block's lanes in step.
  const uint32_t blocks = math::idivup(static_cast<uint32_t>(cls.n_rec), kFourWarps);
  int32_t *const atom_a = atoms;
  int32_t *const atom_b = atoms + cls.n_rec;
  record_atoms<<<blocks, kFourWarps, 0, stream>>>(cls.n_rec, cls.off_a, cls.off_b, ao_atom, atom_a,
                                                  atom_b);
  if (const wwr::wwrError_t err = wwr::wwrGetLastError(); err != wwr::wwrSuccess) {
    return err;
  }
  // The kernels take the origin C and the charge q at run time; this is
  // <a| r |b>'s derivative about the origin, as dipole's: C = 0, q = 1.
  dipole_gradient_kernels[cls.la][cls.lb]<<<blocks, kFourWarps, 0, stream>>>(
      cls.n_rec, n_ao, n_atom, cls.zeta, cls.pa_x, cls.pa_y, cls.pa_z, cls.pb_x, cls.pb_y, cls.pb_z,
      cls.p_x, cls.p_y, cls.p_z, cls.s, cls.ea, cls.eb, cls.off_a, cls.off_b, atom_a, atom_b, m,
      kZero<double>, kZero<double>, kZero<double>, kOne<double>, g);
  return wwr::wwrGetLastError();
}

} // namespace free_breakfast::device
