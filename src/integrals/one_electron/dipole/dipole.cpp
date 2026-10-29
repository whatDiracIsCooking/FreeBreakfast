/// @file
/// @brief The dipole build.

module;

// The kernel launcher, defined in dipole_launch.cu: declared in the GMF so it
// keeps external linkage (see overlap_bridge.h).
#include "dipole_bridge.h"

module free_breakfast.integrals.one_electron.dipole;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

static_assert(kDipoleMaxL < kOneElectronMaxL,
              "the dipole is the one family below f: drop its own check when it reaches f");

Status dipole(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
              CartesianMatrixAO &mu_x, CartesianMatrixAO &mu_y, CartesianMatrixAO &mu_z) {
  FBF_TRY(check_one_electron_build(res, classes, mu_x, "dipole"));
  const uint32_t n = mu_x.n();
  for (const CartesianMatrixAO *m : {&mu_y, &mu_z}) {
    if (m->dev_idx() != mu_x.dev_idx() || m->n() != n) {
      return err_config("dipole: the matrices differ: mu_x is {} square on device {}, another {} "
                        "square on device {}",
                        n, mu_x.dev_idx(), m->n(), m->dev_idx());
    }
  }
  for (const PrimitivePairClass &cls : classes) {
    if (cls.lb > kDipoleMaxL) {
      return err_config("dipole: no kernel for class ({}, {}); the dipole kernels cover la <= lb "
                        "<= {} (FelizJueves validates them through (dd)), so a basis with an f "
                        "shell is refused",
                        cls.la, cls.lb, kDipoleMaxL);
    }
  }
  // The kernels index all three components as one 32-bit (c * n + m) * n + n'.
  if (3 * static_cast<uint64_t>(n) * n >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("dipole: {} Cartesian AOs is too many for the kernels' 32-bit index", n);
  }
  if (n == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The kernels write x, y and z one after the other into one array, which three
  // separately allocated matrices are not, so they write into scratch.
  const size_t count = mu_x.num_elements();
  const auto [mu] = res.device_regions<double>(3 * count);
  // The kernels only add, so the build starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(mu, 0, sizeof(double) * 3 * count, stream)));
  const auto n_ao = static_cast<int32_t>(n);
  FBF_TRY(gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    return device::dipole_class(cls.device_stream()->stream(), cls.view(), n_ao, mu);
  })));
  const std::array<CartesianMatrixAO *, 3> out{&mu_x, &mu_y, &mu_z};
  for (size_t c = 0; c < out.size(); ++c) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out[c]->data(), mu + c * count, sizeof(double) * count,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  return {};
}

} // namespace free_breakfast
