/// @file
/// @brief The V build, and the add-only form core_hamiltonian uses.

module;

// The kernel launcher, defined in potential_en_launch.cu: declared in the GMF so
// it keeps external linkage (see the header).
#include "potential_en_bridge.h"

module free_breakfast.integrals.one_electron.potential_en;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// Every check passed: launch the kernel of every class into m, on res's device,
// each on its own stream between a fork from and a join into res.stream().
Status launch_potential_en(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
                           const Molecule &molecule, CartesianMatrixAO &m) {
  const NucleiView nuclei_view = molecule.view();
  const auto n_ao = static_cast<int32_t>(m.n());
  return gpu_status(launch_each(classes, res.stream(), [&](const PrimitivePairClass &cls) {
    return device::potential_en_class(cls.device_stream()->stream(), cls.view(), nuclei_view, n_ao,
                                      m.data());
  }));
}

Status nuclear_attraction(DeviceResources &res, const std::span<const PrimitivePairClass> classes,
                          const Molecule &molecule, CartesianMatrixAO &v) {
  FBF_TRY(check_one_electron_build(res, classes, v, "nuclear_attraction"));
  FBF_TRY(check_nuclei(res, molecule, "nuclear_attraction"));
  if (v.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  // The kernels only add, so the build starts from zero.
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(v.data(), 0, sizeof(double) * v.num_elements(), res.stream())));
  return launch_potential_en(res, classes, molecule, v);
}

Status add_nuclear_attraction(DeviceResources &res,
                              const std::span<const PrimitivePairClass> classes,
                              const Molecule &molecule, CartesianMatrixAO &m) {
  FBF_TRY(check_one_electron_build(res, classes, m, "add_nuclear_attraction"));
  FBF_TRY(check_nuclei(res, molecule, "add_nuclear_attraction"));
  if (m.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return launch_potential_en(res, classes, molecule, m);
}

} // namespace free_breakfast
