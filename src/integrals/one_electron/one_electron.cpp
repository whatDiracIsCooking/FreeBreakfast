/// @file
/// @brief The T, V and H builds.

module;

// The T and V launchers, defined in the *_launch.cu: declared in the GMF so they
// keep external linkage (see the header).
#include "one_electron_bridge.h"

module free_breakfast.integrals.one_electron;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.one_electron.upload;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// What V and H check beyond check_one_electron_build: the nuclei on res's
// device.
Status check_nuclei(const DeviceResources &res, const DeviceNuclei &nuclei,
                    const std::string_view what) {
  if (res.dev_idx() != nuclei.dev_idx) {
    return err_config("{}: the nuclei are on device {}, but the call passed device {}", what,
                      nuclei.dev_idx, res.dev_idx());
  }
  return {};
}

// The kernels only add, so every build starts from zero.
Status zero(const wwr::wwrStream_t stream, CartesianMatrixAO &m) {
  return gpu_status(wwr::wwrMemsetAsync(m.data(), 0, sizeof(double) * m.num_elements(), stream));
}

// Launch the kinetic kernel of every class into m, without zeroing it.
Status add_kinetic(const wwr::wwrStream_t stream,
                   const std::span<const DevicePrimitivePairClass> classes, CartesianMatrixAO &m) {
  for (const DevicePrimitivePairClass &cls : classes) {
    FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(
        device::kinetic_class(stream, cls.view(), static_cast<int32_t>(m.n()), m.data()))));
  }
  return {};
}

// Launch the nuclear-attraction kernel of every class into m, without zeroing
// it.
Status add_nuclear(const wwr::wwrStream_t stream,
                   const std::span<const DevicePrimitivePairClass> classes,
                   const DeviceNuclei &nuclei, CartesianMatrixAO &m) {
  const NucleiView nuclei_view = nuclei.view();
  for (const DevicePrimitivePairClass &cls : classes) {
    FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(device::nuclear_class(
        stream, cls.view(), nuclei_view, static_cast<int32_t>(m.n()), m.data()))));
  }
  return {};
}

Status kinetic(DeviceResources &res, const std::span<const DevicePrimitivePairClass> classes,
               CartesianMatrixAO &t) {
  FBF_TRY(check_one_electron_build(res, classes, t, "kinetic"));
  if (t.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(zero(res.stream(), t));
  return add_kinetic(res.stream(), classes, t);
}

Status nuclear_attraction(DeviceResources &res,
                          const std::span<const DevicePrimitivePairClass> classes,
                          const DeviceNuclei &nuclei, CartesianMatrixAO &v) {
  FBF_TRY(check_one_electron_build(res, classes, v, "nuclear_attraction"));
  FBF_TRY(check_nuclei(res, nuclei, "nuclear_attraction"));
  if (v.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(zero(res.stream(), v));
  return add_nuclear(res.stream(), classes, nuclei, v);
}

Status core_hamiltonian(DeviceResources &res,
                        const std::span<const DevicePrimitivePairClass> classes,
                        const DeviceNuclei &nuclei, CartesianMatrixAO &h) {
  FBF_TRY(check_one_electron_build(res, classes, h, "core_hamiltonian"));
  FBF_TRY(check_nuclei(res, nuclei, "core_hamiltonian"));
  if (h.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  FBF_TRY(zero(res.stream(), h));
  FBF_TRY(add_kinetic(res.stream(), classes, h));
  return add_nuclear(res.stream(), classes, nuclei, h);
}

} // namespace free_breakfast
