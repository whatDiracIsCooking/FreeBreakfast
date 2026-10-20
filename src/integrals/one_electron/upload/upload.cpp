/// @file
/// @brief upload_nuclei, check_one_electron_build and check_nuclei.

module free_breakfast.integrals.one_electron.upload;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.mirrored_buffer;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

Result<DeviceNuclei> upload_nuclei(DeviceResources &res, const Molecule &molecule) {
  const std::vector<Atom> &atoms = molecule.atoms();
  if (atoms.empty() || atoms.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("upload_nuclei: a molecule of {} atoms", atoms.size());
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto ds = std::make_shared<DeviceStream>(res.device());
  DeviceNuclei out{
      .n_nuc = static_cast<int32_t>(atoms.size()),
      .z = MirroredBuffer<double>(atoms.size(), ds),
      .c = MirroredBuffer<double>(3 * atoms.size(), ds),
  };
  double *z = out.z.pinned().data();
  double *c = out.c.pinned().data();
  for (size_t a = 0; a < atoms.size(); ++a) {
    z[a] = static_cast<double>(nuclear_charge(atoms[a]));
    c[(3 * a) + 0] = atoms[a].position.x;
    c[(3 * a) + 1] = atoms[a].position.y;
    c[(3 * a) + 2] = atoms[a].position.z;
  }
  FBF_TRY(gpu_status(out.z.to_device()));
  // Both copies are on one stream, so this one sync covers z's as well.
  FBF_TRY(gpu_status(out.c.to_device_sync()));
  return out;
}

Status check_one_electron_build(const DeviceResources &res,
                                const std::span<const PrimitivePairClass> classes,
                                const CartesianMatrixAO &m, const std::string_view what) {
  if (res.dev_idx() != m.dev_idx()) {
    return err_config("{}: the matrix is on device {}, but the call passed device {}", what,
                      m.dev_idx(), res.dev_idx());
  }
  const uint32_t n = m.n();
  // The generated kernels index the matrix as a 32-bit m * n_ao + n.
  if (static_cast<uint64_t>(n) * n > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("{}: {} Cartesian AOs is too many for the kernels' 32-bit index", what, n);
  }
  for (const PrimitivePairClass &cls : classes) {
    if (const int32_t dev = cls.device_stream()->dev_idx(); dev != res.dev_idx()) {
      return err_config("{}: class ({}, {}) is on device {}, but the call passed device {}", what,
                        cls.la, cls.lb, dev, res.dev_idx());
    }
    if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum) {
      return err_config("{}: no kernel for class ({}, {}); the kernels cover la <= lb <= {}", what,
                        cls.la, cls.lb, kMaxAngularMomentum);
    }
  }
  return {};
}

Status check_nuclei(const DeviceResources &res, const DeviceNuclei &nuclei,
                    const std::string_view what) {
  if (const int32_t dev = nuclei.device_stream()->dev_idx(); dev != res.dev_idx()) {
    return err_config("{}: the nuclei are on device {}, but the call passed device {}", what, dev,
                      res.dev_idx());
  }
  return {};
}

} // namespace free_breakfast
