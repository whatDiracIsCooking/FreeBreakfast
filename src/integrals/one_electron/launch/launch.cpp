/// @file
/// @brief check_one_electron_build and check_nuclei.

module free_breakfast.integrals.one_electron.launch;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.recipes.matrix_ao;

namespace free_breakfast {

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
    if (cls.la < 0 || cls.lb < cls.la || cls.lb > kOneElectronMaxL) {
      return err_config("{}: no kernel for class ({}, {}); the kernels cover la <= lb <= {}", what,
                        cls.la, cls.lb, kOneElectronMaxL);
    }
  }
  return {};
}

Status check_nuclei(const DeviceResources &res, const Molecule &molecule,
                    const std::string_view what) {
  if (const int32_t dev = molecule.device_stream()->dev_idx(); dev != res.dev_idx()) {
    return err_config("{}: the nuclei are on device {}, but the call passed device {}", what, dev,
                      res.dev_idx());
  }
  return {};
}

} // namespace free_breakfast
