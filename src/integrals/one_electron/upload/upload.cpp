/// @file
/// @brief upload_primitive_pairs, upload_nuclei, check_one_electron_build and
/// check_nuclei.

module free_breakfast.integrals.one_electron.upload;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.linalg.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// Copy a host vector into a fresh device buffer of the same length, on stream.
template <typename T>
Result<Buffers::device<T>> upload_pair_field(DeviceResources &res, const std::vector<T> &host) {
  Buffers::device<T> out(host.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.data(), host.data(), sizeof(T) * host.size(),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  return out;
}

Result<std::vector<DevicePrimitivePairClass>>
upload_primitive_pairs(DeviceResources &res, const std::span<const PrimitivePairClass> classes) {
  const ScopedDeviceIndex on{res.dev_idx()};
  std::vector<DevicePrimitivePairClass> out;
  out.reserve(classes.size());
  for (const PrimitivePairClass &cls : classes) {
    if (cls.size() == 0 || cls.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
      return err_config("upload_primitive_pairs: class ({}, {}) has {} records", cls.la, cls.lb,
                        cls.size());
    }
    auto zeta = FBF_TRY(upload_pair_field(res, cls.zeta));
    auto p = FBF_TRY(upload_pair_field(res, cls.p));
    auto xpa = FBF_TRY(upload_pair_field(res, cls.xpa));
    auto xpb = FBF_TRY(upload_pair_field(res, cls.xpb));
    auto s = FBF_TRY(upload_pair_field(res, cls.s));
    auto eb = FBF_TRY(upload_pair_field(res, cls.eb));
    auto off_a = FBF_TRY(upload_pair_field(res, cls.off_a));
    auto off_b = FBF_TRY(upload_pair_field(res, cls.off_b));
    out.push_back(DevicePrimitivePairClass{
        .la = cls.la,
        .lb = cls.lb,
        .n_rec = static_cast<int32_t>(cls.size()),
        .zeta = std::move(zeta),
        .p = std::move(p),
        .xpa = std::move(xpa),
        .xpb = std::move(xpb),
        .s = std::move(s),
        .eb = std::move(eb),
        .off_a = std::move(off_a),
        .off_b = std::move(off_b),
    });
  }
  // A pageable source does not promise the copy has finished reading it until
  // the stream does, and the caller may release classes on return.
  FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(res.stream())));
  return out;
}

Result<DeviceNuclei> upload_nuclei(DeviceResources &res, const Molecule &molecule) {
  const std::vector<Atom> &atoms = molecule.atoms();
  if (atoms.empty() || atoms.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("upload_nuclei: a molecule of {} atoms", atoms.size());
  }
  std::vector<double> z;
  std::vector<double> c;
  z.reserve(atoms.size());
  c.reserve(3 * atoms.size());
  for (const Atom &atom : atoms) {
    z.push_back(static_cast<double>(nuclear_charge(atom)));
    c.insert(c.end(), {atom.position.x, atom.position.y, atom.position.z});
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  auto z_dev = FBF_TRY(upload_pair_field(res, z));
  auto c_dev = FBF_TRY(upload_pair_field(res, c));
  // As upload_primitive_pairs: z and c are pageable and die on return.
  FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(res.stream())));
  return DeviceNuclei{
      .dev_idx = res.dev_idx(),
      .n_nuc = static_cast<int32_t>(atoms.size()),
      .z = std::move(z_dev),
      .c = std::move(c_dev),
  };
}

Status check_one_electron_build(const DeviceResources &res,
                                const std::span<const DevicePrimitivePairClass> classes,
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
  for (const DevicePrimitivePairClass &cls : classes) {
    if (cls.la < 0 || cls.lb < cls.la || cls.lb > kMaxAngularMomentum) {
      return err_config("{}: no kernel for class ({}, {}); the kernels cover la <= lb <= {}", what,
                        cls.la, cls.lb, kMaxAngularMomentum);
    }
  }
  return {};
}

Status check_nuclei(const DeviceResources &res, const DeviceNuclei &nuclei,
                    const std::string_view what) {
  if (res.dev_idx() != nuclei.dev_idx) {
    return err_config("{}: the nuclei are on device {}, but the call passed device {}", what,
                      nuclei.dev_idx, res.dev_idx());
  }
  return {};
}

} // namespace free_breakfast
