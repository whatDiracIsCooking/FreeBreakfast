/// @file
/// @brief upload_primitive_pairs and overlap.

module;

// The kernel launcher, defined in overlap_launch.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "overlap_bridge.h"

module free_breakfast.integrals.one_electron;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
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

Status overlap(DeviceResources &res, const std::span<const DevicePrimitivePairClass> classes,
               CartesianMatrixAO &s) {
  if (res.dev_idx() != s.dev_idx()) {
    return err_config("overlap: S is on device {}, but the call passed device {}", s.dev_idx(),
                      res.dev_idx());
  }
  const uint32_t n = s.n();
  if (n == 0) {
    return {};
  }
  // The generated kernels index S as a 32-bit m * n_ao + n.
  if (static_cast<uint64_t>(n) * n > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("overlap: {} Cartesian AOs is too many for the kernels' 32-bit S index", n);
  }
  for (const DevicePrimitivePairClass &cls : classes) {
    if (cls.la < 0 || cls.lb < cls.la || cls.lb > device::kOverlapMaxAngularMomentum) {
      return err_config("overlap: no kernel for class ({}, {}); the kernels cover la <= lb <= {}",
                        cls.la, cls.lb, device::kOverlapMaxAngularMomentum);
    }
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The kernels only add, so S starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(s.data(), 0, sizeof(double) * s.num_elements(), stream)));
  for (const DevicePrimitivePairClass &cls : classes) {
    FBF_TRY(gpu_status(static_cast<wwr::wwrError_t>(device::overlap_class(
        stream, cls.la, cls.lb, cls.n_rec, static_cast<int32_t>(n), cls.zeta.data(), cls.xpa.data(),
        cls.xpb.data(), cls.s.data(), cls.off_a.data(), cls.off_b.data(), s.data()))));
  }
  return {};
}

} // namespace free_breakfast
