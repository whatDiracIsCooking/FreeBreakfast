/// @file
/// @brief The mixed S build.

module;

// The kernel launcher, defined in overlap_mixed_launch.cu: declared in the GMF
// so it keeps external linkage (see the header).
#include "overlap_mixed_bridge.h"

module free_breakfast.integrals.one_electron.overlap_mixed;

import std;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.one_electron.launch;

#include "diagnostics.h"

namespace free_breakfast {

// InvalidConfig naming the first shell of basis above kMaxAngularMomentum, the
// limit the kernels are generated to; which names the basis in the message.
Status check_shells(const BasisSet &basis, const std::string_view which) {
  for (uint32_t i = 0; i < basis.nshells(); ++i) {
    if (const int32_t l = basis.shells()[i].angular_momentum(); l > kMaxAngularMomentum) {
      return err_config("overlap_mixed: {} shell {} has l = {}; the kernels cover l <= {}", which,
                        i, l, kMaxAngularMomentum);
    }
  }
  return {};
}

Status overlap_mixed(DeviceResources &res, const BasisSet &rows, const BasisSet &cols,
                     Buffers::device<double> &s) {
  const uint64_t n_rows = rows.nao_cart();
  const uint64_t n_cols = cols.nao_cart();
  if (s.num_elements() != n_rows * n_cols) {
    return err_config("overlap_mixed: the buffer holds {} doubles, but S is {} x {}",
                      s.num_elements(), n_rows, n_cols);
  }
  // The generated kernels index the matrix as a 32-bit m * n_b + n.
  if (n_rows * n_cols > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("overlap_mixed: {} x {} Cartesian AOs is too many for the kernels' 32-bit "
                      "index",
                      n_rows, n_cols);
  }
  FBF_TRY(check_shells(rows, "row"));
  FBF_TRY(check_shells(cols, "column"));
  if (s.num_elements() == 0) {
    return {};
  }
  const std::vector<PrimitivePairClass> classes =
      FBF_TRY(mixed_primitive_pair_classes(res, rows, cols));
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The kernels only add, so the build starts from zero.
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(s.data(), 0, sizeof(double) * s.num_elements(), stream)));
  // The records are this call's: each class's are freed on its own stream,
  // after its kernel, when classes goes out of scope.
  return gpu_status(launch_each(classes, stream, [&](const PrimitivePairClass &cls) {
    return device::overlap_mixed_class(cls.device_stream()->stream(), cls.view(),
                                       static_cast<int32_t>(n_rows), static_cast<int32_t>(n_cols),
                                       s.data());
  }));
}

Status overlap_mixed_spherical(DeviceResources &res, const BasisSet &rows, const BasisSet &cols,
                               Buffers::device<double> &s) {
  const auto n_rows = static_cast<int32_t>(rows.nao_sphe());
  const auto n_cols = static_cast<int32_t>(cols.nao_sphe());
  const auto n_rows_cart = static_cast<int32_t>(rows.nao_cart());
  const auto n_cols_cart = static_cast<int32_t>(cols.nao_cart());
  if (s.num_elements() != static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(n_cols)) {
    return err_config("overlap_mixed_spherical: the buffer holds {} doubles, but S is {} x {}",
                      s.num_elements(), n_rows, n_cols);
  }
  Buffers::device<double> cart(static_cast<size_t>(n_rows_cart) * n_cols_cart, res.device());
  FBF_TRY(overlap_mixed(res, rows, cols, cart));
  if (s.num_elements() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const std::vector<double> u_rows_host = spherical_transform_matrix(rows);
  const std::vector<double> u_cols_host = spherical_transform_matrix(cols);
  Buffers::device<double> u_rows(u_rows_host.size(), res.device());
  Buffers::device<double> u_cols(u_cols_host.size(), res.device());
  Buffers::device<double> t(static_cast<size_t>(n_rows_cart) * n_cols, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(u_rows.data(), u_rows_host.data(),
                                         u_rows_host.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(u_cols.data(), u_cols_host.data(),
                                         u_cols_host.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // T = S_cart U_cols, then S = U_rows^T T.
  FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n_rows_cart, n_cols, n_cols_cart,
      &kOne<double>, cart.data(), n_rows_cart, u_cols.data(), n_cols_cart, &kZero<double>, t.data(),
      n_rows_cart)));
  FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(res.blas().get(), wwr::WWRBLAS_OP_T,
                                                wwr::WWRBLAS_OP_N, n_rows, n_cols, n_rows_cart,
                                                &kOne<double>, u_rows.data(), n_rows_cart, t.data(),
                                                n_rows_cart, &kZero<double>, s.data(), n_rows)));
  // The uploads read pageable host memory that goes with this frame.
  return res.sync();
}

} // namespace free_breakfast
