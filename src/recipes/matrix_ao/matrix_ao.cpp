/// @file
/// @brief upload_spherical_transform_matrix's upload of U, and MatrixAOPair's cartesian_to_spherical /
/// spherical_to_cartesian as gemmwich's two OpX forms over it.

module free_breakfast.recipes.matrix_ao;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.blas;

#include "diagnostics.h"

namespace free_breakfast {

Result<std::shared_ptr<const Buffers::device<double>>>
upload_spherical_transform_matrix(DeviceResources &res, const BasisSet &basis) {
  const uint32_t nao_cart = basis.nao_cart();
  const uint32_t nao_sphe = basis.nao_sphe();
  if (nao_cart == nao_sphe) {
    return std::shared_ptr<const Buffers::device<double>>{};
  }

  const std::vector<double> staging = spherical_transform_matrix(basis);
  auto u = std::make_shared<Buffers::device<double>>(staging.size(), res.device());
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(u->data(), staging.data(), sizeof(double) * staging.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // The staging vector dies at return; a pageable source does not promise the
  // copy has finished reading it until the stream does.
  FBF_TRY(gpu_status(wwr::wwrStreamSynchronize(stream)));
  return std::shared_ptr<const Buffers::device<double>>{std::move(u)};
}

// The scratch a pair over basis needs, after checking u fits it: nullptr
// exactly for an identity U, and nao_cart x nao_sphe otherwise.
size_t matrix_ao_pair_work_count(const BasisSet &basis,
                                 const std::shared_ptr<const Buffers::device<double>> &u) {
  const size_t count = static_cast<size_t>(basis.nao_cart()) * basis.nao_sphe();
  if (basis.nao_cart() == basis.nao_sphe()) {
    if (u != nullptr) {
      check_or_abort(err_config("MatrixAOPair: the basis has no shell above p, so U is the "
                                "identity and must be null"));
    }
    return 0;
  }
  if (u == nullptr || u->num_elements() != count) {
    check_or_abort(err_config("MatrixAOPair: U does not fit the basis ({} x {})", basis.nao_cart(),
                              basis.nao_sphe()));
  }
  return count;
}

MatrixAOPair::MatrixAOPair(DeviceResources &res, const BasisSet &basis,
                           std::shared_ptr<const Buffers::device<double>> u)
    : cartesian_(res, basis), spherical_(res, basis), u_(std::move(u)),
      work_(matrix_ao_pair_work_count(basis, u_), res.device()) {}

Status MatrixAOPair::cartesian_to_spherical(DeviceResources &res) {
  if (res.dev_idx() != dev_idx()) {
    return err_config(
        "MatrixAOPair::cartesian_to_spherical: pair is on device {}, but the call passed device {}",
        dev_idx(), res.dev_idx());
  }
  if (spherical_.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  if (u_ == nullptr) { // U is the identity: no shell above p
    return gpu_status(wwr::wwrMemcpyAsync(spherical_.data(), cartesian_.data(),
                                          sizeof(double) * cartesian_.num_elements(),
                                          wwr::wwrMemcpyDeviceToDevice, res.stream()));
  }
  // OpX = N: X = U stored nao_cart x nao_sphe, so n = nao_cart and
  // m = nao_sphe.
  return gpu_status(gemmwich(res.blas(), static_cast<int32_t>(spherical_.n()),
                             static_cast<int32_t>(cartesian_.n()), u_->data(), cartesian_.data(),
                             work_.data(), spherical_.data()));
}

Status MatrixAOPair::spherical_to_cartesian(DeviceResources &res) {
  if (res.dev_idx() != dev_idx()) {
    return err_config(
        "MatrixAOPair::spherical_to_cartesian: pair is on device {}, but the call passed device {}",
        dev_idx(), res.dev_idx());
  }
  if (cartesian_.n() == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  if (u_ == nullptr) { // U is the identity: no shell above p
    return gpu_status(wwr::wwrMemcpyAsync(cartesian_.data(), spherical_.data(),
                                          sizeof(double) * spherical_.num_elements(),
                                          wwr::wwrMemcpyDeviceToDevice, res.stream()));
  }
  // OpX = T: OpX(X) = U^T is nao_sphe x nao_cart, the same buffer read as
  // stored m x n with m = nao_cart and n = nao_sphe.
  return gpu_status(gemmwich<wwr::WWRBLAS_OP_T>(
      res.blas(), static_cast<int32_t>(cartesian_.n()), static_cast<int32_t>(spherical_.n()),
      u_->data(), spherical_.data(), work_.data(), cartesian_.data()));
}

} // namespace free_breakfast
