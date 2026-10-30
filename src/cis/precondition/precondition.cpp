/// @file
/// @brief The diagonal's and the preconditioner's launches, the guess's host
/// select, and the clusters' grouping, blocks and correction.

module;

// The launchers, defined in cis_precondition.cu: declared in the GMF so they
// keep external linkage (see the header).
#include "cis_precondition_bridge.h"

module free_breakfast.cis.precondition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.blas;        // gemm
import free_breakfast.recipes.davidson;    // DavidsonPreconditioner
import free_breakfast.recipes.diagonalize; // diagonalize

#include "diagnostics.h"

namespace free_breakfast::cis {

static_assert(DavidsonPreconditioner<Preconditioner, double>);

Status diagonal(DeviceResources &res, const int32_t n_occ, const int32_t n_virt,
                const double *const orbital_energies, double *const delta) {
  if (n_occ < 1 || n_virt < 1 ||
      static_cast<int64_t>(n_occ) * n_virt > std::numeric_limits<int32_t>::max()) {
    return err_config("cis::diagonal: needs n_occ >= 1, n_virt >= 1 and n_occ * n_virt to fit an "
                      "int32_t (n_occ = {}, n_virt = {})",
                      n_occ, n_virt);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  return gpu_status(device::diagonal(res.stream(), n_occ, n_virt, orbital_energies, delta));
}

Status guess(DeviceResources &res, const int32_t n_ov, const double *const delta,
             const int32_t guess_count, double *const vectors) {
  if (guess_count < 1 || guess_count > n_ov) {
    return err_config("cis::guess: needs 1 <= guess_count <= n_ov (guess_count = {}, n_ov = {})",
                      guess_count, n_ov);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<size_t>(n_ov);
  const auto count = static_cast<size_t>(guess_count);

  // The select is a host partial sort: it runs once per solve, over a vector
  // a few MB at most, and a host comparator on (Δε, index) is what makes the
  // tie order exact and the result deterministic.
  std::vector<double> host(n);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), delta, n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<int32_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + guess_count, order.end(),
                    [&host](const int32_t l, const int32_t r) {
                      return std::tie(host[static_cast<size_t>(l)], l) <
                             std::tie(host[static_cast<size_t>(r)], r);
                    });

  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(vectors, 0, n * count * sizeof(double), stream)));
  // kOne is static storage, so the pageable source outlives the copy and
  // nothing here needs a sync.
  for (size_t g = 0; g < count; ++g) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors + (g * n) + static_cast<size_t>(order[g]),
                                           &kOne<double>, sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
  }
  return {};
}

Preconditioner::Preconditioner(DeviceResources &res, const int32_t n_ov, const double *const delta,
                               const int32_t max_columns)
    : dev_idx_{res.dev_idx()}, n_ov_{n_ov}, delta_{delta},
      theta_(static_cast<size_t>(std::max(max_columns, 0)), res.device()), index_(0, res.device()),
      lambda_(0, res.device()), vectors_(0, res.device()) {}

Status Preconditioner::operator()(DeviceResources &res, const std::span<const double> theta,
                                  const double *const /*ritz_vectors*/,
                                  const double *const residual, double *const correction) const {
  if (res.dev_idx() != dev_idx_) {
    return err_config("cis::Preconditioner: built on device {}, called on device {}", dev_idx_,
                      res.dev_idx());
  }
  if (theta.size() > theta_.num_elements()) {
    return err_config("cis::Preconditioner: {} columns, more than the {} it was built for",
                      theta.size(), theta_.num_elements());
  }
  if (theta.empty()) {
    return {};
  }
  const ScopedDeviceIndex on{dev_idx_};
  const wwr::wwrStream_t stream = res.stream();
  const auto columns = static_cast<int32_t>(theta.size());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta_.data(), theta.data(), theta.size_bytes(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  if (covered_ == 0) {
    FBF_TRY(gpu_status(
        device::precondition(stream, n_ov_, columns, theta_.data(), delta_, residual, correction)));
  } else {
    // The covered rows of the residual first, as the diagonal correction may
    // overwrite them in place; then the diagonal correction everywhere, and
    // the covered rows replaced by V clamp(θ − λ)⁻¹ Vᵀ r, V block-diagonal.
    // cluster grew the workspace to these two blocks.
    const auto block = static_cast<size_t>(covered_) * theta.size();
    const auto [gathered, projected] = res.device_regions<double>(block, block);
    const wwr::wwrblasHandle_t blas = res.blas().get();
    FBF_TRY(gpu_status(
        device::gather_rows(stream, n_ov_, covered_, columns, index_.data(), residual, gathered)));
    FBF_TRY(gpu_status(
        device::precondition(stream, n_ov_, columns, theta_.data(), delta_, residual, correction)));
    FBF_TRY(
        gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, covered_, columns, covered_, &kOne<double>,
                                           vectors_.data(), gathered, &kZero<double>, projected)));
    FBF_TRY(gpu_status(device::precondition(stream, covered_, columns, theta_.data(),
                                            lambda_.data(), projected, projected)));
    FBF_TRY(gpu_status(gemm(blas, covered_, columns, covered_, &kOne<double>, vectors_.data(),
                            projected, &kZero<double>, gathered)));
    FBF_TRY(gpu_status(device::scatter_rows(stream, n_ov_, covered_, columns, index_.data(),
                                            gathered, correction)));
  }
  // theta is the caller's pageable memory, which a pageable upload does not
  // promise to have read by return (docs/style.md "Host-device copies").
  return res.sync();
}

Result<Buffers::device<double>> Preconditioner::group(DeviceResources &res, const int32_t scope,
                                                      const double gap) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("cis::Preconditioner: built on device {}, clustered on device {}", dev_idx_,
                      res.dev_idx());
  }
  if (scope < 1 || scope > n_ov_ || !(gap >= 0.0)) {
    return err_config("cis::Preconditioner::cluster: needs 1 <= scope <= n_ov and gap >= 0 "
                      "(scope = {}, n_ov = {}, gap = {})",
                      scope, n_ov_, gap);
  }
  covered_ = 0;
  sizes_.clear();
  if (gap == 0.0) {
    return Buffers::device<double>(0, res.device());
  }
  const ScopedDeviceIndex on{dev_idx_};
  const wwr::wwrStream_t stream = res.stream();
  const auto n = static_cast<size_t>(n_ov_);

  // The scope lowest Δε in the guess's order, (Δε, index), on the host: once
  // per solve, as the guess's select is.
  std::vector<double> host(n);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), delta_, n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<int32_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + scope, order.end(),
                    [&host](const int32_t l, const int32_t r) {
                      return std::tie(host[static_cast<size_t>(l)], l) <
                             std::tie(host[static_cast<size_t>(r)], r);
                    });

  // Single linkage over the sorted run: a neighbour within gap joins the
  // cluster, and a cluster of one, which the diagonal already solves exactly,
  // is dropped.
  std::vector<int32_t> index;
  std::vector<int32_t> sizes;
  for (int32_t first = 0; first < scope;) {
    int32_t last = first + 1;
    while (last < scope &&
           host[static_cast<size_t>(order[static_cast<size_t>(last)])] -
                   host[static_cast<size_t>(order[static_cast<size_t>(last - 1)])] <=
               gap) {
      ++last;
    }
    if (last - first >= 2) {
      index.insert(index.end(), order.begin() + first, order.begin() + last);
      sizes.push_back(last - first);
    }
    first = last;
  }
  if (index.empty()) {
    return Buffers::device<double>(0, res.device());
  }

  const auto covered = static_cast<int32_t>(index.size());
  Buffers::device<int32_t> index_d(index.size(), res.device());
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(index_d.data(), index.data(), index.size() * sizeof(int32_t),
                                     wwr::wwrMemcpyHostToDevice, stream)));
  // Column c is 1 on covered entry c: a memset and a copy of kOne per column,
  // as the guess's are.
  Buffers::device<double> units(n * index.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(units.data(), 0, units.size_bytes(), stream)));
  for (size_t c = 0; c < index.size(); ++c) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(units.data() + (c * n) + static_cast<size_t>(index[c]),
                                           &kOne<double>, sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
  }
  // index is pageable and goes out of scope here.
  FBF_TRY(res.sync());
  index_ = std::move(index_d);
  sizes_ = std::move(sizes);
  covered_ = covered;
  return units;
}

Status Preconditioner::solve_blocks(DeviceResources &res, const double *const applied) {
  const ScopedDeviceIndex on{dev_idx_};
  const wwr::wwrStream_t stream = res.stream();
  const int32_t covered = std::exchange(covered_, 0);
  const auto k_all = static_cast<size_t>(covered);

  // A's rows of the covered entries in the sigma of their unit vectors: the
  // covered × covered A_cc, column-major, of which only the diagonal blocks
  // are read.
  std::vector<double> a(k_all * k_all);
  {
    Buffers::device<double> a_d(a.size(), res.device());
    FBF_TRY(gpu_status(
        device::gather_rows(stream, n_ov_, covered, covered, index_.data(), applied, a_d.data())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a.data(), a_d.data(), a.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
  }

  // Each block symmetrized, as its columns come from separate applies, and
  // diagonalized; its eigenvectors placed on V's diagonal, zeros elsewhere.
  std::vector<double> v(k_all * k_all, 0.0);
  std::vector<double> lambda(k_all);
  size_t at = 0;
  for (const int32_t size : sizes_) {
    const auto k = static_cast<size_t>(size);
    std::vector<double> block(k * k);
    for (size_t c = 0; c < k; ++c) {
      for (size_t r = 0; r < k; ++r) {
        block[r + (c * k)] =
            0.5 * (a[(at + r) + ((at + c) * k_all)] + a[(at + c) + ((at + r) * k_all)]);
      }
    }
    Buffers::device<double> block_d(block.size(), res.device());
    Buffers::device<double> w_d(k, res.device());
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(block_d.data(), block.data(), block.size() * sizeof(double),
                                       wwr::wwrMemcpyHostToDevice, stream)));
    FBF_TRY(diagonalize<double>(res, size, block_d.data(), w_d.data()));
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(block.data(), block_d.data(), block.size() * sizeof(double),
                                       wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(lambda.data() + at, w_d.data(), k * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t c = 0; c < k; ++c) {
      std::copy_n(block.begin() + static_cast<std::ptrdiff_t>(c * k), k,
                  v.begin() + static_cast<std::ptrdiff_t>((at + c) * k_all + at));
    }
    at += k;
  }

  Buffers::device<double> lambda_d(k_all, res.device());
  Buffers::device<double> vectors_d(v.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(lambda_d.data(), lambda.data(), k_all * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(vectors_d.data(), v.data(), v.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(res.sync());
  lambda_ = std::move(lambda_d);
  vectors_ = std::move(vectors_d);
  // A call's two scratch blocks, so the workspace is grown here and a call
  // allocates nothing.
  const size_t scratch = k_all * theta_.num_elements();
  std::ignore = res.device_regions<double>(scratch, scratch);
  covered_ = covered;
  return {};
}

} // namespace free_breakfast::cis
