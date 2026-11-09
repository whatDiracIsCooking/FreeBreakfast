/// @file
/// @brief The DF second-derivative integrals, contracted: the spherical
/// matrices made Cartesian, the energy builds' staging with each pair's or
/// cell's atoms (and each bra record's exponents), one launch per class into a
/// zeroed H, and H downloaded. The staging is the DF gradients', whose
/// kernels read the same parameter block: both stage through
/// density_fitting.staging.

module;

// The kernel launchers, defined in hessian_launch.cu: declared in the GMF so
// they keep external linkage (see the header).
#include "hessian_bridge.h"

module free_breakfast.integrals.eri.density_fitting.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.primitives;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // wwrMemsetAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.integrals.eri.density_fitting.staging;

#include "diagnostics.h"

namespace free_breakfast {

Result<std::vector<double>> metric_hessian(DeviceResources &res, const BasisSet &aux,
                                           const double *w) {
  FBF_TRY(check_metric_basis(aux, "metric_hessian"));
  const size_t n = aux.nao_cart();
  const size_t n_sph = aux.nao_sphe();
  // The kernels index W as a 32-bit a * n_aux + b.
  if (n * n > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("metric_hessian: {} Cartesian fitting functions is too many for the "
                      "kernels' 32-bit index",
                      n);
  }
  const size_t n_h = 9 * static_cast<size_t>(aux.natoms()) * aux.natoms();
  if (aux.shells().empty()) {
    return std::vector<double>(n_h, 0.0);
  }

  const std::vector<int32_t> rec_begin = record_begins(aux);
  const auto n_rec = static_cast<size_t>(rec_begin.back());
  const MetricPairs pairs = stage_metric_pairs(aux, rec_begin);
  const std::vector<double> staging = stage_metric_records(aux, rec_begin);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto [d_staging, d_pairs, t, w_cart, d_h] =
      res.device_regions<double, int32_t, double, double, double>(
          staging.size(), pairs.fields.size(), n * n_sph, n * n, n_h);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * staging.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_pairs, pairs.fields.data(),
                                         sizeof(int32_t) * pairs.fields.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  const size_t n_records = static_cast<size_t>(device::kMetricRecordFields) * n_rec;
  FBF_TRY(metric_cartesian(res, n, n_sph, d_staging + n_records, w, t, w_cart));
  const auto n32 = static_cast<int32_t>(n);

  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(d_h, 0, sizeof(double) * n_h, stream)));
  for (size_t lp = 0; lp < kMetricMomenta; ++lp) {
    for (size_t lq = lp; lq < kMetricMomenta; ++lq) {
      const size_t c = (lp * kMetricMomenta) + lq;
      if (pairs.count[c] == 0) {
        continue;
      }
      FBF_TRY(gpu_status(device::metric_hessian_class(
          stream, static_cast<int32_t>(lp), static_cast<int32_t>(lq), static_cast<int32_t>(n_rec),
          d_staging, static_cast<int32_t>(pairs.count[c]),
          d_pairs + (static_cast<size_t>(device::kMetricHessianPairFields) * pairs.begin[c]), n32,
          static_cast<int32_t>(aux.natoms()), w_cart, d_h)));
    }
  }
  std::vector<double> h(n_h);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h.data(), d_h, sizeof(double) * n_h,
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what lets the host staging go: a pageable upload has finished
  // reading its source only once the stream has.
  FBF_TRY(res.sync());
  return h;
}

Result<std::vector<double>> three_center_hessian(DeviceResources &res, const BasisSet &basis,
                                                 const std::span<const PrimitivePairClass> classes,
                                                 const BasisSet &aux, const double *gamma) {
  constexpr std::string_view kWho = "three_center_hessian";
  FBF_TRY(check_three_center_bases(basis, aux, kWho));
  const std::vector<Shell> &shells = basis.shells();
  const std::vector<Shell> &aux_shells = aux.shells();
  const size_t n_h = 9 * static_cast<size_t>(basis.natoms()) * basis.natoms();
  if (shells.empty() || aux_shells.empty()) {
    return std::vector<double>(n_h, 0.0);
  }

  const size_t n = basis.nao_cart();
  const size_t n_sph = basis.nao_sphe();
  const size_t n_aux = aux.nao_cart();
  const size_t n_aux_sph = aux.nao_sphe();
  const size_t n_cart = n * n * n_aux;
  // The kernels index Gam as a 32-bit (m n_ao + n) n_aux + P.
  if (n_cart > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return err_unsupported("three_center_hessian: the Cartesian Gam, {} x {} x {} = {} elements "
                           "({} bytes), is past the kernels' 32-bit index",
                           n, n, n_aux, n_cart, sizeof(double) * n_cart);
  }

  // The counts, from the classes' sizes and the bases alone, so the refusal
  // below comes before anything is downloaded or allocated.
  const size_t n_bra_rec = FBF_TRY(bra_record_count(classes, kWho));
  const std::vector<int32_t> aux_rec_begin = record_begins(aux);
  const auto n_aux_rec = static_cast<size_t>(aux_rec_begin.back());
  const size_t n_pairs = shells.size() * (shells.size() + 1) / 2;
  const size_t n_cells = n_pairs * aux_shells.size();
  const size_t n_bra = static_cast<size_t>(device::kThreeCenterHessianBraFields) * n_bra_rec;
  const size_t n_aux_fields = static_cast<size_t>(device::kMetricRecordFields) * n_aux_rec;
  const size_t n_staging = n_bra + n_aux_fields + (n * n_sph) + (n_aux * n_aux_sph);
  const size_t n_cell_fields = static_cast<size_t>(device::kThreeCenterHessianCellFields) * n_cells;
  // Gam's aux index made Cartesian lands in cart, its n index in work, its m
  // index back in cart.
  const size_t n_work = n_aux * n * n_sph;
  const auto region = [](const size_t bytes) {
    return math::align_up(bytes, DeviceResources::kWorkspaceAlignment);
  };
  const size_t need = region(sizeof(double) * n_staging) + region(sizeof(int32_t) * n_cell_fields) +
                      region(sizeof(double) * n_cart) + region(sizeof(double) * n_work) +
                      region(sizeof(double) * n_h);

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  // The carve's size is the shape's alone: compared against the larger of the
  // workspace and the budget, since growing the workspace past the budget
  // aborts (CreatePolicy). #1123 drops the check: a fixed workspace refuses an
  // oversized borrow itself.
  const size_t workspace = res.device_workspace(0).num_elements();
  const size_t budget = FBF_TRY(res.memory_budget());
  if (const size_t available = std::max(workspace, budget); need > available) {
    return err_unsupported("three_center_hessian: over {} Cartesian orbital and {} Cartesian "
                           "fitting functions it needs {} bytes of device scratch (the Cartesian "
                           "Gam alone {} bytes), and {} bytes are available (the larger of the "
                           "workspace's {} and device {}'s budget {})",
                           n, n_aux, need, sizeof(double) * n_cart, available, workspace,
                           res.dev_idx(), budget);
  }

  ThreeCenterBra bra = FBF_TRY(stage_three_center_bra(res, basis, classes, n_bra_rec, kWho));
  const std::vector<double> staging =
      stage_three_center_records(std::move(bra.fields), basis, aux, aux_rec_begin);
  std::array<size_t, kThreeCenterClasses> n_cell{};
  std::array<size_t, kThreeCenterClasses> cell_begin{};
  std::vector<int32_t> cells;
  cells.reserve(n_cell_fields);
  stage_three_center_cells(bra, aux, aux_rec_begin, 0, aux.nshells(), cells, n_cell, cell_begin);

  const auto [d_staging, d_cells, cart, work, d_h] =
      res.device_regions<double, int32_t, double, double, double>(n_staging, n_cell_fields, n_cart,
                                                                  n_work, n_h);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_staging, staging.data(), sizeof(double) * n_staging,
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_cells, cells.data(), sizeof(int32_t) * n_cell_fields,
                                         wwr::wwrMemcpyHostToDevice, stream)));

  const double *const d_aux = d_staging + n_bra;
  const double *const d_u = d_aux + n_aux_fields;
  FBF_TRY(three_center_cartesian(res, n, n_sph, n_aux, n_aux_sph, d_u, d_u + (n * n_sph), gamma,
                                 cart, work));
  const auto n32 = static_cast<int32_t>(n);
  const auto n_aux32 = static_cast<int32_t>(n_aux);

  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(d_h, 0, sizeof(double) * n_h, stream)));
  for (size_t la = 0; la < kThreeCenterMomenta; ++la) {
    for (size_t lb = la; lb < kThreeCenterMomenta; ++lb) {
      for (size_t lp = 0; lp < kThreeCenterAuxMomenta; ++lp) {
        const size_t c = (((la * kThreeCenterMomenta) + lb) * kThreeCenterAuxMomenta) + lp;
        if (n_cell[c] == 0) {
          continue;
        }
        FBF_TRY(gpu_status(device::three_center_hessian_class(
            stream, static_cast<int32_t>(la), static_cast<int32_t>(lb), static_cast<int32_t>(lp),
            static_cast<int32_t>(n_bra_rec), d_staging, static_cast<int32_t>(n_aux_rec), d_aux,
            static_cast<int32_t>(n_cell[c]),
            d_cells + (static_cast<size_t>(device::kThreeCenterHessianCellFields) * cell_begin[c]),
            n32, n_aux32, static_cast<int32_t>(basis.natoms()), cart, d_h)));
      }
    }
  }
  std::vector<double> h(n_h);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h.data(), d_h, sizeof(double) * n_h,
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  // Also what lets the host staging go (the metric's comment).
  FBF_TRY(res.sync());
  return h;
}

} // namespace free_breakfast
