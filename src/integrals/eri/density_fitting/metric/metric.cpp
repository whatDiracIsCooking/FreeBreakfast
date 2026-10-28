/// @file
/// @brief The metric build: the host staging of FelizJueves's aux records and
/// shell pairs, one launch per class, and the spherical transform.

module;

// The kernel launcher, defined in metric_launch.cu: declared in the GMF so it
// keeps external linkage (see the header).
#include "metric_bridge.h"

module free_breakfast.integrals.eri.density_fitting.metric;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;

#include "diagnostics.h"

namespace free_breakfast {

// One extent per aux momentum the kernels cover; a class (lP, lQ) is entry
// lP * kMetricMomenta + lQ of the per-class tables below.
constexpr auto kMetricMomenta = static_cast<size_t>(device::kMetricMaxMomentum + 1);

Status metric(DeviceResources &res, const BasisSet &aux, MatrixAOPair &m) {
  if (m.dev_idx() != res.dev_idx()) {
    return err_config("metric: the matrices are on device {}, but the call passed device {}",
                      m.dev_idx(), res.dev_idx());
  }
  const uint32_t n = aux.nao_cart();
  if (m.cartesian().n() != n || m.spherical().n() != aux.nao_sphe()) {
    return err_config("metric: the matrices are {} x {} and {} x {}, but the aux basis has {} "
                      "Cartesian and {} spherical functions",
                      m.cartesian().n(), m.cartesian().n(), m.spherical().n(), m.spherical().n(), n,
                      aux.nao_sphe());
  }
  // The kernels index the metric as a 32-bit a * n_aux + b.
  if (static_cast<uint64_t>(n) * n > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return err_config("metric: {} Cartesian fitting functions is too many for the kernels' 32-bit "
                      "index",
                      n);
  }
  const std::vector<Shell> &shells = aux.shells();
  for (const Shell &shell : shells) {
    if (shell.angular_momentum() > device::kMetricMaxMomentum) {
      return err_config("metric: no kernel for an aux shell of momentum {}; the kernels cover l "
                        "<= {}",
                        shell.angular_momentum(), device::kMetricMaxMomentum);
    }
  }
  if (shells.empty()) {
    return {};
  }

  // The records, one per primitive of every shell, field-major
  // (metric_bridge.h); shell i's are [rec_begin[i], rec_begin[i + 1]). One
  // table for every class: a kernel reads only the ranges its pairs name.
  std::vector<int32_t> rec_begin{0};
  rec_begin.reserve(shells.size() + 1);
  for (const Shell &shell : shells) {
    rec_begin.push_back(rec_begin.back() + static_cast<int32_t>(shell.nprimitives()));
  }
  const auto n_rec = static_cast<size_t>(rec_begin.back());
  std::vector<double> records(static_cast<size_t>(device::kMetricRecordFields) * n_rec);
  for (size_t i = 0, r = 0; i < shells.size(); ++i) {
    const Vec<double, 3> &centre = shells[i].center();
    for (const Primitive &prim : shells[i].primitives()) {
      // FelizJueves's whole dressing: c carries every normalisation already
      // (README), and (2/sqrt(pi)) theta, the other half of the prefactor, is
      // in the kernel.
      const double weight = prim.coefficient * std::pow(std::numbers::pi / prim.exponent, 1.5);
      const std::array<double, device::kMetricRecordFields> fields{centre.x, centre.y, centre.z,
                                                                   prim.exponent, weight};
      for (size_t f = 0; f < fields.size(); ++f) {
        records[f * n_rec + r] = fields[f];
      }
      ++r;
    }
  }

  // The pairs P <= Q by shell index, the lower-momentum shell first: a pair
  // whose first shell has the higher momentum is staged swapped, and the
  // kernel's mirrored store writes both of its blocks either way. Counted per
  // class, then written field-major within each class's run.
  const auto class_of = [&](const size_t i, const size_t j) {
    const auto li = static_cast<size_t>(shells[i].angular_momentum());
    const auto lj = static_cast<size_t>(shells[j].angular_momentum());
    return std::min(li, lj) * kMetricMomenta + std::max(li, lj);
  };
  std::array<size_t, kMetricMomenta * kMetricMomenta> n_pair{};
  for (size_t i = 0; i < shells.size(); ++i) {
    for (size_t j = i; j < shells.size(); ++j) {
      ++n_pair[class_of(i, j)];
    }
  }
  std::array<size_t, kMetricMomenta * kMetricMomenta> pair_begin{};
  size_t n_pairs = 0;
  for (size_t c = 0; c < n_pair.size(); ++c) {
    pair_begin[c] = n_pairs;
    n_pairs += n_pair[c];
  }
  std::vector<int32_t> pairs(static_cast<size_t>(device::kMetricPairFields) * n_pairs);
  std::array<size_t, kMetricMomenta * kMetricMomenta> filled{};
  for (size_t i = 0; i < shells.size(); ++i) {
    for (size_t j = i; j < shells.size(); ++j) {
      const bool swapped = shells[i].angular_momentum() > shells[j].angular_momentum();
      const size_t p = swapped ? j : i;
      const size_t q = swapped ? i : j;
      const size_t c = class_of(i, j);
      int32_t *const run =
          pairs.data() + static_cast<size_t>(device::kMetricPairFields) * pair_begin[c];
      const std::array<int32_t, device::kMetricPairFields> fields{
          rec_begin[p],
          rec_begin[p + 1],
          rec_begin[q],
          rec_begin[q + 1],
          static_cast<int32_t>(aux.ao_offset_cart(static_cast<uint32_t>(p))),
          static_cast<int32_t>(aux.ao_offset_cart(static_cast<uint32_t>(q)))};
      for (size_t f = 0; f < fields.size(); ++f) {
        run[f * n_pair[c] + filled[c]] = fields[f];
      }
      ++filled[c];
    }
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const auto [d_records, d_pairs] =
      res.device_regions<double, int32_t>(records.size(), pairs.size());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_records, records.data(), sizeof(double) * records.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d_pairs, pairs.data(), sizeof(int32_t) * pairs.size(),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  // The staging vectors die at return; a pageable source does not promise the
  // copy has finished reading it until the stream does.
  FBF_TRY(res.sync());

  for (size_t lp = 0; lp < kMetricMomenta; ++lp) {
    for (size_t lq = lp; lq < kMetricMomenta; ++lq) {
      const size_t c = lp * kMetricMomenta + lq;
      if (n_pair[c] == 0) {
        continue;
      }
      FBF_TRY(gpu_status(device::metric_class(
          stream, static_cast<int32_t>(lp), static_cast<int32_t>(lq), static_cast<int32_t>(n_rec),
          d_records, static_cast<int32_t>(n_pair[c]),
          d_pairs + static_cast<size_t>(device::kMetricPairFields) * pair_begin[c],
          static_cast<int32_t>(n), m.cartesian().data())));
    }
  }
  FBF_TRY(m.cartesian_to_spherical(res));

  // U^T M U is two gemms, so element (a, b) and its mirror (b, a) are summed
  // in different orders and differ in the last bits (3.6e-15 of max|M| on
  // water in cc-pVDZ-JKFIT). Half of each plus half of its mirror is exactly
  // symmetric: the halving is exact, and a + b = b + a. geam may not write
  // over an operand it transposes, so it goes through the workspace; the
  // records there are dead, their launches queued before this on stream.
  const auto ns = static_cast<int32_t>(m.spherical().n());
  double *const spherical = m.spherical().data();
  const auto [symmetric] = res.device_regions<double>(m.spherical().num_elements());
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(res.blas(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T,
                                                ns, ns, &kHalf<double>, spherical, ns,
                                                &kHalf<double>, spherical, ns, symmetric, ns)));
  return gpu_status(wwr::wwrMemcpyAsync(spherical, symmetric,
                                        sizeof(double) * m.spherical().num_elements(),
                                        wwr::wwrMemcpyDeviceToDevice, stream));
}

} // namespace free_breakfast
