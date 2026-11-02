/// @file
/// @brief rdms' (of one vector and between two) and rdms_adjoint's checks,
/// tile plan and BLAS calls, and their launches.

module;

// The launchers, defined in nevpt2_rdm.cu: declared in the GMF so they keep
// external linkage (see the header).
#include "nevpt2_rdm_bridge.h"

module free_breakfast.nevpt2.rdm;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;  // kZero, kOne
import free_breakfast.core.primitives; // math::align_up, math::idivup
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// What the default budget leaves free beside the workspace: the BLAS
// libraries allocate their own workspaces outside the pool, and the pool
// rounds its allocations up, so the free memory is not all the workspace's to
// take. A growth past the device's memory would abort (CreatePolicy) rather
// than return, so this is what keeps the build on the refusing side of it.
constexpr size_t kRdmsHeadroomBytes = size_t{512} << 20;

size_t rdms_workspace_bytes(const int32_t n_orbitals, const int64_t width,
                            const bool transition) noexcept {
  const auto n = static_cast<size_t>(n_orbitals);
  const size_t nn = n * n;
  const auto w = static_cast<size_t>(width);
  const auto region = [](const size_t count) {
    return math::align_up(sizeof(double) * count, DeviceResources::kWorkspaceAlignment);
  };
  // rdms' device_regions, in its order: eri, consume and fdm2, then R, L2, W,
  // and the ket's L2, which is empty for one vector.
  return (3 * region(nn * nn)) + region(nn * w) + region(nn * nn * w) + region(nn * w) +
         region(transition ? nn * nn * w : 0);
}

size_t rdms_adjoint_workspace_bytes(const int32_t n_orbitals, const int64_t width) noexcept {
  const auto n = static_cast<size_t>(n_orbitals);
  const size_t nn = n * n;
  const size_t n4 = nn * nn;
  const auto w = static_cast<size_t>(width);
  const auto region = [](const size_t count) {
    return math::align_up(sizeof(double) * count, DeviceResources::kWorkspaceAlignment);
  };
  // rdms_adjoint's device_regions, in its order: eri, consume, the two fdm2
  // bars and the consume's; the two raw f3 bars, dm3's bar and dm3; then R,
  // L2, W, R̄, L2̄, W̄.
  return (5 * region(n4)) + (4 * region(n4 * nn)) + (4 * region(nn * w)) + (2 * region(n4 * w));
}

// W = (tu|vw) · L2 for one order over a tile of @p width, as n strided
// batched gemms, one per contracted index x, accumulating after the first:
//   ca: W[a, f, K] = Σ_pqx consume[x, a, p, q] L2[p, q, x, f, K]
//   ac: W[a, f, K] = Σ_pqx consume[x, a, p, q] L2[p, q, f, x, K]
// batched over the free f, so each gemm's k is the pair pq. BLAS is
// column-major, so the row-major W = consume_x · L2 is issued as
// Wᵀ = L2ᵀ consume_xᵀ, L2 first. W is written (a, f)-major for ac and
// (f, a)-major for ca: ca's digest then lands each row's last two axes
// transposed, which is where 'NEVPTkern_cedf_aedf' wants them, with no
// scatter and no n⁶ temporary.
Status rdms_consume(DeviceResources &res, const int64_t n, const int64_t width,
                    const double *const consume, const double *const l2, double *const w,
                    const bool ca) {
  const int64_t nn = n * n;
  const int64_t nw = n * width;
  const int64_t x_stride = ca ? nw : width;
  const int64_t f_stride = ca ? width : nw;
  const int64_t a_stride = ca ? width : nw;
  const int64_t w_stride = ca ? nw : width;
  for (int64_t x = 0; x < n; ++x) {
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int64_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, width, n, nn, &kOne<double>,
        l2 + (x * x_stride), nn * width, f_stride, consume + (x * nn * n), nn, 0,
        x == 0 ? &kZero<double> : &kOne<double>, w, a_stride, w_stride, n)));
  }
  return {};
}

// @p acc (row-major, rows × cols) += L2 · Bᵀ over a tile of @p width: L2 is
// rows × width and B cols × width, both row-major. The column-major call is
// accᵀ += op_T(Bᵀ) · L2ᵀ.
Status rdms_digest(DeviceResources &res, const int64_t rows, const int64_t cols,
                   const int64_t width, const double *const l2, const double *const b,
                   double *const acc) {
  return gpu_status(wwr::gemm<double, int64_t>(res.blas().get(), wwr::WWRBLAS_OP_T,
                                               wwr::WWRBLAS_OP_N, cols, rows, width, &kOne<double>,
                                               b, width, l2, width, &kOne<double>, acc, cols));
}

// The checks rdms and rdms_adjoint share, and the widest tile whose
// @p bytes(n, width) fits the budget, at most every determinant. @p per_det is
// the doubles a determinant adds to @p bytes, an underestimate of its bytes
// (every region is at least its doubles'), so the first guess is at or above
// the answer and the walk down past the rounding is short. The caller holds
// the ScopedDeviceIndex the free memory is read under.
Result<int64_t> rdms_tile_width(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                const ci::DeviceStringSet &beta, const size_t max_bytes,
                                const std::string_view name,
                                const std::function<size_t(int32_t, int64_t)> &bytes,
                                const size_t per_det) {
  if (alpha.dev_idx() != res.dev_idx() || beta.dev_idx() != res.dev_idx()) {
    return err_config("nevpt2::{}: strings are on devices {} and {}, but the call passed device {}",
                      name, alpha.dev_idx(), beta.dev_idx(), res.dev_idx());
  }
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("nevpt2::{}: alpha strings span {} orbitals, beta {}", name,
                      alpha.n_orbitals(), beta.n_orbitals());
  }
  if (alpha.n_orbitals() == 0) {
    return err_config("nevpt2::{}: the strings span no orbitals", name);
  }
  const int64_t ndet = static_cast<int64_t>(alpha.size()) * beta.size();
  if (ndet > std::numeric_limits<int32_t>::max()) {
    return err_config("nevpt2::{}: {} x {} determinants is more than an int32_t counts", name,
                      alpha.size(), beta.size());
  }
  const int32_t n = alpha.n_orbitals();

  // The budget, and the widest tile it holds, before anything is queued or
  // allocated: past the device's memory the workspace's growth would abort,
  // not return. A growth allocates the new block before it frees the old, so
  // it needs the free memory alone.
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  const size_t held = res.device_workspace(0).num_elements();
  const size_t usable = free_bytes > kRdmsHeadroomBytes ? free_bytes - kRdmsHeadroomBytes : 0;
  const size_t budget = std::min(max_bytes, std::max(held, usable));
  if (budget < bytes(n, 1)) {
    return err_unsupported("nevpt2::{}: a tile of one determinant over {} active orbitals needs "
                           "{} bytes of device workspace, and the budget is {} bytes",
                           name, n, bytes(n, 1), budget);
  }
  int64_t width =
      std::min(ndet, static_cast<int64_t>((budget - bytes(n, 0)) / (per_det * sizeof(double))));
  while (bytes(n, width) > budget) {
    --width;
  }
  return width;
}

// The transpose of rdms_consume, for one order over a tile of @p width, from
// W̄ (in W's layout): per x, L2̄ += consume_xᵀ-applied W̄, batched over the
// free f as the forward is,
//   ca: L2̄[p, q, x, f, K] += Σ_a consume[x, a, p, q] W̄[f, a, K],
// and per f, batched over x, the consume's bar summed over the tile,
//   ca: consumē[x, a, p, q] += Σ_K W̄[f, a, K] L2[p, q, x, f, K]
// (ac the same with L2's last two axes and W̄'s two swapped). The second sums
// over f too, which a batch cannot, so it is the loop.
Status rdms_consume_adjoint(DeviceResources &res, const int64_t n, const int64_t width,
                            const double *const consume, const double *const l2,
                            const double *const w_bar, double *const l2_bar,
                            double *const consume_bar, const bool ca) {
  const int64_t nn = n * n;
  const int64_t nw = n * width;
  const int64_t x_stride = ca ? nw : width;
  const int64_t f_stride = ca ? width : nw;
  const int64_t a_stride = ca ? width : nw;
  const int64_t w_stride = ca ? nw : width;
  for (int64_t x = 0; x < n; ++x) {
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int64_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, width, nn, n, &kOne<double>, w_bar,
        a_stride, w_stride, consume + (x * nn * n), nn, 0, &kOne<double>, l2_bar + (x * x_stride),
        nn * width, f_stride, n)));
  }
  for (int64_t f = 0; f < n; ++f) {
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int64_t>(
        res.blas().get(), wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nn, n, width, &kOne<double>,
        l2 + (f * f_stride), nn * width, x_stride, w_bar + (f * w_stride), a_stride, 0,
        &kOne<double>, consume_bar, nn, nn * n, n)));
  }
  return {};
}

// The transpose of rdms_digest's acc += L2 · Bᵀ, from the bar on acc
// (row-major, rows × cols): L2̄ += acc̄ · B, and B̄ = acc̄ᵀ · L2, overwritten.
// Column-major, L2̄ᵀ += Bᵀ acc̄ᵀ and B̄ᵀ = L2ᵀ acc̄.
Status rdms_digest_adjoint(DeviceResources &res, const int64_t rows, const int64_t cols,
                           const int64_t width, const double *const acc_bar, const double *const l2,
                           const double *const b, double *const l2_bar, double *const b_bar) {
  FBF_TRY(gpu_status(wwr::gemm<double, int64_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, width, rows, cols, &kOne<double>, b,
      width, acc_bar, cols, &kOne<double>, l2_bar, width)));
  return gpu_status(wwr::gemm<double, int64_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, width, cols, rows, &kOne<double>, l2,
      width, acc_bar, cols, &kZero<double>, b_bar, width));
}

Result<int64_t> rdms(DeviceResources &res, const ci::DeviceStringSet &alpha,
                     const ci::DeviceStringSet &beta, const double *const vector,
                     const double *const eri, double *const dm3, double *const f3ca,
                     double *const f3ac, const size_t max_bytes) {
  return rdms(res, alpha, beta, vector, vector, eri, nullptr, nullptr, dm3, f3ca, f3ac, max_bytes);
}

Result<int64_t> rdms(DeviceResources &res, const ci::DeviceStringSet &alpha,
                     const ci::DeviceStringSet &beta, const double *const bra,
                     const double *const ket, const double *const eri, double *const dm1,
                     double *const dm2, double *const dm3, double *const f3ca, double *const f3ac,
                     const size_t max_bytes) {
  const ScopedDeviceIndex on{res.dev_idx()};
  // One vector is the build it always was, operation for operation: one L2
  // per tile, the consume reading it as the ket's, and fdm2's own read.
  const bool transition = bra != ket;
  const int64_t n = alpha.n_orbitals();
  const int64_t nn = n * n;
  const int64_t n4 = nn * nn;
  const auto bytes = [transition](const int32_t m, const int64_t w) {
    return rdms_workspace_bytes(m, w, transition);
  };
  const int64_t width =
      FBF_TRY(rdms_tile_width(res, alpha, beta, max_bytes, "rdms", bytes,
                              static_cast<size_t>((2 * nn) + (transition ? 2 * n4 : n4))));
  const int32_t n32 = alpha.n_orbitals();
  const int64_t ndet = static_cast<int64_t>(alpha.size()) * beta.size();
  const int64_t tiles = math::idivup(ndet, width);

  const wwr::wwrStream_t stream = res.stream();
  const size_t n6_bytes = sizeof(double) * static_cast<size_t>(n4 * nn);
  for (double *const out : {dm3, f3ca, f3ac}) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(out, 0, n6_bytes, stream)));
  }
  if (dm1 != nullptr) {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(dm1, 0, sizeof(double) * static_cast<size_t>(nn), stream)));
  }
  if (dm2 != nullptr) {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(dm2, 0, sizeof(double) * static_cast<size_t>(n4), stream)));
  }
  const auto [eri_rm, consume, fdm2, r, l2, w, l2_ket_region] = res.device_regions<double>(
      static_cast<size_t>(n4), static_cast<size_t>(n4), static_cast<size_t>(n4),
      static_cast<size_t>(nn * width), static_cast<size_t>(n4 * width),
      static_cast<size_t>(nn * width), static_cast<size_t>(transition ? n4 * width : 0));
  // L2 is the bra's; the consume reads the ket's, which is the same one for
  // one vector.
  double *const l2_ket = transition ? l2_ket_region : l2;
  FBF_TRY(gpu_status(device::rdm_convert_eri(stream, n32, eri, eri_rm, consume)));

  const int32_t n_beta = beta.size();
  for (int64_t begin = 0; begin < ndet; begin += width) {
    const int64_t count = std::min(width, ndet - begin);
    FBF_TRY(gpu_status(device::rdm_produce(
        stream, n32, n_beta, alpha.device_replacements(), alpha.replacements_per_string(),
        beta.device_replacements(), beta.replacements_per_string(), bra, ket,
        static_cast<int32_t>(begin), static_cast<int32_t>(count), r, l2)));
    if (transition) {
      // The ket's own L2[pqrs, K] = ⟨J| E_pq E_rs |K⟩; this gather writes R
      // again, the same numbers.
      FBF_TRY(gpu_status(device::rdm_produce(
          stream, n32, n_beta, alpha.device_replacements(), alpha.replacements_per_string(),
          beta.device_replacements(), beta.replacements_per_string(), ket, ket,
          static_cast<int32_t>(begin), static_cast<int32_t>(count), r, l2_ket)));
    }
    // dm3[pqrs, tu] += Σ_K L2[pqrs, K] R[tu, K].
    FBF_TRY(rdms_digest(res, n4, nn, count, l2, r, dm3));
    // dm1[tu] += Σ_K R[tu, K] c_I[K] and dm2[pqrs] += Σ_K L2[pqrs, K] c_J[K]:
    // R and L2 are column-major count × n² and count × n⁴.
    if (dm1 != nullptr) {
      FBF_TRY(gpu_status(wwr::gemv<double, int64_t>(res.blas().get(), wwr::WWRBLAS_OP_T, count, nn,
                                                    &kOne<double>, r, count, bra + begin, 1,
                                                    &kOne<double>, dm1, 1)));
    }
    if (dm2 != nullptr) {
      FBF_TRY(gpu_status(wwr::gemv<double, int64_t>(res.blas().get(), wwr::WWRBLAS_OP_T, count, n4,
                                                    &kOne<double>, l2, count, ket + begin, 1,
                                                    &kOne<double>, dm2, 1)));
    }
    FBF_TRY(rdms_consume(res, n, count, consume, l2_ket, w, true));
    FBF_TRY(rdms_digest(res, n4, nn, count, l2, w, f3ca));
    FBF_TRY(rdms_consume(res, n, count, consume, l2_ket, w, false));
    FBF_TRY(rdms_digest(res, n4, nn, count, l2, w, f3ac));
  }

  // fdm2 and the wedge need the fully summed dm3 and f3.
  FBF_TRY(gpu_status(device::rdm_fdm2_ca(stream, n32, eri_rm, dm3, transition, fdm2)));
  FBF_TRY(gpu_status(device::rdm_wedge(stream, n32, fdm2, f3ca)));
  FBF_TRY(gpu_status(device::rdm_fdm2_ac(stream, n32, eri_rm, dm3, fdm2)));
  FBF_TRY(gpu_status(device::rdm_wedge(stream, n32, fdm2, f3ac)));
  return tiles;
}

Result<int64_t> rdms_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                             const ci::DeviceStringSet &beta, const double *const vector,
                             const double *const eri, const double *const dm3_bar,
                             const double *const f3ca_bar, const double *const f3ac_bar,
                             double *const vector_bar, double *const eri_bar,
                             const size_t max_bytes) {
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t n = alpha.n_orbitals();
  const int64_t nn = n * n;
  const int64_t n4 = nn * nn;
  const int64_t width = FBF_TRY(rdms_tile_width(res, alpha, beta, max_bytes, "rdms_adjoint",
                                                &rdms_adjoint_workspace_bytes,
                                                static_cast<size_t>((4 * nn) + (2 * n4))));
  const int32_t n32 = alpha.n_orbitals();
  const int64_t ndet = static_cast<int64_t>(alpha.size()) * beta.size();
  const int64_t tiles = math::idivup(ndet, width);

  const wwr::wwrStream_t stream = res.stream();
  const auto f4 = static_cast<size_t>(n4);
  const auto f6 = static_cast<size_t>(n4 * nn);
  const auto t2 = static_cast<size_t>(nn * width);
  const auto t4 = static_cast<size_t>(n4 * width);
  const auto [eri_rm, consume, fdm2_ca_bar, fdm2_ac_bar, consume_bar, raw_ca_bar, raw_ac_bar,
              dm3_total_bar, dm3, r, l2, w, r_bar, l2_bar, w_bar] =
      res.device_regions<double>(f4, f4, f4, f4, f4, f6, f6, f6, f6, t2, t4, t2, t2, t4, t2);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(consume_bar, 0, sizeof(double) * f4, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(dm3, 0, sizeof(double) * f6, stream)));
  FBF_TRY(gpu_status(
      wwr::wwrMemsetAsync(vector_bar, 0, sizeof(double) * static_cast<size_t>(ndet), stream)));
  FBF_TRY(gpu_status(device::rdm_convert_eri(stream, n32, eri, eri_rm, consume)));

  // The wedge and fdm2, transposed first: they ran last.
  FBF_TRY(gpu_status(device::rdm_wedge_adjoint(stream, n32, f3ca_bar, raw_ca_bar, fdm2_ca_bar)));
  FBF_TRY(gpu_status(device::rdm_wedge_adjoint(stream, n32, f3ac_bar, raw_ac_bar, fdm2_ac_bar)));
  FBF_TRY(gpu_status(device::rdm_fdm2_dm3_adjoint(stream, n32, eri_rm, fdm2_ca_bar, fdm2_ac_bar,
                                                  dm3_bar, dm3_total_bar)));

  const int32_t n_beta = beta.size();
  for (int64_t begin = 0; begin < ndet; begin += width) {
    const int64_t count = std::min(width, ndet - begin);
    FBF_TRY(gpu_status(device::rdm_produce(
        stream, n32, n_beta, alpha.device_replacements(), alpha.replacements_per_string(),
        beta.device_replacements(), beta.replacements_per_string(), vector, vector,
        static_cast<int32_t>(begin), static_cast<int32_t>(count), r, l2)));
    // dm3 as rdms sums it, for the fdm2s' (tu|vw) bar after the last tile.
    FBF_TRY(rdms_digest(res, n4, nn, count, l2, r, dm3));
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(l2_bar, 0, sizeof(double) * static_cast<size_t>(n4 * count), stream)));
    FBF_TRY(rdms_digest_adjoint(res, n4, nn, count, dm3_total_bar, l2, r, l2_bar, r_bar));
    for (const bool ca : {true, false}) {
      FBF_TRY(rdms_consume(res, n, count, consume, l2, w, ca));
      FBF_TRY(rdms_digest_adjoint(res, n4, nn, count, ca ? raw_ca_bar : raw_ac_bar, l2, w, l2_bar,
                                  w_bar));
      FBF_TRY(rdms_consume_adjoint(res, n, count, consume, l2, w_bar, l2_bar, consume_bar, ca));
    }
    FBF_TRY(gpu_status(device::rdm_produce_adjoint(
        stream, n32, n_beta, static_cast<int32_t>(ndet), alpha.device_replacements(),
        alpha.replacements_per_string(), beta.device_replacements(), beta.replacements_per_string(),
        static_cast<int32_t>(begin), static_cast<int32_t>(count), r_bar, l2_bar, vector_bar)));
  }

  FBF_TRY(gpu_status(
      device::rdm_eri_adjoint(stream, n32, dm3, fdm2_ca_bar, fdm2_ac_bar, consume_bar, eri_bar)));
  return tiles;
}

} // namespace free_breakfast::nevpt2
