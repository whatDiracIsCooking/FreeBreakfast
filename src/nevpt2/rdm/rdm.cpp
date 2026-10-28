/// @file
/// @brief rdms' checks, tile plan and BLAS calls, and its launches.

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

size_t rdms_workspace_bytes(const int32_t n_orbitals, const int64_t width) noexcept {
  const auto n = static_cast<size_t>(n_orbitals);
  const size_t nn = n * n;
  const auto w = static_cast<size_t>(width);
  const auto region = [](const size_t count) {
    return math::align_up(sizeof(double) * count, DeviceResources::kWorkspaceAlignment);
  };
  // rdms' device_regions, in its order: eri, consume and fdm2, then R, L2, W.
  return (3 * region(nn * nn)) + region(nn * w) + region(nn * nn * w) + region(nn * w);
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

Result<int64_t> rdms(DeviceResources &res, const ci::DeviceStringSet &alpha,
                     const ci::DeviceStringSet &beta, const double *const vector,
                     const double *const eri, double *const dm3, double *const f3ca,
                     double *const f3ac, const size_t max_bytes) {
  if (alpha.dev_idx() != res.dev_idx() || beta.dev_idx() != res.dev_idx()) {
    return err_config(
        "nevpt2::rdms: strings are on devices {} and {}, but the call passed device {}",
        alpha.dev_idx(), beta.dev_idx(), res.dev_idx());
  }
  if (alpha.n_orbitals() != beta.n_orbitals()) {
    return err_config("nevpt2::rdms: alpha strings span {} orbitals, beta {}", alpha.n_orbitals(),
                      beta.n_orbitals());
  }
  if (alpha.n_orbitals() == 0) {
    return err_config("nevpt2::rdms: the strings span no orbitals");
  }
  const int64_t ndet = static_cast<int64_t>(alpha.size()) * beta.size();
  if (ndet > std::numeric_limits<int32_t>::max()) {
    return err_config("nevpt2::rdms: {} x {} determinants is more than an int32_t counts",
                      alpha.size(), beta.size());
  }
  const int32_t n32 = alpha.n_orbitals();
  const int64_t n = n32;
  const int64_t nn = n * n;
  const int64_t n4 = nn * nn;

  // The budget, and the widest tile it holds, before anything is queued or
  // allocated: past the device's memory the workspace's growth would abort,
  // not return. A growth allocates the new block before it frees the old, so
  // it needs the free memory alone.
  const ScopedDeviceIndex on{res.dev_idx()};
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  const size_t held = res.device_workspace(0).num_elements();
  const size_t usable = free_bytes > kRdmsHeadroomBytes ? free_bytes - kRdmsHeadroomBytes : 0;
  const size_t budget = std::min(max_bytes, std::max(held, usable));
  if (budget < rdms_workspace_bytes(n32, 1)) {
    return err_unsupported("nevpt2::rdms: a tile of one determinant over {} active orbitals needs "
                           "{} bytes of device workspace, and the budget is {} bytes",
                           n, rdms_workspace_bytes(n32, 1), budget);
  }
  // An upper bound, since every region is at least its doubles' bytes; then
  // down past the rounding.
  const auto per_det = static_cast<size_t>((2 * nn) + n4) * sizeof(double);
  int64_t width =
      std::min(ndet, static_cast<int64_t>((budget - rdms_workspace_bytes(n32, 0)) / per_det));
  while (rdms_workspace_bytes(n32, width) > budget) {
    --width;
  }
  const int64_t tiles = math::idivup(ndet, width);

  const wwr::wwrStream_t stream = res.stream();
  const size_t n6_bytes = sizeof(double) * static_cast<size_t>(n4 * nn);
  for (double *const out : {dm3, f3ca, f3ac}) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(out, 0, n6_bytes, stream)));
  }
  const auto [eri_rm, consume, fdm2, r, l2, w] =
      res.device_regions<double>(static_cast<size_t>(n4), static_cast<size_t>(n4),
                                 static_cast<size_t>(n4), static_cast<size_t>(nn * width),
                                 static_cast<size_t>(n4 * width), static_cast<size_t>(nn * width));
  FBF_TRY(gpu_status(device::rdm_convert_eri(stream, n32, eri, eri_rm, consume)));

  const int32_t n_beta = beta.size();
  for (int64_t begin = 0; begin < ndet; begin += width) {
    const int64_t count = std::min(width, ndet - begin);
    FBF_TRY(gpu_status(device::rdm_produce(
        stream, n32, n_beta, alpha.device_replacements(), alpha.replacements_per_string(),
        beta.device_replacements(), beta.replacements_per_string(), vector,
        static_cast<int32_t>(begin), static_cast<int32_t>(count), r, l2)));
    // dm3[pqrs, tu] += Σ_K L2[pqrs, K] R[tu, K].
    FBF_TRY(rdms_digest(res, n4, nn, count, l2, r, dm3));
    FBF_TRY(rdms_consume(res, n, count, consume, l2, w, true));
    FBF_TRY(rdms_digest(res, n4, nn, count, l2, w, f3ca));
    FBF_TRY(rdms_consume(res, n, count, consume, l2, w, false));
    FBF_TRY(rdms_digest(res, n4, nn, count, l2, w, f3ac));
  }

  // fdm2 and the wedge need the fully summed dm3 and f3.
  FBF_TRY(gpu_status(device::rdm_fdm2_ca(stream, n32, eri_rm, dm3, fdm2)));
  FBF_TRY(gpu_status(device::rdm_wedge(stream, n32, fdm2, f3ca)));
  FBF_TRY(gpu_status(device::rdm_fdm2_ac(stream, n32, eri_rm, dm3, fdm2)));
  FBF_TRY(gpu_status(device::rdm_wedge(stream, n32, fdm2, f3ac)));
  return tiles;
}

} // namespace free_breakfast::nevpt2
