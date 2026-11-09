/// @file
/// @brief rdms' and rdms_adjoint's (each of one vector and between two)
/// checks, tile plan and BLAS calls, and their launches.

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
import free_breakfast.core.footprint;
import free_breakfast.ci.strings;

#include "diagnostics.h"

namespace free_breakfast::nevpt2 {

// What the tile leaves of the memory budget: two pool chunks, the
// fragmentation headroom (pipeline/footprint's README). The pool takes memory
// from the driver in 32 MiB chunks, so what it reserves can run up to two of
// them past what is in use; past its limit an allocation aborts
// (CreatePolicy) rather than returns, so this keeps the scratch on the
// refusing side of it.
constexpr size_t kPoolHeadroomBytes = size_t{64} << 20;

// One region's bytes in the scratch: @p count doubles, rounded up to the
// alignment every region starts on.
size_t rdms_region(const size_t count) {
  return math::align_up(sizeof(double) * count, DeviceResources::kWorkspaceAlignment);
}

// @p scratch carved into one region per count, in order, each rdms_region's
// bytes: device_regions' carve, over the call's own scratch.
template <size_t N>
std::array<double *, N> rdms_carve(Buffers::device<std::byte> &scratch,
                                   const std::array<size_t, N> &counts) {
  std::array<double *, N> out{};
  size_t at = 0;
  for (size_t k = 0; k < N; ++k) {
    out[k] = reinterpret_cast<double *>(scratch.data() + at);
    at += rdms_region(counts[k]);
  }
  return out;
}

MemoryFootprint rdms_footprint(const int32_t n_orbitals, const int64_t width, const bool transition,
                               const bool f3) noexcept {
  const auto n = static_cast<size_t>(n_orbitals);
  const size_t nn = n * n;
  const auto w = static_cast<size_t>(width);
  // rdms' regions, in its order: eri, consume and fdm2, then R, L2, W, and
  // the ket's L2, which is empty for one vector. Without f3 only R and L2 are
  // not empty.
  const size_t bytes = f3 ? (3 * rdms_region(nn * nn)) + rdms_region(nn * w) +
                                rdms_region(nn * nn * w) + rdms_region(nn * w) +
                                rdms_region(transition ? nn * nn * w : 0)
                          : rdms_region(nn * w) + rdms_region(nn * nn * w);
  return {.device = {.scratch = bytes}};
}

MemoryFootprint rdms_adjoint_footprint(const int32_t n_orbitals, const int64_t width,
                                       const bool transition, const bool f3) noexcept {
  const auto n = static_cast<size_t>(n_orbitals);
  const size_t nn = n * n;
  const size_t n4 = nn * nn;
  const auto w = static_cast<size_t>(width);
  // rdms_adjoint's regions, in its order: eri, consume, the two fdm2 bars and
  // the consume's; the two raw f3 bars, dm3's bar and dm3; then R, L2, W, R̄,
  // L2̄, W̄; then the ket's L2 and its bar. Without f3 only R, L2 and their
  // bars are not empty.
  const size_t bytes = f3 ? (5 * rdms_region(n4)) + (4 * rdms_region(n4 * nn)) +
                                (4 * rdms_region(nn * w)) + (2 * rdms_region(n4 * w)) +
                                (2 * rdms_region(transition ? n4 * w : 0))
                          : (2 * rdms_region(nn * w)) + (2 * rdms_region(n4 * w));
  return {.device = {.scratch = bytes}};
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
// the answer and the walk down past the rounding is short.
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
  // allocated: the scratch is the call's last allocation, so it may take the
  // memory budget less the headroom, and past the budget it would abort, not
  // return.
  const size_t memory = FBF_TRY(res.memory_budget());
  const size_t usable = memory > kPoolHeadroomBytes ? memory - kPoolHeadroomBytes : 0;
  const size_t budget = std::min(max_bytes, usable);
  if (budget < bytes(n, 1)) {
    return err_unsupported("nevpt2::{}: a tile of one determinant over {} active orbitals needs "
                           "{} bytes of device scratch, and the budget is {} bytes (max_bytes {}, "
                           "device {}'s memory budget {} less {} of headroom)",
                           name, n, bytes(n, 1), budget, max_bytes, res.dev_idx(), memory,
                           kPoolHeadroomBytes);
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
  if ((f3ca == nullptr) != (f3ac == nullptr)) {
    return err_config("nevpt2::rdms: f3ca is {} and f3ac {}: the f3s are built together or not at "
                      "all",
                      f3ca == nullptr ? "null" : "given", f3ac == nullptr ? "null" : "given");
  }
  const bool f3 = f3ca != nullptr;
  const ScopedDeviceIndex on{res.dev_idx()};
  // One vector is the build it always was, operation for operation: one L2
  // per tile, the consume reading it as the ket's, and fdm2's own read.
  const bool transition = bra != ket;
  const int64_t n = alpha.n_orbitals();
  const int64_t nn = n * n;
  const int64_t n4 = nn * nn;
  const auto bytes = [transition, f3](const int32_t m, const int64_t w) {
    return rdms_footprint(m, w, transition, f3).device.scratch;
  };
  const int64_t per_det = f3 ? (2 * nn) + (transition ? 2 * n4 : n4) : nn + n4;
  const int64_t width = FBF_TRY(
      rdms_tile_width(res, alpha, beta, max_bytes, "rdms", bytes, static_cast<size_t>(per_det)));
  const int32_t n32 = alpha.n_orbitals();
  const int64_t ndet = static_cast<int64_t>(alpha.size()) * beta.size();
  const int64_t tiles = math::idivup(ndet, width);

  const wwr::wwrStream_t stream = res.stream();
  const size_t n6_bytes = sizeof(double) * static_cast<size_t>(n4 * nn);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(dm3, 0, n6_bytes, stream)));
  if (f3) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(f3ca, 0, n6_bytes, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(f3ac, 0, n6_bytes, stream)));
  }
  if (dm1 != nullptr) {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(dm1, 0, sizeof(double) * static_cast<size_t>(nn), stream)));
  }
  if (dm2 != nullptr) {
    FBF_TRY(
        gpu_status(wwr::wwrMemsetAsync(dm2, 0, sizeof(double) * static_cast<size_t>(n4), stream)));
  }
  // Every f3-only region is empty without f3: (tu|vw)'s two copies, fdm2, W
  // and the ket's L2, which is also empty for one vector.
  // One scratch block, the call's own: its size is the tile's, which the
  // budget sets, so it is never the workspace's.
  const auto f4 = static_cast<size_t>(f3 ? n4 : 0);
  Buffers::device<std::byte> scratch(bytes(n32, width), res.device());
  const auto [eri_rm, consume, fdm2, r, l2, w, l2_ket_region] = rdms_carve<7>(
      scratch, {f4, f4, f4, static_cast<size_t>(nn * width), static_cast<size_t>(n4 * width),
                static_cast<size_t>(f3 ? nn * width : 0),
                static_cast<size_t>(f3 && transition ? n4 * width : 0)});
  // L2 is the bra's; the consume reads the ket's, which is the same one for
  // one vector.
  double *const l2_ket = transition ? l2_ket_region : l2;
  if (f3) {
    FBF_TRY(gpu_status(device::rdm_convert_eri(stream, n32, eri, eri_rm, consume)));
  }

  const int32_t n_beta = beta.size();
  for (int64_t begin = 0; begin < ndet; begin += width) {
    const int64_t count = std::min(width, ndet - begin);
    FBF_TRY(gpu_status(device::rdm_produce(
        stream, n32, n_beta, alpha.device_replacements(), alpha.replacements_per_string(),
        beta.device_replacements(), beta.replacements_per_string(), bra, ket,
        static_cast<int32_t>(begin), static_cast<int32_t>(count), r, l2)));
    if (f3 && transition) {
      // The ket's own L2[pqrs, K] = ⟨J| E_pq E_rs |K⟩, which only the consume
      // reads; this gather writes R again, the same numbers.
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
    if (f3) {
      FBF_TRY(rdms_consume(res, n, count, consume, l2_ket, w, true));
      FBF_TRY(rdms_digest(res, n4, nn, count, l2, w, f3ca));
      FBF_TRY(rdms_consume(res, n, count, consume, l2_ket, w, false));
      FBF_TRY(rdms_digest(res, n4, nn, count, l2, w, f3ac));
    }
  }

  if (!f3) {
    return tiles;
  }
  // fdm2 and the wedge need the fully summed dm3 and f3.
  FBF_TRY(gpu_status(device::rdm_fdm2_ca(stream, n32, eri_rm, dm3, transition, fdm2)));
  FBF_TRY(gpu_status(device::rdm_wedge(stream, n32, fdm2, f3ca)));
  FBF_TRY(gpu_status(device::rdm_fdm2_ac(stream, n32, eri_rm, dm3, fdm2)));
  FBF_TRY(gpu_status(device::rdm_wedge(stream, n32, fdm2, f3ac)));
  return tiles;
}

// Both adjoints' one body. Between two vectors (@p transition) R̄ goes onto
// the ket and the bra's L2̄ onto the bra, and with the f3 bars the ket's own
// L2 is gathered for the consume, its bar onto the ket. Of one vector
// (@p bra == @p ket, @p bra_bar == @p ket_bar) it is the one-vector adjoint,
// operation for operation: one L2, its bar taking the consume's transpose, and
// one gather of both bars. Null f3 bars skip everything that exists only for
// f3, and leave (tu|vw) unread.
Result<int64_t> rdms_adjoint_of(DeviceResources &res, const ci::DeviceStringSet &alpha,
                                const ci::DeviceStringSet &beta, const double *const bra,
                                const double *const ket, const double *const eri,
                                const double *const dm3_bar, const double *const f3ca_bar,
                                const double *const f3ac_bar, double *const bra_bar,
                                double *const ket_bar, double *const eri_bar,
                                const size_t max_bytes, const bool transition) {
  if ((f3ca_bar == nullptr) != (f3ac_bar == nullptr)) {
    return err_config("nevpt2::rdms_adjoint: f3ca_bar is {} and f3ac_bar {}: the f3 bars are "
                      "given together or not at all",
                      f3ca_bar == nullptr ? "null" : "given",
                      f3ac_bar == nullptr ? "null" : "given");
  }
  const bool f3 = f3ca_bar != nullptr;
  const ScopedDeviceIndex on{res.dev_idx()};
  const int64_t n = alpha.n_orbitals();
  const int64_t nn = n * n;
  const int64_t n4 = nn * nn;
  const auto bytes = [transition, f3](const int32_t m, const int64_t w) {
    return rdms_adjoint_footprint(m, w, transition, f3).device.scratch;
  };
  const int64_t per_det = f3 ? (4 * nn) + (transition ? 4 * n4 : 2 * n4) : (2 * nn) + (2 * n4);
  const int64_t width = FBF_TRY(rdms_tile_width(res, alpha, beta, max_bytes, "rdms_adjoint", bytes,
                                                static_cast<size_t>(per_det)));
  const int32_t n32 = alpha.n_orbitals();
  const int64_t ndet = static_cast<int64_t>(alpha.size()) * beta.size();
  const int64_t tiles = math::idivup(ndet, width);

  const wwr::wwrStream_t stream = res.stream();
  // Every f3-only region is empty without f3, and the ket's L2 and its bar
  // without a transition.
  const auto f4 = static_cast<size_t>(f3 ? n4 : 0);
  const auto f6 = static_cast<size_t>(f3 ? n4 * nn : 0);
  const auto t2 = static_cast<size_t>(nn * width);
  const auto t4 = static_cast<size_t>(n4 * width);
  const size_t tw = f3 ? t2 : 0;
  const size_t tk = f3 && transition ? t4 : 0;
  Buffers::device<std::byte> scratch(bytes(n32, width), res.device());
  const auto [eri_rm, consume, fdm2_ca_bar, fdm2_ac_bar, consume_bar, raw_ca_bar, raw_ac_bar,
              dm3_total_region, dm3, r, l2, w, r_bar, l2_bar, w_bar, l2_ket_region,
              l2_ket_bar_region] =
      rdms_carve<17>(scratch, {f4, f4, f4, f4, f4, f6, f6, f6, f6, t2, t4, tw, t2, t4, tw, tk, tk});
  // The consume reads the ket's L2, the bra's for one vector, and its
  // transpose writes that L2's bar.
  double *const l2_ket = transition ? l2_ket_region : l2;
  double *const l2_ket_bar = transition ? l2_ket_bar_region : l2_bar;
  // Without f3 the bar dm3 carries into the tiles is the caller's.
  const double *const dm3_total_bar = f3 ? dm3_total_region : dm3_bar;
  const auto det_bytes = sizeof(double) * static_cast<size_t>(ndet);
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(bra_bar, 0, det_bytes, stream)));
  if (transition) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(ket_bar, 0, det_bytes, stream)));
  }
  if (f3) {
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(consume_bar, 0, sizeof(double) * f4, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(dm3, 0, sizeof(double) * f6, stream)));
    FBF_TRY(gpu_status(device::rdm_convert_eri(stream, n32, eri, eri_rm, consume)));

    // The wedge and fdm2, transposed first: they ran last.
    FBF_TRY(gpu_status(device::rdm_wedge_adjoint(stream, n32, f3ca_bar, raw_ca_bar, fdm2_ca_bar)));
    FBF_TRY(gpu_status(device::rdm_wedge_adjoint(stream, n32, f3ac_bar, raw_ac_bar, fdm2_ac_bar)));
    FBF_TRY(gpu_status(device::rdm_fdm2_dm3_adjoint(stream, n32, eri_rm, fdm2_ca_bar, fdm2_ac_bar,
                                                    dm3_bar, transition, dm3_total_region)));
  } else {
    // dm3 does not read (tu|vw).
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(eri_bar, 0, sizeof(double) * static_cast<size_t>(n4), stream)));
  }

  const int32_t n_beta = beta.size();
  for (int64_t begin = 0; begin < ndet; begin += width) {
    const int64_t count = std::min(width, ndet - begin);
    FBF_TRY(gpu_status(device::rdm_produce(
        stream, n32, n_beta, alpha.device_replacements(), alpha.replacements_per_string(),
        beta.device_replacements(), beta.replacements_per_string(), bra, ket,
        static_cast<int32_t>(begin), static_cast<int32_t>(count), r, l2)));
    if (f3 && transition) {
      // The ket's own L2, as rdms gathers it; this writes R again, the same
      // numbers.
      FBF_TRY(gpu_status(device::rdm_produce(
          stream, n32, n_beta, alpha.device_replacements(), alpha.replacements_per_string(),
          beta.device_replacements(), beta.replacements_per_string(), ket, ket,
          static_cast<int32_t>(begin), static_cast<int32_t>(count), r, l2_ket)));
    }
    if (f3) {
      // dm3 as rdms sums it, for the fdm2s' (tu|vw) bar after the last tile.
      FBF_TRY(rdms_digest(res, n4, nn, count, l2, r, dm3));
    }
    const auto l2_bytes = sizeof(double) * static_cast<size_t>(n4 * count);
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(l2_bar, 0, l2_bytes, stream)));
    if (f3 && transition) {
      FBF_TRY(gpu_status(wwr::wwrMemsetAsync(l2_ket_bar, 0, l2_bytes, stream)));
    }
    FBF_TRY(rdms_digest_adjoint(res, n4, nn, count, dm3_total_bar, l2, r, l2_bar, r_bar));
    if (f3) {
      for (const bool ca : {true, false}) {
        FBF_TRY(rdms_consume(res, n, count, consume, l2_ket, w, ca));
        FBF_TRY(rdms_digest_adjoint(res, n4, nn, count, ca ? raw_ca_bar : raw_ac_bar, l2, w, l2_bar,
                                    w_bar));
        FBF_TRY(rdms_consume_adjoint(res, n, count, consume, l2_ket, w_bar, l2_ket_bar, consume_bar,
                                     ca));
      }
    }
    // R̄ (and the ket's L2̄) onto the ket, the bra's L2̄ onto the bra: of one
    // vector, one gather of both.
    const auto gather = [&](const double *const from_r, const double *const from_l2,
                            double *const y) {
      return gpu_status(device::rdm_produce_adjoint(
          stream, n32, n_beta, static_cast<int32_t>(ndet), alpha.device_replacements(),
          alpha.replacements_per_string(), beta.device_replacements(),
          beta.replacements_per_string(), static_cast<int32_t>(begin), static_cast<int32_t>(count),
          from_r, from_l2, y));
    };
    if (transition) {
      FBF_TRY(gather(r_bar, f3 ? l2_ket_bar : nullptr, ket_bar));
      FBF_TRY(gather(nullptr, l2_bar, bra_bar));
    } else {
      FBF_TRY(gather(r_bar, l2_bar, bra_bar));
    }
  }

  if (f3) {
    FBF_TRY(gpu_status(device::rdm_eri_adjoint(stream, n32, dm3, fdm2_ca_bar, fdm2_ac_bar,
                                               consume_bar, transition, eri_bar)));
  }
  return tiles;
}

Result<int64_t> rdms_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                             const ci::DeviceStringSet &beta, const double *const vector,
                             const double *const eri, const double *const dm3_bar,
                             const double *const f3ca_bar, const double *const f3ac_bar,
                             double *const vector_bar, double *const eri_bar,
                             const size_t max_bytes) {
  return rdms_adjoint_of(res, alpha, beta, vector, vector, eri, dm3_bar, f3ca_bar, f3ac_bar,
                         vector_bar, vector_bar, eri_bar, max_bytes, false);
}

Result<int64_t> rdms_adjoint(DeviceResources &res, const ci::DeviceStringSet &alpha,
                             const ci::DeviceStringSet &beta, const double *const bra,
                             const double *const ket, const double *const eri,
                             const double *const dm3_bar, const double *const f3ca_bar,
                             const double *const f3ac_bar, double *const bra_bar,
                             double *const ket_bar, double *const eri_bar, const size_t max_bytes) {
  return rdms_adjoint_of(res, alpha, beta, bra, ket, eri, dm3_bar, f3ca_bar, f3ac_bar, bra_bar,
                         ket_bar, eri_bar, max_bytes, true);
}

} // namespace free_breakfast::nevpt2
