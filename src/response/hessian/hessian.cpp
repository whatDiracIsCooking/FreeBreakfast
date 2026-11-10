/// @file
/// @brief Hessian: gen_g_hop's h_op as gemms over MO intermediates
/// built once and two J and K of transition densities per product,
/// integral-direct or density-fitted, and its h_diag assembled on the host
/// from diagonals read back once.

module free_breakfast.response.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, gemv, geam, gemmStridedBatched, gpu_status
import free_breakfast.core.resources;
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.footprint;  // MemoryFootprint, kPoolHeadroomBytes
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.norm; // norm, Norm
import free_breakfast.recipes.linear_operator;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.eri.density_fitting.jk; // df_coulomb, df_exchange(_symmetrized)
import free_breakfast.response.fock;
import free_breakfast.response.orbital_pairs;

#include "diagnostics.h"

namespace free_breakfast::response {

static_assert(LinearOperator<Hessian, double>);

// build's workspace carve, in doubles: gemmwich's intermediate, one projected
// nmo x nmo block, and the permuted Gamma of H_act's exchange part; density-
// fitted (naux > 0), B half- and fully transformed over C_core, (Q|pp), and
// (pp|ii) and (pi|pi) over every p and core i. Shared with the footprints.
std::array<size_t, 8> hessian_build_counts(const size_t n_all, const size_t core, const size_t na,
                                           const size_t n_cart, const size_t naux) {
  const size_t mo_core = naux * core * n_all;
  const size_t pc = naux > 0 ? n_all * core : 0;
  return {n_cart * n_all, n_all * n_all, na * na * na * na, mo_core, mo_core, naux * n_all, pc, pc};
}

// apply's carve for a block of width columns, in doubles: kappa, X2 and the
// ncas x nmo block va lands in, per column; h kappa, the other ncas x nmo
// block, W and an n_cart^2 product, once; the densities D3 then D', per
// column; J C, once; and the right factors R3 (n_cart x ncore) and R'
// (n_cart x nocc), per column. Batched, every density's J and K too. Shared
// with the footprints.
std::array<size_t, 13> hessian_apply_counts(const size_t n_all, const size_t core, const size_t na,
                                            const size_t n_cart, const size_t width,
                                            const bool batched) {
  const size_t nn = n_all * n_all;
  const size_t occ = core + na;
  const size_t square = n_cart * n_cart;
  return {width * nn,
          width * nn,
          n_all * occ,
          width * na * n_all,
          na * n_all,
          std::max(core, na) * n_cart,
          square,
          2 * width * square,
          n_cart * n_all,
          width * n_cart * core,
          width * n_cart * occ,
          batched ? 2 * width * square : 0,
          batched ? 2 * width * square : 0};
}

// The Hessian's own buffers, create's persistent bytes, past its J and K
// matrices: U C (n_cart rows), C^T h C, F + F^T, the two V's, the occupied
// 1-RDM, gamma, H_act and the diagonal.
size_t hessian_bytes(const size_t n_all, const size_t core, const size_t na, const size_t n_cart) {
  const size_t occ = core + na;
  const size_t n_pairs = (core * (n_all - core)) + (na * (n_all - occ));
  return sizeof(double) * ((n_cart * n_all) + (4 * n_all * n_all) + (occ * occ) + (na * na) +
                           (n_all * na * n_all * na) + n_pairs);
}

// The bytes device_regions takes for @p counts doubles.
template <size_t N>
size_t hessian_regions(const std::array<size_t, N> &counts) {
  size_t bytes = 0;
  for (const size_t count : counts) {
    bytes += math::align_up(count * sizeof(double), DeviceResources::kWorkspaceAlignment);
  }
  return bytes;
}

// The scratch a density-fitted batch of @p width columns takes over @p n_all
// orbitals, in bytes: its carve, one allocation, and past it the larger of
// the block df_coulomb's Gamma over the 2 width densities and the
// df_exchange_symmetrized Y at rank nocc (D''s, the larger of its two ranks),
// each allocated and freed in turn while the carve is held.
size_t hessian_batch_bytes(const size_t n_all, const size_t core, const size_t na,
                           const size_t naux, const size_t width) {
  const std::array<size_t, 13> counts = hessian_apply_counts(n_all, core, na, n_all, width, true);
  const auto nmo = static_cast<int32_t>(n_all);
  const auto q = static_cast<int32_t>(naux);
  const auto w = static_cast<int32_t>(width);
  return (sizeof(double) * std::reduce(counts.begin(), counts.end())) +
         std::max(df_coulomb_footprint(nmo, q, 2 * w).device.scratch,
                  df_exchange_symmetrized_footprint(nmo, q, static_cast<int32_t>(core + na), w)
                      .device.scratch);
}

// How many columns Hessian::apply builds a density-fitted block's J and K
// together over: the most, up to @p columns, whose scratch (hessian_batch_bytes,
// every allocation the call has still to make) fits the memory budget less the
// pool's fragmentation headroom, and that the int32_t BLAS dispatch allows.
// The workspace, hessian_apply_footprint's at any width, is DeviceResources',
// held from construction, so the budget is read with it in place. 1 when two
// columns do not fit: the one-column path, whose scratch is
// hessian_apply_footprint's.
Result<size_t> hessian_batch_width(DeviceResources &res, const size_t n_all, const size_t core,
                                   const size_t na, const size_t naux, const size_t columns) {
  const size_t memory = FBF_TRY(res.memory_budget());
  const size_t usable = memory > kPoolHeadroomBytes ? memory - kPoolHeadroomBytes : 0;
  const size_t int32_max = std::numeric_limits<int32_t>::max();
  const size_t widest = std::min(columns, naux > 0 ? int32_max / (naux * (core + na)) : columns);
  const auto fits = [&](const size_t width) {
    return hessian_batch_bytes(n_all, core, na, naux, width) <= usable;
  };
  if (widest < 2 || !fits(2)) {
    return size_t{1};
  }
  // The widest in [2, widest] that fits: the bytes grow with the width.
  size_t lo = 2;
  size_t hi = widest;
  while (lo < hi) {
    const size_t mid = lo + ((hi - lo + 1) / 2);
    if (fits(mid)) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

// Whether the Hessian has pairs to apply over.
bool hessian_has_pairs(const size_t n_all, const size_t core, const size_t na) {
  return (core * (n_all - core)) + (na * (n_all - core - na)) > 0;
}

MemoryFootprint hessian_footprint(const BasisSet &basis, const int32_t ncore, const int32_t ncas) {
  const size_t n_all = basis.nao_sphe();
  const size_t n_cart = basis.nao_cart();
  const auto core = static_cast<size_t>(ncore);
  const auto na = static_cast<size_t>(ncas);
  MemoryFootprint out;
  out.device.persistent =
      hessian_bytes(n_all, core, na, n_cart) + (3 * sizeof(double) * n_cart * n_cart);
  out.device.workspace = hessian_regions(hessian_build_counts(n_all, core, na, n_cart, 0));
  return out;
}

MemoryFootprint hessian_footprint(const int32_t nmo, const int32_t naux, const int32_t ncore,
                                  const int32_t ncas) {
  const auto n_all = static_cast<size_t>(nmo);
  const auto core = static_cast<size_t>(ncore);
  const auto na = static_cast<size_t>(ncas);
  MemoryFootprint out;
  out.device.persistent =
      hessian_bytes(n_all, core, na, n_all) + (2 * sizeof(double) * n_all * n_all);
  out.device.workspace =
      hessian_regions(hessian_build_counts(n_all, core, na, n_all, static_cast<size_t>(naux)));
  return out;
}

MemoryFootprint hessian_apply_footprint(const BasisSet &basis, const int32_t ncore,
                                        const int32_t ncas) {
  const size_t n_all = basis.nao_sphe();
  const auto core = static_cast<size_t>(ncore);
  const auto na = static_cast<size_t>(ncas);
  MemoryFootprint out;
  if (hessian_has_pairs(n_all, core, na)) {
    // One region of every count's doubles, carved by hand.
    const std::array<size_t, 13> counts =
        hessian_apply_counts(n_all, core, na, basis.nao_cart(), 1, false);
    out.device.workspace =
        hessian_regions(std::array<size_t, 1>{std::reduce(counts.begin(), counts.end())});
  }
  return out;
}

MemoryFootprint hessian_apply_footprint(const int32_t nmo, const int32_t naux, const int32_t ncore,
                                        const int32_t ncas) {
  const auto n_all = static_cast<size_t>(nmo);
  const auto core = static_cast<size_t>(ncore);
  const auto na = static_cast<size_t>(ncas);
  MemoryFootprint out;
  if (!hessian_has_pairs(n_all, core, na)) {
    return out;
  }
  const std::array<size_t, 13> counts = hessian_apply_counts(n_all, core, na, n_all, 1, false);
  out.device.scratch = sizeof(double) * std::reduce(counts.begin(), counts.end());
  if (ncore > 0 && nmo > ncore && naux > 0) {
    // The two-factor df_exchange's X and Y at rank nocc, the larger of its
    // two ranks (ncore for D3), each a workspace region.
    const size_t xy =
        math::align_up(sizeof(double) * static_cast<size_t>(naux) * (core + na) * n_all,
                       DeviceResources::kWorkspaceAlignment);
    out.device.workspace = std::max(df_coulomb_footprint(nmo, naux).device.workspace, 2 * xy);
  }
  return out;
}

Hessian::Hessian(DeviceResources &res, const int32_t nmo, const int32_t ncore, const int32_t ncas,
                 const int32_t n_cart, std::variant<Direct, Fitted> jk)
    : jk_(std::move(jk)), dev_idx_(res.dev_idx()), nmo_(nmo), ncore_(ncore), ncas_(ncas),
      n_cart_(n_cart), n_pairs_((ncore * (nmo - ncore)) + (ncas * (nmo - ncore - ncas))),
      c_cart_(static_cast<size_t>(n_cart_) * static_cast<size_t>(nmo), res.device()),
      h_mo_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      fock_sym_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      vhf_c_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      vhf_ca_(static_cast<size_t>(nmo) * static_cast<size_t>(nmo), res.device()),
      dm1_occ_(static_cast<size_t>(ncore + ncas) * static_cast<size_t>(ncore + ncas), res.device()),
      gamma_(static_cast<size_t>(ncas) * static_cast<size_t>(ncas), res.device()),
      h_act_(static_cast<size_t>(nmo) * static_cast<size_t>(ncas) * static_cast<size_t>(nmo) *
                 static_cast<size_t>(ncas),
             res.device()),
      diagonal_(static_cast<size_t>(n_pairs_), res.device()) {}

Result<std::pair<const double *, double>> Hessian::two_j_minus_k(DeviceResources &res,
                                                                 const double *const density,
                                                                 const double *const right,
                                                                 const int32_t rank) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const int32_t nc = n_cart_;
  if (auto *const fitted = std::get_if<Fitted>(&jk_); fitted != nullptr) {
    // K[L R^T + R L^T] = K[L R^T] + K[L R^T]^T: one two-factor call.
    const double *const b = fitted->b;
    double *const j = fitted->j.data();
    double *const k = fitted->k.data();
    FBF_TRY(df_coulomb(res, nc, fitted->naux, b, density, j));
    FBF_TRY(df_exchange(res, nc, fitted->naux, rank, b, c_cart_.data(), right, k));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nc,
                                                  nc, &kTwo<double>, j, nc, &kNegOne<double>, k, nc,
                                                  j, nc)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc,
                                                  nc, &kOne<double>, j, nc, &kNegOne<double>, k, nc,
                                                  j, nc)));
    return std::pair{static_cast<const double *>(j), 1.0};
  }
  Direct &direct = std::get<Direct>(jk_);
  // Host pointer mode, so the call has waited for it.
  double top = 0.0;
  FBF_TRY(norm<Norm::inf>(res, nc * nc, density, 1, &top));
  const double s = top > 2.0 ? 2.0 / top : 1.0;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nc, nc,
                                                &s, density, nc, &kZero<double>, density, nc,
                                                direct.density.data(), nc)));
  FBF_TRY(coulomb(res, direct.j_classes, direct.density, direct.j, direct.tau_j * s));
  FBF_TRY(exchange(res, direct.k_classes, direct.density, direct.k, direct.tau_k * s));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nc, nc, &kTwo<double>, direct.j.data(), nc,
      &kNegOne<double>, direct.k.data(), nc, direct.j.data(), nc)));
  return std::pair{static_cast<const double *>(direct.j.data()), s};
}

Result<Hessian> Hessian::create(DeviceResources &res, const BasisSet &basis,
                                const SphericalMatrixAO &h,
                                const std::shared_ptr<const Buffers::device<double>> &u,
                                const std::span<CoulombPairClass> j_classes,
                                const std::span<OrderedPairClass> k_classes,
                                const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
                                const Buffers::device<double> &ppaa,
                                const Buffers::device<double> &papa, const Fock &fock,
                                const double tau_j, const double tau_k) {
  // Written so that NaN fails too.
  if (!(tau_j >= 0.0) || !(tau_k >= 0.0)) {
    return err_config("response::Hessian::create: tau_j ({}) and tau_k ({}) must be 0 or above",
                      tau_j, tau_k);
  }
  const uint32_t n_sphe = basis.nao_sphe();
  if (h.n() != n_sphe || c.n() != n_sphe) {
    return err_config(
        "response::Hessian::create: h is {} x {} and C {} x {}, but the basis has {} spherical AOs",
        h.n(), h.n(), c.n(), c.n(), n_sphe);
  }
  return build(res, h, u, c, ncore, ncas, ppaa, papa, fock, static_cast<int32_t>(basis.nao_cart()),
               Direct{.j_classes = j_classes,
                      .k_classes = k_classes,
                      .tau_j = tau_j,
                      .tau_k = tau_k,
                      .density = CartesianMatrixAO(res, basis),
                      .j = CartesianMatrixAO(res, basis),
                      .k = CartesianMatrixAO(res, basis)});
}

Result<Hessian> Hessian::create(DeviceResources &res, const SphericalMatrixAO &h,
                                const Buffers::device<double> &b, const int32_t naux,
                                const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
                                const Buffers::device<double> &ppaa,
                                const Buffers::device<double> &papa, const Fock &fock) {
  if (naux < 0) {
    return err_config("response::Hessian::create: naux ({}) must not be negative", naux);
  }
  if (h.n() != c.n()) {
    return err_config("response::Hessian::create: h is {} x {} but C {} x {}", h.n(), h.n(), c.n(),
                      c.n());
  }
  const size_t nn = static_cast<size_t>(c.n()) * c.n();
  if (b.num_elements() != static_cast<size_t>(naux) * nn) {
    return err_config("response::Hessian::create: B holds {} doubles, but naux nmo^2 is {}",
                      b.num_elements(), static_cast<size_t>(naux) * nn);
  }
  return build(res, h, nullptr, c, ncore, ncas, ppaa, papa, fock, static_cast<int32_t>(c.n()),
               Fitted{.b = b.data(),
                      .naux = naux,
                      .j = Buffers::device<double>(nn, res.device()),
                      .k = Buffers::device<double>(nn, res.device())});
}

Result<Hessian> Hessian::build(DeviceResources &res, const SphericalMatrixAO &h,
                               const std::shared_ptr<const Buffers::device<double>> &u,
                               const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
                               const Buffers::device<double> &ppaa,
                               const Buffers::device<double> &papa, const Fock &fock,
                               const int32_t n_cart_rows, std::variant<Direct, Fitted> jk) {
  const uint32_t n_orbitals = c.n();
  if (ncore < 0 || ncas < 0) {
    return err_config("response::Hessian::create: ncore ({}) and ncas ({}) must not be negative",
                      ncore, ncas);
  }
  if (ncore + ncas == 0) {
    return err_config(
        "response::Hessian::create: ncore and ncas are both 0, so nothing is occupied");
  }
  if (std::cmp_greater(int64_t{ncore} + ncas, n_orbitals)) {
    return err_config("response::Hessian::create: ncore {} + ncas {} exceeds the {} orbitals",
                      ncore, ncas, n_orbitals);
  }
  if (h.dev_idx() != res.dev_idx() || c.dev_idx() != res.dev_idx()) {
    return err_config(
        "response::Hessian::create: h is on device {} and C on {}, but the call passed device {}",
        h.dev_idx(), c.dev_idx(), res.dev_idx());
  }
  const size_t n_all = n_orbitals;
  const size_t nn = n_all * n_all;
  const auto na = static_cast<size_t>(ncas);
  const auto core = static_cast<size_t>(ncore);
  const size_t occ = core + na;
  if (ppaa.num_elements() != nn * na * na || papa.num_elements() != nn * na * na) {
    return err_config(
        "response::Hessian::create: ppaa holds {} and papa {} doubles, but nmo^2 ncas^2 is {}",
        ppaa.num_elements(), papa.num_elements(), nn * na * na);
  }
  const size_t n_pairs = (core * (n_all - core)) + (na * (n_all - occ));
  if (fock.fock_inactive.num_elements() != nn || fock.fock_active.num_elements() != nn ||
      fock.fock.num_elements() != nn || fock.gradient.num_elements() != n_pairs ||
      fock.gamma.num_elements() != na * na || fock.gamma2.num_elements() != na * na * na * na) {
    return err_config("response::Hessian::create: the Fock is not of {} orbitals, ncore {} and "
                      "ncas {}",
                      n_all, ncore, ncas);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto nmo = static_cast<int32_t>(n_orbitals);
  const int32_t nocc = ncore + ncas;
  Hessian out(res, nmo, ncore, ncas, n_cart_rows, std::move(jk));
  const Fitted *const fitted = std::get_if<Fitted>(&out.jk_);
  const int32_t nc = out.n_cart_;
  const auto n_cart = static_cast<size_t>(nc);

  // Scratch: gemmwich's intermediate, one projected nmo x nmo block, and
  // the permuted Gamma of H_act's exchange part; density-fitted, B half-
  // and fully transformed over C_core, (Q|pp), and (pp|ii) and (pi|pi) over
  // every p and core i. coulomb and exchange, called while it is live, do
  // not take the workspace; df_coulomb and df_exchange are not called here.
  const size_t naux = fitted != nullptr ? static_cast<size_t>(fitted->naux) : 0;
  const std::array<size_t, 8> counts = hessian_build_counts(n_all, core, na, n_cart, naux);
  const auto [work, block, gamma2_exchange, half, l_core, q_pp, j_core, k_core] = FBF_TRY(
      res.device_regions<double>("response::Hessian::build", counts[0], counts[1], counts[2],
                                 counts[3], counts[4], counts[5], counts[6], counts[7]));

  // U C, then C^T h C, and the MO matrices the product reads.
  if (u != nullptr) {
    FBF_TRY(gpu_status(gemm(blas, nc, nmo, nmo, &kOne<double>, u->data(), c.data(), &kZero<double>,
                            out.c_cart_.data())));
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.c_cart_.data(), c.data(), sizeof(double) * nn,
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  FBF_TRY(gpu_status(gemmwich(blas, nmo, nmo, c.data(), h.data(), work, out.h_mo_.data())));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo,
                                                nmo, &kOne<double>, fock.fock_inactive.data(), nmo,
                                                &kNegOne<double>, out.h_mo_.data(), nmo,
                                                out.vhf_c_.data(), nmo)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nmo, &kOne<double>, out.vhf_c_.data(), nmo,
      &kOne<double>, fock.fock_active.data(), nmo, out.vhf_ca_.data(), nmo)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
      blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo, nmo, &kOne<double>, fock.fock.data(), nmo,
      &kOne<double>, fock.fock.data(), nmo, out.fock_sym_.data(), nmo)));

  // gamma and Gamma back to the host, for the 1-RDM's occupied block, H_act's
  // permuted Gamma and the diagonal; and, integral-direct, the core orbitals'
  // columns of U C, for the prune cap of each C_i C_i^T.
  std::vector<double> gamma(na * na);
  std::vector<double> gamma2(na * na * na * na);
  std::vector<double> c_core(fitted == nullptr ? n_cart * core : 0);
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(out.gamma_.data(), fock.gamma.data(),
                                           gamma.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma.data(), fock.gamma.data(),
                                           gamma.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma2.data(), fock.gamma2.data(),
                                           gamma2.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
  }
  if (!c_core.empty()) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_core.data(), out.c_cart_.data(),
                                           c_core.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
  }
  FBF_TRY(res.sync());

  // The 1-RDM over the occupied orbitals: 2 on the core diagonal, gamma on
  // the active block. Live until the final synchronize.
  std::vector<double> dm1(static_cast<size_t>(nocc) * static_cast<size_t>(nocc), 0.0);
  for (size_t i = 0; i < core; ++i) {
    dm1[i * (occ + 1)] = 2.0;
  }
  for (size_t t = 0; t < na; ++t) {
    for (size_t v = 0; v < na; ++v) {
      dm1[core + t + (occ * (core + v))] = gamma[t + (na * v)];
    }
  }
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(out.dm1_occ_.data(), dm1.data(), dm1.size() * sizeof(double),
                                     wwr::wwrMemcpyHostToDevice, stream)));

  // H_act[(p, u), (r, v)] = sum_wx (pr|wx) Gamma_wxuv + (pw|rx) Gamma'_wxuv,
  // Gamma'_wxuv = Gamma_uwxv + Gamma_wuxv: gen_g_hop's hdm2. For each v, a
  // gemm per r over ppaa's (or papa's) rows (p, r) of every block (w, x),
  // batched over r: A_r = (p, wx) at nmo r + p + nmo^2 wx, B the Gamma
  // columns (wx, u) of v, and C_r the H_act columns (r, v) over (p, u).
  std::vector<double> gamma2_permuted(gamma2.size());
  for (size_t w = 0; w < na; ++w) {
    for (size_t x = 0; x < na; ++x) {
      for (size_t uu = 0; uu < na; ++uu) {
        for (size_t v = 0; v < na; ++v) {
          const auto at = [na](size_t a, size_t b, size_t cc, size_t d) {
            return a + (na * (b + (na * (cc + (na * d)))));
          };
          gamma2_permuted[at(w, x, uu, v)] = gamma2[at(uu, w, x, v)] + gamma2[at(w, uu, x, v)];
        }
      }
    }
  }
  if (ncas > 0) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(gamma2_exchange, gamma2_permuted.data(),
                                           gamma2_permuted.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, stream)));
    const auto nmo64 = static_cast<int64_t>(n_all);
    const int32_t pairs = ncas * ncas;
    const auto ld_pairs = static_cast<int32_t>(nn);
    for (size_t v = 0; v < na; ++v) {
      double *const columns = out.h_act_.data() + (n_all * na * n_all * v);
      const std::array<std::pair<const double *, const double *>, 2> terms{
          {{ppaa.data(), fock.gamma2.data()}, {papa.data(), gamma2_exchange}}};
      for (size_t term = 0; term < terms.size(); ++term) {
        const auto [integrals, weights] = terms[term];
        const double *const beta = term == 0 ? &kZero<double> : &kOne<double>;
        FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, ncas, pairs, &kOne<double>, integrals,
            ld_pairs, nmo64, weights + (na * na * na * v), pairs, 0, beta, columns, nmo,
            nmo64 * ncas, nmo)));
      }
    }
  }

  // The diagonal's inputs, read back: diag of C^T h C, F, F^I - C^T h C and
  // F^I + F^A - C^T h C; diag H_act; ppaa's and papa's core rows; and
  // (pp|ii) and (pi|pi) for each core i.
  const auto diagonal = [&](double *to, const double *matrix, size_t count, size_t ld) {
    return gpu_status(wwr::wwrMemcpy2DAsync(to, sizeof(double), matrix, (ld + 1) * sizeof(double),
                                            sizeof(double), count, wwr::wwrMemcpyDeviceToHost,
                                            stream));
  };
  std::vector<double> h_diag(n_all);
  std::vector<double> f_diag(n_all);
  std::vector<double> vc_diag(n_all);
  std::vector<double> vca_diag(n_all);
  std::vector<double> act_diag(n_all * na);
  std::vector<double> ppaa_core(core * n_all * na * na);
  std::vector<double> papa_core(core * n_all * na * na);
  std::vector<double> j_pc(n_all * core);
  std::vector<double> k_pc(n_all * core);
  FBF_TRY(diagonal(h_diag.data(), out.h_mo_.data(), n_all, n_all));
  FBF_TRY(diagonal(f_diag.data(), fock.fock.data(), n_all, n_all));
  FBF_TRY(diagonal(vc_diag.data(), out.vhf_c_.data(), n_all, n_all));
  FBF_TRY(diagonal(vca_diag.data(), out.vhf_ca_.data(), n_all, n_all));
  if (ncas > 0) {
    FBF_TRY(diagonal(act_diag.data(), out.h_act_.data(), n_all * na, n_all * na));
  }
  if (ncore > 0 && ncas > 0) {
    // Rows [0, ncore) of every column (q, uv) of ppaa and papa.
    for (const auto &[to, from] :
         {std::pair{ppaa_core.data(), ppaa.data()}, std::pair{papa_core.data(), papa.data()}}) {
      FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
          to, core * sizeof(double), from, n_all * sizeof(double), core * sizeof(double),
          n_all * na * na, wwr::wwrMemcpyDeviceToHost, stream)));
    }
  }
  std::vector<double> scale(core, 1.0);
  if (fitted != nullptr && ncore > 0 && naux > 0) {
    // (pp|ii) = sum_Q (Q|pp) (Q|ii) and (pi|pi) = sum_Q (Q|ip)^2 from
    // L_Q,ip = (Q|ip) at Q + naux (i + ncore p) and (Q|pp) at Q + naux p.
    // B is half-transformed ncore columns of C at a time, the first block
    // C_core, which L completes; (Q|pp) is a gemv per p of each block.
    const auto nq = static_cast<int32_t>(naux);
    const auto nq64 = static_cast<int64_t>(naux);
    const double *const b = fitted->b;
    const double *const cc = out.c_cart_.data();
    for (int32_t p0 = 0; p0 < nmo; p0 += ncore) {
      const int32_t width = std::min(ncore, nmo - p0);
      // half[Q, p, m] = sum_n B_Q,nm C_n,p0+p at Q + naux (p + width m): one
      // gemm per m.
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nq, width, nmo, &kOne<double>, b, nq,
          nq64 * nmo, cc + (n_all * static_cast<size_t>(p0)), nmo, 0, &kZero<double>, half, nq,
          nq64 * width, nmo)));
      if (p0 == 0) {
        FBF_TRY(gpu_status(
            gemm(blas, nq * ncore, nmo, nmo, &kOne<double>, half, cc, &kZero<double>, l_core)));
      }
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nq, 1, nmo, &kOne<double>, half, nq * width,
          nq64, cc + (n_all * static_cast<size_t>(p0)), nmo, static_cast<int64_t>(n_all),
          &kZero<double>, q_pp + (naux * static_cast<size_t>(p0)), nq, nq64, width)));
    }
    // (Q|ii) is L's (i, i), every naux (ncore + 1).
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nmo,
                                                  ncore, nq, &kOne<double>, q_pp, nq, l_core,
                                                  nq * (ncore + 1), &kZero<double>, j_core, nmo)));
    for (int32_t i = 0; i < ncore; ++i) {
      // A dot over Q per p, of L's column (i, p) with itself.
      const double *const l_i = l_core + (naux * static_cast<size_t>(i));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, 1, 1, nq, &kOne<double>, l_i, nq,
          nq64 * ncore, l_i, nq, nq64 * ncore, &kZero<double>,
          k_core + (n_all * static_cast<size_t>(i)), 1, 1, nmo)));
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(j_pc.data(), j_core, j_pc.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(k_pc.data(), k_core, k_pc.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
  } else if (auto *const direct = std::get_if<Direct>(&out.jk_); direct != nullptr) {
    // One J and one K of C_i C_i^T per core i, (pp|ii) and (pi|pi) the
    // diagonals of C^T J C and C^T K C.
    for (size_t i = 0; i < core; ++i) {
      // |(C_i C_i^T)_cd| <= m_i^2, m_i = max_mu |(U C)_mu i|: the classes
      // prune for |D| <= 2, so the build is of s C_i C_i^T at s tau.
      double m = 0.0;
      for (size_t mu = 0; mu < n_cart; ++mu) {
        m = std::max(m, std::abs(c_core[mu + (n_cart * i)]));
      }
      const double s = m * m > 2.0 ? 2.0 / (m * m) : 1.0;
      scale[i] = s;
      const double *const c_i = out.c_cart_.data() + (n_cart * i);
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          blas, nc, nc, 1, &s, c_i, c_i, &kZero<double>, direct->density.data())));
      FBF_TRY(coulomb(res, direct->j_classes, direct->density, direct->j, direct->tau_j * s));
      FBF_TRY(exchange(res, direct->k_classes, direct->density, direct->k, direct->tau_k * s));
      FBF_TRY(
          gpu_status(gemmwich(blas, nmo, nc, out.c_cart_.data(), direct->j.data(), work, block)));
      FBF_TRY(diagonal(j_pc.data() + (n_all * i), block, n_all, n_all));
      FBF_TRY(
          gpu_status(gemmwich(blas, nmo, nc, out.c_cart_.data(), direct->k.data(), work, block)));
      FBF_TRY(diagonal(k_pc.data() + (n_all * i), block, n_all, n_all));
    }
  }
  FBF_TRY(res.sync());

  // gen_g_hop's h_diag over the packed pairs (p, q), doubled. With D the
  // 1-RDM's diagonal (2 core, gamma_tt active, 0 virtual):
  //   h_pp D_q + h_qq D_p - F_pp - F_qq, for every pair;
  //   + 2 Vca_pp + 6 (pq|pq) - 2 (pp|qq), q core;
  //   + Vc_qq gamma_uu - sum_v (6 (qu|qv) - 2 (qq|uv)) gamma_uv + H_act[(q, u), (q, u)],
  //     q core and p = ncore + u active;
  //   + Vc_pp gamma_uu + H_act[(p, u), (p, u)], q = ncore + u active.
  const auto occupation = [&](size_t p) {
    if (p < core) {
      return 2.0;
    }
    return p < occ ? gamma[(p - core) * (na + 1)] : 0.0;
  };
  std::vector<double> hd;
  hd.reserve(n_pairs);
  for (size_t q = 0; q < occ; ++q) {
    for (size_t p = q < core ? core : occ; p < n_all; ++p) {
      double value =
          (h_diag[p] * occupation(q)) + (h_diag[q] * occupation(p)) - f_diag[p] - f_diag[q];
      if (q < core) {
        value += (2.0 * vca_diag[p]) +
                 ((6.0 * k_pc[p + (n_all * q)] - 2.0 * j_pc[p + (n_all * q)]) / scale[q]);
        if (p < occ) {
          const size_t uu = p - core;
          double jk = 0.0;
          for (size_t v = 0; v < na; ++v) {
            // (qu|qv) and (qq|uv): row q of column q of block (u, v).
            const size_t at = q + (core * (q + (n_all * (uu + (na * v)))));
            jk += ((6.0 * papa_core[at]) - (2.0 * ppaa_core[at])) * gamma[uu + (na * v)];
          }
          value += (vc_diag[q] * gamma[uu * (na + 1)]) - jk + act_diag[q + (n_all * uu)];
        }
      } else {
        const size_t uu = q - core;
        value += (vc_diag[p] * gamma[uu * (na + 1)]) + act_diag[p + (n_all * uu)];
      }
      hd.push_back(2.0 * value);
    }
  }
  if (n_pairs > 0) {
    FBF_TRY(
        gpu_status(wwr::wwrMemcpyAsync(out.diagonal_.data(), hd.data(), n_pairs * sizeof(double),
                                       wwr::wwrMemcpyHostToDevice, stream)));
  }
  FBF_TRY(res.sync());
  return out;
}

Status Hessian::apply(DeviceResources &res, const int32_t k, const double *const x,
                      double *const y) {
  if (res.dev_idx() != dev_idx_) {
    return err_config("response::Hessian::apply: built on device {}, but the call passed device {}",
                      dev_idx_, res.dev_idx());
  }
  if (k < 0) {
    return err_config("response::Hessian::apply: the block has {} columns", k);
  }
  if (k == 0 || n_pairs_ == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t nmo = nmo_;
  const int32_t ncore = ncore_;
  const int32_t ncas = ncas_;
  const int32_t nocc = ncore + ncas;
  const int32_t nc = n_cart_;
  const auto n_all = static_cast<size_t>(nmo);
  const auto n_cart = static_cast<size_t>(nc);
  const size_t nn = n_all * n_all;
  const size_t square = n_cart * n_cart;
  const auto core = static_cast<size_t>(ncore);
  const auto na = static_cast<size_t>(ncas);
  const size_t occ = core + na;
  const auto pairs = static_cast<size_t>(n_pairs_);
  const auto columns = static_cast<size_t>(k);
  const int32_t n_rot = nmo - ncore; // the active and virtual columns
  const auto n_act = nmo * ncas;     // the order of H_act
  const bool core_jk = ncore > 0 && n_rot > 0;
  const Fitted *const fitted = std::get_if<Fitted>(&jk_);

  // Density-fitted, a block's columns build their J and K together: one
  // df_coulomb of every D3 and D', and one df_exchange_symmetrized of every
  // D3's and one of every D''s right factor against the left one they share.
  // As many columns at a time as hessian_batch_width finds room for in the
  // memory budget. One column, or integral-direct, is built as one column
  // always was.
  const size_t width = fitted != nullptr && core_jk && k > 1
                           ? FBF_TRY(hessian_batch_width(
                                 res, n_all, core, na, static_cast<size_t>(fitted->naux), columns))
                           : 1;
  const bool batched = width > 1;

  // Scratch, carved once for the block: kappa, X2 and the ncas x nmo block
  // va lands in, per column of width; h kappa, the other ncas x nmo block, W
  // and an n_cart^2 product, once; the densities D3 then D', per column; J C,
  // once; and the right factors R3 (n_cart x ncore) and R' (n_cart x nocc),
  // per column. Batched, every density's J and K too. coulomb and exchange do not take the
  // workspace, so the integral-direct path carves it from there; df_coulomb
  // and df_exchange do, so the density-fitted one allocates it.
  const std::array<size_t, 13> counts =
      hessian_apply_counts(n_all, core, na, n_cart, width, batched);
  const size_t total = std::reduce(counts.begin(), counts.end());
  Buffers::device<double> owned(fitted != nullptr ? total : 0, res.device());
  double *at =
      fitted != nullptr
          ? owned.data()
          : std::get<0>(FBF_TRY(res.device_regions<double>("response::Hessian::apply", total)));
  std::array<double *, counts.size()> regions{};
  for (size_t r = 0; r < counts.size(); ++r) {
    regions[r] = std::exchange(at, at + counts[r]);
  }
  const auto [kappas, x2s, hx, t_acts, t2, w, cart, densities, jc, rights3, rights, j_block,
              k_block] = regions;
  const double *const c_cart = c_cart_.data();
  const double *const c_cart_act = c_cart + (n_cart * core);
  // A gemm over a row block of kappa, x2 or V^c + V^a, whose leading dimension
  // is nmo rather than the block's own row count, which recipes/blas's packed
  // gemm cannot express. Every other product here is packed and is gemm.
  const auto block_gemm = [blas](wwr::wwrblasOperation_t op_a, wwr::wwrblasOperation_t op_b,
                                 int32_t m, int32_t n, int32_t kk, double alpha, const double *a,
                                 int32_t lda, const double *b, int32_t ldb, double beta, double *cc,
                                 int32_t ldc) {
    return gpu_status(wwr::gemm<double, int32_t>(blas, op_a, op_b, m, n, kk, &alpha, a, lda, b, ldb,
                                                 &beta, cc, ldc));
  };

  for (size_t first = 0; first < columns; first += width) {
    const size_t m = std::min(width, columns - first);
    // Column c's D3 and D' sit at c and m + c of the densities, so that the
    // batch's 2m lie side by side for df_coulomb, as its J and K do.
    for (size_t c = 0; c < m; ++c) {
      double *const kappa = kappas + (c * nn);
      double *const x2 = x2s + (c * nn);
      double *const t_act = t_acts + (c * na * n_all);
      double *const dm3 = densities + (c * square);
      double *const dm = densities + ((m + c) * square);
      double *const right3 = rights3 + (c * n_cart * core);
      double *const right = rights + (c * n_cart * occ);
      FBF_TRY(unpack_orbital_pairs(res, nmo, ncore, ncas, x + ((first + c) * pairs), kappa));

      // gen_g_hop's x2, then 2 (x2 - x2^T) packed. h D: (C^T h C) kappa over
      // the occupied columns, times the 1-RDM's occupied block.
      FBF_TRY(gpu_status(wwr::wwrMemsetAsync(x2, 0, nn * sizeof(double), stream)));
      FBF_TRY(gpu_status(
          gemm(blas, nmo, nocc, nmo, &kOne<double>, h_mo_.data(), kappa, &kZero<double>, hx)));
      FBF_TRY(gpu_status(
          gemm(blas, nmo, nocc, nocc, &kOne<double>, hx, dm1_occ_.data(), &kZero<double>, x2)));
      // - (F + F^T) kappa / 2.
      FBF_TRY(gpu_status(
          gemm(blas, nmo, &kNegHalf<double>, fock_sym_.data(), kappa, &kOne<double>, x2)));
      // Core rows: 2 kappa[core, ncore:] (V^c + V^a)[ncore:, :].
      if (core_jk) {
        FBF_TRY(block_gemm(wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, ncore, nmo, n_rot, 2.0,
                           kappa + (n_all * core), nmo, vhf_ca_.data() + core, nmo, 1.0, x2, nmo));
      }
      // Active rows, gamma (kappa[act, :] V^c + va), va's J and K below.
      if (ncas > 0) {
        FBF_TRY(block_gemm(wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, ncas, nmo, nmo, 1.0, kappa + core,
                           nmo, vhf_c_.data(), nmo, 0.0, t_act, ncas));
      }

      // update_jk_in_ah: D3 = C_core kappa[core, ncore:] C_rot^T + its
      // transpose, D4 = C_act gamma kappa[act, :] C^T + its transpose;
      // va = gamma C_act^T (2 J[D3] - K[D3]) C, and
      // vc = C_core^T (2 J[D'] - K[D']) C_rot, D' = 2 D3 + D4, onto the
      // core rows' active and virtual columns. All over U C. As two-factor
      // densities, D3 = L R3^T + R3 L^T with L = C_core and
      // R3 = C_rot kappa[core, ncore:]^T, and D' = L R^T + R L^T with
      // L = [C_core C_act], U C's first nocc columns, and R = [2 R3, R4],
      // R4 = C (gamma kappa[act, :])^T: the density-fitted K's factors.
      if (core_jk) {
        FBF_TRY(block_gemm(wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, ncore, nc, n_rot, 1.0,
                           kappa + (n_all * core), nmo, c_cart + (n_cart * core), nc, 0.0, w,
                           ncore));
        FBF_TRY(
            gpu_status(gemm(blas, nc, nc, ncore, &kOne<double>, c_cart, w, &kZero<double>, cart)));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T,
                                                      nc, nc, &kOne<double>, cart, nc,
                                                      &kOne<double>, cart, nc, dm3, nc)));
        FBF_TRY(block_gemm(wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc, ncore, n_rot, 1.0,
                           c_cart + (n_cart * core), nc, kappa + (n_all * core), nmo, 0.0, right3,
                           nc));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                      nc, ncore, &kTwo<double>, right3, nc,
                                                      &kZero<double>, right3, nc, right, nc)));
        if (ncas > 0) {
          // D' = 2 D3 + C_act (gamma kappa[act, :] C^T) + its transpose.
          FBF_TRY(block_gemm(wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, ncas, nmo, ncas, 1.0,
                             gamma_.data(), ncas, kappa + core, nmo, 0.0, t2, ncas));
          FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
              blas, ncas, nc, nmo, &kOne<double>, t2, c_cart, &kZero<double>, w)));
          FBF_TRY(gpu_status(
              gemm(blas, nc, nc, ncas, &kOne<double>, c_cart_act, w, &kZero<double>, cart)));
          FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
              blas, nc, ncas, nmo, &kOne<double>, c_cart, t2, &kZero<double>,
              right + (n_cart * core))));
          FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                        nc, nc, &kTwo<double>, dm3, nc,
                                                        &kOne<double>, cart, nc, dm, nc)));
          FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T,
                                                        nc, nc, &kOne<double>, dm, nc,
                                                        &kOne<double>, cart, nc, dm, nc)));
        } else {
          FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                        nc, nc, &kTwo<double>, dm3, nc,
                                                        &kZero<double>, dm3, nc, dm, nc)));
        }
      }
    }

    if (core_jk) {
      // Batched: 2 J - K[D] = 2 (J - K_sym), K_sym = K[D] / 2 of the
      // symmetrized form, over every density of the batch in one geam.
      if (batched) {
        const auto count = static_cast<int32_t>(m);
        const double *const b = fitted->b;
        FBF_TRY(df_coulomb(res, nc, fitted->naux, 2 * count, b, densities, j_block));
        FBF_TRY(df_exchange_symmetrized(res, nc, fitted->naux, ncore, count, b, c_cart, rights3,
                                        k_block));
        FBF_TRY(df_exchange_symmetrized(res, nc, fitted->naux, nocc, count, b, c_cart, rights,
                                        k_block + (m * square)));
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nc, 2 * count * nc, &kTwo<double>, j_block,
            nc, &kNegTwo<double>, k_block, nc, j_block, nc)));
      }
      for (size_t c = 0; c < m; ++c) {
        double *const x2 = x2s + (c * nn);
        double *const t_act = t_acts + (c * na * n_all);
        // Unbatched, two_j_minus_k's G is its own J matrix, so G3 is
        // read before G' is built.
        const double *g3 = nullptr;
        double s3 = 1.0;
        if (batched) {
          g3 = j_block + (c * square);
        } else {
          std::tie(g3, s3) = FBF_TRY(
              two_j_minus_k(res, densities + (c * square), rights3 + (c * n_cart * core), ncore));
        }
        if (ncas > 0) {
          FBF_TRY(
              gpu_status(gemm(blas, nc, nmo, nc, &kOne<double>, g3, c_cart, &kZero<double>, jc)));
          // BLAS reads alpha through a pointer, and an expression has no address.
          const double inv_s3 = 1.0 / s3;
          FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, ncas, nmo, nc, &inv_s3, c_cart_act, jc,
                                                     &kOne<double>, t_act)));
        }
        const double *g4 = nullptr;
        double s4 = 1.0;
        if (batched) {
          g4 = j_block + ((m + c) * square);
        } else {
          std::tie(g4, s4) = FBF_TRY(two_j_minus_k(res, densities + ((m + c) * square),
                                                   rights + (c * n_cart * occ), nocc));
        }
        FBF_TRY(gpu_status(gemm(blas, nc, n_rot, nc, &kOne<double>, g4, c_cart + (n_cart * core),
                                &kZero<double>, jc)));
        FBF_TRY(block_gemm(wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, ncore, n_rot, nc, 1.0 / s4, c_cart,
                           nc, jc, nc, 1.0, x2 + (n_all * core), nmo));
      }
    }

    for (size_t c = 0; c < m; ++c) {
      double *const kappa = kappas + (c * nn);
      double *const x2 = x2s + (c * nn);
      if (ncas > 0) {
        FBF_TRY(block_gemm(wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, ncas, nmo, ncas, 1.0,
                           gamma_.data(), ncas, t_acts + (c * na * n_all), ncas, 1.0, x2 + core,
                           nmo));
        // Active columns: H_act vec(kappa[:, act]), both nmo ncas contiguous.
        FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
            blas, wwr::WWRBLAS_OP_N, n_act, n_act, &kOne<double>, h_act_.data(), n_act,
            kappa + (n_all * core), 1, &kOne<double>, x2 + (n_all * core), 1)));
      }

      // 2 (x2 - x2^T), the derivative's, over kappa (no longer read).
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo,
                                                    nmo, &kTwo<double>, x2, nmo, &kNegTwo<double>,
                                                    x2, nmo, kappa, nmo)));
      FBF_TRY(pack_orbital_pairs(res, nmo, ncore, ncas, kappa, y + ((first + c) * pairs)));
    }
  }
  return {};
}

} // namespace free_breakfast::response
