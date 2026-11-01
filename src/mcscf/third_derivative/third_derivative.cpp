/// @file
/// @brief third_derivative: T(., v, z) of the state-averaged energy, by Faa di
/// Bruno over the exponential parametrization, every two-electron term over
/// the MO factor L and the occupied pair tensors.

module free_breakfast.mcscf.third_derivative;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne, kHalf, kNegHalf, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemmStridedBatched, geam, dgmm, dot, nrm2, memcpys
import free_breakfast.core.resources;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.rdm;
import free_breakfast.response.orbital_pairs;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// The scratch counts per column (the members' comments in the interface).
constexpr size_t kThirdMatrices = 10;
constexpr size_t kThirdMoBlocks = 2;
constexpr size_t kThirdPairBlocks = 4;
constexpr size_t kThirdOcc4 = 3;
constexpr size_t kThirdOcc2 = 2;
constexpr size_t kThirdCiBlocks = 12;
constexpr size_t kThirdSmall = 8;
// Weights within this of each other are equal; their sum is 1 within it.
constexpr double kThirdWeightTolerance = 1e-12;

// The occupied D and Gamma (E = sum h D + 1/2 sum (pq|rs) Gamma, Gamma with
// the 8-fold symmetry) of active (transition) RDMs g1 (row-major, symmetric),
// g2 (row-major, PySCF's convention) and the core weight n0, added into
// gamma (nocc^4, p fastest) and d (nocc^2).
void add_occupied_rdms(const size_t core, const size_t n, std::span<const double> g1,
                       std::span<const double> g2, const double n0, std::span<double> gamma,
                       std::span<double> d) {
  const size_t occ = core + n;
  const auto at = [occ](size_t p, size_t q, size_t r, size_t s) {
    return p + (occ * (q + (occ * (r + (occ * s)))));
  };
  std::vector<double> g(occ * occ * occ * occ, 0.0);
  for (size_t i = 0; i < core; ++i) {
    d[i + (occ * i)] += 2.0 * n0;
    for (size_t j = 0; j < core; ++j) {
      g[at(i, i, j, j)] += 4.0 * n0;
      g[at(i, j, j, i)] -= 2.0 * n0;
    }
    for (size_t t = 0; t < n; ++t) {
      for (size_t u = 0; u < n; ++u) {
        const double v = g1[(t * n) + u];
        g[at(i, i, core + t, core + u)] += 2.0 * v;
        g[at(core + t, core + u, i, i)] += 2.0 * v;
        g[at(core + t, i, i, core + u)] -= v;
        g[at(i, core + u, core + t, i)] -= v;
      }
    }
  }
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      d[(core + t) + (occ * (core + u))] += g1[(t * n) + u];
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          g[at(core + t, core + u, core + v, core + w)] += g2[(((((t * n) + u) * n) + v) * n) + w];
        }
      }
    }
  }
  // The average over the integrals' symmetries: either index pair swapped,
  // and the pairs swapped.
  for (size_t s = 0; s < occ; ++s) {
    for (size_t r = 0; r < occ; ++r) {
      for (size_t q = 0; q < occ; ++q) {
        for (size_t p = 0; p < occ; ++p) {
          gamma[at(p, q, r, s)] +=
              0.125 *
              (g[at(p, q, r, s)] + g[at(q, p, r, s)] + g[at(p, q, s, r)] + g[at(q, p, s, r)] +
               g[at(r, s, p, q)] + g[at(s, r, p, q)] + g[at(r, s, q, p)] + g[at(s, r, q, p)]);
        }
      }
    }
  }
}

Result<ThirdDerivative> ThirdDerivative::create(
    DeviceResources &res, const SphericalMatrixAO &h, const Buffers::device<double> &b,
    const int32_t naux, const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
    const ci::DeviceStringSet &alpha, const ci::DeviceStringSet &beta, const double *const roots,
    std::span<const double> weights, const double *const z) {
  const auto nmo = static_cast<int32_t>(c.n());
  if (ncas < 1 || ncore < 0 || ncore + ncas > nmo || naux < 1) {
    return err_config("mcscf::ThirdDerivative: needs ncas >= 1, ncore >= 0, ncore + ncas <= nmo "
                      "and naux >= 1 (nmo {}, ncore {}, ncas {}, naux {})",
                      nmo, ncore, ncas, naux);
  }
  const auto mo = static_cast<size_t>(nmo);
  if (h.n() != c.n() || b.num_elements() != static_cast<size_t>(naux) * mo * mo) {
    return err_config("mcscf::ThirdDerivative: h must be nmo x nmo and b naux nmo^2 (nmo {}, naux "
                      "{}; h {}, b {})",
                      nmo, naux, h.n(), b.num_elements());
  }
  if (alpha.n_orbitals() != ncas || beta.n_orbitals() != ncas) {
    return err_config(
        "mcscf::ThirdDerivative: the strings are over {} and {} orbitals, not ncas {}",
        alpha.n_orbitals(), beta.n_orbitals(), ncas);
  }
  if (roots == nullptr || z == nullptr) {
    return err_config("mcscf::ThirdDerivative: roots and z must be given");
  }
  if (weights.empty() || std::ranges::any_of(weights, [](double w) { return !(w > 0.0); }) ||
      std::abs(std::ranges::fold_left(weights, 0.0, std::plus{}) - 1.0) > kThirdWeightTolerance) {
    return err_config(
        "mcscf::ThirdDerivative: the weights must be positive and sum to 1 (got {} of them)",
        weights.size());
  }
  size_t equal_pairs = 0;
  size_t all_pairs = 0;
  for (size_t r = 0; r < weights.size(); ++r) {
    for (size_t s = r + 1; s < weights.size(); ++s) {
      ++all_pairs;
      equal_pairs += std::abs(weights[r] - weights[s]) < kThirdWeightTolerance ? 1 : 0;
    }
  }
  if (equal_pairs != 0 && equal_pairs != all_pairs) {
    return err_config("mcscf::ThirdDerivative: weights equal for {} of {} pairs of roots: the "
                      "rotations among the roots are either all parameters or none",
                      equal_pairs, all_pairs);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  ci::Sigma sigma0 = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  ci::Sigma sigma_z = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  ci::Sigma sigma_work = FBF_TRY(ci::Sigma::create(res, alpha, beta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const int32_t dim = sigma0.size();
  ThirdDerivative op{res,
                     nmo,
                     ncore,
                     ncas,
                     naux,
                     dim,
                     weights,
                     equal_pairs == 0 && all_pairs > 0,
                     std::move(sigma0),
                     std::move(sigma_z),
                     std::move(sigma_work),
                     std::move(rdm)};

  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const int32_t nocc = op.nocc_;
  const auto occ = static_cast<size_t>(nocc);
  const size_t nroots = weights.size();
  const auto nr = static_cast<int32_t>(nroots);
  const auto block = static_cast<size_t>(dim) * nroots;

  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(op.roots_.data(), roots, block * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  std::vector<double> two_w(weights.begin(), weights.end());
  for (double &w : two_w) {
    w *= 2.0;
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(op.two_w_.data(), two_w.data(), nroots * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));

  // h and L in the MO basis. B is naux x nao^2, P fastest ([P][n][m]): B C
  // over m (one gemm, B as (naux nao) x nao), then per q times C over n (a
  // strided-batched gemm), giving L[P, p, q]; one transpose puts Q slowest,
  // L[p, q, Q], each Q one nmo x nmo matrix.
  {
    const auto nx = static_cast<size_t>(naux);
    // Only BLAS runs while they live, which takes no workspace.
    const auto [half, full] = res.device_regions<double>(nx * mo * mo, nx * mo * mo);
    FBF_TRY(gpu_status(gemm(res.blas(), naux * nmo, nmo, nmo, &kOne<double>, b.data(), c.data(),
                            &kZero<double>, half)));
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux, nmo, nmo, &kOne<double>, half, naux,
        static_cast<int64_t>(nx * mo), c.data(), nmo, 0, &kZero<double>, full, naux,
        static_cast<int64_t>(nx * mo), nmo)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
        blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nmo * nmo, naux, &kOne<double>, full, naux,
        &kZero<double>, full, nmo * nmo, op.l_.data(), nmo * nmo)));
    FBF_TRY(
        gpu_status(gemm(res.blas(), nmo, &kOne<double>, h.data(), c.data(), &kZero<double>, half)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nmo, &kOne<double>, c.data(), half,
                                               &kZero<double>, op.h_mo_.data())));
    FBF_TRY(res.sync());
  }
  // Lo: L's occupied columns, packed; P_0: their occupied rows.
  FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
      op.lo_.data(), mo * occ * sizeof(double), op.l_.data(), mo * mo * sizeof(double),
      mo * occ * sizeof(double), static_cast<size_t>(naux), wwr::wwrMemcpyDeviceToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpy2DAsync(
      op.p0_.data(), occ * sizeof(double), op.lo_.data(), mo * sizeof(double), occ * sizeof(double),
      occ * static_cast<size_t>(naux), wwr::wwrMemcpyDeviceToDevice, stream)));

  // H_0 for the base sigma, from the same integrals.
  std::vector<double> h_oo(occ * occ);
  {
    std::vector<double> h_host(mo * mo);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h_host.data(), op.h_mo_.data(), mo * mo * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t q = 0; q < occ; ++q) {
      for (size_t p = 0; p < occ; ++p) {
        h_oo[p + (occ * q)] = h_host[p + (mo * q)];
      }
    }
  }
  const std::array base_pairs{
      std::pair<const double *, const double *>{op.p0_.data(), op.p0_.data()}};
  FBF_TRY(op.set_hamiltonian(res, h_oo, base_pairs, op.sigma0_));

  // z: kappa_z, d_z, Theta_z and A_z c.
  FBF_TRY(op.unpack(res, z, op.k_z_.data(), op.d_zp_.data(), op.theta_z_.data(), op.rot_z_.data()));
  FBF_TRY(gpu_status(gemm(res.blas(), nmo, &kOne<double>, op.h_mo_.data(), op.k_z_.data(),
                          &kZero<double>, op.h_k_z_.data())));
  FBF_TRY(op.right_transform(res, op.k_z_.data(), op.l_k_z_.data()));
  FBF_TRY(op.pair_transform(res, op.k_z_.data(), op.l_k_z_.data(), op.p_k_z_.data()));

  // The averaged RDMs (half the symmetric transition RDMs of c with itself)
  // and A_z c's.
  const size_t occ4 = occ * occ * occ * occ;
  {
    std::vector<double> gamma(occ4, 0.0);
    std::vector<double> d(occ * occ, 0.0);
    FBF_TRY(op.transition(res, op.roots_.data(), op.roots_.data(), gamma, d));
    for (double &v : gamma) {
      v *= 0.5;
    }
    for (double &v : d) {
      v *= 0.5;
    }
    std::vector<double> gamma_z(occ4, 0.0);
    std::vector<double> d_z(occ * occ, 0.0);
    FBF_TRY(op.transition(res, op.roots_.data(), op.rot_z_.data(), gamma_z, d_z));
    for (const auto &[from, to] :
         {std::pair{&gamma, &op.gamma0_}, std::pair{&gamma_z, &op.gamma_z_}, std::pair{&d, &op.d0_},
          std::pair{&d_z, &op.d_z_}}) {
      FBF_TRY(
          gpu_status(wwr::wwrMemcpyAsync(to->data(), from->data(), from->size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
    }
    FBF_TRY(res.sync());
  }
  const auto pq = static_cast<int32_t>(occ * occ);
  // Y_0 = Gamma_0 P_0 and Y_b = Gamma_0 P(K_z) + Gamma_z P_0.
  FBF_TRY(gpu_status(gemm(res.blas(), pq, naux, pq, &kOne<double>, op.gamma0_.data(), op.p0_.data(),
                          &kZero<double>, op.y0_.data())));
  FBF_TRY(gpu_status(gemm(res.blas(), pq, naux, pq, &kOne<double>, op.gamma0_.data(),
                          op.p_k_z_.data(), &kZero<double>, op.y_b_.data())));
  FBF_TRY(gpu_status(gemm(res.blas(), pq, naux, pq, &kOne<double>, op.gamma_z_.data(),
                          op.p0_.data(), &kOne<double>, op.y_b_.data())));

  // M_1 = 2 F at the averaged densities, and M_z its derivative along z:
  // 2 (L Y_b + (L K_z) Y_0 + (h K_z) D_0 + h D_z), occupied columns.
  const auto lmo_k = nocc * naux;
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(op.m1_.data(), 0, mo * mo * sizeof(double), stream)));
  FBF_TRY(gpu_status(wwr::wwrMemsetAsync(op.m_z_.data(), 0, mo * mo * sizeof(double), stream)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), nmo, nocc, lmo_k, &kTwo<double>, op.lo_.data(), op.y0_.data(), &kZero<double>,
      op.m1_.data())));
  FBF_TRY(gpu_status(gemm(res.blas(), nmo, nocc, nocc, &kTwo<double>, op.h_mo_.data(),
                          op.d0_.data(), &kOne<double>, op.m1_.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), nmo, nocc, lmo_k, &kTwo<double>, op.lo_.data(), op.y_b_.data(), &kZero<double>,
      op.m_z_.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      res.blas(), nmo, nocc, lmo_k, &kTwo<double>, op.l_k_z_.data(), op.y0_.data(), &kOne<double>,
      op.m_z_.data())));
  FBF_TRY(gpu_status(gemm(res.blas(), nmo, nocc, nocc, &kTwo<double>, op.h_k_z_.data(),
                          op.d0_.data(), &kOne<double>, op.m_z_.data())));
  FBF_TRY(gpu_status(gemm(res.blas(), nmo, nocc, nocc, &kTwo<double>, op.h_mo_.data(),
                          op.d_z_.data(), &kOne<double>, op.m_z_.data())));

  // H'[K_z] for its own sigma: h K_z + (h K_z)^T and P(K_z) P_0^T + P_0 P(K_z)^T.
  {
    std::vector<double> hk(mo * mo);
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(hk.data(), op.h_k_z_.data(), mo * mo * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    for (size_t q = 0; q < occ; ++q) {
      for (size_t p = 0; p < occ; ++p) {
        h_oo[p + (occ * q)] = hk[p + (mo * q)] + hk[q + (mo * p)];
      }
    }
    const std::array z_pairs{
        std::pair<const double *, const double *>{op.p_k_z_.data(), op.p0_.data()},
        std::pair<const double *, const double *>{op.p0_.data(), op.p_k_z_.data()}};
    FBF_TRY(op.set_hamiltonian(res, h_oo, z_pairs, op.sigma_z_));
  }
  // c_1 = 2 w H c and c_z = 2 w (H'[K_z] c + H A_z c).
  double *const tmp = op.ci_.data();
  FBF_TRY(op.sigma0_.apply(res, nr, op.roots_.data(), tmp));
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim, nr, tmp, dim,
                                                op.two_w_.data(), 1, op.c1_.data(), dim)));
  FBF_TRY(op.sigma_z_.apply(res, nr, op.roots_.data(), tmp));
  FBF_TRY(op.sigma0_.apply(res, nr, op.rot_z_.data(), tmp + block));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim, nr,
                                                &kOne<double>, tmp, dim, &kOne<double>, tmp + block,
                                                dim, tmp, dim)));
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim, nr, tmp, dim,
                                                op.two_w_.data(), 1, op.c_z_.data(), dim)));
  FBF_TRY(res.sync());
  return op;
}

ThirdDerivative::ThirdDerivative(DeviceResources &res, const int32_t nmo, const int32_t ncore,
                                 const int32_t ncas, const int32_t naux, const int32_t dim,
                                 std::span<const double> weights, const bool rotations,
                                 ci::Sigma sigma0, ci::Sigma sigma_z, ci::Sigma sigma_work,
                                 ci::Rdm rdm)
    : nmo_(nmo), ncore_(ncore), ncas_(ncas), nocc_(ncore + ncas), naux_(naux), dim_(dim),
      nroots_(static_cast<int32_t>(weights.size())),
      pairs_((ncore * (nmo - ncore)) + (ncas * (nmo - ncore - ncas))),
      n_theta_(rotations ? nroots_ * (nroots_ - 1) / 2 : 0),
      size_(pairs_ + (dim * nroots_) + n_theta_), weights_(weights.begin(), weights.end()),
      sigma0_(std::move(sigma0)), sigma_z_(std::move(sigma_z)), sigma_work_(std::move(sigma_work)),
      rdm_(std::move(rdm)), two_w_(weights.size(), res.device()),
      roots_(static_cast<size_t>(dim) * weights.size(), res.device()),
      h_mo_(static_cast<size_t>(nmo) * nmo, res.device()),
      l_(static_cast<size_t>(nmo) * nmo * naux, res.device()),
      lo_(static_cast<size_t>(nmo) * nocc_ * naux, res.device()),
      p0_(static_cast<size_t>(nocc_) * nocc_ * naux, res.device()),
      gamma0_(static_cast<size_t>(nocc_) * nocc_ * nocc_ * nocc_, res.device()),
      gamma_z_(static_cast<size_t>(nocc_) * nocc_ * nocc_ * nocc_, res.device()),
      d0_(static_cast<size_t>(nocc_) * nocc_, res.device()),
      d_z_(static_cast<size_t>(nocc_) * nocc_, res.device()),
      k_z_(static_cast<size_t>(nmo) * nmo, res.device()),
      h_k_z_(static_cast<size_t>(nmo) * nmo, res.device()),
      d_zp_(static_cast<size_t>(dim) * weights.size(), res.device()),
      theta_z_(weights.size() * weights.size(), res.device()),
      rot_z_(static_cast<size_t>(dim) * weights.size(), res.device()),
      l_k_z_(static_cast<size_t>(nmo) * nocc_ * naux, res.device()),
      p_k_z_(static_cast<size_t>(nocc_) * nocc_ * naux, res.device()),
      y0_(static_cast<size_t>(nocc_) * nocc_ * naux, res.device()),
      y_b_(static_cast<size_t>(nocc_) * nocc_ * naux, res.device()),
      m1_(static_cast<size_t>(nmo) * nmo, res.device()),
      m_z_(static_cast<size_t>(nmo) * nmo, res.device()),
      c1_(static_cast<size_t>(dim) * weights.size(), res.device()),
      c_z_(static_cast<size_t>(dim) * weights.size(), res.device()),
      mat_(kThirdMatrices * nmo * nmo, res.device()),
      lmo_(kThirdMoBlocks * nmo * nocc_ * naux, res.device()),
      pair_(kThirdPairBlocks * nocc_ * nocc_ * naux, res.device()),
      occ4_(kThirdOcc4 * nocc_ * nocc_ * nocc_ * nocc_, res.device()),
      occ2_(kThirdOcc2 * nocc_ * nocc_, res.device()),
      ci_(kThirdCiBlocks * dim * weights.size(), res.device()),
      small_(kThirdSmall * weights.size() * weights.size(), res.device()),
      act_((2 * static_cast<size_t>(ncas) * ncas) +
               (2 * static_cast<size_t>(ncas) * ncas * ncas * ncas),
           res.device()) {}

Status ThirdDerivative::unpack(DeviceResources &res, const double *const v, double *const kappa,
                               double *const d, double *const theta, double *const rotated) {
  const wwr::wwrStream_t stream = res.stream();
  const auto nroots = static_cast<size_t>(nroots_);
  const auto block = static_cast<size_t>(dim_) * nroots;
  FBF_TRY(response::unpack_orbital_pairs(res, nmo_, ncore_, ncas_, v, kappa));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(d, v + pairs_, block * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  // d -= C (C^T d).
  double *const overlaps = small_.data();
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nroots_, nroots_, dim_, &kOne<double>,
                                             roots_.data(), d, &kZero<double>, overlaps)));
  FBF_TRY(gpu_status(gemm(res.blas(), dim_, nroots_, nroots_, &kNegOne<double>, roots_.data(),
                          overlaps, &kOne<double>, d)));
  std::vector<double> th(nroots * nroots, 0.0);
  if (n_theta_ > 0) {
    std::vector<double> packed(static_cast<size_t>(n_theta_));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(packed.data(), v + pairs_ + block,
                                           packed.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    FBF_TRY(res.sync());
    size_t k = 0;
    for (size_t r = 0; r < nroots; ++r) {
      for (size_t s = r + 1; s < nroots; ++s) {
        th[s + (nroots * r)] = packed[k];
        th[r + (nroots * s)] = -packed[k];
        ++k;
      }
    }
  }
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta, th.data(), th.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(res.sync());
  return rotate(res, d, theta, nroots_, roots_.data(), rotated);
}

Status ThirdDerivative::rotate(DeviceResources &res, const double *const d,
                               const double *const theta, const int32_t k, const double *const v,
                               double *const out) {
  // small_'s first block is unpack's; this takes the next two.
  const auto nr2 = static_cast<size_t>(nroots_) * static_cast<size_t>(k);
  double *const ctv = small_.data() + nr2;
  double *const dtv = ctv + nr2;
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nroots_, k, dim_, &kOne<double>,
                                             roots_.data(), v, &kZero<double>, ctv)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nroots_, k, dim_, &kOne<double>, d, v,
                                             &kZero<double>, dtv)));
  FBF_TRY(gpu_status(
      gemm(res.blas(), nroots_, k, nroots_, &kOne<double>, theta, ctv, &kNegOne<double>, dtv)));
  // dtv now Theta C^T V - d^T V: out = d C^T V + C dtv.
  FBF_TRY(
      gpu_status(gemm(res.blas(), dim_, k, nroots_, &kOne<double>, d, ctv, &kZero<double>, out)));
  return gpu_status(
      gemm(res.blas(), dim_, k, nroots_, &kOne<double>, roots_.data(), dtv, &kOne<double>, out));
}

Status ThirdDerivative::right_transform(DeviceResources &res, const double *const x,
                                        double *const lx) {
  const auto mo = static_cast<int64_t>(nmo_);
  return gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo_, nocc_, nmo_, &kOne<double>,
      l_.data(), nmo_, mo * mo, x, nmo_, 0, &kZero<double>, lx, nmo_, mo * nocc_, naux_));
}

Status ThirdDerivative::pair_transform(DeviceResources &res, const double *const x,
                                       const double *const lx, double *const p) {
  const auto mo = static_cast<int64_t>(nmo_);
  const auto occ = static_cast<int64_t>(nocc_);
  // X^T L over the occupied pair, then + L X (lx's occupied rows).
  FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nocc_, nocc_, nmo_, &kOne<double>, x,
      nmo_, 0, lo_.data(), nmo_, mo * occ, &kZero<double>, p, nocc_, occ * occ, naux_)));
  return gpu_status(wwr::geam<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nocc_, nocc_ * naux_, &kOne<double>,
      lx, nmo_, &kOne<double>, p, nocc_, p, nocc_));
}

Status ThirdDerivative::transition(DeviceResources &res, const double *const a,
                                   const double *const b, std::span<double> gamma,
                                   std::span<double> d) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto nroots = static_cast<size_t>(nroots_);
  const size_t block = static_cast<size_t>(dim_) * nroots;
  const auto len = static_cast<int32_t>(block);
  double a_norm = 0.0;
  double b_norm = 0.0;
  FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, len, a, 1, &a_norm)));
  FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, len, b, 1, &b_norm)));
  if (b_norm == 0.0 || a_norm == 0.0) {
    return {};
  }
  // sum_r w_r (T(a_r, b_r) + T(b_r, a_r)) = [rdm(a + s b) - rdm(a - s b)] / 2s,
  // s putting s b on a's scale so neither term is lost in the difference.
  const double s = a_norm / b_norm;
  const double neg_s = -s;
  // ci_'s last two blocks.
  double *const plus = ci_.data() + ((kThirdCiBlocks - 2) * block);
  double *const minus = plus + block;
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, len, 1,
                                                &kOne<double>, a, len, &s, b, len, plus, len)));
  FBF_TRY(
      gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, len, 1,
                                            &kOne<double>, a, len, &neg_s, b, len, minus, len)));
  const auto n = static_cast<size_t>(ncas_);
  double *const g1p = act_.data();
  double *const g1m = g1p + (n * n);
  double *const g2p = g1m + (n * n);
  double *const g2m = g2p + (n * n * n * n);
  FBF_TRY(rdm_.rdm12(res, weights_, plus, g1p, g2p));
  FBF_TRY(rdm_.rdm12(res, weights_, minus, g1m, g2m));
  std::vector<double> host(act_.num_elements());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), act_.data(), host.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  double n0 = 0.0;
  for (size_t r = 0; r < nroots; ++r) {
    double dot = 0.0;
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, dim_, a + (r * static_cast<size_t>(dim_)), 1,
                                                 b + (r * static_cast<size_t>(dim_)), 1, &dot)));
    n0 += 2.0 * weights_[r] * dot;
  }
  FBF_TRY(res.sync());
  const double scale = 0.5 / s;
  std::vector<double> g1(n * n);
  std::vector<double> g2(n * n * n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      g1[(t * n) + u] = 0.5 * scale *
                        ((host[(t * n) + u] - host[(n * n) + (t * n) + u]) +
                         (host[(u * n) + t] - host[(n * n) + (u * n) + t]));
    }
  }
  const size_t n4 = n * n * n * n;
  for (size_t i = 0; i < n4; ++i) {
    g2[i] = scale * (host[(2 * n * n) + i] - host[(2 * n * n) + n4 + i]);
  }
  add_occupied_rdms(static_cast<size_t>(ncore_), n, g1, g2, n0, gamma, d);
  return {};
}

Status
ThirdDerivative::set_hamiltonian(DeviceResources &res, std::span<const double> h_oo,
                                 std::span<const std::pair<const double *, const double *>> g,
                                 ci::Sigma &sigma) {
  const auto occ = static_cast<size_t>(nocc_);
  const auto core = static_cast<size_t>(ncore_);
  const auto n = static_cast<size_t>(ncas_);
  const auto pq = static_cast<int32_t>(occ * occ);
  double *const ints = occ4_.data() + (2 * occ * occ * occ * occ);
  for (size_t i = 0; i < g.size(); ++i) {
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), pq, pq, naux_, &kOne<double>, g[i].first, g[i].second,
        i == 0 ? &kZero<double> : &kOne<double>, ints)));
  }
  std::vector<double> host(occ * occ * occ * occ);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), ints, host.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  const auto at = [occ](size_t p, size_t q, size_t r, size_t s) {
    return p + (occ * (q + (occ * (r + (occ * s)))));
  };
  double e_core = 0.0;
  for (size_t i = 0; i < core; ++i) {
    e_core += 2.0 * h_oo[i + (occ * i)];
    for (size_t j = 0; j < core; ++j) {
      e_core += (2.0 * host[at(i, i, j, j)]) - host[at(i, j, j, i)];
    }
  }
  std::vector<double> h_eff(n * n);
  std::vector<double> eri(n * n * n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      double sum = h_oo[(core + t) + (occ * (core + u))];
      for (size_t i = 0; i < core; ++i) {
        sum += (2.0 * host[at(core + t, core + u, i, i)]) - host[at(core + t, i, i, core + u)];
      }
      h_eff[(t * n) + u] = sum;
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          eri[(((((t * n) + u) * n) + v) * n) + w] =
              host[at(core + t, core + u, core + v, core + w)];
        }
      }
    }
  }
  return sigma.set_integrals(res, e_core, h_eff, eri);
}

Status ThirdDerivative::pull_ci(DeviceResources &res,
                                std::span<const std::pair<const double *, const double *>> mv,
                                double *const g_d, double *const x_theta) {
  // small_'s blocks 4 and 5 (unpack and rotate hold 0 to 3).
  const size_t nr2 = static_cast<size_t>(nroots_) * static_cast<size_t>(nroots_);
  double *const ctm = small_.data() + (4 * nr2);
  double *const ctv = ctm + nr2;
  for (size_t i = 0; i < mv.size(); ++i) {
    const auto [m, v] = mv[i];
    const double *const beta = i == 0 ? &kZero<double> : &kOne<double>;
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nroots_, nroots_, dim_, &kOne<double>,
                                               roots_.data(), m, &kZero<double>, ctm)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nroots_, nroots_, dim_, &kOne<double>,
                                               roots_.data(), v, &kZero<double>, ctv)));
    // G_D += M (C^T V)^T - V (C^T M)^T; X += (C^T M) (C^T V)^T.
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), dim_, nroots_, nroots_, &kOne<double>, m, ctv, beta, g_d)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), dim_, nroots_, nroots_, &kNegOne<double>, v, ctm, &kOne<double>, g_d)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        res.blas(), nroots_, nroots_, nroots_, &kOne<double>, ctm, ctv, beta, x_theta)));
  }
  // G_D off every root.
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nroots_, nroots_, dim_, &kOne<double>,
                                             roots_.data(), g_d, &kZero<double>, ctm)));
  return gpu_status(gemm(res.blas(), dim_, nroots_, nroots_, &kNegOne<double>, roots_.data(), ctm,
                         &kOne<double>, g_d));
}

Status ThirdDerivative::apply(DeviceResources &res, const int32_t k, const double *const v,
                              double *const y) {
  if (k < 0) {
    return err_config("mcscf::ThirdDerivative::apply: k must not be negative (got {})", k);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto mo = static_cast<size_t>(nmo_);
  const auto occ = static_cast<size_t>(nocc_);
  const auto nroots = static_cast<size_t>(nroots_);
  const size_t block = static_cast<size_t>(dim_) * nroots;
  const size_t mo2 = mo * mo;
  const size_t lmo = mo * occ * static_cast<size_t>(naux_);
  const size_t pair_len = occ * occ * static_cast<size_t>(naux_);
  const size_t occ4 = occ * occ * occ * occ;
  const auto pq = static_cast<int32_t>(occ * occ);
  const int32_t lmo_k = nocc_ * naux_;
  const int32_t nmo = nmo_;
  const int32_t nocc = nocc_;
  const int32_t dim = dim_;
  const int32_t nr = nroots_;
  const double sixth = 1.0 / 6.0;
  const double third = 1.0 / 3.0;
  const double neg_sixth = -sixth;

  // The column's regions.
  double *const ky = mat_.data();
  double *const xm = ky + mo2;   // X = (K_y K_z + K_z K_y) / 2
  double *const hky = xm + mo2;  // h K_y
  double *const hx = hky + mo2;  // h X
  double *const m_n = hx + mo2;  // the orbital covector, summed
  double *const m_y = m_n + mo2; // G2 along y
  double *const t1 = m_y + mo2;
  double *const t2 = t1 + mo2;
  double *const t3 = t2 + mo2;
  double *const t4 = t3 + mo2;
  double *const lky = lmo_.data();
  double *const lx = lky + lmo;
  double *const pky = pair_.data();
  double *const ps = pky + pair_len; // P(X) + P(K_y, K_z)
  double *const yc = ps + pair_len;
  double *const ya = yc + pair_len;
  double *const gamma_y = occ4_.data();
  double *const gamma_a = gamma_y + occ4;
  double *const d_y = occ2_.data();
  double *const d_a = d_y + (occ * occ);
  double *const dyp = ci_.data();       // y's d, off the roots
  double *const rot_y = dyp + block;    // A_y c
  double *const rot_yz = rot_y + block; // (A_y A_z + A_z A_y) c / 2
  double *const c_y = rot_yz + block;   // G2 along y, CI
  double *const c_n = c_y + block;      // the CI covector paired with c
  double *const w1 = c_n + block;
  double *const w2 = w1 + block;
  double *const v_dy = w2 + block;   // paired with A_y c
  double *const v_dz = v_dy + block; // paired with A_z c
  double *const g_d = v_dz + block;
  // small_: 0 unpack's, 1-3 rotate's, 4-5 pull_ci's, 6 Theta_y, 7 X_theta.
  const size_t nr2 = nroots * nroots;
  double *const theta_y = small_.data() + (6 * nr2);
  double *const x_theta = theta_y + nr2;

  const auto mm = [&](const double alpha, const double *a, const double *b, const double beta_v,
                      double *c) {
    return gpu_status(gemm(res.blas(), nmo, &alpha, a, b, &beta_v, c));
  };

  for (int32_t col = 0; col < k; ++col) {
    const double *const vc = v + (static_cast<size_t>(col) * static_cast<size_t>(size_));
    double *const yc_out = y + (static_cast<size_t>(col) * static_cast<size_t>(size_));

    // y's parameters, and the second-order rotations along y and z.
    FBF_TRY(unpack(res, vc, ky, dyp, theta_y, rot_y));
    FBF_TRY(mm(0.5, ky, k_z_.data(), 0.0, xm));
    FBF_TRY(mm(0.5, k_z_.data(), ky, 1.0, xm));
    FBF_TRY(rotate(res, dyp, theta_y, nr, rot_z_.data(), w1));
    FBF_TRY(rotate(res, d_zp_.data(), theta_z_.data(), nr, rot_y, w2));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &kHalf<double>, w1, dim, &kHalf<double>, w2,
                                                  dim, rot_yz, dim)));
    FBF_TRY(mm(1.0, h_mo_.data(), ky, 0.0, hky));
    FBF_TRY(mm(1.0, h_mo_.data(), xm, 0.0, hx));

    // The integrals: L K_y, L X, P(K_y), and P(X) + P(K_y, K_z).
    FBF_TRY(right_transform(res, ky, lky));
    FBF_TRY(right_transform(res, xm, lx));
    FBF_TRY(pair_transform(res, ky, lky, pky));
    FBF_TRY(pair_transform(res, xm, lx, ps));
    {
      const auto mo_l = static_cast<int64_t>(nmo_);
      const auto occ_l = static_cast<int64_t>(nocc_);
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nocc, nocc, nmo, &kOne<double>, ky, nmo, 0,
          l_k_z_.data(), nmo, mo_l * occ_l, &kOne<double>, ps, nocc, occ_l * occ_l, naux_)));
      FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
          blas, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, nocc, nocc, nmo, &kOne<double>, k_z_.data(),
          nmo, 0, lky, nmo, mo_l * occ_l, &kOne<double>, ps, nocc, occ_l * occ_l, naux_)));
    }

    // The RDMs: of A_y c, and of (A_y c, A_z c) with (c, (A_y A_z + A_z A_y) c / 2).
    {
      std::vector<double> gy(occ4, 0.0);
      std::vector<double> dy(occ * occ, 0.0);
      FBF_TRY(transition(res, roots_.data(), rot_y, gy, dy));
      std::vector<double> ga(occ4, 0.0);
      std::vector<double> da(occ * occ, 0.0);
      FBF_TRY(transition(res, rot_y, rot_z_.data(), ga, da));
      FBF_TRY(transition(res, roots_.data(), rot_yz, ga, da));
      for (const auto &[from, to] : {std::pair{&gy, gamma_y}, std::pair{&ga, gamma_a},
                                     std::pair{&dy, d_y}, std::pair{&da, d_a}}) {
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(to, from->data(), from->size() * sizeof(double),
                                               wwr::wwrMemcpyHostToDevice, stream)));
      }
      FBF_TRY(res.sync());
    }

    // Y_c = Gamma_0 P(K_y) + Gamma_y P_0; Y_a = Gamma_0 [P(X) + P(K_y, K_z)]
    // + Gamma_z P(K_y) + Gamma_y P(K_z) + Gamma_a P_0.
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma0_.data(), pky, &kZero<double>, yc)));
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma_y, p0_.data(), &kOne<double>, yc)));
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma0_.data(), ps, &kZero<double>, ya)));
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma_z_.data(), pky, &kOne<double>, ya)));
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma_y, p_k_z_.data(), &kOne<double>, ya)));
    FBF_TRY(gpu_status(
        gemm(res.blas(), pq, naux_, pq, &kOne<double>, gamma_a, p0_.data(), &kOne<double>, ya)));

    // The orbital covectors (occupied columns): N_y = G3(y, z) + G2(X, A_yz c)
    // = 2 [L Y_a + (L K_y) Y_b + (L K_z) Y_c + (L X) Y_0 + (h K_y) D_z +
    // (h K_z) D_y + (h X) D_0 + h D_a], and G2(y) = 2 [L Y_c + (L K_y) Y_0 +
    // (h K_y) D_0 + h D_y].
    FBF_TRY(gpu_status(wwr::wwrMemsetAsync(m_n, 0, 2 * mo2 * sizeof(double), stream)));
    const auto fock_add = [&](const double *lm, const double *ym, double *out) {
      return gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
          res.blas(), nmo, nocc, lmo_k, &kTwo<double>, lm, ym, &kOne<double>, out));
    };
    const auto h_add = [&](const double *hm, const double *dm, double *out) {
      return gpu_status(
          gemm(res.blas(), nmo, nocc, nocc, &kTwo<double>, hm, dm, &kOne<double>, out));
    };
    FBF_TRY(fock_add(lo_.data(), ya, m_n));
    FBF_TRY(fock_add(lky, y_b_.data(), m_n));
    FBF_TRY(fock_add(l_k_z_.data(), yc, m_n));
    FBF_TRY(fock_add(lx, y0_.data(), m_n));
    FBF_TRY(fock_add(lo_.data(), yc, m_y));
    FBF_TRY(fock_add(lky, y0_.data(), m_y));
    FBF_TRY(h_add(hky, d_z_.data(), m_n));
    FBF_TRY(h_add(h_k_z_.data(), d_y, m_n));
    FBF_TRY(h_add(hx, d0_.data(), m_n));
    FBF_TRY(h_add(h_mo_.data(), d_a, m_n));
    FBF_TRY(h_add(hky, d0_.data(), m_y));
    FBF_TRY(h_add(h_mo_.data(), d_y, m_y));

    // The CI covectors: G2(y) = 2 w (H'[K_y] c + H A_y c) and N_y = 2 w
    // ((H''[K_y, K_z] + H'[X]) c + H'[K_y] A_z c + H'[K_z] A_y c + H A_yz c).
    {
      std::vector<double> hk(mo2);
      std::vector<double> h2(mo2);
      // t1 = K_y^T (h K_z), so its sum with its transpose is K_y^T h K_z +
      // K_z^T h K_y.
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), nmo, &kOne<double>, ky, h_k_z_.data(),
                                                 &kZero<double>, t1)));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo,
                                                    nmo, &kOne<double>, t1, nmo, &kOne<double>, hx,
                                                    nmo, t2, nmo)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(hk.data(), hky, mo2 * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, stream)));
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(h2.data(), t2, mo2 * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, stream)));
      FBF_TRY(res.sync());
      std::vector<double> h1_oo(occ * occ);
      std::vector<double> h2_oo(occ * occ);
      for (size_t q = 0; q < occ; ++q) {
        for (size_t p = 0; p < occ; ++p) {
          h1_oo[p + (occ * q)] = hk[p + (mo * q)] + hk[q + (mo * p)];
          h2_oo[p + (occ * q)] = h2[p + (mo * q)] + h2[q + (mo * p)];
        }
      }
      const std::array first{std::pair<const double *, const double *>{pky, p0_.data()},
                             std::pair<const double *, const double *>{p0_.data(), pky}};
      FBF_TRY(set_hamiltonian(res, h1_oo, first, sigma_work_));
      FBF_TRY(sigma_work_.apply(res, nr, roots_.data(), w1)); // H'[K_y] c
      FBF_TRY(sigma_work_.apply(res, nr, rot_z_.data(), w2)); // H'[K_y] A_z c
      FBF_TRY(sigma0_.apply(res, nr, rot_y, c_y));            // H A_y c
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                    nr, &kOne<double>, w1, dim, &kOne<double>, c_y,
                                                    dim, c_y, dim)));
      FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim, nr, c_y,
                                                    dim, two_w_.data(), 1, c_y, dim)));
      FBF_TRY(sigma_z_.apply(res, nr, rot_y, c_n)); // H'[K_z] A_y c
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                    nr, &kOne<double>, w2, dim, &kOne<double>, c_n,
                                                    dim, c_n, dim)));
      FBF_TRY(sigma0_.apply(res, nr, rot_yz, w1)); // H A_yz c
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                    nr, &kOne<double>, w1, dim, &kOne<double>, c_n,
                                                    dim, c_n, dim)));
      const std::array second{std::pair<const double *, const double *>{ps, p0_.data()},
                              std::pair<const double *, const double *>{p0_.data(), ps},
                              std::pair<const double *, const double *>{pky, p_k_z_.data()},
                              std::pair<const double *, const double *>{p_k_z_.data(), pky}};
      FBF_TRY(set_hamiltonian(res, h2_oo, second, sigma_work_));
      FBF_TRY(sigma_work_.apply(res, nr, roots_.data(), w1));
      FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                    nr, &kOne<double>, w1, dim, &kOne<double>, c_n,
                                                    dim, c_n, dim)));
      FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, dim, nr, c_n,
                                                    dim, two_w_.data(), 1, c_n, dim)));
    }

    // The orbital pullback: N_y along a, plus G2(z) along (a, y), G2(y)
    // along (a, z) and G1 along (a, y, z), each a matrix whose
    // antisymmetric part pairs with K_a.
    FBF_TRY(mm(-0.5, m_z_.data(), ky, 1.0, m_n));
    FBF_TRY(mm(-0.5, ky, m_z_.data(), 1.0, m_n));
    FBF_TRY(mm(-0.5, m_y, k_z_.data(), 1.0, m_n));
    FBF_TRY(mm(-0.5, k_z_.data(), m_y, 1.0, m_n));
    FBF_TRY(mm(1.0, m1_.data(), k_z_.data(), 0.0, t1)); // M_1 K_z
    FBF_TRY(mm(1.0, m1_.data(), ky, 0.0, t2));          // M_1 K_y
    FBF_TRY(mm(1.0, ky, m1_.data(), 0.0, t3));          // K_y M_1
    FBF_TRY(mm(1.0, k_z_.data(), m1_.data(), 0.0, t4)); // K_z M_1
    FBF_TRY(mm(sixth, t1, ky, 1.0, m_n));
    FBF_TRY(mm(sixth, ky, t1, 1.0, m_n));
    FBF_TRY(mm(sixth, t2, k_z_.data(), 1.0, m_n));
    FBF_TRY(mm(sixth, k_z_.data(), t2, 1.0, m_n));
    FBF_TRY(mm(sixth, k_z_.data(), t3, 1.0, m_n));
    FBF_TRY(mm(sixth, ky, t4, 1.0, m_n));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nmo,
                                                  nmo, &kOne<double>, m_n, nmo, &kNegOne<double>,
                                                  m_n, nmo, t1, nmo)));
    FBF_TRY(response::pack_orbital_pairs(res, nmo, ncore_, ncas_, t1, yc_out));

    // The CI pullback, the pairs (M, V) of M . A_a V: with c, N_y - A_y c_z /
    // 2 - A_z c_y / 2 + (A_z A_y + A_y A_z) c_1 / 6; with A_y c, c_z / 2 -
    // A_z c_1 / 6; with A_z c, c_y / 2 - A_y c_1 / 6; with A_yz c, c_1 / 3.
    FBF_TRY(rotate(res, dyp, theta_y, nr, c_z_.data(), w1));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &kOne<double>, c_n, dim, &kNegHalf<double>,
                                                  w1, dim, c_n, dim)));
    FBF_TRY(rotate(res, d_zp_.data(), theta_z_.data(), nr, c_y, w1));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &kOne<double>, c_n, dim, &kNegHalf<double>,
                                                  w1, dim, c_n, dim)));
    // A_z c_1 into v_dy's place, A_y c_1 into v_dz's, then the second
    // rotations of each.
    FBF_TRY(rotate(res, d_zp_.data(), theta_z_.data(), nr, c1_.data(), w1)); // A_z c_1
    FBF_TRY(rotate(res, dyp, theta_y, nr, c1_.data(), w2));                  // A_y c_1
    FBF_TRY(rotate(res, dyp, theta_y, nr, w1, v_dy));                        // A_y A_z c_1
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &kOne<double>, c_n, dim, &sixth, v_dy, dim,
                                                  c_n, dim)));
    FBF_TRY(rotate(res, d_zp_.data(), theta_z_.data(), nr, w2, v_dz)); // A_z A_y c_1
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &kOne<double>, c_n, dim, &sixth, v_dz, dim,
                                                  c_n, dim)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &kHalf<double>, c_z_.data(), dim, &neg_sixth,
                                                  w1, dim, v_dy, dim)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &kHalf<double>, c_y, dim, &neg_sixth, w2, dim,
                                                  v_dz, dim)));
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, dim,
                                                  nr, &third, c1_.data(), dim, &kZero<double>,
                                                  c1_.data(), dim, w1, dim)));
    const std::array pulls{std::pair<const double *, const double *>{c_n, roots_.data()},
                           std::pair<const double *, const double *>{v_dy, rot_y},
                           std::pair<const double *, const double *>{v_dz, rot_z_.data()},
                           std::pair<const double *, const double *>{w1, rot_yz}};
    FBF_TRY(pull_ci(res, pulls, g_d, x_theta));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(yc_out + pairs_, g_d, block * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    if (n_theta_ > 0) {
      std::vector<double> x(nr2);
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(x.data(), x_theta, nr2 * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, stream)));
      FBF_TRY(res.sync());
      std::vector<double> packed;
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s) {
          packed.push_back(x[s + (nroots * r)] - x[r + (nroots * s)]);
        }
      }
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(yc_out + pairs_ + block, packed.data(),
                                             packed.size() * sizeof(double),
                                             wwr::wwrMemcpyHostToDevice, stream)));
    }
    FBF_TRY(res.sync());
  }
  return {};
}

} // namespace free_breakfast::mcscf
