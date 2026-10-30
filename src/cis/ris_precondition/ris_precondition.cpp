/// @file
/// @brief RisPreconditioner: the RIS basis fitted and L (and with exchange
/// T_oo and T_vv) built, then each call's Woodbury solve over the capacitance
/// matrix, and with exchange each column's GMRES refinement.

module;

// The diagonal correction's launcher, cis.precondition's: declared in the GMF
// so it keeps external linkage (see the header).
#include "cis_precondition_bridge.h"

module free_breakfast.cis.ris_precondition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;  // kOne, kZero, kTwo, kNegOne, kHalf
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // dgmm, gemmStridedBatched, nrm2, scal, axpy, wwrMemGetInfo
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.blas;     // gemm
import free_breakfast.recipes.davidson; // DavidsonPreconditioner
import free_breakfast.recipes.lu;       // lu_solve
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.cis.ris_basis;

#include "diagnostics.h"

namespace free_breakfast::cis {

static_assert(DavidsonPreconditioner<RisPreconditioner, double>);

// The GMRES's lucky breakdown: the next Arnoldi vector's norm below which the
// Krylov space is taken as invariant. Each v_j is unit and the operator is
// I + M⁻¹ K_RI, near I, so ‖A v_j‖ is of order one and this is relative.
constexpr double kGmresBreakdown = 1e-14;

// The GMRES's device scratch, in the order refine carves it: the Krylov
// basis, K_RI's intermediate, its product, the two aux vectors of the
// Woodbury solve, and the two passes' Gram-Schmidt coefficients.
std::array<size_t, 6> gmres_regions(const int32_t n_ov, const int32_t naux,
                                    const int32_t max_inner_iters) {
  const auto n = static_cast<size_t>(n_ov);
  const auto aux = static_cast<size_t>(naux);
  const auto krylov = static_cast<size_t>(max_inner_iters) + 1;
  return {krylov * n, aux * n, n, aux, aux, 2 * krylov};
}

Result<RisPreconditioner> RisPreconditioner::create(
    DeviceResources &res, const BasisSet &basis, const std::span<const PrimitivePairClass> classes,
    const std::span<const Atom> atoms, const SphericalMatrixAO &c, const int32_t n_occ,
    const double *const delta, const int32_t max_columns, const RisOptions ris) {
  if (c.dev_idx() != res.dev_idx()) {
    return err_config("cis::RisPreconditioner::create: the MO coefficients are on device {}, but "
                      "the call passed device {}",
                      c.dev_idx(), res.dev_idx());
  }
  const auto nao = static_cast<int64_t>(c.n());
  if (nao != static_cast<int64_t>(basis.nao_sphe())) {
    return err_config("cis::RisPreconditioner::create: C is {} x {}, but the basis has {} "
                      "spherical functions",
                      nao, nao, basis.nao_sphe());
  }
  const int64_t n_virt = nao - n_occ;
  if (n_occ < 1 || n_virt < 1 || n_occ * n_virt > std::numeric_limits<int32_t>::max()) {
    return err_config("cis::RisPreconditioner::create: needs 1 <= n_occ < nao and n_occ * n_virt "
                      "to fit an int32_t (n_occ = {}, nao = {})",
                      n_occ, nao);
  }
  if (delta == nullptr || max_columns < 1) {
    return err_config("cis::RisPreconditioner::create: needs delta and max_columns >= 1 "
                      "(max_columns = {})",
                      max_columns);
  }
  if (ris.exchange && (ris.k_max_inner_iters < 0 || !(ris.k_inner_tol > 0.0))) {
    return err_config("cis::RisPreconditioner::create: the exchange needs k_max_inner_iters "
                      ">= 0 and k_inner_tol > 0 (k_max_inner_iters = {}, k_inner_tol = {})",
                      ris.k_max_inner_iters, ris.k_inner_tol);
  }
  const BasisSet aux = FBF_TRY(ris_basis(atoms, ris.theta, ris.shells));
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto n_ov = static_cast<int32_t>(n_occ * n_virt);
  // The exchange's int32_t dispatch: naux n_virt is its batched gemm's
  // leading dimension over T_vv, naux n_occ its second gemm's inner one.
  if (ris.exchange && (int64_t{naux} * n_virt > std::numeric_limits<int32_t>::max() ||
                       int64_t{naux} * n_occ > std::numeric_limits<int32_t>::max())) {
    return err_config("cis::RisPreconditioner::create: the exchange needs naux * n_virt and "
                      "naux * n_occ to fit an int32_t (naux = {}, n_occ = {}, n_virt = {})",
                      naux, n_occ, n_virt);
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  // Past the device's memory an allocation aborts (CreatePolicy) rather than
  // returns, so the largest are refused here, before any: B, and L with its
  // scaled copy. three_center and b_mo refuse their own scratch.
  const auto aux_count = static_cast<size_t>(naux);
  const size_t b_bytes = sizeof(double) * aux_count * static_cast<size_t>(nao * nao);
  const size_t factor_bytes = 2 * sizeof(double) * aux_count * static_cast<size_t>(n_ov);
  // With exchange, T_oo, T_vv and ½ C⁻¹, and the workspace's growth to the
  // GMRES's scratch.
  size_t exchange_bytes = 0;
  if (ris.exchange) {
    const auto occ = static_cast<size_t>(n_occ);
    const auto virt = static_cast<size_t>(n_virt);
    exchange_bytes = sizeof(double) * aux_count * ((occ * occ) + (virt * virt) + aux_count);
    size_t scratch_bytes = 0;
    for (const size_t count : gmres_regions(n_ov, naux, ris.k_max_inner_iters)) {
      scratch_bytes += math::align_up(sizeof(double) * count, DeviceResources::kWorkspaceAlignment);
    }
    if (scratch_bytes > res.device_workspace(0).num_elements()) {
      exchange_bytes += scratch_bytes;
    }
  }
  const size_t need_bytes = b_bytes + factor_bytes + exchange_bytes;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (need_bytes > free_bytes) {
    return err_unsupported("cis::RisPreconditioner::create: B over {} RIS and {} orbital "
                           "functions ({} bytes), L twice over {} excitations ({} bytes) and the "
                           "exchange's blocks and scratch ({} bytes) need {} bytes of device "
                           "memory, and device {} has {} free",
                           naux, nao, b_bytes, n_ov, factor_bytes, exchange_bytes, need_bytes,
                           res.dev_idx(), free_bytes);
  }

  RisPreconditioner p{res, n_occ, static_cast<int32_t>(n_virt), naux, delta, max_columns, ris};
  {
    // B as pipeline.eri fits it, over the RIS basis; the metric's factor and
    // B are freed at the scope's end, stream-ordered after b_mo reads B.
    MatrixAOPair m(res, aux, FBF_TRY(upload_spherical_transform_matrix(res, aux)));
    FBF_TRY(metric(res, aux, m));
    Buffers::device<double> b(b_bytes / sizeof(double), res.device());
    FBF_TRY(three_center(res, basis, classes, aux, b.data()));
    FBF_TRY(density_fit(res, static_cast<int32_t>(nao), naux, m.spherical().data(), b.data()));
    // Lᵀ with the virtual index fast: element (Q, a, i) at Q + naux (a +
    // n_virt i), so column a + n_virt i is excitation ia, as for DfSigma's
    // B_vo.
    const auto nmo = static_cast<int32_t>(nao);
    const OrbitalRange occupied{.first = 0, .count = n_occ};
    const OrbitalRange virtuals{.first = n_occ, .count = static_cast<int32_t>(n_virt)};
    FBF_TRY(b_mo(res, naux, nmo, nmo, b.data(), c.data(), virtuals, occupied, p.factor_.data()));
    // The exchange's oo and vv blocks, over the same B.
    if (ris.exchange) {
      FBF_TRY(
          b_mo(res, naux, nmo, nmo, b.data(), c.data(), occupied, occupied, p.factor_oo_.data()));
      FBF_TRY(
          b_mo(res, naux, nmo, nmo, b.data(), c.data(), virtuals, virtuals, p.factor_vv_.data()));
    }
  }
  // The GMRES's regions, grown to here, once B is freed, so that no call
  // allocates them. The workspace only grows, so a later, larger user leaves
  // it big enough.
  if (ris.exchange) {
    static_cast<void>(
        std::apply([&res](const auto... counts) { return res.device_regions<double>(counts...); },
                   gmres_regions(n_ov, naux, ris.k_max_inner_iters)));
  }

  // The ones and ½ I, uploaded once; the host copies go after the sync.
  const std::vector<double> ones(static_cast<size_t>(n_ov), 1.0);
  std::vector<double> half(aux_count * aux_count, 0.0);
  for (size_t q = 0; q < aux_count; ++q) {
    half[q + (q * aux_count)] = kHalf<double>;
  }
  const wwr::wwrStream_t stream = res.stream();
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(p.ones_.data(), ones.data(), ones.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(p.half_identity_.data(), half.data(),
                                         half.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                         stream)));
  FBF_TRY(res.sync());
  return p;
}

RisPreconditioner::RisPreconditioner(DeviceResources &res, const int32_t n_occ,
                                     const int32_t n_virt, const int32_t naux,
                                     const double *const delta, const int32_t max_columns,
                                     const RisOptions ris)
    : dev_idx_{res.dev_idx()}, n_occ_{n_occ}, n_virt_{n_virt}, n_ov_{n_occ * n_virt}, naux_{naux},
      exchange_{ris.exchange}, k_max_inner_iters_{ris.k_max_inner_iters},
      k_inner_tol_{ris.k_inner_tol}, delta_{delta},
      factor_(static_cast<size_t>(naux) * static_cast<size_t>(n_ov_), res.device()),
      factor_oo_(ris.exchange ? static_cast<size_t>(naux) * static_cast<size_t>(n_occ) *
                                    static_cast<size_t>(n_occ)
                              : 0,
                 res.device()),
      factor_vv_(ris.exchange ? static_cast<size_t>(naux) * static_cast<size_t>(n_virt) *
                                    static_cast<size_t>(n_virt)
                              : 0,
                 res.device()),
      ones_(static_cast<size_t>(n_ov_), res.device()),
      half_identity_(static_cast<size_t>(naux) * static_cast<size_t>(naux), res.device()),
      theta_(static_cast<size_t>(max_columns), res.device()),
      inverse_(static_cast<size_t>(n_ov_), res.device()),
      scaled_(static_cast<size_t>(naux) * static_cast<size_t>(n_ov_), res.device()),
      capacitance_(static_cast<size_t>(naux) * static_cast<size_t>(naux), res.device()),
      y_(static_cast<size_t>(naux) * static_cast<size_t>(max_columns), res.device()),
      half_inverse_(ris.exchange ? static_cast<size_t>(naux) * static_cast<size_t>(naux) : 0,
                    res.device()) {}

Status RisPreconditioner::build_capacitance(DeviceResources &res, const int32_t col) const {
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  // Dt⁻¹ under the diagonal correction's clamp: its kernel on a column of
  // ones, so the two cannot disagree.
  FBF_TRY(gpu_status(device::precondition(stream, n_ov_, 1, theta_.data() + col, delta_,
                                          ones_.data(), inverse_.data())));
  // Lᵀ Dt⁻¹, then the capacitance matrix ½ I − Lᵀ Dt⁻¹ L.
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, naux_, n_ov_,
                                                factor_.data(), naux_, inverse_.data(), 1,
                                                scaled_.data(), naux_)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(capacitance_.data(), half_identity_.data(),
                                         half_identity_.size_bytes(), wwr::wwrMemcpyDeviceToDevice,
                                         stream)));
  return gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
      blas, naux_, naux_, n_ov_, &kNegOne<double>, scaled_.data(), factor_.data(), &kOne<double>,
      capacitance_.data()));
}

Status RisPreconditioner::refine(DeviceResources &res, const int32_t col, double *const z) const {
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const int32_t m = k_max_inner_iters_;
  const auto n = static_cast<size_t>(n_ov_);
  const auto krylov = static_cast<size_t>(m) + 1;

  // ½ C⁻¹, from ½ I as lu_solve's right-hand side; an exactly singular C
  // drops the low-rank term from M⁻¹, as it does from the RIS-J answer.
  FBF_TRY(build_capacitance(res, col));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(half_inverse_.data(), half_identity_.data(),
                                         half_identity_.size_bytes(), wwr::wwrMemcpyDeviceToDevice,
                                         stream)));
  if (const int32_t singular =
          FBF_TRY(lu_solve(res, naux_, naux_, capacitance_.data(), half_inverse_.data()));
      singular != 0) {
    FBF_TRY(gpu_status(
        wwr::wwrMemsetAsync(half_inverse_.data(), 0, half_inverse_.size_bytes(), stream)));
  }
  // The workspace is free once lu_solve has returned, and nothing below
  // calls another of its users.
  const auto regions = gmres_regions(n_ov_, naux_, m);
  const auto [basis, w, kv, t, u, coef] = std::apply(
      [&res](const auto... counts) { return res.device_regions<double>(counts...); }, regions);

  // out = M⁻¹ K_RI in. K_RI in is DfSigma's exchange over one column: per a,
  // W_a = T_vv(:, a, :) in, naux x n_occ, then W's (naux n_occ) x n_virt
  // transpose times T_oo. Then M⁻¹ = (Dt − 2 L Lᵀ)⁻¹ by Woodbury, as
  // operator() applies it: Dt⁻¹ (v + L (2 · ½ C⁻¹) Lᵀ Dt⁻¹ v).
  const auto apply = [&](const double *const in, double *const out) -> Status {
    FBF_TRY(gpu_status(wwr::gemmStridedBatched<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, naux_, n_occ_, n_virt_, &kOne<double>,
        factor_vv_.data(), naux_ * n_virt_, naux_, in, n_virt_, 0, &kZero<double>, w, naux_,
        naux_ * n_occ_, n_virt_)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_virt_, n_occ_, naux_ * n_occ_, &kOne<double>,
                                               w, factor_oo_.data(), &kZero<double>, kv)));
    FBF_TRY(gpu_status(
        gemm(blas, naux_, 1, n_ov_, &kOne<double>, scaled_.data(), kv, &kZero<double>, t)));
    FBF_TRY(gpu_status(
        gemm(blas, naux_, 1, naux_, &kTwo<double>, half_inverse_.data(), t, &kZero<double>, u)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_ov_, 1, naux_, &kOne<double>, factor_.data(),
                                               u, &kOne<double>, kv)));
    return gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, n_ov_, 1, kv, n_ov_,
                                                 inverse_.data(), 1, out, n_ov_));
  };
  // A host-mode nrm2, whose result is the host's once it returns.
  const auto norm = [&](const double *const v) -> Result<double> {
    double result = 0.0;
    FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, n_ov_, v, 1, &result)));
    return result;
  };
  const auto scale = [&](const double factor, double *const v) -> Status {
    return gpu_status(wwr::scal<double, int32_t>(blas, n_ov_, &factor, v, 1));
  };

  // The system is (I + M⁻¹ K_RI) z = b with b = z₀, the column as it
  // stands, so the residual at z₀ is −M⁻¹ K_RI z₀ and z is only written at
  // the end: every early return leaves it at z₀.
  const double b_norm = FBF_TRY(norm(z));
  if (!(b_norm > 0.0) || !std::isfinite(b_norm)) {
    return {};
  }
  const double target = k_inner_tol_ * b_norm;
  FBF_TRY(apply(z, basis));
  FBF_TRY(scale(-1.0, basis));
  const double beta = FBF_TRY(norm(basis));
  if (!std::isfinite(beta) || beta <= target) {
    return {};
  }
  FBF_TRY(scale(1.0 / beta, basis));

  // The Hessenberg matrix (column-major, krylov rows), reduced to R by the
  // Givens rotations (cs, sn) as it grows, and g, the rotated β e₁, whose
  // last entry is the residual's norm.
  std::vector<double> h(krylov * static_cast<size_t>(m), 0.0);
  std::vector<double> cs(static_cast<size_t>(m), 0.0);
  std::vector<double> sn(static_cast<size_t>(m), 0.0);
  std::vector<double> g(krylov, 0.0);
  std::vector<double> dots(2 * krylov, 0.0);
  g[0] = beta;
  int32_t steps = 0;
  for (int32_t j = 0; j < m; ++j) {
    const auto jz = static_cast<size_t>(j);
    const double *const v = basis + (jz * n);
    double *const next = basis + ((jz + 1) * n);
    FBF_TRY(apply(v, next));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, n_ov_, &kOne<double>, v, 1, next, 1)));
    // Classical Gram-Schmidt against v_0 .. v_j, twice.
    for (size_t pass = 0; pass < 2; ++pass) {
      double *const c = coef + (pass * krylov);
      FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, j + 1, 1, n_ov_, &kOne<double>, basis, next,
                                                 &kZero<double>, c)));
      FBF_TRY(
          gpu_status(gemm(blas, n_ov_, 1, j + 1, &kNegOne<double>, basis, c, &kOne<double>, next)));
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(dots.data(), coef, dots.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, stream)));
    const double next_norm = FBF_TRY(norm(next));
    FBF_TRY(res.sync());
    double *const column = h.data() + (jz * krylov);
    for (size_t i = 0; i <= jz; ++i) {
      column[i] = dots[i] + dots[krylov + i];
      if (!std::isfinite(column[i])) {
        return {};
      }
    }
    column[jz + 1] = next_norm;
    if (!std::isfinite(next_norm)) {
      return {};
    }
    const bool breakdown = next_norm <= kGmresBreakdown;
    if (!breakdown) {
      FBF_TRY(scale(1.0 / next_norm, next));
    }
    // The earlier rotations on the new column, then the one that zeroes its
    // subdiagonal.
    for (size_t i = 0; i < jz; ++i) {
      const double upper = column[i];
      const double lower = column[i + 1];
      column[i] = (cs[i] * upper) + (sn[i] * lower);
      column[i + 1] = (-sn[i] * upper) + (cs[i] * lower);
    }
    const double radius = std::hypot(column[jz], column[jz + 1]);
    if (!(radius > 0.0)) {
      break;
    }
    cs[jz] = column[jz] / radius;
    sn[jz] = column[jz + 1] / radius;
    column[jz] = radius;
    column[jz + 1] = 0.0;
    g[jz + 1] = -sn[jz] * g[jz];
    g[jz] *= cs[jz];
    steps = j + 1;
    if (breakdown || std::abs(g[jz + 1]) <= target) {
      break;
    }
  }
  if (steps == 0) {
    return {};
  }

  // R y = g by back substitution, then z += V y.
  const auto k = static_cast<size_t>(steps);
  std::vector<double> y(k, 0.0);
  for (size_t i = k; i-- > 0;) {
    double sum = g[i];
    for (size_t l = i + 1; l < k; ++l) {
      sum -= h[i + (l * krylov)] * y[l];
    }
    y[i] = sum / h[i + (i * krylov)];
    if (!std::isfinite(y[i])) {
      return {};
    }
  }
  FBF_TRY(gpu_status(
      wwr::wwrMemcpyAsync(coef, y.data(), k * sizeof(double), wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(gpu_status(gemm(blas, n_ov_, 1, steps, &kOne<double>, basis, coef, &kOne<double>, z)));
  // y is pageable host memory, read by the upload.
  return res.sync();
}

Status RisPreconditioner::operator()(DeviceResources &res, const std::span<const double> theta,
                                     const double *const /*ritz_vectors*/,
                                     const double *const residual, double *const correction) const {
  if (res.dev_idx() != dev_idx_) {
    return err_config("cis::RisPreconditioner: built on device {}, called on device {}", dev_idx_,
                      res.dev_idx());
  }
  if (theta.size() > theta_.num_elements()) {
    return err_config("cis::RisPreconditioner: {} columns, more than the {} it was built for",
                      theta.size(), theta_.num_elements());
  }
  if (theta.empty()) {
    return {};
  }
  const ScopedDeviceIndex on{dev_idx_};
  const wwr::wwrStream_t stream = res.stream();
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto columns = static_cast<int32_t>(theta.size());
  const auto n = static_cast<size_t>(n_ov_);
  const auto aux = static_cast<size_t>(naux_);
  // theta is the caller's pageable memory; the first lu_solve synchronizes
  // the stream, after which the upload has landed (docs/style.md "Host-device
  // copies").
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(theta_.data(), theta.data(), theta.size_bytes(),
                                         wwr::wwrMemcpyHostToDevice, stream)));

  for (int32_t col = 0; col < columns; ++col) {
    const double *const r = residual + (static_cast<size_t>(col) * n);
    double *const y = y_.data() + (static_cast<size_t>(col) * aux);
    // The capacitance matrix, and the right-hand side Lᵀ Dt⁻¹ r.
    FBF_TRY(build_capacitance(res, col));
    FBF_TRY(gpu_status(
        gemm(blas, naux_, 1, n_ov_, &kOne<double>, scaled_.data(), r, &kZero<double>, y)));
    // An exactly singular capacitance matrix drops the low-rank term: y = 0
    // leaves this column the diagonal correction.
    if (const int32_t singular = FBF_TRY(lu_solve(res, naux_, 1, capacitance_.data(), y));
        singular != 0) {
      FBF_TRY(gpu_status(wwr::wwrMemsetAsync(y, 0, aux * sizeof(double), stream)));
    }
  }

  // z = Dt⁻¹ (r + L y) over the whole block: r copied unless it is already
  // in place, L y accumulated into it, and the diagonal correction applied in
  // place. Every column's r has been read by now, so the copy may overwrite
  // it.
  if (correction != residual) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(correction, residual, n * theta.size() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  FBF_TRY(
      gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n_ov_, columns, naux_, &kOne<double>, factor_.data(),
                                         y_.data(), &kOne<double>, correction)));
  FBF_TRY(gpu_status(
      device::precondition(stream, n_ov_, columns, theta_.data(), delta_, correction, correction)));
  // RIS-JK: each column refined from that RIS-J answer. With no inner
  // iterations allowed, the RIS-J answer stands, bytes and all.
  if (exchange_ && k_max_inner_iters_ > 0) {
    for (int32_t col = 0; col < columns; ++col) {
      FBF_TRY(refine(res, col, correction + (static_cast<size_t>(col) * n)));
    }
  }
  return {};
}

} // namespace free_breakfast::cis
