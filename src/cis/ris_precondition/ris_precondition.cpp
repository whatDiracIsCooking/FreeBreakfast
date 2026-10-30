/// @file
/// @brief RisPreconditioner: the RIS basis fitted and L built, then each
/// call's Woodbury solve over the capacitance matrix.

module;

// The diagonal correction's launcher, cis.precondition's: declared in the GMF
// so it keeps external linkage (see the header).
#include "cis_precondition_bridge.h"

module free_breakfast.cis.ris_precondition;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne, kHalf
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // dgmm, wwrMemGetInfo, gpu_status
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
  const BasisSet aux = FBF_TRY(ris_basis(atoms, ris.theta, ris.shells));
  const auto naux = static_cast<int32_t>(aux.nao_sphe());
  const auto n_ov = static_cast<int32_t>(n_occ * n_virt);

  const ScopedDeviceIndex on{res.dev_idx()};
  // Past the device's memory an allocation aborts (CreatePolicy) rather than
  // returns, so the largest are refused here, before any: B, and L with its
  // scaled copy. three_center and b_mo refuse their own scratch.
  const auto aux_count = static_cast<size_t>(naux);
  const size_t b_bytes = sizeof(double) * aux_count * static_cast<size_t>(nao * nao);
  const size_t factor_bytes = 2 * sizeof(double) * aux_count * static_cast<size_t>(n_ov);
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  if (b_bytes + factor_bytes > free_bytes) {
    return err_unsupported("cis::RisPreconditioner::create: B over {} RIS and {} orbital "
                           "functions ({} bytes) and L twice over {} excitations ({} bytes) need "
                           "{} bytes of device memory, and device {} has {} free",
                           naux, nao, b_bytes, n_ov, factor_bytes, b_bytes + factor_bytes,
                           res.dev_idx(), free_bytes);
  }

  RisPreconditioner p{res, n_ov, naux, delta, max_columns};
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
    FBF_TRY(b_mo(res, naux, nmo, nmo, b.data(), c.data(),
                 OrbitalRange{.first = n_occ, .count = static_cast<int32_t>(n_virt)},
                 OrbitalRange{.first = 0, .count = n_occ}, p.factor_.data()));
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

RisPreconditioner::RisPreconditioner(DeviceResources &res, const int32_t n_ov, const int32_t naux,
                                     const double *const delta, const int32_t max_columns)
    : dev_idx_{res.dev_idx()}, n_ov_{n_ov}, naux_{naux}, delta_{delta},
      factor_(static_cast<size_t>(naux) * static_cast<size_t>(n_ov), res.device()),
      ones_(static_cast<size_t>(n_ov), res.device()),
      half_identity_(static_cast<size_t>(naux) * static_cast<size_t>(naux), res.device()),
      theta_(static_cast<size_t>(max_columns), res.device()),
      inverse_(static_cast<size_t>(n_ov), res.device()),
      scaled_(static_cast<size_t>(naux) * static_cast<size_t>(n_ov), res.device()),
      capacitance_(static_cast<size_t>(naux) * static_cast<size_t>(naux), res.device()),
      y_(static_cast<size_t>(naux) * static_cast<size_t>(max_columns), res.device()) {}

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
    // Dt⁻¹ under the diagonal correction's clamp: its kernel on a column of
    // ones, so the two cannot disagree.
    FBF_TRY(gpu_status(device::precondition(stream, n_ov_, 1, theta_.data() + col, delta_,
                                            ones_.data(), inverse_.data())));
    // Lᵀ Dt⁻¹, then the capacitance matrix ½ I − Lᵀ Dt⁻¹ L and the
    // right-hand side Lᵀ Dt⁻¹ r.
    FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, naux_, n_ov_,
                                                  factor_.data(), naux_, inverse_.data(), 1,
                                                  scaled_.data(), naux_)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(capacitance_.data(), half_identity_.data(),
                                           half_identity_.size_bytes(),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, naux_, naux_, n_ov_, &kNegOne<double>, scaled_.data(), factor_.data(), &kOne<double>,
        capacitance_.data())));
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
  return gpu_status(
      device::precondition(stream, n_ov_, columns, theta_.data(), delta_, correction, correction));
}

} // namespace free_breakfast::cis
