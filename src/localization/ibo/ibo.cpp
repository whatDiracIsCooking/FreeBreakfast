/// @file
/// @brief intrinsic_atomic_orbitals and ibo_localize: the IAO construction of
/// Knizia's Appendix C, and the IBO localization over it.

module free_breakfast.localization.ibo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne, kTwo
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, potrf, potrs, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.orthogonalize;
import free_breakfast.integrals.one_electron.overlap_mixed;

#include "diagnostics.h"

namespace free_breakfast {

// One column-major gemm on res.stream(): C = alpha op(A) op(B) + beta C, op(A)
// m x k and op(B) k x n, each matrix packed (its leading dimension its row
// count as stored).
[[nodiscard]] Status ibo_gemm(DeviceResources &res, const bool trans_a, const bool trans_b,
                              const int32_t m, const int32_t n, const int32_t k, const double alpha,
                              const double *a, const double *b, const double beta, double *c) {
  return gpu_status(
      wwr::gemm<double, int32_t>(res.blas().get(), trans_a ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N,
                                 trans_b ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N, m, n, k, &alpha,
                                 a, trans_a ? k : m, b, trans_b ? n : k, &beta, c, m));
}

// A pool buffer of @p count doubles on res.device().
[[nodiscard]] Buffers::device<double> ibo_buffer(DeviceResources &res, const size_t count) {
  return Buffers::device<double>(count, res.device());
}

[[nodiscard]] Status ibo_copy(DeviceResources &res, double *dst, const double *src,
                              const size_t count) {
  return gpu_status(wwr::wwrMemcpyAsync(dst, src, count * sizeof(double),
                                        wwr::wwrMemcpyDeviceToDevice, res.stream()));
}

[[nodiscard]] Status ibo_upload(DeviceResources &res, double *dst, const std::vector<double> &src) {
  // Pageable: the caller synchronizes before src goes away.
  return gpu_status(wwr::wwrMemcpyAsync(dst, src.data(), src.size() * sizeof(double),
                                        wwr::wwrMemcpyHostToDevice, res.stream()));
}

// The solver's flag: info < 0 is a bad argument, a bug here; info > 0 is
// potrf's leading minor that is not positive, the matrix's doing.
[[nodiscard]] Status ibo_check_info(DeviceResources &res, const std::string_view what) {
  const int32_t info = FBF_TRY(res.fetch_info());
  if (info < 0) {
    return err_gpu("intrinsic_atomic_orbitals: {} rejected argument {}", what, -info);
  }
  if (info > 0) {
    return err_numerical("intrinsic_atomic_orbitals: {} is not positive definite (minor {})", what,
                         info);
  }
  return {};
}

// S = L L^T in place, the lower triangle; the work is the device workspace.
[[nodiscard]] Status ibo_cholesky(DeviceResources &res, const int32_t n, double *a,
                                  const std::string_view what) {
  int32_t lwork = 0;
  FBF_TRY(gpu_status(wwr::potrf_bufferSize<double>(res.solver().get(), wwr::WWRBLAS_FILL_MODE_LOWER,
                                                   n, a, n, &lwork)));
  auto work = wwr::extension::reinterpret_buffer_view<double>(
      res.device_workspace(static_cast<size_t>(std::max(lwork, 1)) * sizeof(double)));
  FBF_TRY(gpu_status(wwr::potrf<double>(res.solver().get(), wwr::WWRBLAS_FILL_MODE_LOWER, n, a, n,
                                        work.data(), lwork, res.info().data())));
  return ibo_check_info(res, what);
}

// B <- S^-1 B for the n x nrhs B, over ibo_cholesky's factor @p l.
[[nodiscard]] Status ibo_solve(DeviceResources &res, const int32_t n, const int32_t nrhs,
                               const double *l, double *b) {
  FBF_TRY(gpu_status(wwr::potrs<double>(res.solver().get(), wwr::WWRBLAS_FILL_MODE_LOWER, n, nrhs,
                                        l, n, b, n, res.info().data())));
  return ibo_check_info(res, "potrs");
}

// X <- X (X^T S X)^-1/2 for the n x k X: Loewdin orthonormalization of its
// columns under S. @p sx (n x k) and @p xn (n x k) are scratch.
[[nodiscard]] Status ibo_orthonormalize(DeviceResources &res, const int32_t n, const int32_t k,
                                        const double *s, double *x, double *sx, double *xn) {
  Buffers::device<double> metric = ibo_buffer(res, static_cast<size_t>(k) * k);
  Buffers::device<double> root = ibo_buffer(res, static_cast<size_t>(k) * k);
  FBF_TRY(ibo_gemm(res, false, false, n, k, n, kOne<double>, s, x, kZero<double>, sx));
  FBF_TRY(ibo_gemm(res, true, false, k, k, n, kOne<double>, x, sx, kZero<double>, metric.data()));
  FBF_TRY(orthogonalize<double>(res, k, metric.data(), root.data()));
  FBF_TRY(ibo_gemm(res, false, false, n, k, k, kOne<double>, x, root.data(), kZero<double>, xn));
  return ibo_copy(res, x, xn, static_cast<size_t>(n) * k);
}

// The spherical overlaps S12 (n x m, the basis by the minimal one) and S2
// (m x m), with the minimal basis's functions normalized: S12 <- S12 D and
// S2 <- D S2 D, D = diag(S2)^-1/2.
[[nodiscard]] Status ibo_overlaps(DeviceResources &res, const BasisSet &basis,
                                  const BasisSet &minimal, Buffers::device<double> &s12,
                                  Buffers::device<double> &s2) {
  const auto n = static_cast<int32_t>(basis.nao_sphe());
  const auto m = static_cast<int32_t>(minimal.nao_sphe());
  FBF_TRY(overlap_mixed_spherical(res, basis, minimal, s12));
  FBF_TRY(overlap_mixed_spherical(res, minimal, minimal, s2));

  std::vector<double> s2_host(static_cast<size_t>(m) * m);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(s2_host.data(), s2.data(), s2_host.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> scale_host(static_cast<size_t>(m));
  for (int32_t r = 0; r < m; ++r) {
    const double norm = s2_host[static_cast<size_t>(r) * (m + 1)];
    if (!(norm > 0.0)) {
      return err_numerical("intrinsic_atomic_orbitals: minimal-basis function {} has norm {}", r,
                           norm);
    }
    scale_host[static_cast<size_t>(r)] = 1.0 / std::sqrt(norm);
  }
  Buffers::device<double> scale = ibo_buffer(res, scale_host.size());
  FBF_TRY(ibo_upload(res, scale.data(), scale_host));
  // dgmm in place: each output element reads only its own input element.
  const wwr::wwrblasHandle_t blas = res.blas().get();
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, n, m, s12.data(), n,
                                                scale.data(), 1, s12.data(), n)));
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_RIGHT, m, m, s2.data(), m,
                                                scale.data(), 1, s2.data(), m)));
  FBF_TRY(gpu_status(wwr::dgmm<double, int32_t>(blas, wwr::WWRBLAS_SIDE_LEFT, m, m, s2.data(), m,
                                                scale.data(), 1, s2.data(), m)));
  // scale_host is read by the upload above; wait before it goes out of scope.
  return res.sync();
}

Status intrinsic_atomic_orbitals(DeviceResources &res, const BasisSet &basis,
                                 const BasisSet &minimal, const int32_t n_occ, const double *s,
                                 const double *c, double *iao) {
  const auto n = static_cast<int32_t>(basis.nao_sphe());
  const auto m = static_cast<int32_t>(minimal.nao_sphe());
  const int32_t k = n_occ;
  if (basis.natoms() == 0) {
    return err_config("intrinsic_atomic_orbitals: the basis has no atoms; place it over a "
                      "molecule");
  }
  if (minimal.natoms() != basis.natoms()) {
    return err_config("intrinsic_atomic_orbitals: the minimal basis is over {} atoms, the basis "
                      "over {}",
                      minimal.natoms(), basis.natoms());
  }
  if (m > n) {
    return err_config("intrinsic_atomic_orbitals: the minimal basis has {} functions, more than "
                      "the basis's {}",
                      m, n);
  }
  if (k < 0 || k > m) {
    return err_config("intrinsic_atomic_orbitals: n_occ {} is outside 0..{}", k, m);
  }
  if (s == nullptr || c == nullptr || iao == nullptr) {
    return err_config("intrinsic_atomic_orbitals: s, c and iao must not be null");
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto nm = static_cast<size_t>(n) * m;
  const auto nk = static_cast<size_t>(n) * k;
  const auto mk = static_cast<size_t>(m) * k;

  Buffers::device<double> s12 = ibo_buffer(res, nm);
  Buffers::device<double> s2 = ibo_buffer(res, static_cast<size_t>(m) * m);
  FBF_TRY(ibo_overlaps(res, basis, minimal, s12, s2));

  // The two Cholesky factors; S1^-1 and S2^-1 are only ever solves.
  Buffers::device<double> l1 = ibo_buffer(res, static_cast<size_t>(n) * n);
  FBF_TRY(ibo_copy(res, l1.data(), s, static_cast<size_t>(n) * n));
  FBF_TRY(ibo_cholesky(res, n, l1.data(), "the basis's overlap"));
  FBF_TRY(ibo_cholesky(res, m, s2.data(), "the minimal basis's overlap"));

  // P12 = S1^-1 S12, the projector from the minimal basis onto the basis.
  Buffers::device<double> p12 = ibo_buffer(res, nm);
  FBF_TRY(ibo_copy(res, p12.data(), s12.data(), nm));
  FBF_TRY(ibo_solve(res, n, m, l1.data(), p12.data()));

  // C~ = orth(S1^-1 S12 S2^-1 S21 C): the occupied orbitals depolarized,
  // projected onto the minimal basis and back.
  Buffers::device<double> x = ibo_buffer(res, std::max(mk, size_t{1}));
  Buffers::device<double> ct = ibo_buffer(res, std::max(nk, size_t{1}));
  Buffers::device<double> sc = ibo_buffer(res, std::max(nk, size_t{1}));
  Buffers::device<double> scratch = ibo_buffer(res, std::max(nm, size_t{1}));
  Buffers::device<double> a = ibo_buffer(res, nm);
  FBF_TRY(ibo_copy(res, a.data(), p12.data(), nm));
  if (k > 0) {
    FBF_TRY(
        ibo_gemm(res, true, false, m, k, n, kOne<double>, s12.data(), c, kZero<double>, x.data()));
    FBF_TRY(ibo_solve(res, m, k, s2.data(), x.data()));
    FBF_TRY(ibo_gemm(res, false, false, n, k, m, kOne<double>, s12.data(), x.data(), kZero<double>,
                     ct.data()));
    FBF_TRY(ibo_solve(res, n, k, l1.data(), ct.data()));
    FBF_TRY(ibo_orthonormalize(res, n, k, s, ct.data(), sc.data(), scratch.data()));

    // A = (C C^T S1 C~ C~^T S1 + (1 - C C^T S1)(1 - C~ C~^T S1)) P12. With
    // S1 P12 = S12 every n x n product folds into an n_occ-wide one:
    // A = P12 - C E - C~ F + 2 C O F, E = C^T S12, F = C~^T S12, O = C^T S1 C~.
    Buffers::device<double> e = ibo_buffer(res, mk);
    Buffers::device<double> f = ibo_buffer(res, mk);
    Buffers::device<double> o = ibo_buffer(res, static_cast<size_t>(k) * k);
    Buffers::device<double> of = ibo_buffer(res, mk);
    FBF_TRY(ibo_gemm(res, false, false, n, k, n, kOne<double>, s, c, kZero<double>, sc.data()));
    FBF_TRY(
        ibo_gemm(res, true, false, k, m, n, kOne<double>, c, s12.data(), kZero<double>, e.data()));
    FBF_TRY(ibo_gemm(res, true, false, k, m, n, kOne<double>, ct.data(), s12.data(), kZero<double>,
                     f.data()));
    FBF_TRY(ibo_gemm(res, true, false, k, k, n, kOne<double>, sc.data(), ct.data(), kZero<double>,
                     o.data()));
    FBF_TRY(ibo_gemm(res, false, false, k, m, k, kOne<double>, o.data(), f.data(), kZero<double>,
                     of.data()));
    FBF_TRY(
        ibo_gemm(res, false, false, n, m, k, kNegOne<double>, c, e.data(), kOne<double>, a.data()));
    FBF_TRY(ibo_gemm(res, false, false, n, m, k, kNegOne<double>, ct.data(), f.data(), kOne<double>,
                     a.data()));
    FBF_TRY(
        ibo_gemm(res, false, false, n, m, k, kTwo<double>, c, of.data(), kOne<double>, a.data()));
  }

  // The IAOs: A orthonormalized under S1.
  Buffers::device<double> sa = ibo_buffer(res, nm);
  FBF_TRY(ibo_orthonormalize(res, n, m, s, a.data(), sa.data(), scratch.data()));
  return ibo_copy(res, iao, a.data(), nm);
}

Result<PipekMezeyInfo> ibo_localize(DeviceResources &res, const BasisSet &basis,
                                    const BasisSet &minimal, const int32_t n_occ, const double *s,
                                    const double *c, double *c_local, double *u,
                                    const PipekMezeyOptions &options) {
  if (c_local == nullptr) {
    return err_config("ibo_localize: c_local must not be null");
  }
  if (c_local == c) {
    return err_config("ibo_localize: c_local must not alias c (C U is a gemm)");
  }
  const auto n = static_cast<int32_t>(basis.nao_sphe());
  const auto m = static_cast<int32_t>(minimal.nao_sphe());
  const int32_t k = n_occ;
  const auto nm = static_cast<size_t>(n) * m;
  Buffers::device<double> iao = ibo_buffer(res, std::max(nm, size_t{1}));
  FBF_TRY(intrinsic_atomic_orbitals(res, basis, minimal, n_occ, s, c, iao.data()));
  if (k == 0) {
    return PipekMezeyInfo{};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto mk = static_cast<size_t>(m) * k;

  // The occupied orbitals over the orthonormal IAOs, C_IAO = A^T S1 C: lossless,
  // since the IAOs span them. Over an orthonormal basis S is the identity, so
  // pipek_mezey_localize's Mulliken populations of C_IAO are the IAO ones,
  // each IAO owned by its minimal-basis AO's atom.
  Buffers::device<double> sc = ibo_buffer(res, static_cast<size_t>(n) * k);
  Buffers::device<double> c_iao = ibo_buffer(res, mk);
  Buffers::device<double> c_iao_local = ibo_buffer(res, mk);
  Buffers::device<double> rotation = ibo_buffer(res, static_cast<size_t>(k) * k);
  Buffers::device<double> identity = ibo_buffer(res, static_cast<size_t>(m) * m);
  FBF_TRY(ibo_gemm(res, false, false, n, k, n, kOne<double>, s, c, kZero<double>, sc.data()));
  FBF_TRY(ibo_gemm(res, true, false, m, k, n, kOne<double>, iao.data(), sc.data(), kZero<double>,
                   c_iao.data()));
  std::vector<double> identity_host(static_cast<size_t>(m) * m, 0.0);
  for (int32_t r = 0; r < m; ++r) {
    identity_host[static_cast<size_t>(r) * (m + 1)] = 1.0;
  }
  FBF_TRY(ibo_upload(res, identity.data(), identity_host));
  FBF_TRY(res.sync());

  Result<PipekMezeyInfo> info = pipek_mezey_localize(res, minimal, k, identity.data(), c_iao.data(),
                                                     c_iao_local.data(), rotation.data(), options);
  if (!info.has_value() && info.error().kind != ErrorKind::MaxIterationsExceeded) {
    return info;
  }
  // C U rather than A C_IAO U: the same orbitals, without the round trip.
  FBF_TRY(ibo_gemm(res, false, false, n, k, k, kOne<double>, c, rotation.data(), kZero<double>,
                   c_local));
  if (u != nullptr) {
    FBF_TRY(ibo_copy(res, u, rotation.data(), static_cast<size_t>(k) * k));
  }
  return info;
}

} // namespace free_breakfast
