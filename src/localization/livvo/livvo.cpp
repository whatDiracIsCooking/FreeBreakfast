/// @file
/// @brief valence_virtual_orbitals: the orthonormal complement of the occupied
/// orbitals within the IAOs, as an eigenproblem over the IAO basis; and
/// livvo_localize and valence_character over them.

module free_breakfast.localization.livvo;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.recipes.diagonalize;

#include "diagnostics.h"

namespace free_breakfast {

// One column-major gemm on res.stream(): C = op(A) op(B), op(A) m x k and
// op(B) k x n, each matrix packed.
[[nodiscard]] Status livvo_gemm(DeviceResources &res, const bool trans_a, const bool trans_b,
                                const int32_t m, const int32_t n, const int32_t k, const double *a,
                                const double *b, double *c) {
  const double one = kOne<double>;
  const double zero = kZero<double>;
  return gpu_status(
      wwr::gemm<double, int32_t>(res.blas().get(), trans_a ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N,
                                 trans_b ? wwr::WWRBLAS_OP_T : wwr::WWRBLAS_OP_N, m, n, k, &one, a,
                                 trans_a ? k : m, b, trans_b ? n : k, &zero, c, m));
}

Status valence_virtual_orbitals(DeviceResources &res, const BasisSet &basis,
                                const BasisSet &minimal, const int32_t n_occ, const double *s,
                                const double *c, const double *iao, double *vvo) {
  const auto n = static_cast<int32_t>(basis.nao_sphe());
  const auto m = static_cast<int32_t>(minimal.nao_sphe());
  const int32_t k = n_occ;
  if (minimal.natoms() != basis.natoms()) {
    return err_config("valence_virtual_orbitals: the minimal basis is over {} atoms, the basis "
                      "over {}",
                      minimal.natoms(), basis.natoms());
  }
  if (k < 0 || k > m) {
    return err_config("valence_virtual_orbitals: n_occ {} is outside 0..{}", k, m);
  }
  if (s == nullptr || c == nullptr || iao == nullptr || vvo == nullptr) {
    return err_config("valence_virtual_orbitals: s, c, iao and vvo must not be null");
  }
  const int32_t n_vvo = m - k;
  if (n_vvo == 0) {
    return {};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  if (k == 0) {
    // Nothing to be orthogonal to: the IAOs are already an orthonormal basis
    // of their span.
    return gpu_status(wwr::wwrMemcpyAsync(vvo, iao, static_cast<size_t>(n) * m * sizeof(double),
                                          wwr::wwrMemcpyDeviceToDevice, res.stream()));
  }
  const auto nk = static_cast<size_t>(n) * k;
  const auto mk = static_cast<size_t>(m) * k;
  const auto mm = static_cast<size_t>(m) * m;

  // P and its eigenvalues are held across diagonalize, which takes the
  // workspace, so they are a buffer of their own.
  Buffers::device<double> projector(mm + static_cast<size_t>(m), res.device());
  double *p = projector.data();
  double *w = p + mm;

  // C_iao = A^T S C, the occupied orbitals over the orthonormal IAOs: m x k
  // with orthonormal columns, since the IAOs span them. Both products are
  // dead before diagonalize, so they are the workspace's.
  {
    const auto [sc, c_iao] = res.device_regions<double>(nk, mk);
    FBF_TRY(livvo_gemm(res, false, false, n, k, n, s, c, sc));
    FBF_TRY(livvo_gemm(res, true, false, m, k, n, iao, sc, c_iao));
    FBF_TRY(livvo_gemm(res, false, true, m, m, k, c_iao, c_iao, p));
  }

  // P = C_iao C_iao^T projects R^m onto the occupied orbitals' coordinates,
  // so its eigenvalues are k ones and m - k zeros. syevd's ascending order
  // puts the zeros first: those m - k columns are the orthonormal complement,
  // the same as I - P's eigenvalue-one vectors without forming I.
  FBF_TRY(diagonalize<double>(res, m, p, w));

  // V = A U_0, U_0 the first m - k eigenvectors, packed in P's first columns.
  // Synchronized, as its siblings are, so V is ready when this returns.
  // projector's free is wwrFreeAsync on the same stream, ordered after the
  // gemm either way.
  FBF_TRY(livvo_gemm(res, false, false, n, n_vvo, m, iao, p, vvo));
  return res.sync();
}

Result<PipekMezeyInfo> livvo_localize(DeviceResources &res, const BasisSet &basis,
                                      const BasisSet &minimal, const int32_t n_occ, const double *s,
                                      const double *iao, const double *vvo, double *livvo,
                                      double *u, const PipekMezeyOptions &options) {
  const auto n = static_cast<int32_t>(basis.nao_sphe());
  const auto m = static_cast<int32_t>(minimal.nao_sphe());
  if (minimal.natoms() != basis.natoms()) {
    return err_config("livvo_localize: the minimal basis is over {} atoms, the basis over {}",
                      minimal.natoms(), basis.natoms());
  }
  if (n_occ < 0 || n_occ > m) {
    return err_config("livvo_localize: n_occ {} is outside 0..{}", n_occ, m);
  }
  if (s == nullptr || iao == nullptr || vvo == nullptr || livvo == nullptr) {
    return err_config("livvo_localize: s, iao, vvo and livvo must not be null");
  }
  if (livvo == vvo) {
    return err_config("livvo_localize: livvo must not alias vvo (V U is a gemm)");
  }
  const int32_t k = m - n_occ;
  if (k == 0) {
    return PipekMezeyInfo{};
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto mk = static_cast<size_t>(m) * k;

  // The VVOs over the orthonormal IAOs, V_IAO = A^T S V: lossless, since V is
  // A U_0. Over an orthonormal basis S is the identity, so
  // pipek_mezey_localize's Mulliken populations of V_IAO are the IAO ones, as
  // for the IBOs. V_IAO, its rotation and the identity are read by the sweep,
  // which takes the workspace, so they are buffers of their own.
  Buffers::device<double> v_iao(mk, res.device());
  Buffers::device<double> v_iao_local(mk, res.device());
  Buffers::device<double> rotation(static_cast<size_t>(k) * k, res.device());
  Buffers::device<double> identity(static_cast<size_t>(m) * m, res.device());
  {
    const auto [sv] = res.device_regions<double>(static_cast<size_t>(n) * k);
    FBF_TRY(livvo_gemm(res, false, false, n, k, n, s, vvo, sv));
    FBF_TRY(livvo_gemm(res, true, false, m, k, n, iao, sv, v_iao.data()));
  }
  std::vector<double> identity_host(static_cast<size_t>(m) * m, 0.0);
  for (int32_t r = 0; r < m; ++r) {
    identity_host[static_cast<size_t>(r) * (m + 1)] = 1.0;
  }
  // Pageable: the sync below keeps identity_host alive until it is read.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(identity.data(), identity_host.data(),
                                         identity_host.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(res.sync());

  Result<PipekMezeyInfo> info = pipek_mezey_localize(res, minimal, k, identity.data(), v_iao.data(),
                                                     v_iao_local.data(), rotation.data(), options);
  if (!info.has_value() && info.error().kind != ErrorKind::MaxIterationsExceeded) {
    return info;
  }
  // V U rather than A V_IAO U: the same orbitals, without the round trip.
  FBF_TRY(livvo_gemm(res, false, false, n, k, k, vvo, rotation.data(), livvo));
  if (u != nullptr) {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(u, rotation.data(),
                                           static_cast<size_t>(k) * k * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
  }
  // As valence_virtual_orbitals: the LIVVOs and U are ready on return.
  FBF_TRY(res.sync());
  return info;
}

Result<std::vector<double>> valence_character(DeviceResources &res, const BasisSet &basis,
                                              const int32_t n_orb, const int32_t n_vvo,
                                              const double *s, const double *c,
                                              const double *livvo) {
  const auto n = static_cast<int32_t>(basis.nao_sphe());
  if (n_orb < 0) {
    return err_config("valence_character: n_orb {} is negative", n_orb);
  }
  if (n_vvo < 0 || n_vvo > n) {
    return err_config("valence_character: n_vvo {} is outside 0..{}", n_vvo, n);
  }
  if (s == nullptr || c == nullptr || livvo == nullptr) {
    return err_config("valence_character: s, c and livvo must not be null");
  }
  std::vector<double> t_val(static_cast<size_t>(n_orb), 0.0);
  if (n_orb == 0 || n_vvo == 0) {
    return t_val;
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const auto n_c = static_cast<size_t>(n) * n_orb;
  const auto lc = static_cast<size_t>(n_vvo) * n_orb;

  // O = L^T S C, n_vvo x n_orb: column i is orbital i's overlap with each
  // LIVVO, and t_val its squared norm (eqs 11-12).
  std::vector<double> o_host(lc);
  {
    const auto [sc, o] = res.device_regions<double>(n_c, lc);
    FBF_TRY(livvo_gemm(res, false, false, n, n_orb, n, s, c, sc));
    FBF_TRY(livvo_gemm(res, true, false, n_vvo, n_orb, n, livvo, sc, o));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(o_host.data(), o, lc * sizeof(double),
                                           wwr::wwrMemcpyDeviceToHost, res.stream())));
    FBF_TRY(res.sync());
  }
  for (size_t i = 0; i < static_cast<size_t>(n_orb); ++i) {
    for (size_t l = 0; l < static_cast<size_t>(n_vvo); ++l) {
      const double x = o_host[l + (static_cast<size_t>(n_vvo) * i)];
      t_val[i] += x * x;
    }
  }
  return t_val;
}

} // namespace free_breakfast
