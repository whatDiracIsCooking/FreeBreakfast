/// @file
/// @brief valence_virtual_orbitals: the orthonormal complement of the occupied
/// orbitals within the IAOs, as an eigenproblem over the IAO basis.

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
    auto scratch = wwr::extension::reinterpret_buffer_view<double>(
        res.device_workspace((nk + mk) * sizeof(double)));
    double *sc = scratch.data();
    double *c_iao = sc + nk;
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
  return livvo_gemm(res, false, false, n, n_vvo, m, iao, p, vvo);
}

} // namespace free_breakfast
