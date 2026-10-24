/// @file
/// @brief active_space_hamiltonian: the core's J and K, and one J per active
/// pair, contracted onto the active orbitals.

module free_breakfast.integrals.active_space;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;  // kOne, kZero, kTwo
import free_breakfast.core.primitives; // math::align_up
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemm, axpy, scal, dot, wwrMemcpyAsync, gpu_status
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.blas; // gemmwich
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;

#include "diagnostics.h"

namespace free_breakfast {

// The call's device scratch, carved from one workspace request, every region a
// whole number of alignment steps. coulomb and exchange, the only routines
// called while it is live, do not take the workspace.
struct ActiveSpaceScratch {
  double *c_cart; // U C over the ncore + ncas columns, nao_cart x (ncore + ncas)
  double *work;   // gemmwich's intermediate, nao_cart x ncas
  double *block;  // C_act^T G C_act, ncas x ncas
  double *p_sphe; // C_core C_core^T over the spherical AOs, for E_core

  ActiveSpaceScratch(DeviceResources &res, const size_t n_cart, const size_t n_sphe, const size_t m,
                     const size_t ncas) {
    size_t bytes = 0;
    const auto region = [&bytes](const size_t count) {
      const size_t at = bytes;
      bytes += math::align_up(count * sizeof(double), DeviceResources::kWorkspaceAlignment);
      return at;
    };
    const size_t at_c_cart = region(n_cart * m);
    const size_t at_work = region(n_cart * ncas);
    const size_t at_block = region(ncas * ncas);
    const size_t at_p_sphe = region(n_sphe * n_sphe);
    std::byte *const base = res.device_workspace(bytes).data();
    c_cart = reinterpret_cast<double *>(base + at_c_cart);
    work = reinterpret_cast<double *>(base + at_work);
    block = reinterpret_cast<double *>(base + at_block);
    p_sphe = reinterpret_cast<double *>(base + at_p_sphe);
  }
};

Result<ActiveSpaceHamiltonian> active_space_hamiltonian(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    const std::span<CoulombPairClass> j_classes, const std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas, const double tau_j,
    const double tau_k) {
  // Everything that can be refused is refused before the device is touched.
  const uint32_t n_sphe = basis.nao_sphe();
  const uint32_t n_cart = basis.nao_cart();
  const uint32_t nmo = c.n();
  if (ncas < 1) {
    return err_config("active_space_hamiltonian: ncas is {}, but must be at least 1", ncas);
  }
  if (ncore < 0) {
    return err_config("active_space_hamiltonian: ncore is {}, but must not be negative", ncore);
  }
  if (std::cmp_greater(int64_t{ncore} + ncas, nmo)) {
    return err_config("active_space_hamiltonian: ncore {} + ncas {} exceeds the {} orbitals", ncore,
                      ncas, nmo);
  }
  // Written so that NaN fails too.
  if (!(tau_j >= 0.0) || !(tau_k >= 0.0)) {
    return err_config("active_space_hamiltonian: tau_j ({}) and tau_k ({}) must be 0 or above",
                      tau_j, tau_k);
  }
  if (h.n() != n_sphe || nmo != n_sphe) {
    return err_config("active_space_hamiltonian: h is {} x {} and C {} x {}, but the basis has {} "
                      "spherical AOs",
                      h.n(), h.n(), nmo, nmo, n_sphe);
  }
  if (h.dev_idx() != res.dev_idx() || c.dev_idx() != res.dev_idx()) {
    return err_config("active_space_hamiltonian: h is on device {} and C on {}, but the call "
                      "passed device {}",
                      h.dev_idx(), c.dev_idx(), res.dev_idx());
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto ns = static_cast<int32_t>(n_sphe);
  const auto nc = static_cast<int32_t>(n_cart);
  const int32_t m = ncore + ncas;
  const auto na = static_cast<size_t>(ncas);

  ActiveSpaceHamiltonian out{.e_core = nuclear_repulsion_energy(molecule),
                             .ncas = ncas,
                             .h_eff = Buffers::device<double>(na * na, res.device()),
                             .eri = Buffers::device<double>(na * na * na * na, res.device())};
  // What coulomb and exchange read and write is their own matrix type, so
  // these three are allocated per call rather than carved from the workspace:
  // the core's density, then each pair's; J; and K.
  CartesianMatrixAO density(res, basis);
  CartesianMatrixAO j(res, basis);
  CartesianMatrixAO k(res, basis);
  const ActiveSpaceScratch scratch(res, n_cart, n_sphe, static_cast<size_t>(m), na);

  // The orbitals over the Cartesian AOs the kernels evaluate, U C: J[U P U^T]
  // projected by U C is J's spherical matrix projected by C.
  if (u != nullptr) {
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nc, m,
                                                  ns, &kOne<double>, u->data(), nc, c.data(), ns,
                                                  &kZero<double>, scratch.c_cart, nc)));
  } else {
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(scratch.c_cart, c.data(),
                                           sizeof(double) * n_cart * static_cast<size_t>(m),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  const double *const c_act = c.data() + (static_cast<size_t>(n_sphe) * ncore);
  const double *const c_cart_act = scratch.c_cart + (static_cast<size_t>(n_cart) * ncore);

  // h_eff's bare part, C_act^T h C_act, over the spherical AOs.
  FBF_TRY(gpu_status(gemmwich(blas, ncas, ns, c_act, h.data(), scratch.work, out.h_eff.data())));

  if (ncore > 0) {
    // P = C_core C_core^T (one spin's density), G = 2 J[P] - K[P] into j.
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
        blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc, nc, ncore, &kOne<double>, scratch.c_cart,
        nc, scratch.c_cart, nc, &kZero<double>, density.data(), nc)));
    FBF_TRY(coulomb(res, j_classes, density, j, tau_j));
    FBF_TRY(exchange(res, k_classes, density, k, tau_k));
    const auto len_cart = static_cast<int32_t>(j.num_elements());
    constexpr double kMinusOne = -1.0;
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, len_cart, &kTwo<double>, j.data(), 1)));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, len_cart, &kMinusOne, k.data(), 1, j.data(), 1)));

    // E_core = E_nuc + 2 tr(P h) + tr(P G); P, h and G are symmetric, so each
    // trace is the elementwise dot. h is spherical, so its P is too.
    FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, ns,
                                                  ns, ncore, &kOne<double>, c.data(), ns, c.data(),
                                                  ns, &kZero<double>, scratch.p_sphe, ns)));
    double ph = 0.0;
    double pg = 0.0;
    FBF_TRY(
        gpu_status(wwr::dot<double, int32_t>(blas, ns * ns, scratch.p_sphe, 1, h.data(), 1, &ph)));
    FBF_TRY(
        gpu_status(wwr::dot<double, int32_t>(blas, len_cart, density.data(), 1, j.data(), 1, &pg)));
    out.e_core += (2.0 * ph) + pg;

    // h_eff += C_act^T G C_act, over the Cartesian AOs.
    FBF_TRY(
        gpu_status(gemmwich(blas, ncas, nc, c_cart_act, j.data(), scratch.work, scratch.block)));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, ncas * ncas, &kOne<double>, scratch.block,
                                                  1, out.h_eff.data(), 1)));
  }

  // m_v = max_mu |(U C)_mu v|, on the host: n_cart x ncas doubles, read once.
  std::vector<double> c_cart_host(static_cast<size_t>(n_cart) * na);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c_cart_host.data(), c_cart_act,
                                         sizeof(double) * c_cart_host.size(),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<double> col_max(na, 0.0);
  for (size_t v = 0; v < na; ++v) {
    for (size_t mu = 0; mu < n_cart; ++mu) {
      col_max[v] = std::max(col_max[v], std::abs(c_cart_host[mu + (n_cart * v)]));
    }
  }

  // (tu|vw) = C_act^T J[D_vw] C_act, D_vw = (C_v C_w^T + C_w C_v^T) / 2, one J
  // per v <= w, on s D_vw at s tau_j and scaled back by 1 / s.
  const size_t block = na * na;
  for (int32_t w = 0; w < ncas; ++w) {
    for (int32_t v = 0; v <= w; ++v) {
      const double bound = col_max[static_cast<size_t>(v)] * col_max[static_cast<size_t>(w)];
      const double s = bound > 2.0 ? 2.0 / bound : 1.0;
      const double half_s = 0.5 * s;
      const double *const c_v = c_cart_act + (static_cast<size_t>(n_cart) * v);
      const double *const c_w = c_cart_act + (static_cast<size_t>(n_cart) * w);
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc,
                                                    nc, 1, &half_s, c_v, nc, c_w, nc,
                                                    &kZero<double>, density.data(), nc)));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc,
                                                    nc, 1, &half_s, c_w, nc, c_v, nc, &kOne<double>,
                                                    density.data(), nc)));
      FBF_TRY(coulomb(res, j_classes, density, j, tau_j * s));
      double *const vw = out.eri.data() + (block * (static_cast<size_t>(v) + (na * w)));
      FBF_TRY(gpu_status(gemmwich(blas, ncas, nc, c_cart_act, j.data(), scratch.work, vw)));
      if (s != 1.0) {
        const double inverse = 1.0 / s;
        FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, ncas * ncas, &inverse, vw, 1)));
      }
      if (v != w) {
        double *const wv = out.eri.data() + (block * (static_cast<size_t>(w) + (na * v)));
        FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(wv, vw, sizeof(double) * block,
                                               wwr::wwrMemcpyDeviceToDevice, stream)));
      }
    }
  }
  return out;
}

} // namespace free_breakfast
