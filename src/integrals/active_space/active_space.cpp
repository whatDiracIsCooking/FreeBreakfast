/// @file
/// @brief active_space_hamiltonian and casscf_integrals: the core's J and K,
/// one J per active pair contracted onto the active orbitals (and, for
/// casscf_integrals, onto all of them), and for casscf_integrals one K per
/// part of each active pair's density.

module free_breakfast.integrals.active_space;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kTwo
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
// called while it is live, do not take the workspace. cols is the number of
// C's columns carried over the Cartesian AOs: ncore + ncas for the
// Hamiltonian, all nmo for casscf_integrals.
struct ActiveSpaceScratch {
  double *c_cart; // U C over the first cols columns, nao_cart x cols
  double *work;   // gemmwich's intermediate, nao_cart x cols
  double *block;  // one projected matrix, at most cols x cols
  double *p_sphe; // C_core C_core^T over the spherical AOs, for E_core

  ActiveSpaceScratch(DeviceResources &res, const size_t n_cart, const size_t n_sphe,
                     const size_t cols) {
    std::tie(c_cart, work, block, p_sphe) =
        res.device_regions<double>(n_cart * cols, n_cart * cols, cols * cols, n_sphe * n_sphe);
  }
};

// Everything either function refuses, refused before the device is touched;
// caller names the function in the message.
Status check_active_space(const std::string_view caller, const DeviceResources &res,
                          const BasisSet &basis, const SphericalMatrixAO &h,
                          const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
                          const double tau_j, const double tau_k) {
  const uint32_t n_sphe = basis.nao_sphe();
  const uint32_t nmo = c.n();
  if (ncas < 1) {
    return err_config("{}: ncas is {}, but must be at least 1", caller, ncas);
  }
  if (ncore < 0) {
    return err_config("{}: ncore is {}, but must not be negative", caller, ncore);
  }
  if (std::cmp_greater(int64_t{ncore} + ncas, nmo)) {
    return err_config("{}: ncore {} + ncas {} exceeds the {} orbitals", caller, ncore, ncas, nmo);
  }
  // Written so that NaN fails too.
  if (!(tau_j >= 0.0) || !(tau_k >= 0.0)) {
    return err_config("{}: tau_j ({}) and tau_k ({}) must be 0 or above", caller, tau_j, tau_k);
  }
  if (h.n() != n_sphe || nmo != n_sphe) {
    return err_config("{}: h is {} x {} and C {} x {}, but the basis has {} spherical AOs", caller,
                      h.n(), h.n(), nmo, nmo, n_sphe);
  }
  if (h.dev_idx() != res.dev_idx() || c.dev_idx() != res.dev_idx()) {
    return err_config("{}: h is on device {} and C on {}, but the call passed device {}", caller,
                      h.dev_idx(), c.dev_idx(), res.dev_idx());
  }
  return {};
}

// The build, on arguments check_active_space has passed. ppaa and papa are
// both null (the Hamiltonian alone) or both nmo^2 ncas^2 doubles on res's
// device, written as CasscfIntegrals documents them.
Result<ActiveSpaceHamiltonian> build_active_space(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    const std::span<CoulombPairClass> j_classes, const std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas, const double tau_j,
    const double tau_k, double *const ppaa, double *const papa) {
  const uint32_t n_sphe = basis.nao_sphe();
  const uint32_t n_cart = basis.nao_cart();
  const bool all_orbitals = ppaa != nullptr;
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto ns = static_cast<int32_t>(n_sphe);
  const auto nc = static_cast<int32_t>(n_cart);
  const auto nmo = static_cast<int32_t>(c.n());
  // C's columns carried over the Cartesian AOs.
  const int32_t m = all_orbitals ? nmo : ncore + ncas;
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
  const ActiveSpaceScratch scratch(res, n_cart, n_sphe, static_cast<size_t>(m));

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

  // The (v, w) and (w, v) blocks of an n x n-per-block buffer.
  const auto blocks = [na](double *const base, const size_t n, const int32_t v, const int32_t w) {
    return std::pair{base + (n * n * (static_cast<size_t>(v) + (na * w))),
                     base + (n * n * (static_cast<size_t>(w) + (na * v)))};
  };
  const auto copy_block = [stream](double *const to, const double *const from, const size_t n) {
    return gpu_status(wwr::wwrMemcpyAsync(to, from, sizeof(double) * n * n,
                                          wwr::wwrMemcpyDeviceToDevice, stream));
  };
  const auto n_all = static_cast<size_t>(nmo);
  const int32_t len_block = nmo * nmo;

  // Per pair v <= w, D_vw = (C_v C_w^T + C_w C_v^T) / 2, built of s D_vw at
  // s tau and scaled back by 1 / s: (tu|vw) = C_act^T J[D_vw] C_act and
  // (pq|vw) = C_all^T J[D_vw] C_all from its one J; (pv|qw) from K[D_vw] plus,
  // for v < w, K of the antisymmetric part (C_v C_w^T - C_w C_v^T) / 2.
  for (int32_t w = 0; w < ncas; ++w) {
    for (int32_t v = 0; v <= w; ++v) {
      const double bound = col_max[static_cast<size_t>(v)] * col_max[static_cast<size_t>(w)];
      const double s = bound > 2.0 ? 2.0 / bound : 1.0;
      const double half_s = 0.5 * s;
      const double minus_half_s = -half_s;
      const double inverse = 1.0 / s;
      const double minus_inverse = -inverse;
      const double *const c_v = c_cart_act + (static_cast<size_t>(n_cart) * v);
      const double *const c_w = c_cart_act + (static_cast<size_t>(n_cart) * w);
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc,
                                                    nc, 1, &half_s, c_v, nc, c_w, nc,
                                                    &kZero<double>, density.data(), nc)));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc,
                                                    nc, 1, &half_s, c_w, nc, c_v, nc, &kOne<double>,
                                                    density.data(), nc)));
      FBF_TRY(coulomb(res, j_classes, density, j, tau_j * s));
      const auto [vw, wv] = blocks(out.eri.data(), na, v, w);
      FBF_TRY(gpu_status(gemmwich(blas, ncas, nc, c_cart_act, j.data(), scratch.work, vw)));
      if (s != 1.0) {
        FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, ncas * ncas, &inverse, vw, 1)));
      }
      if (v != w) {
        FBF_TRY(copy_block(wv, vw, na));
      }
      if (!all_orbitals) {
        continue;
      }

      // (pq|vw), from the same J.
      const auto [pp_vw, pp_wv] = blocks(ppaa, n_all, v, w);
      FBF_TRY(gpu_status(gemmwich(blas, nmo, nc, scratch.c_cart, j.data(), scratch.work, pp_vw)));
      if (s != 1.0) {
        FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, len_block, &inverse, pp_vw, 1)));
      }
      if (v != w) {
        FBF_TRY(copy_block(pp_wv, pp_vw, n_all));
      }

      // (pv|qw): K[D_vw] into both blocks, then K of the antisymmetric part
      // added to (v, w) and taken from (w, v), K[C_w C_v^T] = K[C_v C_w^T]^T.
      const auto [pa_vw, pa_wv] = blocks(papa, n_all, v, w);
      FBF_TRY(exchange(res, k_classes, density, k, tau_k * s));
      FBF_TRY(gpu_status(gemmwich(blas, nmo, nc, scratch.c_cart, k.data(), scratch.work, pa_vw)));
      if (s != 1.0) {
        FBF_TRY(gpu_status(wwr::scal<double, int32_t>(blas, len_block, &inverse, pa_vw, 1)));
      }
      if (v == w) {
        continue;
      }
      FBF_TRY(copy_block(pa_wv, pa_vw, n_all));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc,
                                                    nc, 1, &minus_half_s, c_w, nc, c_v, nc,
                                                    &kZero<double>, density.data(), nc)));
      FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, nc,
                                                    nc, 1, &half_s, c_v, nc, c_w, nc, &kOne<double>,
                                                    density.data(), nc)));
      FBF_TRY(exchange(res, k_classes, density, k, tau_k * s, DensitySymmetry::Antisymmetric));
      FBF_TRY(gpu_status(
          gemmwich(blas, nmo, nc, scratch.c_cart, k.data(), scratch.work, scratch.block)));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, len_block, &inverse, scratch.block, 1, pa_vw, 1)));
      FBF_TRY(gpu_status(
          wwr::axpy<double, int32_t>(blas, len_block, &minus_inverse, scratch.block, 1, pa_wv, 1)));
    }
  }
  return out;
}

Result<ActiveSpaceHamiltonian> active_space_hamiltonian(
    DeviceResources &res, const BasisSet &basis, const Molecule &molecule,
    const SphericalMatrixAO &h, const std::shared_ptr<const Buffers::device<double>> &u,
    const std::span<CoulombPairClass> j_classes, const std::span<OrderedPairClass> k_classes,
    const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas, const double tau_j,
    const double tau_k) {
  FBF_TRY(
      check_active_space("active_space_hamiltonian", res, basis, h, c, ncore, ncas, tau_j, tau_k));
  return build_active_space(res, basis, molecule, h, u, j_classes, k_classes, c, ncore, ncas, tau_j,
                            tau_k, nullptr, nullptr);
}

Result<CasscfIntegrals> casscf_integrals(DeviceResources &res, const BasisSet &basis,
                                         const Molecule &molecule, const SphericalMatrixAO &h,
                                         const std::shared_ptr<const Buffers::device<double>> &u,
                                         const std::span<CoulombPairClass> j_classes,
                                         const std::span<OrderedPairClass> k_classes,
                                         const SphericalMatrixAO &c, const int32_t ncore,
                                         const int32_t ncas, const double tau_j,
                                         const double tau_k) {
  FBF_TRY(check_active_space("casscf_integrals", res, basis, h, c, ncore, ncas, tau_j, tau_k));
  const size_t nmo = c.n();
  const size_t count = nmo * nmo * static_cast<size_t>(ncas) * static_cast<size_t>(ncas);
  Buffers::device<double> ppaa(count, res.device());
  Buffers::device<double> papa(count, res.device());
  ActiveSpaceHamiltonian hamiltonian =
      FBF_TRY(build_active_space(res, basis, molecule, h, u, j_classes, k_classes, c, ncore, ncas,
                                 tau_j, tau_k, ppaa.data(), papa.data()));
  return CasscfIntegrals{
      .hamiltonian = std::move(hamiltonian), .ppaa = std::move(ppaa), .papa = std::move(papa)};
}

} // namespace free_breakfast
