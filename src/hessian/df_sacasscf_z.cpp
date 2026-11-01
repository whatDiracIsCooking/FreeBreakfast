/// @file
/// @brief df_sacasscf_z: a state-averaged DF-CASSCF Hessian's z terms at one
/// root's z, L_xy's z part (df_casscf_exy along the z_o rotation, and at the
/// transition RDMs of z_c by the +/- polarization) and L_xlambda (root I's
/// right-hand side from mcscf::df_derivatives plus I^x z from
/// mcscf::ThirdDerivative::apply_integrals), assembled from the merged parts.

module free_breakfast.hessian;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // axpy, wwrMemcpyAsync, ScopedDeviceIndex, gpu_status
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.mmd;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.rdm;
import free_breakfast.response.orbital_pairs;
import free_breakfast.mcscf.derivative;
import free_breakfast.mcscf.third_derivative;

#include "diagnostics.h"

namespace free_breakfast::hessian {

// Weights within this of each other are equal (mcscf::ThirdDerivative's
// rule): the rotations among the roots are then not parameters.
constexpr double kSaWeightTolerance = 1e-12;

Result<SaCasscfZTerms>
df_sacasscf_z(DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
              const std::span<const PrimitivePairClass> classes, const BasisSet &aux,
              const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
              const SphericalMatrixAO &c, const int32_t ncore, const int32_t ncas,
              const int32_t nalpha, const int32_t nbeta, const Buffers::device<double> &roots,
              const std::span<const double> weights, const int32_t root,
              const Buffers::device<double> &z) {
  const size_t nao = basis.nao_sphe();
  const size_t naux = aux.nao_sphe();
  const uint32_t n_atom = basis.natoms();
  const size_t nroots = weights.size();
  if (c.dev_idx() != res.dev_idx() || c.n() != nao || molecule.natoms() != n_atom ||
      aux.natoms() != n_atom || ncore < 0 || ncas < 1 ||
      int64_t{ncore} + int64_t{ncas} > static_cast<int64_t>(nao) || nroots < 2 || root < 0 ||
      std::cmp_greater_equal(root, nroots)) {
    return err_config("hessian::df_sacasscf_z: needs C nao square on device {}, a molecule and "
                      "two bases on the same atoms, ncore >= 0, ncas >= 1, ncore + ncas <= nao, "
                      "two roots or more and a root among them (C {} square on device {}, {}, {} "
                      "and {} atoms, ncore {}, ncas {}, nao {}, {} weights, root {})",
                      res.dev_idx(), c.n(), c.dev_idx(), molecule.natoms(), n_atom, aux.natoms(),
                      ncore, ncas, nao, nroots, root);
  }
  size_t equal_pairs = 0;
  size_t all_pairs = 0;
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = r + 1; s < nroots; ++s) {
      ++all_pairs;
      equal_pairs += std::abs(weights[r] - weights[s]) < kSaWeightTolerance ? 1 : 0;
    }
  }
  if (equal_pairs != 0 && equal_pairs != all_pairs) {
    return err_config("hessian::df_sacasscf_z: weights equal for {} of {} pairs of roots: the "
                      "rotations among the roots are either all parameters or none",
                      equal_pairs, all_pairs);
  }
  const ScopedDeviceIndex on{res.dev_idx()};
  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  const auto n_act = static_cast<size_t>(ncas);
  const auto nocc = static_cast<size_t>(ncore) + n_act;
  const size_t pairs = (n_act * static_cast<size_t>(ncore)) + ((nao - nocc) * nocc);
  const size_t n_theta = equal_pairs == 0 ? all_pairs : 0;
  const size_t n_ci = dim * nroots;
  const size_t size = pairs + n_ci + n_theta;
  if (roots.num_elements() != n_ci || z.num_elements() != size) {
    return err_config("hessian::df_sacasscf_z: {} roots of {} determinants and z of {} "
                      "parameters (pairs, CI{}), got {} and {} doubles",
                      nroots, dim, size, n_theta == 0 ? "" : ", the roots' rotations",
                      roots.num_elements(), z.num_elements());
  }
  const auto n_x = static_cast<size_t>(3) * n_atom;
  const auto n_x32 = static_cast<int32_t>(n_x);
  const size_t mm = nao * nao;
  const auto nao32 = static_cast<int32_t>(nao);
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto copy = [&res](double *to, const double *from, const size_t count,
                           const wwr::wwrMemcpyKind kind) -> Status {
    return gpu_status(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), kind, res.stream()));
  };
  const auto download = [&](const double *const from,
                            const size_t count) -> Result<std::vector<double>> {
    std::vector<double> to(count);
    FBF_TRY(copy(to.data(), from, count, wwr::wwrMemcpyDeviceToHost));
    FBF_TRY(res.sync());
    return to;
  };
  // gamma and Gamma of @p vectors at the weights, row-major, on the host.
  const auto rdms =
      [&](const double *vectors) -> Result<std::pair<std::vector<double>, std::vector<double>>> {
    Buffers::device<double> gamma(n_act * n_act, res.device());
    Buffers::device<double> gamma2(n_act * n_act * n_act * n_act, res.device());
    FBF_TRY(rdm.rdm12(res, weights, vectors, gamma.data(), gamma2.data()));
    return std::pair{FBF_TRY(download(gamma.data(), gamma.num_elements())),
                     FBF_TRY(download(gamma2.data(), gamma2.num_elements()))};
  };
  const auto derivatives = [&](const double *vectors, std::span<const double> at,
                               const double *rotation) -> Result<mcscf::Derivatives> {
    return mcscf::df_derivatives(res, molecule, basis, classes, aux, h, b, l, c, ncore, ncas,
                                 nalpha, nbeta, vectors, at, rotation);
  };
  // df_casscf_exy reads C as a plain buffer.
  Buffers::device<double> c_columns(mm, res.device());
  FBF_TRY(copy(c_columns.data(), c.data(), mm, wwr::wwrMemcpyDeviceToDevice));
  const auto exy = [&](std::span<const double> gamma, std::span<const double> gamma2,
                       const Buffers::device<double> &fock_x,
                       const double *rotation) -> Result<CasscfExy> {
    return df_casscf_exy(res, molecule, basis, classes, aux, h, b, l, c_columns, ncore, ncas, gamma,
                         gamma2, fock_x, rotation);
  };

  // L_xlambda's first part, root I's own CP right-hand side (one-hot
  // weights), in ThirdDerivative's packed layout: the orbital pairs, the CI
  // rows as they are (the energy's convention), Theta_sr for r < s.
  Buffers::device<double> lxlambda(size * n_x, res.device());
  {
    std::vector<double> one_hot(nroots, 0.0);
    one_hot[static_cast<size_t>(root)] = 1.0;
    const mcscf::Derivatives own = FBF_TRY(derivatives(roots.data(), one_hot, nullptr));
    std::vector<double> theta(n_theta * n_x);
    for (size_t x = 0; x < n_x; ++x) {
      double *column = lxlambda.data() + (x * size);
      FBF_TRY(response::pack_orbital_pairs(res, nao32, ncore, ncas,
                                           own.rhs_orbital.data() + (x * mm), column));
      FBF_TRY(
          copy(column + pairs, own.rhs_ci.data() + (x * n_ci), n_ci, wwr::wwrMemcpyDeviceToDevice));
      size_t at = x * n_theta;
      for (size_t r = 0; r < nroots && n_theta != 0; ++r) {
        for (size_t s = r + 1; s < nroots; ++s) {
          theta[at++] = own.rhs_theta[s + (nroots * (r + (nroots * x)))];
        }
      }
      if (n_theta != 0) {
        FBF_TRY(copy(column + pairs + n_ci, theta.data() + (x * n_theta), n_theta,
                     wwr::wwrMemcpyHostToDevice));
      }
    }
    FBF_TRY(res.sync());
  }

  // L_xy's z_o part: E_xy along C -> C (1 + e K_z) at the averaged RDMs,
  // with F^[x] at the same rotated densities.
  CasscfExy lxy;
  {
    Buffers::device<double> k(mm, res.device());
    FBF_TRY(response::unpack_orbital_pairs(res, nao32, ncore, ncas, z.data(), k.data()));
    const auto [gamma, gamma2] = FBF_TRY(rdms(roots.data()));
    const mcscf::Derivatives rotated = FBF_TRY(derivatives(roots.data(), weights, k.data()));
    lxy = FBF_TRY(exy(gamma, gamma2, *rotated.fock_rotated, k.data()));
  }

  // Its z_c part: the roots' change dc_r = d_r + Sum_s Theta_sr c_s, and
  // E_xy at the symmetric transition RDMs Sum_r w_r (T(dc_r, c_r) +
  // T(c_r, dc_r)) as half the difference of E_xy at the RDMs (and F^[x]) of
  // c_r + dc_r and c_r - dc_r: E_xy is affine in (gamma, Gamma, F^[x]) at a
  // fixed core, so the core's part cancels.
  {
    Buffers::device<double> plus(n_ci, res.device());
    Buffers::device<double> minus(n_ci, res.device());
    Buffers::device<double> dc(n_ci, res.device());
    FBF_TRY(copy(dc.data(), z.data() + pairs, n_ci, wwr::wwrMemcpyDeviceToDevice));
    if (n_theta != 0) {
      // Theta as a full antisymmetric nroots x nroots, (s, r) at s + nroots r.
      const std::vector<double> packed = FBF_TRY(download(z.data() + pairs + n_ci, n_theta));
      std::vector<double> theta(nroots * nroots, 0.0);
      size_t at = 0;
      for (size_t r = 0; r < nroots; ++r) {
        for (size_t s = r + 1; s < nroots; ++s) {
          theta[s + (nroots * r)] = packed[at];
          theta[r + (nroots * s)] = -packed[at];
          ++at;
        }
      }
      Buffers::device<double> theta_d(theta.size(), res.device());
      FBF_TRY(copy(theta_d.data(), theta.data(), theta.size(), wwr::wwrMemcpyHostToDevice));
      const auto nroots32 = static_cast<int32_t>(nroots);
      FBF_TRY(gpu_status(gemm(blas, static_cast<int32_t>(dim), nroots32, nroots32, &kOne<double>,
                              roots.data(), theta_d.data(), &kOne<double>, dc.data())));
      FBF_TRY(res.sync());
    }
    const auto n_ci32 = static_cast<int32_t>(n_ci);
    FBF_TRY(copy(plus.data(), roots.data(), n_ci, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(copy(minus.data(), roots.data(), n_ci, wwr::wwrMemcpyDeviceToDevice));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, n_ci32, &kOne<double>, dc.data(), 1, plus.data(), 1)));
    FBF_TRY(gpu_status(
        wwr::axpy<double, int32_t>(blas, n_ci32, &kNegOne<double>, dc.data(), 1, minus.data(), 1)));
    const auto [g_plus, g2_plus] = FBF_TRY(rdms(plus.data()));
    const auto [g_minus, g2_minus] = FBF_TRY(rdms(minus.data()));
    CasscfExy at_plus;
    {
      const mcscf::Derivatives f = FBF_TRY(derivatives(plus.data(), weights, nullptr));
      at_plus = FBF_TRY(exy(g_plus, g2_plus, f.fock, nullptr));
    }
    CasscfExy at_minus;
    {
      const mcscf::Derivatives f = FBF_TRY(derivatives(minus.data(), weights, nullptr));
      at_minus = FBF_TRY(exy(g_minus, g2_minus, f.fock, nullptr));
    }
    const auto add_half_difference = [](std::vector<double> &to, const std::vector<double> &p,
                                        const std::vector<double> &m) {
      for (size_t i = 0; i < to.size(); ++i) {
        to[i] += 0.5 * (p[i] - m[i]);
      }
    };
    add_half_difference(lxy.one_electron, at_plus.one_electron, at_minus.one_electron);
    add_half_difference(lxy.two_electron.three_center, at_plus.two_electron.three_center,
                        at_minus.two_electron.three_center);
    add_half_difference(lxy.two_electron.metric, at_plus.two_electron.metric,
                        at_minus.two_electron.metric);
    add_half_difference(lxy.two_electron.cross_coulomb, at_plus.two_electron.cross_coulomb,
                        at_minus.two_electron.cross_coulomb);
    add_half_difference(lxy.two_electron.cross_exchange, at_plus.two_electron.cross_exchange,
                        at_minus.two_electron.cross_exchange);
    add_half_difference(lxy.two_electron.cross_active, at_plus.two_electron.cross_active,
                        at_minus.two_electron.cross_active);
    add_half_difference(lxy.two_electron.total, at_plus.two_electron.total,
                        at_minus.two_electron.total);
    add_half_difference(lxy.connection_second, at_plus.connection_second,
                        at_minus.connection_second);
    add_half_difference(lxy.connection_fock, at_plus.connection_fock, at_minus.connection_fock);
    add_half_difference(lxy.connection_twice, at_plus.connection_twice, at_minus.connection_twice);
    add_half_difference(lxy.exy, at_plus.exy, at_minus.exy);
  }

  // L_xlambda's second part, I^x z: the SA Hessian at the derivative
  // integrals and connection, applied to z, added to the right-hand side.
  {
    Buffers::device<double> h1(n_x * mm, res.device());
    Buffers::device<double> w1(n_x * mm * naux, res.device());
    FBF_TRY(mcscf::df_integral_derivatives(res, molecule, basis, classes, aux, h, b, l, c,
                                           h1.data(), w1.data()));
    mcscf::ThirdDerivative op =
        FBF_TRY(mcscf::ThirdDerivative::create(res, h, b, static_cast<int32_t>(naux), c, ncore,
                                               ncas, alpha, beta, roots.data(), weights, z.data()));
    Buffers::device<double> y(size * n_x, res.device());
    FBF_TRY(op.apply_integrals(res, n_x32, h1.data(), w1.data(), y.data()));
    FBF_TRY(gpu_status(wwr::axpy<double, int32_t>(blas, static_cast<int32_t>(size) * n_x32,
                                                  &kOne<double>, y.data(), 1, lxlambda.data(), 1)));
    FBF_TRY(res.sync());
  }
  return SaCasscfZTerms{.lxy = std::move(lxy), .lxlambda = std::move(lxlambda)};
}

} // namespace free_breakfast::hessian
