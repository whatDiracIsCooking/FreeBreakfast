/// @file
/// @brief mcscf::root_rotation: per pair of roots the symmetrized transition
/// RDMs (two ci::Rdm calls, polarized), their orbital gradient by
/// response_gradient packed and contracted with kappa^x by one gemv, then
/// eqs. theta and rot on the host.

module free_breakfast.mcscf.root_rotation;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gemv, geam
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.molecule;
import free_breakfast.integrals.active_space;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.ci.strings;
import free_breakfast.ci.rdm;
import free_breakfast.response.fock;
import free_breakfast.response.orbital_pairs;
import free_breakfast.mcscf.derivative;
import free_breakfast.mcscf.relaxed_hessian;

#include "diagnostics.h"

namespace free_breakfast::mcscf {

// Two roots closer than this (hartree) are refused whatever their coupling:
// RelaxedHessian's floor for dropping their mixing, where the quotient of
// eq. theta is no longer a number.
constexpr double kRootGapFloor = 1e-8;
// Weights this close (relative to the largest) are equal: they are recovered
// from a solution by a normalization, not given exactly.
constexpr double kEqualWeightTolerance = 1e-10;

Result<RootRotation> root_rotation(DeviceResources &res, const Molecule &molecule,
                                   const BasisSet &basis, const SphericalMatrixAO &h,
                                   const Buffers::device<double> &b, const SphericalMatrixAO &c,
                                   const int32_t ncore, const int32_t ncas, const int32_t nalpha,
                                   const int32_t nbeta, const double *const roots,
                                   const std::span<const double> weights,
                                   const std::span<const double> energies,
                                   const Derivatives &derivatives, const double *const kappa,
                                   const double max_rotation) {
  const size_t nao = basis.nao_sphe();
  const size_t n_x = 3 * static_cast<size_t>(basis.natoms());
  const size_t nroots = weights.size();
  if (c.n() != nao || h.n() != nao) {
    return err_config("mcscf::root_rotation: C is {0} x {0} and h {1} x {1}, but the basis has "
                      "{2} functions",
                      c.n(), h.n(), nao);
  }
  if (ncore < 0 || ncas < 1 || static_cast<size_t>(ncore) + static_cast<size_t>(ncas) > nao) {
    return err_config("mcscf::root_rotation: ncore {} and ncas {} do not fit {} orbitals", ncore,
                      ncas, nao);
  }
  if (nao == 0 || b.num_elements() == 0 || b.num_elements() % (nao * nao) != 0) {
    return err_config("mcscf::root_rotation: B holds {} doubles, not a whole number of {}^2 slabs",
                      b.num_elements(), nao);
  }
  if (nroots == 0 || energies.size() != nroots) {
    return err_config("mcscf::root_rotation: {} weights and {} energies, not one each per root",
                      nroots, energies.size());
  }
  const double w_max = std::ranges::max(weights);
  for (const double w : weights) {
    if (std::abs(w - w_max) > kEqualWeightTolerance * w_max) {
      return err_config("mcscf::root_rotation: weight {} beside {}: the term is the equal-weight "
                        "one (at unequal weights the rotations are parameters, in lambda^x)",
                        w, w_max);
    }
  }
  if (derivatives.hamiltonian.size() != nroots * nroots * n_x) {
    return err_config("mcscf::root_rotation: the derivative Hamiltonian holds {} doubles, but "
                      "{} roots over {} coordinates need {}",
                      derivatives.hamiltonian.size(), nroots, n_x, nroots * nroots * n_x);
  }
  if (!(max_rotation > 0.0)) {
    return err_config("mcscf::root_rotation: max_rotation {} is not positive", max_rotation);
  }
  RootRotation out{std::vector<double>(nroots * nroots * n_x, 0.0),
                   std::vector<double>(nroots * nroots * n_x, 0.0),
                   std::vector<double>(n_x * n_x * nroots, 0.0)};
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < r; ++s) {
      const double gap = energies[r] - energies[s];
      if (std::abs(gap) < kRootGapFloor) {
        return err_numerical("mcscf::root_rotation: roots {} and {} are {:.3e} hartree apart, "
                             "under {:.0e}: at a degeneracy the root has no Hessian",
                             s, r, std::abs(gap), kRootGapFloor);
      }
    }
  }
  if (nroots == 1) {
    return out;
  }

  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const wwr::wwrStream_t stream = res.stream();
  const auto n_core = static_cast<size_t>(ncore);
  const auto n_act = static_cast<size_t>(ncas);
  const size_t na2 = n_act * n_act;
  const size_t na4 = na2 * na2;
  const size_t nn = nao * nao;
  const auto naux = static_cast<int32_t>(b.num_elements() / nn);
  const auto n_pairs =
      static_cast<int32_t>((n_core * (nao - n_core)) + (n_act * (nao - n_core - n_act)));

  // The integrals response_gradient contracts: F^I (any RDMs give it; the
  // average's are at hand) and ppaa, papa, on the host.
  const ci::DeviceStringSet alpha = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nalpha));
  const ci::DeviceStringSet beta = FBF_TRY(ci::DeviceStringSet::create(res, ncas, nbeta));
  ci::Rdm rdm = FBF_TRY(ci::Rdm::create(res, alpha, beta));
  const auto dim = static_cast<size_t>(rdm.size());
  const auto dim32 = static_cast<int32_t>(dim);
  // Held across rdm12, casscf_integrals and fock, which take the workspace
  // (docs/style.md "Scratch memory"): c_r + c_s and c_r - c_s, the RDMs of
  // each, the gradient's antisymmetric matrix, its packed pairs and the
  // contraction with kappa.
  Buffers::device<double> pm(2 * dim, res.device());
  Buffers::device<double> rdms(2 * (na2 + na4), res.device());
  Buffers::device<double> a_full(nn, res.device());
  Buffers::device<double> a_packed(std::max<size_t>(static_cast<size_t>(n_pairs), 1), res.device());
  Buffers::device<double> orbital(n_x, res.device());

  std::vector<double> gamma(na2);
  std::vector<double> gamma2(na4);
  const auto download = [&](double *to, const double *from, const size_t count) {
    return gpu_status(
        wwr::wwrMemcpyAsync(to, from, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost, stream));
  };
  FBF_TRY(rdm.rdm12(res, weights, roots, rdms.data(), rdms.data() + na2));
  FBF_TRY(download(gamma.data(), rdms.data(), na2));
  FBF_TRY(download(gamma2.data(), rdms.data() + na2, na4));
  FBF_TRY(res.sync());
  const CasscfIntegrals ints =
      FBF_TRY(casscf_integrals(res, basis, molecule, h, b, naux, c, ncore, ncas));
  const response::Fock f0 = FBF_TRY(response::fock(res, molecule, h, b, naux, c, ncore, ncas,
                                                   ints.ppaa, ints.papa, gamma, gamma2));
  ResponseIntegrals response_ints{nao,
                                  n_core,
                                  n_act,
                                  std::vector<double>(nn),
                                  std::vector<double>(nn * na2),
                                  std::vector<double>(nn * na2)};
  FBF_TRY(download(response_ints.fock_inactive.data(), f0.fock_inactive.data(), nn));
  FBF_TRY(download(response_ints.ppaa.data(), ints.ppaa.data(), nn * na2));
  FBF_TRY(download(response_ints.papa.data(), ints.papa.data(), nn * na2));
  FBF_TRY(res.sync());

  // H^x_rs per pair r > s: H~^x_rs plus <c_r|H'(kappa^x)|c_s>. The latter is
  // h_eff'.gamma_S + 1/2 (tu|vw)'.Gamma_S at the symmetrized transition RDMs,
  // a quarter of the difference of the RDMs of c_r + c_s and c_r - c_s
  // (Gamma_S's half again for the 1/2): H' is real symmetric, so the
  // symmetrized RDMs give <c_r|H'|c_s> exactly. Its gradient in x, packed,
  // against every column of kappa is one gemv.
  const std::array<double, 1> one{1.0};
  std::vector<double> gp(2 * (na2 + na4));
  std::vector<double> a(nn);
  std::vector<double> coupling_orb(n_x);
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < r; ++s) {
      const double *const c_r = roots + (r * dim);
      const double *const c_s = roots + (s * dim);
      for (const auto &[to, sign] :
           {std::pair{pm.data(), &kOne<double>}, std::pair{pm.data() + dim, &kNegOne<double>}}) {
        FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N,
                                                      dim32, 1, &kOne<double>, c_r, dim32, sign,
                                                      c_s, dim32, to, dim32)));
      }
      double *const plus = rdms.data();
      double *const minus = plus + na2 + na4;
      FBF_TRY(rdm.rdm12(res, one, pm.data(), plus, plus + na2));
      FBF_TRY(rdm.rdm12(res, one, pm.data() + dim, minus, minus + na2));
      FBF_TRY(download(gp.data(), rdms.data(), gp.size()));
      FBF_TRY(res.sync());
      const size_t off = na2 + na4;
      for (size_t i = 0; i < na2; ++i) {
        gamma[i] = 0.25 * (gp[i] - gp[off + i]);
      }
      for (size_t i = 0; i < na4; ++i) {
        gamma2[i] = 0.125 * (gp[na2 + i] - gp[off + na2 + i]);
      }
      response_gradient(response_ints, gamma, gamma2, a);
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a_full.data(), a.data(), sizeof(double) * nn,
                                             wwr::wwrMemcpyHostToDevice, stream)));
      FBF_TRY(response::pack_orbital_pairs(res, static_cast<int32_t>(nao), ncore, ncas,
                                           a_full.data(), a_packed.data()));
      FBF_TRY(gpu_status(wwr::gemv<double, int32_t>(
          blas, wwr::WWRBLAS_OP_T, n_pairs, static_cast<int32_t>(n_x), &kOne<double>, kappa,
          n_pairs, a_packed.data(), 1, &kZero<double>, orbital.data(), 1)));
      FBF_TRY(download(coupling_orb.data(), orbital.data(), n_x));
      FBF_TRY(res.sync());
      for (size_t x = 0; x < n_x; ++x) {
        const size_t slab = nroots * nroots * x;
        const double value = derivatives.hamiltonian[r + (nroots * s) + slab] + coupling_orb[x];
        out.coupling[r + (nroots * s) + slab] = value;
        out.coupling[s + (nroots * r) + slab] = value;
      }
    }
  }

  // Eq. theta and the refusal, then eq. rot per root.
  for (size_t r = 0; r < nroots; ++r) {
    for (size_t s = 0; s < nroots; ++s) {
      if (s == r) {
        continue;
      }
      const double gap = energies[r] - energies[s];
      double worst = 0.0;
      for (size_t x = 0; x < n_x; ++x) {
        const size_t slab = nroots * nroots * x;
        const double theta = out.coupling[r + (nroots * s) + slab] / gap;
        out.theta[s + (nroots * r) + slab] = theta;
        worst = std::max(worst, std::abs(theta));
      }
      if (worst > max_rotation) {
        return err_numerical("mcscf::root_rotation: roots {} and {} are {:.3e} hartree apart, "
                             "and their largest coupling over that gap is {:.3e} per bohr, past "
                             "{:.3e}: too near a degeneracy for root {}'s Hessian",
                             std::min(r, s), std::max(r, s), std::abs(gap), worst, max_rotation, r);
      }
    }
  }
  for (size_t i = 0; i < nroots; ++i) {
    double *const term = out.hessian.data() + (n_x * n_x * i);
    for (size_t s = 0; s < nroots; ++s) {
      if (s == i) {
        continue;
      }
      const double scale = 2.0 / (energies[i] - energies[s]);
      for (size_t y = 0; y < n_x; ++y) {
        const double hy = out.coupling[i + (nroots * s) + (nroots * nroots * y)];
        for (size_t x = 0; x < n_x; ++x) {
          term[x + (n_x * y)] +=
              scale * out.coupling[i + (nroots * s) + (nroots * nroots * x)] * hy;
        }
      }
    }
  }
  return out;
}

} // namespace free_breakfast::mcscf
