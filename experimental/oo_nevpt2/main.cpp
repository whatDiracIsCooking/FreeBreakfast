// SPDX-License-Identifier: Apache-2.0
/// @file main.cpp
/// @brief The OO-NEVPT2 probe (design.html): from a config's converged
/// single-state DF-CASSCF, iterate the quasi-Newton step of
///
///     E(kappa) = E_CAS(kappa, c*(kappa)) + E_PT2(kappa, c*(kappa)),
///
/// c* the CASCI root at the rotated orbitals and E_PT2 PC-NEVPT2's over
/// canonical core and virtuals. The step is the NEVPT2 gradient's Z-vector:
/// the right-hand sides g + pack(A) and y / 2 over the CI-relaxed CASSCF
/// Hessian, A and y from nevpt2::pc_adjoint plus the z^c term's. A short
/// line search along it (t = 1, 1/2, then halving) takes the lowest energy.
///
///     free_breakfast_oo_nevpt2 <config.toml> [iterations = 8] [oo|no|mo|cap|sa]
///                              [cap factor = 1]
///
/// With `no`, the natural orbitals of the PT2-corrected density instead,
/// iterated: no functional is minimized (natural() below). With `mo`, the
/// maximum-overlap orbitals: the weight of the first-order wave function,
/// W = <Psi1|Psi1>, minimized in the orbital rotations by BFGS over a
/// finite-difference gradient (maximum_overlap() below), and E read there.
/// With `cap`, E minimized under W <= factor W(CASSCF orbitals): the bare
/// functional with the first-order weight capped (capped() below).
///
/// With `sa`, `no`'s passes over a state average: the natural orbitals of the
/// state-averaged density plus PC-QD-NEVPT2's correction to it, and the QD
/// states in them (averaged() below).
///
/// The config needs [casscf] of one singlet root (two or more singlet roots
/// with `sa`) and density-fitted [eri].
/// Prints one row per iteration; nothing is written but the pipeline's own
/// files, under the system's temporary directory.

#include "diagnostics.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.integrals;
import free_breakfast.recipes;
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

constexpr double kZvectorTolerance = 1e-8;
constexpr double kEnergyTolerance = 1e-9;
// The fallback step's largest rotation, and the longest multiple of a step
// the line search doubles to.
constexpr double kFallbackStep = 0.05;
constexpr double kTrustStep = 0.3;
constexpr double kLongestStep = 2.0;
// The denominator shift W is differenced over (weight() below), the rotation
// its gradient is differenced over, and the gradient that ends the
// minimization.
constexpr double kShift = 1e-3;
constexpr double kRotation = 2e-3;
constexpr double kWeightGradient = 2e-5;
// A rotation whose derivative of W is below this at the CASSCF orbitals is
// taken as zero by symmetry and left out, until the last gradient says not.
constexpr double kSymmetryZero = 2e-6;
// The capped minimization's ends: the Lagrangian's gradient (hartree per
// radian) and how far W may sit over the cap.
constexpr double kLagrangianGradient = 2e-5;
constexpr double kCapTolerance = 1e-7;
// The indices outside the active space each class's perturbers carry, in
// PcClassEnergies' order: Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir.
constexpr std::array<double, 8> kVirtualIndices{1, 0, 2, 1, 2, 2, 0, 1};
constexpr std::array<double, 8> kCoreIndices{0, 1, 2, 2, 1, 0, 2, 1};

fb::Result<std::vector<double>> download(fb::DeviceResources &res, const double *data,
                                         const std::size_t count) {
  std::vector<double> host(count);
  FBF_TRY(fb::gpu_status(wwr::wwrMemcpyAsync(host.data(), data, count * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  return host;
}

fb::Status copy(fb::DeviceResources &res, double *to, const double *from, const std::size_t count,
                const wwr::wwrMemcpyKind kind) {
  FBF_TRY(
      fb::gpu_status(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), kind, res.stream())));
  return res.sync();
}

double norm(const std::span<const double> v) {
  return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

// The eigenvalues of the symmetric @p n x @p n matrix @p a by cyclic Jacobi,
// unsorted, and into @p v (if not null) the eigenvectors, eigenvalue j's at
// v[k n + j].
std::vector<double> eigensystem(std::vector<double> a, const std::size_t n,
                                std::vector<double> *const v = nullptr) {
  if (v != nullptr) {
    v->assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      (*v)[(i * n) + i] = 1.0;
    }
  }
  for (std::int32_t sweep = 0; sweep < 50; ++sweep) {
    double off = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        off += a[(i * n) + j] * a[(i * n) + j];
      }
    }
    if (off < 1e-28) {
      break;
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const double theta =
            0.5 * std::atan2(2.0 * a[(i * n) + j], a[(j * n) + j] - a[(i * n) + i]);
        const double cs = std::cos(theta);
        const double sn = std::sin(theta);
        for (std::size_t k = 0; k < n; ++k) {
          const double ki = a[(k * n) + i];
          const double kj = a[(k * n) + j];
          a[(k * n) + i] = (cs * ki) - (sn * kj);
          a[(k * n) + j] = (sn * ki) + (cs * kj);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double ik = a[(i * n) + k];
          const double jk = a[(j * n) + k];
          a[(i * n) + k] = (cs * ik) - (sn * jk);
          a[(j * n) + k] = (sn * ik) + (cs * jk);
        }
        for (std::size_t k = 0; v != nullptr && k < n; ++k) {
          const double ki = (*v)[(k * n) + i];
          const double kj = (*v)[(k * n) + j];
          (*v)[(k * n) + i] = (cs * ki) - (sn * kj);
          (*v)[(k * n) + j] = (sn * ki) + (cs * kj);
        }
      }
    }
  }
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = a[(i * n) + i];
  }
  return out;
}

// eigensystem's eigenvalues, descending: the active natural occupations of
// a 1-RDM.
std::vector<double> eigenvalues(std::vector<double> a, const std::size_t n) {
  std::vector<double> out = eigensystem(std::move(a), n);
  std::ranges::sort(out, std::greater<>{});
  return out;
}

// What every evaluation shares: the pipeline's products and the active
// space's CI objects.
struct Problem {
  fb::DeviceResources &res;
  const fb::Molecule &molecule;
  const fb::BasisSet &basis;
  const fb::SphericalMatrixAO &h;
  const fb::SphericalMatrixAO &s;
  const fb::Buffers::device<double> &b;
  std::int32_t naux;
  std::int32_t nmo;
  std::int32_t ncore;
  std::int32_t ncas;
  fb::ci::DeviceStringSet &alpha;
  fb::ci::DeviceStringSet &beta;
  fb::ci::Sigma &sigma;
  fb::ci::Rdm &rdm;
};

// The functional at one set of orbitals: the CASCI root there, the core and
// virtuals made canonical over its Fock, and both energies.
struct Point {
  fb::SphericalMatrixAO c;
  fb::ci::Roots roots;
  std::vector<double> gamma;
  std::vector<double> gamma2;
  fb::CasscfIntegrals ints;
  fb::response::Fock fock;
  double e_cas = 0.0;
  double e_pt2 = 0.0;

  [[nodiscard]] double total() const { return e_cas + e_pt2; }
};

fb::Status set_sigma(const Problem &p, const fb::CasscfIntegrals &ints) {
  const auto n = static_cast<std::size_t>(p.ncas);
  const std::vector<double> h_eff = FBF_TRY(download(p.res, ints.hamiltonian.h_eff.data(), n * n));
  const std::vector<double> eri =
      FBF_TRY(download(p.res, ints.hamiltonian.eri.data(), n * n * n * n));
  return p.sigma.set_integrals(p.res, ints.hamiltonian.e_core, h_eff, eri);
}

// E at the orbitals @p c_in (device, nmo^2), the CI started from @p start.
// Without @p pt2 the PC-NEVPT2 energy is left out (e_pt2 a NaN): what a
// point that is only weighed needs.
fb::Result<Point> evaluate(const Problem &p, const double *c_in,
                           const std::span<const double> start, const bool pt2 = true) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const auto n = static_cast<std::size_t>(p.ncas);
  const std::size_t mo2 = nmo * nmo;
  const auto dim = static_cast<std::size_t>(p.rdm.size());
  fb::SphericalMatrixAO c(res, p.basis);
  FBF_TRY(copy(res, c.data(), c_in, mo2, wwr::wwrMemcpyDeviceToDevice));

  // The CASCI root. It does not see the canonicalization below: the active
  // orbitals and the core's span stay.
  const fb::CasscfIntegrals first =
      FBF_TRY(fb::casscf_integrals(res, p.basis, p.molecule, p.h, p.b, p.naux, c, p.ncore, p.ncas));
  FBF_TRY(set_sigma(p, first));
  fb::Buffers::device<double> diagonal(dim, res.device());
  FBF_TRY(p.sigma.diagonal(res, diagonal.data()));
  fb::ci::Roots roots =
      FBF_TRY(fb::ci::solve(res, p.sigma, p.alpha, p.beta, diagonal.data(), 1, {}, start));
  fb::Buffers::device<double> gamma_device(n * n, res.device());
  fb::Buffers::device<double> gamma2_device(n * n * n * n, res.device());
  const std::array<double, 1> one{1.0};
  FBF_TRY(p.rdm.rdm12(res, one, roots.vectors.data(), gamma_device.data(), gamma2_device.data()));
  std::vector<double> gamma = FBF_TRY(download(res, gamma_device.data(), n * n));
  std::vector<double> gamma2 = FBF_TRY(download(res, gamma2_device.data(), n * n * n * n));

  // Canonical core and virtuals over this root's Fock: C <- C R.
  {
    const fb::response::Fock fock =
        FBF_TRY(fb::response::fock(res, p.molecule, p.h, p.b, p.naux, c, p.ncore, p.ncas,
                                   first.ppaa, first.papa, gamma, gamma2));
    std::vector<double> f = FBF_TRY(download(res, fock.fock_inactive.data(), mo2));
    const std::vector<double> fa = FBF_TRY(download(res, fock.fock_active.data(), mo2));
    std::ranges::transform(f, fa, f.begin(), std::plus<>{});
    std::vector<double> r;
    std::vector<double> energies;
    FBF_TRY(fb::canonicalize_core_virtual(res, nmo, static_cast<std::size_t>(p.ncore),
                                          static_cast<std::size_t>(p.ncore + p.ncas), f, r,
                                          energies));
    fb::Buffers::device<double> r_device(mo2, res.device());
    fb::Buffers::device<double> rotated(mo2, res.device());
    FBF_TRY(copy(res, r_device.data(), r.data(), mo2, wwr::wwrMemcpyHostToDevice));
    FBF_TRY(fb::gpu_status(wwr::gemm<double, std::int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, p.nmo, p.nmo, p.nmo,
        &fb::kOne<double>, c.data(), p.nmo, r_device.data(), p.nmo, &fb::kZero<double>,
        rotated.data(), p.nmo)));
    FBF_TRY(copy(res, c.data(), rotated.data(), mo2, wwr::wwrMemcpyDeviceToDevice));
  }
  fb::CasscfIntegrals ints =
      FBF_TRY(fb::casscf_integrals(res, p.basis, p.molecule, p.h, p.b, p.naux, c, p.ncore, p.ncas));
  fb::response::Fock fock = FBF_TRY(fb::response::fock(
      res, p.molecule, p.h, p.b, p.naux, c, p.ncore, p.ncas, ints.ppaa, ints.papa, gamma, gamma2));
  double e_pt2 = std::numeric_limits<double>::quiet_NaN();
  if (pt2) {
    const fb::nevpt2::PcClassEnergies classes = FBF_TRY(
        fb::nevpt2::pc_energy(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c.data(),
                              p.h.data(), p.ncore, p.ncas, roots.vectors.data(), 0));
    e_pt2 = classes.total;
  }
  const double e_cas = fock.energy;
  return Point{.c = std::move(c),
               .roots = std::move(roots),
               .gamma = std::move(gamma),
               .gamma2 = std::move(gamma2),
               .ints = std::move(ints),
               .fock = std::move(fock),
               .e_cas = e_cas,
               .e_pt2 = e_pt2};
}

// The weight of the first-order wave function at a point, W = <Psi1|Psi1>,
// per class and summed. No call reports it, so it is read off the energy:
// with every perturber's denominator raised by w n, n the perturber's count
// of indices outside the active space, a class's energy -sum y^2 / (D + w n)
// has the derivative n sum y^2 / D^2 = n W_class at w = 0. The shift is
// h + w S C d C^T S, d = -1 on the core, 0 on the active and +1 on the
// virtual orbitals: it moves the canonical orbital energies and nothing
// else (no block V or the active Hamiltonian reads), and W is the central
// difference over w = +-kShift. @p virt and @p core scale the two halves of
// d, so the classes with both kinds of index can be weighed either way: the
// check run() prints.
struct Weight {
  std::array<double, 8> classes{};
  double total = 0.0;
};

fb::Result<Weight> weight(const Problem &p, const Point &at, const double virt = 1.0,
                          const double core = 1.0) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const auto ncore = static_cast<std::size_t>(p.ncore);
  const auto nocc = static_cast<std::size_t>(p.ncore + p.ncas);
  const std::size_t mo2 = nmo * nmo;
  const std::vector<double> s = FBF_TRY(download(res, p.s.data(), mo2));
  const std::vector<double> c = FBF_TRY(download(res, at.c.data(), mo2));
  const std::vector<double> h = FBF_TRY(download(res, p.h.data(), mo2));
  // M = S C, column-major, and the shift's matrix sum_q d_q M[:, q] M[:, q]^T.
  std::vector<double> m(mo2, 0.0);
  for (std::size_t q = 0; q < nmo; ++q) {
    for (std::size_t nu = 0; nu < nmo; ++nu) {
      const double c_nu = c[nu + (nmo * q)];
      for (std::size_t mu = 0; mu < nmo; ++mu) {
        m[mu + (nmo * q)] += s[mu + (nmo * nu)] * c_nu;
      }
    }
  }
  std::vector<double> shift(mo2, 0.0);
  for (std::size_t q = 0; q < nmo; ++q) {
    const double d = q < ncore ? -core : (q < nocc ? 0.0 : virt);
    for (std::size_t nu = 0; nu < nmo; ++nu) {
      for (std::size_t mu = 0; mu < nmo; ++mu) {
        shift[mu + (nmo * nu)] += d * m[mu + (nmo * q)] * m[nu + (nmo * q)];
      }
    }
  }
  fb::Buffers::device<double> shifted(mo2, res.device());
  std::array<fb::nevpt2::PcClassEnergies, 2> sides;
  for (std::size_t side = 0; side < 2; ++side) {
    const double w = side == 0 ? kShift : -kShift;
    std::vector<double> host = h;
    for (std::size_t k = 0; k < mo2; ++k) {
      host[k] += w * shift[k];
    }
    FBF_TRY(copy(res, shifted.data(), host.data(), mo2, wwr::wwrMemcpyHostToDevice));
    sides[side] = FBF_TRY(fb::nevpt2::pc_energy(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo,
                                                p.b.data(), at.c.data(), shifted.data(), p.ncore,
                                                p.ncas, at.roots.vectors.data(), 0));
  }
  Weight out;
  for (std::size_t k = 0; k < 8; ++k) {
    const double n = (virt * kVirtualIndices[k]) + (core * kCoreIndices[k]);
    if (n > 0.0) {
      out.classes[k] = (sides[0].energies[k] - sides[1].energies[k]) / (2.0 * kShift * n);
    }
    out.total += out.classes[k];
  }
  return out;
}

// The quasi-Newton step at a point: kappa(Z)'s eigenpairs, for skew_expm.
struct Step {
  fb::Buffers::device<wwr::wwrDoubleComplex> u;
  fb::Buffers::device<double> w;
  // ||g||, ||pack(A)|| and ||g + pack(A)|| over the non-redundant pairs.
  double g_norm = 0.0;
  double a_norm = 0.0;
  double source_norm = 0.0;
  // (g + pack(A)) . Z, and the largest |Z|.
  double slope = 0.0;
  double max_kappa = 0.0;
  std::int32_t products = 0;
  // false: the solve refused, and this is the steepest-descent fallback.
  bool newton = true;
};

fb::Result<Step> quasi_newton(const Problem &p, const Point &at) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const auto n = static_cast<std::size_t>(p.ncas);
  const std::size_t mo2 = nmo * nmo;
  const auto dim = static_cast<std::size_t>(p.rdm.size());
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const double *const c = at.c.data();
  const double *const vector = at.roots.vectors.data();
  FBF_TRY(set_sigma(p, at.ints));

  // E_PT2's adjoint, with the canonical condition's term: A and y.
  fb::nevpt2::PcAdjoint corr =
      FBF_TRY(fb::nevpt2::pc_adjoint(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c,
                                     p.h.data(), p.ncore, p.ncas, vector, 0));
  const fb::nevpt2::ZcMultipliers zc = FBF_TRY(
      fb::nevpt2::zc_multipliers(res, p.nmo, p.ncore, p.ncas, corr.a.data(), corr.epsilon.data()));
  const fb::nevpt2::ZcAdjoint phi = FBF_TRY(
      fb::nevpt2::zc_adjoint(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c, p.h.data(),
                             p.ncore, p.ncas, std::array{1.0}, vector, zc.z.data()));
  FBF_TRY(fb::gpu_status(wwr::axpy<double, std::int32_t>(
      blas, static_cast<std::int32_t>(mo2), &fb::kOne<double>, phi.a.data(), 1, corr.a.data(), 1)));
  FBF_TRY(fb::gpu_status(wwr::axpy<double, std::int32_t>(
      blas, static_cast<std::int32_t>(dim), &fb::kOne<double>, phi.y.data(), 1, corr.y.data(), 1)));

  // The right-hand sides: g + pack(A), and y / 2 in zvector's CI row.
  const std::size_t packed = at.fock.gradient.num_elements();
  const auto packed32 = static_cast<std::int32_t>(packed);
  fb::Buffers::device<double> rhs(packed, res.device());
  FBF_TRY(fb::response::pack_orbital_pairs(res, p.nmo, p.ncore, p.ncas, corr.a.data(), rhs.data()));
  const std::vector<double> a_host = FBF_TRY(download(res, rhs.data(), packed));
  const std::vector<double> g_host = FBF_TRY(download(res, at.fock.gradient.data(), packed));
  FBF_TRY(fb::gpu_status(wwr::axpy<double, std::int32_t>(
      blas, packed32, &fb::kOne<double>, at.fock.gradient.data(), 1, rhs.data(), 1)));
  const std::vector<double> source = FBF_TRY(download(res, rhs.data(), packed));
  FBF_TRY(fb::gpu_status(wwr::scal<double, std::int32_t>(blas, static_cast<std::int32_t>(dim),
                                                         &fb::kHalf<double>, corr.y.data(), 1)));

  // The CI-relaxed CASSCF Hessian at the point, and the solve.
  fb::response::Hessian hessian = FBF_TRY(fb::response::Hessian::create(
      res, p.h, p.b, p.naux, at.c, p.ncore, p.ncas, at.ints.ppaa, at.ints.papa, at.fock));
  fb::response::ResponseIntegrals response_ints{
      nmo,
      static_cast<std::size_t>(p.ncore),
      n,
      FBF_TRY(download(res, at.fock.fock_inactive.data(), mo2)),
      FBF_TRY(download(res, at.ints.ppaa.data(), mo2 * n * n)),
      FBF_TRY(download(res, at.ints.papa.data(), mo2 * n * n))};
  const std::array<double, 1> weights{1.0};
  fb::response::RelaxedHessian relaxed = FBF_TRY(fb::response::RelaxedHessian::create(
      res, hessian, p.sigma, p.rdm, p.alpha, p.beta, at.roots, weights, std::move(response_ints),
      fb::response::response_tolerance_for(kZvectorTolerance)));
  // The solve; where the CASSCF Hessian is no longer positive definite it
  // refuses, and the step is steepest descent on g + pack(A) instead (the
  // CI's response left out), its largest element kFallbackStep.
  const fb::Result<fb::response::ZVector> z = fb::response::zvector(
      res, relaxed, hessian, at.roots, rhs.data(), corr.y.data(), kZvectorTolerance);
  std::vector<double> z_host;
  if (z.has_value()) {
    z_host = FBF_TRY(download(res, z->orbital.data(), packed));
    // A near-singular Hessian (an active orbital at occupation 2) gives a
    // step of many radians: cut it to kTrustStep.
    const double largest = std::ranges::max(
        z_host | std::views::transform([](const double v) { return std::abs(v); }));
    if (largest > kTrustStep) {
      for (double &v : z_host) {
        v *= kTrustStep / largest;
      }
    }
  } else {
    std::println("     Z-vector refused ({}): steepest descent", z.error().message);
    const double largest = std::ranges::max(
        source | std::views::transform([](const double v) { return std::abs(v); }));
    z_host = source;
    for (double &v : z_host) {
      v *= -kFallbackStep / largest;
    }
  }
  fb::Buffers::device<double> z_device(packed, res.device());
  FBF_TRY(copy(res, z_device.data(), z_host.data(), packed, wwr::wwrMemcpyHostToDevice));

  Step out{.u = fb::Buffers::device<wwr::wwrDoubleComplex>(mo2, res.device()),
           .w = fb::Buffers::device<double>(nmo, res.device()),
           .g_norm = norm(g_host),
           .a_norm = norm(a_host),
           .source_norm = norm(source),
           .slope = std::inner_product(source.begin(), source.end(), z_host.begin(), 0.0),
           .max_kappa = std::ranges::max(
               z_host | std::views::transform([](const double v) { return std::abs(v); })),
           .products = z.has_value() ? z->iterations : 0,
           .newton = z.has_value()};
  fb::Buffers::device<double> kappa(mo2, res.device());
  FBF_TRY(fb::response::unpack_orbital_pairs(res, p.nmo, p.ncore, p.ncas, z_device.data(),
                                             kappa.data()));
  FBF_TRY(fb::skew_eig<double>(res, p.nmo, kappa.data(), out.u.data(), out.w.data()));
  return out;
}

// E at C exp(t kappa), the CI started from @p from's.
fb::Result<Point> moved(const Problem &p, const Point &from, const Step &step, const double t) {
  fb::DeviceResources &res = p.res;
  const auto mo2 = static_cast<std::size_t>(p.nmo) * static_cast<std::size_t>(p.nmo);
  fb::Buffers::device<double> rotation(mo2, res.device());
  fb::Buffers::device<double> c(mo2, res.device());
  FBF_TRY(fb::skew_expm<double>(res, p.nmo, step.u.data(), step.w.data(), t, rotation.data()));
  FBF_TRY(fb::gpu_status(wwr::gemm<double, std::int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, p.nmo, p.nmo, p.nmo,
      &fb::kOne<double>, from.c.data(), p.nmo, rotation.data(), p.nmo, &fb::kZero<double>, c.data(),
      p.nmo)));
  FBF_TRY(res.sync());
  return evaluate(p, c.data(), {from.roots.vectors.data(), static_cast<std::size_t>(p.rdm.size())});
}

// The orbitals C U, U's columns the natural orbitals of D = D_ref + @p d,
// D_ref the reference's density over the orbitals @p c (2 on the core, the
// active 1-RDM @p gamma) and @p d a correlation correction to it, nmo^2 in
// that basis. The new active orbitals are the ncas natural orbitals with the
// most weight on the old active space, the new core the ncore of the rest
// with the most on the old core: by overlap, not by occupation, so a
// correlated core orbital whose occupation falls among the active ones does
// not swap in.
fb::Result<fb::Buffers::device<double>> natural_orbitals(const Problem &p, const double *const c,
                                                         std::vector<double> d,
                                                         const std::span<const double> gamma) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const auto ncore = static_cast<std::size_t>(p.ncore);
  const auto n = static_cast<std::size_t>(p.ncas);
  const std::size_t mo2 = nmo * nmo;
  double trace = 0.0;
  double mixing = 0.0;
  for (std::size_t q = 0; q < nmo; ++q) {
    trace += d[(q * nmo) + q];
    for (std::size_t r = 0; r < nmo; ++r) {
      const auto space = [&](const std::size_t k) {
        return k < ncore ? 0 : (k < ncore + n ? 1 : 2);
      };
      if (space(q) != space(r)) {
        mixing = std::max(mixing, std::abs(d[(q * nmo) + r]));
      }
    }
  }
  for (std::size_t i = 0; i < ncore; ++i) {
    d[(i * nmo) + i] += 2.0;
  }
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      d[((ncore + t) * nmo) + ncore + u] += gamma[(t * n) + u];
    }
  }
  std::vector<double> v;
  const std::vector<double> occupations = eigensystem(std::move(d), nmo, &v);

  // The spaces, by weight on the old ones; each by descending occupation.
  const auto weight = [&](const std::size_t j, const std::size_t first, const std::size_t count) {
    double w = 0.0;
    for (std::size_t k = first; k < first + count; ++k) {
      w += v[(k * nmo) + j] * v[(k * nmo) + j];
    }
    return w;
  };
  std::vector<std::size_t> rest(nmo);
  std::iota(rest.begin(), rest.end(), std::size_t{0});
  const auto take = [&](const std::size_t first, const std::size_t count) {
    std::ranges::sort(rest, [&](const std::size_t a, const std::size_t b) {
      return weight(a, first, count) > weight(b, first, count);
    });
    std::vector<std::size_t> taken(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(count));
    rest.erase(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(count));
    std::ranges::sort(taken, [&](const std::size_t a, const std::size_t b) {
      return occupations[a] > occupations[b];
    });
    return taken;
  };
  const std::vector<std::size_t> active = take(ncore, n);
  const std::vector<std::size_t> core = take(0, ncore);
  std::ranges::sort(rest, [&](const std::size_t a, const std::size_t b) {
    return occupations[a] > occupations[b];
  });
  double kept = 0.0;
  for (const std::size_t j : active) {
    kept += weight(j, ncore, n);
  }
  std::print("     trace(dD) {:.1e}, max |dD| between spaces {:.2e}, old active weight {:.4f} of "
             "{}; core down to {:.4f}, virtuals up to {:.4f}, active",
             trace, mixing, kept, n, core.empty() ? 2.0 : occupations[core.back()],
             rest.empty() ? 0.0 : occupations[rest.front()]);
  for (const std::size_t j : active) {
    std::print(" {:.4f}", occupations[j]);
  }
  std::println("");

  // C <- C U, U's columns the natural orbitals: core, active, virtual.
  std::vector<double> u(mo2);
  std::size_t column = 0;
  const std::vector<std::size_t> &virtuals = rest;
  for (const std::vector<std::size_t> *const group : {&core, &active, &virtuals}) {
    for (const std::size_t j : *group) {
      for (std::size_t k = 0; k < nmo; ++k) {
        u[k + (nmo * column)] = v[(k * nmo) + j];
      }
      ++column;
    }
  }
  fb::Buffers::device<double> u_device(mo2, res.device());
  fb::Buffers::device<double> rotated(mo2, res.device());
  FBF_TRY(copy(res, u_device.data(), u.data(), mo2, wwr::wwrMemcpyHostToDevice));
  FBF_TRY(fb::gpu_status(
      wwr::gemm<double, std::int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, p.nmo,
                                      p.nmo, p.nmo, &fb::kOne<double>, c, p.nmo, u_device.data(),
                                      p.nmo, &fb::kZero<double>, rotated.data(), p.nmo)));
  FBF_TRY(res.sync());
  return rotated;
}

// One natural-orbital pass from @p at: E at the natural orbitals of the
// PT2-corrected density D = D_CAS + dE_PT2/dh (pc_adjoint's h_bar plus the
// canonical condition's z^c), the unrelaxed response density, picked by
// natural_orbitals(). The CI starts cold, since the active orbitals have
// changed.
fb::Result<Point> natural(const Problem &p, const Point &at) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const std::size_t mo2 = nmo * nmo;
  const double *const c = at.c.data();
  const double *const vector = at.roots.vectors.data();
  const fb::nevpt2::PcAdjoint corr =
      FBF_TRY(fb::nevpt2::pc_adjoint(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c,
                                     p.h.data(), p.ncore, p.ncas, vector, 0));
  const fb::nevpt2::ZcMultipliers zc = FBF_TRY(
      fb::nevpt2::zc_multipliers(res, p.nmo, p.ncore, p.ncas, corr.a.data(), corr.epsilon.data()));
  const fb::nevpt2::ZcAdjoint phi = FBF_TRY(
      fb::nevpt2::zc_adjoint(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c, p.h.data(),
                             p.ncore, p.ncas, std::array{1.0}, vector, zc.z.data()));
  std::vector<double> d = FBF_TRY(download(res, corr.h_bar.data(), mo2));
  const std::vector<double> z = FBF_TRY(download(res, phi.h_bar.data(), mo2));
  std::ranges::transform(d, z, d.begin(), std::plus<>{});
  const fb::Buffers::device<double> rotated =
      FBF_TRY(natural_orbitals(p, c, std::move(d), at.gamma));
  return evaluate(p, rotated.data(), {});
}

// The state-averaged functional at one set of orbitals: the CASCI roots
// there, the core and virtuals made canonical over the state-averaged Fock,
// and PC-QD-NEVPT2's effective Hamiltonian over the roots with its states.
struct Average {
  fb::SphericalMatrixAO c;
  fb::ci::Roots roots;
  // The state-averaged active 1-RDM.
  std::vector<double> gamma;
  // E_I delta_IJ + (C_IJ + C_JI) / 2, n x n: the pipeline's H_eff.
  std::vector<double> heff;
  // Its eigenvalues, ascending, and each state's weight on CASCI root 0.
  std::vector<double> e_qd;
  std::vector<double> on_root0;
};

// The functional at the orbitals @p c_in (device, nmo^2) over the roots at
// @p weights, the CI started from @p start (every root's, or empty).
fb::Result<Average> average(const Problem &p, const double *c_in,
                            const std::span<const double> weights,
                            const std::span<const double> start) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const auto n = static_cast<std::size_t>(p.ncas);
  const std::size_t mo2 = nmo * nmo;
  const std::size_t nroots = weights.size();
  const auto dim = static_cast<std::size_t>(p.rdm.size());
  fb::SphericalMatrixAO c(res, p.basis);
  FBF_TRY(copy(res, c.data(), c_in, mo2, wwr::wwrMemcpyDeviceToDevice));
  const fb::CasscfIntegrals ints =
      FBF_TRY(fb::casscf_integrals(res, p.basis, p.molecule, p.h, p.b, p.naux, c, p.ncore, p.ncas));
  FBF_TRY(set_sigma(p, ints));
  fb::Buffers::device<double> diagonal(dim, res.device());
  FBF_TRY(p.sigma.diagonal(res, diagonal.data()));
  fb::ci::Roots roots = FBF_TRY(fb::ci::solve(res, p.sigma, p.alpha, p.beta, diagonal.data(),
                                              static_cast<std::int32_t>(nroots), {}, start));
  fb::Buffers::device<double> gamma_device(n * n, res.device());
  fb::Buffers::device<double> gamma2_device(n * n * n * n, res.device());
  FBF_TRY(
      p.rdm.rdm12(res, weights, roots.vectors.data(), gamma_device.data(), gamma2_device.data()));
  std::vector<double> gamma = FBF_TRY(download(res, gamma_device.data(), n * n));
  const std::vector<double> gamma2 = FBF_TRY(download(res, gamma2_device.data(), n * n * n * n));

  // Canonical core and virtuals over the state-averaged Fock: C <- C R. The
  // roots do not see it.
  {
    const fb::response::Fock fock =
        FBF_TRY(fb::response::fock(res, p.molecule, p.h, p.b, p.naux, c, p.ncore, p.ncas, ints.ppaa,
                                   ints.papa, gamma, gamma2));
    std::vector<double> f = FBF_TRY(download(res, fock.fock_inactive.data(), mo2));
    const std::vector<double> fa = FBF_TRY(download(res, fock.fock_active.data(), mo2));
    std::ranges::transform(f, fa, f.begin(), std::plus<>{});
    std::vector<double> r;
    std::vector<double> energies;
    FBF_TRY(fb::canonicalize_core_virtual(res, nmo, static_cast<std::size_t>(p.ncore),
                                          static_cast<std::size_t>(p.ncore + p.ncas), f, r,
                                          energies));
    fb::Buffers::device<double> r_device(mo2, res.device());
    fb::Buffers::device<double> rotated(mo2, res.device());
    FBF_TRY(copy(res, r_device.data(), r.data(), mo2, wwr::wwrMemcpyHostToDevice));
    FBF_TRY(fb::gpu_status(wwr::gemm<double, std::int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, p.nmo, p.nmo, p.nmo,
        &fb::kOne<double>, c.data(), p.nmo, r_device.data(), p.nmo, &fb::kZero<double>,
        rotated.data(), p.nmo)));
    FBF_TRY(copy(res, c.data(), rotated.data(), mo2, wwr::wwrMemcpyDeviceToDevice));
  }
  const std::vector<fb::nevpt2::PcClassEnergies> elements = FBF_TRY(
      fb::nevpt2::qd_pc_energy(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c.data(),
                               p.h.data(), p.ncore, p.ncas, weights, roots.vectors.data(), 0));
  std::vector<double> heff(nroots * nroots);
  for (std::size_t i = 0; i < nroots; ++i) {
    for (std::size_t j = 0; j < nroots; ++j) {
      heff[(i * nroots) + j] =
          (i == j ? roots.energies[i] : 0.0) +
          (0.5 * (elements[(i * nroots) + j].total + elements[(j * nroots) + i].total));
    }
  }
  std::vector<double> v;
  const std::vector<double> unsorted = eigensystem(heff, nroots, &v);
  std::vector<std::size_t> order(nroots);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::ranges::sort(
      order, [&](const std::size_t a, const std::size_t b) { return unsorted[a] < unsorted[b]; });
  std::vector<double> e_qd(nroots);
  std::vector<double> on_root0(nroots);
  for (std::size_t k = 0; k < nroots; ++k) {
    e_qd[k] = unsorted[order[k]];
    on_root0[k] = v[order[k]] * v[order[k]];
  }
  return Average{.c = std::move(c),
                 .roots = std::move(roots),
                 .gamma = std::move(gamma),
                 .heff = std::move(heff),
                 .e_qd = std::move(e_qd),
                 .on_root0 = std::move(on_root0)};
}

// One natural-orbital pass of a state average from @p at: natural()'s, over
// D = D_SA + dE[W]/dh, E[W] = sum_I w_I H_II the weighted trace of the
// PC-QD-NEVPT2 effective Hamiltonian's correlation part (qd_pc_adjoint's
// h_bar at W = diag(w), plus the canonical condition's z^c). At equal
// weights that trace does not depend on how the model space is rotated, so
// D is the average of the QD states' unrelaxed correlated densities.
fb::Result<Average> natural(const Problem &p, const Average &at,
                            const std::span<const double> weights) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const std::size_t mo2 = nmo * nmo;
  const std::size_t nroots = weights.size();
  const double *const c = at.c.data();
  const double *const vectors = at.roots.vectors.data();
  std::vector<double> w(nroots * nroots, 0.0);
  for (std::size_t r = 0; r < nroots; ++r) {
    w[(r * nroots) + r] = weights[r];
  }
  const fb::nevpt2::QdPcAdjoint corr =
      FBF_TRY(fb::nevpt2::qd_pc_adjoint(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c,
                                        p.h.data(), p.ncore, p.ncas, weights, vectors, w, 0));
  const fb::nevpt2::ZcMultipliers zc = FBF_TRY(
      fb::nevpt2::zc_multipliers(res, p.nmo, p.ncore, p.ncas, corr.a.data(), corr.epsilon.data()));
  const fb::nevpt2::ZcAdjoint phi =
      FBF_TRY(fb::nevpt2::zc_adjoint(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c,
                                     p.h.data(), p.ncore, p.ncas, weights, vectors, zc.z.data()));
  std::vector<double> d = FBF_TRY(download(res, corr.h_bar.data(), mo2));
  const std::vector<double> z = FBF_TRY(download(res, phi.h_bar.data(), mo2));
  std::ranges::transform(d, z, d.begin(), std::plus<>{});
  const fb::Buffers::device<double> rotated =
      FBF_TRY(natural_orbitals(p, c, std::move(d), at.gamma));
  return average(p, rotated.data(), weights, {});
}

void print_average(const std::int32_t pass, const Average &at) {
  const std::size_t n = at.e_qd.size();
  std::print("{:>4}  E_CAS", pass);
  for (const double e : at.roots.energies) {
    std::print(" {:.10f}", e);
  }
  std::print("  H_eff");
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i; j < n; ++j) {
      std::print(" {:.10f}", at.heff[(i * n) + j]);
    }
  }
  std::print("  E_QD");
  for (const double e : at.e_qd) {
    std::print(" {:.10f}", e);
  }
  std::print("  on root 0");
  for (const double o : at.on_root0) {
    std::print(" {:.4f}", o);
  }
  std::println("");
}

// The state-averaged natural-orbital passes (design.html, "The state
// average"): from the config's SA-CASSCF, @p passes of natural() above, one
// row each: the CASCI roots' energies, H_eff's upper triangle by rows, the
// QD states' energies and each one's weight on root 0.
fb::Status averaged(const Problem &p, const fb::CasscfOutput &casscf, const std::int32_t passes) {
  fb::Buffers::device<double> start(casscf.vectors.size(), p.res.device());
  FBF_TRY(copy(p.res, start.data(), casscf.vectors.data(), casscf.vectors.size(),
               wwr::wwrMemcpyHostToDevice));
  Average at =
      FBF_TRY(average(p, casscf.c.data(), casscf.weights, {start.data(), casscf.vectors.size()}));
  std::print("SA-CASSCF({},{}) of {} roots, nmo {} ncore {}: pipeline E_CAS", casscf.nelecas,
             casscf.ncas, casscf.weights.size(), p.nmo, p.ncore);
  for (const double e : casscf.energies) {
    std::print(" {:.10f}", e);
  }
  std::println("");
  print_average(0, at);
  for (std::int32_t pass = 1; pass <= passes; ++pass) {
    at = FBF_TRY(natural(p, at, casscf.weights));
    print_average(pass, at);
  }
  return {};
}

// A point and its weight.
struct Weighed {
  Point point;
  Weight w;
};

// The functional at the orbitals C0 exp(kappa(x)), x the non-redundant
// rotations packed (response::unpack_orbital_pairs), the CI started from
// @p start; none where the CASCI or a PC class refuses, @p why said if set.
std::optional<Weighed> weigh(const Problem &p, const double *c0, const std::span<const double> x,
                             const std::span<const double> start, const bool pt2,
                             const char *const why = nullptr) {
  const auto run = [&]() -> fb::Result<Weighed> {
    fb::DeviceResources &res = p.res;
    const auto mo2 = static_cast<std::size_t>(p.nmo) * static_cast<std::size_t>(p.nmo);
    fb::Buffers::device<double> packed(x.size(), res.device());
    fb::Buffers::device<double> kappa(mo2, res.device());
    fb::Buffers::device<wwr::wwrDoubleComplex> u(mo2, res.device());
    fb::Buffers::device<double> w(static_cast<std::size_t>(p.nmo), res.device());
    fb::Buffers::device<double> rotation(mo2, res.device());
    fb::Buffers::device<double> c(mo2, res.device());
    FBF_TRY(copy(res, packed.data(), x.data(), x.size(), wwr::wwrMemcpyHostToDevice));
    FBF_TRY(fb::response::unpack_orbital_pairs(res, p.nmo, p.ncore, p.ncas, packed.data(),
                                               kappa.data()));
    FBF_TRY(fb::skew_eig<double>(res, p.nmo, kappa.data(), u.data(), w.data()));
    FBF_TRY(fb::skew_expm<double>(res, p.nmo, u.data(), w.data(), 1.0, rotation.data()));
    FBF_TRY(fb::gpu_status(wwr::gemm<double, std::int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, p.nmo, p.nmo, p.nmo,
        &fb::kOne<double>, c0, p.nmo, rotation.data(), p.nmo, &fb::kZero<double>, c.data(),
        p.nmo)));
    FBF_TRY(res.sync());
    Point point = FBF_TRY(evaluate(p, c.data(), start, pt2));
    const Weight weighed = FBF_TRY(weight(p, point));
    return Weighed{.point = std::move(point), .w = weighed};
  };
  fb::Result<Weighed> out = run();
  if (!out.has_value()) {
    if (why != nullptr) {
      std::println("     {}: {}", why, out.error().message);
    }
    return std::nullopt;
  }
  return std::move(*out);
}

// dW/dx and d2W/dx2 over the rotations @p take names, at @p x where @p at
// is: central differences over kRotation, the CI relaxed at every point.
// With @p pt2, E's as well.
struct Slope {
  std::vector<double> g;
  std::vector<double> curvature;
  std::vector<double> g_e;
  std::vector<double> curvature_e;
};

fb::Result<Slope> differences(const Problem &p, const double *c0, const std::vector<double> &x,
                              const Weighed &at, const std::vector<bool> &take, const bool pt2) {
  const std::size_t n = x.size();
  Slope out{.g = std::vector<double>(n, 0.0),
            .curvature = std::vector<double>(n, 0.0),
            .g_e = std::vector<double>(n, 0.0),
            .curvature_e = std::vector<double>(n, 0.0)};
  const std::span<const double> start{at.point.roots.vectors.data(),
                                      static_cast<std::size_t>(p.rdm.size())};
  std::vector<double> moved = x;
  for (std::size_t i = 0; i < n; ++i) {
    if (!take[i]) {
      continue;
    }
    moved[i] = x[i] + kRotation;
    const std::optional<Weighed> plus = weigh(p, c0, moved, start, pt2, "gradient");
    moved[i] = x[i] - kRotation;
    const std::optional<Weighed> minus = weigh(p, c0, moved, start, pt2, "gradient");
    moved[i] = x[i];
    if (!plus || !minus) {
      return fb::err_numerical("differences: a gradient point was refused");
    }
    out.g[i] = (plus->w.total - minus->w.total) / (2.0 * kRotation);
    out.curvature[i] =
        (plus->w.total - (2.0 * at.w.total) + minus->w.total) / (kRotation * kRotation);
    if (pt2) {
      out.g_e[i] = (plus->point.total() - minus->point.total()) / (2.0 * kRotation);
      out.curvature_e[i] = (plus->point.total() - (2.0 * at.point.total()) + minus->point.total()) /
                           (kRotation * kRotation);
    }
  }
  return out;
}

void print_weight(const std::string_view label, const Weight &w) {
  std::print("{:<22} W {:.6f} (reference weight {:.6f});", label, w.total, 1.0 / (1.0 + w.total));
  constexpr std::array<std::string_view, 8> kNames{"Sr",   "Si",  "Sijrs", "Sijr",
                                                   "Srsi", "Srs", "Sij",   "Sir"};
  for (std::size_t k = 0; k < 8; ++k) {
    std::print(" {} {:.5f}", kNames[k], w.classes[k]);
  }
  std::println("");
}

// The maximum-overlap orbitals (design.html, "Maximum overlap"): W, the
// weight of the first-order wave function, minimized over the non-redundant
// rotations x from @p origin's orbitals, C = C0 exp(kappa(x)). To first
// order the overlap of the reference with the exact state is
// (1 + W)^(-1/2), so this is Brueckner's criterion at this order. W is
// bounded below, which E_CAS + E_PT2 is not.
//
// The gradient is a central difference over kRotation in each rotation, the
// CI relaxed at every point; the rotations whose derivative vanishes at the
// start (by symmetry) are left out and checked at the end. The step is BFGS
// from the differences' own diagonal curvature, backtracked on W.
fb::Status maximum_overlap(const Problem &p, const Point &origin, const std::int32_t iterations) {
  fb::DeviceResources &res = p.res;
  const std::size_t n = origin.fock.gradient.num_elements();
  const auto dim = static_cast<std::size_t>(p.rdm.size());
  const double *const c0 = origin.c.data();
  std::vector<double> x(n, 0.0);
  std::optional<Weighed> first =
      weigh(p, c0, x, {origin.roots.vectors.data(), dim}, true, "the CASSCF orbitals");
  if (!first) {
    return fb::err_numerical("maximum_overlap: no weight at the CASSCF orbitals");
  }
  Weighed at = std::move(*first);
  const Weight w0 = at.w;
  const double e_cas0 = at.point.e_cas;
  const double e_pt20 = at.point.e_pt2;

  const auto slope = [&](const std::vector<bool> &take) {
    return differences(p, c0, x, at, take, false);
  };
  const auto largest = [&](const std::vector<double> &v, const std::vector<bool> &take) {
    double out = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (take[i]) {
        out = std::max(out, std::abs(v[i]));
      }
    }
    return out;
  };

  std::vector<bool> free(n, true);
  Slope s = FBF_TRY(slope(free));
  std::size_t count = 0;
  double low = std::numeric_limits<double>::infinity();
  double high = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    free[i] = std::abs(s.g[i]) >= kSymmetryZero;
    if (free[i]) {
      ++count;
      low = std::min(low, s.curvature[i]);
      high = std::max(high, s.curvature[i]);
    }
  }
  std::println("{} rotations, {} with a derivative above {:g}; their curvature {:.3e} to {:.3e}", n,
               count, kSymmetryZero, low, high);
  // The inverse Hessian, n x n: the diagonal curvature's, floored.
  const double floor = 0.05 * high;
  std::vector<double> inverse(n * n, 0.0);
  const auto reset = [&](const Slope &from) {
    std::ranges::fill(inverse, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      if (free[i]) {
        inverse[(i * n) + i] = 1.0 / std::max(from.curvature[i], floor);
      }
    }
  };
  reset(s);

  std::println("{:>4} {:>12} {:>16} {:>14} {:>16} {:>10} {:>10} {:>6}", "iter", "W", "E_CAS",
               "E_PT2", "E_total", "max|dW|", "|g_CAS|", "t");
  double t = 0.0;
  for (std::int32_t iter = 0;; ++iter) {
    const std::vector<double> g_cas = FBF_TRY(download(res, at.point.fock.gradient.data(), n));
    double g_max = largest(s.g, free);
    std::println("{:>4} {:>12.8f} {:>16.10f} {:>14.10f} {:>16.10f} {:>10.3e} {:>10.3e} {:>6}", iter,
                 at.w.total, at.point.e_cas, at.point.e_pt2, at.point.total(), g_max, norm(g_cas),
                 t);
    if (g_max < kWeightGradient) {
      // The rotations left out: any that has grown a derivative joins.
      std::vector<bool> rest(n);
      for (std::size_t i = 0; i < n; ++i) {
        rest[i] = !free[i];
      }
      const Slope others = FBF_TRY(slope(rest));
      std::size_t joined = 0;
      for (std::size_t i = 0; i < n; ++i) {
        if (rest[i] && std::abs(others.g[i]) >= kWeightGradient) {
          free[i] = true;
          s.g[i] = others.g[i];
          s.curvature[i] = others.curvature[i];
          inverse[(i * n) + i] = 1.0 / std::max(others.curvature[i], floor);
          ++joined;
        }
      }
      std::println("     max |dW| over the rotations left out {:.3e}; {} join",
                   largest(others.g, rest), joined);
      if (joined == 0) {
        break;
      }
      g_max = largest(s.g, free);
    }
    if (iter >= iterations) {
      std::println("not converged in {} iterations", iterations);
      break;
    }
    const auto direction = [&] {
      std::vector<double> d(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          d[i] -= inverse[(i * n) + j] * s.g[j];
        }
      }
      return d;
    };
    std::vector<double> d = direction();
    double descent = std::inner_product(d.begin(), d.end(), s.g.begin(), 0.0);
    if (!(descent < 0.0)) {
      reset(s);
      d = direction();
      descent = std::inner_product(d.begin(), d.end(), s.g.begin(), 0.0);
    }
    const double longest =
        std::ranges::max(d | std::views::transform([](const double v) { return std::abs(v); }));
    if (longest > kTrustStep) {
      for (double &v : d) {
        v *= kTrustStep / longest;
      }
      descent *= kTrustStep / longest;
    }
    std::optional<Weighed> next;
    std::vector<double> x_next;
    for (t = 1.0; t >= 1.0 / 128.0 && !next; t /= 2.0) {
      x_next = x;
      for (std::size_t i = 0; i < n; ++i) {
        x_next[i] += t * d[i];
      }
      std::optional<Weighed> trial =
          weigh(p, c0, x_next, {at.point.roots.vectors.data(), dim}, true, "trial");
      if (trial && trial->w.total < at.w.total + (1e-4 * t * descent)) {
        next = std::move(trial);
        break;
      }
    }
    if (!next) {
      std::println("no descent along the step down to t = 1/128: stopping");
      break;
    }
    std::vector<double> step(n);
    for (std::size_t i = 0; i < n; ++i) {
      step[i] = x_next[i] - x[i];
    }
    x = std::move(x_next);
    at = std::move(*next);
    const Slope fresh = FBF_TRY(slope(free));
    // BFGS on the inverse: H <- (1 - r s y^T) H (1 - r y s^T) + r s s^T.
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = fresh.g[i] - s.g[i];
    }
    const double sy = std::inner_product(step.begin(), step.end(), y.begin(), 0.0);
    if (sy > 1e-14) {
      const double r = 1.0 / sy;
      std::vector<double> hy(n, 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          hy[i] += inverse[(i * n) + j] * y[j];
        }
      }
      const double yhy = std::inner_product(y.begin(), y.end(), hy.begin(), 0.0);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          inverse[(i * n) + j] += (r * (1.0 + (r * yhy)) * step[i] * step[j]) -
                                  (r * ((hy[i] * step[j]) + (step[i] * hy[j])));
        }
      }
    }
    s = fresh;
  }
  print_weight("CASSCF orbitals:", w0);
  print_weight("maximum overlap:", at.w);
  std::print(
      "largest rotation {:.4f} rad; occupations",
      std::ranges::max(x | std::views::transform([](const double v) { return std::abs(v); })));
  for (const double o : eigenvalues(at.point.gamma, static_cast<std::size_t>(p.ncas))) {
    std::print(" {:.4f}", o);
  }
  std::println("");
  std::println("final {:>16.10f} {:>14.10f} {:>16.10f}", at.point.e_cas, at.point.e_pt2,
               at.point.total());
  std::println("change over CASSCF orbitals: E_CAS {:+.3e}  E_PT2 {:+.3e}  total {:+.3e} hartree",
               at.point.e_cas - e_cas0, at.point.e_pt2 - e_pt20,
               at.point.total() - e_cas0 - e_pt20);
  return {};
}

// The capped functional (design.html, "Capping the weight"): E minimized
// over the non-redundant rotations x from @p origin's orbitals under
// W(x) <= cap, cap = @p factor W(0). E_PT2 >= -W max(D), D the perturbers'
// denominators, so under the cap E is bounded below, which bare it is not.
//
// Both gradients are differences(); the step is the quadratic program's
// with the one linearized constraint, over a BFGS inverse Hessian of the
// Lagrangian E + lambda W, backtracked on the merit E + mu max(0, W - cap).
fb::Status capped(const Problem &p, const Point &origin, const std::int32_t iterations,
                  const double factor) {
  fb::DeviceResources &res = p.res;
  const std::size_t n = origin.fock.gradient.num_elements();
  const auto dim = static_cast<std::size_t>(p.rdm.size());
  const double *const c0 = origin.c.data();
  std::vector<double> x(n, 0.0);
  std::optional<Weighed> first =
      weigh(p, c0, x, {origin.roots.vectors.data(), dim}, true, "the CASSCF orbitals");
  if (!first) {
    return fb::err_numerical("capped: no weight at the CASSCF orbitals");
  }
  Weighed at = std::move(*first);
  const Weight w0 = at.w;
  const double cap = factor * w0.total;
  const double e_cas0 = at.point.e_cas;
  const double e_pt20 = at.point.e_pt2;
  const auto dot = [](const std::vector<double> &a, const std::vector<double> &b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  };

  std::vector<bool> free(n, true);
  Slope s = FBF_TRY(differences(p, c0, x, at, free, true));
  std::size_t count = 0;
  for (std::size_t i = 0; i < n; ++i) {
    free[i] = std::abs(s.g[i]) >= kSymmetryZero || std::abs(s.g_e[i]) >= kSymmetryZero;
    if (free[i]) {
      ++count;
    } else {
      s.g[i] = 0.0;
      s.g_e[i] = 0.0;
    }
  }
  // The multiplier's first estimate, by least squares, and the inverse
  // Hessian: the Lagrangian's diagonal curvature, floored.
  double lambda = std::max(0.0, -dot(s.g_e, s.g) / dot(s.g, s.g));
  double low = std::numeric_limits<double>::infinity();
  double high = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (free[i]) {
      const double curvature = s.curvature_e[i] + (lambda * s.curvature[i]);
      low = std::min(low, curvature);
      high = std::max(high, curvature);
    }
  }
  std::println("cap W <= {:.8f} ({} W at the CASSCF orbitals)", cap, factor);
  std::println("{} rotations, {} with a derivative above {:g}; first multiplier {:.4f} hartree, "
               "the Lagrangian's curvature {:.3e} to {:.3e}",
               n, count, kSymmetryZero, lambda, low, high);
  std::println("     at the CASSCF orbitals |dE/dx| {:.3e}, |dW/dx| {:.3e}, cosine {:.4f}",
               norm(s.g_e), norm(s.g), dot(s.g_e, s.g) / (norm(s.g_e) * norm(s.g)));
  const double floor = 0.05 * high;
  std::vector<double> inverse(n * n, 0.0);
  const auto reset = [&](const Slope &from, const double multiplier) {
    std::ranges::fill(inverse, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      if (free[i]) {
        inverse[(i * n) + i] =
            1.0 / std::max(from.curvature_e[i] + (multiplier * from.curvature[i]), floor);
      }
    }
  };
  reset(s, lambda);
  const auto apply = [&](const std::vector<double> &v) {
    std::vector<double> out(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        out[i] += inverse[(i * n) + j] * v[j];
      }
    }
    return out;
  };
  // The quadratic program's step and multiplier at the current point.
  const auto program = [&] {
    const std::vector<double> he = apply(s.g_e);
    const std::vector<double> hw = apply(s.g);
    const double whw = dot(s.g, hw);
    const double over = at.w.total - cap;
    lambda = whw > 0.0 ? std::max(0.0, (over - dot(s.g, he)) / whw) : 0.0;
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) {
      d[i] = -(he[i] + (lambda * hw[i]));
    }
    return d;
  };
  const auto lagrangian = [&](const Slope &from) {
    double out = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (free[i]) {
        out = std::max(out, std::abs(from.g_e[i] + (lambda * from.g[i])));
      }
    }
    return out;
  };

  std::println("{:>4} {:>12} {:>11} {:>16} {:>14} {:>16} {:>9} {:>10} {:>10} {:>8}", "iter", "W",
               "W - cap", "E_CAS", "E_PT2", "E_total", "lambda", "max|dL|", "|g_CAS|", "t");
  double mu = 1.0;
  double t = 0.0;
  for (std::int32_t iter = 0;; ++iter) {
    std::vector<double> d = program();
    const double over = at.w.total - cap;
    const std::vector<double> g_cas = FBF_TRY(download(res, at.point.fock.gradient.data(), n));
    std::println("{:>4} {:>12.8f} {:>+11.2e} {:>16.10f} {:>14.10f} {:>16.10f} {:>9.4f} {:>10.3e} "
                 "{:>10.3e} {:>8}",
                 iter, at.w.total, over, at.point.e_cas, at.point.e_pt2, at.point.total(), lambda,
                 lagrangian(s), norm(g_cas), t);
    if (lagrangian(s) < kLagrangianGradient && over < kCapTolerance) {
      // The rotations left out: any that has grown a derivative joins.
      std::vector<bool> rest(n);
      for (std::size_t i = 0; i < n; ++i) {
        rest[i] = !free[i];
      }
      const Slope others = FBF_TRY(differences(p, c0, x, at, rest, true));
      std::size_t joined = 0;
      double worst = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        if (!rest[i]) {
          continue;
        }
        const double g = std::max(std::abs(others.g[i]), std::abs(others.g_e[i]));
        worst = std::max(worst, g);
        if (g >= kLagrangianGradient) {
          free[i] = true;
          s.g[i] = others.g[i];
          s.g_e[i] = others.g_e[i];
          s.curvature[i] = others.curvature[i];
          s.curvature_e[i] = others.curvature_e[i];
          inverse[(i * n) + i] =
              1.0 / std::max(others.curvature_e[i] + (lambda * others.curvature[i]), floor);
          ++joined;
        }
      }
      std::println("     max |dE|, |dW| over the rotations left out {:.3e}; {} join", worst,
                   joined);
      if (joined == 0) {
        break;
      }
      d = program();
    }
    if (iter >= iterations) {
      std::println("not converged in {} iterations", iterations);
      break;
    }
    // The merit's slope along a step: E's, less what the step takes off the
    // overshoot.
    const auto merit_slope = [&](const std::vector<double> &step) {
      return dot(s.g_e, step) - (mu * (std::max(0.0, over) - std::max(0.0, over + dot(s.g, step))));
    };
    const auto trust = [&](std::vector<double> &step) {
      const double longest = std::ranges::max(
          step | std::views::transform([](const double v) { return std::abs(v); }));
      if (longest > kTrustStep) {
        for (double &v : step) {
          v *= kTrustStep / longest;
        }
      }
    };
    trust(d);
    mu = std::max(mu, 2.0 * lambda);
    double descent = merit_slope(d);
    if (!(descent < 0.0)) {
      reset(s, lambda);
      d = program();
      trust(d);
      mu = std::max(mu, 2.0 * lambda);
      descent = merit_slope(d);
    }
    const double step_lambda = lambda;
    const double merit = at.point.total() + (mu * std::max(0.0, over));
    std::optional<Weighed> next;
    std::vector<double> x_next;
    for (t = 1.0; t >= 1.0 / 128.0 && !next; t /= 2.0) {
      x_next = x;
      for (std::size_t i = 0; i < n; ++i) {
        x_next[i] += t * d[i];
      }
      std::optional<Weighed> trial =
          weigh(p, c0, x_next, {at.point.roots.vectors.data(), dim}, true, "trial");
      if (trial && trial->point.total() + (mu * std::max(0.0, trial->w.total - cap)) <
                       merit + (1e-4 * t * descent)) {
        next = std::move(trial);
        break;
      }
    }
    if (!next) {
      std::println("no descent of the merit along the step down to t = 1/128: stopping");
      break;
    }
    std::vector<double> step(n);
    for (std::size_t i = 0; i < n; ++i) {
      step[i] = x_next[i] - x[i];
    }
    x = std::move(x_next);
    at = std::move(*next);
    const Slope fresh = FBF_TRY(differences(p, c0, x, at, free, true));
    // BFGS on the inverse, over the Lagrangian's gradient at the step's
    // multiplier: H <- (1 - r s y^T) H (1 - r y s^T) + r s s^T.
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] = (fresh.g_e[i] - s.g_e[i]) + (step_lambda * (fresh.g[i] - s.g[i]));
    }
    const double sy = dot(step, y);
    if (sy > 1e-14) {
      const double r = 1.0 / sy;
      const std::vector<double> hy = apply(y);
      const double yhy = dot(y, hy);
      for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          inverse[(i * n) + j] += (r * (1.0 + (r * yhy)) * step[i] * step[j]) -
                                  (r * ((hy[i] * step[j]) + (step[i] * hy[j])));
        }
      }
    }
    s = fresh;
  }
  print_weight("CASSCF orbitals:", w0);
  print_weight("capped:", at.w);
  std::print(
      "multiplier {:.6f} hartree; largest rotation {:.4f} rad; occupations", lambda,
      std::ranges::max(x | std::views::transform([](const double v) { return std::abs(v); })));
  for (const double o : eigenvalues(at.point.gamma, static_cast<std::size_t>(p.ncas))) {
    std::print(" {:.4f}", o);
  }
  std::println("");
  std::println("final {:>16.10f} {:>14.10f} {:>16.10f}", at.point.e_cas, at.point.e_pt2,
               at.point.total());
  std::println("change over CASSCF orbitals: E_CAS {:+.3e}  E_PT2 {:+.3e}  total {:+.3e} hartree",
               at.point.e_cas - e_cas0, at.point.e_pt2 - e_pt20,
               at.point.total() - e_cas0 - e_pt20);
  return {};
}

fb::Status run(const std::span<char *const> args) {
  const std::string_view mode = args.size() >= 4 ? std::string_view(args[3]) : "oo";
  if (args.size() < 2 || args.size() > 5 ||
      (mode != "no" && mode != "oo" && mode != "mo" && mode != "cap" && mode != "sa") ||
      (args.size() == 5 && mode != "cap")) {
    return fb::err_config("usage: {} <config.toml> [iterations] [oo|no|mo|cap|sa] [cap factor]",
                          args.empty() ? "free_breakfast_oo_nevpt2" : args[0]);
  }
  const std::filesystem::path config = args[1];
  const std::int32_t iterations = args.size() >= 3 ? std::atoi(args[2]) : 8;
  const bool natural_orbitals = mode == "no";
  const std::filesystem::path out =
      std::filesystem::temp_directory_path() / "free_breakfast_oo_nevpt2";
  fb::Calculation calc = FBF_TRY(fb::read_config(config, config.parent_path(), out));
  fb::NodeResources node{{.workspaces = {.device = {.workspace = std::size_t{256} << 20},
                                         .pinned = {.workspace = std::size_t{4} << 20}}}};
  FBF_TRY(calc.run(node));
  const auto &casscf = calc.get<fb::CasscfOutput>();
  const auto &eri = calc.get<fb::EriOutput>();
  if (!casscf.has_value() || !eri.has_value() ||
      !std::holds_alternative<fb::DensityFitted>(eri->eri) ||
      (casscf->energies.size() == 1) == (mode == "sa") || casscf->nelecas % 2 != 0) {
    return fb::err_config("{}: needs [casscf] of one singlet root (of two or more with sa) over "
                          "density-fitted [eri]",
                          config.string());
  }
  fb::DeviceResources &res = node.device(0);
  const auto &df = std::get<fb::DensityFitted>(eri->eri);
  fb::ci::DeviceStringSet alpha =
      FBF_TRY(fb::ci::DeviceStringSet::create(res, casscf->ncas, casscf->nelecas / 2));
  fb::ci::DeviceStringSet beta =
      FBF_TRY(fb::ci::DeviceStringSet::create(res, casscf->ncas, casscf->nelecas / 2));
  fb::ci::Sigma sigma = FBF_TRY(fb::ci::Sigma::create(res, alpha, beta));
  fb::ci::Rdm rdm = FBF_TRY(fb::ci::Rdm::create(res, alpha, beta));
  const Problem p{.res = res,
                  .molecule = calc.get<fb::MoleculeOutput>()->molecule,
                  .basis = calc.get<fb::BasisSetOutput>()->basis,
                  .h = calc.get<fb::CoreHamiltonianOutput>()->h,
                  .s = calc.get<fb::OverlapOutput>()->s,
                  .b = df.b,
                  .naux = df.naux,
                  .nmo = static_cast<std::int32_t>(casscf->c.n()),
                  .ncore = casscf->ncore,
                  .ncas = casscf->ncas,
                  .alpha = alpha,
                  .beta = beta,
                  .sigma = sigma,
                  .rdm = rdm};

  if (mode == "sa") {
    return averaged(p, *casscf, iterations);
  }
  fb::Buffers::device<double> start(casscf->vectors.size(), res.device());
  FBF_TRY(copy(res, start.data(), casscf->vectors.data(), casscf->vectors.size(),
               wwr::wwrMemcpyHostToDevice));
  Point at = FBF_TRY(evaluate(p, casscf->c.data(), {start.data(), casscf->vectors.size()}));
  const double e_cas0 = at.e_cas;
  const double e_pt20 = at.e_pt2;
  std::println("CASSCF({},{}) nmo {} ncore {}: pipeline E_CAS {:.10f}", casscf->nelecas,
               casscf->ncas, p.nmo, p.ncore, casscf->energy);
  if (const auto &fci = calc.get<fb::FciOutput>(); fci.has_value()) {
    // The exact answer, and the functional in the exact state's natural
    // orbitals, by occupation: the ncore most occupied the core, the next
    // ncas the active space. What orbitals that have seen all the
    // correlation are worth to NEVPT2.
    std::println("FCI {:.10f}", fci->energies.front());
    const auto nmo = static_cast<std::size_t>(p.nmo);
    const std::size_t mo2 = nmo * nmo;
    std::vector<double> v;
    const std::vector<double> occupations = eigensystem(
        {fci->rdm1.begin(), fci->rdm1.begin() + static_cast<std::ptrdiff_t>(mo2)}, nmo, &v);
    std::vector<std::size_t> order(nmo);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::sort(order, [&](const std::size_t a, const std::size_t b) {
      return occupations[a] > occupations[b];
    });
    const std::vector<double> c_scf =
        FBF_TRY(download(res, calc.get<fb::ScfOutput>()->c.data(), mo2));
    std::vector<double> c_no(mo2, 0.0);
    for (std::size_t column = 0; column < nmo; ++column) {
      for (std::size_t k = 0; k < nmo; ++k) {
        const double u = v[(k * nmo) + order[column]];
        for (std::size_t mu = 0; mu < nmo; ++mu) {
          c_no[mu + (nmo * column)] += c_scf[mu + (nmo * k)] * u;
        }
      }
    }
    fb::Buffers::device<double> c_device(mo2, res.device());
    FBF_TRY(copy(res, c_device.data(), c_no.data(), mo2, wwr::wwrMemcpyHostToDevice));
    const fb::Result<Point> exact = evaluate(p, c_device.data(), {});
    if (exact.has_value()) {
      std::println("FCI natural orbitals: E_CAS {:.10f}  E_PT2 {:.10f}  total {:.10f}",
                   exact->e_cas, exact->e_pt2, exact->total());
      print_weight("FCI natural orbitals:", FBF_TRY(weight(p, *exact)));
      std::print("     FCI occupations from the last core orbital to the first virtual:");
      for (std::int32_t k = std::max(p.ncore - 1, 0); k <= std::min(p.ncore + p.ncas, p.nmo - 1);
           ++k) {
        std::print(" {:.4f}", occupations[order[static_cast<std::size_t>(k)]]);
      }
      std::println("");
    } else {
      std::println("FCI natural orbitals: {}", exact.error().message);
    }
  }
  // W either way: the virtual orbitals' shift alone and the core's alone
  // weigh the four classes with both kinds of index, and must agree.
  {
    const Weight both = FBF_TRY(weight(p, at));
    const Weight virt = FBF_TRY(weight(p, at, 1.0, 0.0));
    const Weight core = FBF_TRY(weight(p, at, 0.0, 1.0));
    print_weight("CASSCF orbitals:", both);
    double worst = 0.0;
    for (const std::size_t k : {2U, 3U, 4U, 7U}) {
      worst = std::max({worst, std::abs(virt.classes[k] - both.classes[k]),
                        std::abs(core.classes[k] - both.classes[k])});
    }
    std::println("W by the virtual shift alone and by the core's: off by {:.1e} at the most",
                 worst);
  }
  if (mode == "mo") {
    return maximum_overlap(p, at, iterations);
  }
  if (mode == "cap") {
    return capped(p, at, iterations, args.size() == 5 ? std::atof(args[4]) : 1.0);
  }
  if (natural_orbitals) {
    // Natural orbitals of the PT2-corrected density, iterated: no functional
    // is minimized.
    std::println("{:>4} {:>16} {:>14} {:>16} {:>11} {:>11} {:>11}", "pass", "E_CAS", "E_PT2",
                 "E_total", "dE_CAS", "dE_PT2", "dE_total");
    std::println("{:>4} {:>16.10f} {:>14.10f} {:>16.10f}", 0, at.e_cas, at.e_pt2, at.total());
    for (std::int32_t pass = 1; pass <= iterations; ++pass) {
      Point next = FBF_TRY(natural(p, at));
      std::println("{:>4} {:>16.10f} {:>14.10f} {:>16.10f} {:>+11.3e} {:>+11.3e} {:>+11.3e}", pass,
                   next.e_cas, next.e_pt2, next.total(), next.e_cas - e_cas0, next.e_pt2 - e_pt20,
                   next.total() - e_cas0 - e_pt20);
      print_weight("     its weight:", FBF_TRY(weight(p, next)));
      const double change = next.total() - at.total();
      at = std::move(next);
      if (std::abs(change) < kEnergyTolerance) {
        break;
      }
    }
    return {};
  }
  std::println("{:>4} {:>16} {:>14} {:>16} {:>10} {:>10} {:>10} {:>10} {:>9} {:>13} {:>13} {:>6} "
               "{:>5}",
               "iter", "E_CAS", "E_PT2", "E_total", "|g|", "|A|", "|g+A|", "(g+A).Z", "max|Z|",
               "dE(1/2)", "dE(1)", "ratio", "t");
  for (std::int32_t iter = 0; iter < iterations; ++iter) {
    const Step step = FBF_TRY(quasi_newton(p, at));
    // The line search: t = 1 and 1/2, then halving while nothing descends.
    // A refused trial (a PC class's Numerical error) counts as no descent.
    const auto trial = [&](const double t) -> std::optional<Point> {
      fb::Result<Point> point = moved(p, at, step, t);
      if (!point.has_value()) {
        std::println("     t = {}: {}", t, point.error().message);
        return std::nullopt;
      }
      return std::move(*point);
    };
    std::optional<Point> full = trial(1.0);
    std::optional<Point> half = trial(0.5);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double d_full = full ? full->total() - at.total() : nan;
    const double d_half = half ? half->total() - at.total() : nan;
    double t = 0.0;
    std::optional<Point> best;
    if (full && d_full < 0.0 && !(half && d_half < d_full)) {
      t = 1.0;
      best = std::move(full);
      // Still descending at t = 1: double while it does.
      for (double s = 2.0; s <= kLongestStep; s *= 2.0) {
        std::optional<Point> point = trial(s);
        if (!point || !(point->total() < best->total())) {
          break;
        }
        t = s;
        best = std::move(point);
      }
    } else if (half && d_half < 0.0) {
      t = 0.5;
      best = std::move(half);
    } else {
      for (double s = 0.25; s >= 1.0 / 64.0 && !best; s /= 2.0) {
        std::optional<Point> point = trial(s);
        if (point && point->total() < at.total()) {
          t = s;
          best = std::move(point);
        }
      }
    }
    std::println("{:>4} {:>16.10f} {:>14.10f} {:>16.10f} {:>10.3e} {:>10.3e} {:>10.3e} {:>10.3e} "
                 "{:>9.2e} {:>13.3e} {:>13.3e} {:>6.3f} {:>5}",
                 iter, at.e_cas, at.e_pt2, at.total(), step.g_norm, step.a_norm, step.source_norm,
                 step.slope, step.max_kappa, d_half, d_full, d_half / d_full, t);
    std::print("     {} step, taken to dE {:.3e}; occupations", step.newton ? "Newton" : "descent",
               best ? best->total() - at.total() : 0.0);
    for (const double o : eigenvalues(at.gamma, static_cast<std::size_t>(p.ncas))) {
      std::print(" {:.4f}", o);
    }
    std::println("");
    if (const fb::Result<Weight> w = weight(p, at); w.has_value()) {
      print_weight("     its weight:", *w);
    }
    if (!best) {
      std::println("no descent along the step down to t = 1/64: stopping");
      break;
    }
    const double change = best->total() - at.total();
    at = std::move(*best);
    if (std::abs(change) < kEnergyTolerance) {
      std::println("|dE| {:.2e} below {:g}: stopping", std::abs(change), kEnergyTolerance);
      break;
    }
  }
  std::println("final {:>16.10f} {:>14.10f} {:>16.10f}", at.e_cas, at.e_pt2, at.total());
  std::println("change over CASSCF orbitals: E_CAS {:+.3e}  E_PT2 {:+.3e}  total {:+.3e} hartree",
               at.e_cas - e_cas0, at.e_pt2 - e_pt20, at.total() - e_cas0 - e_pt20);
  return {};
}

} // namespace

// int-type-ok: argc's type is fixed by the standard's signature for main.
int main(int argc, char **argv) {
  const fb::Status status = run(std::span<char *const>(argv, static_cast<std::size_t>(argc)));
  if (!status.has_value()) {
    std::println(std::cerr, "error: {}", status.error().message);
    return 1;
  }
  return 0;
}
