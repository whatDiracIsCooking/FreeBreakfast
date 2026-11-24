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
///     free_breakfast_oo_nevpt2 <config.toml> [iterations = 8] [oo|no]
///
/// With `no`, the natural orbitals of the PT2-corrected density instead,
/// iterated: no functional is minimized (natural() below).
///
/// The config needs [casscf] of one singlet root and density-fitted [eri].
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
fb::Result<Point> evaluate(const Problem &p, const double *c_in,
                           const std::span<const double> start) {
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
  const fb::nevpt2::PcClassEnergies pt2 = FBF_TRY(
      fb::nevpt2::pc_energy(res, p.alpha, p.beta, p.naux, p.nmo, p.nmo, p.b.data(), c.data(),
                            p.h.data(), p.ncore, p.ncas, roots.vectors.data(), 0));
  const double e_cas = fock.energy;
  return Point{.c = std::move(c),
               .roots = std::move(roots),
               .gamma = std::move(gamma),
               .gamma2 = std::move(gamma2),
               .ints = std::move(ints),
               .fock = std::move(fock),
               .e_cas = e_cas,
               .e_pt2 = pt2.total};
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

// One natural-orbital pass from @p at: E at the natural orbitals of the
// PT2-corrected density D = D_CAS + dE_PT2/dh (pc_adjoint's h_bar plus the
// canonical condition's z^c), the unrelaxed response density. The new active
// orbitals are the ncas natural orbitals with the most weight on the old
// active space, the new core the ncore of the rest with the most on the old
// core: by overlap, not by occupation, so a correlated core orbital whose
// occupation falls among the active ones does not swap in. The CI starts
// cold, since the active orbitals have changed.
fb::Result<Point> natural(const Problem &p, const Point &at) {
  fb::DeviceResources &res = p.res;
  const auto nmo = static_cast<std::size_t>(p.nmo);
  const auto ncore = static_cast<std::size_t>(p.ncore);
  const auto n = static_cast<std::size_t>(p.ncas);
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
      d[((ncore + t) * nmo) + ncore + u] += at.gamma[(t * n) + u];
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
  return evaluate(p, rotated.data(), {});
}

fb::Status run(const std::span<char *const> args) {
  if (args.size() < 2 || args.size() > 4 ||
      (args.size() == 4 && std::string_view(args[3]) != "no" &&
       std::string_view(args[3]) != "oo")) {
    return fb::err_config("usage: {} <config.toml> [iterations] [oo|no]",
                          args.empty() ? "free_breakfast_oo_nevpt2" : args[0]);
  }
  const std::filesystem::path config = args[1];
  const std::int32_t iterations = args.size() >= 3 ? std::atoi(args[2]) : 8;
  const bool natural_orbitals = args.size() == 4 && std::string_view(args[3]) == "no";
  const std::filesystem::path out =
      std::filesystem::temp_directory_path() / "free_breakfast_oo_nevpt2";
  fb::Calculation calc = FBF_TRY(fb::read_config(config, config.parent_path(), out));
  fb::NodeResources node{{.workspaces = {.device = {.workspace = std::size_t{256} << 20},
                                         .pinned = {.workspace = std::size_t{4} << 20}}}};
  FBF_TRY(calc.run(node));
  const auto &casscf = calc.get<fb::CasscfOutput>();
  const auto &eri = calc.get<fb::EriOutput>();
  if (!casscf.has_value() || !eri.has_value() ||
      !std::holds_alternative<fb::DensityFitted>(eri->eri) || casscf->energies.size() != 1 ||
      casscf->nelecas % 2 != 0) {
    return fb::err_config("{}: needs [casscf] of one singlet root over density-fitted [eri]",
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
                  .b = df.b,
                  .naux = df.naux,
                  .nmo = static_cast<std::int32_t>(casscf->c.n()),
                  .ncore = casscf->ncore,
                  .ncas = casscf->ncas,
                  .alpha = alpha,
                  .beta = beta,
                  .sigma = sigma,
                  .rdm = rdm};

  fb::Buffers::device<double> start(casscf->vectors.size(), res.device());
  FBF_TRY(copy(res, start.data(), casscf->vectors.data(), casscf->vectors.size(),
               wwr::wwrMemcpyHostToDevice));
  Point at = FBF_TRY(evaluate(p, casscf->c.data(), {start.data(), casscf->vectors.size()}));
  const double e_cas0 = at.e_cas;
  const double e_pt20 = at.e_pt2;
  std::println("CASSCF({},{}) nmo {} ncore {}: pipeline E_CAS {:.10f}", casscf->nelecas,
               casscf->ncas, p.nmo, p.ncore, casscf->energy);
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
