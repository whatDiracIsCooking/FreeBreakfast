// GPU cases for nevpt2::qd_sc_hessian_product, the Hessian-vector product of
// E^corr[W] + Phi^SA in (h, B, {c_r}) at a fixed general symmetric W
// (off-diagonal, trace != 1, a negative entry), the root set moving by
// c_r(K) = e^K c_r (docs/nevpt2/nevpt2.tex Eqs. qd-K, qd-Lxx, qd-curv).
//
// - Against a ladder of qd_sc_adjoint + zc_adjoint (at fixed z^c) along one
//   path at a time: h and B straight (AO), the roots along C(t) = e^{t K} C
//   for K the generator of the path's CI part. h_bar, Lambda, X and A are
//   laddered as they come. Each root's y is laddered as the gradient of
//   E(e^K C) in K at t K, over raw CI directions, which the symmetric
//   splitting e^{t K/2} e^{e K_u} e^{t K/2} = e^{t K + e K_u + O(e t^2)}
//   gives exactly to first order in t: G(t) = K^*(sum_r y~_r c~_r^T), y~_r =
//   e^{-t K/2} y_r(t) and c~_r = e^{t K/2} c_r, K^* the adjoint of the
//   generator map at the point. G's derivative at 0 is the Hessian in K,
//   the curvature 1/2 {K_u, K_v} c_r against the raw y_r included. Central
//   differences at t = 1e-3 / 2^k, k < 10, their two-level Richardson
//   extrapolants, and per field the coarsest settled one (settled_rung.h):
//   every field (h_bar, Lambda, each root's y, X, A) within 1e-7 of its
//   largest element. The paths are where SC-QD's adjoint ladders (kQdSmooth in
//   nevpt2_gradient_gpu.cpp): at equal weights h8_sto3g_sa3 along h, B,
//   each root's c and each pure rotation theta_rs; at unequal weights
//   h2co_ccpvdz_sa2_weighted (0.7, 0.3) along each root's c and the
//   rotation, where Theta and gamma_SA's theta term live.
//   h2co_skew_ccpvdz_sa3 is not here: its QD adjoint repeats run to run only
//   to ~1e-9 (#1055), and along h, c_1 and c_2 it does not ladder.
// - Visibility, on the same path: the product's parts (the pair walk's
//   tangent, the one-slot-dotted terms, Phi^SA's, the root-set curvature's)
//   each above ten of the ladder's bars in some field, the curvature on the
//   paths with a CI part (it is zero without one); and the parts summing to
//   the product of a call without them.
//
// nevpt2::qd_pc_hessian_product, PC-QD's, the same way (#964):
// - The ladder of qd_pc_adjoint + zc_adjoint on h2co_skew_ccpvdz_sa3, at
//   the conditioning floor pc_floor = 1e-3, along h, B, each root's c, each
//   pure rotation and all of them at once (mixed), every field within 1e-7.
//   PC's CI directions are singlets, H_k c_r (singlets()), and its h a
//   hundredth of SC's (kPcHScale). At the default cut root 1 keeps Sijr, Srs
//   and Sir modes at 6e-8 to 2e-7 of sigma_max, and the solve's conditioning
//   puts a noise of eps / kept between the ladder's points, past what a
//   1e-7 ladder at t >= 2e-6 can resolve (../../src/nevpt2/gradient/
//   README.md, "The PC-QD Hessian-vector product in (h, B, {c_r})").
// - Visibility (pc_visibility): the r'^T A+ r cross terms, the d12 terms and
//   the rank-projector terms of A+ each above ten of the ladder's bars in
//   some field.
// - The Schur lift on every model state: on the A1/B2 fixture
//   (data/qd_nevpt2_hessian/h2nb_sto3g_sa2.json), over every pair of its
//   C2v vibrations as direction and partner, the product refuses exactly the
//   fixture's PC expected-refusal blocks, naming the coupling and t*, and
//   nothing in the totally symmetric block.
// - The sign gauge (#1203): on h2co_ccpvdz_sa2_weighted, root 0 negated
//   with its part of a seeded direction and W's row and column gives the
//   same lifts and the product but for root 0's y, negated; the same raw
//   direction at the negated roots lifts differently.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.hessian;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"
#include "settled_rung.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

// The issue's ladder bar, relative to each field's largest element.
constexpr double kLadder = 1e-7;
// The visibility rule: a part above this many bars in some field.
constexpr double kVisible = 10.0;
// The ladder's rungs, t = kT0 / 2^k for k < kRungs.
constexpr double kT0 = 1e-3;
constexpr std::int32_t kRungs = 10;
// The parts' sum against a call without them: the same terms in two
// sweeps, so they agree to the walk's and the einsums' rounding, far below
// the ladder's bar.
constexpr double kPartsTolerance = 1e-9;
// The CASSCF's bar: loose, since no stationarity is used.
constexpr double kConvTolGrad = 1e-6;
// B_dot a tenth the length of the others, as the ladders' directions.
constexpr double kBScale = 0.1;
// The Gram-Schmidt of the generator's span drops a vector this short.
constexpr double kSpanDrop = 1e-12;
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  if (count > 0) {
    EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                  wwr::wwrMemcpyDeviceToHost, res.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(res.sync().has_value());
  }
  return host;
}

std::vector<double> download(fb::DeviceResources &res, const fb::Buffers::device<double> &b) {
  return download(res, b.data(), b.num_elements());
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> out(host.size(), res.device());
  if (!host.empty()) {
    EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), host.data(), host.size_bytes(),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(res.sync().has_value());
  }
  return out;
}

double dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double max_abs(std::span<const double> v) {
  double out = 0.0;
  for (const double x : v) {
    out = std::max(out, std::abs(x));
  }
  return out;
}

const fb::json &qd_case(const std::string &name) {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2" / "qd_nevpt2.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name;
  static const fb::json kNone;
  return kNone;
}

// A case's SA-CASSCF point on the host: B as density_fit leaves it, C, the
// AO core Hamiltonian, the roots (root-major) and their weights.
struct Point {
  std::int32_t naux = 0;
  std::int32_t nao = 0;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> h;
  std::vector<double> vectors;
  std::vector<double> weights;
};

// This tree's DF-SA-CASSCF of qd_nevpt2.json's @p name through the real
// registry, as nevpt2_qd_hvp_gpu.cpp's, at kConvTolGrad.
Point sa_casscf(const std::string &name) {
  const fb::json &entry = qd_case(name);
  Point p{.weights = entry.at("weights").get<std::vector<double>>()};
  EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_qd_hessian_product");
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string weights;
  for (const double w : p.weights) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol_grad = {:g}\n",
      entry.at("basis").get<std::string>(), kDfAuxBasis, entry.at("ncas").get<std::int32_t>(),
      entry.at("nelecas").get<std::int32_t>(), p.weights.size(), weights, kConvTolGrad);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return p;
  }
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const fb::Status ran = calc->run(node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return p;
  }
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &eri = calc->get<fb::EriOutput>();
  const auto &h = calc->get<fb::CoreHamiltonianOutput>();
  EXPECT_TRUE(casscf.has_value() && eri.has_value() && h.has_value());
  if (!casscf.has_value() || !eri.has_value() || !h.has_value()) {
    return p;
  }
  fb::DeviceResources &res = node.device(0);
  const auto &df = std::get<fb::DensityFitted>(eri->eri);
  p.naux = df.naux;
  p.nao = static_cast<std::int32_t>(casscf->c.n());
  p.ncore = casscf->ncore;
  p.ncas = casscf->ncas;
  p.nelecas = casscf->nelecas;
  const auto nao = static_cast<std::size_t>(p.nao);
  p.b = download(res, df.b.data(), static_cast<std::size_t>(p.naux) * nao * nao);
  p.c = download(res, casscf->c.data(), nao * nao);
  p.h = download(res, h->h.data(), nao * nao);
  p.vectors = casscf->vectors;
  return p;
}

// sa_casscf once per case per process.
const Point &point(const std::string &name) {
  static std::map<std::string, Point> points;
  auto it = points.find(name);
  if (it == points.end()) {
    it = points.emplace(name, sa_casscf(name)).first;
  }
  return it->second;
}

// A general symmetric W: seeded, off-diagonal, trace != 1, with a negative
// entry, every one of which is checked (nevpt2_qd_hvp_gpu.cpp's).
std::vector<double> general_w(const std::size_t n) {
  std::mt19937_64 engine(9620);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> w(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      w[(i * n) + j] = w[(j * n) + i] = uniform(engine);
    }
  }
  double trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    trace += w[i * (n + 1)];
  }
  EXPECT_GT(std::abs(trace - 1.0), 0.1);
  EXPECT_LT(std::ranges::min(w), 0.0);
  EXPECT_GT(std::abs(w[1]), 0.1);
  return w;
}

// C^T x C over one nao x nao column-major x, C nao x nao column-major.
std::vector<double> to_mo(std::span<const double> c, std::span<const double> x,
                          const std::size_t n) {
  std::vector<double> half(n * n, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        half[mu + (n * q)] += x[mu + (n * nu)] * c[nu + (n * q)];
      }
    }
  }
  std::vector<double> out(n * n, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        out[p + (n * q)] += c[mu + (n * p)] * half[mu + (n * q)];
      }
    }
  }
  return out;
}

// One path: the AO h and B's tangents (h symmetric, B symmetric in its AO
// pair) and every root's raw c_dot, root-major.
struct Line {
  std::vector<double> h;
  std::vector<double> b;
  std::vector<double> c;
};

// Root r's CI direction on a "c_r" path, from @p ci when given (PC's
// singlets), else uniform over every determinant.
using CiDirection = std::function<std::vector<double>(std::size_t root)>;

// The seeded path @p path moves alone: "h", "B", "c_r" (root r's raw c_dot,
// uniform over every determinant, so with components along the other
// roots; @p ci's when given) or "rot_rs" (theta_sr = 1: c_dot_r = c_s,
// c_dot_s = -c_r); or "mixed", every one of them at once.
Line line_of(const Point &p, const std::string &path, const CiDirection &ci = {}) {
  if (path == "mixed") {
    // Every group at once: h, B, each root's raw c and two rotations.
    Line out = line_of(p, "h");
    for (const std::string part : {"B", "c_0", "c_1", "c_2", "rot_01", "rot_12"}) {
      if (part.starts_with("c_") || part.starts_with("rot_")) {
        const auto top = static_cast<std::size_t>(part.back() - '0');
        if (top >= p.weights.size()) {
          continue;
        }
      }
      const Line l = line_of(p, part, ci);
      for (auto [to, from] :
           {std::pair{&out.h, &l.h}, std::pair{&out.b, &l.b}, std::pair{&out.c, &l.c}}) {
        for (std::size_t k = 0; k < to->size(); ++k) {
          (*to)[k] += (*from)[k];
        }
      }
    }
    return out;
  }
  const auto nao = static_cast<std::size_t>(p.nao);
  const auto naux = static_cast<std::size_t>(p.naux);
  const std::size_t ndet = p.vectors.size() / p.weights.size();
  std::mt19937_64 engine(std::hash<std::string>{}(path) % 9631);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  Line l{.h = std::vector<double>(nao * nao, 0.0),
         .b = std::vector<double>(naux * nao * nao, 0.0),
         .c = std::vector<double>(p.vectors.size(), 0.0)};
  if (path == "h") {
    for (std::size_t mu = 0; mu < nao; ++mu) {
      for (std::size_t nu = 0; nu <= mu; ++nu) {
        l.h[mu + (nao * nu)] = l.h[nu + (nao * mu)] = uniform(engine);
      }
    }
  } else if (path == "B") {
    for (std::size_t mu = 0; mu < nao; ++mu) {
      for (std::size_t nu = 0; nu <= mu; ++nu) {
        for (std::size_t q = 0; q < naux; ++q) {
          l.b[q + (naux * (mu + (nao * nu)))] = l.b[q + (naux * (nu + (nao * mu)))] =
              kBScale * uniform(engine);
        }
      }
    }
  } else if (path.starts_with("rot_")) {
    const auto r = static_cast<std::size_t>(path[4] - '0');
    const auto s = static_cast<std::size_t>(path[5] - '0');
    for (std::size_t i = 0; i < ndet; ++i) {
      l.c[(r * ndet) + i] = p.vectors[(s * ndet) + i];
      l.c[(s * ndet) + i] = -p.vectors[(r * ndet) + i];
    }
  } else {
    const std::size_t root = std::stoul(path.substr(2));
    const std::vector<double> given = ci ? ci(root) : std::vector<double>{};
    for (std::size_t i = 0; i < ndet; ++i) {
      l.c[(root * ndet) + i] = ci ? given[i] : uniform(engine);
    }
  }
  return l;
}

// The length of a singlet direction.
constexpr double kSingletLength = 0.5;

// PC's CI directions: root r's c_dot = H_κ c_r, H_κ a seeded random
// spin-free Hamiltonian over the active orbitals (one- and two-body, with
// their symmetries), of length kSingletLength: a spin-free operator on a
// singlet is a singlet, as a CASSCF response is, and reaches every
// configuration a two-body excitation does. A uniform c_dot mixes in
// quintets, along which the metrics' spin-null modes rise (../../src/nevpt2/
// gradient/README.md, "The Schur lift"), which no physical direction does.
CiDirection singlets(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings,
                     const Point &p) {
  return [&res, &strings, &p](const std::size_t root) {
    const auto n = static_cast<std::size_t>(p.ncas);
    const std::size_t ndet = p.vectors.size() / p.weights.size();
    std::mt19937_64 engine(9640 + root);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<double> kappa(n * n);
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b = 0; b <= a; ++b) {
        kappa[a + (n * b)] = kappa[b + (n * a)] = uniform(engine);
      }
    }
    std::vector<double> eri(n * n * n * n, 0.0);
    const auto at = [n](const std::size_t t, const std::size_t u, const std::size_t v,
                        const std::size_t w) { return (((((t * n) + u) * n) + v) * n) + w; };
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t u = 0; u <= t; ++u) {
        for (std::size_t v = 0; v < n; ++v) {
          for (std::size_t w = 0; w <= v; ++w) {
            if ((t * n) + u < (v * n) + w) {
              continue;
            }
            const double x = uniform(engine);
            for (const auto [a, b, c, e] :
                 {std::array{t, u, v, w}, std::array{u, t, v, w}, std::array{t, u, w, v},
                  std::array{u, t, w, v}, std::array{v, w, t, u}, std::array{w, v, t, u},
                  std::array{v, w, u, t}, std::array{w, v, u, t}}) {
              eri[at(a, b, c, e)] = x;
            }
          }
        }
      }
    }
    auto sigma = fb::ci::Sigma::create(res, strings, strings);
    EXPECT_TRUE(sigma.has_value()) << sigma.error().message;
    if (!sigma.has_value()) {
      return std::vector<double>(ndet, 0.0);
    }
    EXPECT_TRUE(sigma->set_integrals(res, 0.0, kappa, eri).has_value());
    const std::vector<double> c(p.vectors.begin() + static_cast<std::ptrdiff_t>(root * ndet),
                                p.vectors.begin() + static_cast<std::ptrdiff_t>((root + 1) * ndet));
    const fb::Buffers::device<double> cd = upload(res, c);
    fb::Buffers::device<double> out(ndet, res.device());
    EXPECT_TRUE(sigma->apply(res, 1, cd.data(), out.data()).has_value());
    std::vector<double> kc = download(res, out.data(), ndet);
    const double length = std::sqrt(dot(kc, kc));
    for (double &x : kc) {
      x *= length > 0.0 ? kSingletLength / length : 0.0;
    }
    return kc;
  };
}

// The path's h and B tangents in the MO basis.
std::array<std::vector<double>, 2> mo_tangent(const Point &p, const Line &l) {
  const auto nao = static_cast<std::size_t>(p.nao);
  const auto naux = static_cast<std::size_t>(p.naux);
  std::vector<double> b(naux * nao * nao, 0.0);
  std::vector<double> slice(nao * nao);
  for (std::size_t q = 0; q < naux; ++q) {
    for (std::size_t mn = 0; mn < nao * nao; ++mn) {
      slice[mn] = l.b[q + (naux * mn)];
    }
    const std::vector<double> mo = to_mo(p.c, slice, nao);
    for (std::size_t pq = 0; pq < nao * nao; ++pq) {
      b[q + (naux * pq)] = mo[pq];
    }
  }
  return {to_mo(p.c, l.h, nao), std::move(b)};
}

// ---- The root set's generators, on the host ---------------------------------

// The roots C (n stacked unit vectors of ndet) and what is formed from
// them: written here in K's matrix form, independently of the call's.
struct Roots {
  std::size_t n = 0;
  std::size_t ndet = 0;
  std::vector<double> c;

  [[nodiscard]] std::span<const double> root(std::span<const double> v, std::size_t r) const {
    return v.subspan(r * ndet, ndet);
  }

  // The generator's part of a raw direction: (1 - P) d_r + C A e_r with
  // A_sr = 1/2 (c_s . d_r - c_r . d_s).
  [[nodiscard]] std::vector<double> project(std::span<const double> d) const {
    std::vector<double> out(d.begin(), d.end());
    for (std::size_t r = 0; r < n; ++r) {
      for (std::size_t s = 0; s < n; ++s) {
        const double along = dot(root(c, s), root(d, r));
        const double a_sr = 0.5 * (along - dot(root(c, r), root(d, s)));
        for (std::size_t e = 0; e < ndet; ++e) {
          out[(r * ndet) + e] += (a_sr - along) * c[(s * ndet) + e];
        }
      }
    }
    return out;
  }

  // K y, K = sum_r (xi_r c_r^T - c_r xi_r^T) + C A C^T over the projected
  // @p u: xi_r = (1 - P) u_r, A = C^T U.
  [[nodiscard]] std::vector<double> apply(std::span<const double> u,
                                          std::span<const double> y) const {
    std::vector<double> out(ndet, 0.0);
    std::vector<double> cy(n);
    for (std::size_t r = 0; r < n; ++r) {
      cy[r] = dot(root(c, r), y);
    }
    for (std::size_t r = 0; r < n; ++r) {
      std::vector<double> xi(root(u, r).begin(), root(u, r).end());
      for (std::size_t s = 0; s < n; ++s) {
        const double a_sr = dot(root(c, s), root(u, r));
        for (std::size_t e = 0; e < ndet; ++e) {
          xi[e] -= a_sr * c[(s * ndet) + e];
          out[e] += a_sr * cy[r] * c[(s * ndet) + e];
        }
      }
      const double xy = dot(xi, y);
      for (std::size_t e = 0; e < ndet; ++e) {
        out[e] += (xi[e] * cy[r]) - (c[(r * ndet) + e] * xy);
      }
    }
    return out;
  }

  // The gradient over raw directions t of <M, K(t)>_F for M = sum_i a_i b_i^T
  // (m stacked pairs), by <M, K(t)> = sum_r t_r . grad_r taken term by term
  // of K's matrix form.
  [[nodiscard]] std::vector<double> adjoint(std::span<const double> a,
                                            std::span<const double> b) const {
    const std::size_t m = a.size() / ndet;
    // M c_r, M^T c_r and Q = C^T M C.
    std::vector<double> out(n * ndet, 0.0);
    std::vector<double> q(n * n, 0.0);
    for (std::size_t s = 0; s < n; ++s) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t i = 0; i < m; ++i) {
          q[(s * n) + r] += dot(root(c, s), root(a, i)) * dot(root(b, i), root(c, r));
        }
      }
    }
    for (std::size_t r = 0; r < n; ++r) {
      std::vector<double> g(ndet, 0.0);
      for (std::size_t i = 0; i < m; ++i) {
        const double bc = dot(root(b, i), root(c, r));
        const double ac = dot(root(a, i), root(c, r));
        for (std::size_t e = 0; e < ndet; ++e) {
          g[e] += (a[(i * ndet) + e] * bc) - (b[(i * ndet) + e] * ac);
        }
      }
      // xi_r = (1 - P) t_r, so its gradient is (1 - P) g; A's is C's
      // antisymmetrized Q.
      std::vector<double> cg(n);
      for (std::size_t s = 0; s < n; ++s) {
        cg[s] = dot(root(c, s), g);
      }
      for (std::size_t e = 0; e < ndet; ++e) {
        double v = g[e];
        for (std::size_t s = 0; s < n; ++s) {
          v += c[(s * ndet) + e] * ((0.5 * (q[(s * n) + r] - q[(r * n) + s])) - cg[s]);
        }
        out[(r * ndet) + e] = v;
      }
    }
    return out;
  }

  // e^{s K} applied to every vector of @p v (stacked ndet), K over the
  // projected @p u: on an orthonormal basis E of span{c_r, u_r}, outside of
  // which K is zero, v + E (e^{s k} - 1) E^T v, k = E^T K E, the series
  // summed term by term.
  [[nodiscard]] std::vector<double> exp_apply(std::span<const double> u, const double s,
                                              std::span<const double> v) const {
    std::vector<std::vector<double>> basis;
    for (std::span<const double> from : {std::span<const double>(c), u}) {
      for (std::size_t r = 0; r < n; ++r) {
        std::vector<double> e(root(from, r).begin(), root(from, r).end());
        for (std::int32_t pass = 0; pass < 2; ++pass) {
          for (const std::vector<double> &f : basis) {
            const double ef = dot(e, f);
            for (std::size_t i = 0; i < ndet; ++i) {
              e[i] -= ef * f[i];
            }
          }
        }
        const double norm = std::sqrt(dot(e, e));
        if (norm > kSpanDrop) {
          for (double &x : e) {
            x /= norm;
          }
          basis.push_back(std::move(e));
        }
      }
    }
    const std::size_t d = basis.size();
    std::vector<double> k(d * d);
    for (std::size_t j = 0; j < d; ++j) {
      const std::vector<double> kj = apply(u, basis[j]);
      for (std::size_t i = 0; i < d; ++i) {
        k[(i * d) + j] = s * dot(basis[i], kj);
      }
    }
    // e^{s k} - 1, term by term.
    std::vector<double> term = k;
    std::vector<double> sum = k;
    for (std::int32_t order = 2; order < 40; ++order) {
      std::vector<double> next(d * d, 0.0);
      for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t j = 0; j < d; ++j) {
          for (std::size_t l = 0; l < d; ++l) {
            next[(i * d) + j] += term[(i * d) + l] * k[(l * d) + j] / order;
          }
        }
      }
      term = std::move(next);
      for (std::size_t e = 0; e < d * d; ++e) {
        sum[e] += term[e];
      }
    }
    std::vector<double> out(v.begin(), v.end());
    const std::size_t count = v.size() / ndet;
    for (std::size_t r = 0; r < count; ++r) {
      std::vector<double> coords(d);
      for (std::size_t i = 0; i < d; ++i) {
        coords[i] = dot(basis[i], root(v, r));
      }
      for (std::size_t i = 0; i < d; ++i) {
        double moved = 0.0;
        for (std::size_t j = 0; j < d; ++j) {
          moved += sum[(i * d) + j] * coords[j];
        }
        for (std::size_t e = 0; e < ndet; ++e) {
          out[(r * ndet) + e] += moved * basis[i][e];
        }
      }
    }
    return out;
  }
};

// ---- The ladder ------------------------------------------------------------

// The fields laddered, by name: h_bar, lambda, x, a and each root's y.
using Fields = std::map<std::string, std::vector<double>>;

Fields fields_of(fb::DeviceResources &res, const Point &p, const nevpt2::AdjointTangent &t) {
  const auto nmo = static_cast<std::size_t>(p.nao);
  const std::size_t ndet = p.vectors.size() / p.weights.size();
  Fields out{
      {"h_bar", download(res, t.h_bar.data(), nmo * nmo)},
      {"lambda", download(res, t.lambda.data(), static_cast<std::size_t>(p.naux) * nmo * nmo)},
      {"x", download(res, t.x.data(), nmo * nmo)},
      {"a", download(res, t.a.data(), nmo * nmo)}};
  const std::vector<double> y = download(res, t.y.data(), p.vectors.size());
  for (std::size_t r = 0; r < p.weights.size(); ++r) {
    out[std::format("y_{}", r)] =
        std::vector<double>(y.begin() + static_cast<std::ptrdiff_t>(r * ndet),
                            y.begin() + static_cast<std::ptrdiff_t>((r + 1) * ndet));
  }
  return out;
}

// Which functional a case ladders: SC-QD, or PC-QD at @p pc_floor with the
// kept span's rotation (@p span) or without it.
struct Method {
  bool pc = false;
  double pc_floor = 0.0;
  bool span = true;
};

// PC's h moves this fraction of SC's: along the raw h, a PC ket meets an
// intruder within 1e-3 of the point on h2co_skew_ccpvdz_sa3
// (../../src/nevpt2/energy/README.md, "The PC reverse pass", PC-QD
// conditioning), so the widest rung would leave the region the functional
// is defined in.
constexpr double kPcHScale = 1e-2;

Line pc_scaled(Line l, const Method &method) {
  if (method.pc) {
    for (double &x : l.h) {
      x *= kPcHScale;
    }
  }
  return l;
}

// The adjoint's fields at a point: qd_sc_adjoint's or qd_pc_adjoint's h_bar,
// Λ, raw y, X, A and ε.
struct Adjoint {
  fb::Buffers::device<double> h_bar, lambda, y, x, a, epsilon;
};

fb::Result<Adjoint> qd_adjoint(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings,
                               const Point &p, const double *b, const double *c, const double *h,
                               const double *vectors, std::span<const double> w,
                               const Method &method) {
  const auto take = [](auto got) -> fb::Result<Adjoint> {
    if (!got) {
      return std::unexpected(got.error());
    }
    return Adjoint{std::move(got->h_bar), std::move(got->lambda), std::move(got->y),
                   std::move(got->x),     std::move(got->a),      std::move(got->epsilon)};
  };
  if (method.pc) {
    return take(nevpt2::qd_pc_adjoint(res, strings, strings, p.naux, p.nao, p.nao, b, c, h, p.ncore,
                                      p.ncas, p.weights, vectors, w, 0, 0, method.pc_floor));
  }
  return take(nevpt2::qd_sc_adjoint(res, strings, strings, p.naux, p.nao, p.nao, b, c, h, p.ncore,
                                    p.ncas, p.weights, vectors, w, 0));
}

// qd_sc_adjoint (or qd_pc_adjoint) at @p w + zc_adjoint at @p z, at the
// point moved by @p t along @p l (h and B straight, the roots e^{t K} C),
// each root's y as G(t).
Fields moved(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings, const Point &p,
             const Roots &roots, const Line &l, std::span<const double> u, const double t,
             std::span<const double> w, std::span<const double> z, const Method &method) {
  std::vector<double> b = p.b;
  std::vector<double> h = p.h;
  for (std::size_t k = 0; k < b.size(); ++k) {
    b[k] += t * l.b[k];
  }
  for (std::size_t k = 0; k < h.size(); ++k) {
    h[k] += t * l.h[k];
  }
  const std::vector<double> vectors = roots.exp_apply(u, t, roots.c);
  const fb::Buffers::device<double> bd = upload(res, b);
  const fb::Buffers::device<double> cd = upload(res, p.c);
  const fb::Buffers::device<double> hd = upload(res, h);
  const fb::Buffers::device<double> vd = upload(res, vectors);
  const fb::Buffers::device<double> zd = upload(res, z);
  const auto corr =
      qd_adjoint(res, strings, p, bd.data(), cd.data(), hd.data(), vd.data(), w, method);
  const auto phi =
      nevpt2::zc_adjoint(res, strings, strings, p.naux, p.nao, p.nao, bd.data(), cd.data(),
                         hd.data(), p.ncore, p.ncas, p.weights, vd.data(), zd.data());
  EXPECT_TRUE(corr) << corr.error().message;
  EXPECT_TRUE(phi) << phi.error().message;
  if (!corr || !phi) {
    return {};
  }
  const auto sum = [&](const fb::Buffers::device<double> &x, const fb::Buffers::device<double> &y) {
    std::vector<double> out = download(res, x);
    const std::vector<double> other = download(res, y);
    for (std::size_t k = 0; k < out.size(); ++k) {
      out[k] += other[k];
    }
    return out;
  };
  Fields out{{"h_bar", sum(corr->h_bar, phi->h_bar)},
             {"lambda", sum(corr->lambda, phi->lambda)},
             {"x", sum(corr->x, phi->x)},
             {"a", sum(corr->a, phi->a)}};
  // G(t) = K^*(sum_r y~_r c~_r^T).
  const std::vector<double> y = sum(corr->y, phi->y);
  const std::vector<double> g =
      roots.adjoint(roots.exp_apply(u, -0.5 * t, y), roots.exp_apply(u, 0.5 * t, roots.c));
  for (std::size_t r = 0; r < roots.n; ++r) {
    const std::span<const double> gr = roots.root(g, r);
    out[std::format("y_{}", r)] = std::vector<double>(gr.begin(), gr.end());
  }
  return out;
}

// The product along one path against its ladder, and its parts' visibility
// and sum (SC's parts; PC's are pc_visibility's). Returns the ladder's bars
// per field, and the parts and the product's fields, for PC's visibility.
struct Laddered {
  std::map<std::string, double> bars;
  Fields product;
  std::vector<Fields> parts;
};

Laddered expect_product(const std::string &name, const std::string &path,
                        const Method &method = {}) {
  Laddered result;
  const Point &p = point(name);
  EXPECT_FALSE(p.b.empty());
  if (p.b.empty()) {
    return result;
  }
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  const std::size_t nroots = p.weights.size();
  const auto nmo = static_cast<std::size_t>(p.nao);
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  EXPECT_TRUE(strings) << strings.error().message;
  if (!strings) {
    return result;
  }
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);
  const std::vector<double> w = general_w(nroots);
  const Roots roots{.n = nroots, .ndet = p.vectors.size() / nroots, .c = p.vectors};

  // z^c over the adjoint's A and the SA Fock's epsilon at W.
  std::vector<double> z;
  {
    const auto adj =
        qd_adjoint(res, *strings, p, b.data(), c.data(), h.data(), vectors.data(), w, method);
    EXPECT_TRUE(adj) << adj.error().message;
    if (!adj) {
      return result;
    }
    const auto zc =
        nevpt2::zc_multipliers(res, p.nao, p.ncore, p.ncas, adj->a.data(), adj->epsilon.data());
    EXPECT_TRUE(zc) << zc.error().message;
    if (!zc) {
      return result;
    }
    z = download(res, zc->z.data(), nmo * nmo);
  }
  const fb::Buffers::device<double> z_d = upload(res, z);

  const Line l =
      pc_scaled(line_of(p, path, method.pc ? singlets(res, *strings, p) : CiDirection{}), method);
  const auto [h_mo, b_mo] = mo_tangent(p, l);
  const fb::Buffers::device<double> h_dot = upload(res, h_mo);
  const fb::Buffers::device<double> b_dot = upload(res, b_mo);
  const fb::Buffers::device<double> c_dot = upload(res, l.c);
  const nevpt2::MoDirections along{
      .count = 1, .h = h_dot.data(), .b = b_dot.data(), .c = c_dot.data()};
  // The product and its parts (walk, one-slot, Phi^SA, curvature), of
  // either functional.
  using Product = std::pair<nevpt2::AdjointTangent, std::vector<nevpt2::AdjointTangent>>;
  const auto product = [&](const bool parts, const bool span) -> fb::Result<Product> {
    const auto take = [](auto got) -> fb::Result<Product> {
      if (!got) {
        return std::unexpected(got.error());
      }
      std::vector<nevpt2::AdjointTangent> split;
      for (auto &q : got->parts) {
        for (auto &part : q) {
          split.push_back(std::move(part));
        }
      }
      return Product{std::move(got->products.front()), std::move(split)};
    };
    if (method.pc) {
      return take(nevpt2::qd_pc_hessian_product(res, *strings, *strings, p.naux, p.nao, p.nao,
                                                b.data(), c.data(), h.data(), p.ncore, p.ncas,
                                                p.weights, vectors.data(), w, z_d.data(), 0, along,
                                                {}, parts, method.pc_floor, span));
    }
    return take(nevpt2::qd_sc_hessian_product(
        res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(), h.data(), p.ncore,
        p.ncas, p.weights, vectors.data(), w, z_d.data(), 0, along, {}, parts));
  };
  const auto got = product(false, method.span);
  EXPECT_TRUE(got) << got.error().message;
  if (!got) {
    return result;
  }
  const Fields predicted = fields_of(res, p, got->first);

  // The ladder: D_k per field, and the coarsest settled two-level Richardson
  // extrapolant (settled_rung.h).
  const std::vector<double> u = roots.project(l.c);
  std::vector<Fields> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double step = std::ldexp(kT0, -k);
    const Fields plus = moved(res, *strings, p, roots, l, u, step, w, z, method);
    const Fields minus = moved(res, *strings, p, roots, l, u, -step, w, z, method);
    EXPECT_EQ(plus.size(), predicted.size());
    EXPECT_EQ(minus.size(), predicted.size());
    if (plus.size() != predicted.size() || minus.size() != predicted.size()) {
      return result;
    }
    Fields rung;
    for (const auto &[field, x] : plus) {
      const std::vector<double> &y = minus.at(field);
      std::vector<double> diff(x.size());
      for (std::size_t e = 0; e < x.size(); ++e) {
        diff[e] = (x[e] - y[e]) / (2.0 * step);
      }
      rung[field] = std::move(diff);
    }
    d.push_back(std::move(rung));
  }
  const auto richardson = [&](const std::size_t k, const std::string &f, const std::size_t e) {
    const auto r = [&](const std::size_t j) {
      return ((4.0 * d[j + 1].at(f)[e]) - d[j].at(f)[e]) / 3.0;
    };
    return ((16.0 * r(k + 1)) - r(k)) / 15.0;
  };
  std::map<std::string, double> &bars = result.bars;
  double worst = 0.0;
  std::string report;
  for (const auto &[field, x] : predicted) {
    const double scale = max_abs(x);
    EXPECT_GT(scale, 0.0) << name << " " << path << " " << field << ": the product is zero";
    EXPECT_EQ(d.front().at(field).size(), x.size()) << field;
    if (!(scale > 0.0) || d.front().at(field).size() != x.size()) {
      return result;
    }
    bars[field] = kLadder * scale;
    const auto change = [&](const std::size_t k) {
      double moved = 0.0;
      for (std::size_t e = 0; e < x.size(); ++e) {
        moved = std::max(moved, std::abs(richardson(k + 1, field, e) - richardson(k, field, e)));
      }
      return moved;
    };
    const std::size_t best =
        fb::test::settled_rung(static_cast<std::size_t>(kRungs) - 3, bars[field], change).rung;
    double miss = 0.0;
    for (std::size_t e = 0; e < x.size(); ++e) {
      miss = std::max(miss, std::abs(x[e] - richardson(best, field, e)));
    }
    EXPECT_LE(miss, bars[field]) << name << " " << path << " " << field;
    worst = std::max(worst, miss / scale);
    report += std::format(" {} {:.1e} (rung {}, |{:.2e}|)", field, miss / scale, best, scale);
  }
  std::cout << std::format("[ nevpt2   ] qd {} hessian product ladder {}{} {}: worst {:.2e};{}\n",
                           method.pc ? "pc" : "sc", name,
                           method.pc_floor > 0.0 ? std::format(" floor {:g}", method.pc_floor) : "",
                           path, worst, report);

  // The parts: each above ten bars in some field (the curvature where the
  // path has a CI part), and their sum the product. PC's visibility is its
  // own (pc_visibility); its parts are returned for it, and with the
  // rotation left out, the walk's share without it.
  const auto split = product(true, method.span);
  EXPECT_TRUE(split) << split.error().message;
  if (!split) {
    return result;
  }
  EXPECT_EQ(split->second.size(), 4U);
  const Fields total = fields_of(res, p, split->first);
  result.product = predicted;
  for (const nevpt2::AdjointTangent &part : split->second) {
    result.parts.push_back(fields_of(res, p, part));
  }
  if (method.pc) {
    const auto fixed = product(true, false);
    EXPECT_TRUE(fixed) << fixed.error().message;
    if (!fixed) {
      return result;
    }
    // The walk's share with the kept span held still.
    result.parts.push_back(fields_of(res, p, fixed->second.front()));
  }
  constexpr std::array<std::string_view, 4> kPartNames{"walk", "one-slot", "phi", "curvature"};
  const bool ci = max_abs(l.c) > 0.0;
  for (std::size_t q = 0; q < kPartNames.size() && !method.pc; ++q) {
    const Fields &part = result.parts[q];
    double visible = 0.0;
    std::string where;
    for (const auto &[field, x] : part) {
      if (const double m = max_abs(x) / bars.at(field); m > visible) {
        visible = m;
        where = field;
      }
    }
    if (kPartNames[q] != "curvature" || ci) {
      EXPECT_GT(visible, kVisible) << name << " " << path << " part " << kPartNames[q];
    } else {
      EXPECT_EQ(visible, 0.0) << name << " " << path << ": a curvature with no CI part";
    }
    std::cout << std::format("[ nevpt2   ] qd hessian product {} {} part {}: {:.3e} bars in {}\n",
                             name, path, kPartNames[q], visible, where);
  }
  double off = 0.0;
  for (const auto &[field, x] : predicted) {
    const std::vector<double> &y = total.at(field);
    double m = 0.0;
    for (std::size_t e = 0; e < x.size(); ++e) {
      m = std::max(m, std::abs(x[e] - y[e]));
    }
    EXPECT_LE(m, kPartsTolerance * max_abs(x)) << name << " " << path << " " << field;
    off = std::max(off, m / max_abs(x));
  }
  std::cout << std::format("[ nevpt2   ] qd hessian product {} {}: parts' sum off by {:.2e}\n",
                           name, path, off);
  return result;
}

class Nevpt2QdHessianProduct
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(Nevpt2QdHessianProduct, MatchesLadder) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("h8_sto3g_sa3_rot_01");
  const auto [name, path] = GetParam();
  if (name != "h8_sto3g_sa3" || path != "rot_01") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  expect_product(name, path);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, Nevpt2QdHessianProduct,
    ::testing::Values(std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"h"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"B"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"c_0"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"c_1"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"c_2"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"rot_01"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"rot_02"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"rot_12"}},
                      std::tuple{std::string{"h2co_ccpvdz_sa2_weighted"}, std::string{"c_0"}},
                      std::tuple{std::string{"h2co_ccpvdz_sa2_weighted"}, std::string{"c_1"}},
                      std::tuple{std::string{"h2co_ccpvdz_sa2_weighted"}, std::string{"rot_01"}}),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

TEST(Nevpt2QdHessianProductRefusal, RefusesBadInputs) {
  FBF_REQUIRE_GPU();
  const Point &p = point("h2co_ccpvdz_sa2_weighted");
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  const std::size_t nroots = p.weights.size();
  const auto nmo = static_cast<std::size_t>(p.nao);
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);
  const std::vector<double> w = general_w(nroots);
  const fb::Buffers::device<double> z = upload(res, std::vector<double>(nmo * nmo, 0.0));
  const Line l = line_of(p, "c_0");
  const auto [h_mo, b_mo] = mo_tangent(p, l);
  const fb::Buffers::device<double> h_dot = upload(res, h_mo);
  const fb::Buffers::device<double> b_dot = upload(res, b_mo);
  const fb::Buffers::device<double> c_dot = upload(res, l.c);
  const nevpt2::MoDirections one{
      .count = 1, .h = h_dot.data(), .b = b_dot.data(), .c = c_dot.data()};
  const auto refused = [&](const std::span<const double> ww, const double *vs, const double *zz,
                           const nevpt2::MoDirections &partners, const std::string_view what) {
    const auto got = nevpt2::qd_sc_hessian_product(res, *strings, *strings, p.naux, p.nao, p.nao,
                                                   b.data(), c.data(), h.data(), p.ncore, p.ncas,
                                                   p.weights, vs, ww, zz, 0, one, partners);
    ASSERT_FALSE(got) << what;
    EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(got.error().message.find("nevpt2::qd_sc_hessian_product"), std::string::npos)
        << got.error().message;
  };
  refused(w, vectors.data(), nullptr, {}, "a null z");
  refused(std::span(w).first(nroots), vectors.data(), z.data(), {}, "W not n^2");
  refused(w, vectors.data(), z.data(),
          {.count = 2, .h = h_dot.data(), .b = b_dot.data(), .c = c_dot.data()},
          "two partners for one direction");
  // Root 1 turned towards root 0: no longer orthogonal to it.
  std::vector<double> skewed = p.vectors;
  const std::size_t ndet = p.vectors.size() / nroots;
  for (std::size_t i = 0; i < ndet; ++i) {
    skewed[ndet + i] += 0.1 * p.vectors[i];
  }
  const fb::Buffers::device<double> skewed_d = upload(res, skewed);
  refused(w, skewed_d.data(), z.data(), {}, "roots not orthonormal");
}

// ---- PC-QD: qd_pc_hessian_product -----------------------------------------

// Zeroed bars for one direction over n states, pc_effective_hamiltonian_hvp's
// m = 1 layout, and what they and the transition hand-over read back to.
struct HvpBars {
  std::deque<fb::Buffers::device<double>> held;
  std::vector<nevpt2::EnergyBars> bars;
  Fields transition;
};

HvpBars hvp_bars(fb::DeviceResources &res, const nevpt2::MoBlocks &blocks, const std::size_t n) {
  const auto na = static_cast<std::size_t>(blocks.ncas);
  const auto nc = static_cast<std::size_t>(blocks.ncore);
  const auto nv = static_cast<std::size_t>(blocks.nvirt);
  const std::size_t nmo = nc + na + nv;
  HvpBars out;
  const auto zeros = [&](const std::size_t count) {
    out.held.push_back(upload(res, std::vector<double>(count, 0.0)));
    return out.held.back().data();
  };
  const nevpt2::EnergyBars shared{.h1e = zeros(na * na),
                                  .h2e = zeros(na * na * na * na),
                                  .h1e_v_sr = zeros(nv * na),
                                  .h1e_v_si = zeros(na * nc),
                                  .h1e_v_sir = zeros(nv * nc),
                                  .e_core = zeros(nc),
                                  .e_virt = zeros(nv),
                                  .lambda =
                                      zeros(static_cast<std::size_t>(blocks.naux) * nmo * nmo)};
  const std::size_t na6 = na * na * na * na * na * na;
  for (std::size_t j = 0; j < n; ++j) {
    nevpt2::EnergyBars b = shared;
    b.dm1 = zeros(na * na);
    b.dm2 = zeros(na * na * na * na);
    b.dm3 = zeros(na6);
    b.f3ca = zeros(na6);
    b.f3ac = zeros(na6);
    out.bars.push_back(b);
  }
  return out;
}

// pc_effective_hamiltonian_hvp over @p pairs along the one direction
// @p directions (n² entries) at @p w, every bar read back by name.
fb::Result<Fields> run_pc_hvp(fb::DeviceResources &res, const nevpt2::MoBlocks &blocks,
                              std::span<const nevpt2::EnergyInputs> pairs,
                              std::span<const double> w,
                              std::span<const nevpt2::EnergyDirection> directions) {
  const std::size_t n = static_cast<std::size_t>(std::llround(std::sqrt(pairs.size())));
  const auto na = static_cast<std::size_t>(blocks.ncas);
  const auto nc = static_cast<std::size_t>(blocks.ncore);
  const auto nv = static_cast<std::size_t>(blocks.nvirt);
  const std::size_t nmo = nc + na + nv;
  const std::size_t na6 = na * na * na * na * na * na;
  HvpBars b = hvp_bars(res, blocks, n);
  const auto got = nevpt2::pc_effective_hamiltonian_hvp(
      res, blocks, pairs, w, 0, directions, b.bars,
      [&](const std::int64_t, const std::int64_t bra, const std::int64_t ket,
          const nevpt2::EnergyBars &t) -> fb::Status {
        const std::string pair = std::format("({}, {})", bra, ket);
        for (const auto &[name, x, count] :
             {std::tuple{"dm1", t.dm1, na * na}, std::tuple{"dm2", t.dm2, na * na * na * na},
              std::tuple{"dm3", t.dm3, na6}}) {
          b.transition[std::format("{} {}", name, pair)] = download(res, x, count);
        }
        return {};
      });
  if (!got) {
    return std::unexpected(got.error());
  }
  Fields out = b.transition;
  const nevpt2::EnergyBars &s = b.bars.front();
  for (const auto &[name, x, count] :
       {std::tuple{"h1e", s.h1e, na * na}, std::tuple{"h2e", s.h2e, na * na * na * na},
        std::tuple{"e_core", s.e_core, nc}, std::tuple{"e_virt", s.e_virt, nv},
        std::tuple{"h1e_v_sr", s.h1e_v_sr, nv * na}, std::tuple{"h1e_v_si", s.h1e_v_si, na * nc},
        std::tuple{"h1e_v_sir", s.h1e_v_sir, nv * nc},
        std::tuple{"lambda", s.lambda, static_cast<std::size_t>(blocks.naux) * nmo * nmo}}) {
    out[name] = download(res, x, count);
  }
  for (std::size_t j = 0; j < n; ++j) {
    const nevpt2::EnergyBars &own = b.bars[j];
    const std::string pair = std::format("({}, {})", j, j);
    for (const auto &[name, x, count] :
         {std::tuple{"dm1", own.dm1, na * na}, std::tuple{"dm2", own.dm2, na * na * na * na},
          std::tuple{"dm3", own.dm3, na6}, std::tuple{"f3ca", own.f3ca, na6},
          std::tuple{"f3ac", own.f3ac, na6}}) {
      out[std::format("{} {}", name, pair)] = download(res, x, count);
    }
  }
  return out;
}

// The largest of @p part's fields over @p whole's ladder bar, kLadder of the
// field's largest element, and the field; fields @p skip names left out.
std::pair<double, std::string> bars_of(const Fields &part, const Fields &whole,
                                       std::span<const std::string_view> skip = {}) {
  double visible = 0.0;
  std::string where;
  for (const auto &[field, x] : part) {
    if (std::ranges::find(skip, field) != skip.end() || !whole.contains(field)) {
      continue;
    }
    const double scale = max_abs(whole.at(field));
    if (scale > 0.0 && max_abs(x) / (kLadder * scale) > visible) {
      visible = max_abs(x) / (kLadder * scale);
      where = field;
    }
  }
  return {visible, where};
}

// PC-QD's three kinds of second-order term in Eq. qd-pcfinish, each above
// ten of the product's ladder bars in some field, along the path's QD
// builders' tangent:
// - the r'ᵀ A⁺ r cross terms: pc_effective_hamiltonian_hvp along the
//   tangent's orbital energies alone, which move Δ (A by Δ̇ S, so r = -Δ̇ S x)
//   and nothing a fixed-multiplier difference or U̇ reads, against the whole
//   tangent's bars; Sijrs's share (SC's, on Λ and the orbital energies) left
//   out;
// - the ∂₁₂ terms: the product's one-slot part, the inputs' second
//   derivatives (∂₁₂ b′, ∂₁₂ b, ∂₁₂ A) against the fixed bars;
// - the rank-projector terms of A⁺ (with b′): the product's walk part with
//   the kept span's rotation against it without, at the floor, where real
//   modes are dropped (along h or B, S_N does not move: they are in y, the
//   CI–orbital block).
void pc_visibility(const std::string &name, const std::string &path, const Method &method,
                   const Laddered &got) {
  ASSERT_EQ(got.parts.size(), 5U) << name << " " << path;
  const Point &p = point(name);
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  const std::size_t nroots = p.weights.size();
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const Line l = pc_scaled(line_of(p, path, singlets(res, *strings, p)), method);
  const Roots roots{.n = nroots, .ndet = p.vectors.size() / nroots, .c = p.vectors};
  const auto [h_mo, b_mo] = mo_tangent(p, l);
  const fb::Buffers::device<double> h_dot = upload(res, h_mo);
  const fb::Buffers::device<double> b_dot = upload(res, b_mo);
  // The CI part along the generators, as the product reads it.
  const fb::Buffers::device<double> c_dot = upload(res, roots.project(l.c));
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);
  auto t = nevpt2::qd_builders_tangent(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(),
                                       c.data(), h.data(), p.ncore, p.ncas, p.weights,
                                       vectors.data(), 1, h_dot.data(), b_dot.data(), c_dot.data());
  ASSERT_TRUE(t) << t.error().message;
  const nevpt2::MoBlocks &blocks = t->value.shared.blocks;
  std::vector<nevpt2::EnergyInputs> pairs = t->value.inputs();
  for (nevpt2::EnergyInputs &in : pairs) {
    in.pc_floor = method.pc_floor;
  }
  std::vector<nevpt2::EnergyDirection> whole;
  std::vector<nevpt2::EnergyDirection> epsilon;
  const nevpt2::BuilderOutputs &ts = t->tangents.front().shared;
  for (const nevpt2::EnergyInputs &in : t->tangents.front().inputs()) {
    whole.push_back({.in = in, .blocks = &ts.blocks});
    epsilon.push_back({.in = {.e_core = ts.e_core.data(), .e_virt = ts.e_virt.data()}});
  }
  const std::vector<double> w = general_w(nroots);
  const auto all = run_pc_hvp(res, blocks, pairs, w, whole);
  ASSERT_TRUE(all) << all.error().message;
  const auto cross = run_pc_hvp(res, blocks, pairs, w, epsilon);
  ASSERT_TRUE(cross) << cross.error().message;
  constexpr std::array<std::string_view, 3> kSijrs{"lambda", "e_core", "e_virt"};
  const auto [cross_bars, cross_at] = bars_of(*cross, *all, kSijrs);
  EXPECT_GT(cross_bars, kVisible) << name << " " << path << ": the r'A+r cross terms";
  const auto [slot_bars, slot_at] = bars_of(got.parts[1], got.product);
  EXPECT_GT(slot_bars, kVisible) << name << " " << path << ": the d12 terms";
  Fields rotation = got.parts[0];
  for (auto &[field, x] : rotation) {
    const std::vector<double> &held = got.parts[4].at(field);
    for (std::size_t e = 0; e < x.size(); ++e) {
      x[e] -= held[e];
    }
  }
  const auto [span_bars, span_at] = bars_of(rotation, got.product);
  EXPECT_GT(span_bars, kVisible) << name << " " << path << ": the rank-projector terms";
  std::cout << std::format("[ nevpt2   ] qd pc hessian product {}{} {}: cross terms {:.2e} bars "
                           "({}), d12 terms {:.2e} bars ({}), rank-projector terms {:.2e} bars "
                           "({})\n",
                           name,
                           method.pc_floor > 0.0 ? std::format(" floor {:g}", method.pc_floor) : "",
                           path, cross_bars, cross_at, slot_bars, slot_at, span_bars, span_at);
}

// The skew case's PC-QD paths, at the conditioning floor pc_floor = 1e-3
// (README "The PC-QD Hessian-vector product in (h, B, {c_r})", why not at
// the default cut): every group alone, every pure rotation, and all of them
// at once, each with pc_visibility's three kinds of term.
class Nevpt2QdPcHessianProduct
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

// The conditioning floor the skew ladders run at.
constexpr double kSkewFloor = 1e-3;

TEST_P(Nevpt2QdPcHessianProduct, MatchesLadder) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("h2co_skew_ccpvdz_sa3_rot_01");
  const auto [name, path] = GetParam();
  if (path != "rot_01") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const Method method{.pc = true, .pc_floor = kSkewFloor, .span = true};
  const Laddered got = expect_product(name, path, method);
  if (HasFailure()) {
    return;
  }
  pc_visibility(name, path, method, got);
}

INSTANTIATE_TEST_SUITE_P(
    Cases, Nevpt2QdPcHessianProduct,
    ::testing::Values(std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"h"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"B"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"c_0"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"c_1"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"c_2"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"rot_01"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"rot_02"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"rot_12"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"mixed"}}),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

// The A1/B2 fixture (data/qd_nevpt2_hessian/h2nb_sto3g_sa2.json, #1058):
// its tree's DF-SA-CASSCF converged tightly, t_x for every Cartesian
// coordinate (hessian::df_qd_nevpt2_tangents), combined into the fixture's
// C2v vibrations; then per pair of vibrations (u, v) qd_pc_hessian_product
// along u with v as partner. A refusal names its coupling, the directions
// and t*, which are the block; the refused blocks must be the fixture's PC
// expected-refusal ones, (0, 1) x the b2 pairs, and nothing in the totally
// symmetric block.
TEST(Nevpt2QdPcHessianProductLift, A1B2FixtureRefusesTheExpectedBlocks) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  FBF_FAMILY_EXEMPT("its one case, the A1/B2 fixture's tangent sweep, is past slow_test.h's rule");
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2_hessian" /
                            "h2nb_sto3g_sa2.json");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  const fb::json &doc = *read;
  const std::vector<double> weights = doc.at("weights").get<std::vector<double>>();
  const auto ncas = doc.at("ncas").get<std::int32_t>();
  const auto nelecas = doc.at("nelecas").get<std::int32_t>();
  const auto labels = doc.at("coordinates").at("labels").get<std::vector<std::string>>();
  const auto irreps = doc.at("coordinates").at("irreps").get<std::vector<std::string>>();
  const auto vectors = doc.at("coordinates").at("vectors").get<std::vector<std::vector<double>>>();
  std::set<std::pair<std::string, std::string>> expected;
  for (const fb::json &block : doc.at("expected_refusals").at("pc")) {
    const auto c = block.at("coordinates").get<std::vector<std::string>>();
    expected.emplace(std::min(c[0], c[1]), std::max(c[0], c[1]));
  }
  ASSERT_FALSE(expected.empty());

  const fb::test::ScratchDir dir("nevpt2_qd_pc_hessian_product_a1b2");
  std::string xyz = std::format("{}\nh2nb\n", doc.at("atoms").size());
  for (const fb::json &atom : doc.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  ASSERT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string w_list;
  for (const double x : weights) {
    w_list += std::format("{}{:.17g}", w_list.empty() ? "" : ", ", x);
  }
  ASSERT_TRUE(fb::write_file(dir / "run.toml",
                             std::format("[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis "
                                         "= \"{}\"\n[eri]\nmode = \"density_fitting\"\naux_basis "
                                         "= \"{}\"\n[casscf]\nncas = {}\nnelecas = {}\nnroots = "
                                         "{}\nweights = [{}]\nconv_tol = 1e-12\nconv_tol_grad = "
                                         "1e-10\n",
                                         doc.at("basis").at("name").get<std::string>(),
                                         doc.at("aux_basis").at("name").get<std::string>(), ncas,
                                         nelecas, weights.size(), w_list))
                  .has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  fb::DeviceResources &res = node.device(0);
  const auto &cas = calc->get<fb::CasscfOutput>();
  ASSERT_TRUE(cas.has_value());
  const fb::Molecule &molecule = calc->get<fb::MoleculeOutput>()->molecule;
  const fb::BasisSet &basis = calc->get<fb::BasisSetOutput>()->basis;
  const std::span<const fb::PrimitivePairClass> classes =
      calc->get<fb::PrimitivePairClassesOutput>()->classes;
  const auto &fit = std::get<fb::DensityFitted>(calc->get<fb::EriOutput>()->eri);
  const fb::SphericalMatrixAO &h = calc->get<fb::CoreHamiltonianOutput>()->h;
  const std::size_t nmo = basis.nao_sphe();
  const std::size_t nn = nmo * nmo;
  const std::size_t naux = fit.aux.nao_sphe();
  const std::size_t nroots = weights.size();
  const std::size_t ndet = cas->vectors.size() / nroots;
  // The Solution as the [hessian] step builds it.
  const auto na2 = static_cast<std::size_t>(cas->ncas) * static_cast<std::size_t>(cas->ncas);
  std::vector<double> gamma(na2, 0.0);
  for (std::size_t r = 0; r < nroots; ++r) {
    for (std::size_t i = 0; i < na2; ++i) {
      gamma[i] += cas->weights[r] * cas->rdm1[(r * na2) + i];
    }
  }
  const fb::Buffers::device<double> roots = upload(res, cas->vectors);
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations,
      .weights = cas->weights};
  for (const auto &[to, from, count] :
       {std::tuple{solution.roots.vectors.data(), roots.data(), cas->vectors.size()},
        std::tuple{solution.c.data(), cas->c.data(), cas->c.num_elements()}}) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(to, from, count * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                  res.stream()),
              wwr::wwrSuccess);
  }
  ASSERT_TRUE(res.sync().has_value());
  const auto cp =
      fb::hessian::df_sacasscf_cp(res, molecule, basis, classes, fit.aux, h, fit.b, fit.l.data(),
                                  solution, weights, cas->ncore, cas->ncas);
  ASSERT_TRUE(cp) << cp.error().message;
  const std::size_t n_x = 3 * basis.natoms();
  std::vector<double> h_x(n_x * nn);
  std::vector<double> b_x(n_x * naux * nn);
  std::vector<double> c_x(n_x * nroots * ndet);
  const auto tx = fb::hessian::df_qd_nevpt2_tangents(
      res, molecule, basis, classes, fit.aux, h, fit.b, fit.l.data(), solution, weights, cas->ncore,
      cas->ncas, cp->cp, false,
      [&](const std::int32_t x0, const std::int32_t count, const double *const h_dot,
          const double *const b_dot, const double *const c_dot, const double *) -> fb::Status {
        const auto x = static_cast<std::size_t>(x0);
        const auto k = static_cast<std::size_t>(count);
        for (const auto &[to, from, size] :
             {std::tuple{&h_x, h_dot, nn}, std::tuple{&b_x, b_dot, naux * nn},
              std::tuple{&c_x, c_dot, nroots * ndet}}) {
          const std::vector<double> host = download(res, from, k * size);
          std::ranges::copy(host, to->begin() + static_cast<std::ptrdiff_t>(x * size));
        }
        return {};
      });
  ASSERT_TRUE(tx) << tx.error().message;

  // The vibrations, each Σ_x q_jx t_x.
  std::vector<std::size_t> vib;
  for (std::size_t j = 0; j < labels.size(); ++j) {
    if (!labels[j].starts_with("T_") && !labels[j].starts_with("R_")) {
      vib.push_back(j);
    }
  }
  std::vector<double> h_q(vib.size() * nn, 0.0);
  std::vector<double> b_q(vib.size() * naux * nn, 0.0);
  std::vector<double> c_q(vib.size() * nroots * ndet, 0.0);
  for (std::size_t q = 0; q < vib.size(); ++q) {
    ASSERT_EQ(vectors[vib[q]].size(), n_x);
    for (std::size_t x = 0; x < n_x; ++x) {
      const double f = vectors[vib[q]][x];
      for (const auto &[to, from, size] :
           {std::tuple{&h_q, &h_x, nn}, std::tuple{&b_q, &b_x, naux * nn},
            std::tuple{&c_q, &c_x, nroots * ndet}}) {
        for (std::size_t e = 0; e < size; ++e) {
          (*to)[(q * size) + e] += f * (*from)[(x * size) + e];
        }
      }
    }
  }
  auto strings = fb::ci::DeviceStringSet::create(res, cas->ncas, cas->nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> h_dot = upload(res, h_q);
  const fb::Buffers::device<double> b_dot = upload(res, b_q);
  const fb::Buffers::device<double> c_dot = upload(res, c_q);
  const std::vector<double> w = general_w(nroots);
  const auto nmo32 = static_cast<std::int32_t>(nmo);
  const auto naux32 = static_cast<std::int32_t>(naux);
  const auto adj = nevpt2::qd_pc_adjoint(res, *strings, *strings, naux32, nmo32, nmo32,
                                         fit.b.data(), solution.c.data(), h.data(), cas->ncore,
                                         cas->ncas, weights, roots.data(), w, 0);
  ASSERT_TRUE(adj) << adj.error().message;
  const auto zc =
      nevpt2::zc_multipliers(res, nmo32, cas->ncore, cas->ncas, adj->a.data(), adj->epsilon.data());
  ASSERT_TRUE(zc) << zc.error().message;
  const auto at = [&](const std::size_t q) {
    return nevpt2::MoDirections{.count = 1,
                                .h = h_dot.data() + (q * nn),
                                .b = b_dot.data() + (q * naux * nn),
                                .c = c_dot.data() + (q * nroots * ndet)};
  };
  std::set<std::pair<std::string, std::string>> refused;
  for (std::size_t a = 0; a < vib.size(); ++a) {
    for (std::size_t bq = a; bq < vib.size(); ++bq) {
      const std::string &u = labels[vib[a]];
      const std::string &v = labels[vib[bq]];
      const auto got = nevpt2::qd_pc_hessian_product(
          res, *strings, *strings, naux32, nmo32, nmo32, fit.b.data(), solution.c.data(), h.data(),
          cas->ncore, cas->ncas, weights, roots.data(), w, zc->z.data(), 0, at(a), at(bq));
      if (!got) {
        EXPECT_EQ(got.error().kind, fb::ErrorKind::Numerical) << got.error().message;
        EXPECT_NE(got.error().message.find("t* "), std::string::npos) << got.error().message;
        EXPECT_NE(got.error().message.find("coupling ("), std::string::npos) << got.error().message;
        refused.emplace(std::min(u, v), std::max(u, v));
        std::cout << std::format("[ nevpt2   ] qd pc product A1/B2 ({}, {}): refused: {}\n", u, v,
                                 got.error().message);
        continue;
      }
      // Accepted: what lifted along u, per ket and class, and its overlaps.
      std::string lifted;
      for (std::size_t ket = 0; ket < nroots; ++ket) {
        for (std::size_t cls = 0; cls < 8; ++cls) {
          const double l = got->lifts.front()[ket][cls];
          double o = 0.0;
          for (std::size_t bra = 0; bra < nroots; ++bra) {
            o = std::max(o, got->overlaps.front()[ket][bra][cls]);
          }
          if (l > 0.0 || o > 0.0) {
            lifted += std::format(" ket {} class {} lift {:.2e} t* {:.2e} overlap {:.2e};", ket,
                                  cls, l, got->lift_steps.front()[ket][cls], o);
          }
        }
      }
      std::cout << std::format("[ nevpt2   ] qd pc product A1/B2 ({}, {}): accepted;{}\n", u, v,
                               lifted);
    }
  }
  EXPECT_EQ(refused, expected);
  // Nothing in the totally symmetric block.
  for (const auto &[u, v] : refused) {
    const auto irrep = [&](const std::string &label) {
      return irreps[static_cast<std::size_t>(std::ranges::find(labels, label) - labels.begin())];
    };
    EXPECT_FALSE(irrep(u) == "A1" && irrep(v) == "A1") << u << ", " << v;
  }
}

// The Schur lifts and the product are functions of a direction written in
// the point's sign gauge, which the SA-CASSCF fixes run to run (#1203): a
// seeded raw c_dot is a different direction once a root (or an active MO)
// comes back with the other sign, and its lifts differ by a few percent
// (0.338 against 0.360 on the footprint test's water), while one process
// repeats them to the walk's rounding. So the check is covariance: root 0
// negated together with its part of the direction and W's row and column 0
// off the diagonal gives the same lifts, and the same product but for root
// 0's y, negated. The bar is the walk's rounding, ~3e-12 of each field's
// largest element measured, with the margin of #1203's other repeats.
constexpr double kGaugeTolerance = 1e-10;
// What the raw c_dot at the flipped roots must move a lift by, of the
// largest: the gauge's own effect, far past the tolerance.
constexpr double kGaugeMoved = 1e-6;

TEST(Nevpt2QdPcHessianProductGauge, FollowsTheRootSigns) {
  FBF_REQUIRE_GPU();
  const Point &p = point("h2co_ccpvdz_sa2_weighted");
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  const std::size_t nroots = p.weights.size();
  const std::size_t ndet = p.vectors.size() / nroots;
  const auto nmo = static_cast<std::size_t>(p.nao);
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  // The direction: every root's raw c_dot, seeded; h and B still, and a
  // partner with no CI part, so no ket lifts on both sides.
  std::vector<double> c_dot(p.vectors.size());
  std::mt19937_64 engine(1203);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (double &x : c_dot) {
    x = uniform(engine);
  }
  const std::vector<double> zeros(p.vectors.size(), 0.0);
  const fb::Buffers::device<double> h_dot = upload(res, std::vector<double>(nmo * nmo, 0.0));
  const fb::Buffers::device<double> b_dot =
      upload(res, std::vector<double>(static_cast<std::size_t>(p.naux) * nmo * nmo, 0.0));
  const fb::Buffers::device<double> still = upload(res, zeros);
  // Root 0 negated: in the roots, the direction and W.
  const auto flip = [&](std::vector<double> v) {
    for (std::size_t e = 0; e < ndet; ++e) {
      v[e] = -v[e];
    }
    return v;
  };
  std::vector<double> w_flip = general_w(nroots);
  for (std::size_t s = 1; s < nroots; ++s) {
    w_flip[s] = -w_flip[s];
    w_flip[s * nroots] = -w_flip[s * nroots];
  }
  struct Got {
    std::vector<std::array<double, 8>> lifts;
    Fields product;
  };
  const auto run = [&](const std::vector<double> &roots, const std::vector<double> &dir,
                       const std::vector<double> &w) -> Got {
    const fb::Buffers::device<double> vectors = upload(res, roots);
    const fb::Buffers::device<double> c_d = upload(res, dir);
    const auto adj = qd_adjoint(res, *strings, p, b.data(), c.data(), h.data(), vectors.data(), w,
                                Method{.pc = true});
    EXPECT_TRUE(adj) << adj.error().message;
    if (!adj) {
      return {};
    }
    const auto zc =
        nevpt2::zc_multipliers(res, p.nao, p.ncore, p.ncas, adj->a.data(), adj->epsilon.data());
    EXPECT_TRUE(zc) << zc.error().message;
    if (!zc) {
      return {};
    }
    const nevpt2::MoDirections along{
        .count = 1, .h = h_dot.data(), .b = b_dot.data(), .c = c_d.data()};
    const nevpt2::MoDirections partner{
        .count = 1, .h = h_dot.data(), .b = b_dot.data(), .c = still.data()};
    const auto got = nevpt2::qd_pc_hessian_product(
        res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(), h.data(), p.ncore,
        p.ncas, p.weights, vectors.data(), w, zc->z.data(), 0, along, partner);
    EXPECT_TRUE(got) << got.error().message;
    if (!got) {
      return {};
    }
    return {.lifts = got->lifts.front(), .product = fields_of(res, p, got->products.front())};
  };
  const Got at = run(p.vectors, c_dot, general_w(nroots));
  const Got flipped = run(flip(p.vectors), flip(c_dot), w_flip);
  // And the same raw c_dot at the flipped roots: a different direction.
  const Got other = run(flip(p.vectors), c_dot, w_flip);
  ASSERT_FALSE(HasFailure());
  ASSERT_EQ(at.lifts.size(), nroots);
  ASSERT_EQ(flipped.lifts.size(), nroots);
  double most = 0.0;
  for (const std::array<double, 8> &ket : at.lifts) {
    most = std::max(most, std::ranges::max(ket));
  }
  ASSERT_GT(most, 0.0) << "no Schur lift along the direction";
  for (std::size_t ket = 0; ket < nroots; ++ket) {
    for (std::size_t k = 0; k < 8; ++k) {
      EXPECT_NEAR(flipped.lifts[ket][k], at.lifts[ket][k], kGaugeTolerance * most)
          << "ket " << ket << " class " << k;
    }
  }
  double moved = 0.0;
  for (std::size_t ket = 0; ket < nroots; ++ket) {
    for (std::size_t k = 0; k < 8; ++k) {
      moved = std::max(moved, std::abs(other.lifts[ket][k] - at.lifts[ket][k]));
    }
  }
  EXPECT_GT(moved, kGaugeMoved * most) << "the raw c_dot at the flipped roots lifts the same";
  for (const auto &[name, field] : at.product) {
    const std::vector<double> &other = flipped.product.at(name);
    const double sign = name == "y_0" ? -1.0 : 1.0;
    const double largest = max_abs(field);
    double off = 0.0;
    for (std::size_t e = 0; e < field.size(); ++e) {
      off = std::max(off, std::abs(other[e] - (sign * field[e])));
    }
    EXPECT_LE(off, kGaugeTolerance * largest) << name;
    std::cout << std::format("[ nevpt2   ] qd pc product root-sign gauge {}: {:.2e} of {:.3e}\n",
                             name, off / largest, largest);
  }
  std::cout << std::format("[ nevpt2   ] qd pc product root-sign gauge: largest lift {:.6e}, "
                           "moved {:.2e} of it at the raw c_dot\n",
                           most, moved / most);
}

} // namespace
