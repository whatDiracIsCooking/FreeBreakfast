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
//   extrapolants, and per field the one that moves least to the next: every
//   field (h_bar, Lambda, each root's y, X, A) within 1e-7 of its largest
//   element. The paths are where SC-QD's adjoint ladders (kQdSmooth in
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

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.ci;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

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
  fb::NodeResources node;
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

// The seeded path @p path moves alone: "h", "B", "c_r" (root r's raw c_dot,
// uniform over every determinant, so with components along the other
// roots) or "rot_rs" (theta_sr = 1: c_dot_r = c_s, c_dot_s = -c_r).
Line line_of(const Point &p, const std::string &path) {
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
    for (std::size_t i = 0; i < ndet; ++i) {
      l.c[(root * ndet) + i] = uniform(engine);
    }
  }
  return l;
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

// qd_sc_adjoint at @p w + zc_adjoint at @p z, at the point moved by @p t
// along @p l (h and B straight, the roots e^{t K} C), each root's y as G(t).
Fields moved(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings, const Point &p,
             const Roots &roots, const Line &l, std::span<const double> u, const double t,
             std::span<const double> w, std::span<const double> z) {
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
      nevpt2::qd_sc_adjoint(res, strings, strings, p.naux, p.nao, p.nao, bd.data(), cd.data(),
                            hd.data(), p.ncore, p.ncas, p.weights, vd.data(), w, 0);
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
// and sum.
void expect_product(const std::string &name, const std::string &path) {
  const Point &p = point(name);
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0);
  const std::size_t nroots = p.weights.size();
  const auto nmo = static_cast<std::size_t>(p.nao);
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);
  const std::vector<double> w = general_w(nroots);
  const Roots roots{.n = nroots, .ndet = p.vectors.size() / nroots, .c = p.vectors};

  // z^c over qd_sc_adjoint's A and the SA Fock's epsilon at W.
  std::vector<double> z;
  {
    const auto adj =
        nevpt2::qd_sc_adjoint(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(),
                              h.data(), p.ncore, p.ncas, p.weights, vectors.data(), w, 0);
    ASSERT_TRUE(adj) << adj.error().message;
    const auto zc =
        nevpt2::zc_multipliers(res, p.nao, p.ncore, p.ncas, adj->a.data(), adj->epsilon.data());
    ASSERT_TRUE(zc) << zc.error().message;
    z = download(res, zc->z.data(), nmo * nmo);
  }
  const fb::Buffers::device<double> z_d = upload(res, z);

  const Line l = line_of(p, path);
  const auto [h_mo, b_mo] = mo_tangent(p, l);
  const fb::Buffers::device<double> h_dot = upload(res, h_mo);
  const fb::Buffers::device<double> b_dot = upload(res, b_mo);
  const fb::Buffers::device<double> c_dot = upload(res, l.c);
  const nevpt2::MoDirections along{
      .count = 1, .h = h_dot.data(), .b = b_dot.data(), .c = c_dot.data()};
  const auto product = [&](const bool parts) {
    return nevpt2::qd_sc_hessian_product(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(),
                                         c.data(), h.data(), p.ncore, p.ncas, p.weights,
                                         vectors.data(), w, z_d.data(), 0, along, {}, parts);
  };
  const auto got = product(false);
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->products.size(), 1U);
  const Fields predicted = fields_of(res, p, got->products.front());

  // The ladder: D_k per field, and the two-level Richardson extrapolant that
  // moves least to the next (nevpt2_qd_hvp_gpu.cpp's).
  const std::vector<double> u = roots.project(l.c);
  std::vector<Fields> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double step = std::ldexp(kT0, -k);
    const Fields plus = moved(res, *strings, p, roots, l, u, step, w, z);
    const Fields minus = moved(res, *strings, p, roots, l, u, -step, w, z);
    ASSERT_EQ(plus.size(), predicted.size());
    ASSERT_EQ(minus.size(), predicted.size());
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
  std::map<std::string, double> bars;
  double worst = 0.0;
  std::string report;
  for (const auto &[field, x] : predicted) {
    const double scale = max_abs(x);
    ASSERT_GT(scale, 0.0) << name << " " << path << " " << field << ": the product is zero";
    ASSERT_EQ(d.front().at(field).size(), x.size()) << field;
    bars[field] = kLadder * scale;
    std::size_t best = 0;
    double best_change = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k + 3 < static_cast<std::size_t>(kRungs); ++k) {
      double change = 0.0;
      for (std::size_t e = 0; e < x.size(); ++e) {
        change = std::max(change, std::abs(richardson(k + 1, field, e) - richardson(k, field, e)));
      }
      if (change < best_change) {
        best_change = change;
        best = k;
      }
    }
    double miss = 0.0;
    for (std::size_t e = 0; e < x.size(); ++e) {
      miss = std::max(miss, std::abs(x[e] - richardson(best, field, e)));
    }
    EXPECT_LE(miss, bars[field]) << name << " " << path << " " << field;
    worst = std::max(worst, miss / scale);
    report += std::format(" {} {:.1e} (rung {}, |{:.2e}|)", field, miss / scale, best, scale);
  }
  std::cout << std::format("[ nevpt2   ] qd hessian product ladder {} {}: worst {:.2e};{}\n", name,
                           path, worst, report);

  // The parts: each above ten bars in some field (the curvature where the
  // path has a CI part), and their sum the product.
  const auto split = product(true);
  ASSERT_TRUE(split) << split.error().message;
  ASSERT_EQ(split->parts.size(), 1U);
  const Fields total = fields_of(res, p, split->products.front());
  constexpr std::array<std::string_view, 4> kPartNames{"walk", "one-slot", "phi", "curvature"};
  const bool ci = max_abs(l.c) > 0.0;
  for (std::size_t q = 0; q < kPartNames.size(); ++q) {
    const Fields part = fields_of(res, p, split->parts.front()[q]);
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
  const Point &p = point("h8_sto3g_sa3");
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0);
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

} // namespace
