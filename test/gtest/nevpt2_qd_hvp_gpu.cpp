// GPU cases for nevpt2::sc_effective_hamiltonian_hvp, the SC-QD-NEVPT2
// effective Hamiltonian's Hessian-vector product in its inputs, forward over
// the pair walk (docs/nevpt2/nevpt2.tex Eq. qd-scfinish).
//
// - Against a ladder of qd_sc_adjoint's builder bars (QdBuilderBars: every
//   bar sc_effective_hamiltonian_adjoint forms, on the shared outputs, every
//   pair's RDMs and B through the slabs) at a general symmetric W
//   (off-diagonal, trace != 1, a negative entry), along one straight line in
//   (h, B, every root's c): the product's direction is that line's tangent
//   of the builders' outputs (qd_builders_tangent), so the bars' derivative
//   along it is the product, field by field. Central differences at
//   t = 1e-3 / 2^k, k < 10, their two-level Richardson extrapolants, and per
//   field the coarsest settled one (settled_rung.h): every field within 1e-7 of
//   its largest element, where SC-QD's adjoint ladders (kQdSmooth in
//   nevpt2_gradient_gpu.cpp; #724, #894): every path on h8_sto3g_sa3, and
//   for the one-function classes h8 lacks, h6_631g_sa2 and h2co_ccpvdz_sa2;
//   and h2co_skew_ccpvdz_sa3 along c_0, every field but the (1, 2)
//   transitions, which sit at its conditioning floor and are printed beside
//   it (below, beside the case list).
// - Visibility, in the same cases: the product along the line's orbital
//   energies alone, which move Δ and nothing a fixed-multiplier difference
//   sees, so it is the finish's second derivatives times Δ̇ alone. At W's
//   off-diagonal part it is Eq. qd-scfinish's off-diagonal terms, at its
//   diagonal part the diagonal's 3 x 3 (Sijrs's share, on Λ and the
//   orbital energies, aside): each above ten bars of the whole product in
//   some field.
// - The A1/B2 fixture (data/qd_nevpt2_hessian/h2nb_sto3g_sa2.json, #1058):
//   along every pair of its C2v vibrations, the parameter response's tangent
//   (hessian::df_qd_nevpt2_tangents) through qd_builders_tangent, with the
//   other coordinate as partner, the product refuses exactly the fixture's
//   SC expected-refusal blocks (none), the totally symmetric block among
//   them.
// - An accepted off-diagonal K6: on h2co_ccpvdz_sa2 (C2v, 1A1 and 1A2,
//   4364 tuples empty for both roots) a random B direction u populates them,
//   and every coupling between the roots takes K6 there. The bars jump as a
//   populated tuple's norm crosses the guard, so no ladder of them settles
//   there; the energy moves by about the guard's 1e-14 through it, so
//   <u, HVP u> is held to a ladder of E[W]'s second difference along the
//   straight line in the inputs, two-level Richardson, to 1e-7, and the
//   couplings' K6 share of it (ScEffectiveHamiltonianHvp::coupling_k6) is
//   above ten of those bars. The roots' relative sign is pinned (d H_01 / ds
//   > 0): the SA-CASSCF does not repeat it, and E[W] at the other sign is E
//   at W's off-diagonal negated (#1071).
// - The ρ_τ refusal on constructed inputs: a ket with an empty active
//   orbital and a virtual whose B to the other active orbitals is zeroed, so
//   that Srsi's (r, r, i) tuples are empty for that ket and populated for
//   the other, and two random B directions that populate them with
//   different functions; refused as Numerical, naming the coupling, the
//   class and the directions. The same direction as its own partner, or
//   none, is accepted.

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
import free_breakfast.response.zvector;
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
// The CASSCF's bar: loose, since no stationarity is used.
constexpr double kConvTolGrad = 1e-6;
// h2co_skew_ccpvdz_sa3's bars' repeat, relative to their largest element
// (README, "The SC-QD reverse pass", Conditioning): a central difference at
// t is floored near it over t.
constexpr double kSkewRepeat = 1e-9;
// B_dot a tenth the length of the others, as the ladders' directions.
constexpr double kBScale = 0.1;
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
// registry, as nevpt2_qd_tangent_gpu.cpp's, at kConvTolGrad.
Point sa_casscf(const std::string &name) {
  const fb::json &entry = qd_case(name);
  Point p{.weights = entry.at("weights").get<std::vector<double>>()};
  EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_qd_hvp");
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

// A general symmetric W: seeded, off-diagonal, trace != 1, with a negative
// entry, every one of which is checked.
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

// One straight line in (h, B, every root's c): the AO h and B's tangents
// (h symmetric, B symmetric in its AO pair) and every root's c, root-major.
struct Line {
  std::vector<double> h;
  std::vector<double> b;
  std::vector<double> c;
};

// The seeded line @p path moves alone: "h", "B" or "c_r".
Line line_of(const Point &p, const std::string &path) {
  const auto nao = static_cast<std::size_t>(p.nao);
  const auto naux = static_cast<std::size_t>(p.naux);
  std::mt19937_64 engine(std::hash<std::string>{}(path) % 9629);
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
  } else {
    const std::size_t root = std::stoul(path.substr(2));
    const std::size_t ndet = p.vectors.size() / p.weights.size();
    for (std::size_t i = 0; i < ndet; ++i) {
      l.c[(root * ndet) + i] = uniform(engine);
    }
  }
  return l;
}

// The line's tangent in the MO basis, as qd_builders_tangent reads it.
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

// The bars, named per field: the shared ones, then each pair's RDMs'.
using Fields = std::map<std::string, std::vector<double>>;

// What qd_sc_adjoint keeps at the point moved by @p t along @p l.
Fields ladder_bars(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings, const Point &p,
                   const Line &l, const double t, std::span<const double> w) {
  std::vector<double> b = p.b;
  std::vector<double> h = p.h;
  std::vector<double> vectors = p.vectors;
  for (std::size_t k = 0; k < b.size(); ++k) {
    b[k] += t * l.b[k];
  }
  for (std::size_t k = 0; k < h.size(); ++k) {
    h[k] += t * l.h[k];
  }
  for (std::size_t k = 0; k < vectors.size(); ++k) {
    vectors[k] += t * l.c[k];
  }
  const fb::Buffers::device<double> bd = upload(res, b);
  const fb::Buffers::device<double> cd = upload(res, p.c);
  const fb::Buffers::device<double> hd = upload(res, h);
  const fb::Buffers::device<double> vd = upload(res, vectors);
  std::optional<nevpt2::QdBuilderBars> kept;
  const auto adj =
      nevpt2::qd_sc_adjoint(res, strings, strings, p.naux, p.nao, p.nao, bd.data(), cd.data(),
                            hd.data(), p.ncore, p.ncas, p.weights, vd.data(), w, 0, 0, &kept);
  EXPECT_TRUE(adj) << adj.error().message;
  Fields out;
  if (!adj || !kept) {
    return out;
  }
  const nevpt2::QdBuilderBars &kb = *kept;
  const nevpt2::BuilderOutputs &s = kb.bars.shared;
  for (const auto &[name, x] :
       {std::pair{"h1e", &s.h1e}, std::pair{"h2e", &s.h2e}, std::pair{"e_core", &s.e_core},
        std::pair{"e_virt", &s.e_virt}, std::pair{"h1e_v_sr", &s.h1e_v_sr},
        std::pair{"h1e_v_si", &s.h1e_v_si}, std::pair{"h1e_v_sir", &s.h1e_v_sir},
        std::pair{"lambda", &kb.lambda}}) {
    out[name] = download(res, *x);
  }
  const std::size_t n = p.weights.size();
  for (std::size_t q = 0; q < n * n; ++q) {
    const nevpt2::PairRdms &r = kb.bars.pairs[q];
    const std::string pair = std::format("({}, {})", q / n, q % n);
    for (const auto &[name, x] :
         {std::pair{"dm1", &r.dm1}, std::pair{"dm2", &r.dm2}, std::pair{"dm3", &r.dm3},
          std::pair{"f3ca", &r.f3ca}, std::pair{"f3ac", &r.f3ac}}) {
      if (x->num_elements() > 0) {
        out[std::format("{} {}", name, pair)] = download(res, *x);
      }
    }
  }
  return out;
}

// Zeroed bars for one direction over n states, sc_effective_hamiltonian_hvp's
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
  for (std::size_t j = 0; j < n; ++j) {
    nevpt2::EnergyBars b = shared;
    b.dm1 = zeros(na * na);
    b.dm2 = zeros(na * na * na * na);
    b.dm3 = zeros(na * na * na * na * na * na);
    b.f3ca = zeros(na * na * na * na * na * na);
    b.f3ac = zeros(na * na * na * na * na * na);
    out.bars.push_back(b);
  }
  return out;
}

// @p b read back by ladder_bars' names.
Fields read_bars(fb::DeviceResources &res, const nevpt2::MoBlocks &blocks, const HvpBars &b) {
  const auto na = static_cast<std::size_t>(blocks.ncas);
  const auto nc = static_cast<std::size_t>(blocks.ncore);
  const auto nv = static_cast<std::size_t>(blocks.nvirt);
  const std::size_t nmo = nc + na + nv;
  const std::size_t n = b.bars.size();
  const nevpt2::EnergyBars &s = b.bars.front();
  Fields out = b.transition;
  for (const auto &[name, x, count] :
       {std::tuple{"h1e", s.h1e, na * na}, std::tuple{"h2e", s.h2e, na * na * na * na},
        std::tuple{"e_core", s.e_core, nc}, std::tuple{"e_virt", s.e_virt, nv},
        std::tuple{"h1e_v_sr", s.h1e_v_sr, nv * na}, std::tuple{"h1e_v_si", s.h1e_v_si, na * nc},
        std::tuple{"h1e_v_sir", s.h1e_v_sir, nv * nc},
        std::tuple{"lambda", s.lambda, static_cast<std::size_t>(blocks.naux) * nmo * nmo}}) {
    out[name] = download(res, x, count);
  }
  const std::size_t na6 = na * na * na * na * na * na;
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

// sc_effective_hamiltonian_hvp over @p pairs along the one direction
// @p directions (n² entries) at @p w, every bar read back.
fb::Result<std::pair<nevpt2::ScEffectiveHamiltonianHvp, Fields>>
run_hvp(fb::DeviceResources &res, const nevpt2::MoBlocks &blocks,
        std::span<const nevpt2::EnergyInputs> pairs, std::span<const double> w,
        std::span<const nevpt2::EnergyDirection> directions,
        std::span<const nevpt2::EnergyDirection> partners = {}) {
  const std::size_t n = static_cast<std::size_t>(std::llround(std::sqrt(pairs.size())));
  const auto na = static_cast<std::size_t>(blocks.ncas);
  HvpBars b = hvp_bars(res, blocks, n);
  const auto got = nevpt2::sc_effective_hamiltonian_hvp(
      res, blocks, pairs, w, 0, directions, b.bars,
      [&](const std::int64_t, const std::int64_t bra, const std::int64_t ket,
          const nevpt2::EnergyBars &t) -> fb::Status {
        const std::string pair = std::format("({}, {})", bra, ket);
        for (const auto &[name, x, count] :
             {std::tuple{"dm1", t.dm1, na * na}, std::tuple{"dm2", t.dm2, na * na * na * na},
              std::tuple{"dm3", t.dm3, na * na * na * na * na * na}}) {
          b.transition[std::format("{} {}", name, pair)] = download(res, x, count);
        }
        return {};
      },
      partners);
  if (!got) {
    return std::unexpected(got.error());
  }
  return std::pair{*got, read_bars(res, blocks, b)};
}

double max_abs(std::span<const double> v) {
  double out = 0.0;
  for (const double x : v) {
    out = std::max(out, std::abs(x));
  }
  return out;
}

// The product along one line against the ladder of the bars, and the
// visibility of its finish's second derivatives.
void expect_ladder(const std::string &name, const std::string &path) {
  const Point p = sa_casscf(name);
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  const std::size_t nroots = p.weights.size();
  const std::size_t n2 = nroots * nroots;
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);
  const Line l = line_of(p, path);
  const auto [h_mo, b_mo] = mo_tangent(p, l);
  const fb::Buffers::device<double> h_dot = upload(res, h_mo);
  const fb::Buffers::device<double> b_dot = upload(res, b_mo);
  const fb::Buffers::device<double> c_dot = upload(res, l.c);
  auto tangent = nevpt2::qd_builders_tangent(
      res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(), h.data(), p.ncore, p.ncas,
      p.weights, vectors.data(), 1, h_dot.data(), b_dot.data(), c_dot.data());
  ASSERT_TRUE(tangent) << tangent.error().message;
  const nevpt2::MoBlocks &blocks = tangent->value.shared.blocks;
  const std::vector<nevpt2::EnergyInputs> pairs = tangent->value.inputs();
  const nevpt2::QdBuilderOutputs &t = tangent->tangents.front();
  std::vector<nevpt2::EnergyDirection> directions;
  for (const nevpt2::EnergyInputs &in : t.inputs()) {
    directions.push_back({.in = in, .blocks = &t.shared.blocks});
  }
  const std::vector<double> w = general_w(nroots);
  const auto got = run_hvp(res, blocks, pairs, w, directions);
  ASSERT_TRUE(got) << got.error().message;
  const Fields &hvp = got->second;

  // The ladder: D_k per field, and per field the coarsest settled two-level
  // Richardson extrapolant (settled_rung.h).
  std::vector<Fields> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double step = std::ldexp(kT0, -k);
    const Fields plus = ladder_bars(res, *strings, p, l, step, w);
    const Fields minus = ladder_bars(res, *strings, p, l, -step, w);
    ASSERT_EQ(plus.size(), hvp.size());
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
  // R_k = (4 D_{k+1} - D_k) / 3 cancels the t² term, and its two-level
  // extrapolant (16 R_{k+1} - R_k) / 15 the t⁴ one: along h on h8_sto3g_sa3
  // R_k's error falls 16-fold a rung and is still 2e-7 of Λ's and h1e_v_sir's
  // largest element at the rung where R_k and R_{k+1} agree best.
  const auto richardson = [&](const std::size_t k, const std::string &f, const std::size_t e) {
    const auto r = [&](const std::size_t j) {
      return ((4.0 * d[j + 1].at(f)[e]) - d[j].at(f)[e]) / 3.0;
    };
    return ((16.0 * r(k + 1)) - r(k)) / 15.0;
  };
  std::map<std::string, double> scale;
  for (const auto &[field, x] : hvp) {
    scale[field] = max_abs(x);
  }
  // Each field's own ladder: its coarsest settled extrapolant against its
  // bar, 1e-7 of its largest element, is its value. On
  // h2co_skew_ccpvdz_sa3 the (1, 2) transitions, both orders, are printed
  // against their floor rather than held to the bar (below, beside the case
  // list).
  const bool skew = name == "h2co_skew_ccpvdz_sa3";
  double worst = 0.0;
  std::string report;
  std::string floored;
  for (const auto &[field, x] : hvp) {
    ASSERT_EQ(d.front().at(field).size(), x.size()) << field;
    if (scale[field] == 0.0) {
      // A field nothing moves along this line is zero on both sides.
      EXPECT_EQ(max_abs(d.back().at(field)), 0.0) << name << " " << path << " " << field;
      continue;
    }
    const auto [best, best_change] = fb::test::settled_rung(
        static_cast<std::size_t>(kRungs) - 3, kLadder * scale[field], [&](const std::size_t k) {
          double change = 0.0;
          for (std::size_t e = 0; e < x.size(); ++e) {
            change =
                std::max(change, std::abs(richardson(k + 1, field, e) - richardson(k, field, e)));
          }
          return change;
        });
    double miss = 0.0;
    for (std::size_t e = 0; e < x.size(); ++e) {
      miss = std::max(miss, std::abs(x[e] - richardson(best, field, e)));
    }
    if (skew && (field.ends_with("(1, 2)") || field.ends_with("(2, 1)"))) {
      // The extrapolant's finest rung, t = kT0 / 2^(best + 2), divides the
      // bars' repeat: (16 · 4 / 45) D_{best+2}'s share of it, ~1.4 kSkewRepeat
      // / t.
      const double floor =
          1.4 * kSkewRepeat / std::ldexp(kT0, -static_cast<std::int32_t>(best + 2));
      floored += std::format(" {} {:.1e} (rung {}, Si floor {:.1e})", field, miss / scale[field],
                             best, floor);
      continue;
    }
    EXPECT_LE(miss, kLadder * scale[field]) << name << " " << path << " " << field;
    worst = std::max(worst, miss / scale[field]);
    report += std::format(" {} {:.1e} (rung {}, change {:.1e})", field, miss / scale[field], best,
                          best_change / scale[field]);
  }
  std::cout << std::format("[ nevpt2   ] qd hvp ladder {} {}: worst {:.2e};{}\n", name, path, worst,
                           report);
  if (!floored.empty()) {
    std::cout << std::format("[ nevpt2   ] qd hvp ladder {} {}: not gated, floored;{}\n", name,
                             path, floored);
  }

  // The visibility: the product along the line's orbital energies alone, at
  // W's off-diagonal and diagonal parts, against the whole product's bars.
  std::vector<nevpt2::EnergyDirection> epsilon;
  for (std::size_t q = 0; q < n2; ++q) {
    epsilon.push_back({.in = {.e_core = t.shared.e_core.data(), .e_virt = t.shared.e_virt.data()}});
  }
  for (const bool off : {true, false}) {
    std::vector<double> part = w;
    for (std::size_t i = 0; i < nroots; ++i) {
      for (std::size_t j = 0; j < nroots; ++j) {
        if ((i != j) != off) {
          part[(i * nroots) + j] = 0.0;
        }
      }
    }
    const auto e = run_hvp(res, blocks, pairs, part, epsilon);
    ASSERT_TRUE(e) << e.error().message;
    double visible = 0.0;
    std::string where;
    for (const auto &[field, x] : e->second) {
      // Sijrs's share, Σ_J H̄_JJ times one state's, is on Λ and the orbital
      // energies alone.
      if (!off && (field == "lambda" || field == "e_core" || field == "e_virt")) {
        continue;
      }
      const double bars = max_abs(x) / (kLadder * scale[field]);
      if (scale[field] > 0.0 && bars > visible) {
        visible = bars;
        where = field;
      }
    }
    EXPECT_GT(visible, kVisible) << name << " " << path << ": "
                                 << (off ? "the off-diagonal finish" : "the diagonal 3 x 3");
    std::cout << std::format("[ nevpt2   ] qd hvp {} {}: {} at most {:.2e} bars ({})\n", name, path,
                             off ? "off-diagonal finish" : "diagonal 3 x 3", visible, where);
  }
}

class Nevpt2QdHvp : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(Nevpt2QdHvp, BarsMatchRichardsonLadder) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("h8_sto3g_sa3_B");
  const auto [name, path] = GetParam();
  if (name != "h8_sto3g_sa3" || path != "B") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  expect_ladder(name, path);
}

// h8_sto3g_sa3, every path. h6_631g_sa2 (CAS(2,3): Srs a one-function
// class) and h2co_ccpvdz_sa2 (CAS(4,3): Sij), along kQdSmooth's random-seed
// paths, carry the one-function classes h8's CAS(4,4) does not.
// h2co_skew_ccpvdz_sa3 along c_0 only (#1055, #1069; README, "The SC-QD
// reverse pass", Conditioning). Root 2's Si tuples have norms of 2e-7 to
// 7e-7, and their bars carry each walk's round-off times 1/N: the transition
// bars repeat only to ~1e-9 of their largest element (kSkewRepeat), so a
// ladder's central difference at t is floored near 1e-9 / t, at least 1e-6
// for t <= 1e-3. c_0 moves the (1, 2) transitions only through ε, so they
// settle at that floor, at the coarsest rung pair, and are printed beside
// it; every other field is held to kLadder. Along B the same tuples have
// N / |Ṅ| of 5e-4 to 8e-4, so the ladder's coarse rungs reach the norm's
// near-zero (no Taylor regime there; root 2's fields end at 4.7e-4 to
// 3e-3), and its fine rungs are floored: no window. h2co_ccpvdz_sa2's B
// populates its empty tuples, whose bars jump at the guard:
// Nevpt2QdHvpEmpty checks it through the energy.
INSTANTIATE_TEST_SUITE_P(
    Cases, Nevpt2QdHvp,
    ::testing::Values(std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"h"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"B"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"c_0"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"c_1"}},
                      std::tuple{std::string{"h8_sto3g_sa3"}, std::string{"c_2"}},
                      std::tuple{std::string{"h6_631g_sa2"}, std::string{"B"}},
                      std::tuple{std::string{"h6_631g_sa2"}, std::string{"c_0"}},
                      std::tuple{std::string{"h2co_ccpvdz_sa2"}, std::string{"c_0"}},
                      std::tuple{std::string{"h2co_ccpvdz_sa2"}, std::string{"c_1"}},
                      std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, std::string{"c_0"}}),
    [](const auto &info) { return std::get<0>(info.param) + "_" + std::get<1>(info.param); });

// ---- An accepted off-diagonal K6 ---------------------------------------------

// The device arrays of one point of the inputs, owned: every pair's
// EnergyInputs over shared fields and the MO blocks, at @p value + s @p dot.
struct MovedInputs {
  std::deque<fb::Buffers::device<double>> held;
  std::optional<nevpt2::MoBlocks> blocks;
  std::vector<nevpt2::EnergyInputs> pairs;
};

MovedInputs moved_inputs(fb::DeviceResources &res, const nevpt2::QdBuilderOutputs &value,
                         const nevpt2::QdBuilderOutputs &dot, const double s) {
  MovedInputs out;
  const auto moved = [&](const fb::Buffers::device<double> &x,
                         const fb::Buffers::device<double> &dx) {
    std::vector<double> host = download(res, x);
    const std::vector<double> d = download(res, dx);
    for (std::size_t e = 0; e < host.size(); ++e) {
      host[e] += s * d[e];
    }
    out.held.push_back(upload(res, host));
    return out.held.back().data();
  };
  const nevpt2::BuilderOutputs &v = value.shared;
  const nevpt2::BuilderOutputs &d = dot.shared;
  const nevpt2::EnergyInputs shared{.h1e = moved(v.h1e, d.h1e),
                                    .h2e = moved(v.h2e, d.h2e),
                                    .e_core = moved(v.e_core, d.e_core),
                                    .e_virt = moved(v.e_virt, d.e_virt),
                                    .h1e_v_sr = moved(v.h1e_v_sr, d.h1e_v_sr),
                                    .h1e_v_si = moved(v.h1e_v_si, d.h1e_v_si),
                                    .h1e_v_sir = moved(v.h1e_v_sir, d.h1e_v_sir)};
  for (std::size_t q = 0; q < value.pairs.size(); ++q) {
    const nevpt2::PairRdms &x = value.pairs[q];
    const nevpt2::PairRdms &dx = dot.pairs[q];
    nevpt2::EnergyInputs in = shared;
    in.dm1 = moved(x.dm1, dx.dm1);
    in.dm2 = moved(x.dm2, dx.dm2);
    in.dm3 = moved(x.dm3, dx.dm3);
    if (x.f3ca.num_elements() > 0) {
      in.f3ca = moved(x.f3ca, dx.f3ca);
      in.f3ac = moved(x.f3ac, dx.f3ac);
    }
    out.pairs.push_back(in);
  }
  const nevpt2::MoBlocks &b = v.blocks;
  const nevpt2::MoBlocks &db = d.blocks;
  const auto block = [&](const fb::Buffers::device<double> &x,
                         const fb::Buffers::device<double> &dx) {
    std::vector<double> host = download(res, x);
    const std::vector<double> dd = download(res, dx);
    for (std::size_t e = 0; e < host.size(); ++e) {
      host[e] += s * dd[e];
    }
    return upload(res, host);
  };
  out.blocks.emplace(nevpt2::MoBlocks{.naux = b.naux,
                                      .ncore = b.ncore,
                                      .ncas = b.ncas,
                                      .nvirt = b.nvirt,
                                      .aa = block(b.aa, db.aa),
                                      .ca = block(b.ca, db.ca),
                                      .va = block(b.va, db.va),
                                      .cv = block(b.cv, db.cv),
                                      .vc = block(b.vc, db.vc)});
  return out;
}

// d H_01 / ds along @p t at @p value, H the SC effective Hamiltonian: a
// central difference at s = 1e-4. It changes sign with either root's, and
// with no orbital's, so its sign is the roots' relative phase.
double coupling_slope(fb::DeviceResources &res, const nevpt2::QdBuilderOutputs &value,
                      const nevpt2::QdBuilderOutputs &t) {
  constexpr double kS = 1e-4;
  const MovedInputs up = moved_inputs(res, value, t, kS);
  const MovedInputs down = moved_inputs(res, value, t, -kS);
  const auto e_up = nevpt2::sc_effective_hamiltonian(res, *up.blocks, up.pairs, 0);
  const auto e_down = nevpt2::sc_effective_hamiltonian(res, *down.blocks, down.pairs, 0);
  EXPECT_TRUE(e_up && e_down);
  if (!e_up || !e_down) {
    return 0.0;
  }
  return ((*e_up)[1].total - (*e_down)[1].total) / (2.0 * kS);
}

TEST(Nevpt2QdHvpEmpty, CouplingK6MatchesTheEnergysSecondDifference) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const std::string name = "h2co_ccpvdz_sa2";
  Point p = sa_casscf(name);
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  const std::size_t nroots = p.weights.size();
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const Line l = line_of(p, "B");
  const auto [h_mo, b_mo] = mo_tangent(p, l);
  const fb::Buffers::device<double> h_dot = upload(res, h_mo);
  const fb::Buffers::device<double> b_dot = upload(res, b_mo);
  const fb::Buffers::device<double> c_dot = upload(res, l.c);
  const auto builders_tangent = [&](const fb::Buffers::device<double> &vectors) {
    return nevpt2::qd_builders_tangent(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(),
                                       c.data(), h.data(), p.ncore, p.ncas, p.weights,
                                       vectors.data(), 1, h_dot.data(), b_dot.data(), c_dot.data());
  };
  // The point, pinned. The SA-CASSCF repeats to 1e-8 in C and 1e-12 in
  // each root's energy, but not in the roots' relative sign, and E[W] at
  // the other sign is E at W's off-diagonal negated: <u, HVP u> is
  // -1963.74 at one and -1803.91 at the other (#1071). Root 1 is negated
  // where needed so that d H_01 / ds is positive, the sign at which this
  // case's numbers below were measured.
  {
    const fb::Buffers::device<double> vectors = upload(res, p.vectors);
    const auto first = builders_tangent(vectors);
    ASSERT_TRUE(first) << first.error().message;
    if (coupling_slope(res, first->value, first->tangents.front()) < 0.0) {
      const std::size_t ndet = p.vectors.size() / nroots;
      for (std::size_t i = ndet; i < 2 * ndet; ++i) {
        p.vectors[i] = -p.vectors[i];
      }
    }
  }
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);
  const auto tangent = builders_tangent(vectors);
  ASSERT_TRUE(tangent) << tangent.error().message;
  const nevpt2::QdBuilderOutputs &value = tangent->value;
  const nevpt2::QdBuilderOutputs &t = tangent->tangents.front();
  const double slope = coupling_slope(res, value, t);
  ASSERT_GT(slope, 0.0) << "the pinned phase";
  std::vector<nevpt2::EnergyDirection> directions;
  for (const nevpt2::EnergyInputs &in : t.inputs()) {
    directions.push_back({.in = in, .blocks = &t.shared.blocks});
  }
  const std::vector<double> w = general_w(nroots);
  const auto got = run_hvp(res, value.shared.blocks, value.inputs(), w, directions);
  ASSERT_TRUE(got) << got.error().message;
  std::int64_t populated = 0;
  for (const auto &ket : got->first.populated.front()) {
    populated += std::ranges::fold_left(ket, std::int64_t{0}, std::plus{});
  }
  EXPECT_GT(populated, 0);

  // <u, HVP u>: each field of the product against the line's tangent of it.
  const Fields &hvp = got->second;
  double predicted = 0.0;
  const auto pair_up = [&](const std::string &field, const fb::Buffers::device<double> &x) {
    if (x.num_elements() > 0) {
      const std::vector<double> d = download(res, x);
      const std::vector<double> &bar = hvp.at(field);
      predicted += std::inner_product(bar.begin(), bar.end(), d.begin(), 0.0);
    }
  };
  const nevpt2::BuilderOutputs &ts = t.shared;
  for (const auto &[field, x] :
       {std::pair{"h1e", &ts.h1e}, std::pair{"h2e", &ts.h2e}, std::pair{"e_core", &ts.e_core},
        std::pair{"e_virt", &ts.e_virt}, std::pair{"h1e_v_sr", &ts.h1e_v_sr},
        std::pair{"h1e_v_si", &ts.h1e_v_si}, std::pair{"h1e_v_sir", &ts.h1e_v_sir}}) {
    pair_up(field, *x);
  }
  for (std::size_t q = 0; q < nroots * nroots; ++q) {
    const std::string pair = std::format("({}, {})", q / nroots, q % nroots);
    const nevpt2::PairRdms &r = t.pairs[q];
    for (const auto &[field, x] :
         {std::pair{"dm1", &r.dm1}, std::pair{"dm2", &r.dm2}, std::pair{"dm3", &r.dm3},
          std::pair{"f3ca", &r.f3ca}, std::pair{"f3ac", &r.f3ac}}) {
      pair_up(std::format("{} {}", field, pair), *x);
    }
  }
  {
    const std::vector<double> &lambda = hvp.at("lambda");
    predicted += std::inner_product(lambda.begin(), lambda.end(), b_mo.begin(), 0.0);
  }

  // E[W] along the line, its second difference D(s) at s = 2.5e-3 / 2^k,
  // k < 10, and its two-level Richardson extrapolants; the coarsest settled
  // one against the bar (settled_rung.h) is the ladder's value. Measured on the RTX 3080: D's
  // error falls from 1.5e-3 to 2.9e-7 of <u, HVP u> by s = 9.8e-6 and then
  // grows (4.8e-5 at 2.4e-6), the populated tuples' H / N round-off near the
  // guard over s²; the extrapolants settle at s = 7.8e-5, to 2e-8.
  const auto energy = [&](const double s) {
    MovedInputs x = moved_inputs(res, value, t, s);
    const auto e = nevpt2::sc_effective_hamiltonian(res, *x.blocks, x.pairs, 0);
    EXPECT_TRUE(e) << e.error().message;
    double sum = 0.0;
    for (std::size_t q = 0; e && q < e->size(); ++q) {
      sum += w[q] * (*e)[q].total;
    }
    return sum;
  };
  const double e0 = energy(0.0);
  std::vector<double> d;
  for (std::int32_t k = 0; k < 10; ++k) {
    const double s = std::ldexp(2.5e-3, -k);
    d.push_back((energy(s) - (2.0 * e0) + energy(-s)) / (s * s));
  }
  const auto r2 = [&](const std::size_t k) {
    const double r = ((4.0 * d[k + 1]) - d[k]) / 3.0;
    const double r_next = ((4.0 * d[k + 2]) - d[k + 1]) / 3.0;
    return ((16.0 * r_next) - r) / 15.0;
  };
  const double bar = kLadder * std::abs(predicted);
  const auto [best, best_change] = fb::test::settled_rung(
      d.size() - 3, bar, [&](const std::size_t k) { return std::abs(r2(k + 1) - r2(k)); });
  const double ladder = r2(best);
  EXPECT_LE(std::abs(ladder - predicted), bar);
  const double k6 = got->first.coupling_k6.front();
  EXPECT_GT(std::abs(k6), kVisible * bar) << "the couplings' K6";
  std::cout << std::format("[ nevpt2   ] qd hvp K6 {}: <u, HVP u> {:+.12e}, E[W]'' ladder "
                           "{:+.12e} (rung {}, change {:.1e}), relative {:.2e}; {} empty tuples "
                           "populated, the couplings' K6 {:+.3e} ({:.2e} bars); dH_01/ds {:+.6e}\n",
                           name, predicted, ladder, best, best_change / std::abs(predicted),
                           std::abs(ladder - predicted) / std::abs(predicted), populated, k6,
                           std::abs(k6) / bar, slope);
}

// ---- The ρ_τ refusal, on constructed inputs --------------------------------

// h8_sto3g_sa3's MO integrals (C = 1, B and h in its SA-CASSCF's MO basis)
// over another partition: three core orbitals, three active with two
// electrons, two virtuals. Ket 1's CI vector never occupies active orbital
// 0, so every RDM element with that index vanishes, and virtual 0's B to the
// other two active orbitals is zeroed: Srsi's (r0, r0, i) tuple then reads
// active orbital 0 alone, which ket 1 leaves empty, so N^1 = 0 exactly
// there, while bra 0 occupies it. Along a B direction the zeroed entries
// open, the tuple is populated at second order for ket 1, and its
// transition T to bra 0 moves at first order: Eq. qd-scfail's one-sided
// tuple, in a class of three functions.
TEST(Nevpt2QdHvpEmpty, OneSidedTupleIsRefusedOnANonProportionalPair) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  const Point base = sa_casscf("h8_sto3g_sa3");
  ASSERT_FALSE(base.b.empty());
  fb::DeviceResources res(0, fb::test::kWorkspaces);
  constexpr std::int32_t kNcore = 3;
  constexpr std::int32_t kNcas = 3;
  constexpr std::int32_t kNelecas = 2;
  const auto nmo = static_cast<std::size_t>(base.nao);
  const auto naux = static_cast<std::size_t>(base.naux);
  Point p = base;
  p.ncore = kNcore;
  p.ncas = kNcas;
  p.nelecas = kNelecas;
  p.weights = {0.5, 0.5};
  // B and h in the MO basis, C the identity.
  p.h = to_mo(base.c, base.h, nmo);
  std::vector<double> slice(nmo * nmo);
  for (std::size_t q = 0; q < naux; ++q) {
    for (std::size_t mn = 0; mn < nmo * nmo; ++mn) {
      slice[mn] = base.b[q + (naux * mn)];
    }
    const std::vector<double> mo = to_mo(base.c, slice, nmo);
    for (std::size_t pq = 0; pq < nmo * nmo; ++pq) {
      p.b[q + (naux * pq)] = mo[pq];
    }
  }
  p.c.assign(nmo * nmo, 0.0);
  for (std::size_t i = 0; i < nmo; ++i) {
    p.c[i * (nmo + 1)] = 1.0;
  }
  const std::size_t r0 = kNcore + kNcas;
  for (std::size_t a = kNcore + 1; a < kNcore + kNcas; ++a) {
    for (std::size_t q = 0; q < naux; ++q) {
      p.b[q + (naux * (r0 + (nmo * a)))] = 0.0;
      p.b[q + (naux * (a + (nmo * r0)))] = 0.0;
    }
  }
  // The roots over the three α and three β strings (one electron each), the
  // string with active orbital 0 first: ket 1 has no weight on it.
  auto strings = fb::ci::DeviceStringSet::create(res, kNcas, kNelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  constexpr std::size_t kStrings = 3;
  std::mt19937_64 engine(9621);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  p.vectors.assign(2 * kStrings * kStrings, 0.0);
  for (std::size_t r = 0; r < 2; ++r) {
    double norm = 0.0;
    for (std::size_t a = 0; a < kStrings; ++a) {
      for (std::size_t bb = 0; bb <= a; ++bb) {
        const double x = (r == 1 && (a == 0 || bb == 0)) ? 0.0 : uniform(engine);
        p.vectors[(r * kStrings * kStrings) + (a * kStrings) + bb] =
            p.vectors[(r * kStrings * kStrings) + (bb * kStrings) + a] = x;
        norm += (a == bb ? 1.0 : 2.0) * x * x;
      }
    }
    for (std::size_t k = 0; k < kStrings * kStrings; ++k) {
      p.vectors[(r * kStrings * kStrings) + k] /= std::sqrt(norm);
    }
  }

  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);
  // Two seeded B directions, symmetric in the pair.
  std::vector<double> b_dots(2 * naux * nmo * nmo, 0.0);
  for (std::size_t u = 0; u < 2; ++u) {
    for (std::size_t mu = 0; mu < nmo; ++mu) {
      for (std::size_t nu = 0; nu <= mu; ++nu) {
        for (std::size_t q = 0; q < naux; ++q) {
          const double x = kBScale * uniform(engine);
          b_dots[(u * naux * nmo * nmo) + q + (naux * (mu + (nmo * nu)))] = x;
          b_dots[(u * naux * nmo * nmo) + q + (naux * (nu + (nmo * mu)))] = x;
        }
      }
    }
  }
  const std::vector<double> zero_h(2 * nmo * nmo, 0.0);
  const std::vector<double> zero_c(2 * p.vectors.size(), 0.0);
  const fb::Buffers::device<double> h_dot = upload(res, zero_h);
  const fb::Buffers::device<double> b_dot = upload(res, b_dots);
  const fb::Buffers::device<double> c_dot = upload(res, zero_c);
  auto tangent = nevpt2::qd_builders_tangent(
      res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(), h.data(), p.ncore, p.ncas,
      p.weights, vectors.data(), 2, h_dot.data(), b_dot.data(), c_dot.data());
  ASSERT_TRUE(tangent) << tangent.error().message;
  const nevpt2::MoBlocks &blocks = tangent->value.shared.blocks;
  const std::vector<nevpt2::EnergyInputs> pairs = tangent->value.inputs();
  // Ket 1 leaves active orbital 0 empty, bra 0 does not.
  const std::vector<double> dm1_ket = download(res, tangent->value.pairs[3].dm1);
  const std::vector<double> dm1_bra = download(res, tangent->value.pairs[0].dm1);
  EXPECT_EQ(dm1_ket[0], 0.0);
  EXPECT_GT(dm1_bra[0], 1e-3);
  const auto direction = [&](const std::size_t u) {
    std::vector<nevpt2::EnergyDirection> out;
    for (const nevpt2::EnergyInputs &in : tangent->tangents[u].inputs()) {
      out.push_back({.in = in, .blocks = &tangent->tangents[u].shared.blocks});
    }
    return out;
  };
  const std::vector<nevpt2::EnergyDirection> u = direction(0);
  const std::vector<nevpt2::EnergyDirection> v = direction(1);
  const std::vector<double> w{0.3, 0.7, 0.7, -0.4};

  // The tuple is one-sided: empty for ket 1, populated for bra 0, with a
  // first-order transition tangent along u.
  const auto tan = nevpt2::sc_effective_hamiltonian_tangent(res, blocks, pairs, 0, u);
  ASSERT_TRUE(tan) << tan.error().message;
  constexpr std::size_t kSrsi = 4;
  const std::vector<std::int64_t> &empty = tan->empty[1][kSrsi];
  ASSERT_FALSE(empty.empty());
  double one_sided = 0.0;
  for (const std::int64_t g : empty) {
    const auto x = static_cast<std::size_t>(g);
    if (std::abs(tan->kets[0].norm[kSrsi][x]) > 1e-10) {
      one_sided = std::max(one_sided, std::abs(tan->transition_dot[1][kSrsi][x]));
    }
  }
  EXPECT_GT(one_sided, 1e-6);
  std::cout << std::format("[ nevpt2   ] qd hvp refusal: ket 1's {} empty Srsi tuples, largest "
                           "one-sided |Ṫ| {:.3e}\n",
                           empty.size(), one_sided);

  // Alone, or with itself as partner, accepted.
  for (const bool partnered : {false, true}) {
    const auto accepted = run_hvp(res, blocks, pairs, w, u,
                                  partnered ? std::span<const nevpt2::EnergyDirection>(u)
                                            : std::span<const nevpt2::EnergyDirection>{});
    ASSERT_TRUE(accepted) << accepted.error().message;
    std::int64_t populated = 0;
    for (const std::int64_t x : accepted->first.populated.front()[1]) {
      populated += x;
    }
    EXPECT_GT(populated, 0);
    std::cout << std::format("[ nevpt2   ] qd hvp refusal: {} accepted, ket 1 {} populated, "
                             "defect {:.2e}, coupling defect {:.2e}\n",
                             partnered ? "u with itself" : "u alone", populated,
                             accepted->first.defect.front(),
                             accepted->first.coupling_defect.front());
  }
  // Against an independent partner, refused.
  const auto refused = run_hvp(res, blocks, pairs, w, u, v);
  ASSERT_FALSE(refused);
  const std::string &message = refused.error().message;
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::Numerical) << message;
  for (const std::string_view part : {"coupling (", "class ", "directions ("}) {
    EXPECT_NE(message.find(part), std::string::npos) << message;
  }
  std::cout << std::format("[ nevpt2   ] qd hvp refused: {}\n", message);
}

// The A1/B2 fixture's blocks: its tree's DF-SA-CASSCF converged tightly
// (the response reads H c_r = E_r c_r), lambda^x, and t_x for every
// Cartesian coordinate, combined into the fixture's C2v coordinates; then
// per pair of vibrations (u, v) the product along u with v as partner. A
// refusal names its coupling's states and the directions, which are the
// block; the refused blocks must be the fixture's expected ones.
TEST(Nevpt2QdHvpEmpty, A1B2FixtureRefusesTheExpectedBlocks) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "qd_nevpt2_hessian" /
                            "h2nb_sto3g_sa2.json");
  ASSERT_TRUE(read.has_value()) << read.error().message;
  const fb::json &doc = *read;
  const std::vector<double> weights = doc.at("weights").get<std::vector<double>>();
  const auto ncas = doc.at("ncas").get<std::int32_t>();
  const auto nelecas = doc.at("nelecas").get<std::int32_t>();
  const auto labels = doc.at("coordinates").at("labels").get<std::vector<std::string>>();
  const auto vectors = doc.at("coordinates").at("vectors").get<std::vector<std::vector<double>>>();
  std::set<std::pair<std::string, std::string>> expected;
  for (const fb::json &block : doc.at("expected_refusals").at("sc")) {
    const auto c = block.at("coordinates").get<std::vector<std::string>>();
    expected.emplace(std::min(c[0], c[1]), std::max(c[0], c[1]));
  }

  const fb::test::ScratchDir dir("nevpt2_qd_hvp_a1b2");
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
  for (const double w : weights) {
    w_list += std::format("{}{:.17g}", w_list.empty() ? "" : ", ", w);
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
  const auto t = fb::hessian::df_qd_nevpt2_tangents(
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
  ASSERT_TRUE(t) << t.error().message;

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
  auto tangent = nevpt2::qd_builders_tangent(
      res, *strings, *strings, static_cast<std::int32_t>(naux), static_cast<std::int32_t>(nmo),
      static_cast<std::int32_t>(nmo), fit.b.data(), solution.c.data(), h.data(), cas->ncore,
      cas->ncas, weights, roots.data(), static_cast<std::int32_t>(vib.size()), h_dot.data(),
      b_dot.data(), c_dot.data());
  ASSERT_TRUE(tangent) << tangent.error().message;
  const nevpt2::MoBlocks &blocks = tangent->value.shared.blocks;
  const std::vector<nevpt2::EnergyInputs> pairs = tangent->value.inputs();
  const auto direction = [&](const std::size_t q) {
    std::vector<nevpt2::EnergyDirection> out;
    for (const nevpt2::EnergyInputs &in : tangent->tangents[q].inputs()) {
      out.push_back({.in = in, .blocks = &tangent->tangents[q].shared.blocks});
    }
    return out;
  };
  const std::vector<double> w = general_w(nroots);
  std::set<std::pair<std::string, std::string>> refused;
  for (std::size_t a = 0; a < vib.size(); ++a) {
    for (std::size_t bq = a; bq < vib.size(); ++bq) {
      const auto got = run_hvp(res, blocks, pairs, w, direction(a), direction(bq));
      const std::string &u = labels[vib[a]];
      const std::string &v = labels[vib[bq]];
      if (!got) {
        EXPECT_EQ(got.error().kind, fb::ErrorKind::Numerical) << got.error().message;
        refused.emplace(std::min(u, v), std::max(u, v));
        std::cout << std::format("[ nevpt2   ] qd hvp A1/B2 ({}, {}): refused: {}\n", u, v,
                                 got.error().message);
        continue;
      }
      std::int64_t populated = 0;
      for (const auto &ket : got->first.populated.front()) {
        populated += std::ranges::fold_left(ket, std::int64_t{0}, std::plus{});
      }
      std::cout << std::format("[ nevpt2   ] qd hvp A1/B2 ({}, {}): accepted, {} empty tuples "
                               "populated, defect {:.2e}, coupling defect {:.2e}\n",
                               u, v, populated, got->first.defect.front(),
                               got->first.coupling_defect.front());
    }
  }
  EXPECT_EQ(refused, expected);
}

} // namespace
