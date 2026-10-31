// GPU cases for nevpt2.gradient: sc_energy and sc_adjoint on this tree's own
// DF-CASSCF of data/casscf/casscf_df.json's h2o_ccpvdz (CAS(4,4), one root,
// so the CASSCF's core and virtuals are the root's canonical ones), run
// through the real registry as pipeline_nevpt2_gpu.cpp runs it. sc_energy
// against PySCF's DF-NEVPT2 (data/nevpt2/nevpt2_df.json); then each adjoint
// against a Richardson ladder of sc_energy itself, with every input rebuilt
// from (h, B, C, c) at each point, the orbital energies included: <u, y>
// along a spin-pure, symmetric u perpendicular to c; Σ_{p>q} κ_pq A_pq along
// a random rotation of each block (core-core and virtual-virtual included),
// the orbitals not re-canonicalized; and <h_bar, C^T dh C> and
// <Λ, C^T dB C> along seeded integral perturbations.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.ci;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

constexpr std::string_view kCase = "h2o_ccpvdz";
// The fitting set both references were made over, compiled in.
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";
// pipeline_nevpt2_gpu.cpp's CASSCF convergence and its bar on E_corr against
// PySCF's, derived there from how closely our CASSCF reaches the reference's
// orbitals.
constexpr double kConvTolGrad = 1e-9;
constexpr double kTolerance = 1e-9;
// sc_adjoint walks the same forward as sc_energy; the einsums' atomics
// reassociate every sum.
constexpr double kSameEnergyTolerance = 1e-12;
// nevpt2_energy_gpu.cpp's ladder: central differences D(t) at t = t0 / 2^k,
// successive changes falling by a factor in (3.2, 4.8) over at least three
// halvings, and the Richardson extrapolant at that run's finest pair against
// the adjoint's prediction to 1e-7 relative.
constexpr double kLadderT0 = 1e-2;
constexpr std::int32_t kRungs = 10;
constexpr double kLadderTolerance = 1e-7;
// The B direction is a tenth the length of the others: uniform in [-1, 1)
// on every fitted element, at full length its ladder ran ratios 5.41 -1.03
// -2.03 -2.02 3.23 4.17 4.08 4.02 (RTX 3080) and its extrapolant missed by
// 3.0e-6 relative, still far from the t^2 regime at t = 2e-5. At a tenth, the
// prediction a tenth of the full one's to all its digits, the ratios settle
// at 4.000 and the miss is 3.6e-10. So the bar was right and the direction
// too long, as nevpt2_energy_gpu.cpp's kRdmScale found; the tolerance is
// unchanged.
constexpr double kBScale = 0.1;

const fb::json &reference(std::string_view dir, std::string_view file) {
  static std::map<std::string, fb::json, std::less<>> docs;
  const std::string key = std::format("{}/{}", dir, file);
  if (const auto it = docs.find(key); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / dir / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(key, read.has_value() ? std::move(*read) : fb::json{}).first->second;
}

const fb::json &reference_case(std::string_view dir, std::string_view file, std::string_view name) {
  for (const fb::json &entry : reference(dir, file).at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in " << dir << "/" << file;
  static const fb::json kNone;
  return kNone;
}

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> out(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), host.data(), host.size_bytes(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return out;
}

// The converged state, on the host: B as density_fit leaves it, C, the AO
// core Hamiltonian and the CI vector.
struct State {
  std::int32_t naux = 0;
  std::int32_t nao = 0;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;
  std::int32_t nelecas = 0;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> h;
  std::vector<double> vector;
};

// Our DF-CASSCF from the reference's start, run once per process.
const State &state() {
  static const State kState = [] {
    State s;
    const fb::json &start = reference_case("casscf", "casscf_df.json", kCase);
    EXPECT_EQ(start.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
    const fb::test::ScratchDir dir("nevpt2_gradient");
    std::string xyz =
        std::format("{}\n{}\n", start.at("atoms").size(), start.at("name").get<std::string>());
    for (const fb::json &atom : start.at("atoms")) {
      const auto coord = atom.at("coord").get<std::vector<double>>();
      xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                         coord[0] * fb::kAngstromPerBohr<double>,
                         coord[1] * fb::kAngstromPerBohr<double>,
                         coord[2] * fb::kAngstromPerBohr<double>);
    }
    EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
    std::string active;
    for (const std::int32_t a : start.at("active").get<std::vector<std::int32_t>>()) {
      active += std::format("{}{}", active.empty() ? "" : ", ", a);
    }
    const std::string toml =
        std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
                    "[casscf]\nconv_tol_grad = {:g}\nncas = {}\nnelecas = {}\nactive = [{}]\n"
                    "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                    start.at("charge").get<std::int32_t>(), start.at("basis").get<std::string>(),
                    kConvTolGrad, start.at("ncas").get<std::int32_t>(),
                    start.at("nelecas").get<std::int32_t>(), active, kDfAuxBasis);
    EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
    auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
    EXPECT_TRUE(calc.has_value()) << calc.error().message;
    if (!calc.has_value()) {
      return s;
    }
    fb::NodeResources node;
    const fb::Status ran = calc->run(node);
    EXPECT_TRUE(ran.has_value()) << ran.error().message;
    const auto &casscf = calc->get<fb::CasscfOutput>();
    const auto &eri = calc->get<fb::EriOutput>();
    const auto &h = calc->get<fb::CoreHamiltonianOutput>();
    EXPECT_TRUE(casscf.has_value() && eri.has_value() && h.has_value());
    if (!ran.has_value() || !casscf.has_value() || !eri.has_value() || !h.has_value()) {
      return s;
    }
    EXPECT_LT(casscf->gradient_norm, kConvTolGrad);
    EXPECT_EQ(casscf->energies.size(), 1U);
    const auto &df = std::get<fb::DensityFitted>(eri->eri);
    fb::DeviceResources &res = node.device(0);
    s.naux = df.naux;
    s.nao = static_cast<std::int32_t>(casscf->c.n());
    s.ncore = casscf->ncore;
    s.ncas = casscf->ncas;
    s.nelecas = casscf->nelecas;
    const auto nao = static_cast<std::size_t>(s.nao);
    s.b = download(res, df.b.data(), static_cast<std::size_t>(s.naux) * nao * nao);
    s.c = download(res, casscf->c.data(), nao * nao);
    s.h = download(res, h->h.data(), nao * nao);
    s.vector = casscf->vectors;
    return s;
  }();
  return kState;
}

// The state on a device, and its strings: one set for both spins.
struct DeviceState {
  fb::Buffers::device<double> b;
  fb::Buffers::device<double> c;
  fb::Buffers::device<double> h;
  fb::Buffers::device<double> vector;
  std::optional<fb::ci::DeviceStringSet> strings;
};

DeviceState upload_state(fb::DeviceResources &res, const State &s) {
  DeviceState d{.b = upload(res, s.b),
                .c = upload(res, s.c),
                .h = upload(res, s.h),
                .vector = upload(res, s.vector),
                .strings = std::nullopt};
  auto strings = fb::ci::DeviceStringSet::create(res, s.ncas, s.nelecas / 2);
  EXPECT_TRUE(strings.has_value()) << strings.error().message;
  if (strings.has_value()) {
    d.strings.emplace(std::move(*strings));
  }
  return d;
}

// sc_energy's E_corr at the state @p s, uploaded.
double energy(fb::DeviceResources &res, const State &s) {
  const DeviceState d = upload_state(res, s);
  const auto e = nevpt2::sc_energy(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                                   d.c.data(), d.h.data(), s.ncore, s.ncas, d.vector.data(), 0);
  EXPECT_TRUE(e.has_value()) << e.error().message;
  return e.has_value() ? e->total : std::nan("");
}

// The adjoint at @p s, on the host: h_bar, Lambda, y and A.
struct HostAdjoint {
  double total = 0.0;
  std::vector<double> h_bar;
  std::vector<double> lambda;
  std::vector<double> y;
  std::vector<double> x;
  std::vector<double> a;
};

HostAdjoint adjoint(fb::DeviceResources &res, const State &s) {
  const DeviceState d = upload_state(res, s);
  const auto got = nevpt2::sc_adjoint(res, *d.strings, *d.strings, s.naux, s.nao, s.nao, d.b.data(),
                                      d.c.data(), d.h.data(), s.ncore, s.ncas, d.vector.data(), 0);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  if (!got.has_value()) {
    return {};
  }
  const auto nmo = static_cast<std::size_t>(s.nao);
  return {.total = got->energies.total,
          .h_bar = download(res, got->h_bar.data(), nmo * nmo),
          .lambda = download(res, got->lambda.data(), static_cast<std::size_t>(s.naux) * nmo * nmo),
          .y = download(res, got->y.data(), s.vector.size()),
          .x = download(res, got->x.data(), nmo * nmo),
          .a = download(res, got->a.data(), nmo * nmo)};
}

// C = A B, column-major: A m x k, B k x n.
std::vector<double> matmul(std::span<const double> a, std::span<const double> b, std::size_t m,
                           std::size_t k, std::size_t n) {
  std::vector<double> c(m * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t l = 0; l < k; ++l) {
      const double blj = b[l + (k * j)];
      for (std::size_t i = 0; i < m; ++i) {
        c[i + (m * j)] += a[i + (m * l)] * blj;
      }
    }
  }
  return c;
}

// exp(K) for a small n x n K, by its Taylor series: |t K| <= 1e-2 |K| here.
std::vector<double> expm(std::span<const double> k, std::size_t n) {
  std::vector<double> out(n * n, 0.0);
  std::vector<double> term(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    out[i * (n + 1)] = 1.0;
    term[i * (n + 1)] = 1.0;
  }
  for (std::int32_t m = 1; m < 60; ++m) {
    term = matmul(term, k, n, n, n);
    double largest = 0.0;
    for (std::size_t i = 0; i < term.size(); ++i) {
      term[i] /= m;
      out[i] += term[i];
      largest = std::max(largest, std::abs(term[i]));
    }
    if (largest < 1e-20) {
      break;
    }
  }
  return out;
}

// C^T M C for an nao x nao M: the MO basis's view of an AO matrix.
std::vector<double> to_mo(const State &s, std::span<const double> m) {
  const auto n = static_cast<std::size_t>(s.nao);
  std::vector<double> ct(n * n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t mu = 0; mu < n; ++mu) {
      ct[p + (n * mu)] = s.c[mu + (n * p)];
    }
  }
  return matmul(matmul(ct, m, n, n, n), s.c, n, n, n);
}

double dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

// Central differences of @p at over the ladder, against @p predicted.
void expect_ladder(const std::string &name, const std::function<double(double)> &at,
                   const double predicted) {
  std::vector<double> d;
  for (std::int32_t k = 0; k < kRungs; ++k) {
    const double t = std::ldexp(kLadderT0, -k);
    d.push_back((at(t) - at(-t)) / (2.0 * t));
  }
  // ratio[k] = (D_k - D_{k+1}) / (D_{k+1} - D_{k+2}); the longest run of
  // ratios in (3.2, 4.8).
  std::int32_t best_end = -1;
  std::int32_t best_len = 0;
  std::int32_t len = 0;
  std::string ladder;
  for (std::int32_t k = 0; k + 2 < kRungs; ++k) {
    const double ratio = (d[k] - d[k + 1]) / (d[k + 1] - d[k + 2]);
    ladder += std::format(" {:.3f}", ratio);
    len = (ratio > 3.2 && ratio < 4.8) ? len + 1 : 0;
    if (len > best_len) {
      best_len = len;
      best_end = k;
    }
  }
  ASSERT_GE(best_len, 3) << name << " ratios:" << ladder;
  // The run's finest pair is (D_{end+1}, D_{end+2}).
  const auto fine = static_cast<std::size_t>(best_end + 2);
  const double richardson = ((4.0 * d[fine]) - d[fine - 1]) / 3.0;
  EXPECT_NEAR(richardson, predicted, kLadderTolerance * std::abs(predicted))
      << name << " ratios:" << ladder;
  std::cout << std::format("[ nevpt2   ] gradient ladder {}: predicted {:.12e}, Richardson "
                           "{:.12e}, relative {:.2e}, ratios{}\n",
                           name, predicted, richardson,
                           std::abs(richardson - predicted) / std::abs(predicted), ladder);
}

TEST(Nevpt2Gradient, EnergyMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const fb::json &want = reference_case("nevpt2", "nevpt2_df.json", kCase);
  const double e_corr = want.at("roots").at(0).at("e_corr").get<double>();
  const double got = energy(res, s);
  EXPECT_NEAR(got, e_corr, kTolerance);
  const HostAdjoint adj = adjoint(res, s);
  EXPECT_NEAR(adj.total, got, kSameEnergyTolerance);
  std::cout << std::format("[ nevpt2   ] {}: sc_energy {:.12f}, dE_corr {:+.2e}\n", kCase, got,
                           got - e_corr);
}

// <u, y> along u = P H_r c, normalized, with c(t) = (c + t u) / |c + t u|.
// H_r is spin-free, so u is in c's spin; and it is a random combination of
// operators that commute with the molecule's point group (the active block
// of C^T h C, its square and the active (tu|vw)), so u is in c's symmetry
// sector too. A direction outside either takes the hole-side metrics where
// they are not positive, and the SC finish has poles there.
TEST(Nevpt2Gradient, CiAdjointMatchesLadder) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const auto n = static_cast<std::size_t>(s.ncas);
  const auto nao = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  const auto nc = static_cast<std::size_t>(s.ncore);

  std::mt19937_64 engine(562);
  std::uniform_real_distribution<double> scale(0.5, 1.5);
  const std::vector<double> h_mo = to_mo(s, s.h);
  std::vector<double> h_act(n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      h_act[(t * n) + u] = h_mo[(nc + t) + (nao * (nc + u))];
    }
  }
  const std::vector<double> h_sq = matmul(h_act, h_act, n, n, n);
  const double s1 = scale(engine);
  const double s2 = scale(engine);
  const double s3 = scale(engine);
  std::vector<double> h_r(n * n);
  for (std::size_t k = 0; k < h_r.size(); ++k) {
    h_r[k] = (s1 * h_act[k]) + (s2 * h_sq[k]);
  }
  // (tu|vw) over the fitted B: B_L,tu = Σ_mn C_m,t C_n,u B_L,mn.
  std::vector<double> b_act(naux * n * n, 0.0);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      for (std::size_t mu = 0; mu < nao; ++mu) {
        for (std::size_t nu = 0; nu < nao; ++nu) {
          const double cc = s.c[mu + (nao * (nc + t))] * s.c[nu + (nao * (nc + u))];
          for (std::size_t l = 0; l < naux; ++l) {
            b_act[l + (naux * (t + (n * u)))] += cc * s.b[l + (naux * (mu + (nao * nu)))];
          }
        }
      }
    }
  }
  std::vector<double> eri_r(n * n * n * n);
  for (std::size_t tu = 0; tu < n * n; ++tu) {
    for (std::size_t vw = 0; vw < n * n; ++vw) {
      double sum = 0.0;
      for (std::size_t l = 0; l < naux; ++l) {
        sum += b_act[l + (naux * tu)] * b_act[l + (naux * vw)];
      }
      // Row-major (t, u, v, w); B_tu is symmetric, so tu's order is moot.
      eri_r[(tu * n * n) + vw] = s3 * sum;
    }
  }
  const DeviceState d = upload_state(res, s);
  auto sigma = fb::ci::Sigma::create(res, *d.strings, *d.strings);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  ASSERT_TRUE(sigma->set_integrals(res, 0.0, h_r, eri_r).has_value());
  fb::Buffers::device<double> hc(s.vector.size(), res.device());
  ASSERT_TRUE(sigma->apply(res, 1, d.vector.data(), hc.data()).has_value());
  std::vector<double> u = download(res, hc.data(), s.vector.size());
  const double overlap = dot(u, s.vector);
  for (std::size_t i = 0; i < u.size(); ++i) {
    u[i] -= overlap * s.vector[i];
  }
  const double norm = std::sqrt(dot(u, u));
  ASSERT_GT(norm, 1e-6);
  for (double &v : u) {
    v /= norm;
  }

  State moved = s;
  expect_ladder(
      "ci",
      [&](const double t) {
        for (std::size_t i = 0; i < u.size(); ++i) {
          moved.vector[i] = s.vector[i] + (t * u[i]);
        }
        const double length = std::sqrt(dot(moved.vector, moved.vector));
        for (double &v : moved.vector) {
          v /= length;
        }
        return energy(res, moved);
      },
      dot(u, adj.y));
}

// Σ_{p>q} κ_pq A_pq along C(t) = C exp(t κ), κ random and antisymmetric on
// one block, everything rebuilt from C(t) and nothing re-canonicalized.
struct Block {
  std::string_view name;
  // The two orbital spaces, each 0 core, 1 active, 2 virtual.
  std::int32_t row;
  std::int32_t col;
};

void PrintTo(const Block &b, std::ostream *os) {
  *os << b.name;
}

class Nevpt2GradientRotation : public ::testing::TestWithParam<Block> {};

TEST_P(Nevpt2GradientRotation, OrbitalSourceMatchesLadder) {
  FBF_REQUIRE_GPU();
  const Block &block = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::array<std::size_t, 4> first{0, static_cast<std::size_t>(s.ncore),
                                         static_cast<std::size_t>(s.ncore + s.ncas), nmo};

  std::mt19937_64 engine(562 + (3 * block.row) + block.col);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> kappa(nmo * nmo, 0.0);
  double predicted = 0.0;
  for (std::size_t p = first[block.row]; p < first[block.row + 1]; ++p) {
    for (std::size_t q = first[block.col]; q < first[block.col + 1]; ++q) {
      if (p <= q && block.row == block.col) {
        continue;
      }
      const double k = uniform(engine);
      kappa[p + (nmo * q)] = k;
      kappa[q + (nmo * p)] = -k;
      // Σ_{p>q} κ_pq A_pq, over whichever of (p, q), (q, p) is below.
      predicted += p > q ? k * adj.a[p + (nmo * q)] : -k * adj.a[q + (nmo * p)];
    }
  }

  State moved = s;
  expect_ladder(
      std::string{block.name},
      [&](const double t) {
        std::vector<double> tk(kappa);
        for (double &v : tk) {
          v *= t;
        }
        moved.c = matmul(s.c, expm(tk, nmo), nmo, nmo, nmo);
        return energy(res, moved);
      },
      predicted);
}

INSTANTIATE_TEST_SUITE_P(Blocks, Nevpt2GradientRotation,
                         ::testing::Values(Block{.name = "core_core", .row = 0, .col = 0},
                                           Block{.name = "active_active", .row = 1, .col = 1},
                                           Block{.name = "virtual_virtual", .row = 2, .col = 2},
                                           Block{.name = "active_core", .row = 1, .col = 0},
                                           Block{.name = "virtual_core", .row = 2, .col = 0},
                                           Block{.name = "virtual_active", .row = 2, .col = 1}),
                         [](const auto &info) { return std::string(info.param.name); });

// <h_bar, C^T dh C> along a seeded symmetric dh, and <Λ, C^T dB C> along a
// seeded dB symmetric in its AO pair, each alone, at fixed C and c.
class Nevpt2GradientIntegrals : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2GradientIntegrals, IntegralAdjointMatchesLadder) {
  FBF_REQUIRE_GPU();
  const std::string &which = GetParam();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const auto nao = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);

  std::mt19937_64 engine(which == "h" ? 5621 : 5622);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> dh(nao * nao, 0.0);
  std::vector<double> db(naux * nao * nao, 0.0);
  double predicted = 0.0;
  if (which == "h") {
    for (std::size_t mu = 0; mu < nao; ++mu) {
      for (std::size_t nu = 0; nu <= mu; ++nu) {
        dh[mu + (nao * nu)] = dh[nu + (nao * mu)] = uniform(engine);
      }
    }
    predicted = dot(adj.h_bar, to_mo(s, dh));
  } else {
    for (std::size_t mu = 0; mu < nao; ++mu) {
      for (std::size_t nu = 0; nu <= mu; ++nu) {
        for (std::size_t l = 0; l < naux; ++l) {
          db[l + (naux * (mu + (nao * nu)))] = db[l + (naux * (nu + (nao * mu)))] =
              kBScale * uniform(engine);
        }
      }
    }
    // C^T dB^L C per L, against Λ^L.
    std::vector<double> slice(nao * nao);
    for (std::size_t l = 0; l < naux; ++l) {
      for (std::size_t mn = 0; mn < nao * nao; ++mn) {
        slice[mn] = db[l + (naux * mn)];
      }
      const std::vector<double> mo = to_mo(s, slice);
      for (std::size_t pq = 0; pq < nao * nao; ++pq) {
        predicted += adj.lambda[l + (naux * pq)] * mo[pq];
      }
    }
  }

  State moved = s;
  expect_ladder(
      which,
      [&](const double t) {
        for (std::size_t k = 0; k < dh.size(); ++k) {
          moved.h[k] = s.h[k] + (t * dh[k]);
        }
        for (std::size_t k = 0; k < db.size(); ++k) {
          moved.b[k] = s.b[k] + (t * db[k]);
        }
        return energy(res, moved);
      },
      predicted);
}

INSTANTIATE_TEST_SUITE_P(Integrals, Nevpt2GradientIntegrals,
                         ::testing::Values(std::string{"h"}, std::string{"B"}),
                         [](const auto &info) { return info.param; });

// The orbital source term's parts: A is X's antisymmetric part, and
// h_bar and every Λ^L symmetric, as the AO half-transform assumes.
TEST(Nevpt2Gradient, SourceTermShapes) {
  FBF_REQUIRE_GPU();
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  fb::DeviceResources res(0);
  const HostAdjoint adj = adjoint(res, s);
  const auto nmo = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  double worst_a = 0.0;
  double worst_h = 0.0;
  double worst_lambda = 0.0;
  for (std::size_t p = 0; p < nmo; ++p) {
    for (std::size_t q = 0; q < nmo; ++q) {
      worst_a = std::max(
          worst_a, std::abs(adj.a[p + (nmo * q)] - (adj.x[p + (nmo * q)] - adj.x[q + (nmo * p)])));
      worst_h = std::max(worst_h, std::abs(adj.h_bar[p + (nmo * q)] - adj.h_bar[q + (nmo * p)]));
      for (std::size_t l = 0; l < naux; ++l) {
        worst_lambda = std::max(worst_lambda, std::abs(adj.lambda[l + (naux * (p + (nmo * q)))] -
                                                       adj.lambda[l + (naux * (q + (nmo * p)))]));
      }
    }
  }
  EXPECT_LT(worst_a, 1e-14);
  EXPECT_EQ(worst_h, 0.0);
  EXPECT_LT(worst_lambda, 1e-15);
}

} // namespace
