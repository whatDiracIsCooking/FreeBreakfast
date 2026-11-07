// GPU cases for nevpt2::sc_effective_hamiltonian_tangent, the SC-QD-NEVPT2
// effective Hamiltonian's forward tangent, against qd_sc_adjoint's bars by
// duality: along a block of random tangents of (h, B, every root's c), each
// column's Σ_MN W_MN Ċ_MN must be <h̄, ḣ> + <Λ, Ḃ> + Σ_r <y_r, ċ_r> to 1e-10
// of the largest of those summands (1e-9 on h2co_skew_ccpvdz_sa3, below), at a general symmetric W (off-diagonal,
// trace != 1, a negative entry). qd_sc_adjoint is ladder-validated
// (nevpt2_gradient_gpu.cpp's Nevpt2QdAdjoint), so duality proves the
// tangent; no ladder of qd_sc_energy is run here.
//
// The tangent reads EnergyInputs, so the chain from (h, B, c) is built
// here, as qd_sc_energy builds its inputs: each root's builders and their
// tangents by builders_tangent; the shared fields its, the orbital energies
// the weights' average of every root's (ε is affine in γ, and γ_SA is the
// weights' average); each off-diagonal pair's transition RDMs by the bra/ket
// nevpt2::rdms, and their tangents the same at the dotted bra plus at the
// dotted ket (Eq. qd-rdm2 of docs/nevpt2/nevpt2.tex). Each of those parts,
// and the diagonal's digests at the dotted (tu|vw), is its own direction in
// the block too, and must be visible: above ten bars in some column.
//
// The points are this tree's DF-SA-CASSCF of data/qd_nevpt2/'s
// h8_sto3g_sa3 (three roots, near-degenerate), h2co_ccpvdz_sa2 (C2v, 1A1
// and 1A2: every coupling and 4364 of the kets' tuples vanish by symmetry,
// so the guard is crossed) and h2co_skew_ccpvdz_sa3 (C1, three roots),
// converged loosely: duality holds at any point, and the symmetry, which
// empties the tuples, is exact at every iterate. The guarded tuples'
// transition tangent is printed, not gated: on h2co_ccpvdz_sa2 the same
// tuples are empty for both roots, so it vanishes at first order (#962
// gates it on a fixture where a tuple is empty for one ket alone).

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

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

// The issue's duality bar, relative to the column's largest summand.
constexpr double kDuality = 1e-10;
// h2co_skew_ccpvdz_sa3's: its qd_sc_adjoint bars repeat run to run only to
// 9e-10 of their largest element (src/nevpt2/energy/README.md, "The SC-QD
// reverse pass", Conditioning: root 2's tuples with own norms near 1e-12),
// so the oracle's floor, to one figure, is the bar there.
constexpr double kSkewDuality = 1e-9;
// The visibility rule: a part above this many bars in some column.
constexpr double kVisible = 10.0;
// Between the tangent's elements and qd_sc_adjoint's: the same inputs, the
// RDMs' dm1 and dm2 rdms' own here and dm3's traces there.
constexpr double kSameElements = 1e-12;
// The CASSCF's bar: loose, since no stationarity is used.
constexpr double kConvTolGrad = 1e-6;
// B_dot a tenth the length of the others, as the ladders' directions.
constexpr double kBScale = 0.1;
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";

constexpr std::array<const char *, 8> kClassNames{"Sr",   "Si",  "Sijrs", "Sijr",
                                                  "Srsi", "Srs", "Sij",   "Sir"};

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
// registry, as nevpt2_gradient_gpu.cpp's run_qd_casscf, at kConvTolGrad.
Point sa_casscf(const std::string &name) {
  const fb::json &entry = qd_case(name);
  Point p{.weights = entry.at("weights").get<std::vector<double>>()};
  EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_qd_tangent");
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

// One column of the block: (h_dot, B_dot) in the MO basis, h_dot symmetric
// and B_dot symmetric in its pair, and every root's c_dot, root-major; a
// field the column does not move is zero.
struct Column {
  std::string name;
  std::vector<double> h;
  std::vector<double> b;
  std::vector<double> c;
};

// Seeded columns: h alone, B alone, each root's c alone, and all at once.
std::vector<Column> columns(const Point &p) {
  const auto nmo = static_cast<std::size_t>(p.nao);
  const auto naux = static_cast<std::size_t>(p.naux);
  const std::size_t nroots = p.weights.size();
  std::mt19937_64 engine(954);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  const auto h_dot = [&] {
    std::vector<double> h(nmo * nmo);
    for (std::size_t q = 0; q < nmo; ++q) {
      for (std::size_t r = 0; r <= q; ++r) {
        h[r + (nmo * q)] = h[q + (nmo * r)] = uniform(engine);
      }
    }
    return h;
  };
  const auto b_dot = [&] {
    std::vector<double> b(naux * nmo * nmo);
    for (std::size_t q = 0; q < nmo; ++q) {
      for (std::size_t r = 0; r <= q; ++r) {
        for (std::size_t l = 0; l < naux; ++l) {
          b[l + (naux * (r + (nmo * q)))] = b[l + (naux * (q + (nmo * r)))] =
              kBScale * uniform(engine);
        }
      }
    }
    return b;
  };
  const auto c_dot = [&](const std::size_t root) {
    std::vector<double> c(p.vectors.size(), 0.0);
    const std::size_t ndet = p.vectors.size() / nroots;
    for (std::size_t i = 0; i < ndet; ++i) {
      c[(root * ndet) + i] = uniform(engine);
    }
    return c;
  };
  const std::vector<double> no_h(nmo * nmo, 0.0);
  const std::vector<double> no_b(naux * nmo * nmo, 0.0);
  const std::vector<double> no_c(p.vectors.size(), 0.0);
  std::vector<Column> out;
  out.push_back({.name = "h", .h = h_dot(), .b = no_b, .c = no_c});
  out.push_back({.name = "B", .h = no_h, .b = b_dot(), .c = no_c});
  for (std::size_t r = 0; r < nroots; ++r) {
    out.push_back({.name = std::format("c_{}", r), .h = no_h, .b = no_b, .c = c_dot(r)});
  }
  Column all{.name = "all", .h = h_dot(), .b = b_dot(), .c = no_c};
  for (std::size_t r = 0; r < nroots; ++r) {
    const std::vector<double> c = c_dot(r);
    for (std::size_t i = 0; i < c.size(); ++i) {
      all.c[i] += c[i];
    }
  }
  out.push_back(std::move(all));
  return out;
}

// A general symmetric W: seeded, off-diagonal, trace != 1, with a negative
// entry, every one of which is checked.
std::vector<double> general_w(const std::size_t n) {
  std::mt19937_64 engine(9540);
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

// The parts of a column's pair tangents, each a direction of its own: the
// whole, the off-diagonal RDMs' ket side alone (rdms at c_M, ċ_N), their
// bra side alone (ċ_M, c_N), and the diagonal's digests at the dotted
// (tu|vw) alone (f3ca and f3ac of rdms at c_r over (tu|vw)'s tangent).
constexpr std::array<std::string_view, 4> kParts{"whole", "ket", "bra", "digest"};

// The duality on @p name, the tangent walked in batches of @p batch.
void expect_duality(const std::string &name, const std::int32_t batch, const double duality) {
  const Point p = sa_casscf(name);
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0);
  const std::size_t nroots = p.weights.size();
  const std::size_t n2 = nroots * nroots;
  const std::size_t ndet = p.vectors.size() / nroots;
  const auto nmo = static_cast<std::size_t>(p.nao);
  const auto na = static_cast<std::size_t>(p.ncas);
  const std::size_t na2 = na * na;
  const std::size_t na4 = na2 * na2;
  const std::size_t na6 = na4 * na2;
  auto strings = fb::ci::DeviceStringSet::create(res, p.ncas, p.nelecas / 2);
  ASSERT_TRUE(strings) << strings.error().message;
  const fb::Buffers::device<double> b = upload(res, p.b);
  const fb::Buffers::device<double> c = upload(res, p.c);
  const fb::Buffers::device<double> h = upload(res, p.h);
  const fb::Buffers::device<double> vectors = upload(res, p.vectors);

  const std::vector<Column> cols = columns(p);
  const std::size_t m = cols.size();
  std::vector<double> h_dots;
  std::vector<double> b_dots;
  for (const Column &col : cols) {
    h_dots.insert(h_dots.end(), col.h.begin(), col.h.end());
    b_dots.insert(b_dots.end(), col.b.begin(), col.b.end());
  }
  const fb::Buffers::device<double> h_dot = upload(res, h_dots);
  const fb::Buffers::device<double> b_dot = upload(res, b_dots);
  // Each root's builders and their tangents along every column, its own
  // c_dot (zero where the column does not move it).
  std::vector<nevpt2::BuildersTangent> roots;
  std::vector<fb::Buffers::device<double>> c_dots;
  for (std::size_t r = 0; r < nroots; ++r) {
    std::vector<double> mine;
    for (const Column &col : cols) {
      mine.insert(mine.end(), col.c.begin() + static_cast<std::ptrdiff_t>(r * ndet),
                  col.c.begin() + static_cast<std::ptrdiff_t>((r + 1) * ndet));
    }
    c_dots.push_back(upload(res, mine));
    auto got = nevpt2::builders_tangent(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(),
                                        c.data(), h.data(), p.ncore, p.ncas,
                                        vectors.data() + (r * ndet), static_cast<std::int32_t>(m),
                                        h_dot.data(), b_dot.data(), c_dots.back().data());
    ASSERT_TRUE(got) << got.error().message;
    roots.push_back(std::move(*got));
  }
  const nevpt2::BuilderOutputs &first = roots.front().value;

  std::deque<fb::Buffers::device<double>> arrays;
  // The weights' average of every root's @p field (the value, or column u's
  // tangent), on the device.
  const auto averaged = [&](const std::optional<std::size_t> u,
                            fb::Buffers::device<double> nevpt2::BuilderOutputs::*field) {
    const auto count = (first.*field).num_elements();
    std::vector<double> sum(count, 0.0);
    for (std::size_t r = 0; r < nroots; ++r) {
      const nevpt2::BuilderOutputs &o = u ? roots[r].tangents[*u] : roots[r].value;
      const std::vector<double> x = download(res, (o.*field).data(), count);
      for (std::size_t i = 0; i < count; ++i) {
        sum[i] += p.weights[r] * x[i];
      }
    }
    arrays.push_back(upload(res, sum));
    return static_cast<const double *>(arrays.back().data());
  };
  // Five zeroed RDM buffers, dm1, dm2, dm3, f3ca and f3ac.
  const auto rdm_buffers = [&] {
    std::array<double *, 5> out{};
    for (std::size_t k = 0; k < 5; ++k) {
      const std::size_t count = k == 0 ? na2 : (k == 1 ? na4 : na6);
      arrays.emplace_back(upload(res, std::vector<double>(count, 0.0)));
      out[k] = arrays.back().data();
    }
    return out;
  };

  // The pairs: the shared fields root 0's, the orbital energies averaged,
  // each root's own RDMs its builders', each transition's rdms'.
  nevpt2::EnergyInputs shared = first.inputs();
  shared.e_core = averaged(std::nullopt, &nevpt2::BuilderOutputs::e_core);
  shared.e_virt = averaged(std::nullopt, &nevpt2::BuilderOutputs::e_virt);
  std::vector<nevpt2::EnergyInputs> pairs(n2, shared);
  for (std::size_t i = 0; i < nroots; ++i) {
    for (std::size_t j = 0; j < nroots; ++j) {
      nevpt2::EnergyInputs &in = pairs[(i * nroots) + j];
      if (i == j) {
        const nevpt2::EnergyInputs own = roots[i].value.inputs();
        in.dm1 = own.dm1;
        in.dm2 = own.dm2;
        in.dm3 = own.dm3;
        in.f3ca = own.f3ca;
        in.f3ac = own.f3ac;
        continue;
      }
      const auto [dm1, dm2, dm3, f3ca, f3ac] = rdm_buffers();
      const auto got = nevpt2::rdms(res, *strings, *strings, vectors.data() + (i * ndet),
                                    vectors.data() + (j * ndet), first.eri.data(), dm1, dm2, dm3,
                                    nullptr, nullptr);
      ASSERT_TRUE(got) << got.error().message;
      in.dm1 = dm1;
      in.dm2 = dm2;
      in.dm3 = dm3;
      in.f3ca = nullptr;
      in.f3ac = nullptr;
    }
  }

  // The directions, [(part m + u) n² + pair]: the whole moves column u's
  // integrals, orbital energies and B too, each other part its RDMs alone.
  std::vector<nevpt2::EnergyDirection> directions;
  for (const std::string_view part : kParts) {
    for (std::size_t u = 0; u < m; ++u) {
      const nevpt2::BuilderOutputs &t0 = roots.front().tangents[u];
      const bool whole = part == "whole";
      nevpt2::EnergyInputs common;
      if (whole) {
        common = t0.inputs();
        common.e_core = averaged(u, &nevpt2::BuilderOutputs::e_core);
        common.e_virt = averaged(u, &nevpt2::BuilderOutputs::e_virt);
        common.dm1 = common.dm2 = common.dm3 = common.f3ca = common.f3ac = nullptr;
      }
      for (std::size_t i = 0; i < nroots; ++i) {
        for (std::size_t j = 0; j < nroots; ++j) {
          nevpt2::EnergyInputs in = common;
          const double *const ci = vectors.data() + (i * ndet);
          const double *const cj = vectors.data() + (j * ndet);
          const double *const di = c_dots[i].data() + (u * ndet);
          const double *const dj = c_dots[j].data() + (u * ndet);
          if (i == j && whole) {
            const nevpt2::EnergyInputs own = roots[i].tangents[u].inputs();
            in.dm1 = own.dm1;
            in.dm2 = own.dm2;
            in.dm3 = own.dm3;
            in.f3ca = own.f3ca;
            in.f3ac = own.f3ac;
          } else if (i == j && part == "digest") {
            // f3 is linear in (tu|vw): rdms at its tangent is the digests'.
            const auto [dm1, dm2, dm3, f3ca, f3ac] = rdm_buffers();
            const auto got = nevpt2::rdms(res, *strings, *strings, ci,
                                          roots[i].tangents[u].eri.data(), dm3, f3ca, f3ac);
            ASSERT_TRUE(got) << got.error().message;
            in.f3ca = f3ca;
            in.f3ac = f3ac;
          } else if (i != j && part != "digest") {
            const auto [dm1, dm2, dm3, f3ca, f3ac] = rdm_buffers();
            if (part != "bra") {
              const auto got = nevpt2::rdms(res, *strings, *strings, ci, dj, first.eri.data(), dm1,
                                            dm2, dm3, nullptr, nullptr);
              ASSERT_TRUE(got) << got.error().message;
            }
            if (part != "ket") {
              const auto [b1, b2, b3, bf1, bf2] = rdm_buffers();
              const auto got = nevpt2::rdms(res, *strings, *strings, di, cj, first.eri.data(), b1,
                                            b2, b3, nullptr, nullptr);
              ASSERT_TRUE(got) << got.error().message;
              for (const auto &[into, from, count] :
                   {std::tuple{dm1, b1, na2}, std::tuple{dm2, b2, na4}, std::tuple{dm3, b3, na6}}) {
                std::vector<double> x = download(res, into, count);
                const std::vector<double> y = download(res, from, count);
                for (std::size_t k = 0; k < count; ++k) {
                  x[k] += y[k];
                }
                ASSERT_EQ(wwr::wwrMemcpyAsync(into, x.data(), sizeof(double) * count,
                                              wwr::wwrMemcpyHostToDevice, res.stream()),
                          wwr::wwrSuccess);
                ASSERT_TRUE(res.sync().has_value());
              }
            }
            in.dm1 = dm1;
            in.dm2 = dm2;
            in.dm3 = dm3;
          }
          directions.push_back({.in = in, .blocks = whole ? &t0.blocks : nullptr});
        }
      }
    }
  }

  const auto got =
      nevpt2::sc_effective_hamiltonian_tangent(res, first.blocks, pairs, batch, directions);
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->total_dot.size(), kParts.size() * m);

  const std::vector<double> w = general_w(nroots);
  const auto adj =
      nevpt2::qd_sc_adjoint(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(),
                            h.data(), p.ncore, p.ncas, p.weights, vectors.data(), w, 0);
  ASSERT_TRUE(adj) << adj.error().message;
  ASSERT_EQ(adj->elements.size(), n2);
  for (std::size_t q = 0; q < n2; ++q) {
    EXPECT_NEAR(got->elements[q].total, adj->elements[q].total, kSameElements) << name << " " << q;
  }
  const std::vector<double> h_bar = download(res, adj->h_bar.data(), nmo * nmo);
  const std::vector<double> lambda =
      download(res, adj->lambda.data(), static_cast<std::size_t>(p.naux) * nmo * nmo);
  const std::vector<double> y = download(res, adj->y.data(), p.vectors.size());

  // Column u's Σ W Ċ for part @p k.
  const auto weighted = [&](const std::size_t k, const std::size_t u) {
    return dot(w, got->total_dot[(k * m) + u]);
  };
  std::array<double, kParts.size()> visible{};
  std::vector<double> bars;
  for (std::size_t u = 0; u < m; ++u) {
    std::vector<double> summands{dot(h_bar, cols[u].h), dot(lambda, cols[u].b)};
    for (std::size_t r = 0; r < nroots; ++r) {
      summands.push_back(dot(std::span<const double>(y).subspan(r * ndet, ndet),
                             std::span<const double>(cols[u].c).subspan(r * ndet, ndet)));
    }
    const double predicted = std::accumulate(summands.begin(), summands.end(), 0.0);
    double scale = 0.0;
    for (const double x : summands) {
      scale = std::max(scale, std::abs(x));
    }
    const double bar = duality * scale;
    bars.push_back(bar);
    const double tangent = weighted(0, u);
    EXPECT_LE(std::abs(tangent - predicted), bar)
        << name << " " << cols[u].name << ": Σ W Ċ " << tangent << ", <bars, u> " << predicted;
    std::cout << std::format("[ nevpt2   ] qd tangent {} batch {} {}: Σ W Ċ {:+.12e}, <bars, u> "
                             "{:+.12e}, miss / largest summand {:.2e}\n",
                             name, batch, cols[u].name, tangent, predicted,
                             std::abs(tangent - predicted) / scale);
    for (std::size_t k = 1; k < kParts.size(); ++k) {
      visible[k] = std::max(visible[k], std::abs(weighted(k, u)) / bar);
    }
  }
  for (std::size_t k = 1; k < kParts.size(); ++k) {
    EXPECT_GT(visible[k], kVisible) << name << ": part " << kParts[k];
    std::cout << std::format("[ nevpt2   ] qd tangent {}: part {} at most {:.2e} bars\n", name,
                             kParts[k], visible[k]);
  }

  // The guarded tuples: each coupling's transition tangent there, against
  // the column's bar.
  std::size_t guarded = 0;
  double largest = 0.0;
  std::string per_class;
  for (std::size_t ket = 0; ket < nroots; ++ket) {
    for (std::size_t k = 0; k < 8; ++k) {
      const std::vector<std::int64_t> &empty = got->empty[ket][k];
      guarded += empty.size();
      per_class += std::format(" {}:{} {}", ket, kClassNames[k], empty.size());
      for (std::size_t bra = 0; bra < nroots; ++bra) {
        if (bra == ket) {
          continue;
        }
        const std::vector<double> &t_dot = got->transition_dot[(bra * nroots) + ket][k];
        for (const std::int64_t t : empty) {
          for (std::size_t u = 0; u < m; ++u) {
            largest = std::max(
                largest,
                std::abs(t_dot[(static_cast<std::size_t>(t) * kParts.size() * m) + u]) / bars[u]);
          }
        }
      }
    }
  }
  std::cout << std::format("[ nevpt2   ] qd tangent {}: {} guarded tuples,{}; largest Ṫ there "
                           "{:.2e} bars\n",
                           name, guarded, per_class, largest);
  if (name == "h2co_ccpvdz_sa2") {
    EXPECT_GT(guarded, 0U) << name;
  }
}

class Nevpt2QdTangent
    : public ::testing::TestWithParam<std::tuple<std::string, std::int32_t, double>> {};

// h8_sto3g_sa3 and h2co_skew_ccpvdz_sa3 whole; h2co_ccpvdz_sa2, the case
// with guarded tuples, in batches of four (29 virtuals: a partial last
// slab). The skew case is 1.4-2.5 s on the gfx1200, so the slow tier's;
// h8_sto3g_sa3 (0.9 s warm) is the family's representative.
TEST_P(Nevpt2QdTangent, DualToQdScAdjoint) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("h8_sto3g_sa3");
  const auto [name, batch, duality] = GetParam();
  if (name == "h2co_skew_ccpvdz_sa3") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  expect_duality(name, batch, duality);
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdTangent,
                         ::testing::Values(std::tuple{std::string{"h8_sto3g_sa3"}, 0, kDuality},
                                           std::tuple{std::string{"h2co_ccpvdz_sa2"}, 4, kDuality},
                                           std::tuple{std::string{"h2co_skew_ccpvdz_sa3"}, 0,
                                                      kSkewDuality}),
                         [](const auto &info) { return std::get<0>(info.param); });

} // namespace
