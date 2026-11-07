// GPU cases for nevpt2::qd_builders_tangent, the QD builders' forward
// tangent over a state average (the shared outputs at γ^SA, f^SA's core and
// virtual blocks, every ordered pair's RDMs), by duality, along one block of
// columns over (h, B, every root's c): h alone, B alone, each root's c alone,
// each pure rotation of two roots (ċ_r = c_s, ċ_s = -c_r), and all at once.
//
// - Against qd_sc_adjoint's builder bars at a general symmetric W
//   (QdBuilderBars, the bars its walk puts on what it reads): each column's
//   tangent paired with them must be <h̄, ḣ> + <Λ, Ḃ> + Σ_r <y_r, ċ_r> of the
//   same pass to 1e-10 of the largest of those summands. The same bars on
//   both sides, so the walk's own conditioning (src/nevpt2/energy/README.md,
//   "The SC-QD reverse pass", Conditioning) cancels: the duality is the
//   builders' alone. qd_sc_adjoint is ladder-validated
//   (nevpt2_gradient_gpu.cpp's Nevpt2QdAdjoint), so duality proves the
//   tangent. Visible, above ten bars in some column: the transition RDMs'
//   tangents, f^SA's (the orbital energies the walk reads), and on the
//   unequal weights γ^SA's θ term (a rotation's shared part).
// - Against zc_adjoint at the SA weights: <z, ḟ^SA> over a random symmetric z
//   on the two blocks, the off-diagonal f the walk never reads, to the same
//   bar, and the θ term visible in it.
// - γ^SA's tangent along a rotation against Eq. qd-gsa of
//   docs/nevpt2/nevpt2.tex, (w_r - w_s)(γ^(sr) + γ^(rs)), from the value's
//   own transition RDMs; zero at equal weights.
//
// The points are this tree's DF-SA-CASSCF of data/qd_nevpt2/'s
// h2co_skew_ccpvdz_sa3 (C1, three roots at equal weights) and
// h2co_skew_ccpvdz_sa3_weighted (the same at 0.4, 0.4, 0.2: rotations of
// roots 0 and 1 stay blind, those with root 2 do not), converged loosely:
// duality holds at any point.

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
// The visibility rule: a part above this many bars in some column.
constexpr double kVisible = 10.0;
// Between two builds of the same sums: the value's elements against
// qd_sc_adjoint's, <z, f^SA> against Φ, and γ^SA's tangent against
// Eq. qd-gsa, relative to the summands (the elements absolute, as
// nevpt2_qd_tangent_gpu.cpp's).
constexpr double kSameSums = 1e-12;
// The CASSCF's bar: loose, since no stationarity is used.
constexpr double kConvTolGrad = 1e-6;
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

double dot(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  return std::inner_product(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(b.size()), b.begin(),
                            0.0);
}

double max_abs(std::span<const double> v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
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
// registry, as nevpt2_qd_tangent_gpu.cpp's sa_casscf, at kConvTolGrad.
Point sa_casscf(const std::string &name) {
  const fb::json &entry = qd_case(name);
  Point p{.weights = entry.at("weights").get<std::vector<double>>()};
  EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_qd_builders_tangent");
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
// field the column does not move is zero. A rotation names its roots.
struct Column {
  std::string name;
  std::vector<double> h;
  std::vector<double> b;
  std::vector<double> c;
  std::optional<std::pair<std::size_t, std::size_t>> rotation;
};

// Seeded columns: h alone, B alone, each root's c alone (a generic vector,
// along the other roots too), each rotation (r, s), r < s, and all at once.
std::vector<Column> columns(const Point &p) {
  const auto nmo = static_cast<std::size_t>(p.nao);
  const auto naux = static_cast<std::size_t>(p.naux);
  const std::size_t nroots = p.weights.size();
  const std::size_t ndet = p.vectors.size() / nroots;
  std::mt19937_64 engine(958);
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
  for (std::size_t r = 0; r < nroots; ++r) {
    for (std::size_t s = r + 1; s < nroots; ++s) {
      Column rot{.name = std::format("rot_{}{}", r, s),
                 .h = no_h,
                 .b = no_b,
                 .c = no_c,
                 .rotation = std::pair{r, s}};
      for (std::size_t i = 0; i < ndet; ++i) {
        rot.c[(r * ndet) + i] = p.vectors[(s * ndet) + i];
        rot.c[(s * ndet) + i] = -p.vectors[(r * ndet) + i];
      }
      out.push_back(std::move(rot));
    }
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
  std::mt19937_64 engine(9580);
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

// <bars, tangent> over the fields @p fields of two BuilderOutputs, each
// field's length the bars' (empty where the walk reads nothing).
using Field = fb::Buffers::device<double> nevpt2::BuilderOutputs::*;
double paired(fb::DeviceResources &res, const nevpt2::BuilderOutputs &bars,
              const nevpt2::BuilderOutputs &tangent, std::span<const Field> fields) {
  double sum = 0.0;
  for (const Field f : fields) {
    const std::vector<double> bar = download(res, bars.*f);
    const std::vector<double> t = download(res, (tangent.*f).data(), bar.size());
    EXPECT_EQ((tangent.*f).num_elements(), bar.size());
    sum += dot(bar, t);
  }
  return sum;
}

// <bars, tangent> over every RDM of one pair.
double paired(fb::DeviceResources &res, const nevpt2::PairRdms &bars,
              const nevpt2::PairRdms &tangent) {
  double sum = 0.0;
  for (const auto f : {&nevpt2::PairRdms::dm1, &nevpt2::PairRdms::dm2, &nevpt2::PairRdms::dm3,
                       &nevpt2::PairRdms::f3ca, &nevpt2::PairRdms::f3ac}) {
    EXPECT_EQ((bars.*f).num_elements(), (tangent.*f).num_elements());
    sum += dot(download(res, bars.*f), download(res, tangent.*f));
  }
  return sum;
}

// The duality on @p name.
void expect_duality(const std::string &name) {
  const Point p = sa_casscf(name);
  ASSERT_FALSE(p.b.empty());
  fb::DeviceResources res(0);
  const std::size_t nroots = p.weights.size();
  const std::size_t n2 = nroots * nroots;
  const std::size_t ndet = p.vectors.size() / nroots;
  const auto nmo = static_cast<std::size_t>(p.nao);
  const auto nc = static_cast<std::size_t>(p.ncore);
  const std::size_t nocc = nc + static_cast<std::size_t>(p.ncas);
  const std::size_t nv = nmo - nocc;
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
  std::vector<double> c_dots;
  for (const Column &col : cols) {
    h_dots.insert(h_dots.end(), col.h.begin(), col.h.end());
    b_dots.insert(b_dots.end(), col.b.begin(), col.b.end());
    c_dots.insert(c_dots.end(), col.c.begin(), col.c.end());
  }
  const fb::Buffers::device<double> h_dot = upload(res, h_dots);
  const fb::Buffers::device<double> b_dot = upload(res, b_dots);
  const fb::Buffers::device<double> c_dot = upload(res, c_dots);
  auto got = nevpt2::qd_builders_tangent(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(),
                                         c.data(), h.data(), p.ncore, p.ncas, p.weights,
                                         vectors.data(), static_cast<std::int32_t>(m), h_dot.data(),
                                         b_dot.data(), c_dot.data());
  ASSERT_TRUE(got) << got.error().message;
  ASSERT_EQ(got->tangents.size(), m);
  ASSERT_EQ(got->value.pairs.size(), n2);

  const std::vector<double> w = general_w(nroots);
  std::optional<nevpt2::QdBuilderBars> kept;
  const auto adj =
      nevpt2::qd_sc_adjoint(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(),
                            h.data(), p.ncore, p.ncas, p.weights, vectors.data(), w, 0, 0, &kept);
  ASSERT_TRUE(adj) << adj.error().message;
  ASSERT_TRUE(kept.has_value());
  ASSERT_EQ(kept->bars.pairs.size(), n2);

  // The value is qd_sc_energy's: its pairs walked give qd_sc_adjoint's
  // elements.
  const auto elements =
      nevpt2::sc_effective_hamiltonian(res, got->value.shared.blocks, got->value.inputs(), 0);
  ASSERT_TRUE(elements) << elements.error().message;
  ASSERT_EQ(elements->size(), n2);
  for (std::size_t q = 0; q < n2; ++q) {
    EXPECT_NEAR((*elements)[q].total, adj->elements[q].total, kSameSums) << name << " " << q;
  }

  const std::vector<double> h_bar = download(res, adj->h_bar);
  const std::vector<double> lambda = download(res, adj->lambda);
  const std::vector<double> y = download(res, adj->y);
  const std::vector<double> walk_lambda = download(res, kept->lambda);
  // What the walk reads of the shared outputs: the orbital energies (f^SA's
  // diagonal, which γ^SA moves) apart from the integrals.
  constexpr std::array<Field, 2> kEpsilon{&nevpt2::BuilderOutputs::e_core,
                                          &nevpt2::BuilderOutputs::e_virt};
  constexpr std::array<Field, 5> kIntegrals{
      &nevpt2::BuilderOutputs::h1e, &nevpt2::BuilderOutputs::h2e, &nevpt2::BuilderOutputs::h1e_v_sr,
      &nevpt2::BuilderOutputs::h1e_v_si, &nevpt2::BuilderOutputs::h1e_v_sir};
  // The summands of column u's <bars, u>: h, B and each root's c.
  const auto summands = [&](const std::span<const double> hb, const std::span<const double> lb,
                            const std::span<const double> yb, const std::size_t u) {
    std::vector<double> out{dot(hb, cols[u].h), dot(lb, cols[u].b)};
    for (std::size_t r = 0; r < nroots; ++r) {
      out.push_back(dot(yb.subspan(r * ndet, ndet),
                        std::span<const double>(cols[u].c).subspan(r * ndet, ndet)));
    }
    return out;
  };
  const auto unequal = [&](const Column &col) {
    return col.rotation &&
           std::abs(p.weights[col.rotation->first] - p.weights[col.rotation->second]) > 1e-12;
  };
  const bool weighted = std::ranges::any_of(cols, unequal);

  // The walk's duality, its parts per column.
  double epsilon_visible = 0.0;
  double transition_visible = 0.0;
  double theta_visible = 0.0;
  double blind = 0.0;
  for (std::size_t u = 0; u < m; ++u) {
    const nevpt2::QdBuilderOutputs &t = got->tangents[u];
    const double epsilon = paired(res, kept->bars.shared, t.shared, kEpsilon);
    const double integrals = paired(res, kept->bars.shared, t.shared, kIntegrals);
    double own = 0.0;
    double transition = 0.0;
    for (std::size_t q = 0; q < n2; ++q) {
      (q % (nroots + 1) == 0 ? own : transition) += paired(res, kept->bars.pairs[q], t.pairs[q]);
    }
    const double slabs = dot(walk_lambda, cols[u].b);
    const double tangent = epsilon + integrals + own + transition + slabs;
    const std::vector<double> terms = summands(h_bar, lambda, y, u);
    const double predicted = std::accumulate(terms.begin(), terms.end(), 0.0);
    const double bar = kDuality * max_abs(terms);
    EXPECT_LE(std::abs(tangent - predicted), bar)
        << name << " " << cols[u].name << ": <bars, tangent> " << tangent << ", <bars, u> "
        << predicted;
    std::cout << std::format(
        "[ nevpt2   ] qd builders tangent {} {}: <bars, tangent> {:+.12e}, "
        "<bars, u> {:+.12e}, miss / largest summand {:.2e}; parts in bars: "
        "epsilon {:.2e}, integrals {:.2e}, own {:.2e}, transition {:.2e}, "
        "slabs {:.2e}\n",
        name, cols[u].name, tangent, predicted, std::abs(tangent - predicted) / max_abs(terms),
        std::abs(epsilon) / bar, std::abs(integrals) / bar, std::abs(own) / bar,
        std::abs(transition) / bar, std::abs(slabs) / bar);
    epsilon_visible = std::max(epsilon_visible, std::abs(epsilon) / bar);
    transition_visible = std::max(transition_visible, std::abs(transition) / bar);
    // A rotation moves the shared outputs through γ^SA alone: its θ term.
    if (cols[u].rotation) {
      const double shared = std::abs(epsilon + integrals) / bar;
      (unequal(cols[u]) ? theta_visible : blind) =
          std::max(unequal(cols[u]) ? theta_visible : blind, shared);
    }
  }
  EXPECT_GT(epsilon_visible, kVisible) << name << ": f^SA's tangent";
  EXPECT_GT(transition_visible, kVisible) << name << ": the transition RDMs' tangents";
  if (weighted) {
    EXPECT_GT(theta_visible, kVisible) << name << ": γ^SA's θ term";
  }
  std::cout << std::format("[ nevpt2   ] qd builders tangent {}: at most epsilon {:.2e}, "
                           "transition {:.2e}, θ {:.2e} bars; equal-weight rotations' shared "
                           "part {:.2e} bars\n",
                           name, epsilon_visible, transition_visible, theta_visible, blind);

  // γ^SA's tangent along each rotation: Eq. qd-gsa, at θ = 1.
  const auto na2 = static_cast<std::size_t>(p.ncas) * static_cast<std::size_t>(p.ncas);
  for (std::size_t u = 0; u < m; ++u) {
    if (!cols[u].rotation) {
      continue;
    }
    const auto [r, s] = *cols[u].rotation;
    const std::vector<double> sr = download(res, got->value.pairs[(s * nroots) + r].dm1);
    const std::vector<double> rs = download(res, got->value.pairs[(r * nroots) + s].dm1);
    const std::vector<double> gsa_dot = download(res, got->tangents[u].shared.dm1);
    ASSERT_EQ(gsa_dot.size(), na2);
    double miss = 0.0;
    for (std::size_t k = 0; k < na2; ++k) {
      miss =
          std::max(miss, std::abs(gsa_dot[k] - ((p.weights[r] - p.weights[s]) * (sr[k] + rs[k]))));
    }
    const double scale = std::max(max_abs(sr), max_abs(rs));
    EXPECT_LE(miss, kSameSums * scale) << name << " " << cols[u].name;
    std::cout << std::format("[ nevpt2   ] qd builders tangent {} {}: |γ̇^SA| {:.3e}, off Eq. "
                             "qd-gsa by {:.2e} of |γ^(sr)|\n",
                             name, cols[u].name, max_abs(gsa_dot), miss / scale);
  }

  // f^SA's blocks against zc_adjoint at the SA weights: the value against Φ,
  // and each column's <z, ḟ> against <bars, u>.
  std::mt19937_64 engine(9581);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> z(nmo * nmo, 0.0);
  for (const auto [p0, count] : {std::pair{std::size_t{0}, nc}, std::pair{nocc, nv}}) {
    for (std::size_t j = 0; j < count; ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
        z[p0 + i + (nmo * (p0 + j))] = z[p0 + j + (nmo * (p0 + i))] = uniform(engine);
      }
    }
  }
  // <z, f> over f's two blocks, and the largest |z f| summand.
  const auto z_dot_f = [&](const nevpt2::BuilderOutputs &o) {
    const std::vector<double> fc = download(res, o.f_core);
    const std::vector<double> fv = download(res, o.f_virt);
    double sum = 0.0;
    double largest = 0.0;
    for (const auto &[f, p0, count] :
         {std::tuple{&fc, std::size_t{0}, nc}, std::tuple{&fv, nocc, nv}}) {
      EXPECT_EQ(f->size(), count * count);
      for (std::size_t j = 0; j < count; ++j) {
        for (std::size_t i = 0; i < count; ++i) {
          const double term = z[p0 + i + (nmo * (p0 + j))] * (*f)[i + (count * j)];
          sum += term;
          largest += std::abs(term);
        }
      }
    }
    return std::pair{sum, largest};
  };
  const fb::Buffers::device<double> zd = upload(res, z);
  const auto zc =
      nevpt2::zc_adjoint(res, *strings, *strings, p.naux, p.nao, p.nao, b.data(), c.data(),
                         h.data(), p.ncore, p.ncas, p.weights, vectors.data(), zd.data());
  ASSERT_TRUE(zc) << zc.error().message;
  const auto [phi, phi_scale] = z_dot_f(got->value.shared);
  EXPECT_NEAR(phi, zc->phi, kSameSums * phi_scale) << name;
  const std::vector<double> zc_h = download(res, zc->h_bar);
  const std::vector<double> zc_lambda = download(res, zc->lambda);
  const std::vector<double> zc_y = download(res, zc->y);
  double zc_theta = 0.0;
  for (std::size_t u = 0; u < m; ++u) {
    const double tangent = z_dot_f(got->tangents[u].shared).first;
    const std::vector<double> terms = summands(zc_h, zc_lambda, zc_y, u);
    const double predicted = std::accumulate(terms.begin(), terms.end(), 0.0);
    const double bar = kDuality * max_abs(terms);
    EXPECT_LE(std::abs(tangent - predicted), bar)
        << name << " " << cols[u].name << ": <z, ḟ> " << tangent << ", <zc bars, u> " << predicted;
    std::cout << std::format("[ nevpt2   ] qd builders tangent {} {}: <z, ḟ^SA> {:+.12e}, "
                             "<zc bars, u> {:+.12e}, miss / largest summand {:.2e}\n",
                             name, cols[u].name, tangent, predicted,
                             std::abs(tangent - predicted) / max_abs(terms));
    if (unequal(cols[u])) {
      zc_theta = std::max(zc_theta, std::abs(tangent) / bar);
    }
  }
  if (weighted) {
    EXPECT_GT(zc_theta, kVisible) << name << ": γ^SA's θ term in f^SA";
  }
  std::cout << std::format("[ nevpt2   ] qd builders tangent {}: <z, f^SA> {:+.12e}, Φ {:+.12e}; "
                           "θ in f^SA at most {:.2e} bars\n",
                           name, phi, zc->phi, zc_theta);
}

class Nevpt2QdBuildersTangent : public ::testing::TestWithParam<std::string> {};

// The weighted case, whose rotations with root 2 carry the θ term, is the
// family's representative; the equal-weight one, the same cost, the slow
// tier's.
TEST_P(Nevpt2QdBuildersTangent, DualToQdScAdjoint) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("h2co_skew_ccpvdz_sa3_weighted");
  if (GetParam() == "h2co_skew_ccpvdz_sa3") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  expect_duality(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdBuildersTangent,
                         ::testing::Values(std::string{"h2co_skew_ccpvdz_sa3"},
                                           std::string{"h2co_skew_ccpvdz_sa3_weighted"}),
                         [](const auto &info) { return info.param; });

} // namespace
