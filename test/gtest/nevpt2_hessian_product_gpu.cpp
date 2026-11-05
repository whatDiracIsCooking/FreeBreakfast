// GPU cases for nevpt2::sc_hessian_product: the tangent of sc_adjoint +
// zc_adjoint at fixed z^c, in the Rayleigh-normalized functional of c,
// along a block of two mixed directions (h_dot, B_dot, c_dot), against a
// Richardson ladder of those two adjoints themselves, every output field;
// each separately coded part above ten bars in some field, and the parts
// summing to the product; partners that leave the product as it is when no
// perturber is empty; and the refusals.
//
// The point is this tree's DF-CASSCF of data/nevpt2_hessian/'s
// h2co_skew_sto3g (C1, CAS(4,4), no SC perturber empty, the geometry #803
// chose for SC's smoothness), z^c its own zc_multipliers. The directions are
// AO perturbations of h and B, taken to the MO basis of C here, and a CI
// direction covariant with the CI vector's signs (c_dot_I = (r_I + 0.3) c_I,
// r symmetric in the two strings), so no gauge needs fixing: every field
// moves covariantly with the orbitals' and determinants' signs.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

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

constexpr std::string_view kCaseFile = "h2co_skew_sto3g.json";
// nevpt2_gradient_gpu.cpp's CASSCF convergence.
constexpr double kConvTolGrad = 1e-9;
// nevpt2_gradient_gpu.cpp's ladder, t = 1e-2 / 2^k, the extrapolant within
// 1e-7 of the prediction, relative here to the field's largest element;
// B_dot a tenth the length of h_dot, as there. Fifteen rungs, not ten: on
// direction 1, X's and A's ladders (and only theirs: X is h_bar's and Λ's
// own, X_ao = X_mo to 3e-15 at every rung) break their run at t ~ 2e-4,
// ratios 3.87 3.97 2.96 3.47 3.85, a max-norm change handed from one element
// to another, and settle to 3.96 3.99 4.00 4.00 only past the tenth rung
// (HIP, gfx1200; CUDA the same): at ten the steadiest run is the coarse one and misses by
// 4.0e-5, at fifteen 2.9e-10. The finest rung, 6.1e-7, is still far above
// the differences' rounding (about 1e-15 |X| / t, 2e-9 relative).
constexpr double kLadderT0 = 1e-2;
constexpr std::int32_t kRungs = 15;
constexpr double kLadderTolerance = 1e-7;
constexpr double kBScale = 0.1;
// A ladder whose changes are all below this (relative to the field's
// largest element) is flat: its rungs agree to rounding.
constexpr double kFlat = 1e-9;
// The visibility rule: each part above ten of the ladder's bars in some
// field.
constexpr double kVisibleBars = 10.0;
// The parts against the product: the same terms, but the class walk's
// differences taken along B_dot and the builders' tangent apart, each at
// its own step, so the sums agree to the walk's differencing round-off, far
// below the ladder's bar.
constexpr double kPartsTolerance = 1e-9;
// Partners on a case with no empty perturber change nothing class_energies_hvp
// does; the einsums' atomics reassociate the sums.
constexpr double kSameTolerance = 1e-12;

const fb::json &case_json() {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_hessian" / kCaseFile);
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc;
}

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

// This tree's DF-CASSCF of the case, through the real registry, once per
// process.
const State &state() {
  static const State kState = [] {
    const fb::json &start = case_json();
    const fb::test::ScratchDir dir("nevpt2_hessian_product");
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
    const std::string toml = std::format(
        "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
        "[casscf]\nconv_tol_grad = {:g}\nncas = {}\nnelecas = {}\nactive = [{}]\n"
        "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
        start.at("charge").get<std::int32_t>(), start.at("basis").at("name").get<std::string>(),
        kConvTolGrad, start.at("ncas").get<std::int32_t>(), start.at("nelecas").get<std::int32_t>(),
        active, start.at("aux_basis").at("name").get<std::string>());
    EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
    auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
    EXPECT_TRUE(calc.has_value()) << calc.error().message;
    if (!calc.has_value()) {
      return State{};
    }
    fb::NodeResources node;
    const fb::Status ran = calc->run(node);
    EXPECT_TRUE(ran.has_value()) << ran.error().message;
    if (!ran.has_value()) {
      return State{};
    }
    fb::DeviceResources &res = node.device(0);
    const auto &casscf = calc->get<fb::CasscfOutput>();
    const auto &eri = calc->get<fb::EriOutput>();
    const auto &h = calc->get<fb::CoreHamiltonianOutput>();
    EXPECT_TRUE(casscf.has_value() && eri.has_value() && h.has_value());
    if (!casscf.has_value() || !eri.has_value() || !h.has_value()) {
      return State{};
    }
    EXPECT_LT(casscf->gradient_norm, kConvTolGrad);
    const auto &df = std::get<fb::DensityFitted>(eri->eri);
    State s;
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

// A random symmetric n x n, uniform in [-scale, scale).
std::vector<double> symmetric(std::mt19937_64 &engine, const std::size_t n, const double scale) {
  std::uniform_real_distribution<double> uniform(-scale, scale);
  std::vector<double> m(n * n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q <= p; ++q) {
      m[p + (n * q)] = m[q + (n * p)] = uniform(engine);
    }
  }
  return m;
}

// C^T M C for the n x n M and the n x n C.
std::vector<double> to_mo(std::span<const double> c, std::span<const double> m,
                          const std::size_t n) {
  std::vector<double> mc(n * n, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        mc[mu + (n * q)] += m[mu + (n * nu)] * c[nu + (n * q)];
      }
    }
  }
  std::vector<double> out(n * n, 0.0);
  for (std::size_t q = 0; q < n; ++q) {
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        out[p + (n * q)] += c[mu + (n * p)] * mc[mu + (n * q)];
      }
    }
  }
  return out;
}

// One direction: AO h_dot and B_dot (B's layout, symmetric in the AO pair),
// their MO images under the state's C, and c_dot.
struct Direction {
  std::vector<double> h_ao;
  std::vector<double> b_ao;
  std::vector<double> h_mo;
  std::vector<double> b_mo;
  std::vector<double> c;
};

Direction random_direction(const State &s, const std::uint64_t seed) {
  const auto n = static_cast<std::size_t>(s.nao);
  const auto naux = static_cast<std::size_t>(s.naux);
  std::mt19937_64 engine(seed);
  Direction d{.h_ao = symmetric(engine, n, 1.0),
              .b_ao = std::vector<double>(naux * n * n),
              .h_mo = {},
              .b_mo = std::vector<double>(naux * n * n),
              .c = std::vector<double>(s.vector.size())};
  d.h_mo = to_mo(s.c, d.h_ao, n);
  for (std::size_t l = 0; l < naux; ++l) {
    const std::vector<double> slice = symmetric(engine, n, kBScale);
    const std::vector<double> moved = to_mo(s.c, slice, n);
    for (std::size_t pq = 0; pq < n * n; ++pq) {
      d.b_ao[l + (naux * pq)] = slice[pq];
      d.b_mo[l + (naux * pq)] = moved[pq];
    }
  }
  // c_dot_I = (r_I + 0.3) c_I, r symmetric under the strings' swap.
  const auto m = static_cast<std::size_t>(std::llround(std::sqrt(double(s.vector.size()))));
  const std::vector<double> r = symmetric(engine, m, 1.0);
  for (std::size_t k = 0; k < d.c.size(); ++k) {
    d.c[k] = (r[k] + 0.3) * s.vector[k];
  }
  return d;
}

// The five output fields of an adjoint (or its tangent), on the host.
struct Fields {
  std::vector<double> h_bar;
  std::vector<double> lambda;
  std::vector<double> y;
  std::vector<double> x;
  std::vector<double> a;
};
constexpr std::array<std::string_view, 5> kFieldNames{"h_bar", "lambda", "y", "x", "a"};
constexpr std::array<std::string_view, 5> kPartNames{"slabs", "builders", "one-slot", "phi",
                                                     "normalization"};

std::array<const std::vector<double> *, 5> fields_of(const Fields &f) {
  return {&f.h_bar, &f.lambda, &f.y, &f.x, &f.a};
}

Fields host_fields(fb::DeviceResources &res, const State &s, const nevpt2::AdjointTangent &t) {
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::size_t nmo2 = nmo * nmo;
  return {.h_bar = download(res, t.h_bar.data(), nmo2),
          .lambda = download(res, t.lambda.data(), static_cast<std::size_t>(s.naux) * nmo2),
          .y = download(res, t.y.data(), s.vector.size()),
          .x = download(res, t.x.data(), nmo2),
          .a = download(res, t.a.data(), nmo2)};
}

double dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double max_abs(std::span<const double> v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double max_abs_diff(std::span<const double> a, std::span<const double> b) {
  double m = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

// The strings, one set for both spins.
fb::ci::DeviceStringSet strings_of(fb::DeviceResources &res, const State &s) {
  auto strings = fb::ci::DeviceStringSet::create(res, s.ncas, s.nelecas / 2);
  EXPECT_TRUE(strings.has_value()) << strings.error().message;
  return std::move(*strings);
}

// The summed adjoint of the Rayleigh-normalized functional at the state
// moved by @p t along @p d (AO h and B, c to (c + t c_dot) / |c + t c_dot|),
// z fixed: sc_adjoint's + zc_adjoint's h_bar, Λ, X and A, and
// P (y + y^c) / |c + t c_dot|.
Fields moved_adjoint(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings,
                     const State &s, const Direction &d, std::span<const double> z,
                     const double t) {
  const auto shift = [t](std::span<const double> x, std::span<const double> dx) {
    std::vector<double> out(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      out[k] = x[k] + (t * dx[k]);
    }
    return out;
  };
  std::vector<double> v = shift(s.vector, d.c);
  const double norm = std::sqrt(dot(v, v));
  for (double &e : v) {
    e /= norm;
  }
  const auto h = upload(res, shift(s.h, d.h_ao));
  const auto b = upload(res, shift(s.b, d.b_ao));
  const auto c = upload(res, s.c);
  const auto vector = upload(res, v);
  const auto zd = upload(res, z);
  const auto corr = nevpt2::sc_adjoint(res, strings, strings, s.naux, s.nao, s.nao, b.data(),
                                       c.data(), h.data(), s.ncore, s.ncas, vector.data(), 0);
  EXPECT_TRUE(corr.has_value()) << corr.error().message;
  const std::array<double, 1> weights{1.0};
  const auto phi =
      nevpt2::zc_adjoint(res, strings, strings, s.naux, s.nao, s.nao, b.data(), c.data(), h.data(),
                         s.ncore, s.ncas, weights, vector.data(), zd.data());
  EXPECT_TRUE(phi.has_value()) << phi.error().message;
  if (!corr.has_value() || !phi.has_value()) {
    return {};
  }
  const auto nmo = static_cast<std::size_t>(s.nao);
  const std::size_t nmo2 = nmo * nmo;
  const std::size_t nl = static_cast<std::size_t>(s.naux) * nmo2;
  const auto sum = [&](const fb::Buffers::device<double> &p, const fb::Buffers::device<double> &q,
                       const std::size_t count) {
    std::vector<double> out = download(res, p.data(), count);
    const std::vector<double> other = download(res, q.data(), count);
    for (std::size_t k = 0; k < count; ++k) {
      out[k] += other[k];
    }
    return out;
  };
  Fields f{.h_bar = sum(corr->h_bar, phi->h_bar, nmo2),
           .lambda = sum(corr->lambda, phi->lambda, nl),
           .y = sum(corr->y, phi->y, v.size()),
           .x = sum(corr->x, phi->x, nmo2),
           .a = sum(corr->a, phi->a, nmo2)};
  const double cy = dot(v, f.y);
  for (std::size_t k = 0; k < v.size(); ++k) {
    f.y[k] = (f.y[k] - (cy * v[k])) / norm;
  }
  return f;
}

// One field's ladder: central differences D_k at t = t0 / 2^k, the ratios of
// successive changes (max norms), the longest run of them in (3.2, 4.8), and
// over that run the steadiest Richardson extrapolant (4 D_{k+2} - D_{k+1}) / 3
// (nevpt2_builders_tangent_gpu.cpp's field_ladder); over every rung when the
// ladder is flat.
struct FieldLadder {
  bool flat = false;
  std::int32_t run = 0;
  double miss = 0.0;
  std::string ratios;
};

FieldLadder field_ladder(const std::vector<std::vector<double>> &d,
                         std::span<const double> predicted) {
  FieldLadder l;
  const double scale = max_abs(predicted);
  std::vector<double> change;
  for (std::size_t k = 0; k + 1 < d.size(); ++k) {
    change.push_back(max_abs_diff(d[k], d[k + 1]));
  }
  l.flat = max_abs(change) <= kFlat * scale;
  std::int32_t best_end = -1;
  std::int32_t len = 0;
  for (std::size_t k = 0; k + 1 < change.size(); ++k) {
    const double ratio = change[k] / change[k + 1];
    l.ratios += std::format(" {:.3f}", ratio);
    len = (ratio > 3.2 && ratio < 4.8) ? len + 1 : 0;
    if (len > l.run) {
      l.run = len;
      best_end = static_cast<std::int32_t>(k);
    }
  }
  std::size_t first = 0;
  std::size_t last = 0;
  if (l.flat) {
    last = d.size() - 3;
  } else if (l.run > 0) {
    first = static_cast<std::size_t>(best_end - l.run + 1);
    last = static_cast<std::size_t>(best_end);
  } else {
    l.miss = std::numeric_limits<double>::infinity();
    return l;
  }
  const auto r = [&](const std::size_t k) {
    std::vector<double> out(d[k].size());
    for (std::size_t e = 0; e < out.size(); ++e) {
      out[e] = ((4.0 * d[k + 2][e]) - d[k + 1][e]) / 3.0;
    }
    return out;
  };
  std::size_t at = last;
  double gap = std::numeric_limits<double>::infinity();
  for (std::size_t k = first; k < last; ++k) {
    const double g = max_abs_diff(r(k + 1), r(k));
    if (g < gap) {
      gap = g;
      at = k + 1;
    }
  }
  l.miss = max_abs_diff(r(at), predicted) / scale;
  return l;
}

// The state on the device, its z^c, and the two directions stacked.
struct ProductCase {
  const State *s = nullptr;
  fb::ci::DeviceStringSet strings;
  fb::Buffers::device<double> b;
  fb::Buffers::device<double> c;
  fb::Buffers::device<double> h;
  fb::Buffers::device<double> vector;
  std::vector<double> z;
  fb::Buffers::device<double> z_d;
  std::array<Direction, 2> dirs;
  fb::Buffers::device<double> h_dot;
  fb::Buffers::device<double> b_dot;
  fb::Buffers::device<double> c_dot;

  [[nodiscard]] nevpt2::MoDirections directions() const {
    return {.count = static_cast<std::int32_t>(dirs.size()),
            .h = h_dot.data(),
            .b = b_dot.data(),
            .c = c_dot.data()};
  }

  [[nodiscard]] fb::Result<nevpt2::ScHessianProduct>
  product(fb::DeviceResources &res, const nevpt2::MoDirections &partners = {},
          const bool parts = false, const double *vector_override = nullptr,
          const double *z_override = z_unset()) const {
    return nevpt2::sc_hessian_product(
        res, strings, strings, s->naux, s->nao, s->nao, b.data(), c.data(), h.data(), s->ncore,
        s->ncas, vector_override != nullptr ? vector_override : vector.data(),
        z_override == z_unset() ? z_d.data() : z_override, 0, directions(), partners, parts);
  }

  // A sentinel distinct from null, so that a test can pass a null z.
  static const double *z_unset() {
    static const double kSentinel = 0.0;
    return &kSentinel;
  }
};

ProductCase make_case(fb::DeviceResources &res) {
  const State &s = state();
  ProductCase u{.s = &s,
                .strings = strings_of(res, s),
                .b = upload(res, s.b),
                .c = upload(res, s.c),
                .h = upload(res, s.h),
                .vector = upload(res, s.vector),
                .z = {},
                .z_d = fb::Buffers::device<double>(0, res.device()),
                .dirs = {random_direction(s, 8141), random_direction(s, 8142)},
                .h_dot = fb::Buffers::device<double>(0, res.device()),
                .b_dot = fb::Buffers::device<double>(0, res.device()),
                .c_dot = fb::Buffers::device<double>(0, res.device())};
  // z^c of the energy at the point, as sc_gradient forms it.
  const auto adj = nevpt2::sc_adjoint(res, u.strings, u.strings, s.naux, s.nao, s.nao, u.b.data(),
                                      u.c.data(), u.h.data(), s.ncore, s.ncas, u.vector.data(), 0);
  EXPECT_TRUE(adj.has_value()) << adj.error().message;
  if (adj.has_value()) {
    auto zc =
        nevpt2::zc_multipliers(res, s.nao, s.ncore, s.ncas, adj->a.data(), adj->epsilon.data());
    EXPECT_TRUE(zc.has_value()) << zc.error().message;
    if (zc.has_value()) {
      u.z = download(res, zc->z.data(), static_cast<std::size_t>(s.nao) * s.nao);
    }
  }
  u.z_d = upload(res, u.z);
  std::vector<double> h_dot;
  std::vector<double> b_dot;
  std::vector<double> c_dot;
  for (const Direction &d : u.dirs) {
    h_dot.insert(h_dot.end(), d.h_mo.begin(), d.h_mo.end());
    b_dot.insert(b_dot.end(), d.b_mo.begin(), d.b_mo.end());
    c_dot.insert(c_dot.end(), d.c.begin(), d.c.end());
  }
  u.h_dot = upload(res, h_dot);
  u.b_dot = upload(res, b_dot);
  u.c_dot = upload(res, c_dot);
  return u;
}

// The product along each direction against a ladder of the summed adjoint,
// every field; then each part's visibility at that ladder's bars, and the
// parts' sum against the product.
TEST(Nevpt2HessianProduct, MatchesLadder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const ProductCase u = make_case(res);
  ASSERT_FALSE(u.z.empty());
  const State &s = *u.s;
  auto got = u.product(res, {}, true);
  ASSERT_TRUE(got.has_value()) << got.error().message;
  ASSERT_EQ(got->products.size(), u.dirs.size());
  ASSERT_EQ(got->parts.size(), u.dirs.size());
  for (const auto &populated : got->populated) {
    EXPECT_EQ(std::accumulate(populated.begin(), populated.end(), std::int64_t{0}), 0)
        << "the case has no empty perturber";
  }

  double worst = 0.0;
  double smallest_margin = std::numeric_limits<double>::infinity();
  double worst_sum = 0.0;
  for (std::size_t k = 0; k < u.dirs.size(); ++k) {
    const Fields predicted = host_fields(res, s, got->products[k]);
    const auto pf = fields_of(predicted);
    // D_k per field, over the rungs.
    std::array<std::vector<std::vector<double>>, 5> d;
    for (std::int32_t rung = 0; rung < kRungs; ++rung) {
      const double t = std::ldexp(kLadderT0, -rung);
      const Fields plus = moved_adjoint(res, u.strings, s, u.dirs[k], u.z, t);
      const Fields minus = moved_adjoint(res, u.strings, s, u.dirs[k], u.z, -t);
      const auto pp = fields_of(plus);
      const auto mm = fields_of(minus);
      for (std::size_t f = 0; f < d.size(); ++f) {
        ASSERT_EQ(pp[f]->size(), pf[f]->size());
        ASSERT_EQ(mm[f]->size(), pf[f]->size());
        std::vector<double> diff(pf[f]->size());
        for (std::size_t e = 0; e < diff.size(); ++e) {
          diff[e] = ((*pp[f])[e] - (*mm[f])[e]) / (2.0 * t);
        }
        d[f].push_back(std::move(diff));
      }
    }
    std::array<double, 5> bars{};
    for (std::size_t f = 0; f < d.size(); ++f) {
      const std::string name = std::format("direction {} {}", k, kFieldNames[f]);
      ASSERT_GT(max_abs(*pf[f]), 0.0) << name << ": the product is zero";
      const FieldLadder l = field_ladder(d[f], *pf[f]);
      EXPECT_TRUE(l.flat || l.run >= 3) << name << " ratios:" << l.ratios;
      EXPECT_LE(l.miss, kLadderTolerance) << name << " ratios:" << l.ratios;
      worst = std::max(worst, l.miss);
      bars[f] = kLadderTolerance * max_abs(*pf[f]);
      std::cout << std::format("[ nevpt2   ] hessian product {}: |product| {:.3e}, relative "
                               "miss {:.2e}, {}ratios{}\n",
                               name, max_abs(*pf[f]), l.miss, l.flat ? "flat, " : "", l.ratios);
    }

    // Each part above ten bars in some field; the parts summing to the
    // product.
    std::array<std::vector<double>, 5> total;
    for (std::size_t f = 0; f < total.size(); ++f) {
      total[f].assign(pf[f]->size(), 0.0);
    }
    for (std::size_t p = 0; p < kPartNames.size(); ++p) {
      const Fields part = host_fields(res, s, got->parts[k][p]);
      const auto qf = fields_of(part);
      double margin = 0.0;
      std::string_view where;
      for (std::size_t f = 0; f < qf.size(); ++f) {
        const double m = max_abs(*qf[f]) / bars[f];
        if (m > margin) {
          margin = m;
          where = kFieldNames[f];
        }
        for (std::size_t e = 0; e < qf[f]->size(); ++e) {
          total[f][e] += (*qf[f])[e];
        }
      }
      EXPECT_GT(margin, kVisibleBars) << "direction " << k << " part " << kPartNames[p];
      smallest_margin = std::min(smallest_margin, margin);
      std::cout << std::format("[ nevpt2   ] hessian product direction {} part {}: {:.3e} bars "
                               "in {}\n",
                               k, kPartNames[p], margin, where);
    }
    for (std::size_t f = 0; f < total.size(); ++f) {
      const double off = max_abs_diff(total[f], *pf[f]) / max_abs(*pf[f]);
      EXPECT_LE(off, kPartsTolerance) << "direction " << k << " " << kFieldNames[f];
      worst_sum = std::max(worst_sum, off);
    }
  }
  std::cout << std::format("[ nevpt2   ] hessian product: worst relative miss {:.2e}, smallest "
                           "visibility {:.3e} bars, parts' sum off by {:.2e}\n",
                           worst, smallest_margin, worst_sum);
}

// Partners are class_energies_hvp's pair check, which reads them only on an
// empty perturber: on this case, none, so the product is the one without.
TEST(Nevpt2HessianProduct, PartnersChangeNothingWithoutEmptyPerturbers) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const ProductCase u = make_case(res);
  ASSERT_FALSE(u.z.empty());
  auto alone = u.product(res);
  ASSERT_TRUE(alone.has_value()) << alone.error().message;
  // Each direction partnered with the other.
  const std::size_t nmo2 = static_cast<std::size_t>(u.s->nao) * u.s->nao;
  std::vector<double> h_dot;
  std::vector<double> b_dot;
  std::vector<double> c_dot;
  for (const Direction &d : {u.dirs[1], u.dirs[0]}) {
    h_dot.insert(h_dot.end(), d.h_mo.begin(), d.h_mo.end());
    b_dot.insert(b_dot.end(), d.b_mo.begin(), d.b_mo.end());
    c_dot.insert(c_dot.end(), d.c.begin(), d.c.end());
  }
  EXPECT_EQ(h_dot.size(), 2 * nmo2);
  const auto hp = upload(res, h_dot);
  const auto bp = upload(res, b_dot);
  const auto cp = upload(res, c_dot);
  auto paired = u.product(res, {.count = 2, .h = hp.data(), .b = bp.data(), .c = cp.data()});
  ASSERT_TRUE(paired.has_value()) << paired.error().message;
  for (std::size_t k = 0; k < u.dirs.size(); ++k) {
    EXPECT_EQ(paired->defect[k], 0.0);
    const Fields a = host_fields(res, *u.s, alone->products[k]);
    const Fields b = host_fields(res, *u.s, paired->products[k]);
    const auto af = fields_of(a);
    const auto bf = fields_of(b);
    for (std::size_t f = 0; f < af.size(); ++f) {
      EXPECT_LE(max_abs_diff(*af[f], *bf[f]), kSameTolerance * max_abs(*af[f]))
          << "direction " << k << " " << kFieldNames[f];
    }
  }
}

TEST(Nevpt2HessianProduct, Refuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const ProductCase u = make_case(res);
  ASSERT_FALSE(u.z.empty());
  const auto refused = [](const fb::Result<nevpt2::ScHessianProduct> &got,
                          const std::string_view what) {
    ASSERT_FALSE(got.has_value()) << what;
    EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(got.error().message.find("nevpt2::sc_hessian_product"), std::string::npos)
        << got.error().message;
  };
  refused(
      u.product(res, {.count = 1, .h = u.h_dot.data(), .b = u.b_dot.data(), .c = u.c_dot.data()}),
      "one partner for two directions");
  refused(u.product(res, {.count = 2, .h = u.h_dot.data(), .b = nullptr, .c = u.c_dot.data()}),
          "a null partner array");
  refused(u.product(res, {}, false, nullptr, nullptr), "a null z");
  std::vector<double> scaled = u.s->vector;
  for (double &v : scaled) {
    v *= 1.5;
  }
  const auto long_vector = upload(res, scaled);
  refused(u.product(res, {}, false, long_vector.data()), "a vector of norm 1.5");
}

} // namespace
