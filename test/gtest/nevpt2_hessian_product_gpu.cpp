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

// The [casscf] and its system, as a run.toml and its molecule.xyz take them:
// the geometry (xyz text, angstrom), and the TOML from [basis_set] on, with
// [molecule] written here.
struct StateSpec {
  std::string name;
  std::string xyz;
  std::int32_t charge = 0;
  std::string rest;
};

// A data/nevpt2_hessian/ case's spec: its atoms (bohr), active space and
// fitting set, the CASSCF at kConvTolGrad.
StateSpec json_spec(const fb::json &start) {
  StateSpec spec{
      .name = start.at("name").get<std::string>(),
      .xyz = std::format("{}\n{}\n", start.at("atoms").size(), start.at("name").get<std::string>()),
      .charge = start.at("charge").get<std::int32_t>(),
      .rest = {}};
  for (const fb::json &atom : start.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    spec.xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                            coord[0] * fb::kAngstromPerBohr<double>,
                            coord[1] * fb::kAngstromPerBohr<double>,
                            coord[2] * fb::kAngstromPerBohr<double>);
  }
  std::string active;
  for (const std::int32_t a : start.at("active").get<std::vector<std::int32_t>>()) {
    active += std::format("{}{}", active.empty() ? "" : ", ", a);
  }
  spec.rest =
      std::format("[basis_set]\nbasis = \"{}\"\n"
                  "[casscf]\nconv_tol_grad = {:g}\nncas = {}\nnelecas = {}\nactive = [{}]\n"
                  "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                  start.at("basis").at("name").get<std::string>(), kConvTolGrad,
                  start.at("ncas").get<std::int32_t>(), start.at("nelecas").get<std::int32_t>(),
                  active, start.at("aux_basis").at("name").get<std::string>());
  return spec;
}

State build_state(const StateSpec &spec);

// This tree's DF-CASSCF of the case, through the real registry, once per
// process.
const State &state() {
  static const State kState = build_state(json_spec(case_json()));
  return kState;
}

// This tree's DF-CASSCF of @p spec, through the real registry; its first
// root's vector when it has several.
State build_state(const StateSpec &spec) {
  const fb::test::ScratchDir dir("nevpt2_hessian_product_" + spec.name);
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", spec.xyz).has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n{}", spec.charge, spec.rest);
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
  const std::size_t ndet = casscf->vectors.size() / casscf->energies.size();
  s.vector.assign(casscf->vectors.begin(),
                  casscf->vectors.begin() + static_cast<std::ptrdiff_t>(ndet));
  return s;
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
// P (y + y^c) / |c + t c_dot|; with @p pc_floor, pc_adjoint's at that
// floor in place of sc_adjoint's.
Fields moved_adjoint(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings,
                     const State &s, const Direction &d, std::span<const double> z, const double t,
                     const std::optional<double> pc_floor = std::nullopt) {
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
  // E_corr's five arrays, SC's or PC's.
  std::optional<std::array<fb::Buffers::device<double>, 5>> corr;
  const auto keep = [&](auto &&got) {
    EXPECT_TRUE(got.has_value()) << got.error().message;
    if (got.has_value()) {
      corr.emplace(std::array{std::move(got->h_bar), std::move(got->lambda), std::move(got->y),
                              std::move(got->x), std::move(got->a)});
    }
  };
  if (pc_floor) {
    keep(nevpt2::pc_adjoint(res, strings, strings, s.naux, s.nao, s.nao, b.data(), c.data(),
                            h.data(), s.ncore, s.ncas, vector.data(), 0, 0, *pc_floor));
  } else {
    keep(nevpt2::sc_adjoint(res, strings, strings, s.naux, s.nao, s.nao, b.data(), c.data(),
                            h.data(), s.ncore, s.ncas, vector.data(), 0));
  }
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
  Fields f{.h_bar = sum((*corr)[0], phi->h_bar, nmo2),
           .lambda = sum((*corr)[1], phi->lambda, nl),
           .y = sum((*corr)[2], phi->y, v.size()),
           .x = sum((*corr)[3], phi->x, nmo2),
           .a = sum((*corr)[4], phi->a, nmo2)};
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

// ---------------------------------------------------------------------------
// PC: pc_hessian_product
// ---------------------------------------------------------------------------

// A PC product: a state, its PC z^c (zc_multipliers of pc_adjoint's A at the
// floor), its directions stacked, and the floor.
struct PcCase {
  const State *s = nullptr;
  fb::ci::DeviceStringSet strings;
  fb::Buffers::device<double> b, c, h, vector;
  std::vector<double> z;
  fb::Buffers::device<double> z_d;
  std::vector<Direction> dirs;
  fb::Buffers::device<double> h_dot, b_dot, c_dot;
  double floor = 0.0;

  [[nodiscard]] nevpt2::MoDirections directions() const {
    return {.count = static_cast<std::int32_t>(dirs.size()),
            .h = h_dot.data(),
            .b = b_dot.data(),
            .c = c_dot.data()};
  }

  [[nodiscard]] fb::Result<nevpt2::PcHessianProduct>
  product(fb::DeviceResources &res, const bool parts = false, const bool span = true) const {
    return nevpt2::pc_hessian_product(res, strings, strings, s->naux, s->nao, s->nao, b.data(),
                                      c.data(), h.data(), s->ncore, s->ncas, vector.data(),
                                      z_d.data(), 0, directions(), parts, floor, span);
  }
};

PcCase make_pc_case(fb::DeviceResources &res, const State &s, std::vector<Direction> dirs,
                    const double floor) {
  PcCase u{.s = &s,
           .strings = strings_of(res, s),
           .b = upload(res, s.b),
           .c = upload(res, s.c),
           .h = upload(res, s.h),
           .vector = upload(res, s.vector),
           .z = {},
           .z_d = fb::Buffers::device<double>(0, res.device()),
           .dirs = std::move(dirs),
           .h_dot = fb::Buffers::device<double>(0, res.device()),
           .b_dot = fb::Buffers::device<double>(0, res.device()),
           .c_dot = fb::Buffers::device<double>(0, res.device()),
           .floor = floor};
  const auto adj =
      nevpt2::pc_adjoint(res, u.strings, u.strings, s.naux, s.nao, s.nao, u.b.data(), u.c.data(),
                         u.h.data(), s.ncore, s.ncas, u.vector.data(), 0, 0, floor);
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

// PC's ladder starts shorter than SC's: at t = 1e-2 the moved states meet
// intruders (|y / (λ + Δ)| past 1 in Sr, Sij or Sir), which PC refuses, and
// at the twist already at 2e-3 (Sr). Its rungs end near SC's 6.1e-7.
struct PcLadderSteps {
  double t0;
  std::int32_t rungs;
};
constexpr PcLadderSteps kPcSkewLadder{.t0 = 2e-3, .rungs = 13};
constexpr PcLadderSteps kPcTwistLadder{.t0 = 1e-3, .rungs = 12};
constexpr double kPcTwistHb = 0.1;

// Direction @p k's product against the ladder of PC's summed adjoint at the
// case's floor, every field: each field's relative miss and bar
// (kLadderTolerance of its largest element), logged under @p what.
struct PcLadder {
  double worst = 0.0;
  std::array<double, 5> bars{};
};

PcLadder pc_ladder(fb::DeviceResources &res, const PcCase &u, const std::size_t k,
                   const Fields &predicted, const std::string &what,
                   const PcLadderSteps &steps = kPcSkewLadder) {
  const auto pf = fields_of(predicted);
  std::array<std::vector<std::vector<double>>, 5> d;
  for (std::int32_t rung = 0; rung < steps.rungs; ++rung) {
    const double t = std::ldexp(steps.t0, -rung);
    const Fields plus = moved_adjoint(res, u.strings, *u.s, u.dirs[k], u.z, t, u.floor);
    const Fields minus = moved_adjoint(res, u.strings, *u.s, u.dirs[k], u.z, -t, u.floor);
    const auto pp = fields_of(plus);
    const auto mm = fields_of(minus);
    for (std::size_t f = 0; f < d.size(); ++f) {
      EXPECT_EQ(pp[f]->size(), pf[f]->size());
      EXPECT_EQ(mm[f]->size(), pf[f]->size());
      if (pp[f]->size() != pf[f]->size() || mm[f]->size() != pf[f]->size()) {
        return {.worst = std::numeric_limits<double>::infinity(), .bars = {}};
      }
      std::vector<double> diff(pf[f]->size());
      for (std::size_t e = 0; e < diff.size(); ++e) {
        diff[e] = ((*pp[f])[e] - (*mm[f])[e]) / (2.0 * t);
      }
      d[f].push_back(std::move(diff));
    }
  }
  PcLadder out;
  for (std::size_t f = 0; f < d.size(); ++f) {
    const std::string name = std::format("{} direction {} {}", what, k, kFieldNames[f]);
    const double scale = max_abs(*pf[f]);
    EXPECT_GT(scale, 0.0) << name << ": the product is zero";
    const FieldLadder l = field_ladder(d[f], *pf[f]);
    EXPECT_TRUE(l.flat || l.run >= 3) << name << " ratios:" << l.ratios;
    EXPECT_LE(l.miss, kLadderTolerance) << name << " ratios:" << l.ratios;
    out.worst = std::max(out.worst, l.miss);
    out.bars[f] = kLadderTolerance * scale;
    std::cout << std::format("[ nevpt2   ] PC hessian product {}: |product| {:.3e}, relative miss "
                             "{:.2e}, {}ratios{}\n",
                             name, scale, l.miss, l.flat ? "flat, " : "", l.ratios);
  }
  return out;
}

// The largest of @p a - @p b over each field, in units of @p bars.
double in_bars(const Fields &a, const Fields &b, const std::array<double, 5> &bars,
               std::string_view &where) {
  const auto fa = fields_of(a);
  const auto fb_ = fields_of(b);
  double most = 0.0;
  for (std::size_t f = 0; f < fa.size(); ++f) {
    const double m = max_abs_diff(*fa[f], *fb_[f]) / bars[f];
    if (m > most) {
      most = m;
      where = kFieldNames[f];
    }
  }
  return most;
}

// Each class's dropped count, and its cut and kept σ over σ_max, for the log.
std::string pc_spectra(const nevpt2::PcClassEnergies &e) {
  constexpr std::array<std::string_view, 8> kNames{"Sr",   "Si",  "Sijrs", "Sijr",
                                                   "Srsi", "Srs", "Sij",   "Sir"};
  std::string out;
  for (std::size_t k = 0; k < 8; ++k) {
    if (k != 2) {
      out += std::format(" {} {} cut {:.1e} kept {:.1e};", kNames[k], e.dropped[k], e.cut[k],
                         e.kept[k]);
    }
  }
  return out;
}

// The length of a singlet direction's part off c.
constexpr double kSingletLength = 0.5;

// A spin-pure direction: random_direction's h and B, and a CI direction
// c_dot = H_κ c + 0.3 c over the state's own c, H_κ a random spin-free
// Hamiltonian over the active orbitals (one- and two-body, κ and (tu|vw)
// random with their symmetries; h and B scaled by @p hb_scale): a spin-free
// operator on a singlet, so c_dot is a singlet, as a CASSCF response is,
// and it reaches every configuration a two-body excitation does. (random_direction's
// covariant c_dot mixes in quintets, along which the metrics' spin-null modes
// rise, README "The Schur lift".) With @p mask, κ's entries off it are zero.
Direction singlet_direction(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings,
                            const State &s, const std::uint64_t seed,
                            const std::function<bool(std::size_t, std::size_t)> &mask = {},
                            const double hb_scale = 1.0) {
  Direction d = random_direction(s, seed);
  for (std::vector<double> *v : {&d.h_ao, &d.h_mo, &d.b_ao, &d.b_mo}) {
    for (double &x : *v) {
      x *= hb_scale;
    }
  }
  const auto n = static_cast<std::size_t>(s.ncas);
  const auto dot_of = [](std::span<const double> a, std::span<const double> b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  };
  std::mt19937_64 engine(seed + 1000);
  std::vector<double> kappa = symmetric(engine, n, 1.0);
  if (mask) {
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = 0; q < n; ++q) {
        if (!mask(p, q)) {
          kappa[p + (n * q)] = 0.0;
        }
      }
    }
  }
  auto sigma = fb::ci::Sigma::create(res, strings, strings);
  EXPECT_TRUE(sigma.has_value()) << sigma.error().message;
  if (!sigma.has_value()) {
    return d;
  }
  // (tu|vw) random with the 8-fold symmetry, masked as κ.
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
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
          const double x = (!mask || (mask(t, u) && mask(v, w))) ? uniform(engine) : 0.0;
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
  EXPECT_TRUE(sigma->set_integrals(res, 0.0, kappa, eri).has_value());
  const auto c = upload(res, s.vector);
  fb::Buffers::device<double> out(s.vector.size(), res.device());
  EXPECT_TRUE(sigma->apply(res, 1, c.data(), out.data()).has_value());
  // κ c off c, scaled to kSingletLength, then 0.3 c along it.
  std::vector<double> kc = download(res, out.data(), s.vector.size());
  const double along = dot_of(kc, s.vector);
  for (std::size_t k = 0; k < kc.size(); ++k) {
    kc[k] -= along * s.vector[k];
  }
  const double length = std::sqrt(dot_of(kc, kc));
  d.c.assign(kc.size(), 0.0);
  for (std::size_t k = 0; k < kc.size(); ++k) {
    d.c[k] = (length > 0.0 ? kSingletLength * kc[k] / length : 0.0) + (0.3 * s.vector[k]);
  }
  return d;
}

// The largest Schur lift over every class and direction.
double largest_lift(const nevpt2::PcHessianProduct &p) {
  double out = 0.0;
  for (const auto &l : p.lifts) {
    for (const double x : l) {
      out = std::max(out, x);
    }
  }
  return out;
}

// The PSB3 twist (nevpt2_psb3_gpu.cpp's kPsb3Twisted, #853's 90° geometry),
// DF-SA3-CASSCF(6,6)/STO-3G over cc-pVDZ-JKFIT, state 0 at the SA orbitals:
// Sr's and Si's cuts drop real modes there (#862), their kept modes at about
// 4e-13 of σ_max at the default cut, where the product refuses (#917).
constexpr std::string_view kPsb3Twisted = R"(14
PSB3, C13=C14 twisted 90 degrees
N     -0.0454402492     0.0522740911    -0.0000000000
C      1.1276796768     0.5920624626     0.0000000000
C      2.3467584180    -0.1268535537     0.0000000000
C      3.5352478218     0.5543991513     0.0000000000
C      4.2381819494     0.9573272501    -1.1898886726
C      5.4137881019     1.6311951607    -1.1318525401
H      1.1448444823     1.6750816790    -0.0000000000
H      2.3229137518    -1.2094227332    -0.0000000000
H      3.9874420661     0.8136009171     0.9516444335
H      3.7996110498     0.7059345067    -2.1474371187
H     -0.1682915976    -0.9466686957    -0.0000000000
H     -0.8778598767     0.6155991196    -0.0000000000
H      5.8725712624     1.8941737519    -0.1873080265
H      5.9322594717     1.9283875648    -2.0319280688
)";

const State &psb3_state() {
  static const State kState =
      build_state({.name = "psb3_twist_sto3g",
                   .xyz = std::string{kPsb3Twisted},
                   .charge = 1,
                   .rest = "[basis_set]\nbasis = \"sto-3g\"\n"
                           "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pvdz-jkfit\"\n"
                           "[scf]\nmax_iterations = 400\n"
                           "[casscf]\nncas = 6\nnelecas = 6\nnroots = 3\nconv_tol_grad = 1e-9\n"});
  return kState;
}

// A cut that drops real modes, at a floor that keeps the solve conditioned
// (#917), per case: h2co_skew_sto3g at kPcSkewFloor, the family's fast case,
// and the PSB3 twist at kPcTwistFloor (README "The PC Hessian-vector product
// in (h, B, c)").
constexpr double kPcSkewFloor = 1e-6;
constexpr double kPcTwistFloor = 1e-8;

// The product along two mixed directions against the ladder of
// pc_adjoint + zc_adjoint, every field to kLadderTolerance; the kept span's
// rotation (the product less its span-free twin) above ten bars in some
// field of every direction, which a cut of real modes makes visible; the
// parts summing to the product; and every Schur lift at round-off.
class Nevpt2PcHessianProduct : public ::testing::TestWithParam<std::string> {};

TEST_P(Nevpt2PcHessianProduct, MatchesLadder) {
  FBF_REQUIRE_GPU();
  const std::string which = GetParam();
  if (which != "h2co_skew_sto3g") {
    // A CASSCF(6,6) and sixty PC adjoints (README), past slow_test.h's rule.
    FBF_SKIP_IF_FAST_TESTS();
  }
  fb::DeviceResources res(0);
  const bool twist = which == "psb3_twist_sto3g";
  const State &s = twist ? psb3_state() : state();
  ASSERT_FALSE(s.b.empty());
  const double floor = twist ? kPcTwistFloor : kPcSkewFloor;
  const fb::ci::DeviceStringSet strings = strings_of(res, s);
  // At the twist h and B move a tenth as far: intruders sit within 2.5e-4
  // of the point along the full-length h (Si, |t| 1.3).
  const double hb = twist ? kPcTwistHb : 1.0;
  const PcCase u = make_pc_case(res, s,
                                {singlet_direction(res, strings, s, 8151, {}, hb),
                                 singlet_direction(res, strings, s, 8152, {}, hb)},
                                floor);
  ASSERT_FALSE(u.z.empty());
  auto got = u.product(res, true);
  ASSERT_TRUE(got.has_value()) << got.error().message;
  auto fixed = u.product(res, false, false);
  ASSERT_TRUE(fixed.has_value()) << fixed.error().message;
  ASSERT_EQ(got->products.size(), u.dirs.size());
  std::int64_t dropped = 0;
  for (std::size_t k = 0; k < 8; ++k) {
    dropped += got->energies.dropped[k];
  }
  EXPECT_GT(dropped, 0) << "the cut drops nothing";
  double worst = 0.0;
  double smallest = std::numeric_limits<double>::infinity();
  double worst_sum = 0.0;
  for (std::size_t k = 0; k < u.dirs.size(); ++k) {
    const Fields predicted = host_fields(res, s, got->products[k]);
    const PcLadder l =
        pc_ladder(res, u, k, predicted, which, twist ? kPcTwistLadder : kPcSkewLadder);
    worst = std::max(worst, l.worst);
    std::string_view where;
    const double span = in_bars(predicted, host_fields(res, s, fixed->products[k]), l.bars, where);
    EXPECT_GT(span, kVisibleBars) << which << " direction " << k << ": the span's rotation";
    smallest = std::min(smallest, span);
    std::cout << std::format("[ nevpt2   ] PC hessian product {} direction {}: the kept span's "
                             "rotation {:.3e} bars in {}\n",
                             which, k, span, where);
    // The parts sum to the product.
    const auto pf = fields_of(predicted);
    std::array<std::vector<double>, 5> total;
    for (std::size_t f = 0; f < total.size(); ++f) {
      total[f].assign(pf[f]->size(), 0.0);
    }
    for (std::size_t p = 0; p < kPartNames.size(); ++p) {
      const Fields part = host_fields(res, s, got->parts[k][p]);
      const auto qf = fields_of(part);
      for (std::size_t f = 0; f < qf.size(); ++f) {
        for (std::size_t e = 0; e < qf[f]->size(); ++e) {
          total[f][e] += (*qf[f])[e];
        }
      }
    }
    for (std::size_t f = 0; f < total.size(); ++f) {
      const double off = max_abs_diff(total[f], *pf[f]) / max_abs(*pf[f]);
      // At the twist the parts' separate walks carry the solve's conditioning
      // (kept 1e-9): only the gapped case holds them to SC's round-off.
      if (!twist) {
        EXPECT_LE(off, kPartsTolerance) << which << " direction " << k << " " << kFieldNames[f];
      }
      worst_sum = std::max(worst_sum, off);
    }
  }

  std::cout << std::format("[ nevpt2   ] PC hessian product {} (floor {:.0e}): worst relative "
                           "miss {:.2e}, the span's rotation at least {:.3e} bars, parts' sum off "
                           "by {:.2e}, largest Schur lift {:.2e};{}\n",
                           which, floor, worst, smallest, worst_sum, largest_lift(*got),
                           pc_spectra(got->energies));
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2PcHessianProduct,
                         ::testing::Values(std::string{"h2co_skew_sto3g"},
                                           std::string{"psb3_twist_sto3g"}),
                         [](const auto &info) { return info.param; });

// data/nevpt2_hessian/'s hcn_sto3g: linear HCN, CAS(4,4) over its π and π*
// (E1x and E1y each), STO-3G. Its Sir metric has two null modes by
// symmetry, which a singlet that breaks the axial symmetry lifts; no nuclear
// displacement does (the fixture's every point drops them).
const State &symmetric_state() {
  static const State kState = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_hessian" /
                              "hcn_sto3g.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? build_state(json_spec(*read)) : State{};
  }();
  return kState;
}

// h's active block in the state's orbitals, n x n: a totally symmetric
// one-body operator.
std::vector<double> active_h(const State &s) {
  const auto nmo = static_cast<std::size_t>(s.nao);
  const auto n = static_cast<std::size_t>(s.ncas);
  const auto nc = static_cast<std::size_t>(s.ncore);
  const std::vector<double> h_mo = to_mo(s.c, s.h, nmo);
  std::vector<double> out(n * n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      out[p + (n * q)] = h_mo[(nc + p) + (nmo * (nc + q))];
    }
  }
  return out;
}

// A singlet direction whose CI part is H_κ c for the given one-body κ
// (n x n, symmetric) alone, h and B as singlet_direction's.
Direction one_body_direction(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings,
                             const State &s, const std::uint64_t seed,
                             const std::vector<double> &kappa) {
  Direction d = random_direction(s, seed);
  const auto n = static_cast<std::size_t>(s.ncas);
  auto sigma = fb::ci::Sigma::create(res, strings, strings);
  EXPECT_TRUE(sigma.has_value()) << sigma.error().message;
  if (!sigma.has_value()) {
    return d;
  }
  EXPECT_TRUE(
      sigma->set_integrals(res, 0.0, kappa, std::vector<double>(n * n * n * n, 0.0)).has_value());
  const auto c = upload(res, s.vector);
  fb::Buffers::device<double> out(s.vector.size(), res.device());
  EXPECT_TRUE(sigma->apply(res, 1, c.data(), out.data()).has_value());
  d.c = download(res, out.data(), s.vector.size());
  return d;
}

// The class energies of @p s with its CI vector moved to
// (c + t u) / |c + t u|, u = c_dot - (c . c_dot) c (the product's sphere
// path), at @p floor; h and B stay.
nevpt2::PcClassEnergies pc_energies_along(fb::DeviceResources &res, const PcCase &u,
                                          const Direction &d, const double t, const double floor) {
  const State &s = *u.s;
  const double along = dot(s.vector, d.c);
  std::vector<double> v(s.vector.size());
  for (std::size_t k = 0; k < v.size(); ++k) {
    v[k] = s.vector[k] + (t * (d.c[k] - (along * s.vector[k])));
  }
  const double norm = std::sqrt(dot(v, v));
  for (double &e : v) {
    e /= norm;
  }
  const auto vd = upload(res, v);
  auto got = nevpt2::pc_energy(res, u.strings, u.strings, s.naux, s.nao, s.nao, u.b.data(),
                               u.c.data(), u.h.data(), s.ncore, s.ncas, vd.data(), 0, 0, floor);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  return got.has_value() ? *got : nevpt2::PcClassEnergies{};
}

// |u|² of direction @p d's CI part off c.
double off_c2(const State &s, const Direction &d) {
  const double along = dot(s.vector, d.c);
  double out = 0.0;
  for (std::size_t k = 0; k < d.c.size(); ++k) {
    const double x = d.c[k] - (along * s.vector[k]);
    out += x * x;
  }
  return out;
}

// Sir, PySCF's class order.
constexpr std::size_t kSir = 7;
// A lifted mode's σ at t, ½ ℓ |u|² t², against the direct probe: third
// order is O(t) of it at the probe's t.
constexpr double kLiftProbeTolerance = 1e-2;
// The floor the breaking direction's ladder runs at: it moves Sir's cut, and
// so t*, to about 1e-2, past the ladder's widest rung, and drops nothing
// else (every other kept σ is above 3e-5).
constexpr double kHcnLadderFloor = 1e-6;

// The symmetric geometry, hcn_sto3g. Along a singlet that breaks the axial
// symmetry, Sir's null modes rise at second order (README "The Schur
// lift"); the product reports the lift ℓ and the step t* within which the
// cut keeps them dropped, and stays the exact derivative of that branch.
// - The report against a direct probe of Sir's metric: at the default cut,
//   the risen mode is Sir's smallest kept one, at ½ ℓ |u|² t²
//   (pc_energy's `kept`, at t = 1e-4); at kHcnLadderFloor Sir drops both
//   modes at t*/2 and one at 2 t*.
// - The product along it, at kHcnLadderFloor, against a ladder whose widest
//   rung is below t*: the dropped branch.
// - A CI part that keeps the symmetry (h's active block on c, a totally
//   symmetric one-body operator) and an orbital-only direction (c_dot zero)
//   lift nothing and pass the ladder at the default cut.
TEST(Nevpt2PcHessianProduct, SymmetricLiftReported) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const State &s = symmetric_state();
  ASSERT_FALSE(s.b.empty());
  const fb::ci::DeviceStringSet strings = strings_of(res, s);
  Direction orbital = random_direction(s, 8162);
  std::ranges::fill(orbital.c, 0.0);
  const Direction kept = one_body_direction(res, strings, s, 8163, active_h(s));
  const Direction breaking = singlet_direction(res, strings, s, 8161);

  // The symmetric pair.
  const PcCase u = make_pc_case(res, s, {kept, orbital}, 0.0);
  ASSERT_FALSE(u.z.empty());
  auto got = u.product(res);
  ASSERT_TRUE(got.has_value()) << got.error().message;
  EXPECT_EQ(got->energies.dropped[kSir], 2) << "Sir's two null modes";
  double worst = 0.0;
  for (std::size_t k = 0; k < u.dirs.size(); ++k) {
    const Fields predicted = host_fields(res, s, got->products[k]);
    worst = std::max(worst, pc_ladder(res, u, k, predicted, "hcn_sto3g").worst);
    for (std::size_t c = 0; c < 8; ++c) {
      EXPECT_LE(got->lifts[k][c], 1e-12) << "direction " << k << " class " << c;
    }
  }

  // The breaking direction: its report against the probe.
  const PcCase v = make_pc_case(res, s, {breaking}, kHcnLadderFloor);
  ASSERT_FALSE(v.z.empty());
  auto lifted = v.product(res);
  ASSERT_TRUE(lifted.has_value()) << lifted.error().message;
  const double ell = lifted->lifts.front()[kSir];
  const double t_star = lifted->lift_steps.front()[kSir];
  const double uu = off_c2(s, breaking);
  ASSERT_GT(ell, 1e-3) << "Sir's null modes do not rise";
  constexpr double kProbeT = 1e-4;
  const nevpt2::PcClassEnergies probe = pc_energies_along(res, v, breaking, kProbeT, 0.0);
  const double predicted_sigma = 0.5 * ell * uu * kProbeT * kProbeT;
  EXPECT_EQ(probe.dropped[kSir], 1) << "one mode risen past the default cut";
  EXPECT_NEAR(probe.kept[kSir], predicted_sigma, kLiftProbeTolerance * predicted_sigma);
  const nevpt2::PcClassEnergies inside =
      pc_energies_along(res, v, breaking, 0.5 * t_star, kHcnLadderFloor);
  const nevpt2::PcClassEnergies outside =
      pc_energies_along(res, v, breaking, 2.0 * t_star, kHcnLadderFloor);
  EXPECT_EQ(inside.dropped[kSir], 2) << "at t*/2";
  EXPECT_LT(outside.dropped[kSir], 2) << "at 2 t*";
  // The product against a ladder inside t*: the dropped branch.
  ASSERT_GT(t_star, 2.0 * kPcSkewLadder.t0) << "t* inside the ladder";
  const PcLadder l =
      pc_ladder(res, v, 0, host_fields(res, s, lifted->products.front()), "hcn_sto3g bend");
  std::cout << std::format("[ nevpt2   ] PC hessian product hcn_sto3g: the symmetric and "
                           "orbital-only directions' worst relative miss {:.2e}, largest Schur "
                           "lift {:.2e}; the bend: Sir's lift {:.3e} per |u|^2, t* {:.3e} at "
                           "floor {:.0e} (|u|^2 {:.3f}), probe σ {:.4e} against ½ ℓ |u|² t² "
                           "{:.4e}, dropped {} / {} at t*/2 / 2t*, ladder inside t* worst "
                           "{:.2e};{}\n",
                           worst, largest_lift(*got), ell, t_star, kHcnLadderFloor, uu,
                           probe.kept[kSir], predicted_sigma, inside.dropped[kSir],
                           outside.dropped[kSir], l.worst, pc_spectra(got->energies));
}

// The AO core Hamiltonian whose MO view C^T h C is @p s's plus @p delta on
// the diagonal of each of @p orbitals and nothing else (nevpt2_gradient_gpu.cpp's
// shifted_h): h + delta Σ_j v_j v_jᵀ, v_j = C^-T e_j, so every orbital
// energy but the moved orbitals' is the state's own, and theirs move by
// delta. C^-1 by Gauss-Jordan with partial pivoting.
std::vector<double> shifted_h(const State &s, std::span<const std::size_t> orbitals,
                              const double delta) {
  const auto n = static_cast<std::size_t>(s.nao);
  // [C | I], row-major, reduced to [I | C^-1].
  std::vector<double> m(n * 2 * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      m[(i * 2 * n) + j] = s.c[i + (n * j)];
    }
    m[(i * 2 * n) + n + i] = 1.0;
  }
  for (std::size_t k = 0; k < n; ++k) {
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < n; ++i) {
      if (std::abs(m[(i * 2 * n) + k]) > std::abs(m[(pivot * 2 * n) + k])) {
        pivot = i;
      }
    }
    for (std::size_t j = 0; j < 2 * n; ++j) {
      std::swap(m[(k * 2 * n) + j], m[(pivot * 2 * n) + j]);
    }
    const double inv = 1.0 / m[(k * 2 * n) + k];
    for (std::size_t j = 0; j < 2 * n; ++j) {
      m[(k * 2 * n) + j] *= inv;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const double f = m[(i * 2 * n) + k];
      if (i != k && f != 0.0) {
        for (std::size_t j = 0; j < 2 * n; ++j) {
          m[(i * 2 * n) + j] -= f * m[(k * 2 * n) + j];
        }
      }
    }
  }
  std::vector<double> h = s.h;
  for (const std::size_t j : orbitals) {
    // v_mu = (C^-T)_{mu j} = (C^-1)_{j mu}.
    const auto v = std::span<const double>(m).subspan((j * 2 * n) + n, n);
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        h[mu + (n * nu)] += delta * v[mu] * v[nu];
      }
    }
  }
  return h;
}

// A degenerate inactive pair, z^c's zero branch: h2co_skew_sto3g with its
// two virtuals, and its closest core pair, each brought to 1e-9 Eh apart
// through h alone (shifted_h; Nevpt2PcZc's construction). PC's A vanishes
// on such a pair, so z^c is zero there; the product passes the same ladder.
TEST(Nevpt2PcHessianProduct, DegenerateInactivePair) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  const State &s = state();
  ASSERT_FALSE(s.b.empty());
  const auto nmo = static_cast<std::size_t>(s.nao);
  const auto nc = static_cast<std::size_t>(s.ncore);
  const std::size_t nocc = nc + static_cast<std::size_t>(s.ncas);
  ASSERT_EQ(nmo - nocc, 2U) << "the case's two virtuals";
  const fb::ci::DeviceStringSet strings = strings_of(res, s);
  // The orbital energies PC reads.
  const auto epsilon_of = [&](const State &x) -> std::vector<double> {
    const auto b = upload(res, x.b);
    const auto c = upload(res, x.c);
    const auto h = upload(res, x.h);
    const auto v = upload(res, x.vector);
    const auto adj = nevpt2::pc_adjoint(res, strings, strings, x.naux, x.nao, x.nao, b.data(),
                                        c.data(), h.data(), x.ncore, x.ncas, v.data(), 0);
    EXPECT_TRUE(adj.has_value()) << adj.error().message;
    return adj.has_value() ? download(res, adj->epsilon.data(), nmo) : std::vector<double>{};
  };
  const std::vector<double> eps = epsilon_of(s);
  ASSERT_EQ(eps.size(), nmo);
  std::size_t core = 0;
  for (std::size_t q = 1; q + 1 < nc; ++q) {
    if (eps[q + 1] - eps[q] < eps[core + 1] - eps[core]) {
      core = q;
    }
  }
  State forced = s;
  for (const std::size_t lower : {nocc, core}) {
    const std::array<std::size_t, 1> upper{lower + 1};
    const std::vector<double> e = epsilon_of(forced);
    forced.h = shifted_h(forced, upper, (e[lower] + 1e-9) - e[lower + 1]);
  }
  const std::vector<double> moved = epsilon_of(forced);
  const PcCase u = make_pc_case(res, forced,
                                {singlet_direction(res, strings, forced, 8171),
                                 singlet_direction(res, strings, forced, 8172)},
                                0.0);
  ASSERT_FALSE(u.z.empty());
  for (const std::size_t lower : {nocc, core}) {
    EXPECT_LT(std::abs(moved[lower + 1] - moved[lower]), 1e-8) << "pair at " << lower;
    EXPECT_EQ(u.z[lower + (nmo * (lower + 1))], 0.0) << "pair at " << lower;
    EXPECT_EQ(u.z[(lower + 1) + (nmo * lower)], 0.0) << "pair at " << lower;
  }
  auto got = u.product(res);
  ASSERT_TRUE(got.has_value()) << got.error().message;
  double worst = 0.0;
  for (std::size_t k = 0; k < u.dirs.size(); ++k) {
    const Fields predicted = host_fields(res, *u.s, got->products[k]);
    worst = std::max(worst, pc_ladder(res, u, k, predicted, "h2co_skew_sto3g degenerate").worst);
  }
  std::cout << std::format("[ nevpt2   ] PC hessian product, degenerate pairs ({}, {}) and ({}, "
                           "{}), gaps {:.1e} and {:.1e}: worst relative miss {:.2e}\n",
                           nocc, nocc + 1, core, core + 1, moved[nocc + 1] - moved[nocc],
                           moved[core + 1] - moved[core], worst);
}

} // namespace
