// GPU cases for nevpt2.rdm's nevpt2::rdms: dm3, f3ca and f3ac against PySCF's
// (data/nevpt2/, written by tools/emit/nevpt2_golden.py) on every case that
// carries them, at the default budget and at budgets forced down to many
// tiles, an unequal last tile and one-determinant tiles; the budget a tile
// width needs, read back as that width; a budget below one determinant
// refused as Unsupported with nothing written or allocated; the other
// refusals; and, slow, a CAS(12,12) build, its tile count and its measured
// peak device memory. And nevpt2::rdms_adjoint against rdms itself: rdms is
// quadratic in the CI vector and linear in (tu|vw), so a polarization
// difference of two builds, contracted with the bars, is the adjoint's
// directional derivative exactly; at one tile and at many, unequal ones.
// And the bra/ket overload: dm1, dm2, dm3, f3ca and f3ac between two
// different vectors against PySCF's (data/nevpt2_transition/, written by
// tools/emit/nevpt2_transition_golden.py), at one tile and many, its own
// budget refusal, and bra == ket as the one-vector build bit for bit.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.npy;
import free_breakfast.ci;
import free_breakfast.nevpt2;

namespace {

namespace fb = free_breakfast;

// Of max|golden| per tensor: the device and PySCF sum the same products in
// different orders (BLAS against PySCF's own kernels).
constexpr double kRelTol = 1e-12;
// A tiled build against the one-tile build: the same products, the sums over
// K split differently.
constexpr double kTileRelTol = 1e-13;
// Σ_pqr dm3[p, p, q, q, r, r] = ⟨N³⟩ = N³, of N³.
constexpr double kTraceRelTol = 1e-12;
// The adjoint's ⟨y, u⟩ (or ⟨h2ē, δ⟩) against the bars on the polarization
// difference, of Σ_i |bar_i| |difference_i|: both are the same products
// summed in different orders, so this is round-off.
constexpr double kAdjointRelTol = 1e-12;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2";
const std::filesystem::path kTransitionDir =
    std::filesystem::path{FBF_TEST_DATA_DIR} / "nevpt2_transition";

// The cases README.md says carry dm3, f3ca and f3ac (CAS(10,10)'s do not, by
// size; #418's chain test covers them). Every one has n_alpha = n_beta.
struct Case {
  const char *name;
  std::int32_t n;
  std::int32_t per_spin;
};
constexpr std::array kCases{Case{"n2_ccpvdz_cas44_df", 4, 2},
                            Case{"n2_ccpvdz_cas44_casscf_df", 4, 2},
                            Case{"n2_ccpvdz_cas88_df", 8, 4}};

// @p a, column-major as io.npy returns it, row-major (C order): what PySCF's
// array holds, element for element.
std::vector<double> row_major(const fb::NpyArray &a) {
  // Axis d's column-major stride: the product of the axes before it.
  std::vector<std::size_t> stride(a.shape.size(), 1);
  for (std::size_t d = 1; d < a.shape.size(); ++d) {
    stride[d] = stride[d - 1] * a.shape[d - 1];
  }
  std::vector<double> out(a.data.size());
  for (std::size_t at = 0; at < out.size(); ++at) {
    // Row-major: the last axis varies fastest, so peel it off first.
    std::size_t rest = at;
    std::size_t cm = 0;
    for (std::size_t d = a.shape.size(); d-- > 0;) {
      cm += (rest % a.shape[d]) * stride[d];
      rest /= a.shape[d];
    }
    out[at] = a.data[cm];
  }
  return out;
}

std::optional<fb::NpyArray> read(const std::filesystem::path &dir, const std::string &array) {
  auto a = fb::read_npy(dir / (array + ".npy"));
  EXPECT_TRUE(a) << dir << "/" << array << ": " << (a ? "" : a.error().message);
  if (!a) {
    return std::nullopt;
  }
  return std::move(*a);
}

std::optional<fb::NpyArray> read(const Case &c, const std::string &array) {
  return read(kDataDir / c.name, array);
}

// (tu|vw) as ActiveSpaceHamiltonian::eri holds it, t + n (u + n (v + n w)),
// from the physicist h2e[p, q, r, s] = (pr|qs) at p + n (q + n (r + n s)):
// (tu|vw) = h2e[t, v, u, w].
std::vector<double> eri_of(const fb::NpyArray &h2e, std::size_t n) {
  std::vector<double> eri(n * n * n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      for (std::size_t v = 0; v < n; ++v) {
        for (std::size_t w = 0; w < n; ++w) {
          eri[t + n * (u + n * (v + n * w))] = h2e.data[t + n * (v + n * (u + n * w))];
        }
      }
    }
  }
  return eri;
}

// One case's inputs and PySCF's answers, row-major: the CI vector alpha-major,
// eri as ActiveSpaceHamiltonian::eri holds (tu|vw), from the physicist h2e.
struct Golden {
  std::vector<double> ci;
  std::vector<double> eri;
  std::vector<double> dm3;
  std::vector<double> f3ca;
  std::vector<double> f3ac;
};

std::optional<Golden> read_golden(const Case &c) {
  const auto ci = read(c, "ci");
  const auto h2e = read(c, "h2e");
  const auto dm3 = read(c, "dm3");
  const auto f3ca = read(c, "f3ca");
  const auto f3ac = read(c, "f3ac");
  if (!ci || !h2e || !dm3 || !f3ca || !f3ac) {
    return std::nullopt;
  }
  return Golden{row_major(*ci), eri_of(*h2e, static_cast<std::size_t>(c.n)), row_major(*dm3),
                row_major(*f3ca), row_major(*f3ac)};
}

fb::Buffers::device<double> on_device(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  // The source may be a temporary.
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> on_host(fb::DeviceResources &res, const fb::Buffers::device<double> &device) {
  std::vector<double> host(device.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

double max_abs(std::span<const double> a) {
  double worst = 0.0;
  for (const double x : a) {
    worst = std::max(worst, std::abs(x));
  }
  return worst;
}

// max|a - b| / max|b|.
double rel_diff(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst / max_abs(b);
}

// One set for both spins, the case's inputs on the device, and n⁶ outputs.
struct Setup {
  fb::ci::DeviceStringSet strings;
  fb::Buffers::device<double> ci;
  fb::Buffers::device<double> eri;
  fb::Buffers::device<double> dm3;
  fb::Buffers::device<double> f3ca;
  fb::Buffers::device<double> f3ac;
};

std::optional<Setup> set_up(fb::DeviceResources &res, std::int32_t n, std::int32_t per_spin,
                            std::span<const double> ci, std::span<const double> eri) {
  auto strings = fb::ci::DeviceStringSet::create(res, n, per_spin);
  EXPECT_TRUE(strings) << (strings ? "" : strings.error().message);
  if (!strings) {
    return std::nullopt;
  }
  const auto n6 = static_cast<std::size_t>(std::pow(n, 6));
  return Setup{std::move(*strings),
               on_device(res, ci),
               on_device(res, eri),
               fb::Buffers::device<double>(n6, res.device()),
               fb::Buffers::device<double>(n6, res.device()),
               fb::Buffers::device<double>(n6, res.device())};
}

struct Built {
  std::int64_t tiles = 0;
  std::vector<double> dm3;
  std::vector<double> f3ca;
  std::vector<double> f3ac;
};

std::optional<Built> build(fb::DeviceResources &res, Setup &s,
                           std::size_t max_bytes = std::numeric_limits<std::size_t>::max()) {
  const auto tiles = fb::nevpt2::rdms(res, s.strings, s.strings, s.ci.data(), s.eri.data(),
                                      s.dm3.data(), s.f3ca.data(), s.f3ac.data(), max_bytes);
  EXPECT_TRUE(tiles) << (tiles ? "" : tiles.error().message);
  if (!tiles) {
    return std::nullopt;
  }
  return Built{*tiles, on_host(res, s.dm3), on_host(res, s.f3ca), on_host(res, s.f3ac)};
}

void expect_matches(const Built &got, const Golden &want, double tol) {
  EXPECT_LE(rel_diff(got.dm3, want.dm3), tol) << "dm3";
  EXPECT_LE(rel_diff(got.f3ca, want.f3ca), tol) << "f3ca";
  EXPECT_LE(rel_diff(got.f3ac, want.f3ac), tol) << "f3ac";
}

TEST(NevptRdm, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const Case &c : kCases) {
    SCOPED_TRACE(c.name);
    const auto golden = read_golden(c);
    ASSERT_TRUE(golden);
    auto s = set_up(res, c.n, c.per_spin, golden->ci, golden->eri);
    ASSERT_TRUE(s);
    const auto got = build(res, *s);
    ASSERT_TRUE(got);
    // These spaces fit one tile on any card.
    EXPECT_EQ(got->tiles, 1);
    expect_matches(*got, *golden, kRelTol);
  }
}

// Each budget is rdms_workspace_bytes at width w + 1, less a byte: the widest
// tile that fits is w, so the tile count says the width was read back exactly.
// Widths with a one-determinant last tile, an unequal last tile wider than
// one, and every tile one determinant (on the 36-determinant cases).
TEST(NevptRdm, TinyBudgetsMatchOneTile) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const Case &c : kCases) {
    SCOPED_TRACE(c.name);
    const auto golden = read_golden(c);
    ASSERT_TRUE(golden);
    auto s = set_up(res, c.n, c.per_spin, golden->ci, golden->eri);
    ASSERT_TRUE(s);
    const auto one = build(res, *s);
    ASSERT_TRUE(one);
    const auto ndet = static_cast<std::int64_t>(golden->ci.size());
    // 36 = 7 · 5 + 1 = 10 · 3 + 6; 4900 = 23 · 213 + 1 = 64 · 76 + 36.
    const std::vector<std::int64_t> widths =
        ndet == 36 ? std::vector<std::int64_t>{1, 7, 10} : std::vector<std::int64_t>{23, 64};
    for (const std::int64_t w : widths) {
      SCOPED_TRACE("width " + std::to_string(w));
      const auto got = build(res, *s, fb::nevpt2::rdms_workspace_bytes(c.n, w + 1) - 1);
      ASSERT_TRUE(got);
      EXPECT_EQ(got->tiles, fb::math::idivup(ndet, w));
      EXPECT_LE(rel_diff(got->dm3, one->dm3), kTileRelTol) << "dm3";
      EXPECT_LE(rel_diff(got->f3ca, one->f3ca), kTileRelTol) << "f3ca";
      EXPECT_LE(rel_diff(got->f3ac, one->f3ac), kTileRelTol) << "f3ac";
      expect_matches(*got, *golden, kRelTol);
    }
  }
}

// A budget below one determinant's tile is refused before anything is
// queued: the outputs keep what they held and the workspace is not grown.
TEST(NevptRdm, BudgetBelowOneDeterminantIsUnsupported) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const Case &c = kCases.back();
  const auto golden = read_golden(c);
  ASSERT_TRUE(golden);
  auto s = set_up(res, c.n, c.per_spin, golden->ci, golden->eri);
  ASSERT_TRUE(s);
  const std::vector<double> sentinel(s->dm3.num_elements(), 0.25);
  for (auto *out : {&s->dm3, &s->f3ca, &s->f3ac}) {
    ASSERT_EQ(wwr::wwrMemcpyAsync(out->data(), sentinel.data(), sentinel.size() * sizeof(double),
                                  wwr::wwrMemcpyHostToDevice, res.stream()),
              wwr::wwrSuccess);
  }
  ASSERT_TRUE(res.sync().has_value());
  const std::size_t held = res.device_workspace(0).num_elements();
  for (const std::size_t budget : {std::size_t{0}, fb::nevpt2::rdms_workspace_bytes(c.n, 1) - 1}) {
    const auto got = fb::nevpt2::rdms(res, s->strings, s->strings, s->ci.data(), s->eri.data(),
                                      s->dm3.data(), s->f3ca.data(), s->f3ac.data(), budget);
    ASSERT_FALSE(got);
    EXPECT_EQ(got.error().kind, fb::ErrorKind::Unsupported) << got.error().message;
    EXPECT_NE(got.error().message.find("nevpt2::rdms:"), std::string::npos);
  }
  EXPECT_EQ(res.device_workspace(0).num_elements(), held);
  for (const auto *out : {&s->dm3, &s->f3ca, &s->f3ac}) {
    EXPECT_EQ(on_host(res, *out), sentinel);
  }
}

TEST(NevptRdm, Refusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  auto four = fb::ci::DeviceStringSet::create(res, 4, 2);
  auto five = fb::ci::DeviceStringSet::create(res, 5, 2);
  auto none = fb::ci::DeviceStringSet::create(res, 0, 0);
  ASSERT_TRUE(four && five && none);
  fb::Buffers::device<double> scratch(std::size_t{1} << 16, res.device());
  double *const p = scratch.data();
  const auto mixed = fb::nevpt2::rdms(res, *four, *five, p, p, p, p, p);
  ASSERT_FALSE(mixed);
  EXPECT_EQ(mixed.error().kind, fb::ErrorKind::InvalidConfig) << mixed.error().message;
  const auto empty = fb::nevpt2::rdms(res, *none, *none, p, p, p, p, p);
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().kind, fb::ErrorKind::InvalidConfig) << empty.error().message;
}

double dot(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::vector<double> random_vector(std::mt19937_64 &rng, std::size_t size) {
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  std::vector<double> v(size);
  for (double &x : v) {
    x = uniform(rng);
  }
  return v;
}

// a + sign · b.
std::vector<double> shifted(std::span<const double> a, std::span<const double> b, double sign) {
  std::vector<double> out(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    out[i] = a[i] + (sign * b[i]);
  }
  return out;
}

// The bars on the three outputs, on the host and the device.
struct Bars {
  std::vector<double> dm3;
  std::vector<double> f3ca;
  std::vector<double> f3ac;
};

// The bars against half the difference of two builds, and the scale the
// round-off is measured against: Σ_i |bar_i| |difference_i| / 2.
std::pair<double, double> contract_difference(const Bars &bars, const Built &plus,
                                              const Built &minus) {
  double value = 0.0;
  double scale = 0.0;
  const auto add = [&](std::span<const double> bar, std::span<const double> p,
                       std::span<const double> m) {
    for (std::size_t i = 0; i < bar.size(); ++i) {
      const double half = 0.5 * (p[i] - m[i]);
      value += bar[i] * half;
      scale += std::abs(bar[i] * half);
    }
  };
  add(bars.dm3, plus.dm3, minus.dm3);
  add(bars.f3ca, plus.f3ca, minus.f3ca);
  add(bars.f3ac, plus.f3ac, minus.f3ac);
  return {value, scale};
}

struct Adjoint {
  std::int64_t tiles = 0;
  std::vector<double> vector_bar;
  std::vector<double> eri_bar;
};

std::optional<Adjoint> adjoint(fb::DeviceResources &res, Setup &s,
                               const fb::Buffers::device<double> &dm3_bar,
                               const fb::Buffers::device<double> &f3ca_bar,
                               const fb::Buffers::device<double> &f3ac_bar,
                               std::size_t max_bytes = std::numeric_limits<std::size_t>::max()) {
  fb::Buffers::device<double> vector_bar(s.ci.num_elements(), res.device());
  fb::Buffers::device<double> eri_bar(s.eri.num_elements(), res.device());
  const auto tiles = fb::nevpt2::rdms_adjoint(res, s.strings, s.strings, s.ci.data(), s.eri.data(),
                                              dm3_bar.data(), f3ca_bar.data(), f3ac_bar.data(),
                                              vector_bar.data(), eri_bar.data(), max_bytes);
  EXPECT_TRUE(tiles) << (tiles ? "" : tiles.error().message);
  if (!tiles) {
    return std::nullopt;
  }
  return Adjoint{*tiles, on_host(res, vector_bar), on_host(res, eri_bar)};
}

// The issue's acceptance: on CAS(4,4) and CAS(6,6), from a random vector, a
// random (tu|vw) with no symmetry (the adjoint is by each stored element) and
// random bars, ⟨y, u⟩ along a random u and ⟨h2ē, δ⟩ along a random δ each
// match the bars on (rdms(+) − rdms(−)) / 2 to round-off; at one tile and at
// a width below the determinant count, with an unequal last tile, which also
// agree with each other.
TEST(NevptRdm, AdjointMatchesPolarization) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  struct Space {
    std::int32_t n;
    std::int32_t per_spin;
    std::size_t strings;
    std::int64_t width; // below the determinant count, not dividing it
  };
  // 36 = 7 · 5 + 1; 400 = 23 · 17 + 9.
  for (const Space sp : {Space{4, 2, 6, 7}, Space{6, 3, 20, 23}}) {
    SCOPED_TRACE("CAS(" + std::to_string(sp.n) + "," + std::to_string(sp.n) + ")");
    const auto n = static_cast<std::size_t>(sp.n);
    const std::size_t ndet = sp.strings * sp.strings;
    const std::size_t n6 = n * n * n * n * n * n;
    std::mt19937_64 rng{static_cast<std::uint64_t>(557 + sp.n)};
    const std::vector<double> ci = random_vector(rng, ndet);
    const std::vector<double> eri = random_vector(rng, n * n * n * n);
    const std::vector<double> u = random_vector(rng, ndet);
    const std::vector<double> delta = random_vector(rng, n * n * n * n);
    const Bars bars{random_vector(rng, n6), random_vector(rng, n6), random_vector(rng, n6)};
    const auto dm3_bar = on_device(res, bars.dm3);
    const auto f3ca_bar = on_device(res, bars.f3ca);
    const auto f3ac_bar = on_device(res, bars.f3ac);

    auto s = set_up(res, sp.n, sp.per_spin, ci, eri);
    ASSERT_TRUE(s);
    ASSERT_EQ(s->strings.size(), static_cast<std::int32_t>(sp.strings));
    const auto one = adjoint(res, *s, dm3_bar, f3ca_bar, f3ac_bar);
    ASSERT_TRUE(one);
    EXPECT_EQ(one->tiles, 1);
    const auto tiled = adjoint(res, *s, dm3_bar, f3ca_bar, f3ac_bar,
                               fb::nevpt2::rdms_adjoint_workspace_bytes(sp.n, sp.width + 1) - 1);
    ASSERT_TRUE(tiled);
    EXPECT_EQ(tiled->tiles, fb::math::idivup(static_cast<std::int64_t>(ndet), sp.width));
    EXPECT_LE(rel_diff(tiled->vector_bar, one->vector_bar), kTileRelTol) << "y";
    EXPECT_LE(rel_diff(tiled->eri_bar, one->eri_bar), kTileRelTol) << "h2e bar";

    // Along u: the builds at c ± u, at the case's (tu|vw).
    auto plus_c = set_up(res, sp.n, sp.per_spin, shifted(ci, u, 1.0), eri);
    auto minus_c = set_up(res, sp.n, sp.per_spin, shifted(ci, u, -1.0), eri);
    ASSERT_TRUE(plus_c && minus_c);
    const auto bp = build(res, *plus_c);
    const auto bm = build(res, *minus_c);
    ASSERT_TRUE(bp && bm);
    const auto [along_u, scale_u] = contract_difference(bars, *bp, *bm);
    // Along δ: the builds at (tu|vw) ± δ, at the case's c.
    auto plus_e = set_up(res, sp.n, sp.per_spin, ci, shifted(eri, delta, 1.0));
    auto minus_e = set_up(res, sp.n, sp.per_spin, ci, shifted(eri, delta, -1.0));
    ASSERT_TRUE(plus_e && minus_e);
    const auto ep = build(res, *plus_e);
    const auto em = build(res, *minus_e);
    ASSERT_TRUE(ep && em);
    const auto [along_delta, scale_delta] = contract_difference(bars, *ep, *em);

    for (const Adjoint *a : {&*one, &*tiled}) {
      SCOPED_TRACE(a->tiles == 1 ? "one tile" : "tiled");
      const double yu = dot(a->vector_bar, u);
      const double hd = dot(a->eri_bar, delta);
      EXPECT_LE(std::abs(yu - along_u), kAdjointRelTol * scale_u)
          << "<y, u> " << yu << " against " << along_u << " (scale " << scale_u << ")";
      EXPECT_LE(std::abs(hd - along_delta), kAdjointRelTol * scale_delta)
          << "<h2e bar, delta> " << hd << " against " << along_delta << " (scale " << scale_delta
          << ")";
    }
  }
}

// The adjoint's own refusals: a budget below one determinant's tile, before
// anything is queued (the outputs keep what they held, the workspace is not
// grown), and the sets rdms refuses.
TEST(NevptRdm, AdjointRefusals) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t kN = 4;
  auto four = fb::ci::DeviceStringSet::create(res, kN, 2);
  auto five = fb::ci::DeviceStringSet::create(res, 5, 2);
  auto none = fb::ci::DeviceStringSet::create(res, 0, 0);
  ASSERT_TRUE(four && five && none);
  const std::vector<double> sentinel(std::size_t{1} << 12, 0.25);
  auto scratch = on_device(res, sentinel);
  double *const p = scratch.data();
  const std::size_t held = res.device_workspace(0).num_elements();
  for (const std::size_t budget :
       {std::size_t{0}, fb::nevpt2::rdms_adjoint_workspace_bytes(kN, 1) - 1}) {
    const auto got = fb::nevpt2::rdms_adjoint(res, *four, *four, p, p, p, p, p, p, p, budget);
    ASSERT_FALSE(got);
    EXPECT_EQ(got.error().kind, fb::ErrorKind::Unsupported) << got.error().message;
    EXPECT_NE(got.error().message.find("nevpt2::rdms_adjoint:"), std::string::npos);
  }
  EXPECT_EQ(res.device_workspace(0).num_elements(), held);
  EXPECT_EQ(on_host(res, scratch), sentinel);
  const auto mixed = fb::nevpt2::rdms_adjoint(res, *four, *five, p, p, p, p, p, p, p);
  ASSERT_FALSE(mixed);
  EXPECT_EQ(mixed.error().kind, fb::ErrorKind::InvalidConfig) << mixed.error().message;
  const auto empty = fb::nevpt2::rdms_adjoint(res, *none, *none, p, p, p, p, p, p, p);
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().kind, fb::ErrorKind::InvalidConfig) << empty.error().message;
}

// The cases nevpt2_transition/README.md lists: a bra and a different ket over
// n orbitals and (n_alpha, n_beta) electrons, the second with n_alpha ≠ n_beta.
struct TransitionCase {
  const char *name;
  std::int32_t n;
  std::int32_t n_alpha;
  std::int32_t n_beta;
};
constexpr std::array kTransitionCases{TransitionCase{"n2_cas44", 4, 2, 2},
                                      TransitionCase{"n2_cas6_3a2b", 6, 3, 2}};

// One transition case's inputs and PySCF's five tensors, row-major.
struct TransitionGolden {
  std::vector<double> bra;
  std::vector<double> ket;
  std::vector<double> eri;
  std::vector<double> dm1;
  std::vector<double> dm2;
  std::vector<double> dm3;
  std::vector<double> f3ca;
  std::vector<double> f3ac;
};

std::optional<TransitionGolden> read_transition(const TransitionCase &c) {
  const std::filesystem::path dir = kTransitionDir / c.name;
  const auto bra = read(dir, "bra");
  const auto ket = read(dir, "ket");
  const auto h2e = read(dir, "h2e");
  const auto dm1 = read(dir, "dm1");
  const auto dm2 = read(dir, "dm2");
  const auto dm3 = read(dir, "dm3");
  const auto f3ca = read(dir, "f3ca");
  const auto f3ac = read(dir, "f3ac");
  if (!bra || !ket || !h2e || !dm1 || !dm2 || !dm3 || !f3ca || !f3ac) {
    return std::nullopt;
  }
  return TransitionGolden{row_major(*bra), row_major(*ket), eri_of(*h2e, c.n), row_major(*dm1),
                          row_major(*dm2), row_major(*dm3), row_major(*f3ca),  row_major(*f3ac)};
}

// The two spins' sets, a transition case's inputs on the device, and the five
// outputs.
struct TransitionSetup {
  fb::ci::DeviceStringSet alpha;
  fb::ci::DeviceStringSet beta;
  fb::Buffers::device<double> bra;
  fb::Buffers::device<double> ket;
  fb::Buffers::device<double> eri;
  fb::Buffers::device<double> dm1;
  fb::Buffers::device<double> dm2;
  fb::Buffers::device<double> dm3;
  fb::Buffers::device<double> f3ca;
  fb::Buffers::device<double> f3ac;
};

std::optional<TransitionSetup> set_up_transition(fb::DeviceResources &res, const TransitionCase &c,
                                                 const TransitionGolden &g) {
  auto alpha = fb::ci::DeviceStringSet::create(res, c.n, c.n_alpha);
  auto beta = fb::ci::DeviceStringSet::create(res, c.n, c.n_beta);
  EXPECT_TRUE(alpha && beta);
  if (!alpha || !beta) {
    return std::nullopt;
  }
  const auto n = static_cast<std::size_t>(c.n);
  const std::size_t n4 = n * n * n * n;
  return TransitionSetup{std::move(*alpha),
                         std::move(*beta),
                         on_device(res, g.bra),
                         on_device(res, g.ket),
                         on_device(res, g.eri),
                         fb::Buffers::device<double>(n * n, res.device()),
                         fb::Buffers::device<double>(n4, res.device()),
                         fb::Buffers::device<double>(n4 * n * n, res.device()),
                         fb::Buffers::device<double>(n4 * n * n, res.device()),
                         fb::Buffers::device<double>(n4 * n * n, res.device())};
}

struct TransitionBuilt {
  std::int64_t tiles = 0;
  std::vector<double> dm1;
  std::vector<double> dm2;
  std::vector<double> dm3;
  std::vector<double> f3ca;
  std::vector<double> f3ac;
};

// The bra/ket overload into @p s's outputs, between @p bra and @p ket.
std::optional<TransitionBuilt>
build_transition(fb::DeviceResources &res, TransitionSetup &s, const double *bra, const double *ket,
                 std::size_t max_bytes = std::numeric_limits<std::size_t>::max()) {
  const auto tiles =
      fb::nevpt2::rdms(res, s.alpha, s.beta, bra, ket, s.eri.data(), s.dm1.data(), s.dm2.data(),
                       s.dm3.data(), s.f3ca.data(), s.f3ac.data(), max_bytes);
  EXPECT_TRUE(tiles) << (tiles ? "" : tiles.error().message);
  if (!tiles) {
    return std::nullopt;
  }
  return TransitionBuilt{*tiles,
                         on_host(res, s.dm1),
                         on_host(res, s.dm2),
                         on_host(res, s.dm3),
                         on_host(res, s.f3ca),
                         on_host(res, s.f3ac)};
}

void expect_matches(const TransitionBuilt &got, const TransitionGolden &want, double tol) {
  EXPECT_LE(rel_diff(got.dm1, want.dm1), tol) << "dm1";
  EXPECT_LE(rel_diff(got.dm2, want.dm2), tol) << "dm2";
  EXPECT_LE(rel_diff(got.dm3, want.dm3), tol) << "dm3";
  EXPECT_LE(rel_diff(got.f3ca, want.f3ca), tol) << "f3ca";
  EXPECT_LE(rel_diff(got.f3ac, want.f3ac), tol) << "f3ac";
}

TEST(NevptRdm, TransitionMatchesPyscf) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const TransitionCase &c : kTransitionCases) {
    SCOPED_TRACE(c.name);
    const auto golden = read_transition(c);
    ASSERT_TRUE(golden);
    auto s = set_up_transition(res, c, *golden);
    ASSERT_TRUE(s);
    const auto got = build_transition(res, *s, s->bra.data(), s->ket.data());
    ASSERT_TRUE(got);
    EXPECT_EQ(got->tiles, 1);
    expect_matches(*got, *golden, kRelTol);
  }
}

// As TinyBudgetsMatchOneTile, against the transition's own bytes: widths with
// every tile one determinant, a one-determinant last tile and an unequal one
// wider than one. And its budget for one determinant, less a byte, is refused,
// though the one-vector build at that budget is not.
TEST(NevptRdm, TransitionTinyBudgetsMatchOneTile) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const TransitionCase &c : kTransitionCases) {
    SCOPED_TRACE(c.name);
    const auto golden = read_transition(c);
    ASSERT_TRUE(golden);
    auto s = set_up_transition(res, c, *golden);
    ASSERT_TRUE(s);
    const auto one = build_transition(res, *s, s->bra.data(), s->ket.data());
    ASSERT_TRUE(one);
    const auto ndet = static_cast<std::int64_t>(golden->bra.size());
    // 36 = 7 · 5 + 1 = 10 · 3 + 6; 300 = 7 · 42 + 6 = 23 · 13 + 1.
    const std::vector<std::int64_t> widths =
        ndet == 36 ? std::vector<std::int64_t>{1, 7, 10} : std::vector<std::int64_t>{7, 23};
    for (const std::int64_t w : widths) {
      SCOPED_TRACE("width " + std::to_string(w));
      const auto got = build_transition(res, *s, s->bra.data(), s->ket.data(),
                                        fb::nevpt2::rdms_workspace_bytes(c.n, w + 1, true) - 1);
      ASSERT_TRUE(got);
      EXPECT_EQ(got->tiles, fb::math::idivup(ndet, w));
      EXPECT_LE(rel_diff(got->dm1, one->dm1), kTileRelTol) << "dm1";
      EXPECT_LE(rel_diff(got->dm2, one->dm2), kTileRelTol) << "dm2";
      EXPECT_LE(rel_diff(got->dm3, one->dm3), kTileRelTol) << "dm3";
      EXPECT_LE(rel_diff(got->f3ca, one->f3ca), kTileRelTol) << "f3ca";
      EXPECT_LE(rel_diff(got->f3ac, one->f3ac), kTileRelTol) << "f3ac";
      expect_matches(*got, *golden, kRelTol);
    }

    const std::size_t held = res.device_workspace(0).num_elements();
    const std::size_t below = fb::nevpt2::rdms_workspace_bytes(c.n, 1, true) - 1;
    const auto refused = fb::nevpt2::rdms(res, s->alpha, s->beta, s->bra.data(), s->ket.data(),
                                          s->eri.data(), s->dm1.data(), s->dm2.data(),
                                          s->dm3.data(), s->f3ca.data(), s->f3ac.data(), below);
    ASSERT_FALSE(refused);
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::Unsupported) << refused.error().message;
    EXPECT_EQ(res.device_workspace(0).num_elements(), held);
    const auto single = fb::nevpt2::rdms(res, s->alpha, s->beta, s->bra.data(), s->eri.data(),
                                         s->dm3.data(), s->f3ca.data(), s->f3ac.data(), below);
    EXPECT_TRUE(single) << (single ? "" : single.error().message);
  }
}

// bra == ket, one pointer, is the one-vector build: dm3, f3ca and f3ac bit for
// bit, at one tile and at many, beside the state's own dm1 and dm2. Two
// pointers holding the same vector take the transition's path, which agrees
// to round-off.
TEST(NevptRdm, BraEqualKetIsTheOneVectorBuild) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  for (const Case &c : kCases) {
    SCOPED_TRACE(c.name);
    const auto golden = read_golden(c);
    const auto dm1_golden = read(c, "dm1");
    const auto dm2_golden = read(c, "dm2");
    ASSERT_TRUE(golden && dm1_golden && dm2_golden);
    auto s = set_up(res, c.n, c.per_spin, golden->ci, golden->eri);
    ASSERT_TRUE(s);
    const auto copy = on_device(res, golden->ci);
    const auto n = static_cast<std::size_t>(c.n);
    fb::Buffers::device<double> dm1(n * n, res.device());
    fb::Buffers::device<double> dm2(n * n * n * n, res.device());
    const auto ndet = static_cast<std::int64_t>(golden->ci.size());
    for (const std::int64_t w : {ndet, ndet == 36 ? std::int64_t{7} : std::int64_t{23}}) {
      SCOPED_TRACE("width " + std::to_string(w));
      const std::size_t budget = fb::nevpt2::rdms_workspace_bytes(c.n, w + 1) - 1;
      const auto one = build(res, *s, budget);
      ASSERT_TRUE(one);
      EXPECT_EQ(one->tiles, fb::math::idivup(ndet, w));
      for (const double *const ket : {static_cast<const double *>(s->ci.data()), copy.data()}) {
        const bool same = ket == s->ci.data();
        SCOPED_TRACE(same ? "one pointer" : "two pointers");
        const auto tiles = fb::nevpt2::rdms(
            res, s->strings, s->strings, s->ci.data(), ket, s->eri.data(), dm1.data(), dm2.data(),
            s->dm3.data(), s->f3ca.data(), s->f3ac.data(),
            same ? budget : fb::nevpt2::rdms_workspace_bytes(c.n, w + 1, true) - 1);
        ASSERT_TRUE(tiles) << (tiles ? "" : tiles.error().message);
        EXPECT_EQ(*tiles, one->tiles);
        const Built got{*tiles, on_host(res, s->dm3), on_host(res, s->f3ca), on_host(res, s->f3ac)};
        if (same) {
          EXPECT_EQ(got.dm3, one->dm3) << "dm3";
          EXPECT_EQ(got.f3ca, one->f3ca) << "f3ca";
          EXPECT_EQ(got.f3ac, one->f3ac) << "f3ac";
        } else {
          EXPECT_LE(rel_diff(got.dm3, one->dm3), kTileRelTol) << "dm3";
          EXPECT_LE(rel_diff(got.f3ca, one->f3ca), kTileRelTol) << "f3ca";
          EXPECT_LE(rel_diff(got.f3ac, one->f3ac), kTileRelTol) << "f3ac";
        }
        EXPECT_LE(rel_diff(on_host(res, dm1), row_major(*dm1_golden)), kRelTol) << "dm1";
        EXPECT_LE(rel_diff(on_host(res, dm2), row_major(*dm2_golden)), kRelTol) << "dm2";
      }
    }
  }
}

// The acceptance's measurement: CAS(12,12), 853 776 determinants, at the
// default budget, from a random unit vector and an eri with (tu|vw)'s
// symmetries, (tu|vw) = Σ_L B_Ltu B_Lvw over a random symmetric B. Prints the
// tile count and the pool's peak; checks Σ_pqr dm3[p, p, q, q, r, r] = N³.
TEST(NevptRdm, Cas1212TilesAtTheDefaultBudget) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  constexpr std::int32_t kN = 12;
  constexpr std::int32_t kPerSpin = 6;
  constexpr std::size_t kAux = 20;
  constexpr std::size_t n = kN;
  fb::DeviceResources res{0};
  const std::size_t strings = 924;
  std::mt19937_64 rng{1212};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};
  std::vector<double> ci(strings * strings);
  for (double &x : ci) {
    x = uniform(rng);
  }
  const double norm = std::sqrt(std::inner_product(ci.begin(), ci.end(), ci.begin(), 0.0));
  for (double &x : ci) {
    x /= norm;
  }
  std::vector<double> b(kAux * n * n);
  for (std::size_t l = 0; l < kAux; ++l) {
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t u = 0; u <= t; ++u) {
        b[(l * n + t) * n + u] = b[(l * n + u) * n + t] = 0.3 * uniform(rng);
      }
    }
  }
  std::vector<double> eri(n * n * n * n, 0.0);
  for (std::size_t tu = 0; tu < n * n; ++tu) {
    for (std::size_t vw = 0; vw < n * n; ++vw) {
      for (std::size_t l = 0; l < kAux; ++l) {
        eri[tu + n * n * vw] += b[l * n * n + tu] * b[l * n * n + vw];
      }
    }
  }
  auto s = set_up(res, kN, kPerSpin, ci, eri);
  ASSERT_TRUE(s);
  ASSERT_EQ(s->strings.size(), static_cast<std::int32_t>(strings));

  wwr::wwrMemPool_t pool = res.device()->pool().get();
  std::uint64_t zero = 0;
  ASSERT_EQ(wwr::wwrMemPoolSetAttribute(pool, wwr::wwrMemPoolAttrUsedMemHigh, &zero),
            wwr::wwrSuccess);
  std::uint64_t before = 0;
  ASSERT_EQ(wwr::wwrMemPoolGetAttribute(pool, wwr::wwrMemPoolAttrUsedMemCurrent, &before),
            wwr::wwrSuccess);
  const auto start = std::chrono::steady_clock::now();
  const auto got = build(res, *s);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ASSERT_TRUE(got);
  std::uint64_t high = 0;
  ASSERT_EQ(wwr::wwrMemPoolGetAttribute(pool, wwr::wwrMemPoolAttrUsedMemHigh, &high),
            wwr::wwrSuccess);
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  ASSERT_EQ(wwr::wwrMemGetInfo(&free_bytes, &total_bytes), wwr::wwrSuccess);
  const std::size_t workspace = res.device_workspace(0).num_elements();
  std::cout << "CAS(12,12): " << got->tiles << " tiles over " << strings * strings
            << " determinants; workspace " << workspace << " bytes; pool peak " << high
            << " bytes (" << before << " held before the build: inputs, outputs, strings); card "
            << total_bytes << " bytes; " << seconds << " s with the readback\n";
  RecordProperty("tiles", std::to_string(got->tiles));
  RecordProperty("pool_peak_bytes", std::to_string(high));
  EXPECT_GE(got->tiles, 1);

  double trace = 0.0;
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        trace += got->dm3[((((p * n + p) * n + q) * n + q) * n + r) * n + r];
      }
    }
  }
  const double electrons = 2.0 * kPerSpin;
  EXPECT_NEAR(trace, std::pow(electrons, 3), kTraceRelTol * std::pow(electrons, 3));
}

} // namespace
