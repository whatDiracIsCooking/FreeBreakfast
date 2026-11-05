// GPU cases for nevpt2::builders_tangent: the forward tangent of the per-root
// builders (every EnergyInputs field, the MO blocks, (tu|vw), and f's
// core-core and virtual-virtual blocks) along a block of two random
// directions (h_dot, B_dot, c_dot), each against a Richardson ladder of the
// builders themselves, every field; and the value's f blocks against
// zc_adjoint's f, which the ladder cannot see (it checks the tangent of
// whatever f the builders form).
//
// The point is synthetic, not a converged CASSCF: the builders are
// polynomials in (h, B, c), with no denominator and no stationarity to
// respect, so a random point is as good a test as any and runs in seconds.
// The tangent is taken at a random non-orthogonal C over AO integrals, and
// the ladder runs in the MO basis itself (C = 1, h and B the MO ones moved
// along the direction), so it also checks that the directions are read in the
// MO basis of C.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.ci;
import free_breakfast.nevpt2;

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

// CAS(4,4) over two core and five virtual orbitals: every class of the
// builders has extent, and n^6 is small.
constexpr std::int32_t kNaux = 24;
constexpr std::int32_t kNcore = 2;
constexpr std::int32_t kNcas = 4;
constexpr std::int32_t kNvirt = 5;
constexpr std::int32_t kNmo = kNcore + kNcas + kNvirt;
constexpr std::int32_t kElectronsPerSpin = 2;
// nevpt2_gradient_gpu.cpp's ladder: t = 1e-2 / 2^k over ten rungs, the
// extrapolant within 1e-7 of the prediction, relative here to the field's
// largest element; B_dot a tenth the length of the others, as there.
constexpr double kLadderT0 = 1e-2;
constexpr std::int32_t kRungs = 10;
constexpr double kLadderTolerance = 1e-7;
constexpr double kBScale = 0.1;
// A field quadratic along the direction (the RDMs: c enters twice, nothing
// else) has exact central differences, so its ladder is flat to rounding,
// about 1e-16 |F| / t <= 1e-11 |F| at the finest rung, and its ratios are
// noise. Below this (relative to the field's largest element) the ladder is
// read as flat: its steadiest pair still meets the same 1e-7 bar.
constexpr double kFlat = 1e-9;
// The value's f against zc_adjoint's Phi and the orbital energies: the same
// sums, reassociated by the einsums' atomics.
constexpr double kSameSumTolerance = 1e-12;

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
  EXPECT_EQ(wwr::wwrMemcpyAsync(out.data(), host.data(), host.size_bytes(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return out;
}

// A point (h, B, C, c) on the host: AO h and B (naux x nao^2, (L, m, n) at
// L + naux (m + nao n)), C nao x nmo, c over the determinants.
struct Point {
  std::vector<double> h;
  std::vector<double> b;
  std::vector<double> c;
  std::vector<double> vector;
};

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

// naux random symmetric n x n slices, (L, p, q) at L + naux (p + n q).
std::vector<double> symmetric_b(std::mt19937_64 &engine, const std::size_t n, const double scale) {
  const auto naux = static_cast<std::size_t>(kNaux);
  std::vector<double> b(naux * n * n);
  for (std::size_t l = 0; l < naux; ++l) {
    const std::vector<double> slice = symmetric(engine, n, scale);
    for (std::size_t pq = 0; pq < n * n; ++pq) {
      b[l + (naux * pq)] = slice[pq];
    }
  }
  return b;
}

double norm2(std::span<const double> v) {
  return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
}

Point random_point(const std::size_t ndet) {
  const auto n = static_cast<std::size_t>(kNmo);
  std::mt19937_64 engine(806);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  Point p{.h = symmetric(engine, n, 1.0),
          .b = symmetric_b(engine, n, 0.5),
          .c = std::vector<double>(n * n),
          .vector = std::vector<double>(ndet)};
  // C = 1 + 0.1 R: square, invertible, not orthogonal.
  for (std::size_t k = 0; k < n * n; ++k) {
    p.c[k] = (k % (n + 1) == 0 ? 1.0 : 0.0) + (0.1 * uniform(engine));
  }
  for (double &v : p.vector) {
    v = uniform(engine);
  }
  const double scale = 1.0 / norm2(p.vector);
  for (double &v : p.vector) {
    v *= scale;
  }
  return p;
}

// C^T M C for the n x n M.
std::vector<double> to_mo(std::span<const double> c, std::span<const double> m) {
  const auto n = static_cast<std::size_t>(kNmo);
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

// The point in its own MO basis: C^T h C, C^T B^L C and C = 1.
Point in_mo(const Point &p) {
  const auto n = static_cast<std::size_t>(kNmo);
  const auto naux = static_cast<std::size_t>(kNaux);
  Point mo{.h = to_mo(p.c, p.h),
           .b = std::vector<double>(p.b.size()),
           .c = std::vector<double>(n * n, 0.0),
           .vector = p.vector};
  std::vector<double> slice(n * n);
  for (std::size_t l = 0; l < naux; ++l) {
    for (std::size_t pq = 0; pq < n * n; ++pq) {
      slice[pq] = p.b[l + (naux * pq)];
    }
    const std::vector<double> moved = to_mo(p.c, slice);
    for (std::size_t pq = 0; pq < n * n; ++pq) {
      mo.b[l + (naux * pq)] = moved[pq];
    }
  }
  for (std::size_t k = 0; k < n; ++k) {
    mo.c[k * (n + 1)] = 1.0;
  }
  return mo;
}

// A direction (h_dot, B_dot, c_dot) in the MO basis: h_dot symmetric, B_dot
// symmetric in its pair, c_dot generic, with a component along c (0.3 c) and
// one across it.
struct Direction {
  std::vector<double> h;
  std::vector<double> b;
  std::vector<double> c;
};

Direction random_direction(const std::uint64_t seed, std::span<const double> vector) {
  const auto n = static_cast<std::size_t>(kNmo);
  std::mt19937_64 engine(seed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  Direction d{.h = symmetric(engine, n, 1.0),
              .b = symmetric_b(engine, n, kBScale),
              .c = std::vector<double>(vector.size())};
  for (double &v : d.c) {
    v = uniform(engine);
  }
  const double along = std::inner_product(d.c.begin(), d.c.end(), vector.begin(), 0.0);
  const double scale = 1.0 / std::sqrt(std::max(0.0, std::pow(norm2(d.c), 2) - (along * along)));
  for (std::size_t k = 0; k < d.c.size(); ++k) {
    d.c[k] = ((d.c[k] - (along * vector[k])) * scale) + (0.3 * vector[k]);
  }
  return d;
}

// Every field of a BuilderOutputs, on the host, under its name.
using Fields = std::vector<std::pair<std::string, std::vector<double>>>;

Fields fields(fb::DeviceResources &res, const nevpt2::BuilderOutputs &o) {
  const auto naux = static_cast<std::size_t>(kNaux);
  const auto nc = static_cast<std::size_t>(kNcore);
  const auto n = static_cast<std::size_t>(kNcas);
  const auto nv = static_cast<std::size_t>(kNvirt);
  const std::size_t n2 = n * n;
  const std::size_t n4 = n2 * n2;
  const std::size_t n6 = n4 * n2;
  const auto get = [&](const fb::Buffers::device<double> &buffer, const std::size_t count) {
    EXPECT_EQ(buffer.num_elements(), count);
    return download(res, buffer.data(), std::min(buffer.num_elements(), count));
  };
  return {{"B_aa", get(o.blocks.aa, naux * n2)},
          {"B_ca", get(o.blocks.ca, naux * nc * n)},
          {"B_va", get(o.blocks.va, naux * nv * n)},
          {"B_cv", get(o.blocks.cv, naux * nc * nv)},
          {"B_vc", get(o.blocks.vc, naux * nv * nc)},
          {"eri", get(o.eri, n4)},
          {"h1e", get(o.h1e, n2)},
          {"h2e", get(o.h2e, n4)},
          {"e_core", get(o.e_core, nc)},
          {"e_virt", get(o.e_virt, nv)},
          {"h1e_v_sr", get(o.h1e_v_sr, nv * n)},
          {"h1e_v_si", get(o.h1e_v_si, n * nc)},
          {"h1e_v_sir", get(o.h1e_v_sir, nv * nc)},
          {"dm1", get(o.dm1, n2)},
          {"dm2", get(o.dm2, n4)},
          {"dm3", get(o.dm3, n6)},
          {"f3ca", get(o.f3ca, n6)},
          {"f3ac", get(o.f3ac, n6)},
          {"f_core", get(o.f_core, nc * nc)},
          {"f_virt", get(o.f_virt, nv * nv)}};
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

// The builders at the MO point @p mo moved by @p t along @p d, on the host.
Fields moved_fields(fb::DeviceResources &res, const fb::ci::DeviceStringSet &strings,
                    const Point &mo, const Direction &d, const double t) {
  const auto shift = [t](std::span<const double> x, std::span<const double> dx) {
    std::vector<double> out(x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
      out[k] = x[k] + (t * dx[k]);
    }
    return out;
  };
  const auto h = upload(res, shift(mo.h, d.h));
  const auto b = upload(res, shift(mo.b, d.b));
  const auto c = upload(res, mo.c);
  const auto vector = upload(res, shift(mo.vector, d.c));
  auto got = nevpt2::builders_tangent(res, strings, strings, kNaux, kNmo, kNmo, b.data(), c.data(),
                                      h.data(), kNcore, kNcas, vector.data(), 0, nullptr, nullptr,
                                      nullptr);
  EXPECT_TRUE(got.has_value()) << got.error().message;
  return got.has_value() ? fields(res, got->value) : Fields{};
}

// One field's ladder: central differences D_k at t = t0 / 2^k, the ratios of
// successive changes |D_k - D_{k+1}| / |D_{k+1} - D_{k+2}| (max norms), the
// longest run of them in (3.2, 4.8), and over that run the steadiest
// Richardson extrapolant R_k = (4 D_{k+2} - D_{k+1}) / 3, the finer of the
// neighbouring pair that agree best (nevpt2_gradient_gpu.cpp's
// richardson_ladder, steadiest); over every rung when the ladder is flat.
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

TEST(Nevpt2BuildersTangent, MatchesLadder) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  auto strings = fb::ci::DeviceStringSet::create(res, kNcas, kElectronsPerSpin);
  ASSERT_TRUE(strings.has_value()) << strings.error().message;
  const auto ndet = static_cast<std::size_t>(strings->size()) * strings->size();
  const Point point = random_point(ndet);
  const Point mo = in_mo(point);
  const std::array<Direction, 2> dirs{random_direction(8061, point.vector),
                                      random_direction(8062, point.vector)};

  // The block call, at the AO point.
  std::vector<double> h_dot;
  std::vector<double> b_dot;
  std::vector<double> c_dot;
  for (const Direction &d : dirs) {
    h_dot.insert(h_dot.end(), d.h.begin(), d.h.end());
    b_dot.insert(b_dot.end(), d.b.begin(), d.b.end());
    c_dot.insert(c_dot.end(), d.c.begin(), d.c.end());
  }
  const auto h = upload(res, point.h);
  const auto b = upload(res, point.b);
  const auto c = upload(res, point.c);
  const auto vector = upload(res, point.vector);
  const auto hd = upload(res, h_dot);
  const auto bd = upload(res, b_dot);
  const auto cd = upload(res, c_dot);
  auto got = nevpt2::builders_tangent(
      res, *strings, *strings, kNaux, kNmo, kNmo, b.data(), c.data(), h.data(), kNcore, kNcas,
      vector.data(), static_cast<std::int32_t>(dirs.size()), hd.data(), bd.data(), cd.data());
  ASSERT_TRUE(got.has_value()) << got.error().message;
  ASSERT_EQ(got->tangents.size(), dirs.size());

  double worst = 0.0;
  for (std::size_t k = 0; k < dirs.size(); ++k) {
    const Fields predicted = fields(res, got->tangents[k]);
    // D_k per field, over the rungs.
    std::vector<std::vector<std::vector<double>>> d(predicted.size());
    for (std::int32_t rung = 0; rung < kRungs; ++rung) {
      const double t = std::ldexp(kLadderT0, -rung);
      const Fields plus = moved_fields(res, *strings, mo, dirs[k], t);
      const Fields minus = moved_fields(res, *strings, mo, dirs[k], -t);
      ASSERT_EQ(plus.size(), predicted.size());
      ASSERT_EQ(minus.size(), predicted.size());
      for (std::size_t f = 0; f < predicted.size(); ++f) {
        std::vector<double> diff(plus[f].second.size());
        for (std::size_t e = 0; e < diff.size(); ++e) {
          diff[e] = (plus[f].second[e] - minus[f].second[e]) / (2.0 * t);
        }
        d[f].push_back(std::move(diff));
      }
    }
    for (std::size_t f = 0; f < predicted.size(); ++f) {
      const std::string name = std::format("direction {} {}", k, predicted[f].first);
      ASSERT_GT(max_abs(predicted[f].second), 0.0) << name << ": the tangent is zero";
      const FieldLadder l = field_ladder(d[f], predicted[f].second);
      EXPECT_TRUE(l.flat || l.run >= 3) << name << " ratios:" << l.ratios;
      EXPECT_LE(l.miss, kLadderTolerance) << name << " ratios:" << l.ratios;
      worst = std::max(worst, l.miss);
      std::cout << std::format("[ nevpt2   ] builders tangent {}: |tangent| {:.3e}, relative "
                               "miss {:.2e}, {}ratios{}\n",
                               name, max_abs(predicted[f].second), l.miss, l.flat ? "flat, " : "",
                               l.ratios);
    }
  }
  std::cout << std::format("[ nevpt2   ] builders tangent: worst relative miss {:.2e}\n", worst);
}

// The value's f is zc_adjoint's: Phi = <z, f> over a random symmetric z on
// the core-core and virtual-virtual blocks, and f's diagonals the orbital
// energies class_energies reads.
TEST(Nevpt2BuildersTangent, FockBlocksAreZcAdjointsFock) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res(0);
  auto strings = fb::ci::DeviceStringSet::create(res, kNcas, kElectronsPerSpin);
  ASSERT_TRUE(strings.has_value()) << strings.error().message;
  const auto ndet = static_cast<std::size_t>(strings->size()) * strings->size();
  const Point point = random_point(ndet);
  const auto h = upload(res, point.h);
  const auto b = upload(res, point.b);
  const auto c = upload(res, point.c);
  const auto vector = upload(res, point.vector);
  auto got = nevpt2::builders_tangent(res, *strings, *strings, kNaux, kNmo, kNmo, b.data(),
                                      c.data(), h.data(), kNcore, kNcas, vector.data(), 0, nullptr,
                                      nullptr, nullptr);
  ASSERT_TRUE(got.has_value()) << got.error().message;
  const Fields value = fields(res, got->value);
  const auto field = [&](const std::string_view name) -> const std::vector<double> & {
    return std::ranges::find(value, name, [](const auto &f) { return f.first; })->second;
  };
  const std::vector<double> &f_core = field("f_core");
  const std::vector<double> &f_virt = field("f_virt");

  const auto n = static_cast<std::size_t>(kNmo);
  const auto nc = static_cast<std::size_t>(kNcore);
  const auto nv = static_cast<std::size_t>(kNvirt);
  const std::size_t nocc = nc + static_cast<std::size_t>(kNcas);
  std::mt19937_64 engine(8063);
  const std::vector<double> zc = symmetric(engine, nc, 1.0);
  const std::vector<double> zv = symmetric(engine, nv, 1.0);
  std::vector<double> z(n * n, 0.0);
  double phi = 0.0;
  double summands = 0.0;
  for (std::size_t j = 0; j < nc; ++j) {
    for (std::size_t i = 0; i < nc; ++i) {
      z[i + (n * j)] = zc[i + (nc * j)];
      phi += zc[i + (nc * j)] * f_core[i + (nc * j)];
      summands += std::abs(zc[i + (nc * j)] * f_core[i + (nc * j)]);
    }
  }
  for (std::size_t bb = 0; bb < nv; ++bb) {
    for (std::size_t a = 0; a < nv; ++a) {
      z[nocc + a + (n * (nocc + bb))] = zv[a + (nv * bb)];
      phi += zv[a + (nv * bb)] * f_virt[a + (nv * bb)];
      summands += std::abs(zv[a + (nv * bb)] * f_virt[a + (nv * bb)]);
    }
  }
  const auto zd = upload(res, z);
  const std::array<double, 1> weights{1.0};
  auto want = nevpt2::zc_adjoint(res, *strings, *strings, kNaux, kNmo, kNmo, b.data(), c.data(),
                                 h.data(), kNcore, kNcas, weights, vector.data(), zd.data());
  ASSERT_TRUE(want.has_value()) << want.error().message;
  EXPECT_NEAR(phi, want->phi, kSameSumTolerance * summands);
  std::cout << std::format("[ nevpt2   ] builders tangent: <z, f> {:.12e}, zc_adjoint's Phi "
                           "{:.12e}, relative {:.2e}\n",
                           phi, want->phi, std::abs(phi - want->phi) / summands);

  const std::vector<double> &e_core = field("e_core");
  const std::vector<double> &e_virt = field("e_virt");
  for (std::size_t i = 0; i < nc; ++i) {
    EXPECT_NEAR(f_core[i * (nc + 1)], e_core[i], kSameSumTolerance * max_abs(e_core)) << i;
  }
  for (std::size_t a = 0; a < nv; ++a) {
    EXPECT_NEAR(f_virt[a * (nv + 1)], e_virt[a], kSameSumTolerance * max_abs(e_virt)) << a;
  }
}

} // namespace
