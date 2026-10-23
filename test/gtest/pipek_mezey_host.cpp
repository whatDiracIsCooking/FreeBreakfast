// The Pipek-Mezey arithmetic (src/localization/pm/pm_math.h) on the host. The
// kernels hold no arithmetic of their own, so the localizer's numerical content
// is checked here with no device: the round-robin schedule is a tournament;
// the p = 2 angle is the exact maximum and the p = 4 angle tracks a brute-force
// scan of the exact quartic, whose guard never accepts a loss; the populations
// are Mulliken's and sum to one; a sweep built from the kernels' own element
// functions rises monotonically, keeps U orthogonal, agrees with populations
// rebuilt from C U and with an independently written sequential Pipek-Mezey;
// and on a symmetric pi ring p = 2 leaves a continuum of answers where p = 4
// picks one.

#include <gtest/gtest.h>

#include "pm_math.h"

import std;

namespace {

namespace fb = free_breakfast;

using Matrix = std::vector<double>;

// Column-major element (r, c) of a matrix with leading dimension ld.
double at(const Matrix &m, std::size_t ld, std::size_t r, std::size_t c) {
  return m[r + (c * ld)];
}
double &at(Matrix &m, std::size_t ld, std::size_t r, std::size_t c) {
  return m[r + (c * ld)];
}

// W = S C for a symmetric n_ao x n_ao S and an n_ao x n_occ C.
Matrix overlap_times(const Matrix &s, const Matrix &c, std::size_t n_ao, std::size_t n_occ) {
  Matrix w(n_ao * n_occ, 0.0);
  for (std::size_t j = 0; j < n_occ; ++j) {
    for (std::size_t mu = 0; mu < n_ao; ++mu) {
      for (std::size_t nu = 0; nu < n_ao; ++nu) {
        at(w, n_ao, mu, j) += at(s, n_ao, mu, nu) * at(c, n_ao, nu, j);
      }
    }
  }
  return w;
}

// Modified Gram-Schmidt of the columns of the n x k matrix m in the metric s
// (the identity when s is empty).
void orthonormalize(Matrix &m, std::size_t n, std::size_t k, const Matrix &s = {}) {
  const auto dot = [&](std::size_t a, std::size_t b) {
    double acc = 0.0;
    for (std::size_t mu = 0; mu < n; ++mu) {
      for (std::size_t nu = 0; nu < n; ++nu) {
        const double metric = s.empty() ? (mu == nu ? 1.0 : 0.0) : at(s, n, mu, nu);
        acc += at(m, n, mu, a) * metric * at(m, n, nu, b);
      }
    }
    return acc;
  };
  for (std::size_t j = 0; j < k; ++j) {
    for (std::size_t i = 0; i < j; ++i) {
      const double ov = dot(i, j);
      for (std::size_t mu = 0; mu < n; ++mu) {
        at(m, n, mu, j) -= ov * at(m, n, mu, i);
      }
    }
    const double norm = std::sqrt(dot(j, j));
    for (std::size_t mu = 0; mu < n; ++mu) {
      at(m, n, mu, j) /= norm;
    }
  }
}

// A model: one AO per atom, so the atom offsets are 0, 1, ..., n_atoms.
struct Model {
  std::int32_t n_atoms = 0;
  std::int32_t n_occ = 0;
  Matrix s;
  Matrix c;
  std::vector<std::int32_t> offsets;
};

// n_atoms centres in a line with an exponentially decaying overlap (a
// Kac-Murdock-Szego matrix, positive definite), and delocalized cosines
// S-orthonormalized: C^T S C = I, and as far from localized as a start gets.
Model make_chain(std::int32_t n_atoms, std::int32_t n_occ, double decay = 0.9) {
  const auto n = static_cast<std::size_t>(n_atoms);
  const auto occ = static_cast<std::size_t>(n_occ);
  Model m{.n_atoms = n_atoms, .n_occ = n_occ, .s = Matrix(n * n), .c = Matrix(n * occ)};
  m.offsets.resize(n + 1);
  std::iota(m.offsets.begin(), m.offsets.end(), 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      at(m.s, n, i, j) =
          std::exp(-decay * std::fabs(static_cast<double>(i) - static_cast<double>(j)));
    }
  }
  for (std::size_t j = 0; j < occ; ++j) {
    for (std::size_t mu = 0; mu < n; ++mu) {
      at(m.c, n, mu, j) = std::cos(static_cast<double>(j + 1) * (static_cast<double>(mu) + 0.5) *
                                   std::numbers::pi / static_cast<double>(n));
    }
  }
  orthonormalize(m.c, n, occ, m.s);
  return m;
}

// The population tensor, through the element function the kernel uses.
Matrix populations(const Matrix &c, const Matrix &w, std::int32_t n_ao, std::int32_t n_occ,
                   const std::vector<std::int32_t> &offsets) {
  const auto n_atoms = static_cast<std::int32_t>(offsets.size() - 1);
  Matrix q(static_cast<std::size_t>(n_atoms) * n_occ * n_occ);
  for (std::int32_t i = 0; i < n_occ; ++i) {
    for (std::int32_t j = i; j < n_occ; ++j) {
      for (std::int32_t a = 0; a < n_atoms; ++a) {
        const double v =
            fb::pm_population_element(i, j, n_ao, c.data(), w.data(), offsets[a], offsets[a + 1]);
        q[fb::pm_pop_index(n_atoms, n_occ, i, j, a)] = v;
        q[fb::pm_pop_index(n_atoms, n_occ, j, i, a)] = v;
      }
    }
  }
  return q;
}

Matrix populations(const Model &m) {
  const auto n = static_cast<std::size_t>(m.n_atoms);
  return populations(m.c, overlap_times(m.s, m.c, n, static_cast<std::size_t>(m.n_occ)), m.n_atoms,
                     m.n_occ, m.offsets);
}

template <std::int32_t kExponent>
double functional(const Matrix &q, std::int32_t n_occ, std::int32_t n_atoms) {
  double l = 0.0;
  for (std::int32_t i = 0; i < n_occ; ++i) {
    for (std::int32_t a = 0; a < n_atoms; ++a) {
      l += fb::pm_functional_term<kExponent>(q[fb::pm_pop_index(n_atoms, n_occ, i, i, a)]);
    }
  }
  return l;
}

struct Sweeps {
  double initial = 0.0;
  double final_value = 0.0;
  std::int32_t sweeps = 0;
  bool converged = false;
  Matrix u;
  Matrix q;
};

// pm.cpp's sweep loop on the host, over the kernels' element functions,
// checking L never falls from one round to the next.
template <std::int32_t kExponent>
Sweeps run_sweeps(Matrix q, std::int32_t n_occ, std::int32_t n_atoms, std::int32_t max_sweeps,
                  double tolerance) {
  const std::int32_t m = (n_occ % 2 == 0) ? n_occ : n_occ + 1;
  std::vector<std::int32_t> schedule(static_cast<std::size_t>(m - 1) * m);
  fb::pm_fill_round_robin(m, schedule.data());

  const auto occ = static_cast<std::size_t>(n_occ);
  Matrix u(occ * occ, 0.0);
  for (std::size_t i = 0; i < occ; ++i) {
    at(u, occ, i, i) = 1.0;
  }
  Matrix q_next(q.size());
  Matrix u_next(u.size());
  std::vector<double> cosines(occ, 1.0);
  std::vector<double> sines(occ, 0.0);
  std::vector<std::int32_t> partner(occ);

  Sweeps out;
  out.initial = functional<kExponent>(q, n_occ, n_atoms);
  out.final_value = out.initial;
  double floor = out.initial;
  for (std::int32_t sweep = 1; sweep <= max_sweeps && !out.converged; ++sweep) {
    for (std::int32_t round = 0; round < m - 1; ++round) {
      const std::int32_t *pairs = schedule.data() + (static_cast<std::size_t>(round) * m);
      for (std::int32_t k = 0; k < m / 2; ++k) {
        const std::int32_t p = pairs[2 * k];
        const std::int32_t r = pairs[(2 * k) + 1];
        if (p >= n_occ || r >= n_occ) {
          const auto real = static_cast<std::size_t>(p < n_occ ? p : r);
          cosines[real] = 1.0;
          sines[real] = 0.0;
          partner[real] = static_cast<std::int32_t>(real);
          continue;
        }
        fb::PmRotation<double> rot;
        if constexpr (kExponent == 2) {
          double sum_a = 0.0;
          double sum_b = 0.0;
          for (std::int32_t a = 0; a < n_atoms; ++a) {
            const auto stats = fb::pm_pair_atom_stats(q[fb::pm_pop_index(n_atoms, n_occ, p, p, a)],
                                                      q[fb::pm_pop_index(n_atoms, n_occ, r, r, a)],
                                                      q[fb::pm_pop_index(n_atoms, n_occ, p, r, a)]);
            sum_a += stats.a;
            sum_b += stats.b;
          }
          rot = fb::pm_rotation(sum_a, sum_b, 1e-16);
        } else {
          fb::PmPairStats4<double> sum;
          for (std::int32_t a = 0; a < n_atoms; ++a) {
            const auto stats =
                fb::pm_pair_atom_stats4(q[fb::pm_pop_index(n_atoms, n_occ, p, p, a)],
                                        q[fb::pm_pop_index(n_atoms, n_occ, r, r, a)],
                                        q[fb::pm_pop_index(n_atoms, n_occ, p, r, a)]);
            sum.a4 += stats.a4;
            sum.b4 += stats.b4;
            sum.a8 += stats.a8;
            sum.b8 += stats.b8;
          }
          rot = fb::pm_rotation4(sum.a4, sum.b4, sum.a8, sum.b8, 1e-16);
        }
        const auto sp = static_cast<std::size_t>(p);
        const auto sr = static_cast<std::size_t>(r);
        cosines[sp] = rot.cos_gamma;
        cosines[sr] = rot.cos_gamma;
        sines[sp] = rot.sin_gamma;
        sines[sr] = -rot.sin_gamma;
        partner[sp] = r;
        partner[sr] = p;
      }
      for (std::int32_t i = 0; i < n_occ; ++i) {
        for (std::int32_t j = i; j < n_occ; ++j) {
          const std::int32_t pi = partner[static_cast<std::size_t>(i)];
          const std::int32_t pj = partner[static_cast<std::size_t>(j)];
          for (std::int32_t a = 0; a < n_atoms; ++a) {
            const double v = fb::pm_rotate_population(
                cosines[static_cast<std::size_t>(i)], sines[static_cast<std::size_t>(i)],
                cosines[static_cast<std::size_t>(j)], sines[static_cast<std::size_t>(j)],
                q[fb::pm_pop_index(n_atoms, n_occ, i, j, a)],
                q[fb::pm_pop_index(n_atoms, n_occ, i, pj, a)],
                q[fb::pm_pop_index(n_atoms, n_occ, pi, j, a)],
                q[fb::pm_pop_index(n_atoms, n_occ, pi, pj, a)]);
            q_next[fb::pm_pop_index(n_atoms, n_occ, i, j, a)] = v;
            q_next[fb::pm_pop_index(n_atoms, n_occ, j, i, a)] = v;
          }
        }
      }
      for (std::size_t col = 0; col < occ; ++col) {
        for (std::size_t row = 0; row < occ; ++row) {
          at(u_next, occ, row, col) = fb::pm_rotate_column(
              at(u, occ, row, col), at(u, occ, row, static_cast<std::size_t>(partner[col])),
              cosines[col], sines[col]);
        }
      }
      q.swap(q_next);
      u.swap(u_next);
      // A round's disjoint rotations each change only their own pair's
      // diagonal, so the round gains the sum of its pairs' non-negative gains.
      const double now = functional<kExponent>(q, n_occ, n_atoms);
      EXPECT_GE(now, floor - 1e-12) << "L fell in sweep " << sweep << " round " << round;
      floor = now;
    }
    const double now = functional<kExponent>(q, n_occ, n_atoms);
    out.converged = std::abs(now - out.final_value) < tolerance;
    out.final_value = now;
    out.sweeps = sweep;
  }
  out.u = std::move(u);
  out.q = std::move(q);
  return out;
}

// A textbook SEQUENTIAL Pipek-Mezey (p = 2): one pair at a time, rotating the
// coefficients and rebuilding the populations after every rotation. Written
// from the definition, sharing nothing with pm_math.h, so agreement is
// evidence rather than a tautology.
double sequential_reference(Matrix c, const Model &m, std::int32_t sweeps) {
  const auto n = static_cast<std::size_t>(m.n_atoms);
  const auto occ = static_cast<std::size_t>(m.n_occ);
  double result = 0.0;
  for (std::int32_t sweep = 0; sweep < sweeps; ++sweep) {
    for (std::size_t p = 0; p < occ; ++p) {
      for (std::size_t r = p + 1; r < occ; ++r) {
        const Matrix w = overlap_times(m.s, c, n, occ);
        double sum_a = 0.0;
        double sum_b = 0.0;
        for (std::size_t mu = 0; mu < n; ++mu) { // one AO per atom
          const double q_pp = at(c, n, mu, p) * at(w, n, mu, p);
          const double q_rr = at(c, n, mu, r) * at(w, n, mu, r);
          const double q_pr =
              0.5 * ((at(c, n, mu, p) * at(w, n, mu, r)) + (at(c, n, mu, r) * at(w, n, mu, p)));
          const double d = q_pp - q_rr;
          sum_a += (q_pr * q_pr) - (0.25 * d * d);
          sum_b += q_pr * d;
        }
        if (sum_a + std::hypot(sum_a, sum_b) < 1e-16) {
          continue;
        }
        const double gamma = 0.25 * std::atan2(sum_b, -sum_a);
        for (std::size_t mu = 0; mu < n; ++mu) {
          const double cp = at(c, n, mu, p);
          const double cr = at(c, n, mu, r);
          at(c, n, mu, p) = (std::cos(gamma) * cp) + (std::sin(gamma) * cr);
          at(c, n, mu, r) = (-std::sin(gamma) * cp) + (std::cos(gamma) * cr);
        }
      }
    }
    result =
        functional<2>(populations(c, overlap_times(m.s, c, n, occ), m.n_atoms, m.n_occ, m.offsets),
                      m.n_occ, m.n_atoms);
  }
  return result;
}

void expect_orthogonal(const Matrix &u, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double dot = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        dot += at(u, n, k, i) * at(u, n, k, j);
      }
      EXPECT_NEAR(dot, i == j ? 1.0 : 0.0, 1e-12) << i << "," << j;
    }
  }
}

// The populations of C U, rebuilt from scratch.
Matrix rebuilt_populations(const Model &m, const Matrix &u) {
  const auto n = static_cast<std::size_t>(m.n_atoms);
  const auto occ = static_cast<std::size_t>(m.n_occ);
  Matrix c_local(n * occ, 0.0);
  for (std::size_t j = 0; j < occ; ++j) {
    for (std::size_t k = 0; k < occ; ++k) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        at(c_local, n, mu, j) += at(m.c, n, mu, k) * at(u, occ, k, j);
      }
    }
  }
  return populations(c_local, overlap_times(m.s, c_local, n, occ), m.n_atoms, m.n_occ, m.offsets);
}

// ---------------------------------------------------------------------------
// The schedule
// ---------------------------------------------------------------------------

TEST(PipekMezeySchedule, RoundsAreDisjointAndCoverEveryPairOnce) {
  for (std::int32_t m = 2; m <= 32; m += 2) {
    std::vector<std::int32_t> schedule(static_cast<std::size_t>(m - 1) * m, -1);
    fb::pm_fill_round_robin(m, schedule.data());
    std::set<std::pair<std::int32_t, std::int32_t>> all_pairs;
    for (std::int32_t r = 0; r < m - 1; ++r) {
      std::set<std::int32_t> used;
      for (std::int32_t k = 0; k < m / 2; ++k) {
        const std::int32_t p = schedule[(static_cast<std::size_t>(r) * m) + (2 * k)];
        const std::int32_t q = schedule[(static_cast<std::size_t>(r) * m) + (2 * k) + 1];
        ASSERT_TRUE(p >= 0 && p < m && q >= 0 && q < m && p != q) << "m=" << m << " r=" << r;
        EXPECT_TRUE(used.insert(p).second) << p << " twice in round " << r;
        EXPECT_TRUE(used.insert(q).second) << q << " twice in round " << r;
        all_pairs.insert({std::min(p, q), std::max(p, q)});
      }
      EXPECT_EQ(std::ssize(used), m) << "m=" << m << " round " << r << " is not a matching";
    }
    EXPECT_EQ(std::ssize(all_pairs), m * (m - 1) / 2) << "m=" << m;
  }
}

// ---------------------------------------------------------------------------
// The p = 2 rotation
// ---------------------------------------------------------------------------

TEST(PipekMezeyRotation, MaximizesTheFunctionalChange) {
  // dL(gamma) = A (1 - cos 4gamma) + B sin 4gamma, scanned finely.
  for (const auto &[a, b] : std::array<std::pair<double, double>, 5>{
           {{0.4, 0.2}, {-0.3, 0.7}, {-0.9, -0.1}, {0.05, -0.6}, {1.2, 0.0}}}) {
    const auto rot = fb::pm_rotation(a, b, 1e-16);
    const auto delta = [&](double g) {
      return (a * (1.0 - std::cos(4.0 * g))) + (b * std::sin(4.0 * g));
    };
    EXPECT_NEAR(delta(std::atan2(rot.sin_gamma, rot.cos_gamma)), rot.gain, 1e-12);
    EXPECT_GE(rot.gain, 0.0);
    for (std::int32_t k = -2000; k <= 2000; ++k) {
      const double g = (std::numbers::pi / 4.0) * (static_cast<double>(k) / 2000.0);
      EXPECT_LE(delta(g), rot.gain + 1e-12) << "a=" << a << " b=" << b << " gamma=" << g;
    }
  }
}

TEST(PipekMezeyRotation, LeavesNegligibleDegenerateAndNanPairsAlone) {
  for (const auto &[a, b, threshold] : std::array<std::array<double, 3>, 3>{
           {{1e-20, 1e-20, 1e-14}, {0.0, 0.0, 1e-16}, {std::nan(""), 1.0, 1e-16}}}) {
    const auto rot = fb::pm_rotation(a, b, threshold);
    EXPECT_EQ(rot.cos_gamma, 1.0);
    EXPECT_EQ(rot.sin_gamma, 0.0);
    EXPECT_EQ(rot.gain, 0.0);
  }
}

TEST(PipekMezeyRotation, PairStatsFollowTheDefinition) {
  const auto stats = fb::pm_pair_atom_stats(0.8, 0.3, 0.2);
  EXPECT_NEAR(stats.a, (0.2 * 0.2) - (0.25 * 0.5 * 0.5), 1e-15);
  EXPECT_NEAR(stats.b, 0.2 * 0.5, 1e-15);
}

// ---------------------------------------------------------------------------
// The p = 4 rotation
// ---------------------------------------------------------------------------

// The exact change in sum over atoms of Qpp^4 + Qqq^4 from rotating by gamma,
// straight from the definition: the oracle the closed form is checked against.
double exact_delta_l4(double gamma, const std::vector<std::array<double, 3>> &atoms) {
  const double c = std::cos(gamma);
  const double s = std::sin(gamma);
  double total = 0.0;
  for (const auto &[q_pp, q_qq, q_pq] : atoms) {
    const double pp = (c * c * q_pp) + (2.0 * c * s * q_pq) + (s * s * q_qq);
    const double qq = (s * s * q_pp) - (2.0 * c * s * q_pq) + (c * c * q_qq);
    total += std::pow(pp, 4) + std::pow(qq, 4) - std::pow(q_pp, 4) - std::pow(q_qq, 4);
  }
  return total;
}

// n random atoms' (Qpp, Qqq, Qpq), and their summed p = 4 statistics.
std::pair<std::vector<std::array<double, 3>>, fb::PmPairStats4<double>>
random_pair(std::mt19937 &rng, std::int32_t n) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<std::array<double, 3>> atoms(static_cast<std::size_t>(n));
  fb::PmPairStats4<double> sum;
  for (auto &atom : atoms) {
    atom = {dist(rng), dist(rng), dist(rng)};
    const auto stats = fb::pm_pair_atom_stats4(atom[0], atom[1], atom[2]);
    sum.a4 += stats.a4;
    sum.b4 += stats.b4;
    sum.a8 += stats.a8;
    sum.b8 += stats.b8;
  }
  return {atoms, sum};
}

TEST(PipekMezeyRotationP4, AngleTracksTheExactQuarticAndReportsItsExactGain) {
  std::mt19937 rng(12345);
  for (std::int32_t trial = 0; trial < 25; ++trial) {
    const auto [atoms, sum] = random_pair(rng, 1 + (trial % 6));
    const auto rot = fb::pm_rotation4(sum.a4, sum.b4, sum.a8, sum.b8, -1e300);
    const double gamma = std::atan2(rot.sin_gamma, rot.cos_gamma);
    EXPECT_NEAR(rot.gain, exact_delta_l4(gamma, atoms), 1e-9) << "trial " << trial;

    double best_g = 0.0;
    double best = exact_delta_l4(0.0, atoms);
    constexpr std::int32_t kSteps = 100000;
    for (std::int32_t k = -kSteps; k <= kSteps; ++k) {
      const double g = (std::numbers::pi / 4.0) * static_cast<double>(k) / kSteps;
      if (const double l = exact_delta_l4(g, atoms); l > best) {
        best = l;
        best_g = g;
      }
    }
    // dL4 has period pi/2, so fold the difference into one period: the closed
    // form and the scan may name the same maximum a period apart. What is left
    // is the dropped 8gamma harmonic's error, which must be modest.
    EXPECT_LT(std::abs(std::remainder(gamma - best_g, std::numbers::pi / 2.0)), 0.3)
        << "trial " << trial;
  }
}

TEST(PipekMezeyRotationP4, NeverAcceptsARotationThatLowersTheExactQuartic) {
  std::mt19937 rng(999);
  std::int32_t rotated = 0;
  for (std::int32_t trial = 0; trial < 500; ++trial) {
    const auto [atoms, sum] = random_pair(rng, 1 + (trial % 8));
    const auto rot = fb::pm_rotation4(sum.a4, sum.b4, sum.a8, sum.b8, 1e-14);
    if (rot.sin_gamma == 0.0 && rot.cos_gamma == 1.0) {
      continue;
    }
    ++rotated;
    EXPECT_GT(exact_delta_l4(std::atan2(rot.sin_gamma, rot.cos_gamma), atoms), -1e-9)
        << "trial " << trial;
  }
  EXPECT_GT(rotated, 0) << "the guard must still let ordinary pairs rotate";
}

TEST(PipekMezeyRotationP4, LeavesNegligibleDegenerateAndNanPairsAlone) {
  for (const auto &[a4, b4, threshold] : std::array<std::array<double, 3>, 3>{
           {{1e-20, 1e-20, 1e-14}, {0.0, 0.0, 1e-16}, {std::nan(""), 1.0, 1e-16}}}) {
    const auto rot = fb::pm_rotation4(a4, b4, 0.0, 0.0, threshold);
    EXPECT_EQ(rot.cos_gamma, 1.0);
    EXPECT_EQ(rot.sin_gamma, 0.0);
    EXPECT_EQ(rot.gain, 0.0);
  }
}

TEST(PipekMezeyRotationP4, GainIsZeroAtZeroAngle) {
  const auto stats = fb::pm_pair_atom_stats4(0.8, 0.3, 0.2);
  EXPECT_NEAR(fb::pm_gain4(0.0, stats.a4, stats.b4, stats.a8, stats.b8), 0.0, 1e-14);
}

// ---------------------------------------------------------------------------
// The populations
// ---------------------------------------------------------------------------

TEST(PipekMezeyPopulations, AreMullikenAndSumToTheMetric) {
  const Model m = make_chain(6, 3);
  const auto n = static_cast<std::size_t>(m.n_atoms);
  const Matrix w = overlap_times(m.s, m.c, n, static_cast<std::size_t>(m.n_occ));
  const Matrix q = populations(m);
  for (std::int32_t i = 0; i < m.n_occ; ++i) {
    for (std::int32_t j = 0; j < m.n_occ; ++j) {
      double total = 0.0;
      for (std::int32_t a = 0; a < m.n_atoms; ++a) {
        const auto mu = static_cast<std::size_t>(a);
        const auto si = static_cast<std::size_t>(i);
        const auto sj = static_cast<std::size_t>(j);
        const double expected = 0.5 * ((at(m.c, n, mu, si) * at(w, n, mu, sj)) +
                                       (at(m.c, n, mu, sj) * at(w, n, mu, si)));
        const double value = q[fb::pm_pop_index(m.n_atoms, m.n_occ, i, j, a)];
        EXPECT_NEAR(value, expected, 1e-14);
        total += value;
      }
      // sum_A Q^A_ij = (C^T S C)_ij = delta_ij: each orbital's populations are
      // a partition of unity, the functional's premise.
      EXPECT_NEAR(total, i == j ? 1.0 : 0.0, 1e-12) << i << "," << j;
    }
  }
}

TEST(PipekMezeyPopulations, AnAtomSumsItsWholeAoRange) {
  // Two atoms owning AOs [0, 2) and [2, 3): an atom's population is its
  // range's, and the two still sum to the metric.
  constexpr std::size_t kN = 3;
  Matrix s(kN * kN, 0.0);
  for (std::size_t i = 0; i < kN; ++i) {
    for (std::size_t j = 0; j < kN; ++j) {
      at(s, kN, i, j) = i == j ? 1.0 : 0.2;
    }
  }
  Matrix c{0.3, 0.5, 0.7};
  orthonormalize(c, kN, 1, s);
  const Matrix q = populations(c, overlap_times(s, c, kN, 1), kN, 1, {0, 2, 3});
  const Matrix w = overlap_times(s, c, kN, 1);
  EXPECT_NEAR(q[0], (c[0] * w[0]) + (c[1] * w[1]), 1e-15);
  EXPECT_NEAR(q[1], c[2] * w[2], 1e-15);
  EXPECT_NEAR(q[0] + q[1], 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Whole sweeps
// ---------------------------------------------------------------------------

template <std::int32_t kExponent>
void check_sweeps_on_a_chain() {
  const Model m = make_chain(8, 4);
  const Sweeps out = run_sweeps<kExponent>(populations(m), m.n_occ, m.n_atoms, 200, 1e-12);
  EXPECT_TRUE(out.converged);
  EXPECT_GT(out.final_value, out.initial);
  expect_orthogonal(out.u, static_cast<std::size_t>(m.n_occ));
  // The point of sweeping the populations: Q^A transforms as U^T Q^A U, so
  // the swept tensor is the one rebuilt from C U.
  const Matrix rebuilt = rebuilt_populations(m, out.u);
  ASSERT_EQ(rebuilt.size(), out.q.size());
  for (std::size_t i = 0; i < rebuilt.size(); ++i) {
    EXPECT_NEAR(rebuilt[i], out.q[i], 1e-11) << "element " << i;
  }
}

TEST(PipekMezeySweep, RisesConvergesAndMatchesTheRebuiltPopulations) {
  check_sweeps_on_a_chain<2>();
}

TEST(PipekMezeySweepP4, RisesConvergesAndMatchesTheRebuiltPopulations) {
  check_sweeps_on_a_chain<4>();
}

TEST(PipekMezeySweep, MatchesTheSequentialAlgorithm) {
  // Rounds of simultaneous disjoint rotations reach the textbook sweep's
  // maximum, odd counts included (the padding index and its bye).
  for (const std::int32_t n_occ : {3, 4, 5}) {
    const Model m = make_chain(8, n_occ);
    const Sweeps out = run_sweeps<2>(populations(m), m.n_occ, m.n_atoms, 200, 1e-14);
    EXPECT_TRUE(out.converged) << "n_occ=" << n_occ;
    EXPECT_NEAR(out.final_value, sequential_reference(m.c, m, 200), 1e-9) << "n_occ=" << n_occ;
  }
}

TEST(PipekMezeySweep, LeavesLocalizedOrbitalsAlone) {
  Model m{.n_atoms = 2, .n_occ = 2, .s = {1.0, 1e-9, 1e-9, 1.0}, .c = {1.0, 0.0, 0.0, 1.0}};
  m.offsets = {0, 1, 2};
  const Sweeps out = run_sweeps<2>(populations(m), 2, 2, 50, 1e-15);
  EXPECT_NEAR(out.initial, 2.0, 1e-8);
  EXPECT_NEAR(out.final_value, 2.0, 1e-8);
}

TEST(PipekMezeySweepP4, ConvergesWithinAHandfulOfSweeps) {
  // Knizia reports 5-10 sweeps for the approximate angle; 25 would still
  // catch one ten times slower.
  const Model m = make_chain(8, 4);
  const Sweeps out = run_sweeps<4>(populations(m), m.n_occ, m.n_atoms, 200, 1e-10);
  EXPECT_TRUE(out.converged);
  EXPECT_LE(out.sweeps, 25);
}

// ---------------------------------------------------------------------------
// The aromatic degeneracy: p = 2 leaves it, p = 4 breaks it
// ---------------------------------------------------------------------------

// The three lowest Hueckel MOs of a six-ring with orthonormal AOs (S = I, to
// isolate the symmetry from any overlap asymmetry): the bonding k = 0 orbital
// and the degenerate k = 1 pair, the textbook case of p = 2's zero mode.
// Localized from eight random near-identity rotations of the occupied space,
// standing in for whatever basis of the degenerate MOs an eigensolver returns,
// it returns each run's L and its sorted, rounded diagonal populations (a
// fingerprint blind to how the orbitals and atoms are labelled).
template <std::int32_t kExponent>
std::pair<std::vector<double>, std::set<std::vector<double>>> localize_ring(double tolerance) {
  constexpr std::int32_t kRing = 6;
  constexpr std::int32_t kOcc = 3;
  Matrix c0(static_cast<std::size_t>(kRing) * kOcc);
  for (std::size_t mu = 0; mu < kRing; ++mu) {
    const double theta = 2.0 * std::numbers::pi * static_cast<double>(mu) / kRing;
    at(c0, kRing, mu, 0) = 1.0;
    at(c0, kRing, mu, 1) = std::cos(theta);
    at(c0, kRing, mu, 2) = std::sin(theta);
  }
  orthonormalize(c0, kRing, kOcc);
  Model m{.n_atoms = kRing, .n_occ = kOcc, .s = Matrix(kRing * kRing, 0.0)};
  for (std::size_t i = 0; i < kRing; ++i) {
    at(m.s, kRing, i, i) = 1.0;
  }
  m.offsets.resize(kRing + 1);
  std::iota(m.offsets.begin(), m.offsets.end(), 0);

  std::mt19937 rng(2024);
  std::normal_distribution<double> noise(0.0, 0.05);
  std::vector<double> values;
  std::set<std::vector<double>> fingerprints;
  for (std::int32_t trial = 0; trial < 8; ++trial) {
    Matrix perturb(kOcc * kOcc);
    for (std::size_t i = 0; i < kOcc; ++i) {
      for (std::size_t j = 0; j < kOcc; ++j) {
        at(perturb, kOcc, i, j) = i == j ? 1.0 : noise(rng);
      }
    }
    orthonormalize(perturb, kOcc, kOcc);
    m.c.assign(static_cast<std::size_t>(kRing) * kOcc, 0.0);
    for (std::size_t j = 0; j < kOcc; ++j) {
      for (std::size_t k = 0; k < kOcc; ++k) {
        for (std::size_t mu = 0; mu < kRing; ++mu) {
          at(m.c, kRing, mu, j) += at(c0, kRing, mu, k) * at(perturb, kOcc, k, j);
        }
      }
    }
    const Sweeps out = run_sweeps<kExponent>(populations(m), kOcc, kRing, 200, tolerance);
    EXPECT_TRUE(out.converged) << "trial " << trial;
    values.push_back(out.final_value);
    std::vector<double> fp;
    for (std::int32_t i = 0; i < kOcc; ++i) {
      for (std::int32_t a = 0; a < kRing; ++a) {
        fp.push_back(std::round(out.q[fb::pm_pop_index(kRing, kOcc, i, i, a)] * 1000.0) / 1000.0);
      }
    }
    std::ranges::sort(fp);
    fingerprints.insert(fp);
  }
  return {values, fingerprints};
}

TEST(PipekMezeyDegeneracy, P2ReachesOneValueByManyLocalizations) {
  const auto [values, fingerprints] = localize_ring<2>(1e-14);
  for (const double v : values) {
    EXPECT_NEAR(v, values.front(), 1e-8);
  }
  EXPECT_GT(fingerprints.size(), 1U) << "p = 2 picked one localization; the ring should be "
                                        "a continuum";
}

TEST(PipekMezeyDegeneracy, P4PicksOneLocalization) {
  const auto [values, fingerprints] = localize_ring<4>(1e-10);
  for (const double v : values) {
    EXPECT_NEAR(v, values.front(), 1e-6);
  }
  EXPECT_EQ(fingerprints.size(), 1U);
}

} // namespace
