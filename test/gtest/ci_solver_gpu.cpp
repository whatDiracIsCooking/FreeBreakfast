// GPU cases for ci.solver and ci.rdm: ci_solve over the host sigma, wrapped as
// a LinearOperator, against PySCF's fci.direct_spin1 (data/ci/direct_spin1.json,
// written by tools/emit_fci_reference.py). Every case there has a triplet below
// its second singlet, so the singlets come out in order only if the spin
// penalty works; with no penalty the triplet appears among them.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.json;
import free_breakfast.ci;

namespace {

namespace fb = free_breakfast;

// Golden data are converged to 1e-13 in the energy; these are the issue's.
constexpr double kEnergyTol = 1e-8;
constexpr double kSpinTol = 1e-8;
constexpr double kRdmTol = 1e-8;

// host_sigma as a LinearOperator: each column copied to the host, through
// host_sigma, and back.
struct HostSigma {
  const fb::StringSet &alpha;
  const fb::StringSet &beta;
  double e_core;
  const std::vector<double> &h1;
  const std::vector<double> &eri;
  std::int32_t applies = 0;

  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const double *x, double *y) {
    ++applies;
    const std::size_t dim =
        static_cast<std::size_t>(alpha.size()) * static_cast<std::size_t>(beta.size());
    const std::size_t count = dim * static_cast<std::size_t>(k);
    std::vector<double> xh(count);
    std::vector<double> yh(count);
    if (auto s = fb::gpu_status(wwr::wwrMemcpyAsync(xh.data(), x, count * sizeof(double),
                                                    wwr::wwrMemcpyDeviceToHost, res.stream()));
        !s) {
      return s;
    }
    if (auto s = res.sync(); !s) {
      return s;
    }
    for (std::size_t col = 0; col < static_cast<std::size_t>(k); ++col) {
      if (auto s = fb::host_sigma(alpha, beta, e_core, h1, eri,
                                  std::span<const double>{xh}.subspan(col * dim, dim),
                                  std::span<double>{yh}.subspan(col * dim, dim));
          !s) {
        return s;
      }
    }
    if (auto s = fb::gpu_status(wwr::wwrMemcpyAsync(y, yh.data(), count * sizeof(double),
                                                    wwr::wwrMemcpyHostToDevice, res.stream()));
        !s) {
      return s;
    }
    return res.sync();
  }
};
static_assert(fb::LinearOperator<HostSigma, double>);

// One direct_spin1 case, its sets in a chosen sector, and its golden roots.
struct Case {
  std::string name;
  std::int32_t n;
  double e_core;
  std::vector<double> h1;
  std::vector<double> eri;
  fb::StringSet alpha;
  fb::StringSet beta;
  std::vector<double> diagonal;
  fb::json roots;
  std::int32_t n_electrons;
};

fb::json load_cases() {
  auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "ci" / "direct_spin1.json");
  EXPECT_TRUE(doc.has_value()) << doc.error().message;
  return doc.has_value() ? (*doc)["cases"] : fb::json::array();
}

// The case in sector (N_α, N_β) = its own, moved by @p shift electrons from β
// to α: shift 1 is the M_s = 1 sector, of the same total electron count.
Case make_case(const fb::json &entry, std::int32_t shift = 0) {
  const auto n = entry["n_orbitals"].get<std::int32_t>();
  const auto na = entry["n_alpha"].get<std::int32_t>() + shift;
  const auto nb = entry["n_beta"].get<std::int32_t>() - shift;
  auto alpha = fb::StringSet::create(n, na);
  auto beta = fb::StringSet::create(n, nb);
  EXPECT_TRUE(alpha.has_value() && beta.has_value());
  Case c{entry["name"].get<std::string>(),
         n,
         entry["e_core"].get<double>(),
         entry["h1"]["values"].get<std::vector<double>>(),
         entry["eri"]["values"].get<std::vector<double>>(),
         std::move(*alpha),
         std::move(*beta),
         {},
         entry["roots"],
         na + nb};
  c.diagonal.resize(static_cast<std::size_t>(c.alpha.size()) *
                    static_cast<std::size_t>(c.beta.size()));
  EXPECT_TRUE(fb::host_diagonal(c.alpha, c.beta, c.e_core, c.h1, c.eri, c.diagonal).has_value());
  return c;
}

// The golden roots whose <S^2> is @p s2, in ascending order.
std::vector<std::size_t> golden_of_spin(const fb::json &roots, double s2) {
  std::vector<std::size_t> out;
  for (std::size_t k = 0; k < roots.size(); ++k) {
    if (std::abs(roots[k]["s2"].get<double>() - s2) < 1e-6) {
      out.push_back(k);
    }
  }
  return out;
}

// The solved roots against the golden ones of the same spin: energy, <S^2>,
// and the 1-RDM (where the golden root is not one of a degenerate level)
// with its trace. The spin-summed 1-RDM does not depend on M_s, so a triplet
// solved at M_s = 1 compares with direct_spin1's M_s = 0 one.
void expect_roots(const Case &c, const fb::CiRoots &got, double s2) {
  const std::vector<std::size_t> golden = golden_of_spin(c.roots, s2);
  ASSERT_GE(golden.size(), got.energies.size());
  const std::size_t dim =
      static_cast<std::size_t>(c.alpha.size()) * static_cast<std::size_t>(c.beta.size());
  const auto m = static_cast<std::size_t>(c.n);
  std::int32_t compared = 0;
  for (std::size_t r = 0; r < got.energies.size(); ++r) {
    SCOPED_TRACE(std::format("root {} (golden {})", r, golden[r]));
    const fb::json &want = c.roots[golden[r]];
    EXPECT_NEAR(got.energies[r], want["energy"].get<double>(), kEnergyTol);
    EXPECT_NEAR(got.spin_squares[r], s2, kSpinTol);

    std::vector<double> gamma(m * m);
    ASSERT_TRUE(
        fb::rdm1(c.alpha, c.beta, std::span<const double>{got.vectors}.subspan(r * dim, dim), gamma)
            .has_value());
    double trace = 0.0;
    for (std::size_t t = 0; t < m; ++t) {
      trace += gamma[t * m + t];
    }
    EXPECT_NEAR(trace, static_cast<double>(c.n_electrons), 1e-10);
    if (want["degenerate"].get<bool>()) {
      continue;
    }
    ++compared;
    const auto reference = want["rdm1"]["values"].get<std::vector<double>>();
    ASSERT_EQ(reference.size(), gamma.size());
    double worst = 0.0;
    for (std::size_t i = 0; i < gamma.size(); ++i) {
      worst = std::max(worst, std::abs(gamma[i] - reference[i]));
    }
    EXPECT_LE(worst, kRdmTol);
  }
  EXPECT_GT(compared, 0) << "no non-degenerate root to compare a 1-RDM against";
}

TEST(CiSolver, LowestSingletsMatchDirectSpin1) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  for (const auto &entry : cases) {
    const Case c = make_case(entry);
    SCOPED_TRACE(c.name);
    // The spin control is exercised: a triplet lies below the second singlet.
    const std::vector<std::size_t> singlets = golden_of_spin(c.roots, 0.0);
    ASSERT_GE(singlets.size(), 3U);
    ASSERT_FALSE(golden_of_spin(c.roots, 2.0).empty());
    ASSERT_LT(golden_of_spin(c.roots, 2.0).front(), singlets[1]);

    HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri};
    const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, c.diagonal, 3);
    ASSERT_TRUE(got.has_value()) << got.error().message;
    EXPECT_EQ(sigma.applies, got->davidson.iterations);
    expect_roots(c, *got, 0.0);
  }
}

// The spin is chosen by the sector: N_α − N_β = 2 keeps triplets, the M_s = 1
// partners of direct_spin1's M_s = 0 ones.
TEST(CiSolver, LowestTripletsFromTheMsOneSector) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  for (const auto &entry : cases) {
    const Case c = make_case(entry, 1);
    SCOPED_TRACE(c.name);
    HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri};
    const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, c.diagonal, 3);
    ASSERT_TRUE(got.has_value()) << got.error().message;
    expect_roots(c, *got, 2.0);
  }
}

// With no penalty every spin is kept: the lowest three roots are
// direct_spin1's lowest three, the triplet among them.
TEST(CiSolver, NoPenaltyKeepsEverySpin) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  const Case c = make_case(cases[0]);
  HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri};
  fb::CiSolverOptions options;
  options.spin_penalty = 0.0;
  const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, c.diagonal, 3, options);
  ASSERT_TRUE(got.has_value()) << got.error().message;
  bool triplet = false;
  for (std::size_t r = 0; r < 3; ++r) {
    EXPECT_NEAR(got->energies[r], c.roots[r]["energy"].get<double>(), kEnergyTol);
    EXPECT_NEAR(got->spin_squares[r], c.roots[r]["s2"].get<double>(), kSpinTol);
    triplet = triplet || std::abs(got->spin_squares[r] - 2.0) < kSpinTol;
  }
  EXPECT_TRUE(triplet);
}

TEST(CiSolver, TooFewIterationsIsMaxIterationsExceeded) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  const Case c = make_case(cases[1]);
  HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri};
  fb::CiSolverOptions options;
  options.davidson.max_iterations = 2;
  const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, c.diagonal, 3, options);
  ASSERT_FALSE(got.has_value());
  EXPECT_EQ(got.error().kind, fb::ErrorKind::MaxIterationsExceeded) << got.error().message;
}

TEST(CiSolver, RejectsBadShapes) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  const Case c = make_case(cases[0]);
  HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri};
  const auto dim = static_cast<std::int32_t>(c.diagonal.size());
  const auto kind = [](const fb::Result<fb::CiRoots> &got) {
    return got.has_value() ? std::optional<fb::ErrorKind>{} : got.error().kind;
  };
  const auto invalid = std::optional{fb::ErrorKind::InvalidConfig};
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, c.diagonal, 0)), invalid);
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, c.diagonal, dim + 1)), invalid);
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta,
                              std::span<const double>{c.diagonal}.first(c.diagonal.size() - 1), 1)),
            invalid);
  fb::CiSolverOptions negative;
  negative.spin_penalty = -1.0;
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, c.diagonal, 1, negative)), invalid);
  EXPECT_EQ(sigma.applies, 0);

  std::vector<double> vector(c.diagonal.size(), 0.0);
  std::vector<double> gamma(static_cast<std::size_t>(c.n * c.n));
  EXPECT_FALSE(
      fb::rdm1(c.alpha, c.beta, std::span<const double>{vector}.first(1), gamma).has_value());
  EXPECT_FALSE(fb::rdm1(c.alpha, c.beta, vector, std::span<double>{gamma}.first(1)).has_value());
  std::vector<double> out(vector.size());
  EXPECT_FALSE(
      fb::host_spin_square(c.alpha, c.beta, vector, std::span<double>{out}.first(1)).has_value());
}

// CiSpinSquare against host_spin_square on random blocks of three vectors,
// each column to 1e-12 of its norm: apply, and penalize into a random Y. The
// sectors cover open and closed shells, an empty spin, a full one, a single
// orbital and N_alpha < N_beta.
TEST(CiSpinSquareGpu, MatchesHost) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  struct Sector {
    std::int32_t n, n_alpha, n_beta;
  };
  constexpr std::int32_t k = 3;
  constexpr double lambda = 0.7;
  constexpr double target = 2.0;
  for (const Sector s : {Sector{6, 3, 3}, Sector{7, 4, 2}, Sector{8, 3, 5}, Sector{5, 0, 3},
                         Sector{5, 5, 2}, Sector{1, 1, 0}, Sector{10, 5, 5}}) {
    SCOPED_TRACE(std::format("n {} alpha {} beta {}", s.n, s.n_alpha, s.n_beta));
    const auto alpha = fb::StringSet::create(s.n, s.n_alpha);
    const auto beta = fb::StringSet::create(s.n, s.n_beta);
    ASSERT_TRUE(alpha.has_value() && beta.has_value());
    auto spin = fb::CiSpinSquare::create(res, *alpha, *beta);
    ASSERT_TRUE(spin.has_value()) << spin.error().message;
    const auto dim = static_cast<std::size_t>(spin->size());
    ASSERT_EQ(dim,
              static_cast<std::size_t>(alpha->size()) * static_cast<std::size_t>(beta->size()));

    std::mt19937_64 engine{dim};
    std::uniform_real_distribution<double> uniform{-1.0, 1.0};
    std::vector<double> x(dim * k);
    std::vector<double> y0(dim * k);
    for (double &v : x) {
      v = uniform(engine);
    }
    for (double &v : y0) {
      v = uniform(engine);
    }
    const auto upload = [&](const std::vector<double> &host) {
      fb::Buffers::device<double> d(host.size(), res.device());
      EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                    wwr::wwrMemcpyHostToDevice, res.stream()),
                wwr::wwrSuccess);
      return d;
    };
    const auto download = [&](const fb::Buffers::device<double> &d) {
      std::vector<double> host(d.num_elements());
      EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d.data(), host.size() * sizeof(double),
                                    wwr::wwrMemcpyDeviceToHost, res.stream()),
                wwr::wwrSuccess);
      EXPECT_TRUE(res.sync().has_value());
      return host;
    };
    const auto xd = upload(x);
    fb::Buffers::device<double> applied(dim * k, res.device());
    ASSERT_TRUE(spin->apply(res, k, xd.data(), applied.data()).has_value());
    const std::vector<double> got = download(applied);
    auto penalized = upload(y0);
    ASSERT_TRUE(spin->penalize(res, k, lambda, target, xd.data(), penalized.data()).has_value());
    const std::vector<double> got_penalized = download(penalized);

    std::vector<double> s2x(dim);
    for (std::size_t j = 0; j < k; ++j) {
      const std::span<const double> column{x.data() + j * dim, dim};
      ASSERT_TRUE(fb::host_spin_square(*alpha, *beta, column, s2x).has_value());
      double norm = 0.0;
      double error = 0.0;
      double penalized_error = 0.0;
      for (std::size_t i = 0; i < dim; ++i) {
        norm += s2x[i] * s2x[i] + column[i] * column[i];
        const double delta = got[j * dim + i] - s2x[i];
        error += delta * delta;
        const double expected = y0[j * dim + i] + lambda * (s2x[i] - target * column[i]);
        const double pdelta = got_penalized[j * dim + i] - expected;
        penalized_error += pdelta * pdelta;
      }
      EXPECT_LE(std::sqrt(error), 1e-12 * std::sqrt(norm)) << "column " << j;
      EXPECT_LE(std::sqrt(penalized_error), 1e-12 * std::sqrt(norm)) << "column " << j;
    }
  }
}

TEST(CiSpinSquareGpu, Refuses) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const auto four = fb::StringSet::create(4, 2);
  const auto five = fb::StringSet::create(5, 2);
  ASSERT_TRUE(four.has_value() && five.has_value());
  EXPECT_FALSE(fb::CiSpinSquare::create(res, *four, *five).has_value());

  auto spin = fb::CiSpinSquare::create(res, *four, *four);
  ASSERT_TRUE(spin.has_value()) << spin.error().message;
  fb::Buffers::device<double> x(static_cast<std::size_t>(spin->size()), res.device());
  fb::Buffers::device<double> y(static_cast<std::size_t>(spin->size()), res.device());
  EXPECT_FALSE(spin->apply(res, -1, x.data(), y.data()).has_value());
  EXPECT_FALSE(spin->penalize(res, -1, 1.0, 0.0, x.data(), y.data()).has_value());
  EXPECT_TRUE(spin->apply(res, 0, x.data(), y.data()).has_value());
}

} // namespace
