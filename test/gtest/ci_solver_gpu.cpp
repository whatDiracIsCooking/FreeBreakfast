// GPU cases for ci.solver and ci.rdm: ci_solve over the host sigma, wrapped as
// a LinearOperator, against PySCF's fci.direct_spin1 (data/ci/direct_spin1.json,
// written by tools/emit_fci_reference.py). Every case there has a triplet below
// its second singlet, so the singlets come out in order only if the spin
// penalty works; with no penalty the triplet appears among them. Over CiSigma,
// a singlet solve runs in the symmetric layout, and the test records that it
// did; a solve started from converged roots, on either path, takes a step or
// two per stage, and an empty start changes nothing.

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
// ci_solve's options each solve runs under: (olsen_correction,
// davidson.canonical_orthonormalization, spin_averaged_diagonal), the defaults
// first.
constexpr std::array<std::tuple<bool, bool, bool>, 4> kSolverVariants{
    {{false, false, false}, {true, false, false}, {false, true, false}, {false, false, true}}};

// host_sigma as a LinearOperator: each column copied to the host, through
// host_sigma, and back. k is (ij|ji) on the device, for the spin-averaged
// diagonal, or null.
struct HostSigma {
  const fb::StringSet &alpha;
  const fb::StringSet &beta;
  double e_core;
  const std::vector<double> &h1;
  const std::vector<double> &eri;
  std::int32_t applies = 0;
  const double *k = nullptr;

  [[nodiscard]] const double *exchange() const noexcept { return k; }

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

// @p host on the device.
fb::Buffers::device<double> on_device(fb::DeviceResources &res, std::span<const double> host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

// The @p count doubles at @p device on the host.
std::vector<double> on_host(fb::DeviceResources &res, const double *device, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), device, count * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// (ij|ji) of @p c's integrals on the device, n × n, as CiSigma::exchange holds
// it.
fb::Buffers::device<double> exchange_on_device(fb::DeviceResources &res, const Case &c) {
  const auto n = static_cast<std::size_t>(c.n);
  std::vector<double> k(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      k[i * n + j] = c.eri[((i * n + j) * n + j) * n + i];
    }
  }
  return on_device(res, k);
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
void expect_roots(fb::DeviceResources &res, const Case &c, const fb::CiRoots &got, double s2) {
  const std::vector<std::size_t> golden = golden_of_spin(c.roots, s2);
  ASSERT_GE(golden.size(), got.energies.size());
  const std::size_t dim =
      static_cast<std::size_t>(c.alpha.size()) * static_cast<std::size_t>(c.beta.size());
  ASSERT_EQ(got.vectors.num_elements(), dim * got.energies.size());
  const std::vector<double> vectors = on_host(res, got.vectors.data(), got.vectors.num_elements());
  const auto m = static_cast<std::size_t>(c.n);
  std::int32_t compared = 0;
  for (std::size_t r = 0; r < got.energies.size(); ++r) {
    SCOPED_TRACE(std::format("root {} (golden {})", r, golden[r]));
    const fb::json &want = c.roots[golden[r]];
    EXPECT_NEAR(got.energies[r], want["energy"].get<double>(), kEnergyTol);
    EXPECT_NEAR(got.spin_squares[r], s2, kSpinTol);

    std::vector<double> gamma(m * m);
    ASSERT_TRUE(
        fb::rdm1(c.alpha, c.beta, std::span<const double>{vectors}.subspan(r * dim, dim), gamma)
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

    const auto diagonal = on_device(res, c.diagonal);
    const auto exchange = exchange_on_device(res, c);
    for (const auto [olsen, canonical, averaged] : kSolverVariants) {
      SCOPED_TRACE(std::format("olsen_correction {}, canonical_orthonormalization {}, "
                               "spin_averaged_diagonal {}",
                               olsen, canonical, averaged));
      HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri, 0, exchange.data()};
      fb::CiSolverOptions options;
      options.olsen_correction = olsen;
      options.davidson.canonical_orthonormalization = canonical;
      options.spin_averaged_diagonal = averaged;
      const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 3, options);
      ASSERT_TRUE(got.has_value()) << got.error().message;
      EXPECT_EQ(sigma.applies, got->davidson.iterations);
      expect_roots(res, c, *got, 0.0);
    }
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
    const auto exchange = exchange_on_device(res, c);
    HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri, 0, exchange.data()};
    const auto diagonal = on_device(res, c.diagonal);
    for (const auto [olsen, canonical, averaged] : kSolverVariants) {
      SCOPED_TRACE(std::format("olsen_correction {}, canonical_orthonormalization {}, "
                               "spin_averaged_diagonal {}",
                               olsen, canonical, averaged));
      fb::CiSolverOptions options;
      options.olsen_correction = olsen;
      options.davidson.canonical_orthonormalization = canonical;
      options.spin_averaged_diagonal = averaged;
      const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 3, options);
      ASSERT_TRUE(got.has_value()) << got.error().message;
      expect_roots(res, c, *got, 2.0);
    }
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
  const auto exchange = exchange_on_device(res, c);
  HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri, 0, exchange.data()};
  const auto diagonal = on_device(res, c.diagonal);
  for (const auto [olsen, canonical, averaged] : kSolverVariants) {
    SCOPED_TRACE(std::format("olsen_correction {}, canonical_orthonormalization {}, "
                             "spin_averaged_diagonal {}",
                             olsen, canonical, averaged));
    fb::CiSolverOptions options;
    options.spin_penalty = 0.0;
    options.olsen_correction = olsen;
    options.davidson.canonical_orthonormalization = canonical;
    options.spin_averaged_diagonal = averaged;
    const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 3, options);
    ASSERT_TRUE(got.has_value()) << got.error().message;
    bool triplet = false;
    for (std::size_t r = 0; r < 3; ++r) {
      EXPECT_NEAR(got->energies[r], c.roots[r]["energy"].get<double>(), kEnergyTol);
      EXPECT_NEAR(got->spin_squares[r], c.roots[r]["s2"].get<double>(), kSpinTol);
      triplet = triplet || std::abs(got->spin_squares[r] - 2.0) < kSpinTol;
    }
    EXPECT_TRUE(triplet);
  }
}

// CiSigma, forwarded, with the layout of every apply recorded: what ci_solve
// chose; and the columns of the first FP64 apply, the ones the FP64 Davidson
// starts from.
struct RecordingSigma {
  fb::CiSigma &sigma;
  std::set<fb::CiLayout> layouts;
  std::int32_t first_fp64_columns = 0;

  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const double *x, double *y,
                   fb::CiLayout layout = fb::CiLayout::full) {
    layouts.insert(layout);
    if (first_fp64_columns == 0) {
      first_fp64_columns = k;
    }
    return sigma.apply(res, k, x, y, layout);
  }
  fb::Status apply_fp32(fb::DeviceResources &res, std::int32_t k, const double *x, double *y,
                        fb::CiLayout layout = fb::CiLayout::full) {
    layouts.insert(layout);
    return sigma.apply_fp32(res, k, x, y, layout);
  }
  [[nodiscard]] const double *exchange() const noexcept { return sigma.exchange(); }
};

// Over CiSigma, which has an apply_fp32, ci_solve runs its single-precision
// stage first by default; the roots meet the same tolerances as the FP64-only
// solve, which fp32_tolerance = 0 asks for. Singlets, in the symmetric layout
// and, from the M_s = 1 sector, triplets, in the full one. The FP64 stage
// starts from X's three columns, or with restart_with_previous from
// [X, X_prev] orthonormalized, more than three.
TEST(CiSolver, TwoStageOverCiSigmaMatchesDirectSpin1) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  for (const auto &entry : cases) {
    for (const std::int32_t shift : {0, 1}) {
      const Case c = make_case(entry, shift);
      SCOPED_TRACE(std::format("{}, M_s {}", c.name, shift));
      auto sigma = fb::CiSigma::create(res, c.alpha, c.beta);
      ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
      ASSERT_TRUE(sigma->set_integrals(res, c.e_core, c.h1, c.eri).has_value());
      fb::Buffers::device<double> diagonal(c.diagonal.size(), res.device());
      ASSERT_TRUE(sigma->diagonal(res, diagonal.data()).has_value());
      const double s2 = shift == 0 ? 0.0 : 2.0;
      const std::set<fb::CiLayout> layout{shift == 0 ? fb::CiLayout::symmetric
                                                     : fb::CiLayout::full};

      for (const auto [olsen, canonical, averaged] : kSolverVariants) {
        SCOPED_TRACE(std::format("olsen_correction {}, canonical_orthonormalization {}, "
                                 "spin_averaged_diagonal {}",
                                 olsen, canonical, averaged));
        fb::CiSolverOptions options;
        options.olsen_correction = olsen;
        options.davidson.canonical_orthonormalization = canonical;
        options.spin_averaged_diagonal = averaged;
        for (const bool previous : {false, true}) {
          SCOPED_TRACE(std::format("restart_with_previous {}", previous));
          RecordingSigma recorded{*sigma, {}};
          fb::CiSolverOptions restart = options;
          restart.restart_with_previous = previous;
          const auto two_stage =
              fb::ci_solve(res, recorded, c.alpha, c.beta, diagonal.data(), 3, restart);
          ASSERT_TRUE(two_stage.has_value()) << two_stage.error().message;
          EXPECT_GT(two_stage->fp32_iterations, 0);
          if (previous) {
            EXPECT_GT(recorded.first_fp64_columns, 3);
          } else {
            EXPECT_EQ(recorded.first_fp64_columns, 3);
          }
          EXPECT_EQ(recorded.layouts, layout);
          expect_roots(res, c, *two_stage, s2);
        }

        RecordingSigma recorded{*sigma, {}};
        fb::CiSolverOptions fp64 = options;
        fp64.fp32_tolerance = 0.0;
        const auto alone = fb::ci_solve(res, recorded, c.alpha, c.beta, diagonal.data(), 3, fp64);
        ASSERT_TRUE(alone.has_value()) << alone.error().message;
        EXPECT_EQ(alone->fp32_iterations, 0);
        EXPECT_EQ(recorded.layouts, layout);
        expect_roots(res, c, *alone, s2);
      }
    }
  }
}

// Started from a solve's own converged roots (CiRoots::vectors, the full
// layout), ci_solve converges in at most two iterations per stage to the same
// energies, to 1e-12, and 1-RDMs, to 1e-10 where the root is not one of a
// degenerate level: singlets on the symmetric path, where the start is
// packed, and triplets on the full one; in two stages and in the FP64 one
// alone. An empty start is the solve without one, to the bit.
TEST(CiSolver, WarmStartFromConvergedRootsOverCiSigma) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  for (const auto &entry : cases) {
    for (const std::int32_t shift : {0, 1}) {
      const Case c = make_case(entry, shift);
      SCOPED_TRACE(std::format("{}, M_s {}", c.name, shift));
      auto sigma = fb::CiSigma::create(res, c.alpha, c.beta);
      ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
      ASSERT_TRUE(sigma->set_integrals(res, c.e_core, c.h1, c.eri).has_value());
      fb::Buffers::device<double> diagonal(c.diagonal.size(), res.device());
      ASSERT_TRUE(sigma->diagonal(res, diagonal.data()).has_value());
      const double s2 = shift == 0 ? 0.0 : 2.0;
      const std::vector<std::size_t> golden = golden_of_spin(c.roots, s2);
      const auto m = static_cast<std::size_t>(c.n);
      const std::size_t dim = c.diagonal.size();
      const std::set<fb::CiLayout> layout{shift == 0 ? fb::CiLayout::symmetric
                                                     : fb::CiLayout::full};

      for (const auto [fp32_tolerance, averaged] :
           std::array<std::pair<double, bool>, 3>{{{fb::CiSolverOptions{}.fp32_tolerance, false},
                                                   {0.0, false},
                                                   {fb::CiSolverOptions{}.fp32_tolerance, true}}}) {
        SCOPED_TRACE(
            std::format("fp32_tolerance {}, spin_averaged_diagonal {}", fp32_tolerance, averaged));
        fb::CiSolverOptions options;
        options.fp32_tolerance = fp32_tolerance;
        options.spin_averaged_diagonal = averaged;
        const auto cold = fb::ci_solve(res, *sigma, c.alpha, c.beta, diagonal.data(), 3, options);
        ASSERT_TRUE(cold.has_value()) << cold.error().message;
        const std::vector<double> cold_vectors =
            on_host(res, cold->vectors.data(), cold->vectors.num_elements());

        const auto empty =
            fb::ci_solve(res, *sigma, c.alpha, c.beta, diagonal.data(), 3, options, {});
        ASSERT_TRUE(empty.has_value()) << empty.error().message;
        EXPECT_EQ(empty->energies, cold->energies);
        EXPECT_EQ(empty->spin_squares, cold->spin_squares);
        EXPECT_EQ(empty->davidson.iterations, cold->davidson.iterations);
        EXPECT_EQ(empty->fp32_iterations, cold->fp32_iterations);
        EXPECT_EQ(on_host(res, empty->vectors.data(), empty->vectors.num_elements()), cold_vectors);

        RecordingSigma recorded{*sigma, {}};
        const auto warm = fb::ci_solve(
            res, recorded, c.alpha, c.beta, diagonal.data(), 3, options,
            std::span<const double>{cold->vectors.data(), cold->vectors.num_elements()});
        ASSERT_TRUE(warm.has_value()) << warm.error().message;
        EXPECT_EQ(recorded.layouts, layout);
        EXPECT_LE(warm->davidson.iterations, 2);
        EXPECT_LE(warm->fp32_iterations, 2);
        if (fp32_tolerance > 0.0) {
          EXPECT_GT(warm->fp32_iterations, 0);
        } else {
          EXPECT_EQ(warm->fp32_iterations, 0);
        }
        std::cout << std::format("[ warm     ] {} M_s {} fp32_tolerance {} spin_averaged {}: "
                                 "cold {} + {}, warm {} + {} iterations\n",
                                 c.name, shift, fp32_tolerance, averaged, cold->fp32_iterations,
                                 cold->davidson.iterations, warm->fp32_iterations,
                                 warm->davidson.iterations);
        expect_roots(res, c, *warm, s2);

        const std::vector<double> warm_vectors =
            on_host(res, warm->vectors.data(), warm->vectors.num_elements());
        for (std::size_t r = 0; r < 3; ++r) {
          SCOPED_TRACE(std::format("root {}", r));
          EXPECT_NEAR(warm->energies[r], cold->energies[r], 1e-12);
          if (c.roots[golden[r]]["degenerate"].get<bool>()) {
            continue;
          }
          std::vector<double> warm_gamma(m * m);
          std::vector<double> cold_gamma(m * m);
          ASSERT_TRUE(fb::rdm1(c.alpha, c.beta,
                               std::span<const double>{warm_vectors}.subspan(r * dim, dim),
                               warm_gamma)
                          .has_value());
          ASSERT_TRUE(fb::rdm1(c.alpha, c.beta,
                               std::span<const double>{cold_vectors}.subspan(r * dim, dim),
                               cold_gamma)
                          .has_value());
          for (std::size_t i = 0; i < m * m; ++i) {
            EXPECT_NEAR(warm_gamma[i], cold_gamma[i], 1e-10) << "element " << i;
          }
        }
      }
    }
  }
}

// With no penalty, a sigma that takes a layout still gets the full one: the
// symmetric layout holds even S alone, and λ = 0 asks for every spin.
TEST(CiSolver, NoPenaltyOverCiSigmaKeepsEverySpin) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  const Case c = make_case(cases[0]);
  auto sigma = fb::CiSigma::create(res, c.alpha, c.beta);
  ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
  ASSERT_TRUE(sigma->set_integrals(res, c.e_core, c.h1, c.eri).has_value());
  const auto diagonal = on_device(res, c.diagonal);
  for (const auto [olsen, canonical, averaged] : kSolverVariants) {
    SCOPED_TRACE(std::format("olsen_correction {}, canonical_orthonormalization {}, "
                             "spin_averaged_diagonal {}",
                             olsen, canonical, averaged));
    RecordingSigma recorded{*sigma, {}};
    fb::CiSolverOptions options;
    options.spin_penalty = 0.0;
    options.olsen_correction = olsen;
    options.davidson.canonical_orthonormalization = canonical;
    options.spin_averaged_diagonal = averaged;
    const auto got = fb::ci_solve(res, recorded, c.alpha, c.beta, diagonal.data(), 3, options);
    ASSERT_TRUE(got.has_value()) << got.error().message;
    EXPECT_EQ(recorded.layouts, std::set<fb::CiLayout>{fb::CiLayout::full});
    bool triplet = false;
    for (std::size_t r = 0; r < 3; ++r) {
      EXPECT_NEAR(got->energies[r], c.roots[r]["energy"].get<double>(), kEnergyTol);
      triplet = triplet || std::abs(got->spin_squares[r] - 2.0) < kSpinTol;
    }
    EXPECT_TRUE(triplet);
  }
}

TEST(CiSolver, TooFewIterationsIsMaxIterationsExceeded) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  const Case c = make_case(cases[1]);
  const auto exchange = exchange_on_device(res, c);
  HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri, 0, exchange.data()};
  const auto diagonal = on_device(res, c.diagonal);
  for (const bool averaged : {false, true}) {
    SCOPED_TRACE(std::format("spin_averaged_diagonal {}", averaged));
    fb::CiSolverOptions options;
    options.davidson.max_iterations = 2;
    options.spin_averaged_diagonal = averaged;
    const auto got = fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 3, options);
    ASSERT_FALSE(got.has_value());
    EXPECT_EQ(got.error().kind, fb::ErrorKind::MaxIterationsExceeded) << got.error().message;
  }
}

TEST(CiSolver, RejectsBadShapes) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  const Case c = make_case(cases[0]);
  HostSigma sigma{c.alpha, c.beta, c.e_core, c.h1, c.eri};
  const auto dim = static_cast<std::int32_t>(c.diagonal.size());
  const auto diagonal = on_device(res, c.diagonal);
  const auto kind = [](const fb::Result<fb::CiRoots> &got) {
    return got.has_value() ? std::optional<fb::ErrorKind>{} : got.error().kind;
  };
  const auto invalid = std::optional{fb::ErrorKind::InvalidConfig};
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 0)), invalid);
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), dim + 1)), invalid);
  const auto other = fb::StringSet::create(c.n + 1, c.beta.n_electrons());
  ASSERT_TRUE(other.has_value());
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, *other, diagonal.data(), 1)), invalid);
  fb::CiSolverOptions negative;
  negative.spin_penalty = -1.0;
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 1, negative)), invalid);
  fb::CiSolverOptions negative_fp32;
  negative_fp32.fp32_tolerance = -1.0;
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 1, negative_fp32)),
            invalid);
  // The spin-averaged diagonal wants the sigma's (ij|ji): this one's is null.
  fb::CiSolverOptions averaged;
  averaged.spin_averaged_diagonal = true;
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 1, averaged)), invalid);
  // A start is n_roots vectors, and independent ones.
  const fb::Buffers::device<double> zeros =
      on_device(res, std::vector<double>(3 * c.diagonal.size(), 0.0));
  const std::span<const double> start{zeros.data(), zeros.num_elements()};
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 2, {}, start)),
            invalid);
  EXPECT_EQ(kind(fb::ci_solve(res, sigma, c.alpha, c.beta, diagonal.data(), 3, {}, start)),
            std::optional{fb::ErrorKind::Numerical});
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

    // penalize_diagonal against diag(S²) = S_z (S_z + 1) + N_β less the
    // doubly occupied orbitals, from the strings, on the first column of x
    // standing in for diag(H); and in place.
    const double two_sz = s.n_alpha - s.n_beta;
    const double constant = (0.25 * two_sz * (two_sz + 2.0)) + s.n_beta;
    std::vector<double> want_diagonal(dim);
    for (std::size_t ia = 0; ia < static_cast<std::size_t>(alpha->size()); ++ia) {
      for (std::size_t ib = 0; ib < static_cast<std::size_t>(beta->size()); ++ib) {
        const std::size_t det = ia * static_cast<std::size_t>(beta->size()) + ib;
        const std::int32_t doubles = std::popcount(alpha->strings()[ia] & beta->strings()[ib]);
        want_diagonal[det] = x[det] + lambda * (constant - doubles - target);
      }
    }
    fb::Buffers::device<double> shifted(dim, res.device());
    ASSERT_TRUE(
        spin->penalize_diagonal(res, lambda, target, xd.data(), shifted.data()).has_value());
    // To roundoff, as the device may fuse the multiply-add.
    const auto expect_diagonal = [&](const std::vector<double> &got_diagonal) {
      ASSERT_EQ(got_diagonal.size(), want_diagonal.size());
      for (std::size_t i = 0; i < dim; ++i) {
        EXPECT_NEAR(got_diagonal[i], want_diagonal[i], 1e-14) << "determinant " << i;
      }
    };
    expect_diagonal(download(shifted));
    auto in_place = upload(std::vector<double>(x.begin(), x.begin() + dim));
    ASSERT_TRUE(
        spin->penalize_diagonal(res, lambda, target, in_place.data(), in_place.data()).has_value());
    expect_diagonal(download(in_place));

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

// CiSpinSquare in the symmetric layout, over one set for both spins, against
// host_spin_square on random swap-symmetric blocks of three vectors, packed:
// apply and penalize to 1e-12 of the norm, and penalize_diagonal as the full
// layout's restricted to the triangle.
TEST(CiSpinSquareGpu, SymmetricLayoutMatchesHost) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr std::int32_t k = 3;
  constexpr double lambda = 0.7;
  constexpr double target = 0.0;
  for (const auto [n, n_electrons] : std::array<std::pair<std::int32_t, std::int32_t>, 6>{
           {{6, 3}, {8, 4}, {7, 2}, {5, 0}, {4, 4}, {1, 1}}}) {
    SCOPED_TRACE(std::format("n {} N {}", n, n_electrons));
    const auto set = fb::StringSet::create(n, n_electrons);
    ASSERT_TRUE(set.has_value());
    auto spin = fb::CiSpinSquare::create(res, *set, *set);
    ASSERT_TRUE(spin.has_value()) << spin.error().message;
    const auto m = static_cast<std::size_t>(set->size());
    const std::size_t dim = m * m;
    const std::size_t packed = m * (m + 1) / 2;
    ASSERT_EQ(static_cast<std::size_t>(spin->size(fb::CiLayout::symmetric)), packed);

    std::mt19937_64 engine{dim + 1};
    std::uniform_real_distribution<double> uniform{-1.0, 1.0};
    std::vector<double> x(dim * k);
    for (std::size_t j = 0; j < k; ++j) {
      for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
          x[j * dim + a * m + b] = x[j * dim + b * m + a] = uniform(engine);
        }
      }
    }
    std::vector<double> y0(packed * k);
    for (double &v : y0) {
      v = uniform(engine);
    }
    // The triangle, off-diagonal elements times sqrt(2).
    const auto pack = [&](std::span<const double> full) {
      std::vector<double> out(packed);
      for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
          out[a * (a + 1) / 2 + b] =
              a == b ? full[a * m + b] : std::numbers::sqrt2 * full[a * m + b];
        }
      }
      return out;
    };
    std::vector<double> u(packed * k);
    for (std::size_t j = 0; j < k; ++j) {
      std::ranges::copy(pack(std::span<const double>{x}.subspan(j * dim, dim)),
                        u.begin() + static_cast<std::ptrdiff_t>(j * packed));
    }
    const auto ud = on_device(res, u);
    fb::Buffers::device<double> applied(packed * k, res.device());
    ASSERT_TRUE(
        spin->apply(res, k, ud.data(), applied.data(), fb::CiLayout::symmetric).has_value());
    const std::vector<double> got = on_host(res, applied.data(), packed * k);
    auto penalized = on_device(res, y0);
    ASSERT_TRUE(
        spin->penalize(res, k, lambda, target, ud.data(), penalized.data(), fb::CiLayout::symmetric)
            .has_value());
    const std::vector<double> got_penalized = on_host(res, penalized.data(), packed * k);

    std::vector<double> s2x(dim);
    for (std::size_t j = 0; j < k; ++j) {
      const std::span<const double> column{x.data() + j * dim, dim};
      ASSERT_TRUE(fb::host_spin_square(*set, *set, column, s2x).has_value());
      const std::vector<double> want = pack(s2x);
      double norm = 0.0;
      double error = 0.0;
      double penalized_error = 0.0;
      for (std::size_t i = 0; i < packed; ++i) {
        norm += want[i] * want[i] + u[j * packed + i] * u[j * packed + i];
        const double delta = got[j * packed + i] - want[i];
        error += delta * delta;
        const double expected =
            y0[j * packed + i] + lambda * (want[i] - target * u[j * packed + i]);
        const double pdelta = got_penalized[j * packed + i] - expected;
        penalized_error += pdelta * pdelta;
      }
      EXPECT_LE(std::sqrt(error), 1e-12 * std::sqrt(norm)) << "column " << j;
      EXPECT_LE(std::sqrt(penalized_error), 1e-12 * std::sqrt(norm)) << "column " << j;
    }

    // The first column of x standing in for diag(H): the full layout's
    // penalized diagonal, read on the triangle unscaled.
    const auto xd = on_device(res, std::span<const double>{x}.first(dim));
    fb::Buffers::device<double> full(dim, res.device());
    ASSERT_TRUE(spin->penalize_diagonal(res, lambda, target, xd.data(), full.data()).has_value());
    fb::Buffers::device<double> triangle(packed, res.device());
    ASSERT_TRUE(spin->penalize_diagonal(res, lambda, target, xd.data(), triangle.data(),
                                        fb::CiLayout::symmetric)
                    .has_value());
    const std::vector<double> want_full = on_host(res, full.data(), dim);
    const std::vector<double> got_triangle = on_host(res, triangle.data(), packed);
    for (std::size_t a = 0; a < m; ++a) {
      for (std::size_t b = 0; b <= a; ++b) {
        EXPECT_EQ(got_triangle[a * (a + 1) / 2 + b], want_full[a * m + b]) << a << ", " << b;
      }
    }
  }
}

// penalize_diagonal with CiSigma::exchange against the spin-averaged diag(H)
// by brute force: each determinant's host_diagonal averaged over the
// determinants of its configuration (the same doubly and the same singly
// occupied orbitals) in the sector, plus the penalty's diagonal, to 1e-12.
// Every direct_spin1 case in four sectors, N_alpha below N_beta among them;
// the singlet sector in the symmetric layout too, on the triangle.
TEST(CiSpinSquareGpu, SpinAveragedDiagonalMatchesHost) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  constexpr double lambda = 0.7;
  constexpr double target = 0.5;
  const fb::json cases = load_cases();
  ASSERT_FALSE(cases.empty());
  for (const auto &entry : cases) {
    for (const std::int32_t shift : {0, 1, 2, -1}) {
      const Case c = make_case(entry, shift);
      SCOPED_TRACE(std::format("{}, shift {}", c.name, shift));
      auto sigma = fb::CiSigma::create(res, c.alpha, c.beta);
      ASSERT_TRUE(sigma.has_value()) << sigma.error().message;
      EXPECT_EQ(sigma->exchange(), nullptr);
      ASSERT_TRUE(sigma->set_integrals(res, c.e_core, c.h1, c.eri).has_value());
      ASSERT_NE(sigma->exchange(), nullptr);
      auto spin = fb::CiSpinSquare::create(res, c.alpha, c.beta);
      ASSERT_TRUE(spin.has_value()) << spin.error().message;

      // The average as diag_J + mean_K (diag_K − diag_J), so the roundoff is
      // the differences', not E_core's.
      const auto nb = static_cast<std::size_t>(c.beta.size());
      std::map<std::pair<std::uint64_t, std::uint64_t>, std::vector<std::size_t>> configurations;
      for (std::size_t ia = 0; ia < static_cast<std::size_t>(c.alpha.size()); ++ia) {
        for (std::size_t ib = 0; ib < nb; ++ib) {
          const std::uint64_t a = c.alpha.strings()[ia];
          const std::uint64_t b = c.beta.strings()[ib];
          configurations[{a & b, a ^ b}].push_back(ia * nb + ib);
        }
      }
      const double two_sz = c.alpha.n_electrons() - c.beta.n_electrons();
      const double constant = (0.25 * two_sz * (two_sz + 2.0)) + c.beta.n_electrons();
      std::vector<double> want(c.diagonal.size());
      bool mixed = false;
      for (const auto &[key, dets] : configurations) {
        mixed = mixed || dets.size() > 1;
        for (const std::size_t j : dets) {
          double spread = 0.0;
          for (const std::size_t k : dets) {
            spread += c.diagonal[k] - c.diagonal[j];
          }
          want[j] = c.diagonal[j] + (spread / static_cast<double>(dets.size())) +
                    lambda * (constant - std::popcount(key.first) - target);
        }
      }
      // The singlet sector has configurations of more than one determinant.
      if (shift == 0) {
        EXPECT_TRUE(mixed);
      }

      const auto diagonal = on_device(res, c.diagonal);
      fb::Buffers::device<double> full(c.diagonal.size(), res.device());
      ASSERT_TRUE(spin->penalize_diagonal(res, lambda, target, diagonal.data(), full.data(),
                                          fb::CiLayout::full, sigma->exchange())
                      .has_value());
      const std::vector<double> got = on_host(res, full.data(), c.diagonal.size());
      for (std::size_t i = 0; i < want.size(); ++i) {
        EXPECT_NEAR(got[i], want[i], 1e-12) << "determinant " << i;
      }
      if (shift != 0) {
        continue;
      }
      const auto m = static_cast<std::size_t>(c.alpha.size());
      fb::Buffers::device<double> triangle(m * (m + 1) / 2, res.device());
      ASSERT_TRUE(spin->penalize_diagonal(res, lambda, target, diagonal.data(), triangle.data(),
                                          fb::CiLayout::symmetric, sigma->exchange())
                      .has_value());
      const std::vector<double> got_triangle = on_host(res, triangle.data(), m * (m + 1) / 2);
      for (std::size_t a = 0; a < m; ++a) {
        for (std::size_t b = 0; b <= a; ++b) {
          EXPECT_NEAR(got_triangle[a * (a + 1) / 2 + b], want[a * m + b], 1e-12) << a << ", " << b;
        }
      }
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

  // The symmetric layout wants one set for both spins.
  const auto one = fb::StringSet::create(4, 1);
  const auto three = fb::StringSet::create(4, 3);
  ASSERT_TRUE(one.has_value() && three.has_value());
  auto mixed = fb::CiSpinSquare::create(res, *one, *three);
  ASSERT_TRUE(mixed.has_value()) << mixed.error().message;
  EXPECT_FALSE(mixed->apply(res, 1, x.data(), y.data(), fb::CiLayout::symmetric).has_value());
  EXPECT_FALSE(
      mixed->penalize(res, 1, 1.0, 0.0, x.data(), y.data(), fb::CiLayout::symmetric).has_value());
  EXPECT_FALSE(mixed->penalize_diagonal(res, 1.0, 0.0, x.data(), y.data(), fb::CiLayout::symmetric)
                   .has_value());
}

} // namespace
