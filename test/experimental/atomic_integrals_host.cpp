// Host cases for experimental/atomic_scf's :integrals: the one-center integrals
// against closed forms, their symmetries, and the oxygen block of
// test/gtest's libcint golden data.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.io.basis;
import free_breakfast.experimental.atomic_scf;

#include "atomic_scf_reference.h"

namespace {

namespace fb = free_breakfast;
using fb::test::kOrigin;
using fb::test::max_abs;
using fb::test::reference_shells;

// U^T M U for M (r x r) and U (r x c), column-major.
std::vector<double> sandwich(const std::vector<double> &m, const std::vector<double> &u,
                             std::size_t r, std::size_t c) {
  std::vector<double> out(c * c, 0.0);
  for (std::size_t j = 0; j < c; ++j) {
    for (std::size_t i = 0; i < c; ++i) {
      double sum = 0.0;
      for (std::size_t b = 0; b < r; ++b) {
        for (std::size_t a = 0; a < r; ++a) {
          sum += u[a + (r * i)] * m[a + (r * b)] * u[b + (r * j)];
        }
      }
      out[i + (c * j)] = sum;
    }
  }
  return out;
}

// --- Closed forms and symmetry ----------------------------------------------

TEST(AtomicIntegrals, NormalizedSPrimitiveMatchesClosedForms) {
  // One normalized s primitive: S = 1, T = 3a/2, V = -2 z sqrt(2a/pi),
  // (ss|ss) = 2 sqrt(a/pi).
  const double a = 0.7;
  const double norm = std::pow(2.0 * a / std::numbers::pi, 0.75);
  const std::vector<fb::Shell> shells{fb::Shell(0, {{a, norm}}, kOrigin)};
  const auto ints = fb::atomic_integrals(3, shells);
  ASSERT_TRUE(ints) << ints.error().message;
  ASSERT_EQ(ints->nao_sphe, 1U);
  EXPECT_NEAR(ints->overlap[0], 1.0, 1e-14);
  EXPECT_NEAR(ints->kinetic[0], 1.5 * a, 1e-14);
  EXPECT_NEAR(ints->nuclear[0], -2.0 * 3.0 * std::sqrt(2.0 * a / std::numbers::pi), 1e-14);
  EXPECT_NEAR(ints->eri[0], 2.0 * std::sqrt(a / std::numbers::pi), 1e-14);
}

TEST(AtomicIntegrals, TranslationInvariant) {
  const std::vector<fb::Shell> at_origin = reference_shells("cc-pvdz", 8);
  std::vector<fb::Shell> moved;
  for (const fb::Shell &s : at_origin) {
    moved.emplace_back(s.angular_momentum(), s.primitives(), fb::Vec<double, 3>{1.5, -0.3, 2.0});
  }
  const auto a = fb::atomic_integrals(8, at_origin);
  const auto b = fb::atomic_integrals(8, moved);
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->eri, b->eri);
}

TEST(AtomicIntegrals, EriHasEightfoldSymmetry) {
  const std::vector<fb::Shell> shells = reference_shells("cc-pvdz", 8);
  const auto ints = fb::atomic_integrals(8, shells);
  ASSERT_TRUE(ints);
  const std::size_t n = ints->nao_sphe;
  const auto at = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return ints->eri[i + (n * (j + (n * (k + (n * l)))))];
  };
  const double scale = max_abs(ints->eri);
  for (std::size_t l = 0; l < n; ++l) {
    for (std::size_t k = 0; k < n; ++k) {
      for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
          const double v = at(i, j, k, l);
          EXPECT_NEAR(at(j, i, k, l), v, 1e-13 * scale);
          EXPECT_NEAR(at(i, j, l, k), v, 1e-13 * scale);
          EXPECT_NEAR(at(k, l, i, j), v, 1e-13 * scale);
        }
      }
    }
  }
}

// --- Against libcint ----------------------------------------------------------

// The oxygen of test/gtest's H2O: it is atom 0 at the origin, so its shells
// come first and its Cartesian block of S and T is one-center. V sums over all
// three nuclei, so it is not compared here (the SCF energies cover it).
class AtomicIntegralsGolden : public ::testing::TestWithParam<std::string> {};

TEST_P(AtomicIntegralsGolden, OxygenBlockMatchesLibcint) {
  const auto doc = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "one_electron" /
                                 (GetParam() + ".json"));
  ASSERT_TRUE(doc) << GetParam();
  std::vector<fb::Shell> shells;
  for (const auto &sh : (*doc)["shells"]) {
    const auto c = sh["centre"].get<std::vector<double>>();
    if (c != std::vector<double>{0.0, 0.0, 0.0}) {
      break;
    }
    shells.emplace_back(sh["l"].get<std::int32_t>(), sh["exponents"].get<std::vector<double>>(),
                        sh["coefficients"].get<std::vector<double>>(), kOrigin);
  }
  ASSERT_FALSE(shells.empty());
  const fb::BasisSet o(shells);
  const std::size_t nao = (*doc)["nao"].get<std::size_t>();
  const std::size_t ncart = o.nao_cart();
  const std::size_t nsph = o.nao_sphe();
  const std::vector<double> u = fb::spherical_transform_matrix(o);
  const auto block = [&](const char *name) {
    const auto all = (*doc)["arrays"][name]["values"].get<std::vector<double>>();
    std::vector<double> out(ncart * ncart);
    for (std::size_t j = 0; j < ncart; ++j) {
      for (std::size_t i = 0; i < ncart; ++i) {
        out[i + (ncart * j)] = all[(i * nao) + j];
      }
    }
    return sandwich(out, u, ncart, nsph);
  };

  const auto ints = fb::atomic_integrals(8, shells);
  ASSERT_TRUE(ints) << ints.error().message;
  ASSERT_EQ(ints->nao_sphe, nsph);
  for (const auto &[name, ours] :
       {std::pair{"int1e_ovlp", &ints->overlap}, std::pair{"int1e_kin", &ints->kinetic}}) {
    const std::vector<double> golden = block(name);
    const double tol = 1e-12 * max_abs(golden);
    for (std::size_t i = 0; i < golden.size(); ++i) {
      EXPECT_NEAR((*ours)[i], golden[i], tol) << name << " element " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(H2O, AtomicIntegralsGolden, ::testing::Values("h2o_sto3g", "h2o_ccpvdz"));

// --- Refusals -----------------------------------------------------------------

TEST(AtomicIntegrals, RefusesNoShellsAndSplitCenters) {
  EXPECT_EQ(fb::atomic_integrals(1, {}).error().kind, fb::ErrorKind::InvalidConfig);
  const std::vector<fb::Shell> split{fb::Shell(0, {{1.0, 1.0}}, kOrigin),
                                     fb::Shell(0, {{0.5, 1.0}}, fb::Vec<double, 3>{0.0, 0.0, 1.0})};
  EXPECT_EQ(fb::atomic_integrals(2, split).error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
