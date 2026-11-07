// Host-only case for data/df_high_aux/, the hand-built basis
// df_high_aux_gpu.cpp runs the top of the density-fitting aux ladder over:
// the fixture is what it claims, every emitted metric and three-centre class
// reached. It reads the manifest into host BasisSets, so nothing touches a
// device.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.io.json;

#include "df_high_aux_fixture.h"

namespace {

namespace fb = free_breakfast;
using fb::test::high_aux::Fixture;
using fb::test::high_aux::fixture;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "df_high_aux";

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too (df_three_center_gpu.cpp's, over one primitive):
// Integral x^2l exp(-g r^2) = (2l-1)!! pi^(3/2) / ((2g)^l g^(3/2)).
double self_overlap(const fb::Shell &shell) {
  const std::int32_t l = shell.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  const fb::Primitive &p = shell.primitives().front();
  const double g = 2.0 * p.exponent;
  return p.coefficient * p.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
         std::pow(2.0 * g, l);
}

// Two shells on the same centre: the bra pair's P - A and P - B are then
// zero and the Boys argument 0, a different arithmetic path from a pair with
// a real separation, so the coverage case wants both for every class.
bool same_centre(const fb::Shell &a, const fb::Shell &b) {
  return a.center().x == b.center().x && a.center().y == b.center().y &&
         a.center().z == b.center().z;
}

std::int32_t max_momentum(const fb::BasisSet &basis) {
  std::int32_t l = 0;
  for (const fb::Shell &shell : basis.shells()) {
    l = std::max(l, shell.angular_momentum());
  }
  return l;
}

// The fixture is what it claims: the two momenta the ladder's limits name,
// every function unit-normalised (which is what lets the arrays be compared
// to libcint's with no norm factor), and -- the point of the file -- every
// emitted class of both families reached, each by a one-centre pair and a
// two-centre one.
TEST(DfHighAux, FixtureCoversEveryEmittedClass) {
  const auto doc = fb::read_json(kDataDir / "df_high_aux.json");
  ASSERT_TRUE(doc) << doc.error().message;
  const Fixture f = fixture(*doc);
  EXPECT_EQ(f.basis.nao_sphe(), f.nao);
  EXPECT_EQ(f.aux.nao_sphe(), f.naux);
  // The emitted ladder's two limits -- kThreeCenterMaxMomentum (f) and
  // kMetricMaxMomentum = kThreeCenterAuxMaxMomentum (i) -- as literals, not
  // as the bridge headers' constants: those headers carry the GPU runtime's
  // includes, which have no business in a module-scanned test TU. Raising
  // either limit means regenerating this fixture (its emitter's docstring
  // says so) and lifting the literal here.
  const auto orbital_max_l = (*doc)["orbital_max_l"].get<std::int32_t>();
  const auto aux_max_l = (*doc)["aux_max_l"].get<std::int32_t>();
  EXPECT_EQ(orbital_max_l, 3);
  EXPECT_EQ(aux_max_l, 6);
  EXPECT_EQ(max_momentum(f.basis), orbital_max_l);
  EXPECT_EQ(max_momentum(f.aux), aux_max_l);

  for (const fb::BasisSet *b : {&f.basis, &f.aux}) {
    for (const fb::Shell &shell : b->shells()) {
      EXPECT_NEAR(self_overlap(shell), 1.0, 1e-13) << "shell l " << shell.angular_momentum();
    }
  }

  // Metric: every (lP, lQ) with lP <= lQ <= kMetricMaxMomentum, and whether
  // the pair's two shells share a centre.
  std::set<std::array<std::int32_t, 3>> metric_seen;
  for (std::uint32_t q = 0; q < f.aux.nshells(); ++q) {
    for (std::uint32_t p = 0; p <= q; ++p) {
      const std::int32_t lp = f.aux.shells()[p].angular_momentum();
      const std::int32_t lq = f.aux.shells()[q].angular_momentum();
      const bool same = same_centre(f.aux.shells()[p], f.aux.shells()[q]);
      metric_seen.insert({std::min(lp, lq), std::max(lp, lq), same ? 1 : 0});
    }
  }
  const auto aux_ls = std::views::iota(0, aux_max_l + 1);
  for (const std::int32_t lp : aux_ls) {
    for (const std::int32_t lq : aux_ls) {
      if (lp > lq) {
        continue;
      }
      for (const std::int32_t same : {0, 1}) {
        EXPECT_TRUE(metric_seen.contains({lp, lq, same}))
            << "no " << (same == 1 ? "one-centre" : "two-centre") << " metric pair (" << lp << "|"
            << lq << ")";
      }
    }
  }

  // eri3: every (la, lb, lP), la <= lb <= kThreeCenterMaxMomentum against
  // lP <= kThreeCenterAuxMaxMomentum, by the bra pair's centres.
  std::set<std::array<std::int32_t, 4>> eri3_seen;
  for (std::uint32_t j = 0; j < f.basis.nshells(); ++j) {
    for (std::uint32_t i = 0; i <= j; ++i) {
      const std::int32_t la = f.basis.shells()[i].angular_momentum();
      const std::int32_t lb = f.basis.shells()[j].angular_momentum();
      const bool same = same_centre(f.basis.shells()[i], f.basis.shells()[j]);
      for (const fb::Shell &aux_shell : f.aux.shells()) {
        eri3_seen.insert(
            {std::min(la, lb), std::max(la, lb), aux_shell.angular_momentum(), same ? 1 : 0});
      }
    }
  }
  for (const std::int32_t la : std::views::iota(0, orbital_max_l + 1)) {
    for (const std::int32_t lb : std::views::iota(la, orbital_max_l + 1)) {
      for (const std::int32_t lp : aux_ls) {
        for (const std::int32_t same : {0, 1}) {
          EXPECT_TRUE(eri3_seen.contains({la, lb, lp, same}))
              << "no " << (same == 1 ? "one-centre" : "two-centre") << " eri3 class (" << la << lb
              << "|" << lp << ")";
        }
      }
    }
  }
}

} // namespace
