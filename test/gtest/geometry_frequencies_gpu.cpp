// GPU cases for geometry::harmonic_analysis: PySCF's RHF/cc-pVDZ Hessians of
// bent water, CO2 along z, HCN off every axis and linear water (a saddle point,
// its degenerate bend imaginary), against pyscf.hessian.thermo.
// harmonic_analysis of the same matrix (test/gtest/data/frequencies/,
// tools/emit/frequencies_reference.py): the frequencies to 1e-3 cm^-1, the
// rigid motions removed (3N - 6, or 3N - 5 linear, modes), and the modes up to
// sign, compared as the projector onto each degenerate set so a doubly
// degenerate bend's arbitrary basis does not matter. Then the negated Hessian
// (every frequency imaginary, the order reversed), a spring between two
// hydrogens against sqrt(k / mu), a single atom (no modes), and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.io.json;
import free_breakfast.geometry;

namespace {

namespace fb = free_breakfast;
namespace geo = free_breakfast::geometry;

// The issue's bar for the frequencies (cm^-1).
constexpr double kFrequencyTol = 1e-3;
// Each degenerate set's mass-weighted projector, element by element.
constexpr double kModeTol = 1e-7;
// Frequencies closer than this (cm^-1) are one degenerate set.
constexpr double kDegenerate = 1e-2;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "frequencies";

struct Case {
  std::vector<fb::Atom> atoms;
  std::vector<double> masses;
  std::vector<double> hessian;
  std::vector<double> frequencies;
  std::vector<double> modes;
};

Case read_case(const std::string &name) {
  const auto doc = fb::read_json(kDataDir / "frequencies.json");
  EXPECT_TRUE(doc) << doc.error().message;
  Case out;
  if (!doc) {
    return out;
  }
  for (const fb::json &c : (*doc)["cases"]) {
    if (c["name"].get<std::string>() != name) {
      continue;
    }
    for (const fb::json &atom : c["atoms"]) {
      const auto xyz = atom["coord"].get<std::vector<double>>();
      out.atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                                   static_cast<fb::Element>(atom["z"].get<std::int32_t>())});
    }
    out.masses = c["masses"].get<std::vector<double>>();
    out.hessian = c["hessian"].get<std::vector<double>>();
    out.frequencies = c["frequencies"].get<std::vector<double>>();
    out.modes = c["modes"].get<std::vector<double>>();
    return out;
  }
  ADD_FAILURE() << name << " is not a case of frequencies.json";
  return out;
}

// The projector onto the mass-weighted modes [first, last), n x n row-major.
std::vector<double> projector(const std::vector<double> &modes, const std::vector<double> &masses,
                              std::size_t first, std::size_t last, std::size_t n) {
  std::vector<double> p(n * n, 0.0);
  for (std::size_t j = first; j < last; ++j) {
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b = 0; b < n; ++b) {
        p[a * n + b] += modes[j * n + a] * std::sqrt(masses[a / 3]) * modes[j * n + b] *
                        std::sqrt(masses[b / 3]);
      }
    }
  }
  return p;
}

class HarmonicAnalysis : public ::testing::TestWithParam<std::string> {};

TEST_P(HarmonicAnalysis, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  const Case c = read_case(GetParam());
  ASSERT_FALSE(c.atoms.empty());
  const std::size_t n = 3 * c.atoms.size();

  fb::DeviceResources res{0};
  const auto vib = geo::harmonic_analysis(res, c.atoms, c.hessian);
  ASSERT_TRUE(vib) << vib.error().message;

  // Six rigid motions removed, five for the linear ones: the count PySCF kept.
  ASSERT_EQ(vib->frequencies.size(), c.frequencies.size());
  ASSERT_EQ(vib->modes.size(), c.frequencies.size() * n);
  for (std::size_t j = 0; j < c.frequencies.size(); ++j) {
    EXPECT_NEAR(vib->frequencies[j], c.frequencies[j], kFrequencyTol) << "mode " << j;
  }

  // The modes, one degenerate set at a time.
  for (std::size_t first = 0; first < c.frequencies.size();) {
    std::size_t last = first + 1;
    while (last < c.frequencies.size() &&
           c.frequencies[last] - c.frequencies[last - 1] < kDegenerate) {
      ++last;
    }
    const auto ours = projector(vib->modes, c.masses, first, last, n);
    const auto theirs = projector(c.modes, c.masses, first, last, n);
    for (std::size_t e = 0; e < n * n; ++e) {
      ASSERT_NEAR(ours[e], theirs[e], kModeTol) << "modes [" << first << ", " << last << ")";
    }
    first = last;
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, HarmonicAnalysis,
                         ::testing::Values("h2o", "co2", "hcn", "h2o_linear"));

// -H: every eigenvalue negated, so every frequency is imaginary, reported as
// minus PySCF's, in reversed (ascending) order.
TEST(HarmonicAnalysisSign, NegatedHessianIsImaginary) {
  FBF_REQUIRE_GPU();
  Case c = read_case("h2o");
  ASSERT_FALSE(c.atoms.empty());
  for (double &value : c.hessian) {
    value = -value;
  }
  fb::DeviceResources res{0};
  const auto vib = geo::harmonic_analysis(res, c.atoms, c.hessian);
  ASSERT_TRUE(vib) << vib.error().message;
  ASSERT_EQ(vib->frequencies.size(), c.frequencies.size());
  for (std::size_t j = 0; j < c.frequencies.size(); ++j) {
    EXPECT_NEAR(vib->frequencies[j], -c.frequencies[c.frequencies.size() - 1 - j], kFrequencyTol);
  }
}

// Two hydrogens joined by a spring of constant k along a skew bond: one mode,
// at sqrt(k / mu), mu = m_H / 2, in cm^-1 (the dalton 1822.888486209 electron
// masses, the hartree 219474.6313632 cm^-1), and its displacement along the
// bond, the atoms moving opposite ways.
TEST(HarmonicAnalysisSpring, DiatomicHasOneMode) {
  FBF_REQUIRE_GPU();
  constexpr double kSpring = 0.4;
  const std::array<double, 3> u{2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0};
  constexpr double kBond = 1.4;
  const std::vector<fb::Atom> atoms{
      fb::Atom{fb::Vec<double, 3>{0.1, -0.2, 0.3}, fb::Element::H},
      fb::Atom{fb::Vec<double, 3>{0.1 + kBond * u[0], -0.2 + kBond * u[1], 0.3 + kBond * u[2]},
               fb::Element::H}};
  std::vector<double> hessian(36);
  for (std::size_t a = 0; a < 2; ++a) {
    for (std::size_t b = 0; b < 2; ++b) {
      for (std::size_t k = 0; k < 3; ++k) {
        for (std::size_t l = 0; l < 3; ++l) {
          hessian[(3 * a + k) * 6 + 3 * b + l] = (a == b ? 1.0 : -1.0) * kSpring * u[k] * u[l];
        }
      }
    }
  }
  fb::DeviceResources res{0};
  const auto vib = geo::harmonic_analysis(res, atoms, hessian);
  ASSERT_TRUE(vib) << vib.error().message;
  ASSERT_EQ(vib->frequencies.size(), 1U);
  const double mu = 1.008 / 2.0 * 1822.888486209;
  EXPECT_NEAR(vib->frequencies[0], std::sqrt(kSpring / mu) * 219474.6313632, kFrequencyTol);
  // norm_mode: the unit mass-weighted vector over sqrt(m), so each atom moves
  // 1 / sqrt(2 m_H) along -/+u (up to the overall sign).
  const double sign = vib->modes[0] * u[0] > 0.0 ? 1.0 : -1.0;
  const double length = 1.0 / std::sqrt(2.0 * 1.008);
  for (std::size_t k = 0; k < 3; ++k) {
    EXPECT_NEAR(vib->modes[k], sign * length * u[k], 1e-12);
    EXPECT_NEAR(vib->modes[3 + k], -sign * length * u[k], 1e-12);
  }
}

// One atom: three translations removed, nothing left.
TEST(HarmonicAnalysisSpring, AtomHasNoModes) {
  FBF_REQUIRE_GPU();
  const std::vector<fb::Atom> atoms{fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, fb::Element::He}};
  const std::vector<double> hessian(9, 0.0);
  fb::DeviceResources res{0};
  const auto vib = geo::harmonic_analysis(res, atoms, hessian);
  ASSERT_TRUE(vib) << vib.error().message;
  EXPECT_TRUE(vib->frequencies.empty());
  EXPECT_TRUE(vib->modes.empty());
}

TEST(HarmonicAnalysisRefusal, RefusesBadInput) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const std::vector<fb::Atom> water{fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, fb::Element::O},
                                    fb::Atom{fb::Vec<double, 3>{0.0, 1.43, -1.1}, fb::Element::H},
                                    fb::Atom{fb::Vec<double, 3>{0.0, -1.43, -1.1}, fb::Element::H}};
  const std::vector<double> hessian(81, 0.0);
  const auto refused = [](const fb::Result<geo::Vibrations> &result) {
    return !result && result.error().kind == fb::ErrorKind::InvalidConfig;
  };

  EXPECT_TRUE(refused(geo::harmonic_analysis(res, {}, {})));
  EXPECT_TRUE(refused(geo::harmonic_analysis(res, water, std::vector<double>(80, 0.0))));
  auto unset = water;
  unset[1].element = fb::Element::Sentinel;
  EXPECT_TRUE(refused(geo::harmonic_analysis(res, unset, hessian)));
  auto nan_position = water;
  nan_position[2].position.y = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(refused(geo::harmonic_analysis(res, nan_position, hessian)));
  auto nan_hessian = hessian;
  nan_hessian[40] = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(refused(geo::harmonic_analysis(res, water, nan_hessian)));
}

} // namespace
