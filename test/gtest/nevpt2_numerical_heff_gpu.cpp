// GPU cases for nevpt2::numerical_heff (tools/qd_nevpt2_gradient's library,
// #783): the numerical dH_eff/dR of the DF-SA-CASSCF + QD-NEVPT2 chain,
// called in process, against the tool's committed data, SC and PC.
//
// Per case, along the committed file's own directions and steps, with its
// energy noise:
//
// - every g_K = R_K^T (dH_eff) R_K against qd_nevpt2_gradient/'s difference
//   of the eigenvalue E_K, within 2x the committed error bar (finite-
//   difference data): for a nondegenerate state the two are one derivative;
// - for the SA3 case, every h_PQ against qd_nevpt2_nac/coupling/'s `{sc,pc}_h`,
//   the same way, each pair up to the sign s_P s_Q of the eigenvectors, read
//   off the data (as the tool's brute-force cross-check does); over three
//   pairs the signs' product must be +1;
// - every error bar the helper reports at or under kCouplingErrorBarTol;
// - and, as algebra on the helper's own dH_eff (no new points), ḡ and the
//   projector onto span{x, h} of every pair unchanged when u_P and u_Q are
//   rotated within the pair by 0.3 rad: ḡ to 1e-10 of the largest |dH_eff|
//   element, the projector (dimensionless) to 1e-10.
//
// The wall time of each case is printed. The SA3 case is a slow-tier one.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.io.json;
import free_breakfast.tools.qd_nevpt2_gradient;

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

// Against committed finite-difference data: twice its error bar.
constexpr double kFiniteDifferenceFactor = 2.0;
// The pair quantities under a rotation within the pair: path agreement.
constexpr double kRotationTol = 1e-10;
// The rotation: arbitrary, and large enough that x and h visibly move.
constexpr double kAngle = 0.3;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR};

// A committed file's chain, directions and steps.
struct Case {
  nevpt2::NumericalChain chain;
  std::vector<std::vector<double>> directions;
  std::array<double, 3> steps{};
  double noise = 0.0;
  fb::json doc;
};

// nlohmann's .at() throws: the exception becomes a failed assertion.
std::optional<Case> read_case(const std::filesystem::path &path) {
  auto doc = fb::read_json(path);
  EXPECT_TRUE(doc) << path << ": " << (doc ? "" : doc.error().message);
  if (!doc) {
    return std::nullopt;
  }
  try {
    Case out;
    out.doc = std::move(*doc);
    const fb::json &d = out.doc;
    for (const fb::json &atom : d.at("atoms")) {
      const auto coord = atom.at("coord").get<std::vector<double>>();
      out.chain.atoms.push_back(
          fb::Atom{{coord.at(0), coord.at(1), coord.at(2)},
                   fb::element_from_symbol(atom.at("symbol").get<std::string>())});
    }
    out.chain.basis = d.at("basis").get<std::string>();
    out.chain.aux_basis = d.at("aux_basis").get<std::string>();
    out.chain.ncas = d.at("ncas").get<std::int32_t>();
    out.chain.nelecas = d.at("nelecas").get<std::int32_t>();
    if (!d.at("active").is_null()) {
      out.chain.active = d.at("active").get<std::vector<std::int32_t>>();
    }
    out.chain.weights = d.at("weights").get<std::vector<double>>();
    const auto schemes = d.at("schemes").get<std::vector<std::string>>();
    out.chain.pc = std::ranges::contains(schemes, std::string{"pc"});
    out.directions = d.at("directions").get<std::vector<std::vector<double>>>();
    const auto steps = d.at("steps").get<std::vector<double>>();
    EXPECT_EQ(steps.size(), 3U);
    std::ranges::copy_n(steps.begin(), 3, out.steps.begin());
    out.noise = d.at("energy_noise").get<double>();
    return out;
  } catch (const std::exception &e) {
    ADD_FAILURE() << path << ": " << e.what();
    return std::nullopt;
  }
}

// Our values against a committed family entry, within twice its error bar,
// times @p sign. Returns the worst |ours - committed| / committed bar.
double against(const nevpt2::Differenced &ours, const fb::json &entry, double sign,
               std::string_view what) {
  const auto value = entry.at("richardson").get<std::vector<double>>();
  const auto bar = entry.at("error_bar").get<std::vector<double>>();
  EXPECT_EQ(ours.value.size(), value.size()) << what;
  double worst = 0.0;
  for (std::size_t c = 0; c < std::min(value.size(), ours.value.size()); ++c) {
    const double miss = std::abs(ours.value[c] - (sign * value[c]));
    worst = std::max(worst, miss / bar[c]);
    EXPECT_LE(miss, kFiniteDifferenceFactor * bar[c])
        << what << " component " << c << ": " << ours.value[c] << " against " << sign * value[c];
  }
  return worst;
}

// The sign of sum_c ours_c theirs_c.
double sign_against(const nevpt2::Differenced &ours, const fb::json &entry) {
  const auto value = entry.at("richardson").get<std::vector<double>>();
  double dot = 0.0;
  for (std::size_t c = 0; c < std::min(value.size(), ours.value.size()); ++c) {
    dot += ours.value[c] * value[c];
  }
  return dot < 0.0 ? -1.0 : 1.0;
}

// The projector onto span{x, h} over the components, Gram-Schmidt.
std::vector<double> plane_projector(const std::vector<double> &x, const std::vector<double> &h) {
  const std::size_t m = x.size();
  const auto norm = [](const std::vector<double> &v) {
    return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
  };
  std::vector<double> e1 = x;
  const double n1 = norm(e1);
  for (double &v : e1) {
    v /= n1;
  }
  std::vector<double> e2 = h;
  const double dot = std::inner_product(e2.begin(), e2.end(), e1.begin(), 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    e2[i] -= dot * e1[i];
  }
  const double n2 = norm(e2);
  // x and h must span a plane for the projector to be defined.
  EXPECT_GT(n2, 1e-6 * norm(h));
  for (double &v : e2) {
    v /= n2;
  }
  std::vector<double> p(m * m);
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      p[(i * m) + j] = (e1[i] * e1[j]) + (e2[i] * e2[j]);
    }
  }
  return p;
}

// Every pair's ḡ and plane over R and over R rotated within the pair.
void check_rotation(const nevpt2::HeffDerivative &d, std::string_view scheme) {
  double scale = 0.0;
  for (const nevpt2::Differenced &e : d.elements) {
    for (const double v : e.value) {
      scale = std::max(scale, std::abs(v));
    }
  }
  const double c = std::cos(kAngle);
  const double s = std::sin(kAngle);
  for (std::size_t p = 0; p < d.n; ++p) {
    for (std::size_t q = p + 1; q < d.n; ++q) {
      std::vector<double> rotated = d.vectors;
      for (std::size_t i = 0; i < d.n; ++i) {
        const double up = d.vectors[(p * d.n) + i];
        const double uq = d.vectors[(q * d.n) + i];
        rotated[(p * d.n) + i] = (c * up) + (s * uq);
        rotated[(q * d.n) + i] = (-s * up) + (c * uq);
      }
      const nevpt2::HeffPair before = nevpt2::heff_pair(d, d.vectors, p, q);
      const nevpt2::HeffPair after = nevpt2::heff_pair(d, rotated, p, q);
      // (x / 2, h) turns by twice the angle in its plane, so it moves by
      // about 2 sin(0.3) = 0.59 of its size.
      double mean_miss = 0.0;
      double moved = 0.0;
      double size = 0.0;
      for (std::size_t k = 0; k < before.mean.value.size(); ++k) {
        mean_miss = std::max(mean_miss, std::abs(before.mean.value[k] - after.mean.value[k]));
        moved = std::max({moved, 0.5 * std::abs(before.x.value[k] - after.x.value[k]),
                          std::abs(before.h.value[k] - after.h.value[k])});
        size = std::max({size, 0.5 * std::abs(before.x.value[k]), std::abs(before.h.value[k])});
      }
      EXPECT_LE(mean_miss, kRotationTol * scale) << scheme << " pair " << p << q;
      // Not vacuous: the rotation moves x and h themselves.
      EXPECT_GT(moved, 0.1 * size) << scheme << " pair " << p << q;
      const std::vector<double> p0 = plane_projector(before.x.value, before.h.value);
      const std::vector<double> p1 = plane_projector(after.x.value, after.h.value);
      double projector_miss = 0.0;
      for (std::size_t k = 0; k < p0.size(); ++k) {
        projector_miss = std::max(projector_miss, std::abs(p0[k] - p1[k]));
      }
      EXPECT_LE(projector_miss, kRotationTol) << scheme << " pair " << p << q;
      std::println("{} pair ({}, {}): ḡ moved {:.1e} ({:.1e} of max |dH_eff| {:.2e}), projector "
                   "{:.1e}, (x/2, h) moved {:.2f} of its size",
                   scheme, p, q, mean_miss, mean_miss / scale, scale, projector_miss, moved / size);
    }
  }
}

// Every error bar the helper reports at or under kCouplingErrorBarTol.
void check_bars(const nevpt2::HeffDerivative &d, std::string_view scheme) {
  double worst = 0.0;
  for (const auto *family : {&d.elements, &d.g, &d.h}) {
    for (const nevpt2::Differenced &v : *family) {
      for (const double bar : v.error_bar) {
        worst = std::max(worst, bar);
      }
    }
  }
  EXPECT_LE(worst, nevpt2::kCouplingErrorBarTol) << scheme;
  std::println("{}: largest error bar {:.2e}", scheme, worst);
}

// One case: the helper along the gradient file's directions, g_K against
// it, and, given @p coupling_file, h_PQ against that.
void run_case(std::string_view name, bool coupling_file) {
  const std::optional<Case> grad =
      read_case(kDataDir / "qd_nevpt2_gradient" / std::format("{}.json", name));
  ASSERT_TRUE(grad.has_value());
  fb::NodeResources node;
  const auto started = std::chrono::steady_clock::now();
  const fb::Result<nevpt2::NumericalHeff> heff =
      nevpt2::numerical_heff(node, grad->chain, grad->directions, grad->steps, grad->noise);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  ASSERT_TRUE(heff) << heff.error().message;
  // The displaced |g| is what an h_PQ miss is read against first (#869).
  std::println("{}: {} points, {:.1f} s, reference |g| {:.1e}, displaced |g| at most {:.1e}, "
               "guard subspace {:.1e} CI {:.1e} prediction {:.3f}",
               name, heff->points, seconds, heff->gradient_norm, heff->displaced_gradient_norm,
               heff->subspace, heff->ci, heff->prediction);
  ::testing::Test::RecordProperty("wall_seconds", std::format("{:.1f}", seconds));
  ASSERT_EQ(heff->pc.has_value(), grad->chain.pc);

  std::optional<Case> coupling;
  if (coupling_file) {
    coupling = read_case(kDataDir / "qd_nevpt2_nac" / "coupling" / std::format("{}.json", name));
    ASSERT_TRUE(coupling.has_value());
    // The same directions and steps as the gradient file, or the
    // comparison would be between different derivatives.
    for (std::size_t c = 0; c < grad->directions.size(); ++c) {
      for (std::size_t x = 0; x < grad->directions[c].size(); ++x) {
        ASSERT_NEAR(coupling->directions.at(c).at(x), grad->directions[c][x], 1e-12);
      }
    }
    ASSERT_EQ(coupling->steps, grad->steps);
  }

  for (const std::string scheme : {"sc", "pc"}) {
    if (scheme == "pc" && !heff->pc) {
      continue;
    }
    const nevpt2::HeffDerivative &d = scheme == "sc" ? heff->sc : *heff->pc;
    check_bars(d, scheme);
    double worst_g = 0.0;
    try {
      const fb::json &states = grad->doc.at("directional").at(scheme);
      ASSERT_EQ(states.size(), d.g.size());
      for (std::size_t k = 0; k < d.n; ++k) {
        worst_g = std::max(worst_g,
                           against(d.g[k], states.at(k), 1.0, std::format("{} g_{}", scheme, k)));
      }
      std::println("{} {}: g_K at most {:.2f} of the committed error bar", name, scheme, worst_g);
      if (coupling) {
        const fb::json &pairs = coupling->doc.at("directional").at(std::format("{}_h", scheme));
        ASSERT_EQ(pairs.size(), d.h.size());
        double product = 1.0;
        double worst_h = 0.0;
        for (std::size_t pq = 0; pq < d.h.size(); ++pq) {
          const double sign = sign_against(d.h[pq], pairs.at(pq));
          product *= sign;
          worst_h = std::max(worst_h, against(d.h[pq], pairs.at(pq), sign,
                                              std::format("{} h pair {}", scheme, pq)));
        }
        if (d.h.size() == 3) {
          EXPECT_GT(product, 0.0) << scheme << ": the pairs' signs are no product of states'";
        }
        std::println("{} {}: h_PQ at most {:.2f} of the committed error bar", name, scheme,
                     worst_h);
      }
    } catch (const std::exception &e) {
      FAIL() << name << " " << scheme << ": " << e.what();
    }
    check_rotation(d, scheme);
  }
}

// The cheapest committed case: two roots, H6 in 6-31G, 18 points.
TEST(Nevpt2NumericalHeff, H6Sa2MatchesCommittedGradients) {
  FBF_REQUIRE_GPU();
  run_case("h6_631g_sa2", false);
}

// The 3x3 H_eff and root matching: skew H2CO, three roots, against both the
// gradient file's g_K and the coupling file's h_PQ. 13.6 s on the RX 9060 XT,
// past the fast tier's ~10 s, and not its family's cheapest (H6 is, 4.1 s).
//
// #869: before #867's CI floor, a point's polish could run out at |g| ~1e-11
// (component 0, finest step, + side): PC's H_eff(1, 2) there moved by
// 2.5e-11, x 4 / (3 x 2 x 2.5e-4) is 6.6e-8 in PC h pair 2, over its 4.1e-8.
// Under the floor every point polishes under kPolishTarget (3e-12) and a
// point's PC H_eff varies by at most 2.5e-12 run to run (A100, 30 runs).
TEST(Nevpt2NumericalHeff, H2coSkewSa3MatchesCommittedGradientsAndCouplings) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  run_case("h2co_skew_ccpvdz_sa3", true);
}

} // namespace
