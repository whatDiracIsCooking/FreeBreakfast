// GPU cases for the Irc step (#1269) through the real registry. First an
// independent reference: HCN <-> HNC at RHF/STO-3G over integral-direct
// ERIs, started from data/irc/hcn_rhf.json's TS and its Hessian loaded by
// `[hessian] from` (the file is a [hessian]'s), against that file's GS2 path,
// whose every hypersphere PySCF's exact energy and SciPy's BFGS minimized
// (tools/emit/irc_reference.py): the points at matched arc length agree in
// energy and geometry within bars derived from the step and the two sphere
// thresholds, and both ends reach the RHF minima, HCN and HNC. Then a
// correlated chain: NH3's inversion at DF-CASSCF(8,7) from its planar TS
// (data/irc/nh3_casscf.json), the Hessian computed in the same run: the
// energy falls at every point, each side leaves along the umbrella mode (and
// the two along its opposite phases), and each end is a minimum, every
// frequency of an analytic Hessian there real, the two ends' energies (mirror
// images) within a step's quadratic rise of each other. Last the refusals of a start that is
// not one saddle, and HessianStep::load's of a file that is not the
// molecule's, before anything runs.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.xyz;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

const fb::json &reference(std::string_view file) {
  static std::map<std::string, fb::json, std::less<>> docs;
  auto it = docs.find(file);
  if (it == docs.end()) {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "irc" / file);
    EXPECT_TRUE(read.has_value()) << read.error().message;
    it = docs.emplace(std::string(file), read.has_value() ? std::move(*read) : fb::json{}).first;
  }
  return it->second;
}

// The reference's TS as ts.xyz in @p dir (Angstrom).
void write_ts(const fb::test::ScratchDir &dir, const fb::json &ref) {
  std::string xyz = std::format("{}\nts\n", ref.at("atoms").size());
  for (const fb::json &atom : ref.at("atoms")) {
    const auto c = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       c[0] * fb::kAngstromPerBohr<double>, c[1] * fb::kAngstromPerBohr<double>,
                       c[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "ts.xyz", xyz).has_value());
}

// The isotope-averaged masses of H, C, N (amu), harmonic_analysis's.
double mass_of(std::string_view symbol) {
  if (symbol == "H") {
    return 1.008;
  }
  if (symbol == "C") {
    return 12.011;
  }
  if (symbol == "N") {
    return 14.007;
  }
  ADD_FAILURE() << "no mass for " << symbol;
  return 1.0;
}

using Points = std::vector<std::array<double, 3>>;

Points to_points(std::span<const double> x) {
  Points out;
  for (std::size_t a = 0; a + 2 < x.size(); a += 3) {
    out.push_back({x[a], x[a + 1], x[a + 2]});
  }
  return out;
}

// The largest distance (bohr) between the points of @p a and @p b, each
// centred on its centroid, after @p b is rotated onto @p a by Horn's
// quaternion (pipeline_optimize_gpu.cpp's).
double aligned_max_deviation(Points a, Points b) {
  for (Points *p : {&a, &b}) {
    std::array<double, 3> c{};
    for (const auto &r : *p) {
      for (int32_t k = 0; k < 3; ++k) {
        c[k] += r[k] / static_cast<double>(p->size());
      }
    }
    for (auto &r : *p) {
      for (int32_t k = 0; k < 3; ++k) {
        r[k] -= c[k];
      }
    }
  }
  std::array<std::array<double, 3>, 3> s{};
  for (std::size_t n = 0; n < a.size(); ++n) {
    for (int32_t i = 0; i < 3; ++i) {
      for (int32_t j = 0; j < 3; ++j) {
        s[i][j] += b[n][i] * a[n][j];
      }
    }
  }
  const auto [xx, xy, xz] = s[0];
  const auto [yx, yy, yz] = s[1];
  const auto [zx, zy, zz] = s[2];
  const std::array<std::array<double, 4>, 4> m{{
      {xx + yy + zz, yz - zy, zx - xz, xy - yx},
      {yz - zy, xx - yy - zz, xy + yx, zx + xz},
      {zx - xz, xy + yx, -xx + yy - zz, yz + zy},
      {xy - yx, zx + xz, yz + zy, -xx - yy + zz},
  }};
  double shift = 0.0;
  for (const auto &row : m) {
    for (const double v : row) {
      shift += std::abs(v);
    }
  }
  std::array<double, 4> q{1.0, 0.0, 0.0, 0.0};
  for (int32_t iteration = 0; iteration < 10000; ++iteration) {
    std::array<double, 4> next{};
    for (int32_t i = 0; i < 4; ++i) {
      next[i] = shift * q[i];
      for (int32_t j = 0; j < 4; ++j) {
        next[i] += m[i][j] * q[j];
      }
    }
    const double norm = std::sqrt((next[0] * next[0]) + (next[1] * next[1]) + (next[2] * next[2]) +
                                  (next[3] * next[3]));
    for (int32_t i = 0; i < 4; ++i) {
      q[i] = next[i] / norm;
    }
  }
  const auto [q0, q1, q2, q3] = q;
  const std::array<std::array<double, 3>, 3> r{{
      {(q0 * q0) + (q1 * q1) - (q2 * q2) - (q3 * q3), 2 * ((q1 * q2) - (q0 * q3)),
       2 * ((q1 * q3) + (q0 * q2))},
      {2 * ((q1 * q2) + (q0 * q3)), (q0 * q0) - (q1 * q1) + (q2 * q2) - (q3 * q3),
       2 * ((q2 * q3) - (q0 * q1))},
      {2 * ((q1 * q3) - (q0 * q2)), 2 * ((q2 * q3) + (q0 * q1)),
       (q0 * q0) - (q1 * q1) - (q2 * q2) + (q3 * q3)},
  }};
  double worst = 0.0;
  for (std::size_t n = 0; n < a.size(); ++n) {
    double d2 = 0.0;
    for (int32_t i = 0; i < 3; ++i) {
      double rotated = 0.0;
      for (int32_t j = 0; j < 3; ++j) {
        rotated += r[i][j] * b[n][j];
      }
      d2 += (rotated - a[n][i]) * (rotated - a[n][i]);
    }
    worst = std::max(worst, std::sqrt(d2));
  }
  return worst;
}

// The mass-weighted distance between @p a and @p b (amu^1/2 bohr), no
// alignment: both walks keep the centre of mass and take no rotation.
double mass_weighted_distance(std::span<const double> a, std::span<const double> b,
                              std::span<const double> masses) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += masses[i / 3] * (a[i] - b[i]) * (a[i] - b[i]);
  }
  return std::sqrt(sum);
}

// The SCF converged as pipeline_gradient_gpu.cpp converges it: the gradient
// is the energy's only at convergence. Integral-direct incremental Fock
// builds stall short of the commutator, so every build is full there.
constexpr std::string_view kScf = "[scf]\nenergy_tolerance = 1e-12\ncommutator_tolerance = 1e-10\n";

// The tree's sphere thresholds in the reference case: tight, so the bar is
// set by analysis rather than by the method's defaults.
constexpr double kSphereGradient = 1e-6;
constexpr double kSphereDisplacement = 1e-5;
// The two RHF implementations' gradients at one geometry, each converged to a
// commutator of 1e-10 (PySCF's conv_tol_grad 1e-10): a tangential error of
// their own, added to the two thresholds.
constexpr double kGradientAgreement = 1e-8;
// The two RHF energies at one geometry (both SCFs to 1e-12 and beyond).
constexpr double kEnergyAgreement = 1e-9;

// HCN <-> HNC at RHF/STO-3G, both sides at the reference's step, against
// PySCF's GS2 path at matched arc length. The bars, from the analysis of a
// point's error: a point whose tangential gradient is t lies within t / kappa
// of the sphere's minimum, kappa the curvature of E on the sphere there (the
// least eigenvalue of H - lambda I over its tangent space, the reference's
// per point), so the two walks' point j differ by at most
// (T + T_ref + G) / kappa_j, T this run's sphere_gradient, T_ref the
// reference's, G the two codes' gradient agreement; and a point's error
// moves the next pivot by as much, so the errors add along the path:
// Delta_k = sum_{j <= k} (T + T_ref + G) / kappa_j (mass-weighted), a
// distance of Delta_k / sqrt(m_H) bohr at most for any atom. The energies
// differ by |g_k| Delta_k (|g_k| = |lambda_k| s / 2) plus the two codes'
// agreement. The ends: each side stops at the minimum, by the model within
// half a step or at a point past it, so its last point is within a step s
// of the RHF minimum (mass-weighted) and its energy within s^2 h_max / 2
// above it, h_max that minimum's largest mass-weighted curvature.
TEST(PipelineIrc, HcnRhfMatchesAnIndependentGs2AndReachesBothMinima) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &ref = reference("hcn_rhf.json");
  const double step = ref.at("step_size").get<double>();
  const double t_ref = ref.at("reference_tangential").get<double>();
  std::vector<double> masses;
  for (const fb::json &atom : ref.at("atoms")) {
    masses.push_back(mass_of(atom.at("symbol").get<std::string>()));
  }
  const double sqrt_m_min = std::sqrt(std::ranges::min(masses));

  const fb::test::ScratchDir dir("pipeline_irc_hcn");
  write_ts(dir, ref);
  const std::string toml = std::format(
      "[molecule]\nxyz = \"ts.xyz\"\n[basis_set]\nbasis = \"sto-3g\"\n"
      "[eri]\nmode = \"integral_direct\"\n{}fock_rebuild_interval = 1\n"
      "[hessian]\nfrom = \"{}\"\n"
      "[irc]\nstep_size = {}\nmax_points = 60\nmax_force = 1e-6\nsphere_gradient = {}\n"
      "sphere_displacement = {}\n",
      kScf, (std::filesystem::path{FBF_TEST_DATA_DIR} / "irc" / "hcn_rhf.json").string(), step,
      kSphereGradient, kSphereDisplacement);
  ASSERT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &out = calc->get<fb::IrcOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;

  double max_curvature = 0.0;
  for (const std::string_view side : {"forward", "backward"}) {
    max_curvature =
        std::max(max_curvature, ref.at("minima").at(side).at("max_curvature").get<double>());
  }
  for (const auto &[side, path] :
       {std::pair{"forward", &out->forward}, std::pair{"backward", &out->backward}}) {
    ASSERT_TRUE(path->has_value()) << side;
    const auto &points = (*path)->points;
    const fb::json &expected = ref.at(side).at("points");
    // Either stop by the minimum: within the next step by the model, or a
    // point past it (the reference's own last stop is the second).
    EXPECT_TRUE((*path)->stop == fb::geometry::IrcStop::MinimumWithinStep ||
                (*path)->stop == fb::geometry::IrcStop::EnergyRose)
        << side << " " << fb::geometry::irc_stop_name((*path)->stop);
    // The two stop rules read the same quantity off two Hessians (the tree's
    // updated model, the reference's exact one), so they may part by a point.
    EXPECT_LE(std::abs(static_cast<double>(points.size()) - static_cast<double>(expected.size())),
              1.0)
        << side;
    const std::size_t common = std::min(points.size(), expected.size());
    double delta = 0.0;
    double worst_energy = 0.0;
    double worst_energy_ratio = 0.0;
    double worst_geometry = 0.0;
    double worst_geometry_ratio = 0.0;
    for (std::size_t k = 1; k < common; ++k) {
      const fb::json &e = expected[k];
      delta += (kSphereGradient + t_ref + kGradientAgreement) / e.at("kappa").get<double>();
      const double g_norm = std::abs(e.at("lam").get<double>()) * 0.5 * step;
      const double energy_bar = (g_norm * delta) + kEnergyAgreement;
      const double geometry_bar = delta / sqrt_m_min;
      const double energy_error = std::abs(points[k].energy - e.at("energy").get<double>());
      const double geometry_error = aligned_max_deviation(
          to_points(e.at("coords").get<std::vector<double>>()), to_points(points[k].x));
      EXPECT_LE(energy_error, energy_bar) << side << " point " << k;
      EXPECT_LE(geometry_error, geometry_bar) << side << " point " << k;
      EXPECT_NEAR(std::abs(points[k].arc_length), std::abs(e.at("arc_length").get<double>()),
                  1e-12);
      worst_energy = std::max(worst_energy, energy_error);
      worst_energy_ratio = std::max(worst_energy_ratio, energy_error / energy_bar);
      worst_geometry = std::max(worst_geometry, geometry_error);
      worst_geometry_ratio = std::max(worst_geometry_ratio, geometry_error / geometry_bar);
    }
    // The end: within a step of the minimum, and within its quadratic rise.
    const fb::json &minimum = ref.at("minima").at(side);
    const auto min_x = minimum.at("coords").get<std::vector<double>>();
    const double distance = mass_weighted_distance(points.back().x, min_x, masses);
    const double rise = points.back().energy - minimum.at("energy").get<double>();
    EXPECT_LE(distance, step) << side;
    EXPECT_GE(rise, -kEnergyAgreement) << side;
    EXPECT_LE(rise, 0.5 * max_curvature * step * step + kEnergyAgreement) << side;
    int32_t evaluations = 0;
    for (const auto &point : points) {
      evaluations += point.sphere_iterations;
    }
    std::cout << std::format(
        "[ irc ] hcn RHF {}: {} points ({} reference), {} evaluations, worst |dE| {:.2e} Eh "
        "({:.2f} of its bar), worst geometry {:.2e} bohr ({:.2f} of its bar), last bar Delta "
        "{:.2e}; end {:.4f} amu^1/2 bohr from the minimum (bar {}), {:.2e} Eh above it (bar "
        "{:.2e})\n",
        side, points.size() - 1, expected.size() - 1, evaluations, worst_energy, worst_energy_ratio,
        worst_geometry, worst_geometry_ratio, delta, distance, step, rise,
        0.5 * max_curvature * step * step);
  }
  // The files: a frame per point per side, and the history with both sides.
  const auto history = fb::read_json(dir / "out" / "irc.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  EXPECT_EQ(history->at("forward").at("points").size(), out->forward->points.size());
  EXPECT_EQ(history->at("backward").at("points").size(), out->backward->points.size());
  EXPECT_EQ(history->at("forward").at("stop").get<std::string>(),
            fb::geometry::irc_stop_name(out->forward->stop));
  EXPECT_LT(history->at("backward").at("points").back().at("arc_length").get<double>(), 0.0);
  const auto frames = fb::read_file(dir / "out" / "irc_forward.xyz");
  ASSERT_TRUE(frames.has_value()) << frames.error().message;
  EXPECT_EQ(std::ranges::count(*frames, '\n'), (2 + 3) * out->forward->points.size());
  std::cout << std::format("[ irc ] hcn RHF: {:.1f} s\n", seconds);
}

// The correlated chain: NH3's umbrella inversion at DF-CASSCF(8,7)/STO-3G,
// the full valence space (a frontier CAS(2,2) has a second, lower solution
// off the plane, which the carried orbitals fell into mid-path), from the
// planar TS (PySCF's, data/irc/nh3_casscf.json), its analytic
// DF-CASSCF Hessian computed in the run, one imaginary frequency as PySCF's
// within PySCF's own error: its Hessian is a central difference at h = 1e-3
// bohr of gradients converged to an orbital gradient of 1e-6, so each
// element is good to about 1e-6 / h, the mass-weighted eigenvalue lambda to
// that over m_H, and the frequency to |omega| (1e-6 / h) / (2 m_H lambda). Each side's energy
// falls at every point; each leaves along the imaginary mode, its first step
// more along the mode than across it (|cos| over 1/sqrt 2), the two along
// its opposite phases; and each end is a minimum, every frequency of a fresh
// analytic Hessian there real. The two ends are mirror images of one
// minimum, each within a step of it, so their energies agree to s^2 h_max / 2
// (h_max the ends' largest mass-weighted curvature, from their highest
// frequency).
TEST(PipelineIrc, Nh3CasscfDescendsAlongTheUmbrellaToMinima) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &ref = reference("nh3_casscf.json");
  constexpr double kStep = 0.1;
  // CODATA 2018, as harmonic_analysis's: a frequency (cm^-1) to its
  // mass-weighted curvature (hartree / (amu bohr^2)).
  const auto curvature = [](double wavenumber) {
    const double omega = wavenumber / 219474.6313632;
    return omega * omega * 1822.888486209;
  };
  const fb::test::ScratchDir dir("pipeline_irc_nh3");
  write_ts(dir, ref);
  const std::string chain =
      std::format("[basis_set]\nbasis = \"sto-3g\"\n[eri]\nmode = \"density_fitting\"\naux_basis = "
                  "\"cc-pvdz-jkfit\"\n{}[casscf]\nncas = 7\nnelecas = 8\nconv_tol_grad = 1e-9\n"
                  "[casscf_gradient]\n[hessian]\n",
                  kScf);
  ASSERT_TRUE(fb::write_file(dir / "run.toml",
                             std::format("[molecule]\nxyz = \"ts.xyz\"\n{}[irc]\nstep_size = {}\n",
                                         chain, kStep))
                  .has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const auto t0 = std::chrono::steady_clock::now();
  const fb::Status ran = calc->run(node);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &hessian = calc->get<fb::HessianOutput>();
  ASSERT_TRUE(hessian.has_value());
  const auto &frequencies = hessian->vibrations.frequencies;
  ASSERT_EQ(std::ranges::count_if(frequencies, [](double f) { return f < 0.0; }), 1);
  const double pyscf_imaginary = ref.at("frequencies").at(0).get<double>();
  const double frequency_bar =
      std::abs(pyscf_imaginary) * (1e-6 / 1e-3) / (2.0 * 1.008 * curvature(pyscf_imaginary));
  EXPECT_NEAR(frequencies.front(), pyscf_imaginary, frequency_bar);
  const auto &out = calc->get<fb::IrcOutput>();
  ASSERT_TRUE(out.has_value()) << out.error().message;

  std::vector<double> masses;
  for (const fb::json &atom : ref.at("atoms")) {
    masses.push_back(atom.at("symbol").get<std::string>() == "N" ? 14.007 : 1.008);
  }
  const std::size_t n = 3 * masses.size();
  // The mode, mass-weighted and unit.
  std::vector<double> mode(n);
  double mode_norm = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    mode[i] = hessian->vibrations.modes[i] * std::sqrt(masses[i / 3]);
    mode_norm += mode[i] * mode[i];
  }
  std::array<double, 2> first_cos{};
  std::array<double, 2> end_energy{};
  double h_max = 0.0;
  std::size_t s_index = 0;
  for (const auto &[side, path] :
       {std::pair{"forward", &out->forward}, std::pair{"backward", &out->backward}}) {
    ASSERT_TRUE(path->has_value()) << side;
    const auto &points = (*path)->points;
    ASSERT_GE(points.size(), 3U) << side;
    EXPECT_TRUE((*path)->stop == fb::geometry::IrcStop::MinimumWithinStep ||
                (*path)->stop == fb::geometry::IrcStop::Minimum)
        << side << " " << fb::geometry::irc_stop_name((*path)->stop);
    for (std::size_t k = 0; k + 1 < points.size(); ++k) {
      EXPECT_LT(points[k + 1].energy, points[k].energy) << side << " point " << k + 1;
    }
    double along = 0.0;
    double length = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double dq = std::sqrt(masses[i / 3]) * (points[1].x[i] - points[0].x[i]);
      along += dq * mode[i];
      length += dq * dq;
    }
    end_energy[s_index] = points.back().energy;
    first_cos[s_index++] = along / std::sqrt(length * mode_norm);

    // A fresh Hessian at the end: every frequency real.
    const fb::test::ScratchDir end_dir(std::format("pipeline_irc_nh3_{}_end", side));
    const fb::Molecule &ts = calc->get<fb::MoleculeOutput>()->molecule;
    std::string xyz = std::format("{}\nend\n", ts.natoms());
    for (std::size_t a = 0; a < ts.natoms(); ++a) {
      xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", fb::element_symbol(ts.atoms()[a].element),
                         points.back().x[3 * a] * fb::kAngstromPerBohr<double>,
                         points.back().x[3 * a + 1] * fb::kAngstromPerBohr<double>,
                         points.back().x[3 * a + 2] * fb::kAngstromPerBohr<double>);
    }
    ASSERT_TRUE(fb::write_file(end_dir / "end.xyz", xyz).has_value());
    ASSERT_TRUE(fb::write_file(end_dir / "run.toml",
                               std::format("[molecule]\nxyz = \"end.xyz\"\n{}", chain))
                    .has_value());
    auto end_calc = fb::read_config(end_dir / "run.toml", end_dir.path(), end_dir / "out");
    ASSERT_TRUE(end_calc.has_value()) << end_calc.error().message;
    const fb::Status end_ran = end_calc->run(node);
    ASSERT_TRUE(end_ran.has_value()) << end_ran.error().message;
    const auto &end_hessian = end_calc->get<fb::HessianOutput>();
    ASSERT_TRUE(end_hessian.has_value());
    const double lowest = end_hessian->vibrations.frequencies.front();
    EXPECT_GT(lowest, 0.0) << side;
    h_max = std::max(h_max, curvature(end_hessian->vibrations.frequencies.back()));
    std::cout << std::format(
        "[ irc ] nh3 CASSCF(8,7) {}: {} points ({}), E {:.10f} -> {:.10f}, first step's cosine "
        "with the mode {:+.6f}, frequencies at the end {:.2f} to {:.2f} cm^-1\n",
        side, points.size() - 1, fb::geometry::irc_stop_name((*path)->stop), points.front().energy,
        points.back().energy, first_cos[s_index - 1], lowest,
        end_hessian->vibrations.frequencies.back());
  }
  EXPECT_GT(first_cos[0], 1.0 / std::sqrt(2.0));
  EXPECT_LT(first_cos[1], -1.0 / std::sqrt(2.0));
  const double end_bar = 0.5 * h_max * kStep * kStep;
  EXPECT_LE(std::abs(end_energy[0] - end_energy[1]), end_bar);
  std::cout << std::format("[ irc ] nh3 CASSCF(8,7): the ends' energies differ by {:.2e} Eh "
                           "(bar {:.2e})\n",
                           std::abs(end_energy[0] - end_energy[1]), end_bar);
  std::cout << std::format("[ irc ] nh3 CASSCF(8,7): imaginary {:.2f} cm^-1 (PySCF {:.2f}, bar "
                           "{:.2f}); {:.1f} s\n",
                           frequencies.front(), pyscf_imaginary, frequency_bar, seconds);
}

// A start that is not one saddle, refused by the step before any
// evaluation, each naming its key; and HessianStep::load's refusal of a file
// that is not the molecule's, and its read of one that is.
TEST(PipelineIrc, RefusesAStartThatIsNotOneSaddle) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
  const fb::test::ScratchDir dir("pipeline_irc_refusals");
  fb::DeviceResources res{0, fb::test::kWorkspaces};
  auto molecule = fb::parse_xyz(res, "3\nhcn\nC 0 0 0\nN 1.2 0 0\nH 0.5 1.1 0\n", 0, 1);
  ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
  const fb::MoleculeOutput ts{std::move(*molecule)};
  const auto with = [](std::vector<double> frequencies) {
    fb::HessianOutput h;
    h.hessian.assign(81, 0.0);
    h.vibrations.modes.assign(frequencies.size() * 9, 0.0);
    h.vibrations.modes[2] = 1.0;
    h.vibrations.frequencies = std::move(frequencies);
    return h;
  };
  fb::IrcStep step;
  step.history = dir / "irc.json";
  fb::NodeResources node{{.workspaces = fb::test::kWorkspaces}};
  const auto expect_refused = [&](const fb::IrcStep &s, const fb::HessianOutput &h,
                                  fb::ErrorKind kind, std::string_view needle) {
    const auto out = s.run(fb::View<fb::MoleculeOutput, fb::HessianOutput>{ts, h}, node);
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().kind, kind) << out.error().message;
    EXPECT_NE(out.error().message.find(needle), std::string::npos) << out.error().message;
    EXPECT_FALSE(std::filesystem::exists(s.history));
  };
  expect_refused(step, with({100.0, 2000.0, 3000.0}), fb::ErrorKind::InvalidConfig,
                 "[irc] starts from a transition state, and the [hessian] has no imaginary "
                 "frequency (the lowest is 100.00 cm^-1)");
  expect_refused(step, with({-500.0, -20.0, 3000.0}), fb::ErrorKind::InvalidConfig,
                 "the [hessian] has 2 (500.00i, 20.00i cm^-1): set [irc] "
                 "allow_multiple_imaginary = true");
  fb::HessianOutput small = with({-500.0, 2000.0, 3000.0});
  small.hessian.resize(36);
  expect_refused(step, small, fb::ErrorKind::InvalidConfig,
                 "[irc] starts from the [hessian] of the molecule's 3 atoms, 9 x 9, and it holds "
                 "36 elements");
  // Allowed, the chain's own refusal still comes first: a CASSCF gradient
  // over integral-direct ERIs.
  fb::IrcStep allowed = step;
  allowed.allow_multiple_imaginary = true;
  allowed.chain.casscf_gradient.emplace();
  expect_refused(allowed, with({-500.0, -20.0, 3000.0}), fb::ErrorKind::Unsupported,
                 "[irc] reruns [casscf_gradient], which runs over density fitting only");

  // HessianStep::load: the molecule's 9 x 9 and its modes, as written.
  const fb::View<fb::MoleculeOutput> view{ts};
  fb::json doc = {{"hessian", fb::json::array()},
                  {"frequencies", {-500.0, 2000.0, 3000.0}},
                  {"modes", fb::json::array()}};
  for (int32_t i = 0; i < 9; ++i) {
    fb::json row = fb::json::array();
    for (int32_t j = 0; j < 9; ++j) {
      row.push_back(i == j ? 1.0 + i : 0.0);
    }
    doc["hessian"].push_back(row);
  }
  for (int32_t m = 0; m < 3; ++m) {
    fb::json row = fb::json::array();
    for (int32_t j = 0; j < 9; ++j) {
      row.push_back(j == m ? -1.0 : 0.0);
    }
    doc["modes"].push_back(row);
  }
  ASSERT_TRUE(fb::write_json(dir / "good.json", doc).has_value());
  const auto loaded = fb::HessianStep::load(dir / "good.json", view);
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  EXPECT_EQ(loaded->hessian.size(), 81U);
  EXPECT_EQ(loaded->hessian[80], 9.0);
  EXPECT_EQ(loaded->vibrations.frequencies, (std::vector<double>{-500.0, 2000.0, 3000.0}));
  EXPECT_EQ(loaded->vibrations.modes[0], -1.0);
  EXPECT_TRUE(loaded->dipole_derivative.empty());
  const auto refused = [&](const fb::json &bad, std::string_view needle) {
    ASSERT_TRUE(fb::write_json(dir / "bad.json", bad).has_value());
    const auto out = fb::HessianStep::load(dir / "bad.json", view);
    ASSERT_FALSE(out.has_value());
    EXPECT_EQ(out.error().kind, fb::ErrorKind::InvalidConfig) << out.error().message;
    EXPECT_NE(out.error().message.find(needle), std::string::npos) << out.error().message;
  };
  fb::json wrong = doc;
  wrong["hessian"].erase(8);
  refused(wrong, "'hessian' is not 9 rows of 9 finite numbers");
  wrong = doc;
  wrong.erase("frequencies");
  refused(wrong, "has no 'frequencies'");
  wrong = doc;
  wrong["frequencies"] = {2000.0, -500.0, 3000.0};
  refused(wrong, "'frequencies' is not an ascending array");
  wrong = doc;
  wrong["modes"].erase(2);
  refused(wrong, "'modes' is not one row of 9 finite numbers per frequency (3)");
}

} // namespace
