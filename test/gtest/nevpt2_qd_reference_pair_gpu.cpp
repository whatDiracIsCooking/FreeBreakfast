// nevpt2::qd_pc_intersection inside a degenerate CASSCF pair (#902,
// src/nevpt2/intersection/README.md "A MECI inside a degenerate CASSCF
// pair"): ethylene's C3v H3C-CH, DF-SA3-CASSCF(2,2)/aug-cc-pVDZ
// (aug-cc-pVDZ-JKFIT), all-electron QD-PC-NEVPT2 states 0 and 1, at the exact
// C3v symmetrization of Park's aug-cc-pVDZ geometry (JCTC (2020), SI p. S3;
// docs/reproduction/park2019/single_points/geometries/ethylene_adz_c3v.xyz),
// where the CASSCF's 1E pair is degenerate by symmetry.
//
// The symmetrization: a frame with z along methyl C -> lone H and methyl
// H 0 in the xz plane, C3v's six operations there (rotations about z by
// 2 pi k / 3, the xz mirror, each with its permutation of the methyl H's),
// and Park's geometry averaged over them. The average is invariant under
// every operation exactly (to rounding), and the same group average of a
// displacement is its a1 part.
//
// - MeanGradientAtTheExactC3vPoint (the fast tier's): the intersection
//   takes the mean-gradient path (reference_pair_tau set, at most
//   mcscf::kLimitTauGate; no g_P, g_Q or h_PQ; NaN sensitivities), and its
//   g_mean is a1: its non-a1 part, against kSymmetryBar. SC at the same
//   CASSCF is still refused (Numerical).
// - MeanGradientMatchesDifferences (slow): directional differences of the
//   mean energy (E_0 + E_1) / 2 of QD-PC single points, every point
//   re-converged from the base's CASSCF orbitals (#696's displaced points),
//   at the exact point: steps 2e-3, 1e-3 and 5e-4
//   bohr, one Richardson step over the two finest, against #853's bar
//   max(2 x the Richardson bar, 3 sigma_E / h_min), sigma_E the sample sd of
//   five repeats of the base point. Directions: g_mean's own, two random
//   unit ones (std::mt19937_64 seeded 902), and two pure e ones (d1's and a
//   third random one's, their a1 part removed). Along the e ones the forward
//   and backward one-sided slopes too (first-order Richardson, 2 F(h) -
//   F(2h)), because a central difference cancels a |delta| (cone) term of
//   the mean energy, against the same bar with its noise term scaled to the
//   one-sided slope's own noise (kOneSidedNoise), 2.28 times the central
//   one's: the same 3.2 standard deviations. Only at the exact point: every
//   point under the floor is within one step of it (5e-6 A is 9.4e-6 bohr,
//   h_min 5e-4), the steps straddle the symmetric point, and there the
//   existing (g_P + g_Q) / 2 path misses the same bar too (at 5e-5 A,
//   README.md). So off it,
// - MeanGradientIsContinuousAcrossTheFloor (slow): g_mean along d1 through
//   the floor, the new path's points and the existing path's on one
//   quadratic in the distance (the test's comment has the bar).

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io;
import free_breakfast.recipes;
import free_breakfast.integrals;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.casscf;
import free_breakfast.pipeline.nevpt2;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

constexpr std::size_t kAtoms = 6;
constexpr std::size_t kCoords = 3 * kAtoms;
using Coords = std::array<double, kCoords>;

// Park's aug-cc-pVDZ C3v geometry (Angstrom): C (methyl), C, three methyl
// H, the lone H.
constexpr std::array<std::string_view, kAtoms> kSymbols{"C", "C", "H", "H", "H", "H"};
constexpr Coords kPark{-4.0040167204, 1.5215194215,  0.0020537368,  -2.5563810211, 1.4217240080,
                       -0.0203780852, -4.3602053719, 2.4377548365,  -0.5084625690, -4.4012510023,
                       1.5538034648,  1.0361353291,  -4.4882276270, 0.6615424234,  -0.5019648020,
                       -1.4827841789, 1.3483546461,  -0.0374080662};
constexpr std::array<std::size_t, 3> kMethyl{2, 3, 4};

// The single points' chain: tier1_ethylene_c3v_augdz.toml's.
constexpr std::string_view kBasis = "aug-cc-pvdz";
constexpr std::string_view kAuxBasis = "aug-cc-pvdz-jkfit";
constexpr std::int32_t kNcas = 2;
constexpr std::int32_t kNelecas = 2;
constexpr std::int32_t kNroots = 3;
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-13;

// The mean gradient's non-a1 part at the exact point: zero by symmetry, so
// what is left is the solves' own error. Each Z-vector is held to a true
// residual of 1e-10, and two independently converged solves of one
// gradient agree to 1e-9 of its largest summand (nevpt2_qd_intersection_gpu's
// kSolveTolerance, #784's bar amendment); a symmetric part removed from a
// gradient is that difference, so 1e-9 absolute (Eh/bohr; the gradient's
// largest summand is under 1 here).
constexpr double kSymmetryBar = 1e-9;
// The FD ladder (bohr), coarse to fine, #853's.
constexpr std::array<double, 3> kSteps{2e-3, 1e-3, 5e-4};
// Repeats of the base energy for sigma_E.
constexpr std::int32_t kRepeats = 5;
// The one-sided slope's noise over the central one's, so that its bar's
// noise term is the same multiple of its own standard deviation as #853's
// 3 sigma_E / h_min is of the central slope's (3.2: the central Richardson
// slope (4 D(h) - D(2h)) / 3 is (2/3h) (E(h) - E(-h)) - (1/12h) (E(2h) -
// E(-2h)), sd sqrt(2 (2/3)^2 + 2 (1/12)^2) sigma_E / h = 0.95 sigma_E / h).
// The one-sided 2 F(h) - F(2h) is (2/h) E(h) - (1/2h) E(2h) - (3/2h) E(0),
// E(0) the mean of the kRepeats runs: sd sqrt(4 + 1/4 + (9/4) / kRepeats)
// sigma_E / h = 2.17 sigma_E / h, 2.28 times the central's.
const double kOneSidedNoise =
    std::sqrt(4.0 + 0.25 + (2.25 / kRepeats)) / std::sqrt((2.0 * (4.0 / 9.0)) + (2.0 / 144.0));
// The continuity check's points along d1 (Angstrom): two under the CASSCF
// floor (gaps 8.8e-10 and 2.2e-9 Eh, #892's 4.9e-4 Eh/A; the floor is at
// 2.0e-5 A) and three over it (4.4e-8 to 8.8e-8). Each side clean of the
// inactive orbitals' floor too (kZcGapFloor, 1e-6 Eh, nevpt2/gradient's
// zc_multipliers): C3v's e core and virtual pairs split linearly as well,
// at 1.7e-2 to 1.6e-1 Eh/A here, so between 6e-6 and 6e-5 A some of their
// z^c are divided and some dropped, pair by pair, and g_mean . random1
// moves by up to 1.5e-9 Eh/bohr where one crosses (README.md). Under 5e-6
// A every such pair is dropped, over 1e-4 A every one divided.
constexpr std::array<double, 5> kAlong{2e-6, 5e-6, 1e-4, 1.5e-4, 2e-4};
constexpr std::size_t kBelow = 2;
constexpr std::int32_t kRuns = 3;

double dot(std::span<const double> a, std::span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

Coords unit(Coords v) {
  const double n = std::sqrt(dot(v, v));
  for (double &x : v) {
    x /= n;
  }
  return v;
}

// One of C3v's operations in the frame: x -> r x, atom a carried to perm[a].
struct Operation {
  std::array<std::array<double, 3>, 3> r;
  std::array<std::size_t, kAtoms> perm;
};

Coords in_frame(const Coords &angstrom);
Coords park_in_frame() {
  return in_frame(kPark);
}

// The methyl H's in the order of their azimuth in the frame, from methyl H
// 0's (0) up: rotation by 2 pi / 3 carries each to the next.
std::array<std::size_t, 3> methyl_order() {
  const Coords x = park_in_frame();
  std::array<std::size_t, 3> order = kMethyl;
  const auto azimuth = [&](std::size_t a) {
    const double phi = std::atan2(x[(3 * a) + 1], x[3 * a]);
    return phi < -1e-6 ? phi + (2.0 * std::numbers::pi) : phi;
  };
  std::ranges::sort(order, {}, azimuth);
  return order;
}

std::array<Operation, 6> operations() {
  const std::array<std::size_t, 3> methyl = methyl_order();
  std::array<Operation, 6> ops{};
  for (std::size_t k = 0; k < 3; ++k) {
    const double t = 2.0 * std::numbers::pi * static_cast<double>(k) / 3.0;
    const double c = std::cos(t);
    const double s = std::sin(t);
    for (const bool mirror : {false, true}) {
      Operation &op = ops[k + (mirror ? 3 : 0)];
      // R_k M, M the xz mirror (y -> -y).
      const double m = mirror ? -1.0 : 1.0;
      op.r = {{{c, -s * m, 0.0}, {s, c * m, 0.0}, {0.0, 0.0, 1.0}}};
      for (std::size_t a = 0; a < kAtoms; ++a) {
        op.perm[a] = a;
      }
      // Methyl H j (in azimuth order) sits near 2 pi j / 3: M takes it to
      // -j, then R_k to k - j (k + j unmirrored).
      for (std::size_t j = 0; j < 3; ++j) {
        const std::size_t to = mirror ? (k + 3 - j) % 3 : (k + j) % 3;
        op.perm[methyl[j]] = methyl[to];
      }
    }
  }
  return ops;
}

// The group average of @p v, a geometry or a displacement in the frame (the
// operations fix the origin): a geometry's exact C3v symmetrization, a
// displacement's a1 part.
Coords average(const Coords &v) {
  Coords out{};
  for (const Operation &op : operations()) {
    for (std::size_t a = 0; a < kAtoms; ++a) {
      for (std::size_t i = 0; i < 3; ++i) {
        double moved = 0.0;
        for (std::size_t j = 0; j < 3; ++j) {
          moved += op.r[i][j] * v[(3 * a) + j];
        }
        out[(3 * op.perm[a]) + i] += moved / 6.0;
      }
    }
  }
  return out;
}

// Park's geometry in the frame, bohr: methyl C at the origin, z toward the
// lone H, methyl H 0 in the xz plane at positive x.
Coords in_frame(const Coords &angstrom) {
  Coords x{};
  for (std::size_t k = 0; k < kCoords; ++k) {
    x[k] = (angstrom[k] - angstrom[k % 3]) / fb::kAngstromPerBohr<double>;
  }
  const auto axis = [&](std::size_t a) {
    return std::array{x[3 * a], x[(3 * a) + 1], x[(3 * a) + 2]};
  };
  std::array<double, 3> z = axis(5);
  const double zn = std::sqrt((z[0] * z[0]) + (z[1] * z[1]) + (z[2] * z[2]));
  for (double &v : z) {
    v /= zn;
  }
  std::array<double, 3> h = axis(kMethyl[0]);
  const double hz = (h[0] * z[0]) + (h[1] * z[1]) + (h[2] * z[2]);
  for (std::size_t i = 0; i < 3; ++i) {
    h[i] -= hz * z[i];
  }
  const double hn = std::sqrt((h[0] * h[0]) + (h[1] * h[1]) + (h[2] * h[2]));
  for (double &v : h) {
    v /= hn;
  }
  const std::array<double, 3> y{(z[1] * h[2]) - (z[2] * h[1]), (z[2] * h[0]) - (z[0] * h[2]),
                                (z[0] * h[1]) - (z[1] * h[0])};
  Coords out{};
  for (std::size_t a = 0; a < kAtoms; ++a) {
    const std::array<double, 3> p = axis(a);
    for (std::size_t i = 0; i < 3; ++i) {
      const std::array<double, 3> &e = i == 0 ? h : (i == 1 ? y : z);
      out[(3 * a) + i] = (p[0] * e[0]) + (p[1] * e[1]) + (p[2] * e[2]);
    }
  }
  return out;
}

// Where a displaced point starts: the base's CASSCF orbitals (AO by MO,
// column-major) and energies, and the RHF's electron counts.
struct Start {
  std::vector<double> c;
  std::vector<double> mo_energies;
  std::int32_t n_alpha = 0;
  std::int32_t n_beta = 0;
};

// The molecule from an xyz file: MoleculeStep's run without its load, which
// reads an [optimize]'s product and so needs OptimizeStep in the registry.
struct GeometryStep {
  static constexpr std::string_view kName = "molecule";
  std::filesystem::path xyz;

  [[nodiscard]] fb::Result<fb::MoleculeOutput> run(fb::View<> /*in*/,
                                                   fb::DeviceResources &res) const {
    auto molecule = fb::read_xyz(res, xyz, 0, 1);
    if (!molecule) {
      return std::unexpected(molecule.error());
    }
    return fb::MoleculeOutput{std::move(*molecule)};
  }
};

// The RHF's stand-in at a displaced point: @ref Start's orbitals
// Loewdin-orthonormalized in this geometry's overlap, C (C^T S C)^-1/2, as
// pipeline/optimize's carried orbitals and #696's displaced points start
// (the RHF is only the CASSCF's start, and near C3v its HOMO and LUMO are
// the e pair, which it splits: from the SAD guess one point of the ladder,
// random1 at 2e-3 bohr, oscillated past 1000 iterations).
struct StartStep {
  static constexpr std::string_view kName = "scf";
  Start start;

  [[nodiscard]] fb::Result<fb::ScfOutput> run(fb::View<fb::BasisSetOutput, fb::OverlapOutput> in,
                                              fb::DeviceResources &res) const {
    const fb::BasisSet &basis = in.get<fb::BasisSetOutput>().basis;
    const fb::SphericalMatrixAO &s = in.get<fb::OverlapOutput>().s;
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    const auto n = static_cast<std::int32_t>(s.n());
    const std::size_t n2 = s.num_elements();
    fb::SphericalMatrixAO previous(res, basis);
    fb::Buffers::device<double> m(n2, res.device());
    fb::Buffers::device<double> m_inv_sqrt(n2, res.device());
    fb::ScfOutput out{.energy = std::numeric_limits<double>::quiet_NaN(),
                      .iterations = 0,
                      .n_alpha = start.n_alpha,
                      .n_beta = start.n_beta,
                      .mo_energies = start.mo_energies,
                      .c = fb::SphericalMatrixAO(res, basis),
                      .d = fb::SphericalMatrixAO(res, basis)};
    // Each call in order, the first error returned.
    const fb::Status done = [&]() -> fb::Status {
      if (auto st = fb::gpu_status(wwr::wwrMemcpyAsync(previous.data(), start.c.data(),
                                                       n2 * sizeof(double),
                                                       wwr::wwrMemcpyHostToDevice, res.stream()));
          !st) {
        return st;
      }
      {
        const auto [work] = res.device_regions<double>(n2);
        if (auto st = fb::gpu_status(
                fb::gemmwich(res.blas(), n, previous.data(), s.data(), work, m.data()));
            !st) {
          return st;
        }
      }
      if (auto st = fb::orthogonalize<double>(res, n, m.data(), m_inv_sqrt.data()); !st) {
        return st;
      }
      if (auto st = fb::gpu_status(fb::gemm(res.blas(), n, &fb::kOne<double>, previous.data(),
                                            m_inv_sqrt.data(), &fb::kZero<double>, out.c.data()));
          !st) {
        return st;
      }
      return res.sync();
    }();
    if (!done) {
      return std::unexpected(done.error());
    }
    return out;
  }
};

// One QD-PC single point: the mean energy of states 0 and 1, the CASSCF's
// smallest root gap, and, with an intersection asked for,
// qd_pc_intersection (and qd_sc_intersection) of 0 and 1 over its CASSCF.
struct Point {
  double mean = 0.0;
  double root_gap = 0.0;
  std::optional<fb::Result<nevpt2::QdIntersection>> pc;
  std::optional<fb::Result<nevpt2::QdIntersection>> sc;
};

// @p calc's products (a Calculation or a Pipeline) as a Point.
template <class C>
Point read_point(fb::NodeResources &node, C &calc, bool intersect, double zvector_tolerance) {
  Point point;
  const auto &out = *calc.template get<fb::Nevpt2Output>();
  EXPECT_TRUE(out.pc_qd_refusal.empty()) << out.pc_qd_refusal;
  EXPECT_EQ(out.pc_qd_energies.size(), 3U);
  point.mean = 0.5 * (out.pc_qd_energies.at(0) + out.pc_qd_energies.at(1));
  const auto &cas = *calc.template get<fb::CasscfOutput>();
  point.root_gap = std::numeric_limits<double>::infinity();
  for (std::size_t r = 0; r < cas.energies.size(); ++r) {
    for (std::size_t s = r + 1; s < cas.energies.size(); ++s) {
      point.root_gap = std::min(point.root_gap, std::abs(cas.energies[s] - cas.energies[r]));
    }
  }
  if (!intersect) {
    return point;
  }
  fb::DeviceResources &res = node.device(0);
  const fb::BasisSet &basis = calc.template get<fb::BasisSetOutput>()->basis;
  const auto n2 = static_cast<std::size_t>(cas.ncas) * static_cast<std::size_t>(cas.ncas);
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas.weights.size(); ++r) {
    for (std::size_t e = 0; e < n2; ++e) {
      gamma[e] += cas.weights[r] * cas.rdm1[(r * n2) + e];
    }
  }
  fb::mcscf::Solution solution{
      .c = fb::SphericalMatrixAO(res, basis),
      .roots = {.energies = cas.energies,
                .spin_squares = cas.spin_squares,
                .vectors = fb::Buffers::device<double>(cas.vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas.energy,
      .gradient_norm = cas.gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas.rdm2,
      .root_gammas = cas.rdm1,
      .iterations = cas.iterations,
      // Carried, not recovered: at the exact point the three roots' 1-RDMs
      // are linearly dependent, and sa_gradient_weights refuses a recovery.
      .weights = cas.weights};
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.roots.vectors.data(), cas.vectors.data(),
                                cas.vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(solution.c.data(), cas.c.data(),
                                cas.c.num_elements() * sizeof(double), wwr::wwrMemcpyDeviceToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  const auto *const fit = std::get_if<fb::DensityFitted>(&calc.template get<fb::EriOutput>()->eri);
  EXPECT_NE(fit, nullptr);
  if (fit == nullptr) {
    return point;
  }
  const fb::Molecule &molecule = calc.template get<fb::MoleculeOutput>()->molecule;
  const auto &classes = calc.template get<fb::PrimitivePairClassesOutput>()->classes;
  const fb::SphericalMatrixAO &h = calc.template get<fb::CoreHamiltonianOutput>()->h;
  point.pc = nevpt2::qd_pc_intersection(res, molecule, basis, classes, fit->aux, h, fit->b,
                                        fit->l.data(), solution, cas.ncore, cas.ncas, 0, 1, 0,
                                        {.zvector_tolerance = zvector_tolerance});
  point.sc = nevpt2::qd_sc_intersection(res, molecule, basis, classes, fit->aux, h, fit->b,
                                        fit->l.data(), solution, cas.ncore, cas.ncas, 0, 1, 0);
  return point;
}

// @p x (bohr) as an XYZ file (Angstrom).
void write_xyz(const std::filesystem::path &path, const Coords &x) {
  std::string xyz = std::format("{}\nethylene C3v\n", kAtoms);
  for (std::size_t a = 0; a < kAtoms; ++a) {
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", kSymbols[a],
                       x[3 * a] * fb::kAngstromPerBohr<double>,
                       x[(3 * a) + 1] * fb::kAngstromPerBohr<double>,
                       x[(3 * a) + 2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(path, xyz).has_value());
}

// The solve's tolerances: the chain's, or #812's tighter re-solve.
struct Tolerances {
  double conv_tol_grad = kConvTolGrad;
  double zvector = 1e-10;
};

// The single point at @p x (bohr): from the SAD guess's RHF with no
// @p start; from @p start's orbitals otherwise. Its Start into *@p keep
// when asked.
Point point_at(fb::NodeResources &node, const Coords &x, bool intersect,
               const Start *start = nullptr, Start *keep = nullptr,
               const Tolerances &tolerances = {}) {
  const fb::test::ScratchDir dir("nevpt2_qd_reference_pair");
  write_xyz(dir / "molecule.xyz", x);
  if (start != nullptr) {
    fb::BasisSetStep basis{.basis = std::string(kBasis)};
    fb::EriStep eri{.mode = fb::EriMode::DensityFitting, .aux_basis = std::string(kAuxBasis)};
    fb::CasscfStep casscf;
    casscf.ncas = kNcas;
    casscf.nelecas = kNelecas;
    casscf.nroots = kNroots;
    casscf.conv_tol_grad = tolerances.conv_tol_grad;
    casscf.conv_tol = kConvTol;
    fb::Nevpt2Step nevpt2{.quasi_degenerate = true};
    auto chain = fb::Pipeline<GeometryStep, fb::BasisSetStep, fb::SphericalTransformStep,
                              fb::PrimitivePairClassesStep, fb::EriStep, fb::OverlapStep,
                              fb::CoreHamiltonianStep, StartStep, fb::CasscfStep,
                              fb::Nevpt2Step>::create({GeometryStep{dir / "molecule.xyz"}, basis,
                                                       fb::SphericalTransformStep{},
                                                       fb::PrimitivePairClassesStep{}, eri,
                                                       fb::OverlapStep{}, fb::CoreHamiltonianStep{},
                                                       StartStep{*start}, casscf, nevpt2});
    EXPECT_TRUE(chain.has_value()) << chain.error().message;
    if (!chain.has_value()) {
      return {};
    }
    const fb::Status ran = chain->run(node);
    EXPECT_TRUE(ran.has_value()) << ran.error().message;
    if (!ran.has_value()) {
      return {};
    }
    return read_point(node, *chain, intersect, tolerances.zvector);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n[scf]\nmax_iterations = 400\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nconv_tol_grad = {:g}\nconv_tol = {:g}\n"
      "[nevpt2]\nquasi_degenerate = true\n",
      kBasis, kAuxBasis, kNcas, kNelecas, kNroots, tolerances.conv_tol_grad, kConvTol);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return {};
  }
  const fb::Status ran = calc->run(node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return {};
  }
  if (keep != nullptr) {
    const auto &cas = *calc->get<fb::CasscfOutput>();
    const auto &scf = *calc->get<fb::ScfOutput>();
    keep->c.resize(cas.c.num_elements());
    keep->mo_energies = cas.mo_energies;
    keep->n_alpha = scf.n_alpha;
    keep->n_beta = scf.n_beta;
    fb::DeviceResources &res = node.device(0);
    EXPECT_EQ(wwr::wwrMemcpyAsync(keep->c.data(), cas.c.data(), keep->c.size() * sizeof(double),
                                  wwr::wwrMemcpyDeviceToHost, res.stream()),
              wwr::wwrSuccess);
    EXPECT_TRUE(res.sync().has_value());
  }
  return read_point(node, *calc, intersect, tolerances.zvector);
}

// The intersection inside the pair at @p x: the mean-gradient path taken,
// its tau under the gate, nothing else returned; SC refused. Returns g_mean.
std::vector<double> expect_mean_gradient(const Point &point, std::string_view where) {
  EXPECT_LT(point.root_gap, fb::mcscf::kRootGapFloor) << where;
  if (!point.pc.has_value() || !point.sc.has_value()) {
    ADD_FAILURE() << where << ": no intersection";
    return {};
  }
  const auto &pc = *point.pc;
  EXPECT_TRUE(pc.has_value()) << where << ": " << pc.error().message;
  if (!pc.has_value()) {
    return {};
  }
  EXPECT_TRUE(pc->reference_pair_tau.has_value()) << where;
  if (pc->reference_pair_tau) {
    EXPECT_LE(std::abs(*pc->reference_pair_tau), fb::mcscf::kLimitTauGate) << where;
  }
  EXPECT_TRUE(pc->g_p.empty() && pc->g_q.empty() && pc->h_pq.empty()) << where;
  EXPECT_TRUE(std::isnan(pc->gap_sensitivity_orbital) && std::isnan(pc->gap_sensitivity_ci))
      << where;
  EXPECT_EQ(pc->g_mean.size(), kCoords) << where;
  const auto &sc = *point.sc;
  EXPECT_FALSE(sc.has_value()) << where << ": SC was not refused";
  if (!sc.has_value()) {
    // At the exact point SC meets its degenerate e core pair first
    // (zc_multipliers, ../gradient/README.md "kZcGapFloor"); off it,
    // root_pair_multipliers' floor. Either is SC's refusal, kept.
    EXPECT_EQ(sc.error().kind, fb::ErrorKind::Numerical) << sc.error().message;
    std::cout << std::format("[ ref pair ] {}: SC refused: {}\n", where, sc.error().message);
  }
  std::cout << std::format("[ ref pair ] {}: CASSCF root gap {:.3e}, tau {:.3e}, |g_mean| {:.6e}\n",
                           where, point.root_gap, pc->reference_pair_tau.value_or(0.0),
                           std::sqrt(dot(pc->g_mean, pc->g_mean)));
  return pc->g_mean;
}

TEST(Nevpt2QdReferencePair, MeanGradientAtTheExactC3vPoint) {
  FBF_REQUIRE_GPU();
  fb::NodeResources node;
  const Coords exact = average(park_in_frame());
  const Point point = point_at(node, exact, true);
  const std::vector<double> g = expect_mean_gradient(point, "exact C3v");
  ASSERT_EQ(g.size(), kCoords);
  Coords gv{};
  std::ranges::copy(g, gv.begin());
  const Coords a1 = average(gv);
  double off = 0.0;
  for (std::size_t k = 0; k < kCoords; ++k) {
    off = std::max(off, std::abs(gv[k] - a1[k]));
  }
  EXPECT_LE(off, kSymmetryBar);
  std::cout << std::format(
      "[ ref pair ] exact C3v: g_mean's non-a1 part {:.3e} (bar {:.0e}) of "
      "its largest {:.3e}\n",
      off, kSymmetryBar,
      std::abs(*std::ranges::max_element(g, {}, [](double v) { return std::abs(v); })));
}

// d1, Park's offset from the exact point (unit), and the random unit
// directions in their std::mt19937_64{902} order.
struct Lines {
  Coords exact;
  Coords d1;
  Coords random0;
  Coords random1;
  Coords random2;
};

Lines lines() {
  Lines out{};
  const Coords park = park_in_frame();
  out.exact = average(park);
  for (std::size_t k = 0; k < kCoords; ++k) {
    out.d1[k] = park[k] - out.exact[k];
  }
  out.d1 = unit(out.d1);
  std::mt19937_64 engine{902};
  std::normal_distribution<double> normal;
  const auto random_unit = [&] {
    Coords v{};
    for (double &x : v) {
      x = normal(engine);
    }
    return unit(v);
  };
  out.random0 = random_unit();
  out.random1 = random_unit();
  out.random2 = random_unit();
  return out;
}

TEST(Nevpt2QdReferencePair, MeanGradientMatchesDifferences) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  fb::NodeResources node;
  const Lines l = lines();
  const Coords &exact = l.exact;
  const Coords &random0 = l.random0;
  const Coords &random1 = l.random1;
  const auto e_part = [](const Coords &v) {
    const Coords a1 = average(v);
    Coords e{};
    for (std::size_t k = 0; k < kCoords; ++k) {
      e[k] = v[k] - a1[k];
    }
    return unit(e);
  };
  const Coords e0 = e_part(l.d1);
  const Coords e1 = e_part(l.random2);

  const double h_min = kSteps.back();
  for (const auto &[where, base] : {std::pair{std::string_view{"exact C3v"}, exact}}) {
    SCOPED_TRACE(where);
    // Every other point starts from this one's CASSCF orbitals (StartStep).
    Start start;
    const Point center = point_at(node, base, true, nullptr, &start);
    const std::vector<double> g = expect_mean_gradient(center, where);
    ASSERT_EQ(g.size(), kCoords);
    Coords gv{};
    std::ranges::copy(g, gv.begin());
    // sigma_E: the base re-run kRepeats times.
    std::vector<double> repeats{center.mean};
    for (std::int32_t k = 1; k < kRepeats; ++k) {
      repeats.push_back(point_at(node, base, false).mean);
    }
    const double average_e =
        std::accumulate(repeats.begin(), repeats.end(), 0.0) / static_cast<double>(repeats.size());
    double ss = 0.0;
    for (const double e : repeats) {
      ss += (e - average_e) * (e - average_e);
    }
    const double sigma = std::sqrt(ss / static_cast<double>(repeats.size() - 1));
    const double noise_bar = 3.0 * sigma / h_min;
    std::cout << std::format("[ ref pair ] {}: sigma_E {:.2e} over {} runs, 3 sigma / h_min "
                             "{:.2e}\n",
                             where, sigma, repeats.size(), noise_bar);

    const std::array<std::pair<std::string_view, Coords>, 5> directions{
        std::pair{std::string_view{"g_mean"}, unit(gv)},
        std::pair{std::string_view{"random0"}, random0},
        std::pair{std::string_view{"random1"}, random1}, std::pair{std::string_view{"e(d1)"}, e0},
        std::pair{std::string_view{"e(random2)"}, e1}};
    for (std::size_t dir = 0; dir < directions.size(); ++dir) {
      const auto &[name, d] = directions[dir];
      const bool one_sided = dir >= 3;
      // E at +h and -h for each step.
      std::array<std::array<double, 2>, 3> e{};
      for (std::size_t s = 0; s < kSteps.size(); ++s) {
        for (const std::size_t side : {std::size_t{0}, std::size_t{1}}) {
          Coords x = base;
          const double t = side == 0 ? kSteps[s] : -kSteps[s];
          for (std::size_t k = 0; k < kCoords; ++k) {
            x[k] += t * d[k];
          }
          e[s][side] = point_at(node, x, false, &start).mean;
          EXPECT_LT(std::abs(e[s][side] - center.mean), 1e-4) << name << " step " << s;
        }
      }
      const double analytic = dot(g, d);
      const auto bar = [&](double fine, double coarse) {
        return std::max(2.0 * std::abs(fine - coarse), noise_bar);
      };
      const auto central = [&](std::size_t s) { return (e[s][0] - e[s][1]) / (2.0 * kSteps[s]); };
      const double c_fine = ((4.0 * central(2)) - central(1)) / 3.0;
      const double c_coarse = ((4.0 * central(1)) - central(0)) / 3.0;
      const double c_bar = bar(c_fine, c_coarse);
      EXPECT_LE(std::abs(c_fine - analytic), c_bar) << name << " central";
      std::string line = std::format("[ ref pair ] {:<16} {:<10} g.d {:+.6e}; central miss "
                                     "{:+.2e} / bar {:.2e}",
                                     where, name, analytic, c_fine - analytic, c_bar);
      if (one_sided) {
        for (const std::size_t side : {std::size_t{0}, std::size_t{1}}) {
          const double sign = side == 0 ? 1.0 : -1.0;
          const auto slope = [&](std::size_t s) {
            return sign * (e[s][side] - average_e) / kSteps[s];
          };
          const double fine = (2.0 * slope(2)) - slope(1);
          const double coarse = (2.0 * slope(1)) - slope(0);
          const double b = std::max(2.0 * std::abs(fine - coarse), kOneSidedNoise * noise_bar);
          EXPECT_LE(std::abs(fine - analytic), b) << name << (side == 0 ? " forward" : " backward");
          line += std::format("; {} miss {:+.2e} / bar {:.2e}", side == 0 ? "forward" : "backward",
                              fine - analytic, b);
        }
      }
      std::cout << line << "\n";
    }
  }
}

// The mean gradient along the line exact + rho d1 across the CASSCF floor:
// g_mean . u at kAlong's points, the mean-gradient path under the floor and
// the (g_P + g_Q) / 2 path over it, for u g_mean's own direction at the
// exact point (asserted) and random1 (reported). d1 is pure e to 1.4e-4, so
// g_mean's a1 part is even in rho and its e part odd: g_mean . u = a +
// b rho + c rho^2 + O(rho^3). Along g_mean's own direction, an a1 one, b = 0
// and the rho^2 term is the whole change (+6.8e-10 Eh/bohr at 1e-4 A), which
// no line fits. One such quadratic is fitted through both paths' points,
// least squares on the means of kRuns runs, and its residuals are held to
// 3 sigma_g, sigma_g the gradient's noise as #812's ladder measures it
// (pipeline/hessian/README.md "The noise floor"), the larger of two terms:
// - restarts: the pooled sample sd of g_mean . u over kRuns independent
//   runs at each point (the SAD start each), what changes where a solve
//   lands;
// - solves: the root mean square over the points of the first run's
//   g_mean . u less a re-solve's from its orbitals with the Z-vector 100
//   times tighter (1e-12) and the CASSCF's conv_tol_grad 10 times (1e-11:
//   1e-12 is under the CASSCF's own floor here, about 3e-12), what every
//   solve leaves unconverged, which no restart averages away.
// So the bar is not divided by sqrt(kRuns): the solves term is common to a
// point's runs. On the C3v manifold g_mean is a1, and its a1 part is what
// the carried-plane walk reads, so g_mean's direction is the one asserted.
// random1's residuals, an e direction's, are printed only: they carry the
// existing path's per-point scatter, about 2e-10 to 4e-10 Eh/bohr above
// the floor too, from C3v's e core and virtual pairs crossing kZcGapFloor
// one by one (#974; README.md, "The mean gradient across the floor").
TEST(Nevpt2QdReferencePair, MeanGradientIsContinuousAcrossTheFloor) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  fb::NodeResources node;
  const Lines l = lines();
  const Point center = point_at(node, l.exact, true);
  const std::vector<double> g0 = expect_mean_gradient(center, "exact C3v");
  ASSERT_EQ(g0.size(), kCoords);
  Coords g0v{};
  std::ranges::copy(g0, g0v.begin());
  const std::array<std::pair<std::string_view, Coords>, 2> directions{
      std::pair{std::string_view{"g_mean"}, unit(g0v)},
      std::pair{std::string_view{"random1"}, l.random1}};

  // values[d][i][r]: direction d, point i, run r; tight[d][i] the re-solve.
  std::array<std::array<std::array<double, kRuns>, kAlong.size()>, 2> values{};
  std::array<std::array<double, kAlong.size()>, 2> tight{};
  const auto mean_gradient = [&](const Point &p, double at) -> std::optional<Coords> {
    EXPECT_TRUE(p.pc.has_value() && p.pc->has_value())
        << at << " A: " << (p.pc ? p.pc->error().message : "no intersection");
    if (!p.pc.has_value() || !p.pc->has_value() || p.pc->value().g_mean.size() != kCoords) {
      return std::nullopt;
    }
    const nevpt2::QdIntersection &out = p.pc->value();
    const bool below = at < kAlong[kBelow];
    EXPECT_EQ(p.root_gap < fb::mcscf::kRootGapFloor, below) << at << " A";
    EXPECT_EQ(out.reference_pair_tau.has_value(), below) << at << " A";
    Coords g{};
    std::ranges::copy(out.g_mean, g.begin());
    return g;
  };
  for (std::size_t i = 0; i < kAlong.size(); ++i) {
    Coords x = l.exact;
    for (std::size_t k = 0; k < kCoords; ++k) {
      x[k] += kAlong[i] / fb::kAngstromPerBohr<double> * l.d1[k];
    }
    Start start;
    for (std::int32_t r = 0; r < kRuns; ++r) {
      const auto g =
          mean_gradient(point_at(node, x, true, nullptr, r == 0 ? &start : nullptr), kAlong[i]);
      ASSERT_TRUE(g.has_value());
      for (std::size_t d = 0; d < directions.size(); ++d) {
        values[d][i][static_cast<std::size_t>(r)] = dot(*g, directions[d].second);
      }
    }
    const auto g = mean_gradient(
        point_at(node, x, true, &start, nullptr, {.conv_tol_grad = 1e-11, .zvector = 1e-12}),
        kAlong[i]);
    ASSERT_TRUE(g.has_value());
    for (std::size_t d = 0; d < directions.size(); ++d) {
      tight[d][i] = dot(*g, directions[d].second);
    }
  }

  for (std::size_t d = 0; d < directions.size(); ++d) {
    const auto &v = values[d];
    std::array<double, kAlong.size()> mean{};
    double ss = 0.0;
    double solves = 0.0;
    for (std::size_t i = 0; i < kAlong.size(); ++i) {
      mean[i] = std::accumulate(v[i].begin(), v[i].end(), 0.0) / kRuns;
      for (const double x : v[i]) {
        ss += (x - mean[i]) * (x - mean[i]);
      }
      solves += (v[i][0] - tight[d][i]) * (v[i][0] - tight[d][i]);
    }
    const double restarts =
        std::sqrt(ss / static_cast<double>(kAlong.size() * static_cast<std::size_t>(kRuns - 1)));
    solves = std::sqrt(solves / static_cast<double>(kAlong.size()));
    const double sigma_g = std::max(restarts, solves);
    const double bar = 3.0 * sigma_g;
    // Least squares of a + b rho + c rho^2 (rho in bohr) by the 3 x 3 normal
    // equations, Cramer's rule.
    std::array<std::array<double, 4>, 3> m{};
    for (std::size_t i = 0; i < kAlong.size(); ++i) {
      const double rho = kAlong[i] / fb::kAngstromPerBohr<double>;
      const std::array<double, 3> basis{1.0, rho, rho * rho};
      for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
          m[a][b] += basis[a] * basis[b];
        }
        m[a][3] += basis[a] * mean[i];
      }
    }
    const auto det = [](const std::array<std::array<double, 3>, 3> &x) {
      return (x[0][0] * ((x[1][1] * x[2][2]) - (x[1][2] * x[2][1]))) -
             (x[0][1] * ((x[1][0] * x[2][2]) - (x[1][2] * x[2][0]))) +
             (x[0][2] * ((x[1][0] * x[2][1]) - (x[1][1] * x[2][0])));
    };
    std::array<std::array<double, 3>, 3> a{};
    for (std::size_t r = 0; r < 3; ++r) {
      for (std::size_t c = 0; c < 3; ++c) {
        a[r][c] = m[r][c];
      }
    }
    const double d0 = det(a);
    std::array<double, 3> coef{};
    for (std::size_t c = 0; c < 3; ++c) {
      std::array<std::array<double, 3>, 3> ac = a;
      for (std::size_t r = 0; r < 3; ++r) {
        ac[r][c] = m[r][3];
      }
      coef[c] = det(ac) / d0;
    }
    std::string line = std::format(
        "[ ref pair ] continuity {:<8} sigma restarts {:.2e}, solves {:.2e}, bar {:.2e}:",
        directions[d].first, restarts, solves, bar);
    for (std::size_t i = 0; i < kAlong.size(); ++i) {
      const double rho = kAlong[i] / fb::kAngstromPerBohr<double>;
      const double residual = mean[i] - (coef[0] + (coef[1] * rho) + (coef[2] * rho * rho));
      if (d == 0) {
        EXPECT_LE(std::abs(residual), bar) << directions[d].first << " at " << kAlong[i] << " A";
      }
      line += std::format(" {:g} A {}{:+.12e} ({:+.1e})", kAlong[i], i < kBelow ? "g_mean " : "",
                          mean[i], residual);
    }
    std::cout << line
              << std::format("; fit a {:+.10e} b {:+.4e} c {:+.4e}\n", coef[0], coef[1], coef[2]);
  }
}

// --- The [optimize] QD-PC MECI through the pair -----------------------------

// tier1_ethylene_c3v_augdz.toml's [optimize] walk at @p start (bohr), into
// @p dir's out/: the chain above with [nevpt2_gradient] QD-PC and
// Opt=Tight's criteria.
fb::Status walk(fb::NodeResources &node, const fb::test::ScratchDir &dir, const Coords &start) {
  write_xyz(dir / "start.xyz", start);
  const std::string toml = std::format(
      "[molecule]\nxyz = \"start.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n[scf]\nmax_iterations = 400\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nconv_tol_grad = 1e-10\n"
      "[nevpt2_gradient]\nquasi_degenerate = true\nvariant = \"pc\"\n"
      "[optimize]\nmeci = [0, 1]\nmax_gap = 1e-7\nmax_force = 1.5e-5\nrms_force = 1e-5\n"
      "max_displacement = 6e-5\nrms_displacement = 4e-5\nmax_steps = 200\n",
      kBasis, kAuxBasis, kNcas, kNelecas, kNroots);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  if (!calc.has_value()) {
    return std::unexpected(calc.error());
  }
  return calc->run(node);
}

// A start inside the pair is refused at evaluation 0, before any step: the
// branching plane has no limit there and there is none to carry. The
// history keeps the error and no iteration.
TEST(PipelineOptimizeReferencePair, RefusesAStartInsideThePair) {
  FBF_REQUIRE_GPU();
  fb::NodeResources node;
  const fb::test::ScratchDir dir("pipeline_optimize_reference_pair_start");
  const fb::Status ran = walk(node, dir, average(park_in_frame()));
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::Numerical) << ran.error().message;
  EXPECT_NE(ran.error().message.find("start off the symmetric point"), std::string::npos)
      << ran.error().message;
  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  EXPECT_TRUE(history->contains("error"));
  EXPECT_TRUE(history->at("iterations").empty());
  EXPECT_FALSE(std::filesystem::exists(dir / "out" / "final.xyz"));
}

// tier 1's C3v walk from Park's own geometry (#847 stopped it at evaluation
// 10): it converges at Opt=Tight, and optimize.json says how. Every
// evaluation under the CASSCF floor carried the plane, with its tau under
// mcscf::kLimitTauGate; every other one has no tau; the first is not
// carried. The head's g_p, g_q and h_pq are null after a carried last
// evaluation, and g_mean is n_atom rows of three. The final geometry's
// largest atom displacement from its C3v symmetrization (in this file's
// frame, whose axis is the methyl C -> lone H line, so the measure carries
// the frame's own tilt) falls by at least kTowardM: the walk went onto M
// rather than stopping near Park's start (7.0e-3 bohr off it; 5.9e-5 at the
// end when this was written, 119 times less, the frame's tilt included).
// About 12 s on the RTX 3080, so the slow tier.
TEST(PipelineOptimizeReferencePair, C3vWalkCarriesThePlaneThroughTheFloor) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  constexpr double kTowardM = 10.0;
  fb::NodeResources node;
  const fb::test::ScratchDir dir("pipeline_optimize_reference_pair_walk");
  Coords start{};
  for (std::size_t k = 0; k < kCoords; ++k) {
    start[k] = kPark[k] / fb::kAngstromPerBohr<double>;
  }
  const fb::Status ran = walk(node, dir, start);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto history = fb::read_json(dir / "out" / "optimize.json");
  ASSERT_TRUE(history.has_value()) << history.error().message;
  const fb::json &iterations = history->at("iterations");
  ASSERT_GE(iterations.size(), 2U);
  std::size_t carried = 0;
  for (std::size_t k = 0; k < iterations.size(); ++k) {
    const fb::json &it = iterations[k];
    const bool below = it.at("root_gap").get<double>() < fb::mcscf::kRootGapFloor;
    EXPECT_EQ(it.at("plane_carried").get<bool>(), below) << "evaluation " << k;
    if (below) {
      ++carried;
      EXPECT_LE(std::abs(it.at("reference_pair_tau").get<double>()), fb::mcscf::kLimitTauGate);
      EXPECT_TRUE(it.at("gap_sensitivity_orbital").is_null()) << "evaluation " << k;
    } else {
      EXPECT_TRUE(it.at("reference_pair_tau").is_null()) << "evaluation " << k;
    }
    std::cout << std::format("[ ref pair ] walk {:>2}: gap {:.3e} max force {:.2e} max step {:.2e} "
                             "root gap {:.3e}{}\n",
                             k, it.at("gap").get<double>(), it.at("max_force").get<double>(),
                             it.at("max_displacement").get<double>(),
                             it.at("root_gap").get<double>(), below ? ", carried" : "");
  }
  EXPECT_FALSE(iterations.front().at("plane_carried").get<bool>());
  EXPECT_GE(carried, 1U);
  EXPECT_LE(history->at("gap").get<double>(), 1e-7);
  const bool last = iterations.back().at("plane_carried").get<bool>();
  for (const std::string key : {"g_p", "g_q", "h_pq"}) {
    EXPECT_EQ(history->at(key).is_null(), last) << key;
  }
  ASSERT_TRUE(history->at("g_mean").is_array());
  EXPECT_EQ(history->at("g_mean").size(), kAtoms);

  // The final geometry, read back, and its C3v deviation.
  const std::string text = fb::read_file(dir / "out" / "final.xyz").value_or("");
  std::istringstream in(text);
  std::string line;
  std::getline(in, line);
  std::getline(in, line);
  Coords angstrom{};
  for (std::size_t a = 0; a < kAtoms; ++a) {
    std::string symbol;
    in >> symbol >> angstrom[3 * a] >> angstrom[(3 * a) + 1] >> angstrom[(3 * a) + 2];
  }
  const auto deviation = [](const Coords &x) {
    const Coords c3v = average(x);
    double worst = 0.0;
    for (std::size_t a = 0; a < kAtoms; ++a) {
      double d2 = 0.0;
      for (std::size_t i = 0; i < 3; ++i) {
        d2 += (x[(3 * a) + i] - c3v[(3 * a) + i]) * (x[(3 * a) + i] - c3v[(3 * a) + i]);
      }
      worst = std::max(worst, std::sqrt(d2));
    }
    return worst;
  };
  const double before = deviation(park_in_frame());
  const double after = deviation(in_frame(angstrom));
  EXPECT_LE(after * kTowardM, before);
  std::cout << std::format("[ ref pair ] walk: {} evaluations, {} carried, final gap {:.3e}, C3v "
                           "deviation {:.2e} -> {:.2e} bohr\n",
                           iterations.size(), carried, history->at("gap").get<double>(), before,
                           after);
}

} // namespace
