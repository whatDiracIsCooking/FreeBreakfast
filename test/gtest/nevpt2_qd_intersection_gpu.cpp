// GPU cases for nevpt2::qd_sc_intersection and nevpt2::qd_pc_intersection,
// the QD-NEVPT2 branching space (g_P, g_Q, h_PQ) of a QD pair at one R and
// the pair's gap sensitivities, on this tree's own DF-SA-CASSCF of
// data/qd_nevpt2/'s cases, converged as #696's generator converges them
// (||g|| < 1e-10), SC and PC (#784):
//
// - against the W forms: g_P, g_Q and h_PQ each against its own
//   qd_*_gradient W-form call at u_P u_P^T, u_Q u_Q^T and W^PQ, the u's the
//   result's own eigenvectors, every pair of h2co_skew_ccpvdz_sa3 and its
//   unequal-weight twin; and s_o, s_c against their formulas on those calls'
//   multipliers. Independently converged Z-vector solves, so 1e-9 of the
//   largest summand (#784's bar amendment, the #767 ruling), PC's s_c 1e-7
//   (kPcCiTolerance);
// - no gap refusal: h2co_ccpvdz_sa2 made exactly degenerate at a consistent
//   solution (make_degenerate), where qd_*_coupling refuses, returns and
//   still equals the W forms; and
//   s_o, s_c are unchanged when u_P, u_Q are rotated within their pair:
//   against separate W-form calls at the rotated u's (1e-9), and by algebra
//   on one set of solves' multipliers (1e-10);
// - the bound: a DF-SA-CASSCF stopped at conv_tol_grad = 1e-5 moves each
//   pair's gap by at most s_o ||g|| + s_c sqrt(n_roots) rho, its ||g|| and
//   rho the stopped solve's own reports and s_o, s_c the converged point's;
// - the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.casscf;
import free_breakfast.pipeline.overlap;
import free_breakfast.pipeline.config;

#include "test_workspaces.h"

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";
// #696's generator's bars.
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-13;
constexpr std::int32_t kMaxCycle = 200;
// Two independently converged Z-vector solves of one quantity (the
// intersection's against a separate W-form call), each to an absolute true
// residual of 1e-10: solver-limited, so 1e-9 of the largest summand
// (#784's bar amendment;
// https://github.com/whatDiracIsCooking/FreeBreakfast/issues/767#issuecomment-6608119792).
constexpr double kSolveTolerance = 1e-9;
// PC's s_c against separately run W-form calls (#784's ruling,
// https://github.com/whatDiracIsCooking/FreeBreakfast/issues/784#issuecomment-6616656181):
// PC's CI adjoint y_bar is reproducible between identical calls only to
// ~5e-8 (measured here, and #729), and z_c is solved from it, so two runs'
// CI multipliers differ by ~1e-9 relative (s_c 4.8e-9 apart on
// h2co_skew_ccpvdz_sa3_weighted, pair (0, 1)). The check is there for a
// wrong pairing factor (2, or 2 w_r), which is O(1). SC's s_c, and g, h
// and s_o of both, keep kSolveTolerance.
constexpr double kPcCiTolerance = 1e-7;
// Algebra on one set of solves' output: round-off.
constexpr double kPathTolerance = 1e-10;
// The stopped DF-SA-CASSCF (#784's bound): its gradient criterion, from the
// converged orbitals turned by a fixed kappa this long over the
// non-redundant pairs; the energy criterion loose enough that the gradient
// alone stops it.
constexpr double kStoppedTolGrad = 1e-5;
constexpr double kStoppedTol = 1e-4;
constexpr double kKappaNorm = 3e-3;
// The fixed angle u_P and u_Q are rotated by within their pair.
constexpr double kPairAngle = 0.3;

const fb::json &reference(std::string_view dir, std::string_view file) {
  static std::map<std::string, fb::json, std::less<>> docs;
  const std::string key = std::format("{}/{}", dir, file);
  if (const auto it = docs.find(key); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / dir / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(key, read.has_value() ? std::move(*read) : fb::json{}).first->second;
}

const fb::json &qd_case(std::string_view name) {
  for (const fb::json &entry : reference("qd_nevpt2", "qd_nevpt2.json").at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in qd_nevpt2/qd_nevpt2.json";
  static const fb::json kNone;
  return kNone;
}

double largest_abs(std::span<const double> v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

double max_diff(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

double norm(std::span<const double> v) {
  double s = 0.0;
  for (const double x : v) {
    s += x * x;
  }
  return std::sqrt(s);
}

std::size_t largest_component(std::span<const double> v) {
  return static_cast<std::size_t>(std::ranges::distance(
      v.begin(), std::ranges::max_element(v, {}, [](const double x) { return std::abs(x); })));
}

// The case's DF-SA-CASSCF through the real registry, and the
// mcscf::Solution of it the NEVPT2 gradient step builds (gamma the weighted
// average of the roots' 1-RDMs).
struct QdRun {
  std::unique_ptr<fb::NodeResources> node;
  std::optional<fb::Calculation> calc;
  std::optional<fb::mcscf::Solution> solution;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;
  // A core Hamiltonian in place of the registry's, and the solution
  // converged over it (DegeneratePair's), or none.
  std::optional<fb::SphericalMatrixAO> h;
};

// The run of a case entry in qd_nevpt2.json's form: name, atoms
// ([symbol, [x, y, z]] in bohr), basis, aux_basis.name, ncas, nelecas and
// weights, and an optional charge (0 when absent).
QdRun qd_run_of(const fb::json &entry) {
  QdRun run{.node = std::make_unique<fb::NodeResources>(
                fb::NodeResourcesConfig{.workspaces = fb::test::kWorkspaces})};
  EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_qd_intersection");
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at(1).get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at(0).get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  std::string weights;
  for (const double w : entry.at("weights").get<std::vector<double>>()) {
    weights += std::format("{}{:.17g}", weights.empty() ? "" : ", ", w);
  }
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol_grad = {:g}\n"
      "conv_tol = {:g}\nmax_cycle = {}\n",
      entry.value("charge", 0), entry.at("basis").get<std::string>(), kDfAuxBasis,
      entry.at("ncas").get<std::int32_t>(), entry.at("nelecas").get<std::int32_t>(),
      entry.at("weights").size(), weights, kConvTolGrad, kConvTol, kMaxCycle);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  EXPECT_TRUE(calc.has_value()) << calc.error().message;
  if (!calc.has_value()) {
    return run;
  }
  const fb::Status ran = calc->run(*run.node);
  EXPECT_TRUE(ran.has_value()) << ran.error().message;
  if (!ran.has_value()) {
    return run;
  }
  run.calc.emplace(std::move(*calc));
  fb::DeviceResources &res = run.node->device(0);
  const auto &cas = run.calc->get<fb::CasscfOutput>();
  const auto &basis = run.calc->get<fb::BasisSetOutput>();
  EXPECT_TRUE(cas.has_value() && basis.has_value());
  if (!cas.has_value() || !basis.has_value()) {
    return run;
  }
  EXPECT_LT(cas->gradient_norm, kConvTolGrad);
  run.ncore = cas->ncore;
  run.ncas = cas->ncas;
  const auto n2 = static_cast<std::size_t>(cas->ncas) * cas->ncas;
  std::vector<double> gamma(n2, 0.0);
  for (std::size_t r = 0; r < cas->weights.size(); ++r) {
    for (std::size_t i = 0; i < n2; ++i) {
      gamma[i] += cas->weights[r] * cas->rdm1[(r * n2) + i];
    }
  }
  run.solution.emplace(fb::mcscf::Solution{
      .c = fb::SphericalMatrixAO(res, basis->basis),
      .roots = {.energies = cas->energies,
                .spin_squares = cas->spin_squares,
                .vectors = fb::Buffers::device<double>(cas->vectors.size(), res.device()),
                .davidson = {},
                .fp32_iterations = 0},
      .energy = cas->energy,
      .gradient_norm = cas->gradient_norm,
      .gamma = std::move(gamma),
      .gamma2 = cas->rdm2,
      .root_gammas = cas->rdm1,
      .iterations = cas->iterations});
  EXPECT_EQ(wwr::wwrMemcpyAsync(run.solution->roots.vectors.data(), cas->vectors.data(),
                                cas->vectors.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(run.solution->c.data(), cas->c.data(),
                                cas->c.num_elements() * sizeof(double),
                                wwr::wwrMemcpyDeviceToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return run;
}

QdRun qd_run(const std::string &name) {
  return qd_run_of(qd_case(name));
}

// The registry's products the calls take, by reference into the run.
struct Inputs {
  const fb::Molecule &molecule;
  const fb::BasisSet &basis;
  std::span<const fb::PrimitivePairClass> classes;
  const fb::DensityFitted &df;
  const fb::SphericalMatrixAO &h;
};

Inputs inputs(QdRun &run) {
  return {.molecule = run.calc->get<fb::MoleculeOutput>()->molecule,
          .basis = run.calc->get<fb::BasisSetOutput>()->basis,
          .classes = run.calc->get<fb::PrimitivePairClassesOutput>()->classes,
          .df = std::get<fb::DensityFitted>(run.calc->get<fb::EriOutput>()->eri),
          .h = run.h ? *run.h : run.calc->get<fb::CoreHamiltonianOutput>()->h};
}

fb::Result<nevpt2::QdIntersection>
intersection_of(QdRun &run, const fb::mcscf::Solution &solution, const std::int32_t p,
                const std::int32_t q, const bool pc,
                const nevpt2::QdIntersectionOptions &options = {}) {
  const Inputs in = inputs(run);
  const auto call = pc ? &nevpt2::qd_pc_intersection : &nevpt2::qd_sc_intersection;
  return call(run.node->device(0), in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
              in.df.l.data(), solution, run.ncore, run.ncas, p, q, 0, options);
}

// What the test reads off one W-form call: G(W), the largest element of
// its two derivative passes' terms (the bar's scale), and its multipliers
// on the host with the SA weights and the Z-vector's report.
struct WForm {
  std::vector<double> total;
  double summand = 0.0;
  std::vector<double> orbital;
  std::vector<double> ci;
  std::vector<double> weights;
  std::int32_t iterations = 0;
  double residual = 0.0;
};

template <typename G>
WForm w_form_fields(fb::DeviceResources &res, const G &g) {
  WForm f{.total = g.total,
          .weights = g.lagrangian.weights,
          .iterations = g.lagrangian.multipliers.iterations,
          .residual = g.lagrangian.multipliers.residual};
  for (const fb::mcscf::GradientTerms *t : {&g.terms, &g.transition}) {
    for (const auto *part :
         {&t->kinetic, &t->potential, &t->overlap, &t->two_electron, &t->nuclear_repulsion}) {
      f.summand = std::max(f.summand, largest_abs(*part));
    }
  }
  const auto &z = g.lagrangian.multipliers;
  f.orbital.resize(z.orbital.num_elements());
  f.ci.resize(z.ci.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(f.orbital.data(), z.orbital.data(),
                                f.orbital.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(f.ci.data(), z.ci.data(), f.ci.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return f;
}

std::optional<WForm> w_form_of(QdRun &run, std::span<const double> w, const bool pc) {
  const Inputs in = inputs(run);
  fb::DeviceResources &res = run.node->device(0);
  if (pc) {
    auto g =
        nevpt2::qd_pc_gradient(res, in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
                               in.df.l.data(), *run.solution, run.ncore, run.ncas, w, 0);
    EXPECT_TRUE(g.has_value()) << g.error().message;
    return g.has_value() ? std::optional{w_form_fields(res, *g)} : std::nullopt;
  }
  auto g = nevpt2::qd_sc_gradient(res, in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
                                  in.df.l.data(), *run.solution, run.ncore, run.ncas, w, 0);
  EXPECT_TRUE(g.has_value()) << g.error().message;
  return g.has_value() ? std::optional{w_form_fields(res, *g)} : std::nullopt;
}

// 1/2 (a b^T + b a^T), n x n row-major, built as the intersection builds it.
std::vector<double> pair_w(std::span<const double> a, std::span<const double> b) {
  const std::size_t n = a.size();
  std::vector<double> w(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      w[(i * n) + j] = 0.5 * ((a[i] * b[j]) + (b[i] * a[j]));
    }
  }
  return w;
}

// s = sqrt(||b - a||^2 + 4 ||c||^2) over (a, b, c) = (z^P, z^Q, z^PQ), and
// the bar's scale, the largest norm summed: max(||a||, ||b||, 2 ||c||).
struct Sensitivity {
  double value = 0.0;
  double scale = 0.0;
};

Sensitivity sensitivity(std::span<const double> a, std::span<const double> b,
                        std::span<const double> c) {
  std::vector<double> d(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    d[i] = b[i] - a[i];
  }
  const double nd = norm(d);
  const double nc = norm(c);
  return {.value = std::sqrt((nd * nd) + (4.0 * nc * nc)),
          .scale = std::max({norm(a), norm(b), 2.0 * nc})};
}

// Root r's CI block scaled by 2 w_r: the multipliers s_c is over.
std::vector<double> scaled_ci(const WForm &f) {
  std::vector<double> out = f.ci;
  const std::size_t nroots = f.weights.size();
  const std::size_t dim = out.size() / nroots;
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] *= 2.0 * f.weights[i / dim];
  }
  return out;
}

// s_o and s_c by their definitions on the three calls' multipliers.
std::pair<Sensitivity, Sensitivity> sensitivities(const WForm &p, const WForm &q, const WForm &pq) {
  return {sensitivity(p.orbital, q.orbital, pq.orbital),
          sensitivity(scaled_ci(p), scaled_ci(q), scaled_ci(pq))};
}

// The three W-form calls at (u_P, u_Q).
std::optional<std::array<WForm, 3>> w_forms(QdRun &run, std::span<const double> u_p,
                                            std::span<const double> u_q, const bool pc) {
  auto gp = w_form_of(run, pair_w(u_p, u_p), pc);
  auto gq = w_form_of(run, pair_w(u_q, u_q), pc);
  auto hpq = w_form_of(run, pair_w(u_p, u_q), pc);
  if (!gp || !gq || !hpq) {
    return std::nullopt;
  }
  return std::array<WForm, 3>{std::move(*gp), std::move(*gq), std::move(*hpq)};
}

// The same three W's in one several-W call (#787's shared path): one set of
// forward arrays, one Hessian, one zvectors solve of the three columns.
std::optional<std::array<WForm, 3>> shared_w_forms(QdRun &run, std::span<const double> u_p,
                                                   std::span<const double> u_q, const bool pc) {
  const Inputs in = inputs(run);
  fb::DeviceResources &res = run.node->device(0);
  const std::array<std::vector<double>, 3> ws{pair_w(u_p, u_p), pair_w(u_q, u_q), pair_w(u_p, u_q)};
  const auto fields = [&](const auto &g) -> std::optional<std::array<WForm, 3>> {
    EXPECT_TRUE(g.has_value()) << g.error().message;
    if (!g.has_value()) {
      return std::nullopt;
    }
    EXPECT_EQ(g->size(), 3U);
    if (g->size() != 3) {
      return std::nullopt;
    }
    return std::array<WForm, 3>{w_form_fields(res, (*g)[0]), w_form_fields(res, (*g)[1]),
                                w_form_fields(res, (*g)[2])};
  };
  if (pc) {
    return fields(nevpt2::qd_pc_gradient(res, in.molecule, in.basis, in.classes, in.df.aux, in.h,
                                         in.df.b, in.df.l.data(), *run.solution, run.ncore,
                                         run.ncas, ws, 0));
  }
  return fields(nevpt2::qd_sc_gradient(res, in.molecule, in.basis, in.classes, in.df.aux, in.h,
                                       in.df.b, in.df.l.data(), *run.solution, run.ncore, run.ncas,
                                       ws, 0));
}

// The shared path's three vectors and two sensitivities against #784's
// three W-form calls at the same u's: independently converged solves, so
// kSolveTolerance of the larger side's largest summand (PC's s_c
// kPcCiTolerance). Each worst ratio folded into @p worst (g_P, g_Q, h_PQ,
// s_o, s_c).
void expect_shared(const std::array<WForm, 3> &shared, const std::array<WForm, 3> &refs,
                   const bool pc, const std::string &label, std::array<double, 5> &worst) {
  constexpr std::array<std::string_view, 3> kNames{"g_P", "g_Q", "h_PQ"};
  for (std::size_t k = 0; k < 3; ++k) {
    const double diff = max_diff(shared[k].total, refs[k].total);
    const double scale = std::max(shared[k].summand, refs[k].summand);
    EXPECT_LE(diff, kSolveTolerance * scale) << label << " shared " << kNames[k];
    worst[k] = std::max(worst[k], diff / scale);
    std::cout << std::format("[ nevpt2   ] qd intersection {} {} shared against three calls: "
                             "{:.2e}, largest summand {:.3e}, relative {:.2e} (shared: {} "
                             "products, true residual {:.2e})\n",
                             label, kNames[k], diff, scale, diff / scale, shared[k].iterations,
                             shared[k].residual);
  }
  const auto [so, sc] = sensitivities(shared[0], shared[1], shared[2]);
  const auto [so_ref, sc_ref] = sensitivities(refs[0], refs[1], refs[2]);
  for (const auto &[ours, def, slot, name] : {std::tuple{so, so_ref, std::size_t{3}, "s_o"},
                                              std::tuple{sc, sc_ref, std::size_t{4}, "s_c"}}) {
    const double diff = std::abs(ours.value - def.value);
    const double scale = std::max(ours.scale, def.scale);
    const double bar = pc && slot == 4 ? kPcCiTolerance : kSolveTolerance;
    EXPECT_LE(diff, bar * scale) << label << " shared " << name;
    worst[slot] = std::max(worst[slot], diff / scale);
    std::cout << std::format("[ nevpt2   ] qd intersection {} {} shared {:.10e} against three "
                             "calls {:.10e}, diff {:.2e}, largest norm summed {:.3e}, relative "
                             "{:.2e}\n",
                             label, name, ours.value, def.value, diff, scale, diff / scale);
  }
}

// The result's three vectors and two sensitivities against the W-form calls
// at its own u's (kSolveTolerance; PC's s_c kPcCiTolerance), each worst
// ratio folded into @p worst (g_P, g_Q, h_PQ, s_o, s_c).
void expect_matches(const nevpt2::QdIntersection &x, const std::array<WForm, 3> &refs,
                    const bool pc, const std::string &label, std::array<double, 5> &worst) {
  const std::array<std::pair<const std::vector<double> *, std::string_view>, 3> fields{
      {{&x.g_p, "g_P"}, {&x.g_q, "g_Q"}, {&x.h_pq, "h_PQ"}}};
  for (std::size_t k = 0; k < 3; ++k) {
    const double diff = max_diff(*fields[k].first, refs[k].total);
    const double scale = refs[k].summand;
    EXPECT_LE(diff, kSolveTolerance * scale) << label << " " << fields[k].second;
    worst[k] = std::max(worst[k], diff / scale);
    std::cout << std::format("[ nevpt2   ] qd intersection {} {}: against its W form {:.2e}, "
                             "largest summand {:.3e}, relative {:.2e} (W form: {} products, true "
                             "residual {:.2e})\n",
                             label, fields[k].second, diff, scale, diff / scale, refs[k].iterations,
                             refs[k].residual);
  }
  const auto [so, sc] = sensitivities(refs[0], refs[1], refs[2]);
  for (const auto &[ours, def, slot, name] :
       {std::tuple{x.gap_sensitivity_orbital, so, std::size_t{3}, "s_o"},
        std::tuple{x.gap_sensitivity_ci, sc, std::size_t{4}, "s_c"}}) {
    const double diff = std::abs(ours - def.value);
    const double bar = pc && slot == 4 ? kPcCiTolerance : kSolveTolerance;
    EXPECT_LE(diff, bar * def.scale) << label << " " << name;
    worst[slot] = std::max(worst[slot], diff / def.scale);
    std::cout << std::format("[ nevpt2   ] qd intersection {} {}: {:.10e} against its definition "
                             "{:.10e}, diff {:.2e}, largest norm summed {:.3e}, relative {:.2e}\n",
                             label, name, ours, def.value, diff, def.scale, diff / def.scale);
  }
}

void print_worst(const std::string &label, const std::array<double, 5> &worst) {
  std::cout << std::format("[ nevpt2   ] qd intersection {} worst relative: g_P {:.2e}, g_Q "
                           "{:.2e}, h_PQ {:.2e}, s_o {:.2e}, s_c {:.2e}\n",
                           label, worst[0], worst[1], worst[2], worst[3], worst[4]);
}

// ---- Against the W forms ---------------------------------------------------

struct WFormCase {
  std::string name;
  bool pc = false;
};

class Nevpt2QdIntersectionWForms : public ::testing::TestWithParam<WFormCase> {};

TEST_P(Nevpt2QdIntersectionWForms, MatchesWForms) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h: 6.3 to 6.5 s each on the RTX 3080 but 20.9 to 24.3 s
  // on the gfx1200 (#793), so every case but the weighted PC one, the
  // cheapest.
  const WFormCase &c = GetParam();
  FBF_FAMILY_REPRESENTATIVE("h2co_skew_ccpvdz_sa3_weighted_pc");
  if (c.name != "h2co_skew_ccpvdz_sa3_weighted" || !c.pc) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  QdRun run = qd_run(c.name);
  ASSERT_TRUE(run.solution.has_value());
  const std::size_t n = run.solution->roots.energies.size();
  std::array<double, 5> worst{};
  std::array<double, 5> shared_worst{};
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p + 1; q < n; ++q) {
      const std::string label = std::format("{} {} ({}, {})", c.pc ? "PC" : "SC", c.name, p, q);
      const auto x = intersection_of(run, *run.solution, static_cast<std::int32_t>(p),
                                     static_cast<std::int32_t>(q), c.pc);
      ASSERT_TRUE(x.has_value()) << label << ": " << x.error().message;
      EXPECT_EQ(x->heff_gap, std::abs(x->eigenvalues[q] - x->eigenvalues[p])) << label;
      const auto refs = w_forms(run, std::span(x->eigenvectors).subspan(p * n, n),
                                std::span(x->eigenvectors).subspan(q * n, n), c.pc);
      ASSERT_TRUE(refs.has_value()) << label;
      expect_matches(*x, *refs, c.pc, label, worst);
      // #787: the shared path at the same u's against the three calls.
      const auto shared = shared_w_forms(run, std::span(x->eigenvectors).subspan(p * n, n),
                                         std::span(x->eigenvectors).subspan(q * n, n), c.pc);
      ASSERT_TRUE(shared.has_value()) << label;
      expect_shared(*shared, *refs, c.pc, label, shared_worst);
      // The intersection's non-default path, once per case, against the same
      // calls: its own H_eff and R, which a gap this size fixes to round-off.
      if (p == 0 && q == 1) {
        const nevpt2::QdIntersectionOptions other{.shared =
                                                      !nevpt2::QdIntersectionOptions{}.shared};
        const std::string other_label = std::format("{} shared = {}", label, other.shared);
        const auto y = intersection_of(run, *run.solution, 0, 1, c.pc, other);
        ASSERT_TRUE(y.has_value()) << other_label << ": " << y.error().message;
        std::array<double, 5> other_worst{};
        expect_matches(*y, *refs, c.pc, other_label, other_worst);
        print_worst(other_label, other_worst);
      }
      std::cout << std::format("[ nevpt2   ] qd intersection {}: heff_gap {:.6e}, root_gap {:.6e}, "
                               "s_o {:.6e}, s_c {:.6e}\n",
                               label, x->heff_gap, x->root_gap, x->gap_sensitivity_orbital,
                               x->gap_sensitivity_ci);
    }
  }
  print_worst(std::format("{} {}", c.pc ? "PC" : "SC", c.name), worst);
  print_worst(std::format("{} {} shared against three calls", c.pc ? "PC" : "SC", c.name),
              shared_worst);
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdIntersectionWForms,
                         ::testing::Values(WFormCase{"h2co_skew_ccpvdz_sa3", false},
                                           WFormCase{"h2co_skew_ccpvdz_sa3", true},
                                           WFormCase{"h2co_skew_ccpvdz_sa3_weighted", false},
                                           WFormCase{"h2co_skew_ccpvdz_sa3_weighted", true}),
                         [](const auto &info) {
                           return std::format("{}_{}", info.param.name,
                                              info.param.pc ? "pc" : "sc");
                         });

// ---- Solutions of the test's own ---------------------------------------------

// nmo x nmo column-major host copies, and their product.
std::vector<double> download(fb::DeviceResources &res, const fb::SphericalMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, std::span<const double> host, fb::SphericalMatrixAO &m) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(m.data(), host.data(), host.size() * sizeof(double),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

std::vector<double> times(std::span<const double> a, std::span<const double> b,
                          const std::size_t n) {
  std::vector<double> out(n * n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t k = 0; k < n; ++k) {
      const double bkj = b[k + (n * j)];
      for (std::size_t i = 0; i < n; ++i) {
        out[i + (n * j)] += a[i + (n * k)] * bkj;
      }
    }
  }
  return out;
}

// The run's DF-SA-CASSCF (its roots and weights) from the orbitals
// @p start (host, nmo x nmo column-major) over inputs(run).h, at
// @p conv_tol and @p conv_tol_grad with the CI at its schedule's tolerance;
// then its core and virtuals canonicalized over its own state-averaged Fock,
// as the Casscf step leaves a solution.
std::optional<fb::mcscf::Solution> canonical_solve(QdRun &run, std::span<const double> start,
                                                   const double conv_tol,
                                                   const double conv_tol_grad) {
  const Inputs in = inputs(run);
  fb::DeviceResources &res = run.node->device(0);
  const auto &cas = run.calc->get<fb::CasscfOutput>();
  const auto nmo = static_cast<std::size_t>(run.solution->c.n());
  const auto ncore = static_cast<std::size_t>(run.ncore);
  const std::size_t nocc = ncore + static_cast<std::size_t>(run.ncas);
  fb::SphericalMatrixAO c0(res, in.basis);
  upload(res, start, c0);

  fb::mcscf::Options options;
  options.conv_tol = conv_tol;
  options.conv_tol_grad = conv_tol_grad;
  options.max_cycle = kMaxCycle;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol = fb::mcscf::solve(
      res, in.basis, in.molecule, in.h, nullptr, {}, {}, c0, run.ncore, run.ncas, cas->nelecas,
      static_cast<std::int32_t>(cas->weights.size()), cas->weights, options, &in.df.b, in.df.naux);
  EXPECT_TRUE(sol.has_value()) << sol.error().message;
  if (!sol.has_value()) {
    return std::nullopt;
  }
  const auto ints = fb::casscf_integrals(res, in.basis, in.molecule, in.h, in.df.b, in.df.naux,
                                         sol->c, run.ncore, run.ncas);
  EXPECT_TRUE(ints.has_value()) << ints.error().message;
  if (!ints.has_value()) {
    return std::nullopt;
  }
  const auto fock =
      fb::response::fock(res, in.molecule, in.h, in.df.b, in.df.naux, sol->c, run.ncore, run.ncas,
                         ints->ppaa, ints->papa, sol->gamma, sol->gamma2);
  EXPECT_TRUE(fock.has_value()) << fock.error().message;
  if (!fock.has_value()) {
    return std::nullopt;
  }
  std::vector<double> f(nmo * nmo);
  std::vector<double> f_active(nmo * nmo);
  EXPECT_EQ(wwr::wwrMemcpyAsync(f.data(), fock->fock_inactive.data(), f.size() * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrMemcpyAsync(f_active.data(), fock->fock_active.data(),
                                f_active.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  std::ranges::transform(f, f_active, f.begin(), std::plus<>{});
  std::vector<double> r;
  std::vector<double> mo_energies;
  const fb::Status canonical =
      fb::canonicalize_core_virtual(res, nmo, ncore, nocc, f, r, mo_energies);
  EXPECT_TRUE(canonical.has_value()) << canonical.error().message;
  upload(res, times(download(res, sol->c), r, nmo), sol->c);
  return std::move(*sol);
}

// H_eff = diag(E_r) + 1/2 (C + C^T) on @p solution, SC's or PC's, n x n
// row-major, through qd_inputs and the effective Hamiltonian.
std::vector<double> qd_heff(QdRun &run, const fb::mcscf::Solution &solution, const bool pc) {
  const Inputs in = inputs(run);
  fb::DeviceResources &res = run.node->device(0);
  const std::size_t n = solution.roots.energies.size();
  std::vector<double> totals;
  const fb::Status got = nevpt2::qd_inputs(
      res, in.molecule, in.basis, in.df.aux, in.h, in.df.b, solution, run.ncore, run.ncas,
      [&](const nevpt2::MoBlocks &blocks,
          const std::span<const nevpt2::EnergyInputs> pairs) -> fb::Status {
        if (pc) {
          auto e = nevpt2::pc_effective_hamiltonian(res, blocks, pairs, 0);
          if (!e.has_value()) {
            return std::unexpected(e.error());
          }
          std::ranges::transform(*e, std::back_inserter(totals), &nevpt2::PcClassEnergies::total);
        } else {
          auto e = nevpt2::sc_effective_hamiltonian(res, blocks, pairs, 0);
          if (!e.has_value()) {
            return std::unexpected(e.error());
          }
          std::ranges::transform(*e, std::back_inserter(totals), &nevpt2::ClassEnergies::total);
        }
        return {};
      });
  EXPECT_TRUE(got.has_value()) << got.error().message;
  if (!got.has_value()) {
    return {};
  }
  std::vector<double> a(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      a[(i * n) + j] = (i == j ? solution.roots.energies[i] : 0.0) +
                       (0.5 * (totals[(i * n) + j] + totals[(j * n) + i]));
    }
  }
  return a;
}

// H_eff's eigenvalues, ascending, by cyclic Jacobi on the host.
std::vector<double> qd_energies(QdRun &run, const fb::mcscf::Solution &solution, const bool pc) {
  std::vector<double> a = qd_heff(run, solution, pc);
  const std::size_t n = solution.roots.energies.size();
  if (a.size() != n * n) {
    return {};
  }
  for (std::int32_t sweep = 0; sweep < 50; ++sweep) {
    double off = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        off = std::max(off, std::abs(a[(i * n) + j]));
      }
    }
    if (off == 0.0) {
      break;
    }
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = i + 1; j < n; ++j) {
        const double aij = a[(i * n) + j];
        if (aij == 0.0) {
          continue;
        }
        const double theta = (a[(j * n) + j] - a[(i * n) + i]) / (2.0 * aij);
        const double t =
            std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0));
        const double c = 1.0 / std::sqrt((t * t) + 1.0);
        const double s = t * c;
        for (std::size_t k = 0; k < n; ++k) {
          const double aki = a[(k * n) + i];
          const double akj = a[(k * n) + j];
          a[(k * n) + i] = (c * aki) - (s * akj);
          a[(k * n) + j] = (s * aki) + (c * akj);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double aik = a[(i * n) + k];
          const double ajk = a[(j * n) + k];
          a[(i * n) + k] = (c * aik) - (s * ajk);
          a[(j * n) + k] = (s * aik) + (c * ajk);
        }
      }
    }
  }
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = a[(i * n) + i];
  }
  std::ranges::sort(values);
  return values;
}

// The DF-SA-CASSCF from the converged orbitals C turned by exp(kappa),
// kappa a fixed antisymmetric matrix of norm kKappaNorm over the
// non-redundant pairs, stopped at kStoppedTolGrad and canonicalized.
std::optional<fb::mcscf::Solution> stopped_solve(QdRun &run) {
  fb::DeviceResources &res = run.node->device(0);
  const auto nmo = static_cast<std::size_t>(run.solution->c.n());
  const auto ncore = static_cast<std::size_t>(run.ncore);
  const std::size_t nocc = ncore + static_cast<std::size_t>(run.ncas);
  // kappa over p > q with q core and p active or virtual, or q active and p
  // virtual; exp(kappa) by its Taylor series (||kappa|| ~ 3e-3: twelve
  // terms are past round-off).
  std::vector<double> kappa(nmo * nmo, 0.0);
  double kk = 0.0;
  std::size_t k = 0;
  for (std::size_t q = 0; q < nocc; ++q) {
    for (std::size_t p = std::max(q + 1, q < ncore ? ncore : nocc); p < nmo; ++p, ++k) {
      const double v = std::sin((1.7 * static_cast<double>(k)) + 0.3);
      kappa[p + (nmo * q)] = v;
      kappa[q + (nmo * p)] = -v;
      kk += v * v;
    }
  }
  for (double &v : kappa) {
    v *= kKappaNorm / std::sqrt(kk);
  }
  std::vector<double> u(nmo * nmo, 0.0);
  std::vector<double> term(nmo * nmo, 0.0);
  for (std::size_t i = 0; i < nmo; ++i) {
    u[i + (nmo * i)] = 1.0;
    term[i + (nmo * i)] = 1.0;
  }
  for (std::int32_t order = 1; order <= 12; ++order) {
    term = times(term, kappa, nmo);
    for (std::size_t i = 0; i < term.size(); ++i) {
      term[i] /= order;
      u[i] += term[i];
    }
  }
  return canonical_solve(run, times(download(res, run.solution->c), u, nmo), kStoppedTol,
                         kStoppedTolGrad);
}

// A QD pair made exactly degenerate on a consistent solution: on
// h2co_ccpvdz_sa2 (C2v, roots 1A1 and 1A2, H_eff's off-diagonal zero by
// symmetry), h + lambda v v^T, v = S C e_k, k the active orbital whose
// occupation the two roots differ most in (so lambda moves one root's
// energy against the other's), and the DF-SA-CASSCF re-converged over it at
// #696's bars from the last orbitals, lambda by secant until H_eff's
// diagonal elements (the roots told apart by gamma_kk) agree to
// kDegenerateTarget. run.h and run.solution become those. The CASSCF is
// stationary over that h, so every W form's Lagrangian is consistent; only
// its nuclear derivative is of a Hamiltonian with no geometry of its own,
// the same function on both sides of every comparison.
constexpr double kDegenerateTarget = 1e-11;

bool make_degenerate(QdRun &run, const bool pc) {
  fb::DeviceResources &res = run.node->device(0);
  const auto nmo = static_cast<std::size_t>(run.solution->c.n());
  const auto n = static_cast<std::size_t>(run.ncas);
  const std::size_t nroots = run.solution->roots.energies.size();
  const std::vector<double> &g = run.solution->root_gammas;
  std::size_t k = 0;
  for (std::size_t t = 1; t < n; ++t) {
    if (std::abs(g[(t * n) + t] - g[(n * n) + (t * n) + t]) >
        std::abs(g[(k * n) + k] - g[(n * n) + (k * n) + k])) {
      k = t;
    }
  }
  const std::vector<double> s = download(res, run.calc->get<fb::OverlapOutput>()->s);
  const std::vector<double> h0 = download(res, run.calc->get<fb::CoreHamiltonianOutput>()->h);
  const std::vector<double> c = download(res, run.solution->c);
  const std::size_t col = static_cast<std::size_t>(run.ncore) + k;
  std::vector<double> v(nmo, 0.0);
  for (std::size_t mu = 0; mu < nmo; ++mu) {
    for (std::size_t nu = 0; nu < nmo; ++nu) {
      v[mu] += s[mu + (nmo * nu)] * c[nu + (nmo * col)];
    }
  }
  // H_eff's diagonal: the root with the larger gamma_kk's less the other's.
  const auto split = [&](const fb::mcscf::Solution &at) {
    const std::vector<double> heff = qd_heff(run, at, pc);
    const std::size_t a =
        at.root_gammas[(k * n) + k] > at.root_gammas[(n * n) + (k * n) + k] ? 0 : 1;
    return heff[a * (nroots + 1)] - heff[(1 - a) * (nroots + 1)];
  };
  const auto at = [&](const double lambda) -> std::optional<double> {
    std::vector<double> h = h0;
    for (std::size_t j = 0; j < nmo; ++j) {
      for (std::size_t i = 0; i < nmo; ++i) {
        h[i + (nmo * j)] += lambda * v[i] * v[j];
      }
    }
    run.h.emplace(res, inputs(run).basis);
    upload(res, h, *run.h);
    auto sol = canonical_solve(run, download(res, run.solution->c), kConvTol, kConvTolGrad);
    if (!sol.has_value()) {
      return std::nullopt;
    }
    run.solution.emplace(std::move(*sol));
    return split(*run.solution);
  };
  double l0 = 0.0;
  double d0 = split(*run.solution);
  const double slope = std::abs(g[(k * n) + k] - g[(n * n) + (k * n) + k]);
  double l1 = -d0 / slope;
  std::optional<double> d1 = at(l1);
  for (std::int32_t step = 0; d1.has_value() && std::abs(*d1) > kDegenerateTarget && step < 20;
       ++step) {
    const double l2 = l1 - (*d1 * (l1 - l0) / (*d1 - d0));
    l0 = l1;
    d0 = *d1;
    l1 = l2;
    d1 = at(l1);
  }
  if (!d1.has_value()) {
    return false;
  }
  std::cout << std::format("[ nevpt2   ] qd intersection {} degenerate: lambda {:.12e} on active "
                           "orbital {}, H_eff diagonal split {:.3e}, ||g|| {:.3e}\n",
                           pc ? "PC" : "SC", l1, k, *d1, run.solution->gradient_norm);
  return std::abs(*d1) <= kDegenerateTarget;
}

// ---- No gap refusal, and the sensitivities' rotation invariance ------------

// The rotated pair's multipliers by linearity: with u_P' = c u_P + s u_Q and
// u_Q' = -s u_P + c u_Q, W^P' = c^2 W^P + s^2 W^Q + 2cs W^PQ, W^Q' = s^2 W^P +
// c^2 W^Q - 2cs W^PQ and W^PQ' = cs (W^Q - W^P) + (c^2 - s^2) W^PQ, and the
// multipliers are linear in W.
std::array<std::vector<double>, 3> rotated(std::span<const double> zp, std::span<const double> zq,
                                           std::span<const double> zpq, const double angle) {
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  std::array<std::vector<double>, 3> out;
  for (auto &v : out) {
    v.resize(zp.size());
  }
  for (std::size_t i = 0; i < zp.size(); ++i) {
    out[0][i] = (c * c * zp[i]) + (s * s * zq[i]) + (2.0 * c * s * zpq[i]);
    out[1][i] = (s * s * zp[i]) + (c * c * zq[i]) - (2.0 * c * s * zpq[i]);
    out[2][i] = (c * s * (zq[i] - zp[i])) + (((c * c) - (s * s)) * zpq[i]);
  }
  return out;
}

// On h2co_ccpvdz_sa2 made exactly degenerate at a consistent solution
// (make_degenerate): qd_*_coupling refuses the pair, and the intersection
// returns and equals the W forms at its own u's. Then s_o and s_c at u's
// rotated by kPairAngle within the pair. (#784 named Nevpt2QdNac.Refuses'
// setup, one CASSCF root energy moved by hand; there the roots are no
// longer eigenvectors at their energies, and every W form's own gate
// refuses its Lagrangian: active-active asymmetry 2.4e-1 in SC.)
TEST(Nevpt2QdIntersection, DegeneratePair) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  constexpr std::size_t kN = 2;
  for (const bool pc : {false, true}) {
    const std::string label = std::format("{} degenerate h2co_ccpvdz_sa2 (0, 1)", pc ? "PC" : "SC");
    QdRun run = qd_run("h2co_ccpvdz_sa2");
    ASSERT_TRUE(run.solution.has_value());
    ASSERT_TRUE(make_degenerate(run, pc)) << label;
    const fb::mcscf::Solution &solution = *run.solution;
    const Inputs in = inputs(run);
    fb::DeviceResources &res = run.node->device(0);

    // The control: the coupling refuses the pair.
    const auto coupled =
        pc ? nevpt2::qd_pc_coupling(res, in.molecule, in.basis, in.classes, in.df.aux, in.h,
                                    in.df.b, in.df.l.data(), solution, run.ncore, run.ncas, 0, 1, 0)
                 .transform([](auto &&) { return 0; })
           : nevpt2::qd_sc_coupling(res, in.molecule, in.basis, in.classes, in.df.aux, in.h,
                                    in.df.b, in.df.l.data(), solution, run.ncore, run.ncas, 0, 1, 0)
                 .transform([](auto &&) { return 0; });
    ASSERT_FALSE(coupled.has_value()) << label;
    EXPECT_EQ(coupled.error().kind, fb::ErrorKind::Numerical) << coupled.error().message;
    EXPECT_NE(coupled.error().message.find("QD states 0 and 1"), std::string::npos)
        << coupled.error().message;

    const auto x = intersection_of(run, solution, 0, 1, pc);
    ASSERT_TRUE(x.has_value()) << label << ": " << x.error().message;
    EXPECT_LT(x->heff_gap, fb::mcscf::kRootGapFloor) << label;
    std::cout << std::format("[ nevpt2   ] qd intersection {}: heff_gap {:.3e} (the coupling "
                             "refuses: {})\n",
                             label, x->heff_gap, coupled.error().message);
    const std::span<const double> r = x->eigenvectors;
    const std::span<const double> u_p = r.subspan(0, kN);
    const std::span<const double> u_q = r.subspan(kN, kN);
    const auto refs = w_forms(run, u_p, u_q, pc);
    ASSERT_TRUE(refs.has_value()) << label;
    std::array<double, 5> worst{};
    expect_matches(*x, *refs, pc, label, worst);
    print_worst(label, worst);
    // #787: the shared path at the same u's against the three calls.
    const auto shared = shared_w_forms(run, u_p, u_q, pc);
    ASSERT_TRUE(shared.has_value()) << label;
    std::array<double, 5> shared_worst{};
    expect_shared(*shared, *refs, pc, label, shared_worst);
    print_worst(label + " shared against three calls", shared_worst);

    // Rotated within the pair: separate W-form calls at the rotated u's.
    const double c = std::cos(kPairAngle);
    const double s = std::sin(kPairAngle);
    std::array<double, kN> up_rot{};
    std::array<double, kN> uq_rot{};
    for (std::size_t i = 0; i < kN; ++i) {
      up_rot[i] = (c * u_p[i]) + (s * u_q[i]);
      uq_rot[i] = (-s * u_p[i]) + (c * u_q[i]);
    }
    const auto turned = w_forms(run, up_rot, uq_rot, pc);
    ASSERT_TRUE(turned.has_value()) << label;
    const auto [so_rot, sc_rot] = sensitivities((*turned)[0], (*turned)[1], (*turned)[2]);
    for (const auto &[ours, other, bar, name] :
         {std::tuple{x->gap_sensitivity_orbital, so_rot, kSolveTolerance, "s_o"},
          std::tuple{x->gap_sensitivity_ci, sc_rot, pc ? kPcCiTolerance : kSolveTolerance,
                     "s_c"}}) {
      const double diff = std::abs(ours - other.value);
      EXPECT_LE(diff, bar * other.scale) << label << " rotated " << name;
      std::cout << std::format("[ nevpt2   ] qd intersection {} {} rotated by {}: {:.10e} "
                               "against {:.10e}, diff {:.2e}, relative {:.2e}\n",
                               label, name, kPairAngle, ours, other.value, diff,
                               diff / other.scale);
    }

    // And by algebra on one set of solves: the rotated pair's multipliers
    // as combinations of the unrotated calls'.
    const auto [so, sc] = sensitivities((*refs)[0], (*refs)[1], (*refs)[2]);
    const auto zo = rotated((*refs)[0].orbital, (*refs)[1].orbital, (*refs)[2].orbital, kPairAngle);
    const auto zc =
        rotated(scaled_ci((*refs)[0]), scaled_ci((*refs)[1]), scaled_ci((*refs)[2]), kPairAngle);
    for (const auto &[unrotated, turned_z, name] :
         {std::tuple{so, &zo, "s_o"}, std::tuple{sc, &zc, "s_c"}}) {
      const Sensitivity algebra = sensitivity((*turned_z)[0], (*turned_z)[1], (*turned_z)[2]);
      const double diff = std::abs(algebra.value - unrotated.value);
      EXPECT_LE(diff, kPathTolerance * unrotated.scale) << label << " algebra " << name;
      std::cout << std::format("[ nevpt2   ] qd intersection {} {} rotated by algebra: {:.10e} "
                               "against {:.10e}, relative {:.2e}\n",
                               label, name, algebra.value, unrotated.value, diff / unrotated.scale);
    }
  }
}

// ---- The bound -------------------------------------------------------------

// On h2co_skew_ccpvdz_sa3, every pair: |gap(stopped) - gap(converged)| <=
// s_o ||g|| + s_c sqrt(n_roots) rho, s_o and s_c the converged point's, ||g||
// and rho the stopped solve's reported gradient_norm and
// roots.davidson.max_residual_norm. A wrong pairing factor on either
// sensitivity (2, or 2 w_r) is what this sees and the definition check does
// not.
class Nevpt2QdIntersectionBound : public ::testing::TestWithParam<bool> {};

TEST_P(Nevpt2QdIntersectionBound, HoldsForAStoppedSolve) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h (#977): PC, 2.2 s on the RTX 3080; SC stands for it.
  if (GetParam()) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const bool pc = GetParam();
  QdRun run = qd_run("h2co_skew_ccpvdz_sa3");
  ASSERT_TRUE(run.solution.has_value());
  const std::size_t n = run.solution->roots.energies.size();
  const std::optional<fb::mcscf::Solution> stopped = stopped_solve(run);
  ASSERT_TRUE(stopped.has_value());
  const double g = stopped->gradient_norm;
  const double rho = stopped->roots.davidson.max_residual_norm;
  EXPECT_LT(g, kStoppedTolGrad);
  std::cout << std::format("[ nevpt2   ] qd intersection {} bound: stopped after {} macro "
                           "iterations, ||g|| {:.3e}, Davidson max residual {:.3e}\n",
                           pc ? "PC" : "SC", stopped->iterations.size(), g, rho);
  const std::vector<double> converged = qd_energies(run, *run.solution, pc);
  const std::vector<double> moved = qd_energies(run, *stopped, pc);
  ASSERT_EQ(converged.size(), n);
  ASSERT_EQ(moved.size(), n);
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = p + 1; q < n; ++q) {
      const std::string label =
          std::format("{} h2co_skew_ccpvdz_sa3 ({}, {})", pc ? "PC" : "SC", p, q);
      const auto x = intersection_of(run, *run.solution, static_cast<std::int32_t>(p),
                                     static_cast<std::int32_t>(q), pc);
      ASSERT_TRUE(x.has_value()) << label << ": " << x.error().message;
      const double shift = std::abs((moved[q] - moved[p]) - (converged[q] - converged[p]));
      const double orbital = x->gap_sensitivity_orbital * g;
      const double ci = x->gap_sensitivity_ci * std::sqrt(static_cast<double>(n)) * rho;
      EXPECT_LE(shift, orbital + ci) << label;
      std::cout << std::format("[ nevpt2   ] qd intersection {} bound: |delta gap| {:.3e} <= "
                               "s_o ||g|| {:.3e} + s_c sqrt(n) rho {:.3e} = {:.3e} (ratio {:.3f}; "
                               "s_o {:.4e}, s_c {:.4e})\n",
                               label, shift, orbital, ci, orbital + ci, shift / (orbital + ci),
                               x->gap_sensitivity_orbital, x->gap_sensitivity_ci);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Methods, Nevpt2QdIntersectionBound, ::testing::Values(false, true),
                         [](const auto &info) { return info.param ? "pc" : "sc"; });

// ---- Refusals ----------------------------------------------------------------

TEST(Nevpt2QdIntersection, Refuses) {
  FBF_REQUIRE_GPU();
  QdRun run = qd_run("h2co_ccpvdz_sa2");
  ASSERT_TRUE(run.solution.has_value());
  for (const bool pc : {false, true}) {
    const std::string_view who = pc ? "nevpt2::qd_pc_intersection" : "nevpt2::qd_sc_intersection";
    for (const auto &[p, q, phrase] :
         {std::tuple{1, 1, "states 1 and 1"}, std::tuple{0, 2, "state 2 of 2 roots"},
          std::tuple{-1, 0, "state -1 of 2 roots"}}) {
      const auto got = intersection_of(run, *run.solution, p, q, pc);
      ASSERT_FALSE(got.has_value()) << who << " (" << p << ", " << q << ")";
      EXPECT_EQ(got.error().kind, fb::ErrorKind::InvalidConfig) << got.error().message;
      EXPECT_NE(got.error().message.find(who), std::string::npos) << got.error().message;
      EXPECT_NE(got.error().message.find(phrase), std::string::npos) << got.error().message;
      std::cout << std::format("[ nevpt2   ] qd intersection refuses ({}, {}): {}\n", p, q,
                               got.error().message);
    }
  }

  // A class PC refuses (root 1's CI vector zero: no positive eigenvalue) is
  // PC's error, as qd_pc_coupling refuses it.
  fb::DeviceResources &res = run.node->device(0);
  const Inputs in = inputs(run);
  const std::size_t total = run.solution->roots.vectors.num_elements();
  const std::vector<double> zero(total / 2, 0.0);
  ASSERT_EQ(wwr::wwrMemcpyAsync(run.solution->roots.vectors.data() + (total / 2), zero.data(),
                                zero.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  ASSERT_TRUE(res.sync().has_value());
  const auto coupled =
      nevpt2::qd_pc_coupling(res, in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
                             in.df.l.data(), *run.solution, run.ncore, run.ncas, 0, 1, 0);
  ASSERT_FALSE(coupled.has_value());
  const auto got = intersection_of(run, *run.solution, 0, 1, true);
  ASSERT_FALSE(got.has_value());
  EXPECT_EQ(got.error().kind, fb::ErrorKind::Numerical) << got.error().message;
  EXPECT_EQ(got.error().kind, coupled.error().kind) << coupled.error().message;
  for (const auto *message : {&got.error().message, &coupled.error().message}) {
    EXPECT_NE(message->find("no positive eigenvalue"), std::string::npos) << *message;
  }
  std::cout << std::format("[ nevpt2   ] qd pc intersection refuses root 1 zero: {}\n",
                           got.error().message);
}

// ---- Timing (#787) -----------------------------------------------------------

// Not a check: the wall time of both paths, which set
// QdIntersectionOptions::shared's default (README.md "Sharing the three
// calls"). Skipped unless FBF_QD_INTERSECTION_TIMING is set, and run alone
// on a held card. Per case and scheme, one warm-up call of each path, then
// kTimingRepeats calls of each, alternating, pair (0, 1). The cases:
// h2co_skew_ccpvdz_sa3, and ethylene SA2-CAS(2,2)/cc-pVDZ at
// data/casscf_meci/'s start geometry and weights. #787 asked for #785's
// ethylene SA3-CAS(2,2), but this tree's SA3 CASSCF from RHF orbitals did
// not converge at the MECI geometry (200 macro iterations, |g| 3.0e-1), the
// start (macro iteration 34, the augmented Hessian's lowest root with no
// gradient component) or planar ethylene (200, |g| 6.3e-2); SA2 at the MECI
// is degenerate in its CASSCF roots, which root_pair_multipliers refuses.
// The size, which sets the cost, is the same; the third root is not.
// QdIntersectionOptions::zvector_tolerance (#858, #864), the one parameter
// of the Z-vector's solve and its true-residual check, on h2co_ccpvdz_sa2's
// pair (0, 1), SC:
// - at 1e-11, 10x under the default, the intersection succeeds (every W's
//   true residual at most 1e-11) and g_P, g_Q and h_PQ agree with the
//   default call's to kSolveTolerance of the W form's largest summand (two
//   independently converged solves);
// - the W form at u_P u_P^T with the same tolerance reports a true residual
//   of at most 1e-11 and the same gradient;
// - a tolerance not positive is InvalidConfig, naming the intersection.
// Not 1e-12 (#885). The conjugate gradients stop on their recurrence under
// the tolerance and the true residual is held to the same number, so it
// passes only if the products' error keeps it under too. With the CI
// response inside H~ at response::response_tolerance_for's floor, 1e-10, that
// error is of order 1e-13 here (mcscf/intersection/README.md "The floor under
// the true residual"): 6.5% of a 1e-12 bar, which column 2 crossed from run
// to run (1.017e-12, 1.065e-12), and about 1% of this one.
constexpr double kTightZvector = 1e-11;

TEST(Nevpt2QdIntersection, ZvectorTolerance) {
  FBF_REQUIRE_GPU();
  QdRun run = qd_run("h2co_ccpvdz_sa2");
  ASSERT_TRUE(run.solution.has_value());
  auto loose = intersection_of(run, *run.solution, 0, 1, false);
  ASSERT_TRUE(loose.has_value()) << loose.error().message;
  auto tight =
      intersection_of(run, *run.solution, 0, 1, false, {.zvector_tolerance = kTightZvector});
  ASSERT_TRUE(tight.has_value()) << tight.error().message;

  const Inputs in = inputs(run);
  const std::size_t n = run.solution->roots.energies.size();
  const std::span<const double> u_p{tight->eigenvectors.data(), n};
  std::vector<double> w(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      w[(i * n) + j] = u_p[i] * u_p[j];
    }
  }
  auto one =
      nevpt2::qd_sc_gradient(run.node->device(0), in.molecule, in.basis, in.classes, in.df.aux,
                             in.h, in.df.b, in.df.l.data(), *run.solution, run.ncore, run.ncas,
                             std::span<const double>{w}, 0, kTightZvector);
  ASSERT_TRUE(one.has_value()) << one.error().message;
  EXPECT_LE(one->lagrangian.multipliers.residual, kTightZvector);
  // The bar's scale: the W form's largest summand, as the other checks here.
  const double scale = w_form_fields(run.node->device(0), *one).summand;
  EXPECT_LE(max_diff(one->total, tight->g_p), kSolveTolerance * scale);
  for (const auto &[ours, want, what] :
       {std::tuple{&tight->g_p, &loose->g_p, "g_P"}, std::tuple{&tight->g_q, &loose->g_q, "g_Q"},
        std::tuple{&tight->h_pq, &loose->h_pq, "h_PQ"}}) {
    EXPECT_LE(max_diff(*ours, *want), kSolveTolerance * scale) << what;
  }

  for (const double bad : {0.0, -1e-10}) {
    const auto refused =
        intersection_of(run, *run.solution, 0, 1, false, {.zvector_tolerance = bad});
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig) << refused.error().message;
    EXPECT_NE(refused.error().message.find("nevpt2::qd_sc_intersection"), std::string::npos)
        << refused.error().message;
  }
  std::cout << std::format("[ qd intersection] zvector_tolerance {:g}: W form {} products to "
                           "{:.2e}; g_P, g_Q, h_PQ against the default's within {:.1e}\n",
                           kTightZvector, one->lagrangian.multipliers.iterations,
                           one->lagrangian.multipliers.residual, kSolveTolerance * scale);
}

// PSB3 (#864): DF-SA3-CASSCF(6,6)/cc-pVDZ of the cation with C13=C14
// twisted 90 degrees, the CASSCF root gap 2.9e-2 Eh, so no seam. Before the
// CI response inside H~ was converged to response::response_tolerance_for the
// Z-vector's tolerance (it was a fixed 1e-8 of its right-hand side, now
// 1e-10), the products' error put a floor under
// the true residual: qd_sc_intersection here was refused, "column 0 of 3 has
// true residual 2.546e-10 after 31 products" (the recurrence 9.8e-11), and
// the state gradient's one-column solve returned 2.6e-10 to 3.7e-10
// unchecked. Now (CUDA, 3080) the three columns end at 9.8e-11, 6.0e-11 and
// 4.2e-11 after 31 products, their recurrence's values. Held:
// - the SC intersection of QD states 0 and 1 succeeds, so each column's true
//   residual is at most 1e-10;
// - the SC state gradient of state 0 succeeds, its one column's true
//   residual at most 1e-10 too, which the one-column path now checks.
// Slow tier: the PSB3 CASSCF alone is over the fast tiers' ten seconds.
constexpr std::string_view kPsb3Twisted = R"(N -0.0454402492  0.0522740911 -0.0000000000
C  1.1276796768  0.5920624626  0.0000000000
C  2.3467584180 -0.1268535537  0.0000000000
C  3.5352478218  0.5543991513  0.0000000000
C  4.2381819494  0.9573272501 -1.1898886726
C  5.4137881019  1.6311951607 -1.1318525401
H  1.1448444823  1.6750816790 -0.0000000000
H  2.3229137518 -1.2094227332 -0.0000000000
H  3.9874420661  0.8136009171  0.9516444335
H  3.7996110498  0.7059345067 -2.1474371187
H -0.1682915976 -0.9466686957 -0.0000000000
H -0.8778598767  0.6155991196 -0.0000000000
H  5.8725712624  1.8941737519 -0.1873080265
H  5.9322594717  1.9283875648 -2.0319280688)";
constexpr double kPsb3ZvectorTolerance = 1e-10;

TEST(Nevpt2QdIntersection, Psb3TrueResidual) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  fb::json entry = {{"name", "psb3_tw90"},
                    {"atoms", fb::json::array()},
                    {"basis", "cc-pVDZ"},
                    {"aux_basis", {{"name", kDfAuxBasis}}},
                    {"ncas", 6},
                    {"nelecas", 6},
                    {"weights", {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}},
                    {"charge", 1}};
  std::istringstream lines{std::string{kPsb3Twisted}};
  std::string symbol;
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  while (lines >> symbol >> x >> y >> z) {
    entry["atoms"].push_back({symbol,
                              {x / fb::kAngstromPerBohr<double>, y / fb::kAngstromPerBohr<double>,
                               z / fb::kAngstromPerBohr<double>}});
  }
  ASSERT_EQ(entry["atoms"].size(), 14U);
  QdRun run = qd_run_of(entry);
  ASSERT_TRUE(run.solution.has_value());
  const fb::ScopedDeviceIndex on{run.node->device(0).dev_idx()};

  auto both = intersection_of(run, *run.solution, 0, 1, false);
  ASSERT_TRUE(both.has_value()) << both.error().message;
  const Inputs in = inputs(run);
  auto state = nevpt2::qd_sc_gradient(run.node->device(0), in.molecule, in.basis, in.classes,
                                      in.df.aux, in.h, in.df.b, in.df.l.data(), *run.solution,
                                      run.ncore, run.ncas, 0, 0);
  ASSERT_TRUE(state.has_value()) << state.error().message;
  const auto &zs = state->lagrangian.multipliers;
  EXPECT_LE(zs.residual, kPsb3ZvectorTolerance);
  std::cout << std::format("[ qd intersection] PSB3 tw90: root gap {:.3e}, H_eff gap {:.3e}; "
                           "state 0's Z-vector {} products to {:.2e}\n",
                           both->root_gap, both->heff_gap, zs.iterations, zs.residual);
}

TEST(Nevpt2QdIntersection, Timing) {
  FBF_REQUIRE_GPU();
  if (std::getenv("FBF_QD_INTERSECTION_TIMING") == nullptr) {
    GTEST_SKIP() << "a measurement, not a check: set FBF_QD_INTERSECTION_TIMING to run it";
  }
  constexpr std::int32_t kTimingRepeats = 3;
  const fb::json &meci = reference("casscf_meci", "casscf_meci.json");
  fb::json ethylene{{"name", "c2h4_ccpvdz_sa2"},
                    {"basis", meci.at("basis")},
                    {"aux_basis", {{"name", meci.at("aux_basis").at("name")}}},
                    {"ncas", meci.at("ncas")},
                    {"nelecas", meci.at("nelecas")},
                    {"weights", meci.at("weights")},
                    {"atoms", fb::json::array()}};
  for (const fb::json &atom : meci.at("start").at("atoms")) {
    ethylene["atoms"].push_back(fb::json::array({atom.at("symbol"), atom.at("coord")}));
  }
  for (const auto &[label, entry] : {std::pair<std::string_view, const fb::json *>{
                                         "h2co_skew_ccpvdz_sa3", &qd_case("h2co_skew_ccpvdz_sa3")},
                                     std::pair<std::string_view, const fb::json *>{
                                         "ethylene SA2-CAS(2,2)/cc-pVDZ", &ethylene}}) {
    QdRun run = qd_run_of(*entry);
    ASSERT_TRUE(run.solution.has_value()) << label;
    for (const bool pc : {false, true}) {
      std::array<std::vector<double>, 2> seconds;
      for (std::int32_t rep = -1; rep < kTimingRepeats; ++rep) {
        for (const bool shared : {false, true}) {
          const auto start = std::chrono::steady_clock::now();
          const auto x = intersection_of(run, *run.solution, 0, 1, pc, {.shared = shared});
          const std::chrono::duration<double> took = std::chrono::steady_clock::now() - start;
          ASSERT_TRUE(x.has_value()) << label << ": " << x.error().message;
          if (rep >= 0) {
            seconds[shared ? 1 : 0].push_back(took.count());
          }
        }
      }
      const auto summary = [](const std::vector<double> &t) {
        return std::format("min {:.3f} s, mean {:.3f} s", std::ranges::min(t),
                           std::accumulate(t.begin(), t.end(), 0.0) /
                               static_cast<double>(t.size()));
      };
      std::cout << std::format("[ nevpt2   ] qd intersection timing {} {} (0, 1): three calls {}; "
                               "shared {}\n",
                               pc ? "PC" : "SC", label, summary(seconds[0]), summary(seconds[1]));
    }
  }
}

} // namespace
