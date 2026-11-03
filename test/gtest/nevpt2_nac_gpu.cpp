// GPU cases for nevpt2::qd_sc_nac and nevpt2::qd_pc_nac, the QD-NEVPT2
// derivative coupling d_PQ = h_PQ / (E_Q - E_P) + d^CAS + d^PT2, on this
// tree's own DF-SA-CASSCF of data/qd_nevpt2/'s cases, converged as #696's
// generator converges them (||g|| < 1e-10), SC and PC, every pair:
//
// - against #743's zeroth-order references (data/qd_nevpt2_nac/coupling/):
//   d^NEV + d^CAS = h_PQ / (E_Q - E_P) + d^CAS against `{sc,pc}_zeroth`,
//   within 2x each component's error bar; then h_QP = h_PQ and
//   d_QP = -d_PQ (independent solves) to 1e-9 of the largest part summed
//   (#748's ruling), total against its formula, and the CSF split against
//   one mcscf::df_nac combination at (T^PQ, D^PT2) built here from the
//   coupling's own R and nevpt2::qd_inputs;
// - against #744's brute force (data/qd_nevpt2_nac/brute_force/): the whole
//   d_PQ (`total`) and d^PT2 alone (`pt2`), within 2x each component's
//   error bar (its `zeroth`, #743's family, printed);
// - total against its formula at every NacOptions setting;
// - the refusals: a degenerate QD pair, and a coupled pair of CASSCF roots
//   under kRootGapFloor.
//
// The data's phases are its own run's: each pair's sign is s_m(P) s_m(Q),
// read off H_eff's element (m(P), m(Q)), m(P) the index of R_P's largest
// component, ours against the data's, never off d itself (#747's rule).

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.mcscf;
import free_breakfast.nevpt2;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
namespace nevpt2 = free_breakfast::nevpt2;

constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";
// #696's generator's bars, at which #743 and #744 converged every point
// (#744 polished further, to ||g|| ~1e-14).
constexpr double kConvTolGrad = 1e-10;
constexpr double kConvTol = 1e-13;
constexpr std::int32_t kMaxCycle = 200;
// Two independently converged Z-vector solves of one quantity (the swapped
// pair, cas + pt2 against the combination) agree to the solver, not
// round-off: 1e-9 of the same scale (#775,
// https://github.com/whatDiracIsCooking/FreeBreakfast/issues/767#issuecomment-6608119792).
constexpr double kSolveTolerance = 1e-9;
// Against committed numerical data: 2x the component's error bar.
constexpr double kBarFactor = 2.0;
// Path agreement and the symmetry checks (#749's "How results are
// compared", #748's ruling on the scale).
constexpr double kPathTolerance = 1e-10;
// H_eff's element that signs a pair must stand clear of round-off, and ours
// must be the data's in magnitude (#747's test's bars).
constexpr double kPhaseElementFloor = 1e-6;
constexpr double kPhaseMagnitudeTolerance = 1e-4;

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

double dot(std::span<const double> a, std::span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  double s = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    s += a[i] * b[i];
  }
  return s;
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
};

QdRun qd_run(const std::string &name) {
  QdRun run{.node = std::make_unique<fb::NodeResources>()};
  const fb::json &entry = qd_case(name);
  EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  const fb::test::ScratchDir dir("nevpt2_nac");
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
      "[molecule]\nxyz = \"molecule.xyz\"\n[basis_set]\nbasis = \"{}\"\n"
      "[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n"
      "[casscf]\nncas = {}\nnelecas = {}\nnroots = {}\nweights = [{}]\nconv_tol_grad = {:g}\n"
      "conv_tol = {:g}\nmax_cycle = {}\n",
      entry.at("basis").get<std::string>(), kDfAuxBasis, entry.at("ncas").get<std::int32_t>(),
      entry.at("nelecas").get<std::int32_t>(), entry.at("weights").size(), weights, kConvTolGrad,
      kConvTol, kMaxCycle);
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

// The registry's products the coupling takes, by reference into the run.
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
          .h = run.calc->get<fb::CoreHamiltonianOutput>()->h};
}

fb::Result<nevpt2::QdNac> nac_of(QdRun &run, const std::int32_t p, const std::int32_t q,
                                 const bool pc, const fb::mcscf::NacOptions &options = {}) {
  const Inputs in = inputs(run);
  const auto call = pc ? &nevpt2::qd_pc_nac : &nevpt2::qd_sc_nac;
  return call(run.node->device(0), in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
              in.df.l.data(), *run.solution, run.ncore, run.ncas, p, q, 0, options);
}

// What the test reads off the interstate coupling: h_PQ, H_eff's
// eigenvalues and R, and the largest element of the two derivative passes'
// terms that sum to h (#748's scale).
struct Coupling {
  std::vector<double> h;
  std::vector<double> eigenvalues;
  std::vector<double> eigenvectors;
  double summand = 0.0;
};

template <typename G>
Coupling coupling_fields(const G &g) {
  Coupling c{.h = g.total,
             .eigenvalues = g.lagrangian.eigenvalues,
             .eigenvectors = g.lagrangian.eigenvectors};
  for (const fb::mcscf::GradientTerms *t : {&g.terms, &g.transition}) {
    for (const auto *part :
         {&t->kinetic, &t->potential, &t->overlap, &t->two_electron, &t->nuclear_repulsion}) {
      c.summand = std::max(c.summand, largest_abs(*part));
    }
  }
  return c;
}

std::optional<Coupling> coupling_of(QdRun &run, const std::int32_t p, const std::int32_t q,
                                    const bool pc) {
  const Inputs in = inputs(run);
  fb::DeviceResources &res = run.node->device(0);
  if (pc) {
    auto g =
        nevpt2::qd_pc_coupling(res, in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
                               in.df.l.data(), *run.solution, run.ncore, run.ncas, p, q, 0);
    EXPECT_TRUE(g.has_value()) << g.error().message;
    return g.has_value() ? std::optional{coupling_fields(*g)} : std::nullopt;
  }
  auto g = nevpt2::qd_sc_coupling(res, in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
                                  in.df.l.data(), *run.solution, run.ncore, run.ncas, p, q, 0);
  EXPECT_TRUE(g.has_value()) << g.error().message;
  return g.has_value() ? std::optional{coupling_fields(*g)} : std::nullopt;
}

// T^PQ = 1/2 (R_P R_Q^T - R_Q R_P^T), n x n row-major, R column P at [P n].
std::vector<double> t_of(std::span<const double> r, const std::size_t n, const std::size_t p,
                         const std::size_t q) {
  std::vector<double> t(n * n);
  for (std::size_t m = 0; m < n; ++m) {
    for (std::size_t k = 0; k < n; ++k) {
      t[(m * n) + k] =
          0.5 * ((r[(p * n) + m] * r[(q * n) + k]) - (r[(q * n) + m] * r[(p * n) + k]));
    }
  }
  return t;
}

// D^PT2 at @p t, on the host, through nevpt2::qd_inputs as the coupling
// builds it.
std::vector<double> density_of(QdRun &run, std::span<const double> t, const bool pc) {
  const Inputs in = inputs(run);
  fb::DeviceResources &res = run.node->device(0);
  const auto nmo = static_cast<std::size_t>(run.solution->c.n());
  std::vector<double> d(nmo * nmo);
  const fb::Status got = nevpt2::qd_inputs(
      res, in.molecule, in.basis, in.df.aux, in.h, in.df.b, *run.solution, run.ncore, run.ncas,
      [&](const nevpt2::MoBlocks &blocks,
          const std::span<const nevpt2::EnergyInputs> pairs) -> fb::Status {
        auto density = pc ? nevpt2::pc_pt2_transition_density(res, blocks, pairs, t, 0)
                          : nevpt2::sc_pt2_transition_density(res, blocks, pairs, t, 0);
        if (!density.has_value()) {
          return std::unexpected(density.error());
        }
        EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), density->data(), d.size() * sizeof(double),
                                      wwr::wwrMemcpyDeviceToHost, res.stream()),
                  wwr::wwrSuccess);
        return res.sync();
      });
  EXPECT_TRUE(got.has_value()) << got.error().message;
  return d;
}

std::size_t largest_component(std::span<const double> v) {
  return static_cast<std::size_t>(std::ranges::distance(
      v.begin(), std::ranges::max_element(v, {}, [](const double x) { return std::abs(x); })));
}

// The pair's sign against the data's phases: s_m(P) s_m(Q) from H_eff's
// element (m(P), m(Q)), ours (R diag(E) R^T) against @p heff_data
// (n x n row-major, the data's roots' basis); @p r_data the data's R, column
// P at [P n]. +1 when m(P) = m(Q).
double pair_sign(const Coupling &c, std::span<const double> heff_data,
                 std::span<const double> r_data, const std::size_t n, const std::size_t p,
                 const std::size_t q, const std::string &label) {
  const std::size_t mp = largest_component(std::span(c.eigenvectors).subspan(p * n, n));
  const std::size_t mq = largest_component(std::span(c.eigenvectors).subspan(q * n, n));
  EXPECT_EQ(mp, largest_component(r_data.subspan(p * n, n))) << label << " R_P";
  EXPECT_EQ(mq, largest_component(r_data.subspan(q * n, n))) << label << " R_Q";
  if (mp == mq) {
    return 1.0;
  }
  double ours = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    ours += c.eigenvectors[(k * n) + mp] * c.eigenvalues[k] * c.eigenvectors[(k * n) + mq];
  }
  const double theirs = heff_data[(mp * n) + mq];
  EXPECT_GT(std::abs(theirs), kPhaseElementFloor) << label << " H_eff " << mp << mq;
  EXPECT_NEAR(std::abs(ours), std::abs(theirs), kPhaseMagnitudeTolerance * std::abs(theirs))
      << label << " H_eff " << mp << mq;
  const double sign = ours * theirs < 0.0 ? -1.0 : 1.0;
  std::cout << std::format("[ nevpt2   ] qd nac {}: pair sign {:+.0f} from H_eff[{}, {}] ours "
                           "{:+.6e}, the data's {:+.6e}\n",
                           label, sign, mp, mq, ours, theirs);
  return sign;
}

std::vector<double> flat(const fb::json &nested) {
  std::vector<double> out;
  for (const fb::json &row : nested) {
    for (const fb::json &x : row) {
      out.push_back(x.get<double>());
    }
  }
  return out;
}

// One family's worst |ours - data| / bar, and where.
struct Worst {
  double ratio = 0.0;
  std::string where;
  void see(const double ratio_in, const std::string &where_in) {
    if (ratio_in > ratio) {
      ratio = ratio_in;
      where = where_in;
    }
  }
};

// One comparison against a committed (Richardson, error bar): within 2x
// the bar when @p gate (else printed only), printed, and folded into
// @p worst.
void expect_within(const std::string &label, const double ours, const double ref, const double bar,
                   Worst &worst, const bool gate = true) {
  const double ratio = std::abs(ours - ref) / bar;
  if (gate) {
    EXPECT_LE(std::abs(ours - ref), kBarFactor * bar) << label;
  }
  worst.see(ratio, label);
  std::cout << std::format("[ nevpt2   ] qd nac {}: analytic {:+.10f} numerical {:+.10f} diff "
                           "{:+.2e} ({:.2f} bars of {:.1e})\n",
                           label, ours, ref, ours - ref, ratio, bar);
}

// d's total against its formula from the returned parts, @p options',
// E_Q - E_P = +-heff_gap by the order of p and q: to 1e-10 of the largest
// part summed.
void expect_total(const nevpt2::QdNac &d, const std::int32_t p, const std::int32_t q,
                  const fb::mcscf::NacOptions &options, const std::string &label) {
  const double ediff = q > p ? d.heff_gap : -d.heff_gap;
  std::vector<double> formula(d.total.size());
  double scale = 0.0;
  for (std::size_t i = 0; i < d.total.size(); ++i) {
    const double rest = d.cas[i] + d.pt2[i] - (options.use_etfs ? d.csf[i] : 0.0);
    formula[i] =
        options.mult_ediff ? d.numerator[i] + (ediff * rest) : (d.numerator[i] / ediff) + rest;
    const double nev = options.mult_ediff ? d.numerator[i] : d.numerator[i] / ediff;
    const double factor = options.mult_ediff ? std::abs(ediff) : 1.0;
    scale = std::max({scale, std::abs(nev), factor * std::abs(d.cas[i]),
                      factor * std::abs(d.pt2[i]), factor * std::abs(d.csf[i])});
  }
  const double diff = max_diff(d.total, formula);
  EXPECT_LE(diff, kPathTolerance * scale) << label;
  std::cout << std::format("[ nevpt2   ] qd nac {} total (mult_ediff {}, use_etfs {}): against "
                           "its formula {:.2e}, largest part {:.3e}, relative {:.2e}\n",
                           label, options.mult_ediff, options.use_etfs, diff, scale, diff / scale);
}

// ---- Against #743's zeroth-order references --------------------------------

const fb::json &coupling_data(std::string_view name) {
  return reference("qd_nevpt2_nac/coupling", std::format("{}.json", name));
}

// Every pair of case @p name, SC and PC: d^NEV + d^CAS against #743's
// `{sc,pc}_zeroth`, the symmetries, the default total, the CSF split and
// the combination path.
void expect_zeroth(const std::string &name) {
  const auto start = std::chrono::steady_clock::now();
  const fb::json &data = coupling_data(name);
  ASSERT_EQ(data.at("name").get<std::string>(), name);
  QdRun run = qd_run(name);
  ASSERT_TRUE(run.solution.has_value());
  const Inputs in = inputs(run);
  fb::DeviceResources &res = run.node->device(0);
  const std::size_t n = run.solution->roots.energies.size();
  const std::size_t natom = data.at("atoms").size();
  const fb::json &directions = data.at("directions");
  const bool full = data.contains("full") && !data.at("full").is_null();
  std::cout << std::format("[ nevpt2   ] qd nac {}: CASSCF ||g|| {:.2e}\n", name,
                           run.solution->gradient_norm);

  for (const bool pc : {false, true}) {
    const std::string_view scheme = pc ? "pc" : "sc";
    const std::string_view tag = pc ? "PC" : "SC";
    const auto heff_data =
        data.at("reference").at(std::format("{}_heff", scheme)).get<std::vector<double>>();
    const auto r_data =
        data.at("reference").at(std::format("{}_heff_vectors", scheme)).get<std::vector<double>>();
    ASSERT_EQ(heff_data.size(), n * n);
    Worst worst;
    double worst_symmetry_h = 0.0;
    double worst_symmetry_d = 0.0;
    double worst_split = 0.0;
    double worst_path = 0.0;
    const fb::json &pairs = data.at("pairs");
    for (std::size_t pq = 0; pq < pairs.size(); ++pq) {
      const auto p = pairs.at(pq).at(0).get<std::int32_t>();
      const auto q = pairs.at(pq).at(1).get<std::int32_t>();
      const auto up = static_cast<std::size_t>(p);
      const auto uq = static_cast<std::size_t>(q);
      const std::string label = std::format("{} {} ({}, {})", tag, name, p, q);
      const auto begin = std::chrono::steady_clock::now();
      const auto d = nac_of(run, p, q, pc);
      ASSERT_TRUE(d.has_value()) << d.error().message;
      const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
      ASSERT_EQ(d->total.size(), 3 * natom);
      std::cout << std::format("[ nevpt2   ] qd nac {} ({:.2f} s): H_eff gap {:.4e}, root gap "
                               "{:.4e}, max |zeta| {:.2e}, Z-vectors {} iterations, residuals "
                               "{::.2e}\n",
                               label, seconds.count(), d->heff_gap, d->root_gap, d->max_zeta,
                               d->iterations, d->residual);

      // The coupling the test reads R and the h scale from, and the sign.
      const std::optional<Coupling> c = coupling_of(run, p, q, pc);
      ASSERT_TRUE(c.has_value());
      const double ediff = c->eigenvalues[uq] - c->eigenvalues[up];
      // Two calls: the same gap to the path-agreement bar.
      EXPECT_NEAR(d->heff_gap, std::abs(ediff), kPathTolerance * std::abs(ediff)) << label;
      const double sign = pair_sign(c.value(), heff_data, r_data, n, up, uq, label);

      // d^NEV + d^CAS against `zeroth`.
      std::vector<double> zeroth(d->total.size());
      for (std::size_t i = 0; i < zeroth.size(); ++i) {
        zeroth[i] = sign * ((d->numerator[i] / ediff) + d->cas[i]);
      }
      const fb::json &directional = data.at("directional").at(std::format("{}_zeroth", scheme));
      for (std::size_t k = 0; k < directions.size(); ++k) {
        expect_within(std::format("{} zeroth u{}", label, k),
                      dot(zeroth, directions.at(k).get<std::vector<double>>()),
                      directional.at(pq).at("richardson").at(k).get<double>(),
                      directional.at(pq).at("error_bar").at(k).get<double>(), worst);
      }
      if (full) {
        const fb::json &f = data.at("full").at(std::format("{}_zeroth", scheme)).at(pq);
        for (std::size_t a = 0; a < natom; ++a) {
          for (std::size_t x = 0; x < 3; ++x) {
            expect_within(std::format("{} zeroth {}{} {}", label,
                                      data.at("atoms").at(a).at("symbol").get<std::string>(), a,
                                      "xyz"[x]),
                          zeroth[(3 * a) + x], f.at("richardson").at(a).at(x).get<double>(),
                          f.at("error_bar").at(a).at(x).get<double>(), worst);
          }
        }
      }

      // The swapped pair: h_QP = h_PQ and d_QP = -d_PQ.
      const auto swapped = nac_of(run, q, p, pc);
      ASSERT_TRUE(swapped.has_value()) << swapped.error().message;
      const double h_diff = max_diff(swapped->numerator, d->numerator);
      const double h_scale = c->summand;
      EXPECT_LE(h_diff, kSolveTolerance * h_scale) << label << " h_QP";
      std::vector<double> minus(swapped->total.size());
      std::ranges::transform(swapped->total, minus.begin(), std::negate<>{});
      const double d_diff = max_diff(minus, d->total);

      // The combination at (T^PQ, D^PT2), one Z-vector: the CSF split and
      // cas + pt2.
      const std::vector<double> t = t_of(c->eigenvectors, n, up, uq);
      const std::vector<double> dx = density_of(run, t, pc);
      const auto combined =
          fb::mcscf::df_nac(res, in.molecule, in.basis, in.classes, in.df.aux, in.h, in.df.b,
                            in.df.l.data(), *run.solution, run.ncore, run.ncas, t, dx);
      ASSERT_TRUE(combined.has_value()) << combined.error().message;
      std::vector<double> split(d->csf.size());
      std::vector<double> sum(d->csf.size());
      for (std::size_t i = 0; i < split.size(); ++i) {
        split[i] = combined->csf[i] + combined->basis_motion[i];
        sum[i] = d->cas[i] + d->pt2[i];
      }
      const double split_diff = max_diff(d->csf, split);
      const double split_scale = std::max(largest_abs(d->csf), largest_abs(split));
      EXPECT_LE(split_diff, kPathTolerance * split_scale) << label << " CSF split";
      const double path_diff = max_diff(sum, combined->total);
      const double path_scale = std::max(largest_abs(sum), largest_abs(combined->total));
      EXPECT_LE(path_diff, kSolveTolerance * path_scale) << label << " cas + pt2";
      const double part_scale =
          std::max({largest_abs(combined->response), largest_abs(combined->csf),
                    largest_abs(combined->basis_motion)});

      // d_QP = -d_PQ, scaled by the largest part summed into d: h's summands
      // over the gap, and the combination's three parts.
      const double d_scale =
          std::max({h_scale / std::abs(ediff), part_scale, largest_abs(d->cas), largest_abs(d->pt2),
                    largest_abs(d->numerator) / std::abs(ediff)});
      EXPECT_LE(d_diff, kSolveTolerance * d_scale) << label << " d_QP";
      std::cout << std::format(
          "[ nevpt2   ] qd nac {}: |h_QP - h_PQ| {:.2e}, largest summand {:.3e}, relative {:.2e} "
          "(of max |h| {:.3e}: {:.2e}); |d_QP + d_PQ| {:.2e}, largest part {:.3e}, relative "
          "{:.2e} (of max |d| {:.3e}: {:.2e}); CSF split {:.2e} relative {:.2e}; cas + pt2 vs "
          "the combination {:.2e} relative {:.2e} (of its largest part {:.3e}: {:.2e}); max "
          "|cas| {:.3e}, max |pt2| {:.3e}, max |csf| {:.3e}\n",
          label, h_diff, h_scale, h_diff / h_scale, largest_abs(d->numerator),
          h_diff / largest_abs(d->numerator), d_diff, d_scale, d_diff / d_scale,
          largest_abs(d->total), d_diff / largest_abs(d->total), split_diff,
          split_diff / split_scale, path_diff, path_diff / path_scale, part_scale,
          path_diff / part_scale, largest_abs(d->cas), largest_abs(d->pt2), largest_abs(d->csf));
      worst_symmetry_h = std::max(worst_symmetry_h, h_diff / h_scale);
      worst_symmetry_d = std::max(worst_symmetry_d, d_diff / d_scale);
      worst_split = std::max(worst_split, split_diff / split_scale);
      worst_path = std::max(worst_path, path_diff / path_scale);

      expect_total(*d, p, q, {}, label);
    }
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    std::cout << std::format("[ nevpt2   ] qd nac {} {} against #743: worst {:.3f} bars ({}); "
                             "h_QP {:.2e}, d_QP {:.2e}, CSF split {:.2e}, cas + pt2 {:.2e}; "
                             "{:.2f} s\n",
                             tag, name, worst.ratio, worst.where, worst_symmetry_h,
                             worst_symmetry_d, worst_split, worst_path, seconds.count());
  }
}

class Nevpt2QdNacZeroth : public ::testing::TestWithParam<std::string_view> {};

TEST_P(Nevpt2QdNacZeroth, MatchesNumerical) {
  FBF_REQUIRE_GPU();
  expect_zeroth(std::string{GetParam()});
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdNacZeroth,
                         ::testing::Values(std::string_view{"h8_ring_sto3g_sa3"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3_weighted"}),
                         [](const auto &info) { return std::string(info.param); });

// ---- Against #744's brute force --------------------------------------------

const fb::json &brute_force_data(std::string_view name) {
  return reference("qd_nevpt2_nac/brute_force", std::format("{}.json", name));
}

// Every pair of case @p name, SC and PC: the whole d_PQ (`total`) and d^PT2
// (`pt2`) against #744's, directions and (the ring) every Cartesian
// component.
void expect_brute_force(const std::string &name) {
  const auto start = std::chrono::steady_clock::now();
  const fb::json &data = brute_force_data(name);
  ASSERT_EQ(data.at("name").get<std::string>(), name);
  QdRun run = qd_run(name);
  ASSERT_TRUE(run.solution.has_value());
  const std::size_t n = run.solution->roots.energies.size();
  const std::size_t natom = data.at("atoms").size();
  const fb::json &directions = data.at("directions");
  const bool full = !data.at("full").is_null();
  std::cout << std::format("[ nevpt2   ] qd nac brute force {}: CASSCF ||g|| {:.2e} (#744's "
                           "polished {:.1e})\n",
                           name, run.solution->gradient_norm,
                           data.at("reference").at("gradient_norm").get<double>());

  for (const bool pc : {false, true}) {
    const std::string_view scheme = pc ? "pc" : "sc";
    const std::string_view tag = pc ? "PC" : "SC";
    const std::vector<double> heff_data =
        flat(data.at("reference").at(std::format("{}_heff", scheme)));
    // #744's R is [M][P]; column P at [P n] for pair_sign.
    const std::vector<double> rows =
        flat(data.at("reference").at(std::format("{}_vectors", scheme)));
    std::vector<double> r_data(n * n);
    for (std::size_t m = 0; m < n; ++m) {
      for (std::size_t k = 0; k < n; ++k) {
        r_data[(k * n) + m] = rows[(m * n) + k];
      }
    }
    std::map<std::string, Worst, std::less<>> worst;
    const fb::json &pairs = data.at("pairs");
    for (std::size_t pq = 0; pq < pairs.size(); ++pq) {
      const auto p = pairs.at(pq).at(0).get<std::int32_t>();
      const auto q = pairs.at(pq).at(1).get<std::int32_t>();
      const auto up = static_cast<std::size_t>(p);
      const auto uq = static_cast<std::size_t>(q);
      const std::string label = std::format("{} {} ({}, {})", tag, name, p, q);
      const auto begin = std::chrono::steady_clock::now();
      const auto d = nac_of(run, p, q, pc);
      ASSERT_TRUE(d.has_value()) << d.error().message;
      const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
      ASSERT_EQ(d->total.size(), 3 * natom);
      std::cout << std::format("[ nevpt2   ] qd nac {} ({:.2f} s): H_eff gap {:.4e}, root gap "
                               "{:.4e}, max |zeta| {:.2e}, Z-vectors {} iterations, residuals "
                               "{::.2e}\n",
                               label, seconds.count(), d->heff_gap, d->root_gap, d->max_zeta,
                               d->iterations, d->residual);
      const std::optional<Coupling> c = coupling_of(run, p, q, pc);
      ASSERT_TRUE(c.has_value());
      const double ediff = c->eigenvalues[uq] - c->eigenvalues[up];
      const double sign = pair_sign(c.value(), heff_data, r_data, n, up, uq, label);
      std::vector<double> zeroth(d->total.size());
      for (std::size_t i = 0; i < zeroth.size(); ++i) {
        zeroth[i] = (d->numerator[i] / ediff) + d->cas[i];
      }
      for (const auto &[term, values] :
           {std::pair{std::string_view{"total"}, &d->total},
            std::pair{std::string_view{"pt2"}, &d->pt2},
            std::pair{std::string_view{"zeroth"},
                      static_cast<const std::vector<double> *>(&zeroth)}}) {
        std::vector<double> v = *values;
        for (double &x : v) {
          x *= sign;
        }
        Worst &w = worst[std::string{term}];
        // #744's zeroth is #743's family, gated there; here it is printed.
        const bool gate = term != "zeroth";
        const fb::json &directional = data.at("directional").at(scheme).at(term);
        for (std::size_t k = 0; k < directions.size(); ++k) {
          expect_within(std::format("{} {} u{}", label, term, k),
                        dot(v, directions.at(k).get<std::vector<double>>()),
                        directional.at(k).at("richardson").at(pq).get<double>(),
                        directional.at(k).at("error_bar").at(pq).get<double>(), w, gate);
        }
        if (full) {
          const fb::json &f = data.at("full").at(scheme).at(term);
          for (std::size_t comp = 0; comp < 3 * natom; ++comp) {
            expect_within(std::format("{} {} {}{} {}", label, term,
                                      data.at("atoms").at(comp / 3).at("symbol").get<std::string>(),
                                      comp / 3, "xyz"[comp % 3]),
                          v[comp], f.at(comp).at("richardson").at(pq).get<double>(),
                          f.at(comp).at("error_bar").at(pq).get<double>(), w, gate);
          }
        }
      }
      expect_total(*d, p, q, {}, label);
    }
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    std::cout << std::format("[ nevpt2   ] qd nac {} {} against #744: total worst {:.3f} bars "
                             "({}); pt2 worst {:.3f} bars ({}); zeroth worst {:.3f} bars ({}); "
                             "{:.2f} s\n",
                             tag, name, worst["total"].ratio, worst["total"].where,
                             worst["pt2"].ratio, worst["pt2"].where, worst["zeroth"].ratio,
                             worst["zeroth"].where, seconds.count());
  }
}

class Nevpt2QdNacBruteForce : public ::testing::TestWithParam<std::string_view> {};

TEST_P(Nevpt2QdNacBruteForce, MatchesNumerical) {
  FBF_REQUIRE_GPU();
  expect_brute_force(std::string{GetParam()});
}

INSTANTIATE_TEST_SUITE_P(Cases, Nevpt2QdNacBruteForce,
                         ::testing::Values(std::string_view{"h6_631g_sa2"},
                                           std::string_view{"h8_sto3g_sa3"},
                                           std::string_view{"h8_ring_sto3g_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

// ---- The options, and the refusals -----------------------------------------

// total against its formula at every NacOptions setting, SC and PC, on
// h6_631g_sa2's one pair.
TEST(Nevpt2QdNac, TotalFollowsTheOptions) {
  FBF_REQUIRE_GPU();
  QdRun run = qd_run("h6_631g_sa2");
  ASSERT_TRUE(run.solution.has_value());
  for (const bool pc : {false, true}) {
    for (const bool mult_ediff : {false, true}) {
      for (const bool use_etfs : {false, true}) {
        const fb::mcscf::NacOptions options{.mult_ediff = mult_ediff, .use_etfs = use_etfs};
        for (const auto &[p, q] : {std::pair{0, 1}, std::pair{1, 0}}) {
          const auto d = nac_of(run, p, q, pc, options);
          ASSERT_TRUE(d.has_value()) << d.error().message;
          expect_total(*d, p, q, options,
                       std::format("{} h6_631g_sa2 ({}, {})", pc ? "PC" : "SC", p, q));
        }
      }
    }
  }
}

// On h2co_ccpvdz_sa2: QD states made degenerate (root 1's CASSCF energy
// moved so that H_eff's diagonal is equal; its off-diagonal is zero by
// symmetry) are the interstate coupling's Numerical refusal; and two CASSCF
// roots made equal, which T^PQ couples (H_eff's eigenvalues still apart by
// their correlation), are refused naming the pair, by the first call that
// divides by their gap.
TEST(Nevpt2QdNac, Refuses) {
  FBF_REQUIRE_GPU();
  QdRun run = qd_run("h2co_ccpvdz_sa2");
  ASSERT_TRUE(run.solution.has_value());
  fb::mcscf::Solution &solution = *run.solution;
  const std::vector<double> energies = solution.roots.energies;
  const auto refused = [&](const bool pc, const std::string_view who,
                           const std::string_view phrase) {
    const auto got = nac_of(run, 0, 1, pc);
    ASSERT_FALSE(got.has_value());
    EXPECT_EQ(got.error().kind, fb::ErrorKind::Numerical) << got.error().message;
    EXPECT_NE(got.error().message.find(who), std::string::npos) << got.error().message;
    EXPECT_NE(got.error().message.find(phrase), std::string::npos) << got.error().message;
    std::cout << std::format("[ nevpt2   ] qd nac {} refuses: {}\n", pc ? "PC" : "SC",
                             got.error().message);
  };
  for (const bool pc : {false, true}) {
    const std::optional<Coupling> c = coupling_of(run, 0, 1, pc);
    ASSERT_TRUE(c.has_value());
    // H_eff's diagonal correlation: lambda - E on each (diagonal) state.
    const std::size_t m0 = largest_component(std::span(c->eigenvectors).subspan(0, 2));
    const double corr0 = c->eigenvalues[0] - energies[m0];
    const double corr1 = c->eigenvalues[1] - energies[1 - m0];
    solution.roots.energies[1 - m0] = energies[m0] + corr0 - corr1;
    refused(pc, pc ? "nevpt2::qd_pc_coupling" : "nevpt2::qd_sc_coupling", "QD states 0 and 1");
    solution.roots.energies = energies;
    solution.roots.energies[1] = energies[0];
    // The interstate coupling's Lagrangian refuses the pair first
    // (mcscf::root_pair_multipliers, its rotation multiplier divides by
    // the same gap), before df_nac's own refusal is reached.
    refused(pc, "mcscf::", "roots 0 and 1 are");
    solution.roots.energies = energies;
  }
}

} // namespace
