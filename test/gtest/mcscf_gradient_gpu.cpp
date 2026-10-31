// mcscf::df_gradient on the device: this tree's own DF-CASSCF
// (mcscf::solve under DensityFitting::Full over the case's fitting set, from
// the exact RHF and the case's active orbitals) converged tightly, then its
// nuclear gradient, every term PySCF records and the total, against
// df.grad.casscf in data/casscf_gradient/casscf_gradient.json
// (tools/emit/casscf_gradient_reference.py) to 1e-8 absolute, and the
// total's sum over atoms below 1e-9; for each state-specific case (H2O and the
// skewed H2O, cc-pVDZ / cc-pVDZ-JKFIT, CAS(4,4)) and, against
// df.grad.sacasscf, every root of the state-averaged one (H2CO, SA-2 CAS(4,3)),
// whose parts include the multipliers' terms. Then the caller's pieces: for
// one root, E_cas's own MO adjoint (h_bar = D, Lambda^L = Sum Gamma B^L, X its
// X, built on the host) fed as the extra adjoint doubles the electronic
// gradient to 1e-10, so the extra term alone is the state-specific one; for a
// state average, each root's multipliers from mcscf::zvector passed in
// reproduce the internal path to 1e-10. And the refusals: a root the
// solution does not have, multipliers of the wrong length, and a solution
// whose roots and RDMs disagree.
//
// The setup is mcscf_solve_gpu.cpp's: the setup steps through the real
// registry, the SCF run by hand (kExactScf), and the start picked by the Casci
// step's selection. So this executable links pipeline; mcscf does not. Runs the
// device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.mcscf;
import free_breakfast.mcscf.relaxed_hessian;
import free_breakfast.mcscf.zvector;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars: every term and the total within 1e-8 of PySCF, and the
// total's sum over atoms below 1e-9.
constexpr double kTermTolerance = 1e-8;
constexpr double kSumTolerance = 1e-9;
// The energy against PySCF's, as mcscf_solve_gpu.cpp holds the DF-CASSCF's.
constexpr double kEnergyTolerance = 1e-8;
// The caller's pieces against the call's own: the same formula over the same
// inputs, to round-off.
constexpr double kSelfTolerance = 1e-10;
// The solve's convergence. The gradient formula takes W for the orbitals'
// response, so its error is first order in the orbital gradient: ~||g|| times
// the orbitals' change per bohr, ~1e-9 here at the bar's tenth.
constexpr double kConvTolGrad = 1e-9;

const fb::json &reference_case(std::string_view name) {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                              "casscf_gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : doc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_gradient.json";
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule through the real registry to the Scf step's
// inputs, as mcscf_solve_gpu.cpp's run_setup_steps.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_gradient");
  // Bohr in the reference, Angstrom in an XYZ file.
  std::string xyz =
      std::format("{}\n{}\n", entry.at("atoms").size(), entry.at("name").get<std::string>());
  for (const fb::json &atom : entry.at("atoms")) {
    const auto coord = atom.at("coord").get<std::vector<double>>();
    xyz += std::format("{} {:.17g} {:.17g} {:.17g}\n", atom.at("symbol").get<std::string>(),
                       coord[0] * fb::kAngstromPerBohr<double>,
                       coord[1] * fb::kAngstromPerBohr<double>,
                       coord[2] * fb::kAngstromPerBohr<double>);
  }
  EXPECT_TRUE(fb::write_file(dir / "molecule.xyz", xyz).has_value());
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
                  "[orthogonalizer]\n[core_hamiltonian]\n[guess]\n",
                  entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), dir / "out");
  if (!calc.has_value()) {
    ADD_FAILURE() << calc.error().message;
    return std::nullopt;
  }
  if (const fb::Status ran = calc->run(node); !ran.has_value()) {
    ADD_FAILURE() << ran.error().message;
    return std::nullopt;
  }
  return std::move(*calc);
}

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

void upload(fb::DeviceResources &res, std::span<const double> host, double *data) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(data, host.data(), host.size_bytes(), wwr::wwrMemcpyHostToDevice,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
}

// Every term of @p got against @p want to kSelfTolerance, @p doubled: got's
// electronic terms twice want's and its nuclear repulsion want's, so got's
// total less want's is want's electronic gradient. The worst error.
double compare_terms(const std::string &what, const fb::mcscf::GradientTerms &got,
                     const fb::mcscf::GradientTerms &want, bool doubled) {
  const double scale = doubled ? 2.0 : 1.0;
  double worst = 0.0;
  for (const auto &[key, ours, theirs, factor] :
       {std::tuple{"kinetic", &got.kinetic, &want.kinetic, scale},
        {"potential", &got.potential, &want.potential, scale},
        {"overlap", &got.overlap, &want.overlap, scale},
        {"two_electron", &got.two_electron, &want.two_electron, scale},
        {"nuclear_repulsion", &got.nuclear_repulsion, &want.nuclear_repulsion, 1.0}}) {
    EXPECT_EQ(ours->size(), theirs->size()) << what << " " << key;
    for (std::size_t i = 0; i < std::min(ours->size(), theirs->size()); ++i) {
      EXPECT_NEAR((*ours)[i], factor * (*theirs)[i], kSelfTolerance) << what << " " << key << i;
      worst = std::max(worst, std::abs((*ours)[i] - (factor * (*theirs)[i])));
    }
  }
  for (std::size_t i = 0; i < std::min(got.total.size(), want.total.size()); ++i) {
    const double electronic = want.total[i] - want.nuclear_repulsion[i];
    const double expected = doubled ? want.total[i] + electronic : want.total[i];
    EXPECT_NEAR(got.total[i], expected, kSelfTolerance) << what << " total " << i;
    worst = std::max(worst, std::abs(got.total[i] - expected));
  }
  return worst;
}

// E_cas's MO adjoint at the one-root @p sol, on the host (nuclear_gradient.tex
// Eq. total, "CAS"): h_bar = D, Lambda^L_pq = Sum_rs Gamma_pqrs B^L_rs over
// the full-range Gamma (its core parts D (x) D - 1/2 D (x) D exchanged, less
// the active-active of either, plus the active Gamma), and X = 2 h h_bar +
// 2 Sum_L B^L Lambda^L, in MoAdjoint's layouts.
struct HostAdjoint {
  std::vector<double> h_bar;
  std::vector<double> lambda;
  std::vector<double> x;
};
HostAdjoint cas_adjoint(fb::DeviceResources &res, const fb::mcscf::Solution &sol,
                        const fb::SphericalMatrixAO &h, const fb::Buffers::device<double> &b,
                        std::size_t naux, std::size_t ncore, std::size_t ncas) {
  const auto n = static_cast<std::size_t>(sol.c.n());
  const std::size_t nn = n * n;
  const std::vector<double> c = download(res, sol.c.data(), nn);
  const std::vector<double> b_ao = download(res, b.data(), naux * nn);
  const std::vector<double> h_ao = download(res, h.data(), nn);
  // Column-major n x n products: a b and a^T b.
  const auto product = [n](const std::vector<double> &a, const std::vector<double> &m,
                           const bool transpose) {
    std::vector<double> out(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k) {
        const double mkj = m[k + (n * j)];
        for (std::size_t i = 0; i < n; ++i) {
          out[i + (n * j)] += (transpose ? a[k + (n * i)] : a[i + (n * k)]) * mkj;
        }
      }
    }
    return out;
  };
  const auto mo = [&](const std::vector<double> &ao) {
    return product(c, product(ao, c, false), true);
  };
  const auto dot = [](const std::vector<double> &a, const std::vector<double> &m) {
    return std::inner_product(a.begin(), a.end(), m.begin(), 0.0);
  };
  std::vector<double> d(nn, 0.0);
  std::vector<double> d_a(nn, 0.0);
  for (std::size_t i = 0; i < ncore; ++i) {
    d[i * (n + 1)] = 2.0;
  }
  for (std::size_t t = 0; t < ncas; ++t) {
    for (std::size_t u = 0; u < ncas; ++u) {
      d_a[(ncore + t) + (n * (ncore + u))] = sol.gamma[(t * ncas) + u];
      d[(ncore + t) + (n * (ncore + u))] = sol.gamma[(t * ncas) + u];
    }
  }
  HostAdjoint out{d, std::vector<double>(naux * nn), product(mo(h_ao), d, false)};
  std::vector<double> b_l(nn);
  for (std::size_t l = 0; l < naux; ++l) {
    for (std::size_t k = 0; k < nn; ++k) {
      b_l[k] = b_ao[l + (naux * k)];
    }
    const std::vector<double> b_mo = mo(b_l);
    const double j = dot(d, b_mo);
    const double j_a = dot(d_a, b_mo);
    const std::vector<double> k_full = product(d, product(b_mo, d, false), false);
    const std::vector<double> k_act = product(d_a, product(b_mo, d_a, false), false);
    std::vector<double> lambda(nn);
    for (std::size_t k = 0; k < nn; ++k) {
      lambda[k] = (d[k] * j) - (d_a[k] * j_a) - (0.5 * (k_full[k] - k_act[k]));
    }
    for (std::size_t t = 0; t < ncas; ++t) {
      for (std::size_t u = 0; u < ncas; ++u) {
        double sum = 0.0;
        for (std::size_t v = 0; v < ncas; ++v) {
          for (std::size_t w = 0; w < ncas; ++w) {
            sum += sol.gamma2[(((((t * ncas) + u) * ncas) + v) * ncas) + w] *
                   b_mo[(ncore + v) + (n * (ncore + w))];
          }
        }
        lambda[(ncore + t) + (n * (ncore + u))] += sum;
      }
    }
    const std::vector<double> bl = product(b_mo, lambda, false);
    for (std::size_t k = 0; k < nn; ++k) {
      out.lambda[l + (naux * k)] = lambda[k];
      out.x[k] += bl[k];
    }
  }
  for (double &v : out.x) {
    v *= 2.0;
  }
  return out;
}

// @p ours against the reference rows @p ref to kTermTolerance absolute; the
// worst error, for the record a PR reports.
double compare(const std::string &what, const std::vector<double> &ours, const fb::json &ref) {
  const auto rows = ref.get<std::vector<std::vector<double>>>();
  EXPECT_EQ(ours.size(), 3 * rows.size()) << what;
  if (ours.size() != 3 * rows.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double worst = 0.0;
  for (std::size_t a = 0; a < rows.size(); ++a) {
    for (std::size_t k = 0; k < 3; ++k) {
      EXPECT_NEAR(ours[(3 * a) + k], rows[a][k], kTermTolerance)
          << what << " [" << a << "][" << k << "]";
      worst = std::max(worst, std::abs(ours[(3 * a) + k] - rows[a][k]));
    }
  }
  return worst;
}

class McscfGradient : public ::testing::TestWithParam<std::string_view> {};

// The case's DF-CASSCF, its gradient against PySCF's term by term, the sum
// over atoms, and the refusals on the same solution.
TEST_P(McscfGradient, DfCasscfMatchesPyscf) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());
  const auto start = std::chrono::steady_clock::now();

  fb::NodeResources node;
  auto calc = run_setup_steps(entry, node);
  ASSERT_TRUE(calc.has_value());
  fb::DeviceResources &res = node.device(0);
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  const auto &transform = calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &x = calc->get<fb::OrthogonalizerOutput>();
  const auto &h = calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = calc->get<fb::GuessOutput>();
  ASSERT_TRUE(molecule.has_value() && basis.has_value() && transform.has_value() &&
              pairs.has_value() && overlap.has_value() && x.has_value() && h.has_value() &&
              guess.has_value())
      << "a product of the setup steps is missing";
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  ASSERT_EQ(basis->basis.nao_sphe(), entry.at("nao").get<std::size_t>());

  // The exact RHF, which the reference's start is made from too.
  auto direct = fb::EriStep{}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          *molecule, *basis, *pairs},
      res);
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  auto scf = fb::test::kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis, *transform, *direct, *overlap, *x, *h, *guess},
      res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  auto j_classes = fb::coulomb_pair_classes(res, pairs->classes);
  auto k_classes = fb::ordered_pair_classes(res, basis->basis, pairs->classes);
  ASSERT_TRUE(j_classes.has_value() && k_classes.has_value());

  fb::CasciStep step;
  step.ncas = entry.at("ncas").get<std::int32_t>();
  step.nelecas = entry.at("nelecas").get<std::int32_t>();
  if (!entry.at("active").is_null()) {
    step.active = entry.at("active").get<std::vector<std::int32_t>>();
  }
  step.nroots = nroots;
  auto space = step.select(res, molecule->molecule, basis->basis, *scf);
  ASSERT_TRUE(space.has_value()) << space.error().message;
  ASSERT_EQ(space->ncore, entry.at("ncore").get<std::int32_t>());

  // B, L and the placed aux basis over the reference's set.
  const fb::json &aux = entry.at("aux_basis");
  auto fitted =
      fb::EriStep{.mode = fb::EriMode::DensityFitting,
                  .aux_basis = aux.at("name").get<std::string>()}
          .run(
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
                  *molecule, *basis, *pairs},
              res);
  ASSERT_TRUE(fitted.has_value()) << fitted.error().message;
  const auto &df = std::get<fb::DensityFitted>(fitted->eri);
  ASSERT_EQ(df.naux, aux.at("naux").get<std::int32_t>());

  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol = fb::mcscf::solve(res, basis->basis, molecule->molecule, h->h, transform->u, *j_classes,
                              *k_classes, space->c, space->ncore, space->ncas, space->nelecas,
                              nroots, weights, options, &df.b, df.naux);
  ASSERT_TRUE(sol.has_value()) << sol.error().message;
  EXPECT_NEAR(sol->energy, entry.at("energy").get<double>(), kEnergyTolerance);

  // Every root's gradient, each term against the reference's part (for a
  // state average, the multipliers' terms included).
  for (std::int32_t root = 0; root < nroots; ++root) {
    const auto root_start = std::chrono::steady_clock::now();
    const auto g =
        fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                               df.b, df.l.data(), *sol, space->ncore, space->ncas, root);
    ASSERT_TRUE(g.has_value()) << g.error().message;
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start;
    const std::chrono::duration<double> root_seconds =
        std::chrono::steady_clock::now() - root_start;

    const fb::json &want = entry.at("roots").at(static_cast<std::size_t>(root));
    const fb::json &parts = want.at("parts");
    const std::string label = std::format("{} root {}", name, root);
    std::string report = std::format(
        "[ casgrad  ] {} ({} macro iterations, gradient {:.2f} s, {:.2f} s in all):", label,
        sol->iterations.size(), root_seconds.count(), seconds.count());
    for (const auto &[key, ours] : {std::pair{"kinetic", &g->kinetic},
                                    {"potential", &g->potential},
                                    {"overlap", &g->overlap},
                                    {"two_electron", &g->two_electron},
                                    {"nuclear_repulsion", &g->nuclear_repulsion},
                                    {"total", &g->total}}) {
      const double worst =
          compare(label + " " + key, *ours,
                  key == std::string_view{"total"} ? want.at("total") : parts.at(key));
      report += std::format(" {} {:.2e}", key, worst);
    }
    std::array<double, 3> sum{};
    for (std::size_t i = 0; i < g->total.size(); ++i) {
      sum[i % 3] += g->total[i];
    }
    const double worst_sum = std::ranges::max(sum, {}, [](double s) { return std::abs(s); });
    EXPECT_LT(std::abs(worst_sum), kSumTolerance) << label << " translational sum";
    std::cout << report << std::format(" sum {:.2e}\n", std::abs(worst_sum));
  }

  // The caller's pieces, against the call's own path.
  const auto naux = static_cast<std::size_t>(df.naux);
  const auto ncore = static_cast<std::size_t>(space->ncore);
  const auto ncas = static_cast<std::size_t>(space->ncas);
  if (nroots == 1) {
    // E_cas's own adjoint as the extra term: the gradient doubles, less the
    // nuclear repulsion.
    const auto own =
        fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                               df.b, df.l.data(), *sol, space->ncore, space->ncas, 0);
    ASSERT_TRUE(own.has_value()) << own.error().message;
    const HostAdjoint host = cas_adjoint(res, *sol, h->h, df.b, naux, ncore, ncas);
    fb::Buffers::device<double> h_bar(host.h_bar.size(), res.device());
    fb::Buffers::device<double> lambda(host.lambda.size(), res.device());
    fb::Buffers::device<double> x_mo(host.x.size(), res.device());
    upload(res, host.h_bar, h_bar.data());
    upload(res, host.lambda, lambda.data());
    upload(res, host.x, x_mo.data());
    const fb::mcscf::MoAdjoint extra{
        .h_bar = h_bar.data(), .lambda = lambda.data(), .x = x_mo.data()};
    const auto doubled = fb::mcscf::df_gradient(
        res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h, df.b, df.l.data(),
        *sol, space->ncore, space->ncas, 0, nullptr, &extra);
    ASSERT_TRUE(doubled.has_value()) << doubled.error().message;
    std::cout << std::format("[ casgrad  ] {} extra E_cas adjoint: {:.2e}\n", name,
                             compare_terms(name + " extra", *doubled, *own, true));
  } else {
    // Each root's multipliers from zvector, as the call solves them, passed
    // in: the same gradient.
    const fb::SphericalMatrixAO &c = sol->c;
    const auto mo = static_cast<std::size_t>(c.n());
    auto ints = fb::casscf_integrals(res, basis->basis, molecule->molecule, h->h, df.b, df.naux, c,
                                     space->ncore, space->ncas);
    ASSERT_TRUE(ints.has_value()) << ints.error().message;
    const std::int32_t n_beta = (space->nelecas - (molecule->molecule.multiplicity() - 1)) / 2;
    auto alpha = fb::ci::DeviceStringSet::create(res, space->ncas, space->nelecas - n_beta);
    auto beta = fb::ci::DeviceStringSet::create(res, space->ncas, n_beta);
    ASSERT_TRUE(alpha.has_value() && beta.has_value());
    auto sigma = fb::ci::Sigma::create(res, *alpha, *beta);
    auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
    ASSERT_TRUE(sigma.has_value() && rdm.has_value());
    // Row-major with no reorder, as mcscf::solve and the call set them.
    ASSERT_TRUE(
        sigma
            ->set_integrals(res, ints->hamiltonian.e_core,
                            download(res, ints->hamiltonian.h_eff.data(), ncas * ncas),
                            download(res, ints->hamiltonian.eri.data(), ncas * ncas * ncas * ncas))
            .has_value());
    auto sa_fock = fb::response::fock(res, molecule->molecule, h->h, df.b, df.naux, c, space->ncore,
                                      space->ncas, ints->ppaa, ints->papa, sol->gamma, sol->gamma2);
    ASSERT_TRUE(sa_fock.has_value()) << sa_fock.error().message;
    auto hessian = fb::response::Hessian::create(res, h->h, df.b, df.naux, c, space->ncore,
                                                 space->ncas, ints->ppaa, ints->papa, *sa_fock);
    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;
    fb::mcscf::ResponseIntegrals response_ints{
        mo,
        ncore,
        ncas,
        download(res, sa_fock->fock_inactive.data(), mo * mo),
        download(res, ints->ppaa.data(), mo * mo * ncas * ncas),
        download(res, ints->papa.data(), mo * mo * ncas * ncas)};
    auto relaxed = fb::mcscf::RelaxedHessian::create(res, *hessian, *sigma, *rdm, *alpha, *beta,
                                                     sol->roots, weights, std::move(response_ints));
    ASSERT_TRUE(relaxed.has_value()) << relaxed.error().message;
    fb::Buffers::device<double> gamma_device(ncas * ncas, res.device());
    fb::Buffers::device<double> gamma2_device(ncas * ncas * ncas * ncas, res.device());
    for (std::int32_t root = 0; root < nroots; ++root) {
      std::vector<double> one_hot(weights.size(), 0.0);
      one_hot[static_cast<std::size_t>(root)] = 1.0;
      ASSERT_TRUE(rdm->rdm12(res, one_hot, sol->roots.vectors.data(), gamma_device.data(),
                             gamma2_device.data())
                      .has_value());
      auto root_fock = fb::response::fock(
          res, molecule->molecule, h->h, df.b, df.naux, c, space->ncore, space->ncas, ints->ppaa,
          ints->papa, download(res, gamma_device.data(), ncas * ncas),
          download(res, gamma2_device.data(), ncas * ncas * ncas * ncas));
      ASSERT_TRUE(root_fock.has_value()) << root_fock.error().message;
      auto z = fb::mcscf::zvector(res, *relaxed, *hessian, sol->roots, root_fock->gradient.data());
      ASSERT_TRUE(z.has_value()) << z.error().message;
      const auto internal =
          fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux,
                                 h->h, df.b, df.l.data(), *sol, space->ncore, space->ncas, root);
      const auto given = fb::mcscf::df_gradient(res, molecule->molecule, basis->basis,
                                                pairs->classes, df.aux, h->h, df.b, df.l.data(),
                                                *sol, space->ncore, space->ncas, root, &*z);
      ASSERT_TRUE(internal.has_value() && given.has_value());
      std::cout << std::format(
          "[ casgrad  ] {} root {} given multipliers: {:.2e}\n", name, root,
          compare_terms(std::format("{} root {} given", name, root), *given, *internal, false));
    }
  }
  // Multipliers of the wrong length are InvalidConfig.
  const fb::mcscf::ZVector short_z{fb::Buffers::device<double>(1, res.device()),
                                   fb::Buffers::device<double>(1, res.device()), 0, 0.0};
  const auto wrong =
      fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                             df.b, df.l.data(), *sol, space->ncore, space->ncas, 0, &short_z);
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().kind, fb::ErrorKind::InvalidConfig) << wrong.error().message;
  EXPECT_NE(wrong.error().message.find("multipliers"), std::string::npos);

  // A root the solution does not have is InvalidConfig.
  const auto past =
      fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                             df.b, df.l.data(), *sol, space->ncore, space->ncas, nroots);
  ASSERT_FALSE(past.has_value());
  EXPECT_EQ(past.error().kind, fb::ErrorKind::InvalidConfig) << past.error().message;
  EXPECT_NE(past.error().message.find("mcscf::df_gradient"), std::string::npos);
  // So is a solution with one root more than its per-root RDMs.
  sol->roots.energies.push_back(sol->roots.energies.front());
  const auto mismatched =
      fb::mcscf::df_gradient(res, molecule->molecule, basis->basis, pairs->classes, df.aux, h->h,
                             df.b, df.l.data(), *sol, space->ncore, space->ncas, 0);
  ASSERT_FALSE(mismatched.has_value());
  EXPECT_EQ(mismatched.error().kind, fb::ErrorKind::InvalidConfig) << mismatched.error().message;
  EXPECT_NE(mismatched.error().message.find("root gamma"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfGradient,
                         ::testing::Values(std::string_view{"h2o_ccpvdz"},
                                           std::string_view{"h2o_skew_ccpvdz"},
                                           std::string_view{"h2co_ccpvdz_sa2"}),
                         [](const auto &info) { return std::string(info.param); });

} // namespace
