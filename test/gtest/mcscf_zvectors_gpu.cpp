// mcscf::zvectors on the device: several right-hand sides at once, by each
// CpSolver, against mcscf::zvector on each right-hand side alone. Over a
// state-specific case and a state average of data/casscf_gradient/
// (casscf_gradient.json names the molecule, basis, fitting set, active space
// and weights; its multipliers are not read): the DF-CASSCF by mcscf::solve
// from the Casci step's start, the CI re-solved at its orbitals, and H~ there.
// Each column is an orbital right-hand side (root 0's gradient, then random
// ones) and a random CI one, y_r = P H_y c_r for a random spin-free H_y (so in
// the roots' spin, as zvector's own CI test makes it).
//
// The setup is mcscf_zvector_gpu.cpp's, so this executable links pipeline;
// mcscf itself does not. Runs the device chain, so needs a GPU.

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

// The right-hand sides solved at once.
constexpr std::int32_t kColumns = 3;
// Each column's solve, and zvector's on it alone.
constexpr double kResidualTolerance = 1e-10;
// The CASSCF's convergence, as mcscf_zvector_gpu.cpp's.
constexpr double kConvTolGrad = 1e-10;
// Each column against zvector's on it alone, relative to the column's
// largest |z_o| and |z_c|. Schur is compared over the same H~: the same
// conjugate gradients, the CI responses' run in lockstep and in another
// summation order. Coupled solves the equations with no CI response inside,
// and H~'s products are only good to that response's tolerance (1e-8 of its
// right-hand side by default, which leaves zvector's own true residual near
// 1e-9 for a CI right-hand side of order 1): so it is compared with zvector
// over H~ with the responses solved to kExactResponseTolerance.
constexpr double kAgreement = 1e-10;
constexpr double kExactResponseTolerance = 1e-12;
// z_c is projected off every root: max |<z_c,r|c_s>|.
constexpr double kCiOverlapTolerance = 1e-10;

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                              "casscf_gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_gradient/casscf_gradient.json";
  static const fb::json kNone;
  return kNone;
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

// The case's molecule through the real registry to the Scf step's inputs.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_zvectors");
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

// The active-space Hamiltonian of ints in the row-major layout the sigma
// takes.
void set_sigma_integrals(fb::DeviceResources &res, const fb::CasscfIntegrals &ints, std::size_t n,
                         fb::ci::Sigma &sigma) {
  const std::vector<double> h_col = download(res, ints.hamiltonian.h_eff.data(), n * n);
  const std::vector<double> eri_col = download(res, ints.hamiltonian.eri.data(), n * n * n * n);
  std::vector<double> h_eff(n * n);
  std::vector<double> eri(n * n * n * n);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u < n; ++u) {
      h_eff[(t * n) + u] = h_col[t + (n * u)];
      for (std::size_t v = 0; v < n; ++v) {
        for (std::size_t w = 0; w < n; ++w) {
          eri[(((((t * n) + u) * n) + v) * n) + w] = eri_col[t + (n * (u + (n * (v + (n * w)))))];
        }
      }
    }
  }
  EXPECT_TRUE(sigma.set_integrals(res, ints.hamiltonian.e_core, h_eff, eri).has_value());
}

// A random spin-free active-space Hamiltonian, h1 and (tu|vw) uniform in
// [-1, 1) given their symmetries, row-major.
void random_hamiltonian(std::mt19937_64 &engine, std::size_t n, std::vector<double> &h1,
                        std::vector<double> &eri) {
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  const auto at = [n](std::size_t t, std::size_t u, std::size_t v, std::size_t w) {
    return (((((t * n) + u) * n) + v) * n) + w;
  };
  h1.assign(n * n, 0.0);
  eri.assign(n * n * n * n, 0.0);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u <= t; ++u) {
      h1[(t * n) + u] = h1[(u * n) + t] = uniform(engine);
    }
  }
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t u = 0; u <= t; ++u) {
      for (std::size_t v = 0; v < n; ++v) {
        for (std::size_t w = 0; w <= v; ++w) {
          if ((t * n) + u < (v * n) + w) {
            continue;
          }
          const double value = uniform(engine);
          for (const auto [p, q, r, s] :
               {std::array{t, u, v, w}, std::array{u, t, v, w}, std::array{t, u, w, v},
                std::array{u, t, w, v}, std::array{v, w, t, u}, std::array{w, v, t, u},
                std::array{v, w, u, t}, std::array{w, v, u, t}}) {
            eri[at(p, q, r, s)] = value;
          }
        }
      }
    }
  }
}

class McscfZvectors : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfZvectors, EachColumnIsZvectorsAlone) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam());
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());
  const std::size_t roots_count = weights.size();

  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};

  // The SCF, B over the case's fitting set, the start and the DF-CASSCF.
  std::optional<fb::Calculation> calc = run_setup_steps(entry, node);
  ASSERT_TRUE(calc.has_value());
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis_set = calc->get<fb::BasisSetOutput>();
  const auto &transform = calc->get<fb::SphericalTransformOutput>();
  const auto &pairs_out = calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &x = calc->get<fb::OrthogonalizerOutput>();
  const auto &h = calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = calc->get<fb::GuessOutput>();
  ASSERT_TRUE(molecule.has_value() && basis_set.has_value() && transform.has_value() &&
              pairs_out.has_value() && overlap.has_value() && x.has_value() && h.has_value() &&
              guess.has_value());
  const fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput> view{
      *molecule, *basis_set, *pairs_out};
  auto eri = fb::EriStep{}.run(view, res);
  ASSERT_TRUE(eri.has_value()) << eri.error().message;
  auto scf = fb::test::kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform, *eri, *overlap, *x, *h, *guess},
      res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                            .aux_basis = entry.at("aux_basis").at("name").get<std::string>()}
                    .run(view, res);
  ASSERT_TRUE(fitted.has_value()) << fitted.error().message;
  const auto df = std::get<fb::DensityFitted>(std::move(fitted->eri));
  fb::CasciStep casci;
  casci.ncas = entry.at("ncas").get<std::int32_t>();
  casci.nelecas = entry.at("nelecas").get<std::int32_t>();
  if (!entry.at("active").is_null()) {
    casci.active = entry.at("active").get<std::vector<std::int32_t>>();
  }
  casci.nroots = nroots;
  auto start = casci.select(res, molecule->molecule, basis_set->basis, *scf);
  ASSERT_TRUE(start.has_value()) << start.error().message;
  const std::int32_t ncore = start->ncore;
  const std::int32_t ncas = start->ncas;
  const std::int32_t nelecas = start->nelecas;
  const fb::BasisSet &basis = basis_set->basis;
  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol = fb::mcscf::solve(res, basis, molecule->molecule, h->h, transform->u, {}, {}, start->c,
                              ncore, ncas, nelecas, nroots, weights, options, &df.b, df.naux);
  ASSERT_TRUE(sol.has_value()) << sol.error().message;
  const fb::SphericalMatrixAO &c = sol->c;
  const auto mo = static_cast<std::size_t>(basis.nao_sphe());
  const auto core = static_cast<std::size_t>(ncore);
  const auto n = static_cast<std::size_t>(ncas);

  // At the solution's orbitals: the CI re-solved, the SA RDMs, H and H~.
  auto ints =
      fb::casscf_integrals(res, basis, molecule->molecule, h->h, df.b, df.naux, c, ncore, ncas);
  ASSERT_TRUE(ints.has_value()) << ints.error().message;
  const std::int32_t n_beta = (nelecas - (molecule->molecule.multiplicity() - 1)) / 2;
  auto alpha = fb::ci::DeviceStringSet::create(res, ncas, nelecas - n_beta);
  auto beta = fb::ci::DeviceStringSet::create(res, ncas, n_beta);
  ASSERT_TRUE(alpha.has_value() && beta.has_value());
  auto sigma = fb::ci::Sigma::create(res, *alpha, *beta);
  auto response_sigma = fb::ci::Sigma::create(res, *alpha, *beta);
  auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  ASSERT_TRUE(sigma.has_value() && response_sigma.has_value() && rdm.has_value());
  set_sigma_integrals(res, *ints, n, *sigma);
  const auto dim = static_cast<std::size_t>(sigma->size());
  const std::size_t ci_block = dim * roots_count;
  fb::Buffers::device<double> diagonal(dim, res.device());
  ASSERT_TRUE(sigma->diagonal(res, diagonal.data()).has_value());
  auto roots = fb::ci::solve(res, *sigma, *alpha, *beta, diagonal.data(), nroots);
  ASSERT_TRUE(roots.has_value()) << roots.error().message;
  const std::vector<double> c_host = download(res, roots->vectors.data(), ci_block);

  fb::Buffers::device<double> gamma_device(n * n, res.device());
  fb::Buffers::device<double> gamma2_device(n * n * n * n, res.device());
  const auto rdms = [&](std::span<const double> w) {
    EXPECT_TRUE(rdm->rdm12(res, w, roots->vectors.data(), gamma_device.data(), gamma2_device.data())
                    .has_value());
    return std::pair{download(res, gamma_device.data(), n * n),
                     download(res, gamma2_device.data(), n * n * n * n)};
  };
  const auto [gamma, gamma2] = rdms(weights);
  auto sa_fock = fb::response::fock(res, molecule->molecule, h->h, df.b, df.naux, c, ncore, ncas,
                                    ints->ppaa, ints->papa, gamma, gamma2);
  ASSERT_TRUE(sa_fock.has_value()) << sa_fock.error().message;
  auto hessian = fb::response::Hessian::create(res, h->h, df.b, df.naux, c, ncore, ncas, ints->ppaa,
                                               ints->papa, *sa_fock);
  ASSERT_TRUE(hessian.has_value()) << hessian.error().message;
  const fb::mcscf::ResponseIntegrals response_ints{
      mo,
      core,
      n,
      download(res, sa_fock->fock_inactive.data(), mo * mo),
      download(res, ints->ppaa.data(), mo * mo * n * n),
      download(res, ints->papa.data(), mo * mo * n * n)};
  // H~ as the tree builds it, and with its CI responses solved tightly.
  auto relaxed = fb::mcscf::RelaxedHessian::create(res, *hessian, *sigma, *rdm, *alpha, *beta,
                                                   *roots, weights, response_ints);
  ASSERT_TRUE(relaxed.has_value()) << relaxed.error().message;
  auto exact = fb::mcscf::RelaxedHessian::create(res, *hessian, *sigma, *rdm, *alpha, *beta, *roots,
                                                 weights, response_ints, kExactResponseTolerance);
  ASSERT_TRUE(exact.has_value()) << exact.error().message;
  const auto pairs = static_cast<std::size_t>(relaxed->size());
  const auto columns = static_cast<std::size_t>(kColumns);

  // The right-hand sides: root 0's gradient, then uniform ones of its size;
  // and per column y = P H_y c.
  std::vector<double> one_hot(roots_count, 0.0);
  one_hot[0] = 1.0;
  const auto [gamma_i, gamma2_i] = rdms(one_hot);
  auto root_fock = fb::response::fock(res, molecule->molecule, h->h, df.b, df.naux, c, ncore, ncas,
                                      ints->ppaa, ints->papa, gamma_i, gamma2_i);
  ASSERT_TRUE(root_fock.has_value()) << root_fock.error().message;
  const std::vector<double> g0 = download(res, root_fock->gradient.data(), pairs);
  const double g_scale = std::ranges::max(g0, {}, [](double v) { return std::abs(v); });
  std::mt19937_64 engine(611);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<double> gradients(pairs * columns);
  std::ranges::copy(g0, gradients.begin());
  std::generate(gradients.begin() + static_cast<std::ptrdiff_t>(pairs), gradients.end(),
                [&] { return g_scale * uniform(engine); });
  std::vector<double> y(ci_block * columns);
  fb::Buffers::device<double> sigma_out(ci_block, res.device());
  for (std::size_t col = 0; col < columns; ++col) {
    std::vector<double> h1;
    std::vector<double> eri1;
    random_hamiltonian(engine, n, h1, eri1);
    ASSERT_TRUE(response_sigma->set_integrals(res, 0.0, h1, eri1).has_value());
    ASSERT_TRUE(
        response_sigma->apply(res, nroots, roots->vectors.data(), sigma_out.data()).has_value());
    std::vector<double> block = download(res, sigma_out.data(), ci_block);
    for (std::size_t r = 0; r < roots_count; ++r) {
      for (std::size_t s = 0; s < roots_count; ++s) {
        double dot = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
          dot += block[(r * dim) + i] * c_host[(s * dim) + i];
        }
        for (std::size_t i = 0; i < dim; ++i) {
          block[(r * dim) + i] -= dot * c_host[(s * dim) + i];
        }
      }
    }
    std::ranges::copy(block, y.begin() + static_cast<std::ptrdiff_t>(col * ci_block));
  }
  fb::Buffers::device<double> gradients_device(gradients.size(), res.device());
  fb::Buffers::device<double> y_device(y.size(), res.device());
  upload(res, gradients, gradients_device.data());
  upload(res, y, y_device.data());

  // No columns is refused.
  {
    auto none = fb::mcscf::zvectors(res, *relaxed, *hessian, *roots, 0, gradients_device.data());
    ASSERT_FALSE(none.has_value());
    EXPECT_EQ(none.error().kind, fb::ErrorKind::InvalidConfig);
  }

  // zvector on each column alone, over H~ as built and over the exact one.
  const auto alone = [&](fb::mcscf::RelaxedHessian &over) {
    std::pair<std::vector<double>, std::vector<double>> want{
        std::vector<double>(pairs * columns), std::vector<double>(ci_block * columns)};
    for (std::size_t col = 0; col < columns; ++col) {
      auto z =
          fb::mcscf::zvector(res, over, *hessian, *roots, gradients_device.data() + (col * pairs),
                             y_device.data() + (col * ci_block), kResidualTolerance);
      EXPECT_TRUE(z.has_value()) << z.error().message;
      if (!z.has_value()) {
        break;
      }
      std::ranges::copy(download(res, z->orbital.data(), pairs),
                        want.first.begin() + static_cast<std::ptrdiff_t>(col * pairs));
      std::ranges::copy(download(res, z->ci.data(), ci_block),
                        want.second.begin() + static_cast<std::ptrdiff_t>(col * ci_block));
    }
    return want;
  };
  const auto as_built = alone(*relaxed);
  const auto tight = alone(*exact);

  for (const auto [solver, name, want] :
       {std::tuple{fb::mcscf::CpSolver::Schur, "Schur", &as_built},
        std::tuple{fb::mcscf::CpSolver::Coupled, "Coupled", &tight}}) {
    SCOPED_TRACE(name);
    const std::vector<double> &want_o = want->first;
    const std::vector<double> &want_c = want->second;
    const auto begin = std::chrono::steady_clock::now();
    auto z = fb::mcscf::zvectors(res, *relaxed, *hessian, *roots, kColumns, gradients_device.data(),
                                 y_device.data(), solver, kResidualTolerance);
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
    ASSERT_TRUE(z.has_value()) << z.error().message;
    ASSERT_EQ(z->iterations.size(), columns);
    ASSERT_EQ(z->residuals.size(), columns);
    const std::vector<double> got_o = download(res, z->orbital.data(), pairs * columns);
    const std::vector<double> got_c = download(res, z->ci.data(), ci_block * columns);
    for (std::size_t col = 0; col < columns; ++col) {
      double orbital_error = 0.0;
      double orbital_scale = 0.0;
      for (std::size_t k = 0; k < pairs; ++k) {
        const std::size_t at = (col * pairs) + k;
        orbital_error = std::max(orbital_error, std::abs(got_o[at] - want_o[at]));
        orbital_scale = std::max(orbital_scale, std::abs(want_o[at]));
      }
      double ci_error = 0.0;
      double ci_scale = 0.0;
      double ci_overlap = 0.0;
      for (std::size_t r = 0; r < roots_count; ++r) {
        const std::size_t first = (col * ci_block) + (r * dim);
        for (std::size_t i = 0; i < dim; ++i) {
          ci_error = std::max(ci_error, std::abs(got_c[first + i] - want_c[first + i]));
          ci_scale = std::max(ci_scale, std::abs(want_c[first + i]));
        }
        for (std::size_t s = 0; s < roots_count; ++s) {
          double dot = 0.0;
          for (std::size_t i = 0; i < dim; ++i) {
            dot += got_c[first + i] * c_host[(s * dim) + i];
          }
          ci_overlap = std::max(ci_overlap, std::abs(dot));
        }
      }
      EXPECT_LT(orbital_error, kAgreement * orbital_scale) << "column " << col;
      EXPECT_LT(ci_error, kAgreement * ci_scale) << "column " << col;
      EXPECT_LT(ci_overlap, kCiOverlapTolerance) << "column " << col;
      std::cout << std::format("[ zvectors ] {} {} column {}: {} products, residual {:.3e}, max "
                               "|dz_o| {:.3e} (max |z_o| {:.3e}), max |dz_c| {:.3e} (max |z_c| "
                               "{:.3e}), max |<z_c|c>| {:.3e}\n",
                               entry.at("name").get<std::string>(), name, col, z->iterations[col],
                               z->residuals[col], orbital_error, orbital_scale, ci_error, ci_scale,
                               ci_overlap);
    }
    std::cout << std::format("[ zvectors ] {} {}: {} columns in {:.2f} s\n",
                             entry.at("name").get<std::string>(), name, kColumns, seconds.count());
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfZvectors, ::testing::Values("h2o_ccpvdz", "h2co_ccpvdz_sa2"),
                         [](const auto &info) { return std::string(info.param); });

} // namespace
