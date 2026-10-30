// mcscf::zvector on the device against PySCF's df.grad.sacasscf multipliers
// in data/casscf_gradient/casscf_gradient.json
// (tools/emit/casscf_gradient_reference.py), for each state-averaged case:
// the DF-CASSCF by mcscf::solve (DensityFitting::Full over the case's fitting
// set) from the RHF and the case's start, put in the reference's canonical
// orbitals (state-averaged Fock eigenvectors in the core and the virtuals,
// natural orbitals in the active space) with the CI re-solved there, as the
// tool does; then per root I the residual ||H~ z_o + g_I||, z_o and z_c
// against the reference to 1e-7, and z_c off every root.
//
// The reference's orbital signs are PySCF's, which a solve from its own start
// does not reproduce, and an orbital's sign flips every multiplier it carries
// (and, for an active one, CI coefficients): so z_o is matched by (p, q) and
// both are compared in magnitude, as the data's README allows.
//
// The setup is mcscf_solve_gpu.cpp's: the SCF (run with J unscreened) and the
// start picked by the Casci step, so this executable links pipeline; mcscf
// itself does not. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes;
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

// The issue's bar on z_o and z_c.
constexpr double kMultiplierTolerance = 1e-7;
// The solve's ||H~ z_o + g_I||: the bar a 1e-8 gradient needs, and the bar
// the test holds it to.
constexpr double kResidualTolerance = 1e-10;
// The CASSCF's convergence. z_o moves with the orbitals at first order, by
// their error ~||g|| / (H~'s least eigenvalue, above 1e-3 here): at 1e-10 that
// is ~1e-7 at worst, inside the bar for multipliers of order 1 and below.
constexpr double kConvTolGrad = 1e-10;
// The energies and the canonical orbitals' spectra against the reference.
constexpr double kEnergyTolerance = 1e-8;
constexpr double kOrbitalEnergyTolerance = 1e-6;
// z_c is projected off every root: max |<z_c,r|c_s>|.
constexpr double kCiOverlapTolerance = 1e-10;

const fb::json &reference() {
  static const fb::json kDoc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_gradient" /
                              "casscf_gradient.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return kDoc;
}

const fb::json &reference_case(std::string_view name) {
  for (const fb::json &entry : reference().at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_gradient/casscf_gradient.json";
  static const fb::json kNone;
  return kNone;
}

// Copies in res.stream()'s order, waited for.
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

// The symmetric n x n a (column-major) diagonalized on the device: the
// ascending eigenvalues, and a overwritten by its eigenvectors.
std::vector<double> eigenvectors(fb::DeviceResources &res, std::vector<double> &a, std::int32_t n) {
  const auto size = static_cast<std::size_t>(n);
  fb::Buffers::device<double> matrix(std::max<std::size_t>(size * size, 1), res.device());
  fb::Buffers::device<double> values(std::max<std::size_t>(size, 1), res.device());
  upload(res, a, matrix.data());
  EXPECT_TRUE(fb::diagonalize<double>(res, n, matrix.data(), values.data()).has_value());
  a = download(res, matrix.data(), size * size);
  return download(res, values.data(), size);
}

// The reference's molecule through the real registry to the Scf step's
// inputs. The Scf step itself is run by hand: tau_j is not a TOML key.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("mcscf_zvector");
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

// The packed index of the pair (p, q), p > q (../response/README.md).
std::size_t packed_index(std::size_t p, std::size_t q, std::size_t nmo, std::size_t ncore,
                         std::size_t nocc) {
  return q < ncore ? (q * (nmo - ncore)) + (p - ncore)
                   : (ncore * (nmo - ncore)) + ((q - ncore) * (nmo - nocc)) + (p - nocc);
}

// The active-space Hamiltonian of ints in the row-major layout the sigma
// takes (h_eff symmetric; (tu|vw) transposed index by index).
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

class McscfZvector : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfZvector, MatchesPyscfSaCasscfMultipliers) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam());
  ASSERT_TRUE(entry.contains("pairs"));
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());

  // The SCF, B over the case's fitting set, and the start.
  fb::NodeResources node;
  auto calc = run_setup_steps(entry, node);
  ASSERT_TRUE(calc.has_value());
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis_set = calc->get<fb::BasisSetOutput>();
  const auto &transform = calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &x = calc->get<fb::OrthogonalizerOutput>();
  const auto &h = calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = calc->get<fb::GuessOutput>();
  ASSERT_TRUE(molecule.has_value() && basis_set.has_value() && transform.has_value() &&
              pairs.has_value() && overlap.has_value() && x.has_value() && h.has_value() &&
              guess.has_value());
  const fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput> view{
      *molecule, *basis_set, *pairs};
  auto eri = fb::EriStep{}.run(view, res);
  ASSERT_TRUE(eri.has_value()) << eri.error().message;
  auto scf = fb::test::kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform, *eri, *overlap, *x, *h, *guess},
      res);
  ASSERT_TRUE(scf.has_value()) << scf.error().message;
  const fb::json &aux = entry.at("aux_basis");
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting,
                            .aux_basis = aux.at("name").get<std::string>()}
                    .run(view, res);
  ASSERT_TRUE(fitted.has_value()) << fitted.error().message;
  const auto df = std::get<fb::DensityFitted>(std::move(fitted->eri));
  ASSERT_EQ(df.naux, aux.at("naux").get<std::int32_t>());
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
  ASSERT_EQ(ncore, entry.at("ncore").get<std::int32_t>());
  ASSERT_EQ(ncas, entry.at("ncas").get<std::int32_t>());
  ASSERT_EQ(nelecas, entry.at("nelecas").get<std::int32_t>());

  // The DF-CASSCF, wholly fitted.
  const fb::BasisSet &basis = basis_set->basis;
  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol = fb::mcscf::solve(res, basis, molecule->molecule, h->h, transform->u, {}, {}, start->c,
                              ncore, ncas, nelecas, nroots, weights, options, &df.b, df.naux);
  ASSERT_TRUE(sol.has_value()) << sol.error().message;
  const auto energies = entry.at("energies").get<std::vector<double>>();
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_NEAR(sol->roots.energies[r], energies[r], kEnergyTolerance) << "root " << r;
  }

  // The canonical orbitals: F = F^I + F^A at the state-averaged gamma
  // diagonalized in the core and in the virtuals, ascending, and gamma in the
  // active space, occupation descending.
  const auto mo = static_cast<std::size_t>(basis.nao_sphe());
  const auto core = static_cast<std::size_t>(ncore);
  const auto n = static_cast<std::size_t>(ncas);
  const std::size_t nocc = core + n;
  std::vector<double> u(mo * mo, 0.0);
  {
    auto ints = fb::casscf_integrals(res, basis, molecule->molecule, h->h, df.b, df.naux, sol->c,
                                     ncore, ncas);
    ASSERT_TRUE(ints.has_value()) << ints.error().message;
    auto fock = fb::response::fock(res, molecule->molecule, h->h, df.b, df.naux, sol->c, ncore,
                                   ncas, ints->ppaa, ints->papa, sol->gamma, sol->gamma2);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;
    const std::vector<double> fi = download(res, fock->fock_inactive.data(), mo * mo);
    const std::vector<double> fa = download(res, fock->fock_active.data(), mo * mo);
    const auto block = [&](std::size_t first, std::size_t count,
                           const auto &element) -> std::vector<double> {
      std::vector<double> a(count * count);
      for (std::size_t j = 0; j < count; ++j) {
        for (std::size_t i = 0; i < count; ++i) {
          a[i + (count * j)] = element(first + i, first + j);
        }
      }
      std::vector<double> values = eigenvectors(res, a, static_cast<std::int32_t>(count));
      for (std::size_t j = 0; j < count; ++j) {
        for (std::size_t i = 0; i < count; ++i) {
          u[(first + i) + (mo * (first + j))] = a[i + (count * j)];
        }
      }
      return values;
    };
    const auto f = [&](std::size_t p, std::size_t q) {
      return fi[p + (mo * q)] + fa[p + (mo * q)];
    };
    const std::vector<double> core_energies = block(0, core, f);
    const std::vector<double> virtual_energies = block(nocc, mo - nocc, f);
    // -gamma ascending is gamma descending.
    std::vector<double> occupations = block(core, n, [&](std::size_t p, std::size_t q) {
      return -sol->gamma[((p - core) * n) + (q - core)];
    });
    const fb::json &orbitals = entry.at("orbitals");
    const auto want_core = orbitals.at("core_energies").get<std::vector<double>>();
    const auto want_virtual = orbitals.at("virtual_energies").get<std::vector<double>>();
    const auto want_occupations = orbitals.at("natural_occupations").get<std::vector<double>>();
    ASSERT_EQ(want_core.size(), core);
    ASSERT_EQ(want_virtual.size(), mo - nocc);
    ASSERT_EQ(want_occupations.size(), n);
    for (std::size_t i = 0; i < core; ++i) {
      EXPECT_NEAR(core_energies[i], want_core[i], kOrbitalEnergyTolerance) << "core " << i;
    }
    for (std::size_t i = 0; i < mo - nocc; ++i) {
      EXPECT_NEAR(virtual_energies[i], want_virtual[i], kOrbitalEnergyTolerance) << "virtual " << i;
    }
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(-occupations[i], want_occupations[i], kOrbitalEnergyTolerance)
          << "occupation " << i;
    }
  }
  // C <- C U on the host.
  fb::SphericalMatrixAO c(res, basis);
  {
    const std::vector<double> c_old = download(res, sol->c.data(), mo * mo);
    std::vector<double> c_new(mo * mo, 0.0);
    for (std::size_t j = 0; j < mo; ++j) {
      for (std::size_t k = 0; k < mo; ++k) {
        const double ukj = u[k + (mo * j)];
        for (std::size_t i = 0; i < mo; ++i) {
          c_new[i + (mo * j)] += c_old[i + (mo * k)] * ukj;
        }
      }
    }
    upload(res, c_new, c.data());
  }

  // At the canonical orbitals: the CI re-solved, the state-averaged RDMs, H
  // and H~.
  auto ints =
      fb::casscf_integrals(res, basis, molecule->molecule, h->h, df.b, df.naux, c, ncore, ncas);
  ASSERT_TRUE(ints.has_value()) << ints.error().message;
  // N_alpha - N_beta = 2S, the multiplicity 2S + 1.
  const std::int32_t n_beta = (nelecas - (molecule->molecule.multiplicity() - 1)) / 2;
  auto alpha = fb::ci::DeviceStringSet::create(res, ncas, nelecas - n_beta);
  auto beta = fb::ci::DeviceStringSet::create(res, ncas, n_beta);
  ASSERT_TRUE(alpha.has_value() && beta.has_value());
  auto sigma = fb::ci::Sigma::create(res, *alpha, *beta);
  auto rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  ASSERT_TRUE(sigma.has_value() && rdm.has_value());
  set_sigma_integrals(res, *ints, n, *sigma);
  const auto dim = static_cast<std::size_t>(sigma->size());
  fb::Buffers::device<double> diagonal(dim, res.device());
  ASSERT_TRUE(sigma->diagonal(res, diagonal.data()).has_value());
  auto roots = fb::ci::solve(res, *sigma, *alpha, *beta, diagonal.data(), nroots);
  ASSERT_TRUE(roots.has_value()) << roots.error().message;
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_NEAR(roots->energies[r], energies[r], kEnergyTolerance) << "canonical root " << r;
  }
  fb::Buffers::device<double> gamma_device(n * n, res.device());
  fb::Buffers::device<double> gamma2_device(n * n * n * n, res.device());
  // The RDMs at weights w, read back.
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
  fb::mcscf::ResponseIntegrals response_ints{mo,
                                             core,
                                             n,
                                             download(res, sa_fock->fock_inactive.data(), mo * mo),
                                             download(res, ints->ppaa.data(), mo * mo * n * n),
                                             download(res, ints->papa.data(), mo * mo * n * n)};
  auto relaxed = fb::mcscf::RelaxedHessian::create(res, *hessian, *sigma, *rdm, *alpha, *beta,
                                                   *roots, weights, std::move(response_ints));
  ASSERT_TRUE(relaxed.has_value()) << relaxed.error().message;
  const auto n_pairs = static_cast<std::size_t>(relaxed->size());

  const auto pair_list = entry.at("pairs").get<std::vector<std::array<std::size_t, 2>>>();
  ASSERT_EQ(pair_list.size(), n_pairs);
  const std::vector<double> c_host = download(res, roots->vectors.data(), dim * weights.size());
  for (std::int32_t state = 0; state < nroots; ++state) {
    SCOPED_TRACE(std::format("root {}", state));
    const fb::json &want = entry.at("roots").at(static_cast<std::size_t>(state)).at("z");

    // g_I: the gradient at root I's own RDMs, the same orbitals.
    std::vector<double> one_hot(weights.size(), 0.0);
    one_hot[static_cast<std::size_t>(state)] = 1.0;
    const auto [gamma_i, gamma2_i] = rdms(one_hot);
    auto root_fock = fb::response::fock(res, molecule->molecule, h->h, df.b, df.naux, c, ncore,
                                        ncas, ints->ppaa, ints->papa, gamma_i, gamma2_i);
    ASSERT_TRUE(root_fock.has_value()) << root_fock.error().message;

    const auto begin = std::chrono::steady_clock::now();
    auto z = fb::mcscf::zvector(res, *relaxed, *hessian, *roots, root_fock->gradient.data(),
                                kResidualTolerance);
    const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - begin;
    ASSERT_TRUE(z.has_value()) << z.error().message;
    EXPECT_LT(z->residual, kResidualTolerance);

    // z_o by (p, q), in magnitude.
    const std::vector<double> z_o = download(res, z->orbital.data(), n_pairs);
    const auto want_o = want.at("orbital").get<std::vector<double>>();
    double orbital_error = 0.0;
    for (std::size_t k = 0; k < n_pairs; ++k) {
      const auto [p, q] = pair_list[k];
      const double got = z_o[packed_index(p, q, mo, core, nocc)];
      orbital_error = std::max(orbital_error, std::abs(std::abs(got) - std::abs(want_o[k])));
    }
    EXPECT_LT(orbital_error, kMultiplierTolerance);

    // z_c per root r of the average, in magnitude, and off every root.
    const std::vector<double> z_c = download(res, z->ci.data(), dim * weights.size());
    const auto want_c = want.at("ci").get<std::vector<std::vector<std::vector<double>>>>();
    ASSERT_EQ(want_c.size(), weights.size());
    double ci_error = 0.0;
    double ci_overlap = 0.0;
    for (std::size_t r = 0; r < weights.size(); ++r) {
      std::vector<double> flat;
      for (const auto &row : want_c[r]) {
        flat.insert(flat.end(), row.begin(), row.end());
      }
      ASSERT_EQ(flat.size(), dim);
      for (std::size_t i = 0; i < dim; ++i) {
        ci_error = std::max(ci_error, std::abs(std::abs(z_c[(r * dim) + i]) - std::abs(flat[i])));
      }
      for (std::size_t s = 0; s < weights.size(); ++s) {
        double dot = 0.0;
        for (std::size_t i = 0; i < dim; ++i) {
          dot += z_c[(r * dim) + i] * c_host[(s * dim) + i];
        }
        ci_overlap = std::max(ci_overlap, std::abs(dot));
      }
    }
    EXPECT_LT(ci_error, kMultiplierTolerance);
    EXPECT_LT(ci_overlap, kCiOverlapTolerance);
    std::cout << std::format("[ zvector  ] {} root {}: {} products, residual {:.3e} (PySCF "
                             "{:.3e}), max |z_o| error {:.3e}, max |z_c| error {:.3e}, "
                             "max |<z_c|c>| {:.3e}, {:.2f} s\n",
                             entry.at("name").get<std::string>(), state, z->iterations, z->residual,
                             want.at("residual").get<double>(), orbital_error, ci_error, ci_overlap,
                             seconds.count());
    ::testing::Test::RecordProperty(std::format("root{}_products", state), z->iterations);
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfZvector, ::testing::Values("h2co_ccpvdz_sa2"),
                         [](const auto &info) { return std::string(info.param); });

} // namespace
