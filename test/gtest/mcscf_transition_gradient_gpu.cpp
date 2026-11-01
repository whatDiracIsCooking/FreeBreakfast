// mcscf::df_transition_gradient on the device against PySCF's DF SA-CASSCF
// couplings in data/casscf_nac/casscf_nac.json
// (tools/emit/casscf_nac_reference.py), for both cases (H2CO SA-2 and the
// skewed H2CO SA-3, CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT): this tree's
// DF-CASSCF (mcscf::solve under DensityFitting::Full) from the exact RHF and
// the Casci step's start, put in the reference's canonical orbitals with the
// CI re-solved there (as mcscf_zvector_gpu.cpp does); then for every ordered
// pair (I, J), at ci::Rdm::trans_rdm12's symmetrized transition RDMs,
//
// - the RDMs against the reference's sym_1 and sym_2, in magnitude;
// - the electronic terms' sum against `hellmann_feynman` (PySCF's
//   grad_elec_active) to 1e-9, up to the one sign s_I s_J the two CI phases
//   carry, with no nuclear repulsion;
// - the transition orbital gradient, response::fock's at the transition RDMs
//   less its gradient at zero RDMs, against `b`'s gen_g_hop_active vector,
//   matched by (p, q) and in magnitude, since an orbital's sign flips every
//   element it carries. gen_g_hop returns 2 pack(F - F^T), this tree's
//   g = 2 (F - F^T) itself (measured: equal to 7e-9, and off by 0.04 at
//   half), whatever casscf_nac/README.md says of the factor.
//
// Then the multipliers on top: with an arbitrary (z_o, z_c), each root's
// df_gradient less its df_transition_gradient at the root's own RDMs is the
// core's own energy and the nuclear repulsion, the same for every root and
// every z (zero z included, against the transition terms with no
// multipliers), to 1e-10. And RDMs of the wrong length are refused.
//
// The setup is mcscf_zvector_gpu.cpp's, so this executable links pipeline;
// mcscf does not. Runs the device chain, so needs a GPU.

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
import free_breakfast.mcscf.zvector;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bar on the Hellmann-Feynman terms.
constexpr double kTermTolerance = 1e-9;
// The transition orbital gradient and RDMs, elements of order 0.01 to 1: they
// move with the orbitals and the CI at first order, by ~||g|| / (the orbital
// Hessian's least eigenvalue) at the solve's convergence. The RDMs, a sanity
// check on the states, measured 1.4e-7 at worst on the 3080 (the skewed
// three roots, 2e-12 for two); the gradient 7e-9 (1e-11 for two).
constexpr double kOrbitalGradientTolerance = 1e-8;
constexpr double kRdmTolerance = 1e-6;
// The multipliers' path against itself: the same formula, to round-off.
constexpr double kSelfTolerance = 1e-10;
// The CASSCF's convergence, as mcscf_zvector_gpu.cpp's.
constexpr double kConvTolGrad = 1e-10;
// The energies and the canonical orbitals' spectra against the reference.
constexpr double kEnergyTolerance = 1e-8;
constexpr double kOrbitalEnergyTolerance = 1e-6;

const fb::json &reference_case(std::string_view name) {
  static const fb::json kDoc = [] {
    auto read =
        fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_nac" / "casscf_nac.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  for (const fb::json &entry : kDoc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf_nac/casscf_nac.json";
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
  const fb::test::ScratchDir dir("mcscf_transition_gradient");
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

// The electronic terms' sum, element 3 A + k.
std::vector<double> electronic(const fb::mcscf::GradientTerms &g) {
  std::vector<double> sum(g.total.size(), 0.0);
  for (const std::vector<double> *term : {&g.kinetic, &g.potential, &g.overlap, &g.two_electron}) {
    std::ranges::transform(sum, *term, sum.begin(), std::plus{});
  }
  return sum;
}

class McscfTransitionGradient : public ::testing::TestWithParam<std::string_view> {};

TEST_P(McscfTransitionGradient, MatchesPyscfGradElecActive) {
  FBF_REQUIRE_GPU();
  const std::string name{GetParam()};
  const fb::json &entry = reference_case(name);
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto nroots = static_cast<std::int32_t>(weights.size());
  const auto energies = entry.at("energies").get<std::vector<double>>();
  const auto start_time = std::chrono::steady_clock::now();

  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  std::optional<fb::Calculation> calc = run_setup_steps(entry, node);
  ASSERT_TRUE(calc.has_value());
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
  const fb::BasisSet &basis = basis_set->basis;
  ASSERT_EQ(basis.nao_sphe(), entry.at("nao").get<std::size_t>());
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
  const auto &df = std::get<fb::DensityFitted>(fitted->eri);
  ASSERT_EQ(df.naux, aux.at("naux").get<std::int32_t>());
  fb::CasciStep casci;
  casci.ncas = entry.at("ncas").get<std::int32_t>();
  casci.nelecas = entry.at("nelecas").get<std::int32_t>();
  casci.nroots = nroots;
  auto space = casci.select(res, molecule->molecule, basis, *scf);
  ASSERT_TRUE(space.has_value()) << space.error().message;
  const std::int32_t ncore = space->ncore;
  const std::int32_t ncas = space->ncas;
  const std::int32_t nelecas = space->nelecas;
  ASSERT_EQ(ncore, entry.at("ncore").get<std::int32_t>());
  ASSERT_EQ(nelecas, entry.at("nelecas").get<std::int32_t>());

  fb::mcscf::Options options;
  options.conv_tol_grad = kConvTolGrad;
  options.density_fitting = fb::mcscf::DensityFitting::Full;
  auto sol = fb::mcscf::solve(res, basis, molecule->molecule, h->h, transform->u, {}, {}, space->c,
                              ncore, ncas, nelecas, nroots, weights, options, &df.b, df.naux);
  ASSERT_TRUE(sol.has_value()) << sol.error().message;
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
    const std::vector<double> occupations = block(core, n, [&](std::size_t p, std::size_t q) {
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

  // At the canonical orbitals: the CI re-solved, and the solution there, its
  // weighted and per-root RDMs, so that df_gradient reads it as a solve's.
  auto ints =
      fb::casscf_integrals(res, basis, molecule->molecule, h->h, df.b, df.naux, c, ncore, ncas);
  ASSERT_TRUE(ints.has_value()) << ints.error().message;
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
  fb::mcscf::Solution canon{.c = std::move(c),
                            .roots = std::move(*roots),
                            .energy = sol->energy,
                            .gradient_norm = sol->gradient_norm,
                            .gamma = {},
                            .gamma2 = {},
                            .root_gammas = {},
                            .iterations = {}};
  fb::Buffers::device<double> gamma_device(n * n, res.device());
  fb::Buffers::device<double> gamma2_device(n * n * n * n, res.device());
  const auto rdms = [&](std::span<const double> w) {
    EXPECT_TRUE(
        rdm->rdm12(res, w, canon.roots.vectors.data(), gamma_device.data(), gamma2_device.data())
            .has_value());
    return std::pair{download(res, gamma_device.data(), n * n),
                     download(res, gamma2_device.data(), n * n * n * n)};
  };
  // The symmetrized transition RDMs of (bra, ket).
  const auto transition = [&](std::size_t bra, std::size_t ket) {
    EXPECT_TRUE(rdm->trans_rdm12(res, canon.roots.vectors.data() + (bra * dim),
                                 canon.roots.vectors.data() + (ket * dim), gamma_device.data(),
                                 gamma2_device.data())
                    .has_value());
    return std::pair{download(res, gamma_device.data(), n * n),
                     download(res, gamma2_device.data(), n * n * n * n)};
  };
  std::tie(canon.gamma, canon.gamma2) = rdms(weights);
  for (std::int32_t r = 0; r < nroots; ++r) {
    std::vector<double> one_hot(weights.size(), 0.0);
    one_hot[static_cast<std::size_t>(r)] = 1.0;
    const std::vector<double> root_gamma = rdms(one_hot).first;
    canon.root_gammas.insert(canon.root_gammas.end(), root_gamma.begin(), root_gamma.end());
  }
  // The Fock at zero RDMs, whose gradient is the core's own.
  const std::vector<double> zero_gamma(n * n, 0.0);
  const std::vector<double> zero_gamma2(n * n * n * n, 0.0);
  auto core_fock = fb::response::fock(res, molecule->molecule, h->h, df.b, df.naux, canon.c, ncore,
                                      ncas, ints->ppaa, ints->papa, zero_gamma, zero_gamma2);
  ASSERT_TRUE(core_fock.has_value()) << core_fock.error().message;
  const std::size_t n_pairs = (core * (mo - core)) + (n * (mo - nocc));
  const std::vector<double> core_gradient = download(res, core_fock->gradient.data(), n_pairs);
  const auto pair_list = entry.at("pairs").get<std::vector<std::array<std::size_t, 2>>>();
  ASSERT_EQ(pair_list.size(), n_pairs);

  for (const fb::json &coupling : entry.at("couplings")) {
    const auto bra = coupling.at("bra").get<std::size_t>();
    const auto ket = coupling.at("ket").get<std::size_t>();
    const std::string label = std::format("{} ({}, {})", name, bra, ket);
    SCOPED_TRACE(label);
    const auto [gamma, gamma2] = transition(bra, ket);

    // The RDMs in magnitude: the active orbitals' and the CI's signs are each
    // side's own.
    const auto want_1 = coupling.at("rdm").at("sym_1").get<std::vector<std::vector<double>>>();
    const auto want_2 = coupling.at("rdm").at("sym_2").get<std::vector<double>>();
    ASSERT_EQ(want_1.size(), n);
    ASSERT_EQ(want_2.size(), n * n * n * n);
    double rdm_error = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t v = 0; v < n; ++v) {
        rdm_error =
            std::max(rdm_error, std::abs(std::abs(gamma[(t * n) + v]) - std::abs(want_1[t][v])));
      }
    }
    for (std::size_t i = 0; i < want_2.size(); ++i) {
      rdm_error = std::max(rdm_error, std::abs(std::abs(gamma2[i]) - std::abs(want_2[i])));
    }
    EXPECT_LT(rdm_error, kRdmTolerance);

    // The Hellmann-Feynman terms, up to s_I s_J.
    const auto pair_start = std::chrono::steady_clock::now();
    const auto g = fb::mcscf::df_transition_gradient(res, molecule->molecule, basis, pairs->classes,
                                                     df.aux, h->h, df.b, df.l.data(), canon, ncore,
                                                     ncas, gamma, gamma2);
    ASSERT_TRUE(g.has_value()) << g.error().message;
    const std::chrono::duration<double> pair_seconds =
        std::chrono::steady_clock::now() - pair_start;
    EXPECT_TRUE(std::ranges::all_of(g->nuclear_repulsion, [](double v) { return v == 0.0; }));
    const std::vector<double> ours = electronic(*g);
    const auto rows = coupling.at("hellmann_feynman").get<std::vector<std::vector<double>>>();
    ASSERT_EQ(ours.size(), 3 * rows.size());
    double dot = 0.0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
      dot += ours[i] * rows[i / 3][i % 3];
    }
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    double hf_error = 0.0;
    for (std::size_t i = 0; i < ours.size(); ++i) {
      EXPECT_NEAR(ours[i], sign * rows[i / 3][i % 3], kTermTolerance) << "hellmann_feynman " << i;
      hf_error = std::max(hf_error, std::abs(ours[i] - (sign * rows[i / 3][i % 3])));
      EXPECT_NEAR(g->total[i], ours[i], kSelfTolerance) << "total " << i;
    }

    // The transition orbital gradient: fock's at the transition RDMs less
    // its at zero ones, against b by (p, q), in magnitude.
    auto fock = fb::response::fock(res, molecule->molecule, h->h, df.b, df.naux, canon.c, ncore,
                                   ncas, ints->ppaa, ints->papa, gamma, gamma2);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;
    std::vector<double> b_ours = download(res, fock->gradient.data(), n_pairs);
    std::ranges::transform(b_ours, core_gradient, b_ours.begin(), std::minus{});
    const auto want_b = coupling.at("b").at("orbital").get<std::vector<double>>();
    ASSERT_EQ(want_b.size(), n_pairs);
    double b_error = 0.0;
    for (std::size_t k = 0; k < n_pairs; ++k) {
      const auto [p, q] = pair_list[k];
      const double got = b_ours[packed_index(p, q, mo, core, nocc)];
      b_error = std::max(b_error, std::abs(std::abs(got) - std::abs(want_b[k])));
    }
    EXPECT_LT(b_error, kOrbitalGradientTolerance);
    std::cout << std::format("[ transgrad] {}: hellmann_feynman {:.2e} ({:.2f} s), orbital "
                             "gradient {:.2e}, rdms {:.2e}\n",
                             label, hf_error, pair_seconds.count(), b_error, rdm_error);
  }

  // The multipliers on top. Each root's df_gradient less its
  // df_transition_gradient at its own RDMs, under the same multipliers, is
  // the core's own energy and the nuclear repulsion: one gradient for every
  // root and every z.
  const std::size_t ci_count = dim * weights.size();
  std::mt19937 engine(661);
  std::uniform_real_distribution<double> uniform(-1e-2, 1e-2);
  std::vector<double> z_o(n_pairs);
  std::vector<double> z_c(ci_count);
  std::ranges::generate(z_o, [&] { return uniform(engine); });
  std::ranges::generate(z_c, [&] { return uniform(engine); });
  const auto make_z = [&](bool zero) {
    fb::mcscf::ZVector z{fb::Buffers::device<double>(n_pairs, res.device()),
                         fb::Buffers::device<double>(ci_count, res.device()),
                         {},
                         0,
                         0.0};
    upload(res, zero ? std::vector<double>(n_pairs, 0.0) : z_o, z.orbital.data());
    upload(res, zero ? std::vector<double>(ci_count, 0.0) : z_c, z.ci.data());
    return z;
  };
  const fb::mcscf::ZVector arbitrary = make_z(false);
  const fb::mcscf::ZVector zero = make_z(true);
  std::optional<std::vector<double>> core_part;
  double self_error = 0.0;
  for (std::int32_t r = 0; r < nroots; ++r) {
    const auto root = static_cast<std::size_t>(r);
    const auto [gamma, gamma2] = transition(root, root);
    for (const fb::mcscf::ZVector *z : {&arbitrary, &zero}) {
      const auto full =
          fb::mcscf::df_gradient(res, molecule->molecule, basis, pairs->classes, df.aux, h->h, df.b,
                                 df.l.data(), canon, ncore, ncas, r, z);
      ASSERT_TRUE(full.has_value()) << full.error().message;
      // Zero multipliers stand against the transition terms with none.
      const auto active = fb::mcscf::df_transition_gradient(
          res, molecule->molecule, basis, pairs->classes, df.aux, h->h, df.b, df.l.data(), canon,
          ncore, ncas, gamma, gamma2, z == &zero ? nullptr : z);
      ASSERT_TRUE(active.has_value()) << active.error().message;
      std::vector<double> difference(full->total.size());
      std::ranges::transform(full->total, active->total, difference.begin(), std::minus{});
      if (!core_part.has_value()) {
        core_part = difference;
        continue;
      }
      for (std::size_t i = 0; i < difference.size(); ++i) {
        EXPECT_NEAR(difference[i], (*core_part)[i], kSelfTolerance)
            << "root " << r << (z == &zero ? " zero z " : " z ") << i;
        self_error = std::max(self_error, std::abs(difference[i] - (*core_part)[i]));
      }
    }
  }
  const std::chrono::duration<double> seconds = std::chrono::steady_clock::now() - start_time;
  std::cout << std::format("[ transgrad] {}: multipliers' core part {:.2e}, {:.2f} s in all\n",
                           name, self_error, seconds.count());

  // RDMs of the wrong length are InvalidConfig.
  const auto wrong = fb::mcscf::df_transition_gradient(
      res, molecule->molecule, basis, pairs->classes, df.aux, h->h, df.b, df.l.data(), canon, ncore,
      ncas, zero_gamma, std::span<const double>{zero_gamma2}.first(1));
  ASSERT_FALSE(wrong.has_value());
  EXPECT_EQ(wrong.error().kind, fb::ErrorKind::InvalidConfig) << wrong.error().message;
  EXPECT_NE(wrong.error().message.find("mcscf::df_transition_gradient"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, McscfTransitionGradient,
                         ::testing::Values(std::string_view{"h2co_ccpvdz_sa2"},
                                           std::string_view{"h2co_skew_ccpvdz_sa3"}),
                         [](const auto &info) { return std::string(info.param); });

} // namespace
