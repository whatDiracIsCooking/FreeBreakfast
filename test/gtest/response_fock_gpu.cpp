// response::fock on the device, at each case's CASCI orbitals, against PySCF's
// mc1step.gen_g_hop (data/casscf/casscf.json, tools/emit_casscf_reference.py):
// the weighted energy against the CASCI energy; the gradient's sign-invariant
// sqrt(sum g_pq^2) per pair of orbital sets, and |g|; the gradient against a
// central finite difference of the energy along random kappa, the RDMs fixed;
// the RHF limit, ncas = 0; the packed order; and the refusals.
//
// The orbitals are the setup's own: the SCF (run with J unscreened), the active
// space picked by pipeline.active_space as the Casci step picks it (by index,
// or by AVAS), and the CI and RDMs from the Casci step's roots and ci.rdm's
// ci::Rdm.
// So this executable links pipeline; response itself does not. Runs the device
// chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.recipes.expm;
import free_breakfast.integrals;
import free_breakfast.ci;
import free_breakfast.response;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.pipeline.casci;
import free_breakfast.core.nontransferrable;

#include "mcscf_setup.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars.
constexpr double kEnergyTolerance = 1e-9;
constexpr double kGradientTolerance = 1e-7;
constexpr double kRhfGradientTolerance = 1e-6;
constexpr double kFiniteDifferenceTolerance = 1e-6;
// The orbital energies only pair our orbitals with PySCF's by index.
constexpr double kOrbitalEnergyTolerance = 1e-6;
// The finite difference: Richardson's extrapolation of two central
// differences, at steps kStep and kStep / 2, which cancels the step^2 term;
// the step^4 one left is far below the bar, and the energy's noise over
// kStep is too.
constexpr double kStep = 2e-3;
constexpr std::int32_t kDirections = 3;

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf" / "casscf.json");
    EXPECT_TRUE(read.has_value()) << read.error().message;
    return read.has_value() ? std::move(*read) : fb::json{};
  }();
  return doc;
}

const fb::json &reference_case(std::string_view name) {
  for (const fb::json &entry : reference().at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf/casscf.json";
  static const fb::json kNone;
  return kNone;
}

// Copies in res.stream()'s order, waited for.
std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

void upload(fb::DeviceResources &res, const std::vector<double> &host, double *data) {
  EXPECT_EQ(wwr::wwrMemcpyAsync(data, host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
}

// The packed order, as src/response/README.md documents it: (p, q) per packed
// element.
std::vector<std::pair<std::size_t, std::size_t>> packed_pairs(std::size_t nmo, std::size_t ncore,
                                                              std::size_t ncas) {
  std::vector<std::pair<std::size_t, std::size_t>> pairs;
  for (std::size_t q = 0; q < ncore; ++q) {
    for (std::size_t p = ncore; p < nmo; ++p) {
      pairs.emplace_back(p, q);
    }
  }
  for (std::size_t q = ncore; q < ncore + ncas; ++q) {
    for (std::size_t p = ncore + ncas; p < nmo; ++p) {
      pairs.emplace_back(p, q);
    }
  }
  return pairs;
}

// The reference's molecule through the real registry to the Scf step's inputs, in
// basis if it is not empty and the reference's own otherwise. The Scf step itself
// is run by hand: tau_j is not a TOML key.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node,
                                               std::string_view basis = {}) {
  const fb::test::ScratchDir dir("response_fock");
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
                  entry.at("charge").get<std::int32_t>(),
                  basis.empty() ? entry.at("basis").get<std::string>() : std::string{basis});
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

// The reference's active space as the Casci step's selection.
fb::CasciStep casci_step(const fb::json &entry) {
  fb::CasciStep step;
  if (!entry.at("avas").is_null()) {
    step.avas = entry.at("avas").at("labels").get<std::vector<std::string>>();
    step.avas_threshold = entry.at("avas").at("threshold").get<double>();
  } else {
    step.ncas = entry.at("ncas").get<std::int32_t>();
    step.nelecas = entry.at("nelecas").get<std::int32_t>();
    if (!entry.at("active").is_null()) {
      step.active = entry.at("active").get<std::vector<std::int32_t>>();
    }
  }
  step.nroots = static_cast<std::int32_t>(entry.at("weights").size());
  return step;
}

// Everything response::fock reads, from one converged SCF, and with a CASCI, its
// orbitals and the weighted RDMs of its roots.
struct Setup {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  const fb::Molecule *molecule = nullptr;
  const fb::BasisSet *basis = nullptr;
  const fb::SphericalTransformOutput *transform = nullptr;
  const fb::CoreHamiltonianOutput *h = nullptr;
  std::optional<fb::ScfOutput> scf;
  std::vector<fb::CoulombPairClass> j_classes;
  std::vector<fb::OrderedPairClass> k_classes;
  std::optional<fb::CasciOutput> casci;
  std::vector<double> weights;
  std::vector<double> gamma;
  std::vector<double> gamma2;
  std::size_t nmo = 0;

  [[nodiscard]] fb::DeviceResources &res() { return node.device(0); }

  // response::fock over orbitals with the setup's RDMs, ppaa built over them.
  [[nodiscard]] fb::Result<fb::response::Fock> fock(const fb::SphericalMatrixAO &orbitals) {
    auto integrals = fb::casscf_integrals(res(), *basis, *molecule, h->h, transform->u, j_classes,
                                          k_classes, orbitals, casci->ncore, casci->ncas);
    if (!integrals.has_value()) {
      return std::unexpected(std::move(integrals.error()));
    }
    return fb::response::fock(res(), *basis, *molecule, h->h, transform->u, j_classes, k_classes,
                              orbitals, casci->ncore, casci->ncas, integrals->ppaa, gamma, gamma2);
  }

  // The energy of orbitals at the setup's RDMs, from active_space_hamiltonian:
  // E_core + sum h_eff gamma + 1/2 sum (tu|vw) Gamma, independent of response::fock.
  [[nodiscard]] fb::Result<double> energy(const fb::SphericalMatrixAO &orbitals) {
    auto built =
        fb::active_space_hamiltonian(res(), *basis, *molecule, h->h, transform->u, j_classes,
                                     k_classes, orbitals, casci->ncore, casci->ncas, 1e-14, 1e-14);
    if (!built.has_value()) {
      return std::unexpected(std::move(built.error()));
    }
    const fb::ActiveSpaceHamiltonian &ham = *built;
    const auto n = static_cast<std::size_t>(casci->ncas);
    const std::vector<double> h_eff = download(res(), ham.h_eff.data(), n * n);
    const std::vector<double> eri = download(res(), ham.eri.data(), n * n * n * n);
    double e = ham.e_core;
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t u = 0; u < n; ++u) {
        e += h_eff[t + (n * u)] * gamma[(t * n) + u];
        for (std::size_t v = 0; v < n; ++v) {
          for (std::size_t w = 0; w < n; ++w) {
            e += 0.5 * eri[t + (n * (u + (n * (v + (n * w)))))] *
                 gamma2[(((((t * n) + u) * n) + v) * n) + w];
          }
        }
      }
    }
    return e;
  }
};

// The setup over the reference's molecule (in basis if not empty), with the
// CASCI and its RDMs when with_casci.
std::unique_ptr<Setup> make_setup(const fb::json &entry, bool with_casci,
                                  std::string_view basis = {}) {
  auto setup = std::make_unique<Setup>();
  auto calc = run_setup_steps(entry, setup->node, basis);
  if (!calc.has_value()) {
    return nullptr;
  }
  setup->calc.emplace(std::move(*calc)); // a Pipeline moves, but is not move-assigned
  const auto &molecule = setup->calc->get<fb::MoleculeOutput>();
  const auto &basis_set = setup->calc->get<fb::BasisSetOutput>();
  const auto &transform = setup->calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = setup->calc->get<fb::PrimitivePairClassesOutput>();
  const auto &overlap = setup->calc->get<fb::OverlapOutput>();
  const auto &x = setup->calc->get<fb::OrthogonalizerOutput>();
  const auto &h = setup->calc->get<fb::CoreHamiltonianOutput>();
  const auto &guess = setup->calc->get<fb::GuessOutput>();
  if (!molecule.has_value() || !basis_set.has_value() || !transform.has_value() ||
      !pairs.has_value() || !overlap.has_value() || !x.has_value() || !h.has_value() ||
      !guess.has_value()) {
    ADD_FAILURE() << "a product of the setup steps is missing";
    return nullptr;
  }
  setup->molecule = &molecule->molecule;
  setup->basis = &basis_set->basis;
  setup->transform = &*transform;
  setup->h = &*h;
  setup->nmo = basis_set->basis.nao_sphe();
  auto scf = fb::test::kExactScf.run(fb::StepInput<fb::ScfStep>{*molecule, *basis_set, *transform,
                                                                *pairs, *overlap, *x, *h, *guess},
                                     setup->res());
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return nullptr;
  }
  setup->scf.emplace(std::move(*scf));

  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto j_classes = fb::coulomb_pair_classes(res, pairs->classes);
  auto k_classes = fb::ordered_pair_classes(res, basis_set->basis, pairs->classes);
  if (!j_classes.has_value() || !k_classes.has_value()) {
    ADD_FAILURE() << "the J or K classes failed";
    return nullptr;
  }
  setup->j_classes = std::move(*j_classes);
  setup->k_classes = std::move(*k_classes);
  if (!with_casci) {
    return setup;
  }

  // The CASCI at the reference's start, and the selection's rotation R, which
  // gives each column's energy in the ScfOutput's Fock matrix: sum_k R_kj^2 eps_k.
  const fb::CasciStep step = casci_step(entry);
  auto space = step.select(res, molecule->molecule, basis_set->basis, *setup->scf);
  if (!space.has_value()) {
    ADD_FAILURE() << space.error().message;
    return nullptr;
  }
  auto casci = step.run(
      fb::StepInput<fb::CasciStep>{*molecule, *basis_set, *transform, *pairs, *h, *setup->scf},
      res);
  if (!casci.has_value()) {
    ADD_FAILURE() << casci.error().message;
    return nullptr;
  }
  setup->casci.emplace(std::move(*casci));
  EXPECT_EQ(setup->casci->ncore, entry.at("ncore").get<std::int32_t>());
  EXPECT_EQ(setup->casci->ncas, entry.at("ncas").get<std::int32_t>());
  EXPECT_EQ(setup->casci->nelecas, entry.at("nelecas").get<std::int32_t>());
  const auto expected = entry.at("casci").at("orbital_energies").get<std::vector<double>>();
  EXPECT_EQ(expected.size(), setup->nmo);
  for (std::size_t col = 0; col < std::min(expected.size(), setup->nmo); ++col) {
    double energy = 0.0;
    for (std::size_t k = 0; k < setup->nmo; ++k) {
      const double r = space->r[k + (setup->nmo * col)];
      energy += r * r * setup->scf->mo_energies[k];
    }
    EXPECT_NEAR(energy, expected[col], kOrbitalEnergyTolerance) << "column " << col;
  }

  setup->weights = entry.at("weights").get<std::vector<double>>();
  const std::int32_t ncas = setup->casci->ncas;
  const std::int32_t n_beta = setup->casci->nelecas / 2;
  auto alpha = fb::ci::StringSet::create(ncas, setup->casci->nelecas - n_beta);
  auto beta = fb::ci::StringSet::create(ncas, n_beta);
  if (!alpha.has_value() || !beta.has_value()) {
    ADD_FAILURE() << "the string sets failed";
    return nullptr;
  }
  const auto n = static_cast<std::size_t>(ncas);
  // ci::Rdm reads and writes the device; response::fock takes gamma and Gamma
  // on the host, as a CASSCF loop reads them back for its energy anyway.
  auto ci_rdm = fb::ci::Rdm::create(res, *alpha, *beta);
  if (!ci_rdm.has_value()) {
    ADD_FAILURE() << ci_rdm.error().message;
    return nullptr;
  }
  fb::Buffers::device<double> vectors(setup->casci->vectors.size(), res.device());
  fb::Buffers::device<double> gamma(n * n, res.device());
  fb::Buffers::device<double> gamma2(n * n * n * n, res.device());
  upload(res, setup->casci->vectors, vectors.data());
  const fb::Status rdm =
      ci_rdm->rdm12(res, setup->weights, vectors.data(), gamma.data(), gamma2.data());
  if (!rdm.has_value()) {
    ADD_FAILURE() << rdm.error().message;
    return nullptr;
  }
  setup->gamma = download(res, gamma.data(), n * n);
  setup->gamma2 = download(res, gamma2.data(), n * n * n * n);
  return setup;
}

// C exp(t kappa(x)) for the packed x: kappa unpacked on the device, then
// skew_eig and skew_expm.
std::optional<fb::SphericalMatrixAO> rotate(Setup &setup, const fb::SphericalMatrixAO &c,
                                            std::int32_t ncore, std::int32_t ncas,
                                            const std::vector<double> &x, double t) {
  fb::DeviceResources &res = setup.res();
  const auto nmo = static_cast<std::int32_t>(setup.nmo);
  const std::size_t nn = setup.nmo * setup.nmo;
  fb::Buffers::device<double> packed(x.size(), res.device());
  fb::Buffers::device<double> kappa(nn, res.device());
  fb::Buffers::device<wwr::wwrDoubleComplex> u(nn, res.device());
  fb::Buffers::device<double> w(setup.nmo, res.device());
  fb::Buffers::device<double> e(nn, res.device());
  upload(res, x, packed.data());
  if (!fb::response::unpack_orbital_pairs(res, nmo, ncore, ncas, packed.data(), kappa.data())
           .has_value() ||
      !fb::skew_eig<double>(res, nmo, kappa.data(), u.data(), w.data()).has_value() ||
      !fb::skew_expm<double>(res, nmo, u.data(), w.data(), t, e.data()).has_value()) {
    ADD_FAILURE() << "the rotation failed";
    return std::nullopt;
  }
  std::optional<fb::SphericalMatrixAO> rotated;
  rotated.emplace(res, *setup.basis);
  constexpr double kOne = 1.0;
  constexpr double kZero = 0.0;
  const auto status = wwr::gemm<double, std::int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, nmo, nmo, nmo, &kOne, c.data(), nmo,
      e.data(), nmo, &kZero, rotated->data(), nmo);
  EXPECT_EQ(status, wwr::WWRBLAS_STATUS_SUCCESS);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return rotated;
}

// Unit random directions in the packed space, from a fixed seed.
std::vector<std::vector<double>> directions(std::size_t count, std::size_t n_pairs) {
  std::mt19937_64 engine{248};
  std::normal_distribution<double> normal;
  std::vector<std::vector<double>> out(count, std::vector<double>(n_pairs));
  for (auto &x : out) {
    double norm = 0.0;
    for (double &value : x) {
      value = normal(engine);
      norm += value * value;
    }
    for (double &value : x) {
      value /= std::sqrt(norm);
    }
  }
  return out;
}

// dE/dt at t = 0 along x by Richardson's extrapolation of two central
// differences of energy(C exp(t kappa(x))).
template <typename Energy>
std::optional<double> slope(Setup &setup, const fb::SphericalMatrixAO &c, std::int32_t ncore,
                            std::int32_t ncas, const std::vector<double> &x, Energy &&energy) {
  std::array<double, 2> central{};
  for (std::size_t i = 0; i < 2; ++i) {
    const double step = kStep / static_cast<double>(1U << i);
    auto plus = rotate(setup, c, ncore, ncas, x, step);
    auto minus = rotate(setup, c, ncore, ncas, x, -step);
    if (!plus.has_value() || !minus.has_value()) {
      return std::nullopt;
    }
    const fb::Result<double> e_plus = energy(*plus);
    const fb::Result<double> e_minus = energy(*minus);
    if (!e_plus.has_value() || !e_minus.has_value()) {
      ADD_FAILURE() << "an energy failed";
      return std::nullopt;
    }
    central[i] = (*e_plus - *e_minus) / (2.0 * step);
  }
  return ((4.0 * central[1]) - central[0]) / 3.0;
}

struct Case {
  std::string_view name;
};

// ctest names a discovered case by this, so `ctest -R n2_ccpvdz` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

class ResponseFock : public ::testing::TestWithParam<Case> {};

TEST_P(ResponseFock, MatchesPyscfAtTheCasciOrbitals) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name);
  const fb::json &casci = entry.at("casci");
  auto setup = make_setup(entry, true);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::size_t nmo = setup->nmo;
  const auto ncore = static_cast<std::size_t>(setup->casci->ncore);
  const auto ncas = static_cast<std::size_t>(setup->casci->ncas);

  auto fock = setup->fock(setup->casci->c);
  ASSERT_TRUE(fock.has_value()) << fock.error().message;

  // The energy: PySCF's CASCI, our own Casci step's roots weighted, and
  // active_space_hamiltonian's over the same RDMs.
  EXPECT_NEAR(fock->energy, casci.at("energy").get<double>(), kEnergyTolerance);
  double weighted = 0.0;
  for (std::size_t r = 0; r < setup->weights.size(); ++r) {
    weighted += setup->weights[r] * setup->casci->energies[r];
  }
  EXPECT_NEAR(fock->energy, weighted, kEnergyTolerance);
  const fb::Result<double> independent = setup->energy(setup->casci->c);
  ASSERT_TRUE(independent.has_value()) << independent.error().message;
  EXPECT_NEAR(fock->energy, *independent, kEnergyTolerance);

  // The gradient, per pair of orbital sets and in norm.
  const auto order = packed_pairs(nmo, ncore, ncas);
  ASSERT_EQ(fock->gradient.num_elements(), order.size());
  const std::vector<double> g = download(res, fock->gradient.data(), order.size());
  std::map<std::pair<std::size_t, std::size_t>, std::size_t> index;
  for (std::size_t i = 0; i < order.size(); ++i) {
    index[order[i]] = i;
  }
  std::size_t covered = 0;
  for (const fb::json &pair : casci.at("pairs")) {
    const auto ps = pair.at("p").get<std::vector<std::size_t>>();
    const auto qs = pair.at("q").get<std::vector<std::size_t>>();
    double sum = 0.0;
    for (const std::size_t p : ps) {
      for (const std::size_t q : qs) {
        const auto at = index.find({p, q});
        ASSERT_NE(at, index.end()) << "(" << p << ", " << q << ") is not a packed pair";
        sum += g[at->second] * g[at->second];
        ++covered;
      }
    }
    EXPECT_NEAR(std::sqrt(sum), pair.at("gradient").get<double>(), kGradientTolerance)
        << "p " << ps.front() << ", q " << qs.front();
  }
  EXPECT_EQ(covered, order.size());
  double norm = 0.0;
  for (const double value : g) {
    norm += value * value;
  }
  norm = std::sqrt(norm);
  EXPECT_NEAR(norm, casci.at("gradient_norm").get<double>(), kGradientTolerance);

  // The gradient against the energy's slope along random kappa, RDMs fixed.
  for (const auto &x : directions(kDirections, order.size())) {
    const auto fd =
        slope(*setup, setup->casci->c, setup->casci->ncore, setup->casci->ncas, x,
              [&](const fb::SphericalMatrixAO &orbitals) { return setup->energy(orbitals); });
    ASSERT_TRUE(fd.has_value());
    const double gx = std::inner_product(g.begin(), g.end(), x.begin(), 0.0);
    EXPECT_NEAR(*fd, gx, kFiniteDifferenceTolerance * std::max(std::abs(gx), norm));
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, ResponseFock,
                         ::testing::Values(Case{"h2o_ccpvdz"}, Case{"n2_ccpvdz"},
                                           Case{"h2co_ccpvdz_sa2"},
                                           Case{"h2co_ccpvdz_sa2_weighted"},
                                           Case{"h2o_ccpvdz_avas"}),
                         [](const auto &info) { return std::string(info.param.name); });

// ncas = 0: the generalized Fock is 2 F^I in the core columns, the gradient
// 4 F^I_ai, the RHF one: zero at the converged SCF, whose energy is given
// back; and away from it, the slope of response::fock's own energy.
TEST(ResponseFockRhfLimit, IsTheRhfGradient) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz");
  auto setup = make_setup(entry, false);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::size_t nmo = setup->nmo;
  const std::int32_t nocc = setup->molecule->electron_count() / 2;
  const fb::Buffers::device<double> no_ppaa(0, res.device());
  const auto rhf = [&](const fb::SphericalMatrixAO &orbitals) {
    return fb::response::fock(res, *setup->basis, *setup->molecule, setup->h->h,
                              setup->transform->u, setup->j_classes, setup->k_classes, orbitals,
                              nocc, 0, no_ppaa, {}, {});
  };

  auto at_scf = rhf(setup->scf->c);
  ASSERT_TRUE(at_scf.has_value()) << at_scf.error().message;
  EXPECT_NEAR(at_scf->energy, setup->scf->energy, kEnergyTolerance);
  const auto order = packed_pairs(nmo, static_cast<std::size_t>(nocc), 0);
  ASSERT_EQ(at_scf->gradient.num_elements(), order.size());
  for (const double value : download(res, at_scf->gradient.data(), order.size())) {
    EXPECT_NEAR(value, 0.0, kRhfGradientTolerance);
  }

  // Off the SCF, along a random kappa of 0.05.
  std::vector<double> step = directions(1, order.size()).front();
  auto off = rotate(*setup, setup->scf->c, nocc, 0, step, 0.05);
  ASSERT_TRUE(off.has_value());
  auto there = rhf(*off);
  ASSERT_TRUE(there.has_value()) << there.error().message;
  const std::vector<double> g = download(res, there->gradient.data(), order.size());
  const std::vector<double> f = download(res, there->fock_inactive.data(), nmo * nmo);
  const std::vector<double> fock = download(res, there->fock.data(), nmo * nmo);
  double norm = 0.0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    const auto [p, q] = order[i];
    EXPECT_NEAR(g[i], 4.0 * f[p + (nmo * q)], 1e-12) << "(" << p << ", " << q << ")";
    norm += g[i] * g[i];
  }
  norm = std::sqrt(norm);
  for (std::size_t q = 0; q < nmo; ++q) {
    for (std::size_t p = 0; p < nmo; ++p) {
      const double want = q < static_cast<std::size_t>(nocc) ? 2.0 * f[p + (nmo * q)] : 0.0;
      EXPECT_NEAR(fock[p + (nmo * q)], want, 1e-12) << "(" << p << ", " << q << ")";
    }
  }
  ASSERT_GT(norm, 1e-2); // far enough off the ScfOutput to test anything
  for (const auto &x : directions(kDirections, order.size())) {
    const auto fd = slope(*setup, *off, nocc, 0, x, [&](const fb::SphericalMatrixAO &orbitals) {
      return rhf(orbitals).transform([](const fb::response::Fock &r) { return r.energy; });
    });
    ASSERT_TRUE(fd.has_value());
    const double gx = std::inner_product(g.begin(), g.end(), x.begin(), 0.0);
    EXPECT_NEAR(*fd, gx, kFiniteDifferenceTolerance * std::max(std::abs(gx), norm));
  }
}

// The packed order, element by element, and unpack as its antisymmetric
// inverse, over every combination of empty and non-empty spaces.
TEST(ResponseFockPackedOrder, PacksAndUnpacksTheNonRedundantPairs) {
  FBF_REQUIRE_GPU();
  fb::NodeResources node;
  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  constexpr std::size_t kNmo = 7;
  const std::vector<std::pair<std::int32_t, std::int32_t>> spaces{{2, 3}, {0, 3}, {3, 0}, {0, 7},
                                                                  {7, 0}, {2, 5}, {1, 1}};
  for (const auto &[ncore, ncas] : spaces) {
    const auto order =
        packed_pairs(kNmo, static_cast<std::size_t>(ncore), static_cast<std::size_t>(ncas));
    std::vector<double> full(kNmo * kNmo);
    for (std::size_t i = 0; i < full.size(); ++i) {
      full[i] = static_cast<double>(i + 1);
    }
    fb::Buffers::device<double> full_device(full.size(), res.device());
    fb::Buffers::device<double> packed(std::max<std::size_t>(order.size(), 1), res.device());
    upload(res, full, full_device.data());
    ASSERT_TRUE(
        fb::response::pack_orbital_pairs(res, kNmo, ncore, ncas, full_device.data(), packed.data())
            .has_value());
    const std::vector<double> got = download(res, packed.data(), order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      const auto [p, q] = order[i];
      EXPECT_EQ(got[i], full[p + (kNmo * q)]) << ncore << ", " << ncas << ": element " << i;
    }

    std::vector<double> x(order.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i] = static_cast<double>(i) + 0.5;
    }
    if (!x.empty()) {
      upload(res, x, packed.data());
    }
    ASSERT_TRUE(fb::response::unpack_orbital_pairs(res, kNmo, ncore, ncas, packed.data(),
                                                   full_device.data())
                    .has_value());
    const std::vector<double> kappa = download(res, full_device.data(), full.size());
    std::vector<double> want(kNmo * kNmo, 0.0);
    for (std::size_t i = 0; i < order.size(); ++i) {
      const auto [p, q] = order[i];
      want[p + (kNmo * q)] = x[i];
      want[q + (kNmo * p)] = -x[i];
    }
    for (std::size_t i = 0; i < want.size(); ++i) {
      EXPECT_EQ(kappa[i], want[i]) << ncore << ", " << ncas << ": element " << i;
    }
  }
  const fb::Status refused = fb::response::pack_orbital_pairs(res, kNmo, 4, 4, nullptr, nullptr);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
  const fb::Status negative =
      fb::response::unpack_orbital_pairs(res, kNmo, -1, 2, nullptr, nullptr);
  ASSERT_FALSE(negative.has_value());
  EXPECT_EQ(negative.error().kind, fb::ErrorKind::InvalidConfig);
}

// Everything response::fock refuses, refused before the device is touched.
TEST(ResponseFockRefusals, AreInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz");
  auto setup = make_setup(entry, false, "sto-3g");
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const std::size_t nmo = setup->nmo;
  const std::vector<double> gamma(4, 0.0);
  const std::vector<double> gamma2(16, 0.0);
  const fb::Buffers::device<double> ppaa(nmo * nmo * 4, res.device());
  const fb::Buffers::device<double> short_ppaa(nmo * nmo, res.device());
  const auto call = [&](std::int32_t ncore, std::int32_t ncas, const fb::Buffers::device<double> &p,
                        std::span<const double> g1, std::span<const double> g2,
                        double tau = 1e-14) {
    return fb::response::fock(res, *setup->basis, *setup->molecule, setup->h->h,
                              setup->transform->u, setup->j_classes, setup->k_classes,
                              setup->scf->c, ncore, ncas, p, g1, g2, tau, tau);
  };
  const auto refused = [](const fb::Result<fb::response::Fock> &r, std::string_view what) {
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(r.error().message.find(what), std::string::npos) << r.error().message;
  };
  ASSERT_TRUE(call(4, 2, ppaa, gamma, gamma2).has_value());
  refused(call(-1, 2, ppaa, gamma, gamma2), "must not be negative");
  refused(call(6, 2, ppaa, gamma, gamma2), "exceeds the 7 orbitals");
  refused(call(0, 0, short_ppaa, {}, {}), "nothing is occupied");
  refused(call(4, 2, ppaa, gamma, gamma2, -1.0), "must be 0 or above");
  refused(call(4, 2, ppaa, gamma, gamma2, std::nan("")), "must be 0 or above");
  refused(call(4, 2, short_ppaa, gamma, gamma2), "ppaa holds");
  refused(call(4, 2, ppaa, std::span{gamma}.first(3), gamma2), "gamma holds");
  refused(call(4, 2, ppaa, gamma, std::span{gamma2}.first(8)), "gamma holds");
}

} // namespace
