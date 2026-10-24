// active_space_hamiltonian on the device, over the Scf step's orbitals (run with
// J unscreened, kExactScf below), against
// PySCF's CASCI (data/active_space/active_space.json,
// tools/emit_active_integrals_reference.py): E_core and the sign-invariant
// elements h_eff_tt, (tt|uu) and (tu|tu); the RHF energy given back by an
// active space spanning occupied and virtual orbitals; (tu|vw)'s 8-fold
// symmetry; the prune cap, on orbitals scaled past it; and the refusals. Each
// at tau = 0 and at the defaults. casscf_integrals' ppaa and papa likewise:
// their sign-invariant (pp|uu) and (pu|pu), summed over degenerate sets,
// against PySCF's CASSCF.ao2mo; their active blocks against
// active_space_hamiltonian's (tu|vw); their permutational symmetries; and the
// cap, for K as for J. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.integrals;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// PySCF's integrals are exact and its orbitals converged to conv_tol_grad
// 1e-11; ours carry kExactScf's tolerances, K's screen in the Scf, and, at the
// defaults, J's and K's screens here.
constexpr double kPyscfTolerance = 1e-10;
// The orbital energies only pair our orbitals with PySCF's by index.
constexpr double kOrbitalEnergyTolerance = 1e-6;

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "active_space" /
                              "active_space.json");
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
  ADD_FAILURE() << "no case " << name << " in active_space/active_space.json";
  static const fb::json kNone;
  return kNone;
}

// Copies in res.stream()'s order, waited for: what the call under test
// queued there is finished before the host reads it.
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

// The reference's molecule through the real registry to the Scf's inputs.
// The Scf itself is run by hand: tau_j is not a TOML key.
std::optional<fb::Calculation> run_setup_steps(const fb::json &entry, fb::NodeResources &node) {
  const fb::test::ScratchDir dir("active_space");
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
  auto calc = fb::read_config(dir / "run.toml");
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

// The Scf the comparisons need: unscreened J (tau_j = 0; at the default, J's
// screen moves N2/cc-pVDZ's energy by 2.6e-10 Eh and an h_eff_tt by 1.4e-10),
// every build full, and converged past the step's defaults, since h_eff_tt
// moves to first order in the orbitals. K's default screen does not bind here.
const fb::ScfStep kExactScf{.energy_tolerance = 1e-13,
                            .commutator_tolerance = 1e-12,
                            .fock_rebuild_interval = 1,
                            .tau_j = 0.0};

// Everything active_space_hamiltonian reads, from one converged Scf, and the
// Scf's orbitals reordered to the reference's core, active, rest.
struct Setup {
  fb::NodeResources node;
  std::optional<fb::Calculation> calc;
  const fb::Molecule *molecule = nullptr;
  const fb::BasisSet *basis = nullptr;
  const fb::SphericalTransform *transform = nullptr;
  const fb::CoreHamiltonian *h = nullptr;
  std::optional<fb::Scf> scf;
  std::vector<fb::CoulombPairClass> j_classes;
  std::vector<fb::CoulombPairClass> j_classes_unpruned;
  std::vector<fb::OrderedPairClass> k_classes;
  std::optional<fb::SphericalMatrixAO> c;
  std::int32_t ncore = 0;
  std::int32_t ncas = 0;

  [[nodiscard]] fb::DeviceResources &res() { return node.device(0); }

  [[nodiscard]] fb::Result<fb::ActiveSpaceHamiltonian> build(const fb::SphericalMatrixAO &orbitals,
                                                             std::int32_t core, std::int32_t cas,
                                                             double tau_j, double tau_k,
                                                             bool unpruned = false) {
    return fb::active_space_hamiltonian(res(), *basis, *molecule, h->h, transform->u,
                                        unpruned ? j_classes_unpruned : j_classes, k_classes,
                                        orbitals, core, cas, tau_j, tau_k);
  }

  [[nodiscard]] fb::Result<fb::CasscfIntegrals> build_casscf(const fb::SphericalMatrixAO &orbitals,
                                                             double tau_j, double tau_k,
                                                             bool unpruned = false) {
    return fb::casscf_integrals(res(), *basis, *molecule, h->h, transform->u,
                                unpruned ? j_classes_unpruned : j_classes, k_classes, orbitals,
                                ncore, ncas, tau_j, tau_k);
  }
};

std::unique_ptr<Setup> make_setup(const fb::json &entry) {
  auto setup = std::make_unique<Setup>();
  auto calc = run_setup_steps(entry, setup->node);
  if (!calc.has_value()) {
    return nullptr;
  }
  setup->calc.emplace(std::move(*calc)); // a Pipeline moves, but is not move-assigned
  const auto &molecule = setup->calc->get<fb::Molecule>();
  const auto &basis = setup->calc->get<fb::BasisSet>();
  const auto &transform = setup->calc->get<fb::SphericalTransform>();
  const auto &pairs = setup->calc->get<fb::PrimitivePairClasses>();
  const auto &overlap = setup->calc->get<fb::Overlap>();
  const auto &x = setup->calc->get<fb::Orthogonalizer>();
  const auto &h = setup->calc->get<fb::CoreHamiltonian>();
  const auto &guess = setup->calc->get<fb::Guess>();
  if (!molecule.has_value() || !basis.has_value() || !transform.has_value() || !pairs.has_value() ||
      !overlap.has_value() || !x.has_value() || !h.has_value() || !guess.has_value()) {
    ADD_FAILURE() << "a product of the setup steps is missing";
    return nullptr;
  }
  setup->molecule = &*molecule;
  setup->basis = &*basis;
  setup->transform = &*transform;
  setup->h = &*h;
  auto scf = kExactScf.run(
      fb::StepInput<fb::ScfStep>{*molecule, *basis, *transform, *pairs, *overlap, *x, *h, *guess},
      setup->res());
  if (!scf.has_value()) {
    ADD_FAILURE() << scf.error().message;
    return nullptr;
  }
  setup->scf.emplace(std::move(*scf));

  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  auto j_classes = fb::coulomb_pair_classes(res, pairs->classes);
  auto j_classes_unpruned = fb::coulomb_pair_classes(res, pairs->classes, 0.0);
  auto k_classes = fb::ordered_pair_classes(res, *basis, pairs->classes);
  if (!j_classes.has_value() || !j_classes_unpruned.has_value() || !k_classes.has_value()) {
    ADD_FAILURE() << "the J or K classes failed";
    return nullptr;
  }
  setup->j_classes = std::move(*j_classes);
  setup->j_classes_unpruned = std::move(*j_classes_unpruned);
  setup->k_classes = std::move(*k_classes);

  // Our orbitals pair with PySCF's by index only if the two agree on the
  // order, which the orbital energies show.
  const auto mo_energies = entry.at("mo_energies").get<std::vector<double>>();
  EXPECT_EQ(setup->scf->mo_energies.size(), mo_energies.size());
  for (std::size_t i = 0; i < std::min(mo_energies.size(), setup->scf->mo_energies.size()); ++i) {
    EXPECT_NEAR(setup->scf->mo_energies[i], mo_energies[i], kOrbitalEnergyTolerance)
        << "orbital " << i;
  }

  // C's columns in the reference's order: core, active, the rest.
  const std::size_t n = basis->nao_sphe();
  const auto orbitals = entry.at("orbitals").get<std::vector<std::size_t>>();
  EXPECT_EQ(orbitals.size(), n);
  const std::vector<double> c = download(res, setup->scf->c.data(), n * n);
  std::vector<double> reordered(n * n);
  for (std::size_t col = 0; col < std::min(n, orbitals.size()); ++col) {
    std::copy_n(c.begin() + static_cast<std::ptrdiff_t>(n * orbitals[col]), n,
                reordered.begin() + static_cast<std::ptrdiff_t>(n * col));
  }
  setup->c.emplace(res, *basis);
  upload(res, reordered, setup->c->data());
  setup->ncore = entry.at("ncore").get<std::int32_t>();
  setup->ncas = entry.at("ncas").get<std::int32_t>();
  return setup;
}

// The Hamiltonian on the host.
struct Host {
  double e_core;
  std::size_t n;
  std::vector<double> h_eff;
  std::vector<double> eri;

  [[nodiscard]] double h(std::size_t t, std::size_t u) const { return h_eff[t + (n * u)]; }
  [[nodiscard]] double g(std::size_t t, std::size_t u, std::size_t v, std::size_t w) const {
    return eri[t + (n * (u + (n * (v + (n * w)))))];
  }
};

Host to_host(fb::DeviceResources &res, const fb::ActiveSpaceHamiltonian &ham) {
  const auto n = static_cast<std::size_t>(ham.ncas);
  return {.e_core = ham.e_core,
          .n = n,
          .h_eff = download(res, ham.h_eff.data(), n * n),
          .eri = download(res, ham.eri.data(), n * n * n * n)};
}

// ppaa and papa on the host: (p, q) of block (u, v) at
// p + nmo (q + nmo (u + ncas v)).
struct HostOrbital {
  std::size_t nmo;
  std::size_t ncas;
  std::vector<double> ppaa;
  std::vector<double> papa;

  [[nodiscard]] std::size_t at(std::size_t p, std::size_t q, std::size_t u, std::size_t v) const {
    return p + (nmo * (q + (nmo * (u + (ncas * v)))));
  }
};

HostOrbital to_host(fb::DeviceResources &res, const fb::CasscfIntegrals &ints, std::size_t nmo) {
  const auto ncas = static_cast<std::size_t>(ints.hamiltonian.ncas);
  const std::size_t count = nmo * nmo * ncas * ncas;
  EXPECT_EQ(ints.ppaa.num_elements(), count);
  EXPECT_EQ(ints.papa.num_elements(), count);
  return {.nmo = nmo,
          .ncas = ncas,
          .ppaa = download(res, ints.ppaa.data(), count),
          .papa = download(res, ints.papa.data(), count)};
}

double max_abs(const std::vector<double> &v) {
  double m = 0.0;
  for (const double x : v) {
    m = std::max(m, std::abs(x));
  }
  return m;
}

struct Case {
  std::string_view name;
  bool screened; // the defaults, or tau = 0 for J and K both
};

// ctest names a discovered case by this, so `ctest -R n2_ccpvdz_tau0` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name << (c.screened ? "_default" : "_tau0");
}

class ActiveSpaceGpu : public ::testing::TestWithParam<Case> {};

TEST_P(ActiveSpaceGpu, MatchesPyscfGivesBackTheRhfEnergyAndIsSymmetric) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name);
  auto setup = make_setup(entry);
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  // The screened case takes the function's own defaults, whatever they are.
  auto ham =
      GetParam().screened
          ? fb::active_space_hamiltonian(setup->res(), *setup->basis, *setup->molecule, setup->h->h,
                                         setup->transform->u, setup->j_classes, setup->k_classes,
                                         *setup->c, setup->ncore, setup->ncas)
          : setup->build(*setup->c, setup->ncore, setup->ncas, 0.0, 0.0);
  ASSERT_TRUE(ham.has_value()) << ham.error().message;
  ASSERT_EQ(ham->ncas, setup->ncas);
  const Host host = to_host(setup->res(), *ham);
  const std::size_t n = host.n;

  // Against PySCF: E_core and the elements no orbital's sign changes.
  EXPECT_NEAR(host.e_core, entry.at("e_core").get<double>(), kPyscfTolerance);
  const auto h_diag = entry.at("h_eff_diagonal").get<std::vector<double>>();
  const auto coulomb = entry.at("coulomb").get<std::vector<std::vector<double>>>();
  const auto exchange = entry.at("exchange").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(h_diag.size(), n);
  for (std::size_t t = 0; t < n; ++t) {
    EXPECT_NEAR(host.h(t, t), h_diag[t], kPyscfTolerance) << "h_eff " << t;
    for (std::size_t u = 0; u < n; ++u) {
      EXPECT_NEAR(host.g(t, t, u, u), coulomb[t][u], kPyscfTolerance) << "(tt|uu) " << t << u;
      EXPECT_NEAR(host.g(t, u, t, u), exchange[t][u], kPyscfTolerance) << "(tu|tu) " << t << u;
    }
  }

  // The RHF energy, from the core and the active occupied orbitals: those of
  // the active columns that were occupied in the Scf.
  const auto orbitals = entry.at("orbitals").get<std::vector<std::size_t>>();
  const std::size_t nocc = entry.at("electrons").get<std::size_t>() / 2;
  std::vector<std::size_t> occupied;
  for (std::size_t t = 0; t < n; ++t) {
    if (orbitals[static_cast<std::size_t>(setup->ncore) + t] < nocc) {
      occupied.push_back(t);
    }
  }
  ASSERT_FALSE(occupied.empty());
  ASSERT_LT(occupied.size(), n) << "the active space must span virtual orbitals too";
  double energy = host.e_core;
  for (const std::size_t i : occupied) {
    energy += 2.0 * host.h(i, i);
    for (const std::size_t j : occupied) {
      energy += (2.0 * host.g(i, i, j, j)) - host.g(i, j, i, j);
    }
  }
  EXPECT_NEAR(energy, setup->scf->energy, 1e-10);

  // The 8-fold permutational symmetry of real orbitals' (tu|vw).
  const double tolerance = 1e-12 * max_abs(host.eri);
  for (std::size_t w = 0; w < n; ++w) {
    for (std::size_t v = 0; v < n; ++v) {
      for (std::size_t u = 0; u < n; ++u) {
        for (std::size_t t = 0; t < n; ++t) {
          const double x = host.g(t, u, v, w);
          EXPECT_NEAR(host.g(u, t, v, w), x, tolerance);
          EXPECT_NEAR(host.g(t, u, w, v), x, tolerance);
          EXPECT_NEAR(host.g(v, w, t, u), x, tolerance) << t << u << v << w;
        }
      }
    }
  }
}

TEST_P(ActiveSpaceGpu, CasscfIntegralsMatchPyscfTheHamiltonianAndAreSymmetric) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name);
  auto setup = make_setup(entry);
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const bool screened = GetParam().screened;
  // The screened case takes the functions' own defaults, whatever they are.
  auto ints = screened
                  ? fb::casscf_integrals(res, *setup->basis, *setup->molecule, setup->h->h,
                                         setup->transform->u, setup->j_classes, setup->k_classes,
                                         *setup->c, setup->ncore, setup->ncas)
                  : setup->build_casscf(*setup->c, 0.0, 0.0);
  auto ham = screened ? fb::active_space_hamiltonian(res, *setup->basis, *setup->molecule,
                                                     setup->h->h, setup->transform->u,
                                                     setup->j_classes, setup->k_classes, *setup->c,
                                                     setup->ncore, setup->ncas)
                      : setup->build(*setup->c, setup->ncore, setup->ncas, 0.0, 0.0);
  ASSERT_TRUE(ints.has_value()) << ints.error().message;
  ASSERT_TRUE(ham.has_value()) << ham.error().message;
  const std::size_t nmo = setup->basis->nao_sphe();
  const HostOrbital host = to_host(res, *ints, nmo);
  const Host active = to_host(res, *ham);
  const std::size_t n = active.n;
  ASSERT_EQ(host.ncas, n);
  const auto core = static_cast<std::size_t>(setup->ncore);

  // The Hamiltonian it carries is active_space_hamiltonian's.
  const Host carried = to_host(res, ints->hamiltonian);
  EXPECT_NEAR(carried.e_core, active.e_core, 1e-12 * std::abs(active.e_core));
  const double eri_tolerance = 1e-12 * max_abs(active.eri);
  for (std::size_t k = 0; k < active.eri.size(); ++k) {
    EXPECT_NEAR(carried.eri[k], active.eri[k], eri_tolerance) << k;
  }

  // Against PySCF: sum (pp|uu) and sum (pu|pu) over p's degenerate set and
  // u's, which no orbital's sign, nor any rotation within a set, changes.
  const auto sets = entry.at("orbital_sets").get<std::vector<std::vector<std::size_t>>>();
  const auto active_sets = entry.at("active_sets").get<std::vector<std::vector<std::size_t>>>();
  const auto pp_uu = entry.at("ppaa_diagonal").get<std::vector<std::vector<double>>>();
  const auto pu_pu = entry.at("papa_diagonal").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(pp_uu.size(), sets.size());
  ASSERT_EQ(pu_pu.size(), sets.size());
  for (std::size_t a = 0; a < sets.size(); ++a) {
    ASSERT_EQ(pp_uu[a].size(), active_sets.size());
    ASSERT_EQ(pu_pu[a].size(), active_sets.size());
    for (std::size_t b = 0; b < active_sets.size(); ++b) {
      double coulomb = 0.0;
      double exchange = 0.0;
      for (const std::size_t p : sets[a]) {
        for (const std::size_t u : active_sets[b]) {
          coulomb += host.ppaa[host.at(p, p, u, u)];
          exchange += host.papa[host.at(p, p, u, u)];
        }
      }
      EXPECT_NEAR(coulomb, pp_uu[a][b], kPyscfTolerance) << "(pp|uu) set " << a << " " << b;
      EXPECT_NEAR(exchange, pu_pu[a][b], kPyscfTolerance) << "(pu|pu) set " << a << " " << b;
    }
  }

  // The active blocks: (tu|vw) is ppaa's (t, u) of block (v, w), and papa's
  // (t, v) of block (u, w), (pu|qv) at p = t, q = v.
  for (std::size_t w = 0; w < n; ++w) {
    for (std::size_t v = 0; v < n; ++v) {
      for (std::size_t u = 0; u < n; ++u) {
        for (std::size_t t = 0; t < n; ++t) {
          const double x = active.g(t, u, v, w);
          EXPECT_NEAR(host.ppaa[host.at(core + t, core + u, v, w)], x, eri_tolerance)
              << "ppaa " << t << u << v << w;
          EXPECT_NEAR(host.papa[host.at(core + t, core + v, u, w)], x, eri_tolerance)
              << "papa " << t << u << v << w;
        }
      }
    }
  }

  // ppaa: (pq|uv) = (qp|uv) = (pq|vu). papa: (pu|qv) = (qv|pu).
  const double pp_tolerance = 1e-12 * max_abs(host.ppaa);
  const double pa_tolerance = 1e-12 * max_abs(host.papa);
  for (std::size_t v = 0; v < n; ++v) {
    for (std::size_t u = 0; u < n; ++u) {
      for (std::size_t q = 0; q < nmo; ++q) {
        for (std::size_t p = 0; p < nmo; ++p) {
          const double x = host.ppaa[host.at(p, q, u, v)];
          EXPECT_NEAR(host.ppaa[host.at(q, p, u, v)], x, pp_tolerance);
          EXPECT_NEAR(host.ppaa[host.at(p, q, v, u)], x, pp_tolerance);
          EXPECT_NEAR(host.papa[host.at(q, p, v, u)], host.papa[host.at(p, q, u, v)], pa_tolerance)
              << p << " " << q << " " << u << " " << v;
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Cases, ActiveSpaceGpu,
                         ::testing::Values(Case{"h2o_sto3g", false}, Case{"h2o_sto3g", true},
                                           Case{"h2o_ccpvdz", false}, Case{"h2o_ccpvdz", true},
                                           Case{"n2_ccpvdz", false}, Case{"n2_ccpvdz", true}),
                         [](const auto &info) {
                           return std::format("{}_{}", info.param.name,
                                              info.param.screened ? "default" : "tau0");
                         });

TEST(ActiveSpaceGpuCap, ScaledOrbitalsOnPrunedClassesMatchUnprunedOnes) {
  // Orbitals scaled so that every m_v m_w is past the prune's |D| <= 2: the
  // pruned classes, with the cap, give unpruned classes' (tu|vw) at tau = 0.
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_ccpvdz"));
  ASSERT_NE(setup, nullptr);
  fb::DeviceResources &res = setup->res();
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::BasisSet &basis = *setup->basis;
  const std::size_t n = basis.nao_sphe();
  const std::size_t n_cart = basis.nao_cart();
  constexpr double kScale = 8.0;
  std::vector<double> c = download(res, setup->c->data(), n * n);
  for (double &x : c) {
    x *= kScale;
  }
  fb::SphericalMatrixAO scaled(res, basis);
  upload(res, c, scaled.data());

  // m_v = max_mu |(U C)_mu v| over the active columns: every product past 2.
  ASSERT_NE(setup->transform->u, nullptr);
  const std::vector<double> u = download(res, setup->transform->u->data(), n_cart * n);
  double smallest = std::numeric_limits<double>::infinity();
  for (std::size_t v = static_cast<std::size_t>(setup->ncore);
       v < static_cast<std::size_t>(setup->ncore + setup->ncas); ++v) {
    double m_v = 0.0;
    for (std::size_t mu = 0; mu < n_cart; ++mu) {
      double x = 0.0;
      for (std::size_t k = 0; k < n; ++k) {
        x += u[mu + (n_cart * k)] * c[k + (n * v)];
      }
      m_v = std::max(m_v, std::abs(x));
    }
    smallest = std::min(smallest, m_v);
  }
  ASSERT_GT(smallest * smallest, 2.0);

  auto pruned = setup->build(scaled, setup->ncore, setup->ncas, 0.0, 0.0);
  auto unpruned = setup->build(scaled, setup->ncore, setup->ncas, 0.0, 0.0, /*unpruned=*/true);
  ASSERT_TRUE(pruned.has_value()) << pruned.error().message;
  ASSERT_TRUE(unpruned.has_value()) << unpruned.error().message;
  const Host a = to_host(res, *pruned);
  const Host b = to_host(res, *unpruned);
  const double tolerance = 1e-12 * max_abs(b.eri);
  ASSERT_EQ(a.eri.size(), b.eri.size());
  for (std::size_t k = 0; k < a.eri.size(); ++k) {
    EXPECT_NEAR(a.eri[k], b.eri[k], tolerance) << k;
  }

  // casscf_integrals' J is capped the same: ppaa on the pruned classes is the
  // unpruned classes'. K's classes are not pruned, but K is capped as J is:
  // papa of the scaled orbitals is kScale^4 times the unscaled orbitals',
  // whose m_v m_w (below the cap or not) take another s.
  auto pruned_ints = setup->build_casscf(scaled, 0.0, 0.0);
  auto unpruned_ints = setup->build_casscf(scaled, 0.0, 0.0, /*unpruned=*/true);
  auto unscaled_ints = setup->build_casscf(*setup->c, 0.0, 0.0);
  ASSERT_TRUE(pruned_ints.has_value()) << pruned_ints.error().message;
  ASSERT_TRUE(unpruned_ints.has_value()) << unpruned_ints.error().message;
  ASSERT_TRUE(unscaled_ints.has_value()) << unscaled_ints.error().message;
  const HostOrbital x = to_host(res, *pruned_ints, n);
  const HostOrbital y = to_host(res, *unpruned_ints, n);
  const HostOrbital z = to_host(res, *unscaled_ints, n);
  const double pp_tolerance = 1e-12 * max_abs(y.ppaa);
  for (std::size_t k = 0; k < x.ppaa.size(); ++k) {
    EXPECT_NEAR(x.ppaa[k], y.ppaa[k], pp_tolerance) << k;
  }
  constexpr double kScale4 = kScale * kScale * kScale * kScale;
  const double pa_tolerance = 1e-12 * max_abs(x.papa);
  for (std::size_t k = 0; k < x.papa.size(); ++k) {
    EXPECT_NEAR(x.papa[k], kScale4 * z.papa[k], pa_tolerance) << k;
  }
}

TEST(ActiveSpaceGpuRefusals, RefusesAnActiveSpacePastTheOrbitalsOrEmpty) {
  FBF_REQUIRE_GPU();
  auto setup = make_setup(reference_case("h2o_sto3g"));
  ASSERT_NE(setup, nullptr);
  const fb::ScopedDeviceIndex on{setup->res().dev_idx()};
  const auto nmo = static_cast<std::int32_t>(setup->basis->nao_sphe());
  const auto refused = [&](std::int32_t ncore, std::int32_t ncas, double tau = 0.0) {
    auto ham = setup->build(*setup->c, ncore, ncas, tau, tau);
    return !ham.has_value() && ham.error().kind == fb::ErrorKind::InvalidConfig;
  };
  EXPECT_TRUE(refused(1, nmo));     // ncore + ncas > nmo
  EXPECT_TRUE(refused(0, nmo + 1)); // ncas > nmo
  EXPECT_TRUE(refused(1, 0));       // ncas = 0
  EXPECT_TRUE(refused(-1, 2));
  EXPECT_TRUE(refused(0, 2, -1.0));
  EXPECT_TRUE(refused(0, 2, std::numeric_limits<double>::quiet_NaN()));
  // The edges it takes: FCI, ncore = 0 and ncas = nmo, and every orbital used.
  auto fci = setup->build(*setup->c, 0, nmo, 0.0, 0.0);
  ASSERT_TRUE(fci.has_value()) << fci.error().message;
  EXPECT_NEAR(fci->e_core, fb::nuclear_repulsion_energy(*setup->molecule), 1e-12);
  EXPECT_TRUE(setup->build(*setup->c, 1, nmo - 1, 0.0, 0.0).has_value());
  // casscf_integrals refuses alike.
  setup->ncore = 1;
  setup->ncas = nmo;
  auto past = setup->build_casscf(*setup->c, 0.0, 0.0);
  EXPECT_TRUE(!past.has_value() && past.error().kind == fb::ErrorKind::InvalidConfig);
  setup->ncore = 0;
  setup->ncas = 2;
  auto negative = setup->build_casscf(*setup->c, -1.0, 0.0);
  EXPECT_TRUE(!negative.has_value() && negative.error().kind == fb::ErrorKind::InvalidConfig);
}

} // namespace
