// The Casscf step against PySCF's converged, polished mcscf.CASSCF
// (data/casscf/casscf.json, tools/emit/casscf_reference.py), through the real
// registry: a config naming [casscf] plans the whole chain to the ScfOutput,
// picks the start (by index, or by AVAS), optimizes it and canonicalizes the
// core and the virtuals. Compared: each root's energy and the weighted one,
// the state-averaged natural occupations, and the canonical core and virtual
// orbital energies, the state-averaged Fock's eigenvalues in each space. Each
// case also writes its natural orbitals as Molden, read back as
// io_molden_host reads one. Each case runs again with [eri] density-fitted,
// against the DF-CASSCF of data/casscf/casscf_df.json; so do that file's
// "rohf" cases, a doublet and a triplet from the Scf step's DF-ROHF at their
// multiplicity, which have no exact counterpart. And the start from another
// run's .npy files (orbitals, CI): at the same geometry, at a displaced one,
// and the files it refuses. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The issue's bars.
constexpr double kEnergyTolerance = 1e-8;
constexpr double kOccupationTolerance = 1e-6;
constexpr double kOrbitalEnergyTolerance = 1e-6;
// Each root's energy, the natural occupations and the orbital energies are
// first order in the orbitals' error, ~||g|| / (H's least eigenvalue): at 1e-9,
// as mcscf_solve_gpu.cpp runs, ~1e-8 (see src/mcscf/solve/README.md).
constexpr double kConvTolGrad = 1e-9;

// The density-fitted references' fitting set, compiled in, by the name
// casscf_df.json records.
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";

// data/casscf/<file>, read once per file.
const fb::json &reference(std::string_view file) {
  static std::map<std::string, fb::json, std::less<>> docs;
  if (const auto it = docs.find(file); it != docs.end()) {
    return it->second;
  }
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf" / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return docs.emplace(file, read.has_value() ? std::move(*read) : fb::json{}).first->second;
}

const fb::json &reference_case(std::string_view name, std::string_view file = "casscf.json") {
  for (const fb::json &entry : reference(file).at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in casscf/" << file;
  static const fb::json kNone;
  return kNone;
}

// The reference's molecule written as an XYZ file in dir, and a config naming
// it with the given tables after it; at its multiplicity for an "rohf" case,
// whose Scf step is then ROHF.
std::filesystem::path write_config(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view tables) {
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
  const std::int32_t multiplicity = entry.value("scf", std::string{"rhf"}) == "rohf"
                                        ? entry.at("multiplicity").get<std::int32_t>()
                                        : 1;
  const std::string toml = std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n"
                                       "multiplicity = {}\n[basis_set]\nbasis = \"{}\"\n{}",
                                       entry.at("charge").get<std::int32_t>(), multiplicity,
                                       entry.at("basis").get<std::string>(), tables);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The [casscf] table of the reference's start and weights.
std::string casscf_table(const fb::json &entry) {
  std::string table = std::format("[casscf]\nconv_tol_grad = {:g}\n", kConvTolGrad);
  if (!entry.at("avas").is_null()) {
    table += std::format("avas_threshold = {:.17g}\navas = [",
                         entry.at("avas").at("threshold").get<double>());
    const auto labels = entry.at("avas").at("labels").get<std::vector<std::string>>();
    for (std::size_t i = 0; i < labels.size(); ++i) {
      table += std::format("{}\"{}\"", i == 0 ? "" : ", ", labels[i]);
    }
    table += "]\n";
  } else {
    table += std::format("ncas = {}\nnelecas = {}\n", entry.at("ncas").get<std::int32_t>(),
                         entry.at("nelecas").get<std::int32_t>());
    if (!entry.at("active").is_null()) {
      const auto active = entry.at("active").get<std::vector<std::int32_t>>();
      table += "active = [";
      for (std::size_t i = 0; i < active.size(); ++i) {
        table += std::format("{}{}", i == 0 ? "" : ", ", active[i]);
      }
      table += "]\n";
    }
  }
  const auto weights = entry.at("weights").get<std::vector<double>>();
  table += std::format("nroots = {}\nweights = [", weights.size());
  for (std::size_t i = 0; i < weights.size(); ++i) {
    table += std::format("{}{:.17g}", i == 0 ? "" : ", ", weights[i]);
  }
  return table + "]\n";
}

// The eigenvalues of a symmetric n x n matrix, descending: cyclic Jacobi.
std::vector<double> eigenvalues(std::vector<double> a, std::size_t n) {
  for (std::int32_t sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        off += a[(p * n) + q] * a[(p * n) + q];
      }
    }
    if (off < 1e-30) {
      break;
    }
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        const double apq = a[(p * n) + q];
        if (apq == 0.0) {
          continue;
        }
        const double theta = (a[(q * n) + q] - a[(p * n) + p]) / (2.0 * apq);
        const double t =
            std::copysign(1.0, theta) / (std::abs(theta) + std::sqrt((theta * theta) + 1.0));
        const double c = 1.0 / std::sqrt((t * t) + 1.0);
        const double s = t * c;
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[(k * n) + p];
          const double akq = a[(k * n) + q];
          a[(k * n) + p] = (c * akp) - (s * akq);
          a[(k * n) + q] = (s * akp) + (c * akq);
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = a[(p * n) + k];
          const double aqk = a[(q * n) + k];
          a[(p * n) + k] = (c * apk) - (s * aqk);
          a[(q * n) + k] = (s * apk) + (c * aqk);
        }
      }
    }
  }
  std::vector<double> values(n);
  for (std::size_t i = 0; i < n; ++i) {
    values[i] = a[(i * n) + i];
  }
  std::ranges::sort(values, std::greater<>{});
  return values;
}

// Each orbital's Occup= of the Molden file at path, and its coefficient count.
std::pair<std::vector<double>, std::size_t> molden_occupations(const std::filesystem::path &path) {
  const auto text = fb::read_file(path);
  EXPECT_TRUE(text.has_value()) << text.error().message;
  if (!text.has_value()) {
    return {};
  }
  EXPECT_TRUE(text->starts_with("[Molden Format]\n"));
  const std::size_t mo_at = text->find("[MO]");
  EXPECT_NE(mo_at, std::string::npos);
  if (mo_at == std::string::npos) {
    return {};
  }
  std::vector<double> occupations;
  std::size_t coefficients = 0;
  std::istringstream in{text->substr(mo_at)};
  std::string line;
  while (std::getline(in, line)) {
    std::int32_t index = 0;
    double value = 0.0;
    std::istringstream fields(line);
    if (line.starts_with(" Occup=")) {
      occupations.push_back(std::stod(line.substr(7)));
    } else if (fields >> index >> value) {
      ++coefficients;
    }
  }
  return {occupations, coefficients};
}

struct Case {
  std::string_view name;
  // Under slow_test.h: every case but H2O cc-pVDZ, the cheapest. H2O 6-31G is
  // the PySCF comparison's fast case (#1008), and slow everywhere else.
  bool slow;
};

// ctest names a discovered case by this, so `ctest -R n2_ccpvdz` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

// Runs the config of entry's molecule and the [casscf] table casscf, and
// checks the step's product against entry's reference.
void expect_matches(const fb::json &entry, const std::string &casscf_toml) {
  const fb::json &want = entry.at("casscf");
  const fb::test::ScratchDir dir("pipeline_casscf");
  const std::filesystem::path molden = dir / "out/casscf.molden";
  auto calc = fb::read_config(write_config(dir, entry, casscf_toml), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  ASSERT_TRUE(casscf.has_value() && basis.has_value());
  std::cout << std::format("[ casscf   ] {}: {} macro iterations, |g| {:.3e}\n",
                           entry.at("name").get<std::string>(), casscf->iterations.size(),
                           casscf->gradient_norm);

  EXPECT_EQ(casscf->ncore, entry.at("ncore").get<std::int32_t>());
  EXPECT_EQ(casscf->ncas, entry.at("ncas").get<std::int32_t>());
  EXPECT_EQ(casscf->nelecas, entry.at("nelecas").get<std::int32_t>());
  EXPECT_LT(casscf->gradient_norm, kConvTolGrad);

  // The energies: each root's and the weighted one; each root of the
  // molecule's spin S, <S^2> = S(S + 1).
  const double s = (entry.value("multiplicity", std::int32_t{1}) - 1) / 2.0;
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto energies = want.at("energies").get<std::vector<double>>();
  ASSERT_EQ(casscf->energies.size(), weights.size());
  ASSERT_EQ(casscf->spin_squares.size(), weights.size());
  EXPECT_EQ(casscf->weights, weights);
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_NEAR(casscf->energies[r], energies[r], kEnergyTolerance) << "root " << r;
    EXPECT_NEAR(casscf->spin_squares[r], s * (s + 1.0), 1e-8) << "root " << r;
  }
  EXPECT_NEAR(casscf->energy, want.at("energy").get<double>(), kEnergyTolerance);

  // The state-averaged natural occupations, of the weighted 1-RDMs; and each
  // CI vector of unit norm.
  const auto n = static_cast<std::size_t>(casscf->ncas);
  ASSERT_EQ(casscf->rdm1.size(), weights.size() * n * n);
  ASSERT_EQ(casscf->vectors.size() % weights.size(), 0U);
  const std::size_t dim = casscf->vectors.size() / weights.size();
  std::vector<double> gamma(n * n, 0.0);
  for (std::size_t r = 0; r < weights.size(); ++r) {
    double norm = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      norm += casscf->vectors[(r * dim) + i] * casscf->vectors[(r * dim) + i];
    }
    EXPECT_NEAR(norm, 1.0, 1e-12) << "root " << r;
    for (std::size_t i = 0; i < n * n; ++i) {
      gamma[i] += weights[r] * casscf->rdm1[(r * n * n) + i];
    }
  }
  const auto occupations = want.at("natural_occupations").get<std::vector<double>>();
  const std::vector<double> ours = eigenvalues(gamma, n);
  ASSERT_EQ(ours.size(), occupations.size());
  for (std::size_t k = 0; k < n; ++k) {
    EXPECT_NEAR(ours[k], occupations[k], kOccupationTolerance) << "occupation " << k;
  }

  // The canonical core and virtual orbital energies, ascending in each space.
  const std::size_t nao = basis->basis.nao_sphe();
  ASSERT_EQ(casscf->mo_energies.size(), nao);
  const auto ncore = static_cast<std::size_t>(casscf->ncore);
  const auto core = want.at("orbital_energies").at("core").get<std::vector<double>>();
  const auto virtuals = want.at("orbital_energies").at("virtual").get<std::vector<double>>();
  ASSERT_EQ(core.size(), ncore);
  ASSERT_EQ(virtuals.size(), nao - ncore - n);
  for (std::size_t i = 0; i < ncore; ++i) {
    EXPECT_NEAR(casscf->mo_energies[i], core[i], kOrbitalEnergyTolerance) << "core " << i;
  }
  for (std::size_t i = 0; i < virtuals.size(); ++i) {
    EXPECT_NEAR(casscf->mo_energies[ncore + n + i], virtuals[i], kOrbitalEnergyTolerance)
        << "virtual " << i;
  }

  // The Molden file: every orbital, the core at 2, the natural ones at the
  // natural occupations, descending, the rest at 0. Occup= has six decimals.
  const auto [written, coefficients] = molden_occupations(molden);
  ASSERT_EQ(written.size(), nao);
  EXPECT_EQ(coefficients, nao * nao);
  for (std::size_t j = 0; j < nao; ++j) {
    const double expected = j < ncore ? 2.0 : (j < ncore + n ? occupations[j - ncore] : 0.0);
    EXPECT_NEAR(written[j], expected, 1e-6) << "orbital " << j;
  }
}

class PipelineCasscf : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineCasscf, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  // Under slow_test.h: every case but H2O 6-31G, the family's representative
  // (#1008). H2O cc-pVDZ integral-direct costs seconds here, as McscfSolve's
  // fast case does over the same reference.
  FBF_FAMILY_REPRESENTATIVE("h2o_631g");
  if (GetParam().name != "h2o_631g") {
    FBF_SKIP_IF_FAST_TESTS();
  }
  const fb::json &entry = reference_case(GetParam().name);
  expect_matches(entry, casscf_table(entry));
}

// [eri] mode = "density_fitting" over the compiled-in cc-pVDZ-JKFIT, against
// PySCF's DF-CASSCF (data/casscf/casscf_df.json), to the same bars: the step
// runs mcscf::DensityFitting::Full over the pipeline's B and canonicalizes
// over the fitted Fock, whose eigenvalues the reference's orbital energies
// are. One [eri] serves the whole pipeline, so the start is the DF-RHF's
// orbitals where the reference's is the exact RHF's: the same active
// orbitals, and the converged DF-CASSCF is the start's only through which
// stationary point it finds.
void expect_matches_density_fitted(std::string_view name) {
  const fb::json &entry = reference_case(name, "casscf_df.json");
  ASSERT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
  expect_matches(entry, casscf_table(entry) +
                            std::format("[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                                        kDfAuxBasis));
}

TEST_P(PipelineCasscf, MatchesPyscfDensityFitted) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE("h2o_ccpvdz");
  if (GetParam().slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  expect_matches_density_fitted(GetParam().name);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineCasscf,
                         ::testing::Values(Case{.name = "h2o_631g", .slow = true},
                                           Case{.name = "h2o_ccpvdz", .slow = false},
                                           Case{.name = "n2_ccpvdz", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2", .slow = true},
                                           Case{.name = "h2co_ccpvdz_sa2_weighted", .slow = true},
                                           Case{.name = "h2o_ccpvdz_avas", .slow = true}),
                         [](const auto &info) { return std::string(info.param.name); });

// From the Scf step's DF-ROHF, at the molecule's multiplicity, against PySCF's
// DF-CASSCF from its DF-ROHF (casscf_df.json's "rohf" cases), to the same
// bars: the core doubly occupied, the singly occupied orbitals active, the
// roots of the molecule's spin. Density-fitted only, as the ROHF is.
class PipelineCasscfOverRohf : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineCasscfOverRohf, MatchesPyscfDensityFitted) {
  FBF_REQUIRE_GPU();
  if (GetParam().slow) {
    FBF_SKIP_IF_FAST_TESTS();
  }
  expect_matches_density_fitted(GetParam().name);
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineCasscfOverRohf,
                         ::testing::Values(Case{.name = "nh2_ccpvdz_rohf", .slow = false},
                                           Case{.name = "ch2_ccpvdz_rohf", .slow = false}),
                         [](const auto &info) { return std::string(info.param.name); });

// max_cycle running out fails the step, and the error is its product.
TEST(PipelineCasscf, OutOfMacroIterationsIsMaxIterationsExceeded) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz");
  const fb::test::ScratchDir dir("pipeline_casscf");
  auto calc =
      fb::read_config(write_config(dir, entry, "[casscf]\nncas = 4\nnelecas = 4\nmax_cycle = 1\n"),
                      dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::MaxIterationsExceeded) << ran.error().message;
  ASSERT_TRUE(calc->get<fb::ScfOutput>().has_value());
  EXPECT_FALSE(calc->get<fb::CasscfOutput>().has_value());
}

// The start from another run's .npy files. H2O cc-pVDZ, density-fitted over
// cc-pVDZ-JKFIT, CAS(4,4) [2, 3, 5, 6] over two equally weighted roots, so
// the CI file holds more than one column.
constexpr std::string_view kTwoRootTables =
    "[eri]\nmode = \"density_fitting\"\naux_basis = \"cc-pVDZ-JKFIT\"\n"
    "[casscf]\nncas = 4\nnelecas = 4\nnroots = 2\nconv_tol_grad = 1e-9\n";

// The config of entry's molecule and tables run, its output into dir/out.
fb::Result<fb::Calculation> run_in(const fb::test::ScratchDir &dir, const fb::json &entry,
                                   std::string_view tables, std::string_view out) {
  auto calc = fb::read_config(write_config(dir, entry, tables), dir.path(), dir / out);
  if (!calc.has_value()) {
    return calc;
  }
  fb::NodeResources node;
  if (const fb::Status ran = calc->run(node); !ran.has_value()) {
    return std::unexpected(ran.error());
  }
  return calc;
}

// The start a converged run writes, read back at the same geometry: the
// Löwdin step is the identity to rounding there, the orbitals and the CI are
// the solution's, so the energies are the first run's and the loop is done in
// two macro iterations, the fewest it can stop in (it stops on an accepted
// step's energy change, so after a second evaluation).
TEST(PipelineCasscfStart, ItsOwnNpyFilesReproduceARun) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz", "casscf_df.json");
  const fb::test::ScratchDir dir("pipeline_casscf_start");
  auto first = run_in(
      dir, entry, std::string(kTwoRootTables) + "active = [2, 3, 5, 6]\nwrite_npy = true\n", "s0");
  ASSERT_TRUE(first.has_value()) << first.error().message;
  const fb::CasscfOutput &want = *first->get<fb::CasscfOutput>();
  const fb::NpyArray orbitals = *fb::read_npy(dir / "s0/casscf_orbitals.npy");
  const fb::NpyArray ci = *fb::read_npy(dir / "s0/casscf_ci.npy");
  const std::size_t nao = want.c.n();
  EXPECT_EQ(orbitals.shape, (std::vector<std::size_t>{nao, nao}));
  EXPECT_EQ(ci.shape, (std::vector<std::size_t>{want.vectors.size() / 2, 2}));
  EXPECT_EQ(ci.data, want.vectors);

  auto again = run_in(dir, entry,
                      std::string(kTwoRootTables) +
                          "orbitals = \"s0/casscf_orbitals.npy\"\nci = \"s0/casscf_ci.npy\"\n",
                      "again");
  ASSERT_TRUE(again.has_value()) << again.error().message;
  const fb::CasscfOutput &got = *again->get<fb::CasscfOutput>();
  std::cout << std::format("[ casscf   ] round trip: {} macro iterations (from RHF {})\n",
                           got.iterations.size(), want.iterations.size());
  EXPECT_LE(got.iterations.size(), 2U);
  ASSERT_EQ(got.energies.size(), want.energies.size());
  for (std::size_t r = 0; r < got.energies.size(); ++r) {
    EXPECT_NEAR(got.energies[r], want.energies[r], kEnergyTolerance) << "root " << r;
  }
  EXPECT_NEAR(got.energy, want.energy, kEnergyTolerance);
}

// Carried to a displaced geometry, one O-H bond 0.1 bohr longer and the
// other 0.05 shorter, and carried back: the start is followed, not replaced.
// Back at the first geometry the solution is the first run's, to the
// energies' bar, so the displaced one is the state continuously connected to
// it. (A start from the displaced geometry's RHF by the same indices is not
// a check of that: its canonical order there is its own, and from it this
// SA-CASSCF reaches another solution, -75.9010 Eh against -75.8487.)
TEST(PipelineCasscfStart, CarriedToADisplacedGeometryAndBackIsTheSameState) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &entry = reference_case("h2o_ccpvdz", "casscf_df.json");
  const fb::test::ScratchDir dir("pipeline_casscf_start");
  auto first = run_in(
      dir, entry, std::string(kTwoRootTables) + "active = [2, 3, 5, 6]\nwrite_npy = true\n", "s0");
  ASSERT_TRUE(first.has_value()) << first.error().message;

  fb::json displaced = entry;
  fb::json &atoms = displaced.at("atoms");
  const auto o = atoms[0].at("coord").get<std::vector<double>>();
  for (std::size_t a = 1; a < 3; ++a) {
    // Each H along its O-H bond.
    auto h = atoms[a].at("coord").get<std::vector<double>>();
    const double length = std::hypot(h[0] - o[0], h[1] - o[1], h[2] - o[2]);
    const double stretch = a == 1 ? 0.1 : -0.05;
    for (std::size_t k = 0; k < 3; ++k) {
      h[k] += stretch * (h[k] - o[k]) / length;
    }
    atoms[a]["coord"] = h;
  }
  const std::string carry =
      "orbitals = \"{0}/casscf_orbitals.npy\"\nci = \"{0}/casscf_ci.npy\"\nwrite_npy = true\n";
  auto there = run_in(
      dir, displaced,
      std::string(kTwoRootTables) + std::vformat(carry, std::make_format_args("s0")), "there");
  ASSERT_TRUE(there.has_value()) << there.error().message;
  auto back = run_in(
      dir, entry, std::string(kTwoRootTables) + std::vformat(carry, std::make_format_args("there")),
      "back");
  ASSERT_TRUE(back.has_value()) << back.error().message;
  const fb::CasscfOutput &want = *first->get<fb::CasscfOutput>();
  const fb::CasscfOutput &moved = *there->get<fb::CasscfOutput>();
  const fb::CasscfOutput &got = *back->get<fb::CasscfOutput>();
  std::cout << std::format("[ casscf   ] displaced: {} macro iterations there, {} back; E {:.10f} "
                           "there, {:.10f} at the start\n",
                           moved.iterations.size(), got.iterations.size(), moved.energy,
                           want.energy);
  ASSERT_EQ(got.energies.size(), want.energies.size());
  for (std::size_t r = 0; r < got.energies.size(); ++r) {
    EXPECT_NEAR(got.energies[r], want.energies[r], kEnergyTolerance) << "root " << r;
  }
  // The displaced solution is not the first geometry's: the files did move.
  EXPECT_GT(std::abs(moved.energy - want.energy), 1e-4);
}

// The selection is shared, so [casci] takes the same file: CASCI over a
// converged CASSCF's own orbitals is that CASSCF's energy. Integral-direct,
// as the Casci step is, one root.
TEST(PipelineCasscfStart, ACasciOverACasscfsOrbitalsIsItsEnergy) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  const fb::json &entry = reference_case("h2o_ccpvdz");
  const fb::test::ScratchDir dir("pipeline_casscf_start");
  auto first =
      run_in(dir, entry,
             "[casscf]\nncas = 4\nnelecas = 4\nactive = [2, 3, 5, 6]\nconv_tol_grad = 1e-9\n"
             "write_npy = true\n",
             "s0");
  ASSERT_TRUE(first.has_value()) << first.error().message;
  auto casci =
      run_in(dir, entry, "[casci]\nncas = 4\nnelecas = 4\norbitals = \"s0/casscf_orbitals.npy\"\n",
             "casci");
  ASSERT_TRUE(casci.has_value()) << casci.error().message;
  ASSERT_EQ(casci->get<fb::CasciOutput>()->energies.size(), 1U);
  EXPECT_NEAR(casci->get<fb::CasciOutput>()->energies[0], first->get<fb::CasscfOutput>()->energy,
              kEnergyTolerance);
}

// The files the start refuses, once the SCF has run: orbitals of another
// shape, and a CI of another root count or length, each InvalidConfig naming
// the shapes; a missing file is read_npy's IO error.
TEST(PipelineCasscfStart, AMismatchedFileIsRefused) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz", "casscf_df.json");
  const fb::test::ScratchDir dir("pipeline_casscf_start");
  auto first = run_in(
      dir, entry, std::string(kTwoRootTables) + "active = [2, 3, 5, 6]\nwrite_npy = true\n", "s0");
  ASSERT_TRUE(first.has_value()) << first.error().message;
  const fb::CasscfOutput &cas = *first->get<fb::CasscfOutput>();
  const std::size_t nao = cas.c.n();
  const std::size_t dim = cas.vectors.size() / 2;
  std::filesystem::create_directories(dir / "bad");
  const auto write = [&](std::string_view name, std::vector<std::size_t> shape) {
    fb::NpyArray array{.shape = std::move(shape), .data = {}};
    array.data.assign(*fb::element_count(array.shape), 0.0);
    EXPECT_TRUE(fb::write_npy(dir / "bad" / name, array).has_value());
  };
  write("narrow.npy", {nao, nao - 1});
  write("one_root.npy", {dim, 1});
  write("short.npy", {dim - 1, 2});

  const auto expect_refused = [&](std::string_view keys, fb::ErrorKind kind,
                                  std::string_view needle) {
    auto calc = run_in(dir, entry, std::string(kTwoRootTables) + std::string(keys), "out");
    ASSERT_FALSE(calc.has_value()) << keys;
    EXPECT_EQ(calc.error().kind, kind) << calc.error().message;
    EXPECT_NE(calc.error().message.find(needle), std::string::npos) << calc.error().message;
  };
  expect_refused("orbitals = \"bad/narrow.npy\"\n", fb::ErrorKind::InvalidConfig,
                 std::format("have shape [{}, {}], not the basis's {} AOs by {} orbitals", nao,
                             nao - 1, nao, nao));
  expect_refused("orbitals = \"s0/casscf_orbitals.npy\"\nci = \"bad/one_root.npy\"\n",
                 fb::ErrorKind::InvalidConfig,
                 std::format("has shape [{}, 1], not (dim, nroots) with nroots 2", dim));
  expect_refused("orbitals = \"s0/casscf_orbitals.npy\"\nci = \"bad/short.npy\"\n",
                 fb::ErrorKind::InvalidConfig, "ci_guess must be empty or nroots dim");
  expect_refused("orbitals = \"bad/absent.npy\"\n", fb::ErrorKind::IO, "absent.npy");
}

} // namespace
