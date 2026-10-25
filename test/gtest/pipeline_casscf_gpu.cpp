// The Casscf step against PySCF's converged, polished mcscf.CASSCF
// (data/casscf/casscf.json, tools/emit_casscf_reference.py), through the real
// registry: a config naming [casscf] plans the whole chain to the ScfOutput,
// picks the start (by index, or by AVAS), optimizes it and canonicalizes the
// core and the virtuals. Compared: each root's energy and the weighted one,
// the state-averaged natural occupations, and the canonical core and virtual
// orbital energies, the state-averaged Fock's eigenvalues in each space. Each
// case also writes its natural orbitals as Molden, read back as
// io_molden_host reads one. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

static_assert(fb::PipelineStep<fb::CasscfStep>);
static_assert(fb::kRunShape<fb::CasscfStep> == fb::RunShape::Device);
static_assert(std::same_as<
              fb::StepInput<fb::CasscfStep>,
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                       fb::PrimitivePairClassesOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::CasscfStep>, fb::CasscfOutput>);

// The issue's bars.
constexpr double kEnergyTolerance = 1e-8;
constexpr double kOccupationTolerance = 1e-6;
constexpr double kOrbitalEnergyTolerance = 1e-6;
// Each root's energy, the natural occupations and the orbital energies are
// first order in the orbitals' error, ~||g|| / (H's least eigenvalue): at 1e-9,
// as mcscf_solve_gpu.cpp runs, ~1e-8 (see src/mcscf/solve/README.md).
constexpr double kConvTolGrad = 1e-9;

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

// The reference's molecule written as an XYZ file in dir, and a config naming
// it with the given tables after it.
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
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n{}",
      entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>(), tables);
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
};

// ctest names a discovered case by this, so `ctest -R n2_ccpvdz` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

class PipelineCasscf : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineCasscf, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name);
  const fb::json &want = entry.at("casscf");
  const fb::test::ScratchDir dir("pipeline_casscf");
  const std::filesystem::path molden = dir / "out/casscf.molden";
  auto calc =
      fb::read_config(write_config(dir, entry, casscf_table(entry)), dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casscf = calc->get<fb::CasscfOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  ASSERT_TRUE(casscf.has_value() && basis.has_value());
  std::cout << std::format("[ casscf   ] {}: {} macro iterations, |g| {:.3e}\n", GetParam().name,
                           casscf->iterations.size(), casscf->gradient_norm);

  EXPECT_EQ(casscf->ncore, entry.at("ncore").get<std::int32_t>());
  EXPECT_EQ(casscf->ncas, entry.at("ncas").get<std::int32_t>());
  EXPECT_EQ(casscf->nelecas, entry.at("nelecas").get<std::int32_t>());
  EXPECT_LT(casscf->gradient_norm, kConvTolGrad);

  // The energies: each root's and the weighted one.
  const auto weights = entry.at("weights").get<std::vector<double>>();
  const auto energies = want.at("energies").get<std::vector<double>>();
  ASSERT_EQ(casscf->energies.size(), weights.size());
  ASSERT_EQ(casscf->spin_squares.size(), weights.size());
  EXPECT_EQ(casscf->weights, weights);
  for (std::size_t r = 0; r < weights.size(); ++r) {
    EXPECT_NEAR(casscf->energies[r], energies[r], kEnergyTolerance) << "root " << r;
    EXPECT_NEAR(casscf->spin_squares[r], 0.0, 1e-8) << "root " << r;
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

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineCasscf,
                         ::testing::Values(Case{"h2o_ccpvdz"}, Case{"n2_ccpvdz"},
                                           Case{"h2co_ccpvdz_sa2"},
                                           Case{"h2co_ccpvdz_sa2_weighted"},
                                           Case{"h2o_ccpvdz_avas"}),
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

} // namespace
