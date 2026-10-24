// The Casci step against PySCF's mcscf.CASCI (data/casci/casci.json,
// tools/emit_casci_reference.py), through the real registry: a config naming
// [casci] plans the whole chain to the ScfOutput, picks the active space (by index,
// or by AVAS against PySCF's mcscf.avas), builds its Hamiltonian and solves it
// over the GPU sigma. Compared: each singlet's total
// energy and <S^2>, and, for a root of no degenerate level, its natural
// occupations, the 1-RDM's eigenvalues, which unlike the 1-RDM do not depend
// on the active orbitals' signs. Every 1-RDM is symmetric with trace nelecas,
// and the orbitals are the ScfOutput's, reordered. Runs the device chain, so needs a
// GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

static_assert(fb::PipelineStep<fb::CasciStep>);
static_assert(fb::kRunShape<fb::CasciStep> == fb::RunShape::Device);
static_assert(std::same_as<
              fb::StepInput<fb::CasciStep>,
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                       fb::PrimitivePairClassesOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::CasciStep>, fb::CasciOutput>);

// The issue's bar, for the energies and <S^2>. Measured: within 1e-9.
constexpr double kEnergyTolerance = 1e-8;
// The natural occupations move to first order in the CI vector and in the two
// SCFs' orbitals. Measured: within 2e-9.
constexpr double kOccupationTolerance = 1e-7;
// The orbital energies only pair our orbitals with PySCF's by index.
constexpr double kOrbitalEnergyTolerance = 1e-6;

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "casci" / "casci.json");
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
  ADD_FAILURE() << "no case " << name << " in casci/casci.json";
  static const fb::json kNone;
  return kNone;
}

std::vector<double> download(const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpy(host.data(), data, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost),
            wwr::wwrSuccess);
  return host;
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

// The [casci] table of the reference's active space, nroots singlets. The Scf step
// it fills in runs at its defaults: they leave the energies within 1e-9 of
// PySCF's here, though a CASCI energy moves to first order in the orbitals.
std::string casci_tables(const fb::json &entry, std::size_t nroots) {
  if (!entry.at("avas").is_null()) {
    // AVAS picks ncas and nelecas; the test compares them with PySCF's.
    std::string tables = std::format("[casci]\nnroots = {}\navas_threshold = {:.17g}\navas = [",
                                     nroots, entry.at("avas").at("threshold").get<double>());
    const auto labels = entry.at("avas").at("labels").get<std::vector<std::string>>();
    for (std::size_t i = 0; i < labels.size(); ++i) {
      tables += std::format("{}\"{}\"", i == 0 ? "" : ", ", labels[i]);
    }
    return tables + "]\n";
  }
  std::string tables = std::format("[casci]\nncas = {}\nnelecas = {}\nnroots = {}\n",
                                   entry.at("ncas").get<std::int32_t>(),
                                   entry.at("nelecas").get<std::int32_t>(), nroots);
  if (!entry.at("active").is_null()) {
    const auto active = entry.at("active").get<std::vector<std::int32_t>>();
    tables += "active = [";
    for (std::size_t i = 0; i < active.size(); ++i) {
      tables += std::format("{}{}", i == 0 ? "" : ", ", active[i]);
    }
    tables += "]\n";
  }
  return tables;
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

struct Case {
  std::string_view name;
};

// ctest names a discovered case by this, so `ctest -R n2_ccpvdz` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name;
}

class PipelineCasci : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineCasci, MatchesPyscf) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name);
  const fb::json &roots = entry.at("roots");
  const fb::test::ScratchDir dir("pipeline_casci");
  auto calc = fb::read_config(write_config(dir, entry, casci_tables(entry, roots.size())));
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casci = calc->get<fb::CasciOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  ASSERT_TRUE(casci.has_value() && scf.has_value() && basis.has_value());

  // The frontier selection, and PySCF's, pick orbitals by index: the two
  // agree only if their orbital order does.
  const auto mo_energies = entry.at("mo_energies").get<std::vector<double>>();
  ASSERT_EQ(scf->mo_energies.size(), mo_energies.size());
  for (std::size_t i = 0; i < mo_energies.size(); ++i) {
    EXPECT_NEAR(scf->mo_energies[i], mo_energies[i], kOrbitalEnergyTolerance) << "orbital " << i;
  }

  EXPECT_EQ(casci->ncore, entry.at("ncore").get<std::int32_t>());
  EXPECT_EQ(casci->ncas, entry.at("ncas").get<std::int32_t>());
  EXPECT_EQ(casci->nelecas, entry.at("nelecas").get<std::int32_t>());
  const auto n = static_cast<std::size_t>(casci->ncas);
  ASSERT_EQ(casci->energies.size(), roots.size());
  ASSERT_EQ(casci->spin_squares.size(), roots.size());
  ASSERT_EQ(casci->rdm1.size(), roots.size() * n * n);
  ASSERT_FALSE(casci->vectors.empty());
  ASSERT_EQ(casci->vectors.size() % roots.size(), 0U);
  const std::size_t dim = casci->vectors.size() / roots.size();
  for (std::size_t r = 0; r < roots.size(); ++r) {
    const fb::json &root = roots.at(r);
    EXPECT_NEAR(casci->energies[r], root.at("energy").get<double>(), kEnergyTolerance)
        << "root " << r;
    EXPECT_NEAR(casci->spin_squares[r], root.at("s2").get<double>(), kEnergyTolerance)
        << "root " << r;

    double norm = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      norm += casci->vectors[(r * dim) + i] * casci->vectors[(r * dim) + i];
    }
    EXPECT_NEAR(norm, 1.0, 1e-12) << "root " << r;
    const std::vector<double> gamma(casci->rdm1.begin() + static_cast<std::ptrdiff_t>(r * n * n),
                                    casci->rdm1.begin() +
                                        static_cast<std::ptrdiff_t>((r + 1) * n * n));
    double trace = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
      trace += gamma[(t * n) + t];
      for (std::size_t u = 0; u < n; ++u) {
        EXPECT_NEAR(gamma[(t * n) + u], gamma[(u * n) + t], 1e-12) << "root " << r;
      }
    }
    EXPECT_NEAR(trace, casci->nelecas, 1e-10) << "root " << r;
    if (root.at("degenerate").get<bool>()) {
      continue; // its 1-RDM depends on the rotation within the level
    }
    const auto expected = root.at("natural_occupations").get<std::vector<double>>();
    const std::vector<double> ours = eigenvalues(gamma, n);
    ASSERT_EQ(ours.size(), expected.size());
    for (std::size_t k = 0; k < n; ++k) {
      EXPECT_NEAR(ours[k], expected[k], kOccupationTolerance) << "root " << r << ", " << k;
    }
  }

  if (!entry.at("avas").is_null()) {
    return; // AVAS rotates the ScfOutput's orbitals rather than ordering them
  }
  // The orbitals are the ScfOutput's, in orbital_order's order.
  const auto order = calc->step<fb::CasciStep>()->orbital_order(
      calc->get<fb::MoleculeOutput>()->molecule.electron_count(),
      static_cast<std::int32_t>(mo_energies.size()));
  ASSERT_TRUE(order.has_value()) << order.error().message;
  const std::size_t nao = basis->basis.nao_sphe();
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const std::vector<double> c = download(casci->c.data(), nao * nao);
  const std::vector<double> scf_c = download(scf->c.data(), nao * nao);
  for (std::size_t col = 0; col < nao; ++col) {
    for (std::size_t mu = 0; mu < nao; ++mu) {
      ASSERT_EQ(c[(col * nao) + mu], scf_c[(static_cast<std::size_t>((*order)[col]) * nao) + mu])
          << "column " << col;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineCasci,
                         ::testing::Values(Case{"h2o_sto3g"}, Case{"h2o_ccpvdz"}, Case{"n2_ccpvdz"},
                                           Case{"h2o_ccpvdz_explicit"}, Case{"h2o_ccpvdz_avas"},
                                           Case{"n2_ccpvdz_avas"}, Case{"n2_ccpvdz_avas_2p"}),
                         [](const auto &info) { return std::string(info.param.name); });

// An active space past the orbitals is refused when the step runs, and the
// error is the step's product.
TEST(PipelineCasci, AnActiveSpacePastTheOrbitalsIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("pipeline_casci");
  // Water's STO-3G has 7 orbitals; 3 core and 5 active do not fit.
  auto calc = fb::read_config(write_config(dir, entry, "[casci]\nncas = 5\nnelecas = 4\n"));
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(ran.error().message.find("exceeds the 7 orbitals"), std::string::npos)
      << ran.error().message;
  ASSERT_TRUE(calc->get<fb::ScfOutput>().has_value());
  EXPECT_FALSE(calc->get<fb::CasciOutput>().has_value());
}

// molden set: the lowest root's natural orbitals land in a Molden file, read
// back as io_molden_host reads one. Every orbital is written, the core at 2,
// the natural ones at PySCF's natural occupations, descending, the rest at 0.
TEST(PipelineCasciMolden, WritesTheNaturalOrbitals) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz_avas");
  const fb::test::ScratchDir dir("pipeline_casci_molden");
  const std::filesystem::path path = dir / "h2o_avas.molden";
  auto calc = fb::read_config(write_config(
      dir, entry,
      casci_tables(entry, 1) + std::format("molden = \"{}\"\n", path.generic_string())));
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &casci = calc->get<fb::CasciOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  ASSERT_TRUE(casci.has_value() && basis.has_value());

  const auto text = fb::read_file(path);
  ASSERT_TRUE(text.has_value()) << text.error().message;
  ASSERT_TRUE(text->starts_with("[Molden Format]\n"));
  const std::size_t mo_at = text->find("[MO]");
  ASSERT_NE(mo_at, std::string::npos);
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
  const std::size_t nao = basis->basis.nao_sphe();
  ASSERT_EQ(occupations.size(), nao);
  EXPECT_EQ(coefficients, nao * nao);
  const auto ncore = static_cast<std::size_t>(casci->ncore);
  const auto n = static_cast<std::size_t>(casci->ncas);
  const auto expected =
      entry.at("roots").at(0).at("natural_occupations").get<std::vector<double>>();
  ASSERT_EQ(expected.size(), n);
  for (std::size_t j = 0; j < nao; ++j) {
    // Molden's Occup= has six decimals.
    const double want = j < ncore ? 2.0 : (j < ncore + n ? expected[j - ncore] : 0.0);
    EXPECT_NEAR(occupations[j], want, 1e-6) << "orbital " << j;
  }
}

// An AVAS label naming no MINAO AO of the molecule is refused when the step
// runs: MINAO's oxygen has no 3d.
TEST(PipelineCasci, AnAvasLabelNamingNoMinaoAoIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz_avas");
  const fb::test::ScratchDir dir("pipeline_casci");
  auto calc = fb::read_config(write_config(dir, entry, "[casci]\navas = [\"O 2p\", \"O 3d\"]\n"));
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(ran.error().message.find("'O 3d' names no MINAO AO"), std::string::npos)
      << ran.error().message;
  EXPECT_FALSE(calc->get<fb::CasciOutput>().has_value());
}

} // namespace
