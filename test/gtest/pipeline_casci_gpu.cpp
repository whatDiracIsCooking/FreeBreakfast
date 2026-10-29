// The Casci step against PySCF's mcscf.CASCI (data/casci/casci.json,
// tools/emit/casci_reference.py), through the real registry: a config naming
// [casci] plans the whole chain to the ScfOutput, picks the active space (by index,
// or by AVAS against PySCF's mcscf.avas), builds its Hamiltonian and solves it
// over the GPU sigma. Compared: each root's total
// energy and <S^2>, and, for a root of no degenerate level, its natural
// occupations, the 1-RDM's eigenvalues, which unlike the 1-RDM do not depend
// on the active orbitals' signs. Every 1-RDM is symmetric with trace nelecas,
// and the orbitals are the ScfOutput's, reordered; an AVAS case that records
// "active_overlaps" has its active orbitals, up to sign, compared. A case at a multiplicity
// other than 1 is a CASCI of that spin over the RHF orbitals: the chain runs as
// the singlet and the step again by hand with the molecule at the
// multiplicity. An "rohf" case is over the Scf step's own open shell instead:
// the chain runs at its multiplicity with [eri] density-fitted, the DF-ROHF the
// reference's orbitals are. Then the open shell's refusals, over a hand-built
// ScfOutput. Runs the device chain, so needs a GPU.

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
// AVAS's active orbitals over the ScfOutput's, |R|, move to first order in
// the two SCFs' orbitals. Measured: within 4e-9.
constexpr double kOverlapTolerance = 1e-7;

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

std::vector<double> download(fb::DeviceResources &res, const double *data, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), data, sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The reference's fitting set for an "rohf" case's DF-ROHF, compiled in, by
// the name casci.json records.
constexpr std::string_view kDfAuxBasis = "cc-pVDZ-JKFIT";

// Whether the reference's orbitals are its DF-ROHF's.
bool over_rohf(const fb::json &entry) {
  return entry.value("scf", std::string{"rhf"}) == "rohf";
}

// The reference's molecule written as an XYZ file in dir, and a config naming
// it with the given tables after it: at its multiplicity for an "rohf" case,
// whose [eri] is then density-fitted.
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
  std::string toml = std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n",
                                 entry.at("charge").get<std::int32_t>());
  if (over_rohf(entry)) {
    EXPECT_EQ(entry.at("aux_basis").at("name").get<std::string>(), kDfAuxBasis);
    toml += std::format("multiplicity = {}\n[eri]\nmode = \"density_fitting\"\naux_basis = "
                        "\"{}\"\n",
                        entry.at("multiplicity").get<std::int32_t>(), kDfAuxBasis);
  }
  toml +=
      std::format("[basis_set]\nbasis = \"{}\"\n{}", entry.at("basis").get<std::string>(), tables);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  return dir / "run.toml";
}

// The [casci] table of the reference's active space, nroots roots. The Scf step
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
  auto calc = fb::read_config(write_config(dir, entry, casci_tables(entry, roots.size())),
                              dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;
  const auto &chain = calc->get<fb::CasciOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  ASSERT_TRUE(chain.has_value() && scf.has_value() && basis.has_value());

  // The step again by hand over the same products, but the molecule's
  // multiplicity, when the reference's is not 1 over RHF orbitals.
  std::optional<fb::CasciOutput> by_hand;
  if (const auto multiplicity = entry.value("multiplicity", std::int32_t{1});
      multiplicity != 1 && !over_rohf(entry)) {
    fb::DeviceResources &res = node.device(0);
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    const fb::Molecule &singlet = calc->get<fb::MoleculeOutput>()->molecule;
    const fb::MoleculeOutput molecule{
        fb::Molecule(res, singlet.atoms(), singlet.charge(), multiplicity)};
    auto ran_by_hand = calc->step<fb::CasciStep>()->run(
        fb::StepInput<fb::CasciStep>{molecule, *basis, *calc->get<fb::SphericalTransformOutput>(),
                                     *calc->get<fb::PrimitivePairClassesOutput>(),
                                     *calc->get<fb::CoreHamiltonianOutput>(), *scf},
        res);
    ASSERT_TRUE(ran_by_hand.has_value()) << ran_by_hand.error().message;
    by_hand.emplace(std::move(*ran_by_hand));
  }
  const fb::CasciOutput &casci = by_hand ? *by_hand : *chain;

  // The frontier selection, and PySCF's, pick orbitals by index: the two
  // agree only if their orbital order does.
  const auto mo_energies = entry.at("mo_energies").get<std::vector<double>>();
  ASSERT_EQ(scf->mo_energies.size(), mo_energies.size());
  for (std::size_t i = 0; i < mo_energies.size(); ++i) {
    EXPECT_NEAR(scf->mo_energies[i], mo_energies[i], kOrbitalEnergyTolerance) << "orbital " << i;
  }

  EXPECT_EQ(casci.ncore, entry.at("ncore").get<std::int32_t>());
  EXPECT_EQ(casci.ncas, entry.at("ncas").get<std::int32_t>());
  EXPECT_EQ(casci.nelecas, entry.at("nelecas").get<std::int32_t>());
  const auto n = static_cast<std::size_t>(casci.ncas);
  ASSERT_EQ(casci.energies.size(), roots.size());
  ASSERT_EQ(casci.spin_squares.size(), roots.size());
  ASSERT_EQ(casci.rdm1.size(), roots.size() * n * n);
  ASSERT_FALSE(casci.vectors.empty());
  ASSERT_EQ(casci.vectors.size() % roots.size(), 0U);
  const std::size_t dim = casci.vectors.size() / roots.size();
  for (std::size_t r = 0; r < roots.size(); ++r) {
    const fb::json &root = roots.at(r);
    EXPECT_NEAR(casci.energies[r], root.at("energy").get<double>(), kEnergyTolerance)
        << "root " << r;
    EXPECT_NEAR(casci.spin_squares[r], root.at("s2").get<double>(), kEnergyTolerance)
        << "root " << r;

    double norm = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      norm += casci.vectors[(r * dim) + i] * casci.vectors[(r * dim) + i];
    }
    EXPECT_NEAR(norm, 1.0, 1e-12) << "root " << r;
    const std::vector<double> gamma(casci.rdm1.begin() + static_cast<std::ptrdiff_t>(r * n * n),
                                    casci.rdm1.begin() +
                                        static_cast<std::ptrdiff_t>((r + 1) * n * n));
    double trace = 0.0;
    for (std::size_t t = 0; t < n; ++t) {
      trace += gamma[(t * n) + t];
      for (std::size_t u = 0; u < n; ++u) {
        EXPECT_NEAR(gamma[(t * n) + u], gamma[(u * n) + t], 1e-12) << "root " << r;
      }
    }
    EXPECT_NEAR(trace, casci.nelecas, 1e-10) << "root " << r;
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

  if (entry.contains("active_overlaps")) {
    // AVAS's active orbitals up to sign: |R|'s active columns, R the
    // selection's rotation of the ScfOutput's orbitals, against PySCF's.
    fb::DeviceResources &res = node.device(0);
    const fb::ScopedDeviceIndex on{res.dev_idx()};
    const auto space = calc->step<fb::CasciStep>()->select(
        res, calc->get<fb::MoleculeOutput>()->molecule, basis->basis, *scf);
    ASSERT_TRUE(space.has_value()) << space.error().message;
    const auto overlaps = entry.at("active_overlaps").get<std::vector<std::vector<double>>>();
    ASSERT_EQ(overlaps.size(), n);
    const std::size_t nmo = mo_energies.size();
    const auto ncore = static_cast<std::size_t>(space->ncore);
    for (std::size_t j = 0; j < n; ++j) {
      ASSERT_EQ(overlaps[j].size(), nmo);
      for (std::size_t k = 0; k < nmo; ++k) {
        EXPECT_NEAR(std::abs(space->r[k + (nmo * (ncore + j))]), overlaps[j][k], kOverlapTolerance)
            << "active orbital " << j << ", orbital " << k;
      }
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
  fb::DeviceResources res(scf->c.dev_idx());
  const std::vector<double> c = download(res, casci.c.data(), nao * nao);
  const std::vector<double> scf_c = download(res, scf->c.data(), nao * nao);
  for (std::size_t col = 0; col < nao; ++col) {
    for (std::size_t mu = 0; mu < nao; ++mu) {
      ASSERT_EQ(c[(col * nao) + mu], scf_c[(static_cast<std::size_t>((*order)[col]) * nao) + mu])
          << "column " << col;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineCasci,
                         ::testing::Values(Case{"h2o_sto3g"}, Case{"h2o_ccpvdz"}, Case{"n2_ccpvdz"},
                                           Case{"h2o_ccpvdz_explicit"}, Case{"h2o_ccpvdz_triplet"},
                                           Case{"h2o_ccpvdz_avas"}, Case{"n2_ccpvdz_avas"},
                                           Case{"n2_ccpvdz_avas_2p"}, Case{"nh2_ccpvdz_rohf"},
                                           Case{"ch2_ccpvdz_rohf"}, Case{"nh2_ccpvdz_rohf_avas"},
                                           Case{"ch2_ccpvdz_rohf_avas"}),
                         [](const auto &info) { return std::string(info.param.name); });

// An active space past the orbitals is refused when the step runs, and the
// error is the step's product.
TEST(PipelineCasci, AnActiveSpacePastTheOrbitalsIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("pipeline_casci");
  // Water's STO-3G has 7 orbitals; 3 core and 5 active do not fit.
  auto calc = fb::read_config(write_config(dir, entry, "[casci]\nncas = 5\nnelecas = 4\n"),
                              dir.path(), dir / "out");
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

// The lowest root's natural orbitals land in the output directory's
// casci.molden, read back as io_molden_host reads one. Every orbital is
// written, the core at 2, the natural ones at PySCF's natural occupations,
// descending, the rest at 0.
TEST(PipelineCasciMolden, WritesTheNaturalOrbitals) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_ccpvdz_avas");
  const fb::test::ScratchDir dir("pipeline_casci_molden");
  const std::filesystem::path path = dir / "out/casci.molden";
  auto calc =
      fb::read_config(write_config(dir, entry, casci_tables(entry, 1)), dir.path(), dir / "out");
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
  auto calc = fb::read_config(write_config(dir, entry, "[casci]\navas = [\"O 2p\", \"O 3d\"]\n"),
                              dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(ran.error().message.find("'O 3d' names no MINAO AO"), std::string::npos)
      << ran.error().message;
  EXPECT_FALSE(calc->get<fb::CasciOutput>().has_value());
}

// An open-shell ScfOutput, n_alpha != n_beta, by index, with its core doubly
// occupied and every singly occupied orbital active: each broken is refused by
// the selection, naming the orbital. The rest of the chain is water's in STO-3G, 7 orbitals, under
// a hand-built ScfOutput of 6 alpha and 4 beta electrons, orbitals 4 and 5
// singly occupied; its orbitals are never read.
TEST(PipelineCasci, AnOpenShellSelectionKeepsItsOccupations) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("pipeline_casci");
  auto calc = fb::read_config(write_config(dir, entry, "[casci]\nncas = 4\nnelecas = 4\n"),
                              dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::Molecule &molecule = calc->get<fb::MoleculeOutput>()->molecule;
  const fb::BasisSet &basis = calc->get<fb::BasisSetOutput>()->basis;
  const fb::ScfOutput open{.energy = 0.0,
                           .iterations = 0,
                           .n_alpha = 6,
                           .n_beta = 4,
                           .mo_energies = std::vector<double>(7, 0.0),
                           .c = fb::SphericalMatrixAO(res, basis),
                           .d = fb::SphericalMatrixAO(res, basis)};
  const auto refusal = [&](const fb::ActiveSpaceSelection &selection) {
    const auto selected = selection.select(res, molecule, basis, open);
    EXPECT_FALSE(selected.has_value());
    return selected.has_value() ? fb::Error{} : selected.error();
  };

  // The frontier CAS(4,4) is 3..6 over a core of 0..2: both kept.
  const auto frontier =
      fb::ActiveSpaceSelection{.ncas = 4, .nelecas = 4}.select(res, molecule, basis, open);
  ASSERT_TRUE(frontier.has_value()) << frontier.error().message;
  EXPECT_EQ(frontier->ncore, 3);
  EXPECT_EQ(frontier->nelecas, 4);

  // Active 0, 1, 4 and 5 push the core to 2, 3 and 6: 6 is empty.
  const fb::Error core =
      refusal(fb::ActiveSpaceSelection{.ncas = 4, .nelecas = 4, .active = {0, 1, 4, 5}});
  EXPECT_EQ(core.kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(core.message.find("core orbital 6 is not doubly occupied"), std::string::npos)
      << core.message;

  // CAS(4,2) over 3 and 4 leaves 5, singly occupied, outside.
  const fb::Error single =
      refusal(fb::ActiveSpaceSelection{.ncas = 2, .nelecas = 4, .active = {3, 4}});
  EXPECT_EQ(single.kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(single.message.find("singly occupied orbital 5"), std::string::npos) << single.message;
}

// A multiplicity that does not fit the active space is refused when the step
// runs, naming it: 2S below 0, of the other parity from nelecas, or above it.
// Water's CAS(4,4) in STO-3G, the step run by hand with the molecule at each.
TEST(PipelineCasci, AMultiplicityThatDoesNotFitIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case("h2o_sto3g");
  const fb::test::ScratchDir dir("pipeline_casci");
  auto calc = fb::read_config(write_config(dir, entry, "[casci]\nncas = 4\nnelecas = 4\n"),
                              dir.path(), dir / "out");
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_TRUE(ran.has_value()) << ran.error().message;

  fb::DeviceResources &res = node.device(0);
  const fb::ScopedDeviceIndex on{res.dev_idx()};
  const fb::Molecule &singlet = calc->get<fb::MoleculeOutput>()->molecule;
  for (const std::int32_t multiplicity : {0, 2, 7}) {
    const fb::MoleculeOutput molecule{
        fb::Molecule(res, singlet.atoms(), singlet.charge(), multiplicity)};
    const auto refused = calc->step<fb::CasciStep>()->run(
        fb::StepInput<fb::CasciStep>{
            molecule, *calc->get<fb::BasisSetOutput>(), *calc->get<fb::SphericalTransformOutput>(),
            *calc->get<fb::PrimitivePairClassesOutput>(), *calc->get<fb::CoreHamiltonianOutput>(),
            *calc->get<fb::ScfOutput>()},
        res);
    ASSERT_FALSE(refused.has_value()) << "multiplicity " << multiplicity;
    EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(refused.error().message.find(
                  std::format("casci: multiplicity {} does not fit 4", multiplicity)),
              std::string::npos)
        << refused.error().message;
  }
}

} // namespace
