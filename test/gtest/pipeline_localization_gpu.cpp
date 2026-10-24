// The Localization step against PySCF's Pipek-Mezey and intrinsic bond
// orbitals (data/localization/{pipek_mezey,ibo}.json,
// tools/emit_localization_reference.py), through the real registry: a config
// naming [localization] plans the whole chain to the Scf and localizes its
// occupied orbitals. The two codes normalise and order their AOs differently,
// so the orbitals are compared by what is invariant to that: the functional,
// and each orbital's Mulliken population on each atom, matched to PySCF's
// orbitals up to order. Localization is a rotation of the occupied space, so
// the localized orbitals are also S-orthonormal and give back the SCF's own
// density, D = 2 C_local C_local^T. The IAOs are checked on their own by their
// partial charges. Runs the device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

static_assert(fb::PipelineStep<fb::LocalizationStep>);
static_assert(fb::kRunShape<fb::LocalizationStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::LocalizationStep>,
                           fb::View<fb::Molecule, fb::BasisSet, fb::Overlap, fb::Scf>>);
static_assert(std::same_as<fb::StepOutput<fb::LocalizationStep>, fb::Localization>);

// Both sides converge L to ~1e-12 at a maximum, where L is quadratic in the
// orbitals, so the populations agree only to about its square root. What is
// left beyond that is the two SCFs' own tolerances.
constexpr double kFunctionalTolerance = 1e-8;
constexpr double kPopulationTolerance = 1e-5;
// The IAO charges are linear in the SCF's density, so they carry its
// tolerance and nothing of the localization's.
constexpr double kChargeTolerance = 1e-7;
// An IBO with more than this population on one atom is a one-center orbital
// (a core or a lone pair); a bond's largest is ~0.6.
constexpr double kOneCenter = 0.9;

fb::json read_reference(std::string_view file) {
  auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "localization" / file);
  EXPECT_TRUE(read.has_value()) << read.error().message;
  return read.has_value() ? std::move(*read) : fb::json{};
}

// One case of localization/@p file: pipek_mezey.json or ibo.json.
const fb::json &reference_case(std::string_view name, std::string_view file = "pipek_mezey.json") {
  static const fb::json kPipekMezey = read_reference("pipek_mezey.json");
  static const fb::json kIbo = read_reference("ibo.json");
  for (const fb::json &entry : (file == "ibo.json" ? kIbo : kPipekMezey).at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in localization/" << file;
  static const fb::json kNone;
  return kNone;
}

// The entry's record for @p exponent, or null.
const fb::json *exponent_record(const fb::json &entry, std::int32_t exponent) {
  for (const fb::json &localized : entry.at("localized")) {
    if (localized.at("exponent").get<std::int32_t>() == exponent) {
      return &localized;
    }
  }
  return nullptr;
}

template <typename Buffer>
std::vector<double> download(const Buffer &buffer, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpy(host.data(), buffer.data(), sizeof(double) * count,
                           wwr::wwrMemcpyDeviceToHost),
            wwr::wwrSuccess);
  return host;
}

// The reference's molecule localized with the given exponent, from a config
// naming only the inputs and [localization], plus @p extra in that table. A
// @p basis document, if given, is read from a file in place of the entry's
// basis.
std::optional<fb::Calculation> localize(const fb::json &entry, std::int32_t exponent,
                                        std::string_view extra = "",
                                        const fb::json *basis = nullptr) {
  const fb::test::ScratchDir dir("pipeline_localization");
  std::string basis_spec = entry.at("basis").get<std::string>();
  if (basis != nullptr) {
    basis_spec = "basis.json";
    EXPECT_TRUE(fb::write_json(dir / basis_spec, *basis).has_value());
  }
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
  // Converged past the comparison's needs: the default 1e-10 on L leaves the
  // populations ~1e-5 from the maximum.
  const std::string toml =
      std::format("[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
                  "[localization]\nexponent = {}\nfunctional_tolerance = 1e-13\n{}",
                  entry.at("charge").get<std::int32_t>(), basis_spec, exponent, extra);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml");
  if (!calc.has_value()) {
    ADD_FAILURE() << calc.error().message;
    return std::nullopt;
  }
  fb::NodeResources node;
  if (const fb::Status ran = calc->run(node); !ran.has_value()) {
    ADD_FAILURE() << ran.error().message;
    return std::nullopt;
  }
  return std::move(*calc);
}

// The localized orbitals of @p calc against the reference record @p expected:
// the functional, every orbital's Mulliken population on every atom matched to
// the reference's up to order, S-orthonormality and the SCF's density.
//
// @p one_center_sets is for IBO. Two orbitals both wholly on one atom by the
// IAO populations (water's O 1s and lone pairs) can rotate into each other at
// almost no cost in L, only the other atoms' small tails telling them apart:
// L moves by 1e-13 where their Mulliken populations move by 1e-3, between
// PySCF's own two IBO optimizers as much as between it and this code. Such
// orbitals are fixed only as a set, so they are compared by the set's summed
// populations, which are.
void expect_matches(const fb::json &entry, const fb::json &expected, const fb::Calculation &calc,
                    bool one_center_sets = false) {
  const auto &local = calc.get<fb::Localization>();
  const auto &scf = calc.get<fb::Scf>();
  const auto &overlap = calc.get<fb::Overlap>();
  const auto &basis = calc.get<fb::BasisSet>();
  ASSERT_TRUE(local.has_value() && scf.has_value() && overlap.has_value() && basis.has_value());

  const std::size_t n = basis->nao_sphe();
  const auto occ = static_cast<std::size_t>(local->n_occ);
  ASSERT_EQ(occ * 2, entry.at("electrons").get<std::size_t>());
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const std::vector<double> c = download(local->c, n * occ);
  const std::vector<double> s = download(overlap->s, n * n);
  const std::vector<double> d = download(scf->d, n * n);

  EXPECT_NEAR(local->info.functional, expected.at("functional").get<double>(),
              kFunctionalTolerance);
  EXPECT_GT(local->info.functional, local->info.initial_functional);

  // Each orbital's population on each atom: sum over the atom's AOs of
  // C (S C). A placed basis keeps an atom's shells, and so its AOs, together.
  const std::uint32_t n_atoms = basis->natoms();
  ASSERT_EQ(n_atoms, entry.at("atoms").size());
  std::vector<std::size_t> first_ao(n_atoms + 1, n);
  for (std::uint32_t a = 0; a < n_atoms; ++a) {
    first_ao[a] = basis->ao_offset_sphe(basis->atom_shell_offset(a));
  }
  std::vector<std::vector<double>> ours(occ, std::vector<double>(n_atoms, 0.0));
  for (std::size_t i = 0; i < occ; ++i) {
    for (std::uint32_t a = 0; a < n_atoms; ++a) {
      for (std::size_t mu = first_ao[a]; mu < first_ao[a + 1]; ++mu) {
        double sc = 0.0;
        for (std::size_t nu = 0; nu < n; ++nu) {
          sc += s[mu + (n * nu)] * c[nu + (n * i)];
        }
        ours[i][a] += c[mu + (n * i)] * sc;
      }
    }
  }
  // Every PySCF orbital matches a distinct one of ours, population by
  // population: the orbitals are the same up to order (and sign, which the
  // populations do not see).
  const auto theirs = expected.at("populations").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(theirs.size(), occ);
  // With @p one_center_sets, orbitals on one atom are compared as a set: the
  // atom an orbital sits on (population above kOneCenter), or n_atoms for one
  // that spans several.
  const auto center = [&](const std::vector<double> &q) {
    for (std::uint32_t a = 0; a < n_atoms; ++a) {
      if (one_center_sets && q[a] > kOneCenter) {
        return a;
      }
    }
    return n_atoms;
  };
  std::vector<std::vector<double>> our_sets(n_atoms, std::vector<double>(n_atoms, 0.0));
  std::vector<std::vector<double>> their_sets = our_sets;
  std::vector<bool> taken(occ, false);
  for (std::size_t i = 0; i < occ; ++i) {
    if (const std::uint32_t at = center(ours[i]); at < n_atoms) {
      taken[i] = true;
      std::ranges::transform(our_sets[at], ours[i], our_sets[at].begin(), std::plus{});
    }
  }
  for (std::size_t k = 0; k < occ; ++k) {
    if (const std::uint32_t at = center(theirs[k]); at < n_atoms) {
      std::ranges::transform(their_sets[at], theirs[k], their_sets[at].begin(), std::plus{});
      continue;
    }
    std::size_t best = occ;
    double best_gap = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < occ; ++i) {
      double gap = 0.0;
      for (std::uint32_t a = 0; a < n_atoms; ++a) {
        gap = std::max(gap, std::abs(ours[i][a] - theirs[k][a]));
      }
      if (!taken[i] && gap < best_gap) {
        best = i;
        best_gap = gap;
      }
    }
    ASSERT_LT(best, occ);
    taken[best] = true;
    EXPECT_LT(best_gap, kPopulationTolerance) << "PySCF orbital " << k << " has no match";
  }
  // The trace of a set's population block is invariant under rotations within
  // it, so the sums are as determined as a lone orbital's populations.
  for (std::uint32_t at = 0; at < n_atoms; ++at) {
    for (std::uint32_t a = 0; a < n_atoms; ++a) {
      EXPECT_NEAR(our_sets[at][a], their_sets[at][a], kPopulationTolerance)
          << "orbitals on atom " << at << ", population on atom " << a;
    }
  }

  for (std::size_t i = 0; i < occ; ++i) {
    for (std::size_t j = 0; j < occ; ++j) {
      double ctsc = 0.0;
      for (std::size_t mu = 0; mu < n; ++mu) {
        for (std::size_t nu = 0; nu < n; ++nu) {
          ctsc += c[mu + (n * i)] * s[mu + (n * nu)] * c[nu + (n * j)];
        }
      }
      EXPECT_NEAR(ctsc, i == j ? 1.0 : 0.0, 1e-10) << i << "," << j;
    }
  }
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      double density = 0.0;
      for (std::size_t i = 0; i < occ; ++i) {
        density += 2.0 * c[mu + (n * i)] * c[nu + (n * i)];
      }
      EXPECT_NEAR(density, d[mu + (n * nu)], 1e-10) << mu << "," << nu;
    }
  }
}

struct Case {
  std::string_view name;
  std::int32_t exponent;
};

// ctest names a discovered case by this, so `ctest -R h2o_ccpvdz_p4` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name << "_p" << c.exponent;
}

const auto kCases =
    ::testing::Values(Case{"h2o_sto3g", 2}, Case{"h2o_sto3g", 4}, Case{"h2o_ccpvdz", 2},
                      Case{"h2o_ccpvdz", 4}, Case{"nh4_ccpvdz", 2}, Case{"nh4_ccpvdz", 4});

std::string case_name(const ::testing::TestParamInfo<Case> &info) {
  return std::format("{}_p{}", info.param.name, info.param.exponent);
}

class PipelineLocalization : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineLocalization, MatchesPyscfAndKeepsTheDensity) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name);
  const fb::json *expected = exponent_record(entry, GetParam().exponent);
  ASSERT_NE(expected, nullptr) << "no exponent " << GetParam().exponent;
  auto calc = localize(entry, GetParam().exponent);
  ASSERT_TRUE(calc.has_value());
  expect_matches(entry, *expected, *calc);
  const auto &local = calc->get<fb::Localization>();
  ASSERT_TRUE(local.has_value());
  EXPECT_TRUE(local->populations.empty() && local->charges.empty()) << "IBO only";
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineLocalization, kCases, case_name);

class PipelineIbo : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineIbo, MatchesPyscfAndKeepsTheDensity) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name, "ibo.json");
  const fb::json *expected = exponent_record(entry, GetParam().exponent);
  ASSERT_NE(expected, nullptr) << "no exponent " << GetParam().exponent;
  auto calc = localize(entry, GetParam().exponent, "mode = \"ibo\"\n");
  ASSERT_TRUE(calc.has_value());
  expect_matches(entry, *expected, *calc, true);

  // The step's IAO populations of the IBOs are the ones the sweep maximized:
  // each orbital's sum to one, and their p-th powers sum to its L. The
  // charges are the density's, so PySCF's whichever orbitals gave them.
  const auto &local = calc->get<fb::Localization>();
  ASSERT_TRUE(local.has_value());
  const std::size_t n_atoms = entry.at("atoms").size();
  const auto occ = static_cast<std::size_t>(local->n_occ);
  ASSERT_EQ(local->populations.size(), n_atoms * occ);
  double functional = 0.0;
  for (std::size_t i = 0; i < occ; ++i) {
    double weight = 0.0;
    for (std::size_t a = 0; a < n_atoms; ++a) {
      const double q = local->populations[a + (n_atoms * i)];
      weight += q;
      functional += std::pow(q, GetParam().exponent);
    }
    EXPECT_NEAR(weight, 1.0, 1e-10) << "orbital " << i;
  }
  EXPECT_NEAR(functional, local->info.functional, 1e-10);
  const auto charges = entry.at("iao_charges").get<std::vector<double>>();
  ASSERT_EQ(local->charges.size(), charges.size());
  for (std::size_t a = 0; a < n_atoms; ++a) {
    EXPECT_NEAR(local->charges[a], charges[a], kChargeTolerance) << "atom " << a;
  }

  // The bond reading of the IBOs. Water and NH4+ are a heavy atom (atom 0)
  // and hydrogens: every IBO is either on the heavy atom alone (core, lone
  // pair) or a bond to one hydrogen, one per hydrogen. Each center is the
  // step's own population, and its character sums to one; a hydrogen's is
  // all s, MINAO giving it nothing else. The bonds are symmetry-equivalent,
  // so the heavy atom's hybrid is the same in each.
  ASSERT_EQ(local->bonds.size(), occ);
  std::vector<bool> bonded(n_atoms, false);
  std::optional<std::array<double, 4>> hybrid;
  for (std::size_t i = 0; i < occ; ++i) {
    SCOPED_TRACE(std::format("IBO {}", i));
    const auto &centers = local->bonds[i].centers;
    ASSERT_FALSE(centers.empty());
    for (std::size_t k = 0; k < centers.size(); ++k) {
      const fb::BondCenter &center = centers[k];
      EXPECT_DOUBLE_EQ(center.population, local->populations[center.atom + (n_atoms * i)]);
      EXPECT_NEAR(std::accumulate(center.character.begin(), center.character.end(), 0.0), 1.0,
                  1e-12);
      if (k > 0) {
        EXPECT_GE(centers[k - 1].population, center.population);
      }
    }
    EXPECT_EQ(centers[0].atom, 0U) << "the heavy atom holds the most of every IBO";
    if (centers.size() == 1) {
      continue;
    }
    ASSERT_EQ(centers.size(), 2U);
    const std::uint32_t hydrogen = centers[1].atom;
    EXPECT_FALSE(bonded[hydrogen]) << "two bonds to atom " << hydrogen;
    bonded[hydrogen] = true;
    EXPECT_NEAR(centers[1].character[0], 1.0, 1e-12);
    if (!hybrid.has_value()) {
      hybrid = centers[0].character;
    }
    for (std::size_t l = 0; l < 4; ++l) {
      EXPECT_NEAR(centers[0].character[l], (*hybrid)[l], kPopulationTolerance) << "l = " << l;
    }
  }
  EXPECT_EQ(std::ranges::count(bonded, true), static_cast<std::ptrdiff_t>(n_atoms - 1));
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineIbo, kCases, case_name);

// The IAOs on their own, from the Scf's orbitals over the entry's basis (or
// @p basis_document): orthonormal, spanning the occupied orbitals (each has
// unit weight on them), and giving PySCF's partial charges,
// Z_A - 2 sum_i sum_{r in A} (A^T S C)_ri^2.
void expect_iaos(const fb::json &entry, const fb::json *basis_document = nullptr) {
  auto calc = localize(entry, 4, "mode = \"ibo\"\n", basis_document);
  ASSERT_TRUE(calc.has_value());
  const auto &molecule = calc->get<fb::Molecule>();
  const auto &basis = calc->get<fb::BasisSet>();
  const auto &overlap = calc->get<fb::Overlap>();
  const auto &scf = calc->get<fb::Scf>();
  ASSERT_TRUE(molecule.has_value() && basis.has_value() && overlap.has_value() && scf.has_value());
  auto stamp = fb::resolve_basis("minao");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  auto minao = fb::place(stamp->primitive_normalized(), *molecule);
  ASSERT_TRUE(minao.has_value()) << minao.error().message;

  const std::size_t n = basis->nao_sphe();
  const std::size_t m = minao->nao_sphe();
  const auto occ = static_cast<std::size_t>(molecule->electron_count() / 2);
  fb::DeviceResources res(scf->c.dev_idx());
  fb::Buffers::device<double> iao(n * m, res.device());
  const fb::Status built =
      fb::intrinsic_atomic_orbitals(res, *basis, *minao, static_cast<std::int32_t>(occ),
                                    overlap->s.data(), scf->c.data(), iao.data());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  ASSERT_TRUE(res.sync().has_value());
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const std::vector<double> a = download(iao, n * m);
  const std::vector<double> s = download(overlap->s, n * n);

  // A^T S A = I.
  std::vector<double> ats(m * n, 0.0);
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      for (std::size_t mu = 0; mu < n; ++mu) {
        ats[r + (m * nu)] += a[mu + (n * r)] * s[mu + (n * nu)];
      }
    }
  }
  for (std::size_t r = 0; r < m; ++r) {
    for (std::size_t q = 0; q < m; ++q) {
      double atsa = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu) {
        atsa += ats[r + (m * nu)] * a[nu + (n * q)];
      }
      EXPECT_NEAR(atsa, r == q ? 1.0 : 0.0, 1e-10) << r << "," << q;
    }
  }
  // The canonical orbitals' populations over the IAOs: each orbital's sum
  // to one, as the IAOs span them, and the charges are PySCF's.
  const auto populations =
      fb::orbital_populations(res, *basis, *minao, static_cast<std::int32_t>(occ),
                              overlap->s.data(), scf->c.data(), iao.data());
  ASSERT_TRUE(populations.has_value()) << populations.error().message;
  const std::uint32_t n_atoms = minao->natoms();
  ASSERT_EQ(populations->size(), n_atoms * occ);
  for (std::size_t i = 0; i < occ; ++i) {
    double weight = 0.0;
    for (std::uint32_t atom = 0; atom < n_atoms; ++atom) {
      weight += (*populations)[atom + (n_atoms * i)];
    }
    EXPECT_NEAR(weight, 1.0, 1e-10) << "orbital " << i;
  }

  const auto ours = fb::partial_charges(molecule->atoms(), *populations);
  ASSERT_TRUE(ours.has_value()) << ours.error().message;
  const auto charges = entry.at("iao_charges").get<std::vector<double>>();
  ASSERT_EQ(charges.size(), n_atoms);
  ASSERT_EQ(ours->size(), n_atoms);
  for (std::uint32_t atom = 0; atom < n_atoms; ++atom) {
    EXPECT_NEAR((*ours)[atom], charges[atom], kChargeTolerance) << "atom " << atom;
  }
  // Each electron is counted once: the charges sum to the molecule's.
  EXPECT_NEAR(std::accumulate(ours->begin(), ours->end(), 0.0), molecule->charge(), 1e-10);
}

TEST(PipelineIbo, TheIaosAreOrthonormalSpanTheOccupiedAndGivePyscfsCharges) {
  FBF_REQUIRE_GPU();
  for (std::string_view name : {"h2o_sto3g", "h2o_ccpvdz", "nh4_ccpvdz"}) {
    SCOPED_TRACE(name);
    expect_iaos(reference_case(name, "ibo.json"));
  }
}

// cc-pVDZ with oxygen's outer s function given twice, read from a path: the
// same space, so the same SCF and IAOs, over an exactly singular S1. Its
// Cholesky factor has a pivot at rounding, below orthogonalize's floor, so
// S1^-1 is applied as X X^T over X = S1^-1/2, and the charges are still the
// plain cc-pVDZ ones PySCF gave. (PySCF's own fallback, canonical
// orthogonalization, is the same pseudo-inverse here: both drop the one null
// mode and nothing else.)
TEST(PipelineIbo, ASingularOverlapFallsBackToTheInverseRoot) {
  FBF_REQUIRE_GPU();
  auto document = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / ".." / ".." / ".." /
                                "data" / "basis" / "cc-pVDZ.json");
  ASSERT_TRUE(document.has_value()) << document.error().message;
  fb::json &coefficients =
      document->at("elements").at("8").at("electron_shells").at(0).at("coefficients");
  ASSERT_EQ(coefficients.size(), 3U);
  coefficients.push_back(coefficients.back());
  expect_iaos(reference_case("h2o_ccpvdz", "ibo.json"), &*document);
}

// partial_charges is host arithmetic, and refuses populations that are not
// whole columns of the atoms.
TEST(Population, PartialChargesAreZLessTheOccupiedPopulations) {
  const std::array atoms{fb::Atom{{0.0, 0.0, 0.0}, fb::Element::O},
                         fb::Atom{{0.0, 0.0, 1.8}, fb::Element::H}};
  // Two orbitals: one all on O, one split 0.75 / 0.25.
  const std::array populations{1.0, 0.0, 0.75, 0.25};
  const auto charges = fb::partial_charges(atoms, populations);
  ASSERT_TRUE(charges.has_value()) << charges.error().message;
  EXPECT_DOUBLE_EQ((*charges)[0], 8.0 - (2.0 * 1.75));
  EXPECT_DOUBLE_EQ((*charges)[1], 1.0 - (2.0 * 0.25));
  const auto one = fb::partial_charges(atoms, populations, 1.0);
  ASSERT_TRUE(one.has_value());
  EXPECT_DOUBLE_EQ((*one)[0], 8.0 - 1.75);

  const std::array ragged{1.0, 0.0, 0.5};
  const auto refused = fb::partial_charges(atoms, ragged);
  ASSERT_FALSE(refused.has_value());
  EXPECT_EQ(refused.error().kind, fb::ErrorKind::InvalidConfig);
}

// orbital_populations and bond_orbitals over hand-written weights, on a
// two-atom minimal basis: an s and a p shell on atom 0, an s on atom 1.
TEST(Bond, CentersAndCharacterFromTheWeights) {
  const fb::Vec<double, 3> origin{0.0, 0.0, 0.0};
  const auto shell = [&](std::int32_t l) {
    return fb::Shell(l, std::vector<double>{1.0}, std::vector<double>{1.0}, origin);
  };
  const fb::BasisSet minimal({shell(0), shell(1), shell(0)}, {0, 2, 3});
  ASSERT_EQ(minimal.nao_sphe(), 5U);
  // Functions s0 p0x p0y p0z s1. A bond, 60% on atom 0 (a third s) and 40%
  // on atom 1; then a lone pair, 97% on atom 0's s.
  const std::array weights{0.2, 0.1, 0.2, 0.1, 0.4, 0.97, 0.0, 0.0, 0.0, 0.03};

  const auto populations = fb::orbital_populations(minimal, 2, weights);
  ASSERT_TRUE(populations.has_value()) << populations.error().message;
  ASSERT_EQ(populations->size(), 4U);
  EXPECT_DOUBLE_EQ((*populations)[0], 0.6);
  EXPECT_DOUBLE_EQ((*populations)[1], 0.4);
  EXPECT_DOUBLE_EQ((*populations)[2], 0.97);
  EXPECT_DOUBLE_EQ((*populations)[3], 0.03);

  const auto bonds = fb::bond_orbitals(minimal, 2, weights);
  ASSERT_TRUE(bonds.has_value()) << bonds.error().message;
  ASSERT_EQ(bonds->size(), 2U);
  const auto &bond = (*bonds)[0].centers;
  ASSERT_EQ(bond.size(), 2U);
  EXPECT_EQ(bond[0].atom, 0U);
  EXPECT_DOUBLE_EQ(bond[0].population, 0.6);
  EXPECT_NEAR(bond[0].character[0], 1.0 / 3.0, 1e-15);
  EXPECT_NEAR(bond[0].character[1], 2.0 / 3.0, 1e-15);
  EXPECT_EQ(bond[0].character[2], 0.0);
  EXPECT_EQ(bond[1].atom, 1U);
  EXPECT_DOUBLE_EQ(bond[1].character[0], 1.0);
  const auto &lone = (*bonds)[1].centers;
  ASSERT_EQ(lone.size(), 1U) << "3% is under the default threshold";
  EXPECT_EQ(lone[0].atom, 0U);
  EXPECT_DOUBLE_EQ(lone[0].character[0], 1.0);

  // A lower threshold takes the 3% too; the largest atom is a center even
  // above every population.
  EXPECT_EQ(fb::bond_orbitals(minimal, 2, weights, 0.01)->at(1).centers.size(), 2U);
  EXPECT_EQ(fb::bond_orbitals(minimal, 2, weights, 1.0)->at(0).centers.size(), 1U);

  EXPECT_FALSE(fb::bond_orbitals(minimal, 2, weights, 0.0).has_value());
  EXPECT_FALSE(fb::bond_orbitals(minimal, 3, weights).has_value());
  EXPECT_FALSE(fb::orbital_populations(minimal, 3, weights).has_value());
}

// molden set: the localized orbitals land in a Molden file, n_occ of them at
// occupation 2. Each energy is <i|F|i> = sum_k U_ki^2 eps_k, and U is
// orthogonal, so they sum to the canonical occupied energies.
TEST(PipelineLocalizationMolden, WritesTheLocalizedOrbitals) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_localization_molden");
  const std::filesystem::path path = dir / "h2o_pm.molden";
  auto calc = localize(reference_case("h2o_sto3g"), 2,
                       std::format("molden = \"{}\"\n", path.generic_string()));
  ASSERT_TRUE(calc.has_value());
  const auto &local = calc->get<fb::Localization>();
  const auto &scf = calc->get<fb::Scf>();
  ASSERT_TRUE(local.has_value() && scf.has_value());
  const auto occ = static_cast<std::size_t>(local->n_occ);

  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(file, line));
  EXPECT_EQ(line, "[Molden Format]");
  std::size_t orbitals = 0;
  std::size_t occupied = 0;
  double energy_sum = 0.0;
  while (std::getline(file, line)) {
    if (line.starts_with(" Ene=")) {
      ++orbitals;
      energy_sum += std::stod(line.substr(5));
    }
    occupied += line == " Occup= 2.000000" ? 1 : 0;
  }
  EXPECT_EQ(orbitals, occ);
  EXPECT_EQ(occupied, occ);
  double expected_sum = 0.0;
  for (std::size_t k = 0; k < occ; ++k) {
    expected_sum += scf->mo_energies[k];
  }
  EXPECT_NEAR(energy_sum, expected_sum, 1e-10);
}

} // namespace
