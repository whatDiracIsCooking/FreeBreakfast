// The Localization step against PySCF's Pipek-Mezey and intrinsic bond
// orbitals (data/localization/{pipek_mezey,ibo,rohf}.json,
// tools/emit/localization_reference.py), through the real registry: a config
// naming [localization] plans the whole chain to the ScfOutput and localizes its
// occupied orbitals. The two codes normalise and order their AOs differently,
// so the orbitals are compared by what is invariant to that: the functional,
// and each orbital's Mulliken population on each atom, matched to PySCF's
// orbitals up to order. LocalizationOutput is a rotation of the occupied space, so
// the localized orbitals are also S-orthonormal and give back the SCF's own
// density, D = 2 C_local C_local^T. The IAOs are checked on their own by their
// partial charges. An ROHF reference's doubly and singly occupied spaces are
// each localized within itself, and matched to PySCF's per space. Runs the
// device chain, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "slow_test.h"

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

// One case of localization/@p file: pipek_mezey.json, ibo.json or rohf.json.
const fb::json &reference_case(std::string_view name, std::string_view file = "pipek_mezey.json") {
  static const fb::json kPipekMezey = read_reference("pipek_mezey.json");
  static const fb::json kIbo = read_reference("ibo.json");
  static const fb::json kRohf = read_reference("rohf.json");
  const fb::json &doc = file == "ibo.json" ? kIbo : file == "rohf.json" ? kRohf : kPipekMezey;
  for (const fb::json &entry : doc.at("cases")) {
    if (entry.at("name").get<std::string>() == name) {
      return entry;
    }
  }
  ADD_FAILURE() << "no case " << name << " in localization/" << file;
  static const fb::json kNone;
  return kNone;
}

// The entry's record for @p exponent in its @p key list, or null.
const fb::json *exponent_record(const fb::json &entry, std::int32_t exponent,
                                std::string_view key = "localized") {
  for (const fb::json &localized : entry.at(key)) {
    if (localized.at("exponent").get<std::int32_t>() == exponent) {
      return &localized;
    }
  }
  return nullptr;
}

template <typename Buffer>
std::vector<double> download(fb::DeviceResources &res, const Buffer &buffer, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), buffer.data(), sizeof(double) * count,
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

// The reference's molecule localized with the given exponent, from a config
// naming only the inputs and [localization], plus @p extra in that table. An
// entry with an aux_basis (rohf.json's) is run at its multiplicity over
// density-fitted ERIs, which the ROHF loop needs. A
// @p basis document, if given, is read from a file in place of the entry's
// basis. The run's outputs land in @p output_dir, or in a directory removed
// on return if none is given.
std::optional<fb::Calculation> localize(const fb::json &entry, std::int32_t exponent,
                                        std::string_view extra = "",
                                        const fb::json *basis = nullptr,
                                        std::filesystem::path output_dir = {}) {
  const fb::test::ScratchDir dir("pipeline_localization");
  if (output_dir.empty()) {
    output_dir = dir / "out";
  }
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
  std::string multiplicity;
  std::string eri;
  if (entry.contains("aux_basis")) {
    multiplicity = std::format("multiplicity = {}\n", entry.at("multiplicity").get<std::int32_t>());
    eri = std::format("[eri]\nmode = \"density_fitting\"\naux_basis = \"{}\"\n",
                      entry.at("aux_basis").get<std::string>());
  }
  // Converged past the comparison's needs: the default 1e-10 on L leaves the
  // populations ~1e-5 from the maximum.
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n{}[basis_set]\nbasis = \"{}\"\n{}"
      "[localization]\nexponent = {}\nfunctional_tolerance = 1e-13\n{}",
      entry.at("charge").get<std::int32_t>(), multiplicity, basis_spec, eri, exponent, extra);
  EXPECT_TRUE(fb::write_file(dir / "run.toml", toml).has_value());
  auto calc = fb::read_config(dir / "run.toml", dir.path(), output_dir);
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

// Each of @p occ orbitals' Mulliken population on each atom, orbital by
// atom: sum over the atom's AOs of C (S C). A placed basis keeps an atom's
// shells, and so its AOs, together.
std::vector<std::vector<double>> mulliken_populations(const fb::BasisSet &basis,
                                                      const std::vector<double> &s,
                                                      const std::vector<double> &c,
                                                      std::size_t occ) {
  const std::size_t n = basis.nao_sphe();
  const std::uint32_t n_atoms = basis.natoms();
  std::vector<std::size_t> first_ao(n_atoms + 1, n);
  for (std::uint32_t a = 0; a < n_atoms; ++a) {
    first_ao[a] = basis.ao_offset_sphe(basis.atom_shell_offset(a));
  }
  std::vector<std::vector<double>> q(occ, std::vector<double>(n_atoms, 0.0));
  for (std::size_t i = 0; i < occ; ++i) {
    for (std::uint32_t a = 0; a < n_atoms; ++a) {
      for (std::size_t mu = first_ao[a]; mu < first_ao[a + 1]; ++mu) {
        double sc = 0.0;
        for (std::size_t nu = 0; nu < n; ++nu) {
          sc += s[mu + (n * nu)] * c[nu + (n * i)];
        }
        q[i][a] += c[mu + (n * i)] * sc;
      }
    }
  }
  return q;
}

// Every PySCF orbital of @p theirs matches a distinct one of @p ours,
// population by population: the orbitals are the same up to order (and sign,
// which the populations do not see).
//
// @p one_center_sets is for IBO. Two orbitals both wholly on one atom by the
// IAO populations (water's O 1s and lone pairs) can rotate into each other at
// almost no cost in L, only the other atoms' small tails telling them apart:
// L moves by 1e-13 where their Mulliken populations move by 1e-3, between
// PySCF's own two IBO optimizers as much as between it and this code. Such
// orbitals are fixed only as a set, so they are compared by the set's summed
// populations, which are.
void expect_populations_match(const std::vector<std::vector<double>> &ours,
                              const std::vector<std::vector<double>> &theirs,
                              bool one_center_sets) {
  const std::size_t occ = ours.size();
  ASSERT_EQ(theirs.size(), occ);
  if (occ == 0) {
    return;
  }
  const auto n_atoms = static_cast<std::uint32_t>(ours[0].size());
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
}

// The @p occ localized orbitals @p c are S-orthonormal and give back the
// SCF's density @p d: twice their first @p n_doubly, once the rest.
void expect_orthonormal_with_density(const std::vector<double> &c, const std::vector<double> &s,
                                     const std::vector<double> &d, std::size_t n, std::size_t occ,
                                     std::size_t n_doubly) {
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
        density += (i < n_doubly ? 2.0 : 1.0) * c[mu + (n * i)] * c[nu + (n * i)];
      }
      EXPECT_NEAR(density, d[mu + (n * nu)], 1e-10) << mu << "," << nu;
    }
  }
}

// The localized orbitals of @p calc against the reference record @p expected:
// the functional, every orbital's Mulliken population on every atom matched to
// the reference's up to order (expect_populations_match, which has
// @p one_center_sets), S-orthonormality and the SCF's density.
void expect_matches(const fb::json &entry, const fb::json &expected, const fb::Calculation &calc,
                    bool one_center_sets = false) {
  const auto &local = calc.get<fb::LocalizationOutput>();
  const auto &scf = calc.get<fb::ScfOutput>();
  const auto &overlap = calc.get<fb::OverlapOutput>();
  const auto &basis = calc.get<fb::BasisSetOutput>();
  ASSERT_TRUE(local.has_value() && scf.has_value() && overlap.has_value() && basis.has_value());

  const std::size_t n = basis->basis.nao_sphe();
  const auto occ = static_cast<std::size_t>(local->n_occ);
  ASSERT_EQ(occ * 2, entry.at("electrons").get<std::size_t>());
  EXPECT_EQ(local->n_doubly, local->n_occ);
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  fb::DeviceResources res(scf->c.dev_idx());
  const std::vector<double> c = download(res, local->c, n * occ);
  const std::vector<double> s = download(res, overlap->s, n * n);
  const std::vector<double> d = download(res, scf->d, n * n);

  EXPECT_NEAR(local->info.functional, expected.at("functional").get<double>(),
              kFunctionalTolerance);
  EXPECT_GT(local->info.functional, local->info.initial_functional);
  ASSERT_EQ(basis->basis.natoms(), entry.at("atoms").size());
  expect_populations_match(mulliken_populations(basis->basis, s, c, occ),
                           expected.at("populations").get<std::vector<std::vector<double>>>(),
                           one_center_sets);
  expect_orthonormal_with_density(c, s, d, n, occ, occ);
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
  const auto &local = calc->get<fb::LocalizationOutput>();
  ASSERT_TRUE(local.has_value());
  EXPECT_TRUE(local->populations.empty() && local->charges.empty()) << "IBO only";
  EXPECT_EQ(local->n_vvo, 0) << "LIVVO only";
  EXPECT_TRUE(local->livvo_bonds.empty() && local->valence_character.empty()) << "LIVVO only";
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
  const auto &local = calc->get<fb::LocalizationOutput>();
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

// The IAOs on their own, from the ScfOutput's orbitals over the entry's basis (or
// @p basis_document): orthonormal, spanning the occupied orbitals (each has
// unit weight on them), and giving PySCF's partial charges,
// Z_A - 2 sum_i sum_{r in A} (A^T S C)_ri^2.
void expect_iaos(const fb::json &entry, const fb::json *basis_document = nullptr) {
  auto calc = localize(entry, 4, "mode = \"ibo\"\n", basis_document);
  ASSERT_TRUE(calc.has_value());
  const auto &molecule = calc->get<fb::MoleculeOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(molecule.has_value() && basis.has_value() && overlap.has_value() && scf.has_value());
  auto stamp = fb::resolve_basis("minao");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  auto minao = fb::place(stamp->primitive_normalized(), molecule->molecule);
  ASSERT_TRUE(minao.has_value()) << minao.error().message;

  const std::size_t n = basis->basis.nao_sphe();
  const std::size_t m = minao->nao_sphe();
  const auto occ = static_cast<std::size_t>(molecule->molecule.electron_count() / 2);
  fb::DeviceResources res(scf->c.dev_idx());
  fb::Buffers::device<double> iao(n * m, res.device());
  const fb::Status built = fb::localization::intrinsic_atomic_orbitals(
      res, basis->basis, *minao, static_cast<std::int32_t>(occ), overlap->s.data(), scf->c.data(),
      iao.data());
  ASSERT_TRUE(built.has_value()) << built.error().message;
  ASSERT_TRUE(res.sync().has_value());
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const std::vector<double> a = download(res, iao, n * m);
  const std::vector<double> s = download(res, overlap->s, n * n);

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
      fb::orbital_populations(res, basis->basis, *minao, static_cast<std::int32_t>(occ),
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

  const auto ours = fb::partial_charges(molecule->molecule.atoms(), *populations);
  ASSERT_TRUE(ours.has_value()) << ours.error().message;
  const auto charges = entry.at("iao_charges").get<std::vector<double>>();
  ASSERT_EQ(charges.size(), n_atoms);
  ASSERT_EQ(ours->size(), n_atoms);
  for (std::uint32_t atom = 0; atom < n_atoms; ++atom) {
    EXPECT_NEAR((*ours)[atom], charges[atom], kChargeTolerance) << "atom " << atom;
  }
  // Each electron is counted once: the charges sum to the molecule's.
  EXPECT_NEAR(std::accumulate(ours->begin(), ours->end(), 0.0), molecule->molecule.charge(), 1e-10);
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

// bond_orbitals' sigma / pi kind over hand-written weights, on two atoms
// each with an s and a p shell: pi when every center is at least
// pi_threshold p, one-center orbitals included.
TEST(Bond, PiWhenEveryCenterIsAlmostAllP) {
  const fb::Vec<double, 3> origin{0.0, 0.0, 0.0};
  const auto shell = [&](std::int32_t l) {
    return fb::Shell(l, std::vector<double>{1.0}, std::vector<double>{1.0}, origin);
  };
  const fb::BasisSet minimal({shell(0), shell(1), shell(0), shell(1)}, {0, 2, 4});
  ASSERT_EQ(minimal.nao_sphe(), 8U);
  // Functions s0 p0x p0y p0z s1 p1x p1y p1z. A pi bond, all p on both
  // atoms; a bond all p on atom 0 but 20% s on atom 1; a one-center pure-p
  // lone pair, with an s tail under the threshold; a lone pair half s.
  const std::array weights{0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5, //
                           0.0, 0.0,  0.0, 0.5, 0.1,  0.0, 0.0, 0.4, //
                           0.0, 0.97, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, //
                           0.5, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.0};
  const auto bonds = fb::bond_orbitals(minimal, 4, weights);
  ASSERT_TRUE(bonds.has_value()) << bonds.error().message;
  ASSERT_EQ(bonds->size(), 4U);
  EXPECT_EQ((*bonds)[0].kind, fb::BondKind::Pi);
  EXPECT_EQ((*bonds)[1].kind, fb::BondKind::Sigma) << "atom 1 is s-bearing";
  ASSERT_EQ((*bonds)[2].centers.size(), 1U);
  EXPECT_EQ((*bonds)[2].kind, fb::BondKind::Pi) << "a pure-p lone pair is pi";
  EXPECT_EQ((*bonds)[3].kind, fb::BondKind::Sigma);

  // The rule at its edge, on the second orbital's s-bearing center: pi just
  // below its p character, sigma just above.
  const double p_share = (*bonds)[1].centers[1].character[1];
  EXPECT_NEAR(p_share, 0.8, 1e-15);
  EXPECT_EQ(fb::bond_orbitals(minimal, 4, weights, 0.1, p_share - 1e-12)->at(1).kind,
            fb::BondKind::Pi);
  EXPECT_EQ(fb::bond_orbitals(minimal, 4, weights, 0.1, p_share + 1e-12)->at(1).kind,
            fb::BondKind::Sigma);
  // A pi_threshold of one keeps only the pure-p orbitals pi.
  const auto pure = fb::bond_orbitals(minimal, 4, weights, 0.1, 1.0);
  ASSERT_TRUE(pure.has_value()) << pure.error().message;
  EXPECT_EQ(pure->at(0).kind, fb::BondKind::Pi);
  EXPECT_EQ(pure->at(2).kind, fb::BondKind::Pi);

  for (const double refused : {0.0, -0.5, 1.0 + 1e-12, std::numeric_limits<double>::quiet_NaN()}) {
    const auto out = fb::bond_orbitals(minimal, 4, weights, 0.1, refused);
    ASSERT_FALSE(out.has_value()) << refused;
    EXPECT_EQ(out.error().kind, fb::ErrorKind::InvalidConfig);
  }
}

// A localize() entry for a closed-shell neutral molecule from Angstrom
// coordinates, as the reference files give one in Bohr.
fb::json molecule_entry(std::string_view name,
                        std::span<const std::pair<std::string_view, fb::Vec<double, 3>>> atoms) {
  fb::json entry{{"name", name}, {"charge", 0}, {"basis", "cc-pvdz"}, {"atoms", fb::json::array()}};
  for (const auto &[symbol, coord] : atoms) {
    entry.at("atoms").push_back(
        {{"symbol", symbol},
         {"coord", std::vector<double>{coord.x / fb::kAngstromPerBohr<double>,
                                       coord.y / fb::kAngstromPerBohr<double>,
                                       coord.z / fb::kAngstromPerBohr<double>}}});
  }
  return entry;
}

// The kinds of the Localization step's IBOs, at the default exponent of four
// (which separates sigma and pi, rather than giving banana bonds): the
// cores, and the sigma and pi orbitals over two or more centers, counted.
struct KindCounts {
  std::size_t cores = 0;
  std::size_t sigma = 0;
  std::size_t pi = 0;
};
KindCounts ibo_kinds(const fb::json &entry) {
  KindCounts counts;
  const auto calc = localize(entry, 4, "mode = \"ibo\"\n");
  if (!calc.has_value()) {
    return counts;
  }
  const auto &local = calc->get<fb::LocalizationOutput>();
  EXPECT_TRUE(local.has_value());
  if (!local.has_value()) {
    return counts;
  }
  EXPECT_EQ(local->bonds.size(), static_cast<std::size_t>(local->n_occ));
  for (const fb::BondOrbital &orbital : local->bonds) {
    if (orbital.centers.size() == 1) {
      EXPECT_EQ(orbital.kind, fb::BondKind::Sigma) << "a carbon 1s core";
      ++counts.cores;
    } else {
      ++(orbital.kind == fb::BondKind::Pi ? counts.pi : counts.sigma);
    }
  }
  return counts;
}

// Ethylene, cc-pVDZ: two sigma cores, five sigma bonds (four CH, one CC)
// and one pi bond.
TEST(PipelineIbo, EthyleneHasOnePiBond) {
  FBF_REQUIRE_GPU();
  const std::array<std::pair<std::string_view, fb::Vec<double, 3>>, 6> atoms{{
      {"C", {0.6695, 0.0, 0.0}},
      {"C", {-0.6695, 0.0, 0.0}},
      {"H", {1.2321, 0.9289, 0.0}},
      {"H", {1.2321, -0.9289, 0.0}},
      {"H", {-1.2321, 0.9289, 0.0}},
      {"H", {-1.2321, -0.9289, 0.0}},
  }};
  const KindCounts counts = ibo_kinds(molecule_entry("ethylene", atoms));
  EXPECT_EQ(counts.cores, 2U);
  EXPECT_EQ(counts.sigma, 5U);
  EXPECT_EQ(counts.pi, 1U);
}

// Benzene, cc-pVDZ: six cores, twelve sigma bonds and three pi bonds. Each
// pi IBO is half on one carbon and a fifth on each neighbour, so three
// centers at the default threshold, every one of them all p.
TEST(PipelineIbo, BenzeneHasThreePiBonds) {
  FBF_REQUIRE_GPU();
  FBF_SKIP_IF_FAST_TESTS();
  constexpr double kCarbon = 1.3915;
  constexpr double kHydrogen = 2.4715;
  std::vector<std::pair<std::string_view, fb::Vec<double, 3>>> atoms;
  for (const double radius : {kCarbon, kHydrogen}) {
    for (std::int32_t k = 0; k < 6; ++k) {
      const double angle = k * fb::kPi<double> / 3.0;
      atoms.emplace_back(
          radius == kCarbon ? "C" : "H",
          fb::Vec<double, 3>{radius * std::cos(angle), radius * std::sin(angle), 0.0});
    }
  }
  const KindCounts counts = ibo_kinds(molecule_entry("benzene", atoms));
  EXPECT_EQ(counts.cores, 6U);
  EXPECT_EQ(counts.sigma, 12U);
  EXPECT_EQ(counts.pi, 3U);
}

// The localized orbitals land in the output directory's localization.molden,
// n_occ of them at occupation 2, then the canonical virtuals at occupation 0.
// Each localized energy is <i|F|i> = sum_k U_ki^2 eps_k, and U is orthogonal,
// so they sum to the canonical occupied energies; each virtual keeps its own.
// The step sorts the localized ones, so the whole file ascends.
TEST(PipelineLocalizationMolden, WritesTheLocalizedOrbitals) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_localization_molden");
  const std::filesystem::path path = dir / "localization.molden";
  auto calc = localize(reference_case("h2o_sto3g"), 2, "", nullptr, dir.path());
  ASSERT_TRUE(calc.has_value());
  const auto &local = calc->get<fb::LocalizationOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(local.has_value() && scf.has_value());
  const auto occ = static_cast<std::size_t>(local->n_occ);
  const std::size_t n_mo = scf->mo_energies.size();
  ASSERT_GT(n_mo, occ);

  std::ifstream file(path);
  ASSERT_TRUE(file.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(file, line));
  EXPECT_EQ(line, "[Molden Format]");
  std::vector<double> energies;
  std::vector<std::string> occupations;
  while (std::getline(file, line)) {
    if (line.starts_with(" Ene=")) {
      energies.push_back(std::stod(line.substr(5)));
    }
    if (line.starts_with(" Occup=")) {
      occupations.push_back(line);
    }
  }
  ASSERT_EQ(energies.size(), n_mo);
  ASSERT_EQ(occupations.size(), n_mo);
  EXPECT_TRUE(std::ranges::is_sorted(energies));
  double energy_sum = 0.0;
  double expected_sum = 0.0;
  for (std::size_t k = 0; k < occ; ++k) {
    EXPECT_EQ(occupations[k], " Occup= 2.000000") << k;
    energy_sum += energies[k];
    expected_sum += scf->mo_energies[k];
  }
  EXPECT_NEAR(energy_sum, expected_sum, 1e-10);
  for (std::size_t a = occ; a < n_mo; ++a) {
    EXPECT_EQ(occupations[a], " Occup= 0.000000") << a;
    EXPECT_NEAR(energies[a], scf->mo_energies[a], 1e-10) << a;
  }
}

// Water's LIVVOs over MINAO, in either mode: its two sigma*_OH, each on O
// (atom 0) and one hydrogen, a different one each; and the canonical
// virtuals' valence characters, each in [0, 1] and summing to n_vvo, the
// LIVVOs lying wholly in the virtual space (localization/livvo's README).
// IBO's populations, charges and bonds describe the occupied IBOs, so
// Pipek-Mezey leaves them empty even though LIVVO placed MINAO.
class PipelineLivvo : public ::testing::TestWithParam<std::string_view> {};

TEST_P(PipelineLivvo, WatersTwoSigmaStarOh) {
  FBF_REQUIRE_GPU();
  const bool ibo = GetParam() == "ibo";
  auto calc = localize(reference_case("h2o_ccpvdz"), ibo ? 4 : 2,
                       std::format("mode = \"{}\"\nlivvo = true\n", GetParam()));
  ASSERT_TRUE(calc.has_value());
  const auto &local = calc->get<fb::LocalizationOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(local.has_value() && scf.has_value());
  const auto occ = static_cast<std::size_t>(local->n_occ);
  ASSERT_EQ(local->n_vvo, 2);
  EXPECT_EQ(local->livvo.num_elements(), calc->get<fb::BasisSetOutput>()->basis.nao_sphe() * 2);
  EXPECT_GE(local->livvo_info.functional, local->livvo_info.initial_functional);
  EXPECT_EQ(local->populations.empty(), !ibo) << "IBO only";
  EXPECT_EQ(local->charges.empty(), !ibo) << "IBO only";
  EXPECT_EQ(local->bonds.empty(), !ibo) << "IBO only";

  ASSERT_EQ(local->livvo_bonds.size(), 2U);
  std::array<bool, 3> on_hydrogen{};
  for (const fb::BondOrbital &orbital : local->livvo_bonds) {
    EXPECT_EQ(orbital.kind, fb::BondKind::Sigma);
    ASSERT_EQ(orbital.centers.size(), 2U);
    const std::uint32_t a = orbital.centers[0].atom;
    const std::uint32_t b = orbital.centers[1].atom;
    ASSERT_TRUE(std::min(a, b) == 0 && std::max(a, b) > 0) << "on " << a << " and " << b;
    const std::uint32_t hydrogen = std::max(a, b);
    EXPECT_FALSE(on_hydrogen[hydrogen]) << "two LIVVOs on atom " << hydrogen;
    on_hydrogen[hydrogen] = true;
    EXPECT_NEAR(orbital.centers[0].population + orbital.centers[1].population, 1.0, 1e-3);
  }

  ASSERT_EQ(local->valence_character.size(), scf->mo_energies.size() - occ);
  double sum = 0.0;
  for (const double t_val : local->valence_character) {
    EXPECT_GE(t_val, -1e-12);
    EXPECT_LE(t_val, 1.0 + 1e-12);
    sum += t_val;
  }
  EXPECT_NEAR(sum, static_cast<double>(local->n_vvo), 1e-8);
}

INSTANTIATE_TEST_SUITE_P(Modes, PipelineLivvo, ::testing::Values("pipek_mezey", "ibo"),
                         [](const ::testing::TestParamInfo<std::string_view> &info) {
                           return std::string{info.param};
                         });

// The localized orbitals come out in ascending order of <i|F|i> =
// sum_k U_ki^2 eps_k, in either mode, with C and U permuted together: C is
// still C_occ U, column for column.
class PipelineLocalizationOrder : public ::testing::TestWithParam<std::string_view> {};

TEST_P(PipelineLocalizationOrder, AscendsInFockEnergy) {
  FBF_REQUIRE_GPU();
  const bool ibo = GetParam() == "ibo";
  auto calc = localize(reference_case("h2o_ccpvdz"), ibo ? 4 : 2,
                       std::format("mode = \"{}\"\n", GetParam()));
  ASSERT_TRUE(calc.has_value());
  const auto &local = calc->get<fb::LocalizationOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(local.has_value() && scf.has_value());
  const std::size_t n = calc->get<fb::BasisSetOutput>()->basis.nao_sphe();
  const auto occ = static_cast<std::size_t>(local->n_occ);
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  fb::DeviceResources res(scf->c.dev_idx());
  const std::vector<double> c = download(res, local->c, n * occ);
  const std::vector<double> u = download(res, local->u, occ * occ);
  const std::vector<double> c_occ = download(res, scf->c, n * occ);

  std::vector<double> energies(occ, 0.0);
  for (std::size_t i = 0; i < occ; ++i) {
    for (std::size_t k = 0; k < occ; ++k) {
      energies[i] += u[k + (occ * i)] * u[k + (occ * i)] * scf->mo_energies[k];
    }
  }
  EXPECT_TRUE(std::ranges::is_sorted(energies)) << "water's 1s first, then up to the HOMO";
  for (std::size_t i = 0; i < occ; ++i) {
    for (std::size_t mu = 0; mu < n; ++mu) {
      double cu = 0.0;
      for (std::size_t k = 0; k < occ; ++k) {
        cu += c_occ[mu + (n * k)] * u[k + (occ * i)];
      }
      EXPECT_NEAR(c[mu + (n * i)], cu, 1e-12) << "AO " << mu << ", orbital " << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Modes, PipelineLocalizationOrder, ::testing::Values("pipek_mezey", "ibo"),
                         [](const ::testing::TestParamInfo<std::string_view> &info) {
                           return std::string{info.param};
                         });

// The LIVVOs land in localization_livvo.molden beside the localized
// orbitals, n_vvo of them at occupation 0. Each energy is a weighted mean of
// the canonical virtual energies, sum_a O_al^2 eps_a with sum_a O_al^2 = 1,
// so it lies within their range.
TEST(PipelineLocalizationMolden, WritesTheLivvos) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_localization_livvo_molden");
  auto calc = localize(reference_case("h2o_ccpvdz"), 2, "livvo = true\n", nullptr, dir.path());
  ASSERT_TRUE(calc.has_value());
  const auto &local = calc->get<fb::LocalizationOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  ASSERT_TRUE(local.has_value() && scf.has_value());
  const auto occ = static_cast<std::size_t>(local->n_occ);

  std::ifstream file(dir / "localization_livvo.molden");
  ASSERT_TRUE(file.is_open());
  std::string line;
  ASSERT_TRUE(std::getline(file, line));
  EXPECT_EQ(line, "[Molden Format]");
  std::vector<double> energies;
  std::size_t empty = 0;
  while (std::getline(file, line)) {
    if (line.starts_with(" Ene=")) {
      energies.push_back(std::stod(line.substr(5)));
    }
    empty += line == " Occup= 0.000000" ? 1 : 0;
  }
  ASSERT_EQ(energies.size(), static_cast<std::size_t>(local->n_vvo));
  EXPECT_EQ(empty, energies.size());
  const double lowest = scf->mo_energies[occ];
  const double highest = scf->mo_energies.back();
  for (const double energy : energies) {
    EXPECT_GE(energy, lowest - 1e-10);
    EXPECT_LE(energy, highest + 1e-10);
  }
  EXPECT_TRUE(std::filesystem::exists(dir / "localization.molden"));
}

// livvo_molden names the LIVVOs' file, so a hand-built step setting it
// without livvo is refused when it runs. A config cannot ask for that: it
// sets livvo_molden only beside livvo.
TEST(PipelineLocalization, ALivvoMoldenWithoutLivvoIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  const fb::test::ScratchDir dir("pipeline_localization_livvo_without");
  ASSERT_TRUE(fb::write_file(dir / "water.xyz", "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\n"
                                                "H 0 -0.7572 -0.4692\n")
                  .has_value());
  auto calc = fb::Calculation::create({fb::MoleculeStep{.xyz = dir / "water.xyz"},
                                       fb::BasisSetStep{.basis = "sto-3g"},
                                       fb::LocalizationStep{.livvo_molden = dir / "livvo.molden"}},
                                      []<class S>() -> fb::Result<S> { return S{}; });
  ASSERT_TRUE(calc.has_value()) << calc.error().message;
  fb::NodeResources node;
  const fb::Status ran = calc->run(node);
  ASSERT_FALSE(ran.has_value());
  EXPECT_EQ(ran.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(ran.error().message.find("livvo_molden needs livvo"), std::string::npos)
      << ran.error().message;
  EXPECT_FALSE(std::filesystem::exists(dir / "livvo.molden"));
}

// An ROHF reference (rohf.json): the doubly occupied orbitals localized
// among themselves, then the singly occupied ones among themselves, in
// either mode, each space matched to PySCF's localization of that space
// alone. H2O+ is a doublet, whose one singly occupied orbital has nothing to
// rotate with; triplet CH2's two are localized as a pair. The step's c holds
// the doubly occupied space first (n_doubly = n_beta columns), U is
// block-diagonal, so no rotation mixes the spaces, and the occupied density,
// 2 C_d C_d^T + C_s C_s^T, is the SCF's. Each space ascends in <i|F|i> on its
// own, and the Molden file has them at occupations 2 and 1.
struct OpenShellCase {
  std::string_view name;
  std::string_view mode;
  std::int32_t exponent;
};

void PrintTo(const OpenShellCase &c, std::ostream *os) {
  *os << c.name << "_" << c.mode << "_p" << c.exponent;
}

class PipelineLocalizationOpenShell : public ::testing::TestWithParam<OpenShellCase> {};

TEST_P(PipelineLocalizationOpenShell, EachSpaceMatchesPyscfAndKeepsTheDensity) {
  FBF_REQUIRE_GPU();
  const OpenShellCase param = GetParam();
  const bool ibo = param.mode == "ibo";
  const fb::json &entry = reference_case(param.name, "rohf.json");
  const fb::json *expected = exponent_record(entry, param.exponent, ibo ? "ibo" : "pipek_mezey");
  ASSERT_NE(expected, nullptr) << "no exponent " << param.exponent;
  const fb::test::ScratchDir dir("pipeline_localization_open_shell");
  auto calc = localize(entry, param.exponent, std::format("mode = \"{}\"\n", param.mode), nullptr,
                       dir.path());
  ASSERT_TRUE(calc.has_value());
  const auto &local = calc->get<fb::LocalizationOutput>();
  const auto &scf = calc->get<fb::ScfOutput>();
  const auto &overlap = calc->get<fb::OverlapOutput>();
  const auto &basis = calc->get<fb::BasisSetOutput>();
  ASSERT_TRUE(local.has_value() && scf.has_value() && overlap.has_value() && basis.has_value());

  ASSERT_EQ(local->n_occ, entry.at("n_alpha").get<std::int32_t>());
  ASSERT_EQ(local->n_doubly, entry.at("n_beta").get<std::int32_t>());
  ASSERT_LT(local->n_doubly, local->n_occ);
  const std::size_t n = basis->basis.nao_sphe();
  const auto occ = static_cast<std::size_t>(local->n_occ);
  const auto doubly = static_cast<std::size_t>(local->n_doubly);
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  fb::DeviceResources res(scf->c.dev_idx());
  const std::vector<double> c = download(res, local->c, n * occ);
  const std::vector<double> u = download(res, local->u, occ * occ);
  const std::vector<double> c_occ = download(res, scf->c, n * occ);
  const std::vector<double> s = download(res, overlap->s, n * n);
  const std::vector<double> d = download(res, scf->d, n * n);

  // L summed over the spaces, each PySCF's own.
  const fb::json &doubly_ref = expected->at("doubly");
  const fb::json &singly_ref = expected->at("singly");
  EXPECT_NEAR(local->info.functional,
              doubly_ref.at("functional").get<double>() + singly_ref.at("functional").get<double>(),
              kFunctionalTolerance);
  EXPECT_GT(local->info.functional, local->info.initial_functional);

  const auto ours = mulliken_populations(basis->basis, s, c, occ);
  {
    SCOPED_TRACE("doubly occupied");
    expect_populations_match({ours.begin(), ours.begin() + static_cast<std::ptrdiff_t>(doubly)},
                             doubly_ref.at("populations").get<std::vector<std::vector<double>>>(),
                             ibo);
  }
  {
    SCOPED_TRACE("singly occupied");
    expect_populations_match({ours.begin() + static_cast<std::ptrdiff_t>(doubly), ours.end()},
                             singly_ref.at("populations").get<std::vector<std::vector<double>>>(),
                             ibo);
  }
  expect_orthonormal_with_density(c, s, d, n, occ, doubly);

  // U is block-diagonal, C is C_occ U column for column, and each space
  // ascends in <i|F|i> = sum_k U_ki^2 eps_k.
  std::vector<double> energies(occ, 0.0);
  for (std::size_t i = 0; i < occ; ++i) {
    for (std::size_t k = 0; k < occ; ++k) {
      if ((i < doubly) != (k < doubly)) {
        EXPECT_EQ(u[k + (occ * i)], 0.0) << "U mixes the spaces at " << k << "," << i;
      }
      energies[i] += u[k + (occ * i)] * u[k + (occ * i)] * scf->mo_energies[k];
    }
    for (std::size_t mu = 0; mu < n; ++mu) {
      double cu = 0.0;
      for (std::size_t k = 0; k < occ; ++k) {
        cu += c_occ[mu + (n * k)] * u[k + (occ * i)];
      }
      EXPECT_NEAR(c[mu + (n * i)], cu, 1e-12) << "AO " << mu << ", orbital " << i;
    }
  }
  const auto split = energies.begin() + static_cast<std::ptrdiff_t>(doubly);
  EXPECT_TRUE(std::is_sorted(energies.begin(), split));
  EXPECT_TRUE(std::is_sorted(split, energies.end()));

  if (ibo) {
    // The IAO populations' p-th powers sum to L over both spaces, and the
    // charges count the doubly occupied orbitals twice and the singly
    // occupied once, PySCF's whichever orbitals gave them.
    const std::size_t n_atoms = entry.at("atoms").size();
    ASSERT_EQ(local->populations.size(), n_atoms * occ);
    double functional = 0.0;
    for (std::size_t i = 0; i < occ; ++i) {
      double weight = 0.0;
      for (std::size_t a = 0; a < n_atoms; ++a) {
        const double q = local->populations[a + (n_atoms * i)];
        weight += q;
        functional += std::pow(q, param.exponent);
      }
      EXPECT_NEAR(weight, 1.0, 1e-10) << "orbital " << i;
    }
    EXPECT_NEAR(functional, local->info.functional, 1e-10);
    const auto charges = entry.at("iao_charges").get<std::vector<double>>();
    ASSERT_EQ(local->charges.size(), charges.size());
    for (std::size_t a = 0; a < n_atoms; ++a) {
      EXPECT_NEAR(local->charges[a], charges[a], kChargeTolerance) << "atom " << a;
    }
    EXPECT_EQ(local->bonds.size(), occ);
  } else {
    EXPECT_TRUE(local->populations.empty() && local->charges.empty()) << "IBO only";
  }

  // The Molden file: the doubly occupied at 2, the singly occupied at 1, then
  // the virtuals at 0.
  std::ifstream file(dir / "localization.molden");
  ASSERT_TRUE(file.is_open());
  std::vector<std::string> occupations;
  for (std::string line; std::getline(file, line);) {
    if (line.starts_with(" Occup=")) {
      occupations.push_back(line);
    }
  }
  ASSERT_EQ(occupations.size(), scf->mo_energies.size());
  for (std::size_t k = 0; k < occupations.size(); ++k) {
    EXPECT_EQ(occupations[k], k < doubly ? " Occup= 2.000000"
                              : k < occ  ? " Occup= 1.000000"
                                         : " Occup= 0.000000")
        << k;
  }
}

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineLocalizationOpenShell,
                         ::testing::Values(OpenShellCase{"h2o_cation_ccpvdz", "pipek_mezey", 2},
                                           OpenShellCase{"h2o_cation_ccpvdz", "pipek_mezey", 4},
                                           OpenShellCase{"h2o_cation_ccpvdz", "ibo", 2},
                                           OpenShellCase{"h2o_cation_ccpvdz", "ibo", 4},
                                           OpenShellCase{"ch2_ccpvdz", "pipek_mezey", 2},
                                           OpenShellCase{"ch2_ccpvdz", "pipek_mezey", 4},
                                           OpenShellCase{"ch2_ccpvdz", "ibo", 2},
                                           OpenShellCase{"ch2_ccpvdz", "ibo", 4}),
                         [](const ::testing::TestParamInfo<OpenShellCase> &info) {
                           return std::format("{}_{}_p{}", info.param.name, info.param.mode,
                                              info.param.exponent);
                         });

} // namespace
