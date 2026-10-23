// The Localization step against PySCF's Pipek-Mezey (data/localization/
// pipek_mezey.json, tools/emit_localization_reference.py), through the real
// registry: a config naming [localization] plans the whole chain to the Scf
// and localizes its occupied orbitals. The two codes normalise and order their
// AOs differently, so the orbitals are compared by what is invariant to that:
// the functional, and each orbital's Mulliken population on each atom, matched
// to PySCF's orbitals up to order. Localization is a rotation of the occupied
// space, so the localized orbitals are also S-orthonormal and give back the
// SCF's own density, D = 2 C_local C_local^T. Runs the device chain, so needs a
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

const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "localization" /
                              "pipek_mezey.json");
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
  ADD_FAILURE() << "no case " << name << " in localization/pipek_mezey.json";
  static const fb::json kNone;
  return kNone;
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
// naming only the inputs and [localization].
std::optional<fb::Calculation> localize(const fb::json &entry, std::int32_t exponent) {
  const fb::test::ScratchDir dir("pipeline_localization");
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
  const std::string toml = std::format(
      "[molecule]\nxyz = \"molecule.xyz\"\ncharge = {}\n[basis_set]\nbasis = \"{}\"\n"
      "[localization]\nexponent = {}\nfunctional_tolerance = 1e-13\n",
      entry.at("charge").get<std::int32_t>(), entry.at("basis").get<std::string>(), exponent);
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

struct Case {
  std::string_view name;
  std::int32_t exponent;
};

// ctest names a discovered case by this, so `ctest -R h2o_ccpvdz_p4` selects one.
void PrintTo(const Case &c, std::ostream *os) {
  *os << c.name << "_p" << c.exponent;
}

class PipelineLocalization : public ::testing::TestWithParam<Case> {};

TEST_P(PipelineLocalization, MatchesPyscfAndKeepsTheDensity) {
  FBF_REQUIRE_GPU();
  const fb::json &entry = reference_case(GetParam().name);
  const fb::json *expected = nullptr;
  for (const fb::json &localized : entry.at("localized")) {
    if (localized.at("exponent").get<std::int32_t>() == GetParam().exponent) {
      expected = &localized;
    }
  }
  ASSERT_NE(expected, nullptr) << "no exponent " << GetParam().exponent;

  auto calc = localize(entry, GetParam().exponent);
  ASSERT_TRUE(calc.has_value());
  const auto &local = calc->get<fb::Localization>();
  const auto &scf = calc->get<fb::Scf>();
  const auto &overlap = calc->get<fb::Overlap>();
  const auto &basis = calc->get<fb::BasisSet>();
  ASSERT_TRUE(local.has_value() && scf.has_value() && overlap.has_value() && basis.has_value());

  const std::size_t n = basis->nao_sphe();
  const auto occ = static_cast<std::size_t>(local->n_occ);
  ASSERT_EQ(occ * 2, entry.at("electrons").get<std::size_t>());
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const std::vector<double> c = download(local->c, n * occ);
  const std::vector<double> s = download(overlap->s, n * n);
  const std::vector<double> d = download(scf->d, n * n);

  EXPECT_NEAR(local->info.functional, expected->at("functional").get<double>(),
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
  const auto theirs = expected->at("populations").get<std::vector<std::vector<double>>>();
  ASSERT_EQ(theirs.size(), occ);
  std::vector<bool> taken(occ, false);
  for (std::size_t k = 0; k < occ; ++k) {
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

INSTANTIATE_TEST_SUITE_P(Pyscf, PipelineLocalization,
                         ::testing::Values(Case{"h2o_sto3g", 2}, Case{"h2o_sto3g", 4},
                                           Case{"h2o_ccpvdz", 2}, Case{"h2o_ccpvdz", 4},
                                           Case{"nh4_ccpvdz", 2}, Case{"nh4_ccpvdz", 4}),
                         [](const auto &info) {
                           return std::format("{}_p{}", info.param.name, info.param.exponent);
                         });

} // namespace
