// The PipekMezey step through the real registry: a config naming
// [pipek_mezey] plans the whole chain to the Scf and localizes its occupied
// orbitals. Localization is a rotation of the occupied space, so the localized
// orbitals are S-orthonormal and give back the SCF's own density,
// D = 2 C_local C_local^T. Runs the device chain, so needs a GPU.

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

static_assert(fb::PipelineStep<fb::PipekMezeyStep>);
static_assert(fb::kRunShape<fb::PipekMezeyStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::PipekMezeyStep>,
                           fb::View<fb::Molecule, fb::BasisSet, fb::Overlap, fb::Scf>>);
static_assert(std::same_as<fb::StepOutput<fb::PipekMezeyStep>, fb::PipekMezey>);

template <typename Buffer>
std::vector<double> download(const Buffer &buffer, std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpy(host.data(), buffer.data(), sizeof(double) * count,
                           wwr::wwrMemcpyDeviceToHost),
            wwr::wwrSuccess);
  return host;
}

// Water in STO-3G, the reference geometry of data/scf/rhf.json's h2o_sto3g,
// localized with the given [pipek_mezey] table.
std::optional<fb::Calculation> localize_water(std::string_view table) {
  const fb::test::ScratchDir dir("pipeline_pipek_mezey");
  // Bohr in the reference, Angstrom in an XYZ file.
  const auto angstrom = [](double bohr) { return bohr * fb::kAngstromPerBohr<double>; };
  const std::string xyz = std::format("3\nwater\nO 0 0 {:.17g}\nH 0 {:.17g} {:.17g}\n"
                                      "H 0 {:.17g} {:.17g}\n",
                                      angstrom(0.2216), angstrom(1.4309), angstrom(-0.8867),
                                      angstrom(-1.4309), angstrom(-0.8867));
  EXPECT_TRUE(fb::write_file(dir / "water.xyz", xyz).has_value());
  EXPECT_TRUE(fb::write_file(dir / "run.toml", std::format("[molecule]\nxyz = \"water.xyz\"\n"
                                                           "[basis_set]\nbasis = \"sto-3g\"\n{}",
                                                           table))
                  .has_value());
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

class PipelinePipekMezey : public ::testing::TestWithParam<std::string_view> {};

TEST_P(PipelinePipekMezey, LocalizesTheScfOrbitalsWithoutChangingTheDensity) {
  FBF_REQUIRE_GPU();
  auto calc = localize_water(GetParam());
  ASSERT_TRUE(calc.has_value());
  const auto &pm = calc->get<fb::PipekMezey>();
  const auto &scf = calc->get<fb::Scf>();
  const auto &overlap = calc->get<fb::Overlap>();
  ASSERT_TRUE(pm.has_value() && scf.has_value() && overlap.has_value());

  const std::size_t n = calc->get<fb::BasisSet>()->nao_sphe();
  const auto occ = static_cast<std::size_t>(pm->n_occ);
  ASSERT_EQ(n, 7U);
  ASSERT_EQ(occ, 5U);
  const fb::ScopedDeviceIndex on{scf->c.dev_idx()};
  const std::vector<double> c = download(pm->c, n * occ);
  const std::vector<double> s = download(overlap->s, n * n);
  const std::vector<double> d = download(scf->d, n * n);

  EXPECT_GT(pm->info.functional, pm->info.initial_functional);
  EXPECT_GE(pm->info.sweeps, 1);
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

INSTANTIATE_TEST_SUITE_P(Exponents, PipelinePipekMezey,
                         ::testing::Values("[pipek_mezey]\n", "[pipek_mezey]\nexponent = 4\n"),
                         [](const auto &info) {
                           return info.param.find("4") == std::string_view::npos ? "P2" : "P4";
                         });

} // namespace
