// The BasisSet step against a direct read_basis + primitive_normalized + place.
// The step's contract is test/pipeline/steps_host.cpp's; running it needs a
// Molecule, which owns its nuclei on the device, so the rest needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.basis_set;
import free_breakfast.core.nontransferrable;

#include "io_basis_json.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;
using fb::test::kBasis;

fb::MoleculeOutput water(fb::DeviceResources &res) {
  return {fb::Molecule(res, {Atom{{0.0, 0.0, 0.2217}, Element::O},
                             Atom{{0.0, 1.4309, -0.8867}, Element::H},
                             Atom{{0.0, -1.4309, -0.8867}, Element::H}})};
}

TEST(PipelineBasisSet, MatchesReadBasisThenPlace) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::test::ScratchDir dir("pipeline_basis_set");
  const auto path = dir / "basis.json";
  ASSERT_TRUE(fb::write_file(path, kBasis).has_value());
  const fb::MoleculeOutput molecule = water(res);

  const fb::BasisSetStep step{.basis = path.string()};
  auto basis = step.run(fb::View<fb::MoleculeOutput>{molecule});
  auto stamp = fb::read_basis(path);
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  auto direct = fb::place(stamp->primitive_normalized(), molecule.molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;

  EXPECT_EQ(basis->basis.nao_cart(), direct->nao_cart());
  EXPECT_EQ(basis->basis.nao_sphe(), direct->nao_sphe());
  ASSERT_EQ(basis->basis.natoms(), direct->natoms());
  for (std::uint32_t a = 0; a < direct->natoms(); ++a) {
    EXPECT_EQ(basis->basis.atom_shell_count(a), direct->atom_shell_count(a)) << a;
  }
  ASSERT_EQ(basis->basis.nshells(), direct->nshells());
  for (std::uint32_t s = 0; s < direct->nshells(); ++s) {
    const fb::Shell &got = basis->basis.shells()[s];
    const fb::Shell &want = direct->shells()[s];
    EXPECT_EQ(got.angular_momentum(), want.angular_momentum()) << s;
    EXPECT_EQ(got.center().x, want.center().x) << s;
    EXPECT_EQ(got.center().y, want.center().y) << s;
    EXPECT_EQ(got.center().z, want.center().z) << s;
    ASSERT_EQ(got.nprimitives(), want.nprimitives()) << s;
    for (std::uint32_t p = 0; p < want.nprimitives(); ++p) {
      EXPECT_EQ(got.primitives()[p].exponent, want.primitives()[p].exponent) << s << ' ' << p;
      EXPECT_EQ(got.primitives()[p].coefficient, want.primitives()[p].coefficient) << s << ' ' << p;
    }
  }
}

TEST(PipelineBasisSet, MissingFileIsIoError) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::test::ScratchDir dir("pipeline_basis_set");
  const fb::MoleculeOutput molecule = water(res);

  const fb::BasisSetStep step{.basis = (dir / "absent.json").string()};
  auto basis = step.run(fb::View<fb::MoleculeOutput>{molecule});
  ASSERT_FALSE(basis.has_value());
  EXPECT_EQ(basis.error().kind, fb::ErrorKind::IO);
}

TEST(PipelineBasisSet, UncoveredElementIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::test::ScratchDir dir("pipeline_basis_set");
  const auto path = dir / "basis.json";
  ASSERT_TRUE(fb::write_file(path, kBasis).has_value());
  const fb::MoleculeOutput co{
      fb::Molecule(res, {Atom{{0.0, 0.0, 0.0}, Element::C}, Atom{{0.0, 0.0, 2.1}, Element::O}})};

  const fb::BasisSetStep step{.basis = path.string()};
  auto basis = step.run(fb::View<fb::MoleculeOutput>{co});
  ASSERT_FALSE(basis.has_value());
  EXPECT_EQ(basis.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
