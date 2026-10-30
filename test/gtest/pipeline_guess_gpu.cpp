// The Guess step against a direct sad_density, and its load against the
// density it was saved from. The step's contract is checked by static_assert;
// running it needs a Molecule, which owns its nuclei on the device, so the
// rest needs a GPU. The guess itself is host work, run here on a subflow of a
// taskflow of its own, as the pipeline would.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.sad;
import free_breakfast.io.basis;
import free_breakfast.io.sad;
import free_breakfast.io.npy;
import free_breakfast.core.nontransferrable;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.guess;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

using GuessInput = fb::View<fb::MoleculeOutput, fb::BasisSetOutput>;

static_assert(fb::PipelineStep<fb::GuessStep>);
static_assert(fb::kRunShape<fb::GuessStep> == fb::RunShape::HostSubflow);
static_assert(std::same_as<fb::StepInput<fb::GuessStep>, GuessInput>);
static_assert(std::same_as<fb::StepOutput<fb::GuessStep>, fb::GuessOutput>);
static_assert(fb::LoadableStep<fb::GuessStep>);
static_assert(fb::kLoadShape<fb::GuessStep> == fb::RunShape::Host);
static_assert(std::same_as<fb::LoadInput<fb::GuessStep>, fb::View<fb::BasisSetOutput>>);

// O: s, s, p, d (the d unoccupied); H: s.
fb::BasisStamp water_stamp() {
  std::map<std::int32_t, std::vector<fb::ShellStamp>> table;
  table.emplace(1, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{1.2, 1.0}})});
  table.emplace(8, std::vector<fb::ShellStamp>{
                       fb::ShellStamp(0, {{90.0, 1.0}}), fb::ShellStamp(0, {{9.0, 1.0}}),
                       fb::ShellStamp(1, {{2.0, 1.0}}), fb::ShellStamp(2, {{0.8, 1.0}})});
  return fb::BasisStamp(std::move(table));
}

fb::MoleculeOutput water(fb::DeviceResources &res) {
  return {fb::Molecule(res, {Atom{{0.0, 0.0, 0.2}, Element::O}, Atom{{0.0, 1.4, -0.9}, Element::H},
                             Atom{{0.0, -1.4, -0.9}, Element::H}})};
}

// The stamp placed over the molecule, as the BasisSet step publishes it.
fb::Result<fb::BasisSetOutput> placed(const fb::BasisStamp &stamp,
                                      const fb::MoleculeOutput &molecule) {
  return fb::place(stamp, molecule.molecule).transform([](fb::BasisSet basis) {
    return fb::BasisSetOutput{std::move(basis)};
  });
}

// The step's run on a subflow, the shape it is scheduled in.
fb::Result<fb::GuessOutput> run_step(const fb::GuessStep &step, const GuessInput &in) {
  fb::tf::Executor executor(4);
  fb::tf::Taskflow flow;
  fb::Result<fb::GuessOutput> out = fb::err_config("the guess task did not run");
  flow.emplace([&](fb::tf::Subflow &subflow) { out = step.run(in, subflow); });
  executor.run(flow).wait();
  return out;
}

TEST(PipelineGuess, SadMatchesSadDensityBitwise) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  auto basis = placed(water_stamp(), molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;

  auto guess = run_step(fb::GuessStep{.type = fb::GuessType::Sad}, GuessInput{molecule, *basis});
  auto direct = fb::sad_density(basis->basis, molecule.molecule.atoms());
  ASSERT_TRUE(guess.has_value()) << guess.error().message;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  ASSERT_EQ(guess->d.size(), std::size_t{basis->basis.nao_sphe()} * basis->basis.nao_sphe());
  EXPECT_EQ(guess->d, *direct);
}

// A compiled-in set carries its name through place, and the step takes that
// set's baked stamps: exactly the guess sad_density gives with them.
// test/io/sad_host.cpp holds each stamp to the SCF it stands in for.
TEST(PipelineGuess, ACompiledInBasisTakesTheBakedStamps) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  auto stamp = fb::resolve_basis("cc-pvdz");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  auto basis = placed(stamp->primitive_normalized(), molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  EXPECT_EQ(basis->basis.name(), "cc-pVDZ");

  auto guess = run_step(fb::GuessStep{}, GuessInput{molecule, *basis});
  auto baked = fb::sad_density(basis->basis, molecule.molecule.atoms(), fb::builtin_sad_density);
  ASSERT_TRUE(guess.has_value()) << guess.error().message;
  ASSERT_TRUE(baked.has_value()) << baked.error().message;
  EXPECT_EQ(guess->d, *baked);
}

TEST(PipelineGuess, ABasisPlacedOverOtherAtomsIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  const fb::MoleculeOutput hydrogen{fb::Molecule(res, {Atom{{0.0, 0.0, 0.0}, Element::H}})};
  auto basis = placed(water_stamp(), hydrogen);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;

  auto guess = run_step(fb::GuessStep{}, GuessInput{molecule, *basis});
  ASSERT_FALSE(guess.has_value());
  EXPECT_EQ(guess.error().kind, fb::ErrorKind::InvalidConfig);
}

// --- load -------------------------------------------------------------------

TEST(PipelineGuess, ALoadedGuessIsTheSavedDensityBitwise) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  auto basis = placed(water_stamp(), molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  auto guess = run_step(fb::GuessStep{}, GuessInput{molecule, *basis});
  ASSERT_TRUE(guess.has_value()) << guess.error().message;

  const fb::test::ScratchDir dir("pipeline-guess");
  const std::size_t n = basis->basis.nao_sphe();
  ASSERT_TRUE(fb::write_npy(dir / "d.npy", fb::NpyArray{.shape = {n, n}, .data = guess->d}));
  auto loaded = fb::GuessStep::load(dir / "d.npy", fb::View<fb::BasisSetOutput>{*basis});
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  EXPECT_EQ(loaded->d, guess->d);
}

TEST(PipelineGuess, ALoadOfTheWrongShapeIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  auto basis = placed(water_stamp(), molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;
  const std::size_t n = basis->basis.nao_sphe();

  const fb::test::ScratchDir dir("pipeline-guess");
  // A basis one function smaller, and the right count as a vector.
  for (const std::vector<std::size_t> &shape :
       {std::vector<std::size_t>{n - 1, n - 1}, std::vector<std::size_t>{n * n}}) {
    const fb::NpyArray array{.shape = shape,
                             .data = std::vector<double>(*fb::element_count(shape), 0.0)};
    ASSERT_TRUE(fb::write_npy(dir / "d.npy", array));
    auto loaded = fb::GuessStep::load(dir / "d.npy", fb::View<fb::BasisSetOutput>{*basis});
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(loaded.error().message.find("[guess]"), std::string::npos) << loaded.error().message;
  }
}

TEST(PipelineGuess, ALoadOfAMissingFileIsIo) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  auto basis = placed(water_stamp(), molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;

  const fb::test::ScratchDir dir("pipeline-guess");
  auto loaded = fb::GuessStep::load(dir / "absent.npy", fb::View<fb::BasisSetOutput>{*basis});
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().kind, fb::ErrorKind::IO);
}

} // namespace
