// The Guess step against a direct sad_density. The step's contract is checked
// by static_assert; running it needs a Molecule, which owns its nuclei on the
// device, so the rest needs a GPU. The guess itself is host work, run here on
// a subflow of a taskflow of its own, as the pipeline would.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.sad;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.guess;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

using GuessInput = fb::View<fb::Molecule, fb::BasisSet>;

static_assert(fb::PipelineStep<fb::GuessStep>);
static_assert(fb::kRunShape<fb::GuessStep> == fb::RunShape::HostSubflow);
static_assert(std::same_as<fb::StepInput<fb::GuessStep>, GuessInput>);
static_assert(std::same_as<fb::StepOutput<fb::GuessStep>, fb::Guess>);

// O: s, s, p, d (the d unoccupied); H: s.
fb::BasisStamp water_stamp() {
  std::map<std::int32_t, std::vector<fb::ShellStamp>> table;
  table.emplace(1, std::vector<fb::ShellStamp>{fb::ShellStamp(0, {{1.2, 1.0}})});
  table.emplace(8, std::vector<fb::ShellStamp>{
                       fb::ShellStamp(0, {{90.0, 1.0}}), fb::ShellStamp(0, {{9.0, 1.0}}),
                       fb::ShellStamp(1, {{2.0, 1.0}}), fb::ShellStamp(2, {{0.8, 1.0}})});
  return fb::BasisStamp(std::move(table));
}

fb::Molecule water(fb::DeviceResources &res) {
  return fb::Molecule(res, {Atom{{0.0, 0.0, 0.2}, Element::O}, Atom{{0.0, 1.4, -0.9}, Element::H},
                            Atom{{0.0, -1.4, -0.9}, Element::H}});
}

// The step's run on a subflow, the shape it is scheduled in.
fb::Result<fb::Guess> run_step(const fb::GuessStep &step, const GuessInput &in) {
  fb::tf::Executor executor(4);
  fb::tf::Taskflow flow;
  fb::Result<fb::Guess> out = fb::err_config("the guess task did not run");
  flow.emplace([&](fb::tf::Subflow &subflow) { out = step.run(in, subflow); });
  executor.run(flow).wait();
  return out;
}

TEST(PipelineGuess, SadMatchesSadDensityBitwise) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule = water(res);
  auto basis = fb::place(water_stamp(), molecule);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;

  auto guess = run_step(fb::GuessStep{.type = fb::GuessType::Sad}, GuessInput{molecule, *basis});
  auto direct = fb::sad_density(*basis, molecule.atoms());
  ASSERT_TRUE(guess.has_value()) << guess.error().message;
  ASSERT_TRUE(direct.has_value()) << direct.error().message;
  ASSERT_EQ(guess->d.size(), std::size_t{basis->nao_sphe()} * basis->nao_sphe());
  EXPECT_EQ(guess->d, *direct);
}

TEST(PipelineGuess, ABasisPlacedOverOtherAtomsIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::Molecule molecule = water(res);
  const fb::Molecule hydrogen(res, {Atom{{0.0, 0.0, 0.0}, Element::H}});
  auto basis = fb::place(water_stamp(), hydrogen);
  ASSERT_TRUE(basis.has_value()) << basis.error().message;

  auto guess = run_step(fb::GuessStep{}, GuessInput{molecule, *basis});
  ASSERT_FALSE(guess.has_value());
  EXPECT_EQ(guess.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
