// The Guess step's load: a density read back from .npy against the basis it
// is loaded over. A load reads only the BasisSetOutput, and a BasisSet built
// from its shells needs no Molecule, so nothing touches a device. The guess
// itself, and a load of the density it saved, are pipeline_guess_gpu.cpp's.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.io.npy;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.guess;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// s, s, p, d on one centre, s on another: nao_sphe = 11.
fb::BasisSetOutput water_like_basis() {
  const fb::Vec<double, 3> o{0.0, 0.0, 0.2};
  const fb::Vec<double, 3> h{0.0, 1.4, -0.9};
  return {fb::BasisSet({fb::Shell(0, {{90.0, 1.0}}, o), fb::Shell(0, {{9.0, 1.0}}, o),
                        fb::Shell(1, {{2.0, 1.0}}, o), fb::Shell(2, {{0.8, 1.0}}, o),
                        fb::Shell(0, {{1.2, 1.0}}, h)})};
}

TEST(PipelineGuessLoad, ALoadIsTheFileBitwise) {
  const fb::BasisSetOutput basis = water_like_basis();
  const std::size_t n = basis.basis.nao_sphe();
  ASSERT_EQ(n, 11U);
  // Not symmetric, so a transposed read would not pass for the original.
  std::vector<double> d(n * n);
  for (std::size_t i = 0; i < d.size(); ++i) {
    d[i] = 1.0 / static_cast<double>(i + 3);
  }

  const fb::test::ScratchDir dir("pipeline-guess-load");
  ASSERT_TRUE(fb::write_npy(dir / "d.npy", fb::NpyArray{.shape = {n, n}, .data = d}));
  auto loaded = fb::GuessStep::load(dir / "d.npy", fb::View<fb::BasisSetOutput>{basis});
  ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
  EXPECT_EQ(loaded->d, d);
}

TEST(PipelineGuessLoad, ALoadOfTheWrongShapeIsInvalidConfig) {
  const fb::BasisSetOutput basis = water_like_basis();
  const std::size_t n = basis.basis.nao_sphe();

  const fb::test::ScratchDir dir("pipeline-guess-load");
  // A basis one function smaller, and the right count as a vector.
  for (const std::vector<std::size_t> &shape :
       {std::vector<std::size_t>{n - 1, n - 1}, std::vector<std::size_t>{n * n}}) {
    const fb::NpyArray array{.shape = shape,
                             .data = std::vector<double>(*fb::element_count(shape), 0.0)};
    ASSERT_TRUE(fb::write_npy(dir / "d.npy", array));
    auto loaded = fb::GuessStep::load(dir / "d.npy", fb::View<fb::BasisSetOutput>{basis});
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().kind, fb::ErrorKind::InvalidConfig);
    EXPECT_NE(loaded.error().message.find("[guess]"), std::string::npos) << loaded.error().message;
  }
}

TEST(PipelineGuessLoad, ALoadOfAMissingFileIsIo) {
  const fb::BasisSetOutput basis = water_like_basis();

  const fb::test::ScratchDir dir("pipeline-guess-load");
  auto loaded = fb::GuessStep::load(dir / "absent.npy", fb::View<fb::BasisSetOutput>{basis});
  ASSERT_FALSE(loaded.has_value());
  EXPECT_EQ(loaded.error().kind, fb::ErrorKind::IO);
}

} // namespace
