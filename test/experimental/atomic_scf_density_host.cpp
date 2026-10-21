// Host cases for experimental/atomic_scf's :density: the block-diagonal
// molecular guess assembled from atomic SCFs, and its refusal.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.io.basis;
import free_breakfast.experimental.atomic_scf;

#include "atomic_scf_reference.h"

namespace {

namespace fb = free_breakfast;
using fb::test::kOrigin;
using fb::test::reference_shells;

TEST(AtomicScfDensity, IsBlockDiagonalOverTheAtoms) {
  // H2O laid out as place would: O's shells, then each H's, grouped per atom.
  const std::vector<fb::Atom> atoms{
      fb::Atom{kOrigin, fb::Element::O},
      fb::Atom{fb::Vec<double, 3>{1.43, 0.21, -1.07}, fb::Element::H},
      fb::Atom{fb::Vec<double, 3>{-1.38, 0.35, -1.12}, fb::Element::H},
  };
  const std::vector<fb::Shell> o = reference_shells("sto-3g", 8);
  const std::vector<fb::Shell> h = reference_shells("sto-3g", 1);
  std::vector<fb::Shell> shells;
  std::vector<std::uint32_t> offsets{0};
  for (const fb::Atom &atom : atoms) {
    for (const fb::Shell &s : (atom.element == fb::Element::O) ? o : h) {
      shells.emplace_back(s.angular_momentum(), s.primitives(), atom.position);
    }
    offsets.push_back(static_cast<std::uint32_t>(shells.size()));
  }
  const fb::BasisSet basis(shells, offsets);

  const auto density = fb::atomic_scf_density(basis, atoms);
  ASSERT_TRUE(density) << density.error().message;
  const std::size_t n = basis.nao_sphe();
  ASSERT_EQ(n, 7U);
  const auto o_scf = fb::atomic_scf(8, o);
  const auto h_scf = fb::atomic_scf(1, h);
  ASSERT_TRUE(o_scf && h_scf);
  // O is the leading 5 x 5 block, each H one diagonal entry; nothing between.
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      double expected = 0.0;
      if (i < 5 && j < 5) {
        expected = o_scf->density[i + (5 * j)];
      } else if (i == j) {
        expected = h_scf->density[0];
      }
      EXPECT_EQ((*density)[i + (n * j)], expected) << i << ", " << j;
    }
  }
}

TEST(AtomicScfDensity, RefusesAMismatchedAtomCount) {
  const std::vector<fb::Atom> one{fb::Atom{kOrigin, fb::Element::H}};
  const fb::BasisSet ungrouped(reference_shells("sto-3g", 1));
  EXPECT_EQ(fb::atomic_scf_density(ungrouped, one).error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
