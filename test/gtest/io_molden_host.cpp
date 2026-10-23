// Host-only cases for io/molden: the writer's sections, its per-shell
// reordering into Molden's m = 0, +1, -1, ... order, and the two
// normalizations it undoes.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.nontransferrable;
import free_breakfast.atom;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.molden;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

const fb::Vec<double, 3> kOxygen{0.0, 0.0, 0.25};
const fb::Vec<double, 3> kHydrogen{0.0, 1.5, -0.75};

// A shell as the pipeline makes one: its basis-file coefficients with the
// primitive normalization folded in, placed at a center.
fb::Shell placed(int32_t l, std::vector<fb::Primitive> raw, const fb::Vec<double, 3> &center) {
  return fb::place(fb::ShellStamp(l, std::move(raw)).primitive_normalized(), center);
}

// O carrying a contracted s, a p and a d; H a single s. AO layout
// (spherical): O s 0, O p 1..3, O d 4..8, H s 9.
struct Fixture {
  std::vector<fb::Atom> atoms{{kOxygen, fb::Element::O}, {kHydrogen, fb::Element::H}};
  fb::BasisSet basis{{placed(0, {{130.7, 0.300}, {23.81, 0.535}, {6.444, 0.445}}, kOxygen),
                      placed(1, {{5.033, 1.0}}, kOxygen), placed(2, {{1.2, 1.0}}, kOxygen),
                      placed(0, {{0.5, 1.0}}, kHydrogen)},
                     {0, 3, 4}};
  static constexpr size_t kNao = 10;
};

// The [MO] section read back as a column-major nao x nmo matrix, in Molden's
// row order.
std::vector<double> mo_block(std::string_view text, size_t nao) {
  std::vector<double> c;
  std::istringstream in{std::string(text.substr(text.find("[MO]")))};
  std::string line;
  while (std::getline(in, line)) {
    int32_t index = 0;
    double value = 0.0;
    std::istringstream fields(line);
    if (fields >> index >> value) {
      c.push_back(value);
    }
  }
  EXPECT_EQ(c.size() % nao, 0U);
  return c;
}

// The contraction norm of an s shell over normalized primitives, written out
// for two primitives rather than through the writer's general-l sum.
double two_primitive_s_norm(double a, double ca, double b, double cb) {
  const double overlap = std::pow(2.0 * std::sqrt(a * b) / (a + b), 1.5);
  return std::sqrt((ca * ca) + (cb * cb) + (2.0 * ca * cb * overlap));
}

TEST(IoMolden, WritesTheSectionsInOrder) {
  const Fixture f;
  const std::vector<double> c(Fixture::kNao * 2, 0.0);
  const auto text = fb::molden_to_string(f.atoms, f.basis, c, std::vector<double>{-20.5, 0.4},
                                         std::vector<double>{2.0, 0.0});
  ASSERT_TRUE(text) << text.error().message;
  size_t at = 0;
  for (std::string_view section :
       {"[Molden Format]", "[Atoms] AU", "[GTO]", "[5D7F]", "[9G]", "[MO]"}) {
    const size_t found = text->find(section, at);
    ASSERT_NE(found, std::string::npos) << section;
    at = found;
  }
  EXPECT_NE(text->find("O      1   8"), std::string::npos);
  EXPECT_NE(text->find("H      2   1"), std::string::npos);
  EXPECT_NE(text->find(" s 3 1.00\n"), std::string::npos);
  EXPECT_NE(text->find(" d 1 1.00\n"), std::string::npos);
  EXPECT_NE(text->find("Ene= -2.050000000000000e+01"), std::string::npos);
  EXPECT_NE(text->find("Occup= 2.000000"), std::string::npos);
}

TEST(IoMolden, GtoCarriesTheBasisFileCoefficients) {
  const Fixture f;
  const std::vector<double> c(Fixture::kNao, 0.0);
  const auto text =
      fb::molden_to_string(f.atoms, f.basis, c, std::vector<double>{0.0}, std::vector<double>{0.0});
  ASSERT_TRUE(text) << text.error().message;
  // The folded N(alpha, l) is divided back out, so the file's own numbers
  // come back.
  EXPECT_NE(text->find(" 1.307000000000000e+02  3.000000000000000e-01"), std::string::npos);
  EXPECT_NE(text->find(" 2.381000000000000e+01  5.350000000000000e-01"), std::string::npos);
  EXPECT_NE(text->find(" 1.200000000000000e+00  1.000000000000000e+00"), std::string::npos);
}

TEST(IoMolden, ReordersEachShellToMoldenOrder) {
  const Fixture f;
  // C = I: MO j is AO j, so Molden row i of MO j is nonzero exactly where row
  // i reads AO j.
  std::vector<double> c(Fixture::kNao * Fixture::kNao, 0.0);
  for (size_t i = 0; i < Fixture::kNao; ++i) {
    c[i + (i * Fixture::kNao)] = 1.0;
  }
  const std::vector<double> energies(Fixture::kNao, 0.0);
  const auto text = fb::molden_to_string(f.atoms, f.basis, c, energies, energies);
  ASSERT_TRUE(text) << text.error().message;
  const std::vector<double> m = mo_block(*text, Fixture::kNao);
  ASSERT_EQ(m.size(), Fixture::kNao * Fixture::kNao);

  // s, p (x, y, z) and the H s stay put; the d block (AOs 4..8, m = -2..+2)
  // is written m = 0, +1, -1, +2, -2.
  const std::array<size_t, Fixture::kNao> reads{0, 1, 2, 3, 6, 7, 5, 8, 4, 9};
  for (size_t i = 0; i < Fixture::kNao; ++i) {
    for (size_t j = 0; j < Fixture::kNao; ++j) {
      const double value = m[i + (j * Fixture::kNao)];
      if (j == reads[i]) {
        EXPECT_NE(value, 0.0) << "row " << i;
      } else {
        EXPECT_EQ(value, 0.0) << "row " << i << ", MO " << j;
      }
    }
  }
}

TEST(IoMolden, ScalesEachRowByItsContractionNorm) {
  const Fixture f;
  std::vector<double> c(Fixture::kNao * Fixture::kNao, 0.0);
  for (size_t i = 0; i < Fixture::kNao; ++i) {
    c[i + (i * Fixture::kNao)] = 1.0;
  }
  const std::vector<double> energies(Fixture::kNao, 0.0);
  const auto text = fb::molden_to_string(f.atoms, f.basis, c, energies, energies);
  ASSERT_TRUE(text) << text.error().message;
  const std::vector<double> m = mo_block(*text, Fixture::kNao);
  ASSERT_EQ(m.size(), Fixture::kNao * Fixture::kNao);

  // A single primitive with coefficient 1 is already a unit AO, at any l: the
  // writer's two normalizations cancel exactly.
  for (size_t ao : {1, 2, 3, 9}) {
    EXPECT_NEAR(m[ao + (ao * Fixture::kNao)], 1.0, 1e-14) << ao;
  }
  EXPECT_NEAR(m[4 + (6 * Fixture::kNao)], 1.0, 1e-14);

  // The three-primitive O s: its first two primitives' norm, plus the third.
  const double a = 130.7;
  const double b = 23.81;
  const double g = 6.444;
  const auto pair = [](double x, double y) {
    return std::pow(2.0 * std::sqrt(x * y) / (x + y), 1.5);
  };
  const double ab = two_primitive_s_norm(a, 0.300, b, 0.535);
  const double norm2 = (ab * ab) + (0.445 * 0.445) + (2.0 * 0.300 * 0.445 * pair(a, g)) +
                       (2.0 * 0.535 * 0.445 * pair(b, g));
  EXPECT_NEAR(m[0], std::sqrt(norm2), 1e-14);
  EXPECT_GT(std::abs(m[0] - 1.0), 1e-3) << "the fixture's contraction should not be unit";
}

TEST(IoMolden, RejectsSizesThatDisagree) {
  const Fixture f;
  const std::vector<double> c(Fixture::kNao * 2, 0.0);
  const auto short_occ = fb::molden_to_string(f.atoms, f.basis, c, std::vector<double>{0.0, 0.0},
                                              std::vector<double>{2.0});
  ASSERT_FALSE(short_occ);
  EXPECT_EQ(short_occ.error().kind, fb::ErrorKind::InvalidConfig);
  const auto short_c = fb::molden_to_string(f.atoms, f.basis, std::span(c).first(5),
                                            std::vector<double>{0.0}, std::vector<double>{2.0});
  ASSERT_FALSE(short_c);
  EXPECT_EQ(short_c.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(IoMolden, RejectsABasisWithoutTheAtomsGrouping) {
  const Fixture f;
  const fb::BasisSet bare({placed(0, {{1.0, 1.0}}, kOxygen)});
  const std::vector<double> c(1, 1.0);
  const auto bare_text =
      fb::molden_to_string(f.atoms, bare, c, std::vector<double>{0.0}, std::vector<double>{2.0});
  ASSERT_FALSE(bare_text);
  EXPECT_EQ(bare_text.error().kind, fb::ErrorKind::InvalidConfig);
  const auto one_atom =
      fb::molden_to_string(std::span(f.atoms).first(1), f.basis, std::vector<double>(10, 0.0),
                           std::vector<double>{0.0}, std::vector<double>{2.0});
  ASSERT_FALSE(one_atom);
  EXPECT_EQ(one_atom.error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(IoMolden, RefusesAShellAboveG) {
  const std::vector<fb::Atom> atoms{{kOxygen, fb::Element::O}};
  const fb::BasisSet basis({placed(5, {{1.0, 1.0}}, kOxygen)}, {0, 1});
  const std::vector<double> c(11, 0.0);
  const auto text =
      fb::molden_to_string(atoms, basis, c, std::vector<double>{0.0}, std::vector<double>{0.0});
  ASSERT_FALSE(text);
  EXPECT_EQ(text.error().kind, fb::ErrorKind::Unsupported);
}

TEST(IoMolden, WriteMoldenWritesTheSameText) {
  const Fixture f;
  const std::vector<double> c(Fixture::kNao, 0.5);
  const fb::test::ScratchDir dir("io-molden");
  ASSERT_TRUE(fb::write_molden(dir / "h2o.molden", f.atoms, f.basis, c, std::vector<double>{-1.0},
                               std::vector<double>{2.0}));
  const auto expected = fb::molden_to_string(f.atoms, f.basis, c, std::vector<double>{-1.0},
                                             std::vector<double>{2.0});
  const auto written = fb::read_file(dir / "h2o.molden");
  ASSERT_TRUE(expected);
  ASSERT_TRUE(written);
  EXPECT_EQ(*written, *expected);
}

} // namespace
