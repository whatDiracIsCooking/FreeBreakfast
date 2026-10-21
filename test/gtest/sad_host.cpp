// Host-only cases for the SAD initial guess: each atom's block against PySCF's
// AtomSphAverageRHF density (data/sad/atomic_scf.json, written by
// tools/emit_sad_reference.py), the block-diagonal assembly, and the refusals.
// The guess over a basis placed on a Molecule is sad_gpu.cpp's.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.io.json;
import free_breakfast.io.basis;
import free_breakfast.sad;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

const fb::Vec<double, 3> kOrigin{0.0, 0.0, 0.0};

// Read once per binary: every case that wants a basis shares it.
const fb::json &reference() {
  static const fb::json doc = [] {
    auto read = fb::read_json(std::filesystem::path{FBF_TEST_DATA_DIR} / "sad" / "atomic_scf.json");
    if (!read) {
      ADD_FAILURE() << read.error().message;
      return fb::json{};
    }
    return *read;
  }();
  return doc;
}

// One element's shells from the reference file's BSE basis, at the origin, the
// way a production run gets them: parse_basis, primitive_normalized, at.
std::vector<fb::Shell> reference_shells(const std::string &basis, std::int32_t z) {
  auto stamp = fb::parse_basis(reference()["bases"][basis].dump());
  EXPECT_TRUE(stamp) << basis;
  if (!stamp) {
    return {};
  }
  const fb::BasisStamp normalized = stamp->primitive_normalized();
  const std::vector<fb::ShellStamp> *element = normalized.shells_for(z);
  EXPECT_NE(element, nullptr) << basis << " Z=" << z;
  std::vector<fb::Shell> shells;
  if (element != nullptr) {
    for (const fb::ShellStamp &s : *element) {
      shells.push_back(s.at(kOrigin));
    }
  }
  return shells;
}

// The guess for one atom alone, its shells grouped as place would.
fb::Result<std::vector<double>> atom_density(Element element, std::vector<fb::Shell> shells) {
  const auto count = static_cast<std::uint32_t>(shells.size());
  const fb::BasisSet basis(std::move(shells), {0, count});
  const std::vector<Atom> atoms{Atom{kOrigin, element}};
  return fb::sad_density(basis, atoms);
}

// <a|b> for one m of two same-l shells on one center, in this tree's
// normalization: coefficients multiply unnormalized primitives, and a spherical
// function has the norm of the shell's x^l primitive,
// Integral x^2l exp(-g r^2) = (2l-1)!! pi^(3/2) / ((2g)^l g^(3/2)).
double same_l_overlap(const fb::Shell &a, const fb::Shell &b) {
  const std::int32_t l = a.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  double sum = 0.0;
  for (const fb::Primitive &p : a.primitives()) {
    for (const fb::Primitive &q : b.primitives()) {
      const double g = p.exponent + q.exponent;
      sum += p.coefficient * q.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
             std::pow(2.0 * g, l);
    }
  }
  return sum;
}

// --- Against PySCF ------------------------------------------------------------

// PySCF's per-l density blocks are over unit-normalized AOs; a function here
// is N_i = sqrt(<i|i>) times PySCF's, so its density element is the reference
// one over N_i N_j. Every element is compared, so the zeros between different
// (l, m) are checked too.
TEST(Sad, AtomBlockMatchesPyscfAtomSphAverageRhf) {
  ASSERT_FALSE(reference()["atoms"].empty());
  for (const auto &ref : reference()["atoms"]) {
    const auto basis_name = ref["basis"].get<std::string>();
    const auto z = ref["z"].get<std::int32_t>();
    SCOPED_TRACE(ref["symbol"].get<std::string>() + "/" + basis_name);
    const auto blocks = ref["density_blocks"].get<std::vector<std::vector<std::vector<double>>>>();
    const std::vector<fb::Shell> shells = reference_shells(basis_name, z);

    // Per l, each shell of that momentum's first AO and norm, in shell order.
    std::vector<std::vector<std::size_t>> first(blocks.size());
    std::vector<std::vector<double>> norm(blocks.size());
    std::size_t n = 0;
    for (const fb::Shell &shell : shells) {
      const auto l = static_cast<std::size_t>(shell.angular_momentum());
      ASSERT_LT(l, blocks.size());
      first[l].push_back(n);
      norm[l].push_back(std::sqrt(same_l_overlap(shell, shell)));
      n += shell.function_count_sphe();
    }
    std::vector<double> expected(n * n, 0.0);
    for (std::size_t l = 0; l < blocks.size(); ++l) {
      ASSERT_EQ(blocks[l].size(), first[l].size()) << "l=" << l;
      for (std::size_t j = 0; j < first[l].size(); ++j) {
        for (std::size_t i = 0; i < first[l].size(); ++i) {
          for (std::size_t m = 0; m < (2 * l) + 1; ++m) {
            expected[(first[l][i] + m) + (n * (first[l][j] + m))] =
                blocks[l][i][j] / (norm[l][i] * norm[l][j]);
          }
        }
      }
    }

    const auto density = atom_density(static_cast<Element>(z), shells);
    ASSERT_TRUE(density) << density.error().message;
    ASSERT_EQ(density->size(), n * n);
    for (std::size_t i = 0; i < n * n; ++i) {
      EXPECT_NEAR((*density)[i], expected[i], 1e-7) << i % n << ", " << i / n;
    }
  }
}

// --- The assembly ---------------------------------------------------------------

TEST(Sad, IsBlockDiagonalOverTheAtoms) {
  // H2O laid out as place would: O's shells, then each H's, grouped per atom.
  const std::vector<Atom> atoms{
      Atom{kOrigin, Element::O},
      Atom{fb::Vec<double, 3>{1.43, 0.21, -1.07}, Element::H},
      Atom{fb::Vec<double, 3>{-1.38, 0.35, -1.12}, Element::H},
  };
  const std::vector<fb::Shell> o = reference_shells("sto-3g", 8);
  const std::vector<fb::Shell> h = reference_shells("sto-3g", 1);
  std::vector<fb::Shell> shells;
  std::vector<std::uint32_t> offsets{0};
  for (const Atom &atom : atoms) {
    for (const fb::Shell &s : (atom.element == Element::O) ? o : h) {
      shells.emplace_back(s.angular_momentum(), s.primitives(), atom.position);
    }
    offsets.push_back(static_cast<std::uint32_t>(shells.size()));
  }
  const fb::BasisSet basis(shells, offsets);

  const auto density = fb::sad_density(basis, atoms);
  ASSERT_TRUE(density) << density.error().message;
  const std::size_t n = basis.nao_sphe();
  ASSERT_EQ(n, 7U);
  const auto o_alone = atom_density(Element::O, o);
  const auto h_alone = atom_density(Element::H, h);
  ASSERT_TRUE(o_alone && h_alone);
  // O is the leading 5 x 5 block, each H one diagonal entry; nothing between.
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      double expected = 0.0;
      if (i < 5 && j < 5) {
        expected = (*o_alone)[i + (5 * j)];
      } else if (i == j) {
        expected = (*h_alone)[0];
      }
      EXPECT_EQ((*density)[i + (n * j)], expected) << i << ", " << j;
    }
  }
}

TEST(Sad, SeatsSubshellsByMomentumNotShellPosition) {
  // O's STO-3G shells are s, s, p; put the p first. The k-th subshell of
  // momentum l still goes on the l block's k-th orbital, so the density is the
  // same one with its rows and columns moved: p (3 AOs) then the two s.
  const std::vector<fb::Shell> o = reference_shells("sto-3g", 8);
  ASSERT_EQ(o.size(), 3U);
  const auto in_order = atom_density(Element::O, o);
  const auto p_first = atom_density(Element::O, {o[2], o[0], o[1]});
  ASSERT_TRUE(in_order && p_first);
  const std::array<std::size_t, 5> moved{3, 4, 0, 1, 2}; // in_order AO -> p_first AO
  for (std::size_t j = 0; j < 5; ++j) {
    for (std::size_t i = 0; i < 5; ++i) {
      EXPECT_NEAR((*p_first)[moved[i] + (5 * moved[j])], (*in_order)[i + (5 * j)], 1e-10)
          << i << ", " << j;
    }
  }
}

TEST(Sad, GhostAtomIsEmpty) {
  const auto density = atom_density(Element::Sentinel, reference_shells("sto-3g", 1));
  ASSERT_TRUE(density) << density.error().message;
  EXPECT_EQ(*density, (std::vector<double>{0.0}));
}

// --- Refusals -------------------------------------------------------------------

TEST(Sad, RefusesABasisNotGroupedOverTheAtoms) {
  const std::vector<Atom> one{Atom{kOrigin, Element::H}};
  const fb::BasisSet ungrouped(reference_shells("sto-3g", 1));
  EXPECT_EQ(fb::sad_density(ungrouped, one).error().kind, fb::ErrorKind::InvalidConfig);
  const std::vector<Atom> two{Atom{kOrigin, Element::H}, Atom{kOrigin, Element::H}};
  const fb::BasisSet grouped(reference_shells("sto-3g", 1), {0, 1});
  EXPECT_EQ(fb::sad_density(grouped, two).error().kind, fb::ErrorKind::InvalidConfig);
}

TEST(Sad, RefusesABasisThatCannotSeatTheAtom) {
  // O needs a p orbital; two s shells have none.
  const auto density = atom_density(
      Element::O, {fb::Shell(0, {{10.0, 1.0}}, kOrigin), fb::Shell(0, {{1.0, 1.0}}, kOrigin)});
  ASSERT_FALSE(density);
  EXPECT_EQ(density.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
