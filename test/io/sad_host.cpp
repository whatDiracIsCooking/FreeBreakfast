// The compiled-in SAD stamps (data/sad/, written by tools/sad and baked in by
// src/io/sad/CMakeLists.txt) against the SCF they stand in for. Each one is
// recomputed exactly as the generator computed it: the compiled-in set through
// resolve_basis, its primitives normalized as the BasisSet step normalizes
// them, the element's shells alone, and sad_density with no lookup.
//
// Recomputing them all takes hours of CPU (each 3d element of cc-pVTZ runs for
// minutes), so the fast tier recomputes H, C and O of every set, and the slow
// tier every element through argon, but cc-pVTZ and aug-cc-pVTZ only through
// neon: their third rows alone are most of a minute and two minutes, past the
// timeout under a sanitizer. The rest
// are held to their size only; tools/sad's run is what computed them.

#include <gtest/gtest.h>

#include "slow_test.h"

import std;
import free_breakfast.core;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.sad;
import free_breakfast.sad;
import free_breakfast.tools.sad;
import free_breakfast.io.base;
import free_breakfast.core.nontransferrable;

#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;

// The canonical names, as BasisSet::name holds them.
constexpr std::array<std::string_view, 11> kCompiledInSets{
    "STO-3G",  "6-31G",   "6-31G*",      "6-31G**",     "def2-SVP", "def2-TZVP",
    "cc-pVDZ", "cc-pVTZ", "aug-cc-pVDZ", "aug-cc-pVTZ", "MINAO"};

// Element @p z of @p normalized alone at the origin, grouped as one atom.
fb::BasisSet atom_basis(const fb::BasisStamp &normalized, std::int32_t z) {
  std::vector<fb::Shell> shells;
  for (const fb::ShellStamp &shell : *normalized.shells_for(z)) {
    shells.push_back(fb::place(shell, fb::Vec<double, 3>{0.0, 0.0, 0.0}));
  }
  const auto count = static_cast<std::uint32_t>(shells.size());
  return fb::BasisSet(std::move(shells), {0, count});
}

// Every baked stamp of @p name whose Z passes @p keep, against its SCF.
void expect_baked_matches_computed(std::string_view name,
                                   const std::function<bool(std::int32_t)> &keep) {
  auto stamp = fb::resolve_basis(name);
  ASSERT_TRUE(stamp) << name << ": " << stamp.error().message;
  const fb::BasisStamp normalized = stamp->primitive_normalized();
  for (const auto &[z, element] : normalized.element_shells()) {
    if (!keep(z)) {
      continue;
    }
    SCOPED_TRACE(std::format("{} Z={}", name, z));
    const fb::BasisSet basis = atom_basis(normalized, z);
    const std::vector<fb::Atom> atom{
        fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, static_cast<fb::Element>(z)}};
    const auto computed = fb::sad_density(basis, atom);
    ASSERT_TRUE(computed) << computed.error().message;
    const auto baked = fb::builtin_sad_density(name, z);
    ASSERT_TRUE(baked);
    EXPECT_EQ(*baked, *computed);
  }
}

TEST(IoSad, EveryElementOfEveryCompiledInSetIsBaked) {
  for (const std::string_view name : kCompiledInSets) {
    auto stamp = fb::resolve_basis(name);
    ASSERT_TRUE(stamp) << name << ": " << stamp.error().message;
    ASSERT_EQ(stamp->name(), name);
    const fb::BasisStamp normalized = stamp->primitive_normalized();
    for (const auto &[z, element] : stamp->element_shells()) {
      std::size_t n = 0;
      for (const fb::ShellStamp &shell : element) {
        n += shell.function_count_sphe();
      }
      const auto baked = fb::builtin_sad_density(name, z);
      if (!baked) {
        // The one exemption: an atom too large for the atomic SCF, which
        // tools/sad skips because the runtime refuses it the same way (the 3d
        // row of aug-cc-pVTZ). Anything else unbaked is a stale data/sad.
        const std::vector<fb::Atom> atom{
            fb::Atom{fb::Vec<double, 3>{0.0, 0.0, 0.0}, static_cast<fb::Element>(z)}};
        const auto computed = fb::sad_density(atom_basis(normalized, z), atom);
        ASSERT_FALSE(computed) << name << " Z=" << z << " is not baked; regenerate data/sad";
        EXPECT_EQ(computed.error().kind, fb::ErrorKind::Unsupported) << name << " Z=" << z;
        continue;
      }
      EXPECT_EQ(baked->size(), n * n) << name << " Z=" << z;
    }
  }
}

TEST(IoSad, StampSetsNameEachDocumentAsItIsCompiledIn) {
  // tools/sad's --all over the real data/basis: each document by its BSE name,
  // never its file stem, which resolve_basis does not always fold to the name
  // (6-31gstar.json is 6-31G*, and "6-31gstar" resolves to nothing; #1446).
  const auto sets = fb::sad_stamp_sets(std::filesystem::path{FBF_DATA_DIR} / "basis");
  ASSERT_TRUE(sets) << sets.error().message;
  // File-name order: 6-31g, 6-31gstar, 6-31gstarstar, aug-cc-pVDZ, ...
  const std::vector<std::string> expected{"6-31G",       "6-31G*",  "6-31G**", "aug-cc-pVDZ",
                                          "aug-cc-pVTZ", "cc-pVDZ", "cc-pVTZ", "def2-SVP",
                                          "def2-TZVP",   "MINAO",   "STO-3G"};
  EXPECT_EQ(*sets, expected);
  for (const std::string &name : *sets) {
    EXPECT_TRUE(std::ranges::contains(kCompiledInSets, name)) << name;
    const auto stamp = fb::resolve_basis(name);
    ASSERT_TRUE(stamp) << name << ": " << stamp.error().message;
    EXPECT_EQ(stamp->name(), name);
  }
  EXPECT_FALSE(fb::is_builtin_basis("6-31gstar"));
}

TEST(IoSad, StampSetsRefuseADocumentWithNoName) {
  const fb::test::ScratchDir dir("io-sad-sets");
  ASSERT_TRUE(fb::write_file(dir / "a.json", R"({"name": "A"})").has_value());
  ASSERT_TRUE(fb::write_file(dir / "b.json", R"({"elements": {}})").has_value());
  ASSERT_TRUE(fb::write_file(dir / "README.md", "not a set").has_value());
  const auto sets = fb::sad_stamp_sets(dir.path());
  ASSERT_FALSE(sets);
  EXPECT_EQ(sets.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_NE(sets.error().message.find("b.json"), std::string::npos) << sets.error().message;

  std::filesystem::remove(dir / "b.json");
  const auto one = fb::sad_stamp_sets(dir.path());
  ASSERT_TRUE(one) << one.error().message;
  EXPECT_EQ(*one, std::vector<std::string>{"A"});
}

TEST(IoSad, NothingElseIsBaked) {
  // The canonical name, not a spelling: only a basis resolved from the binary
  // carries a name, and it carries this one.
  EXPECT_FALSE(fb::builtin_sad_density("cc-pvdz", 8));
  EXPECT_FALSE(fb::builtin_sad_density("", 8));
  // cc-pVDZ has no potassium, and nothing has element 0.
  EXPECT_FALSE(fb::builtin_sad_density("cc-pVDZ", 19));
  EXPECT_FALSE(fb::builtin_sad_density("STO-3G", 0));
}

TEST(IoSad, BakedMatchesComputedForHCO) {
  // The fast set's one check of the stamps' values, the bitwise check a
  // data/basis or atomic-SCF change fails until data/sad is regenerated; the
  // cheaper two check only which stamps exist.
  FBF_FAMILY_REPRESENTATIVE();
  for (const std::string_view name : kCompiledInSets) {
    expect_baked_matches_computed(name, [](std::int32_t z) { return z == 1 || z == 6 || z == 8; });
  }
}

TEST(IoSad, BakedMatchesComputedThroughArgon) {
  FBF_SKIP_IF_FAST_TESTS();
  for (const std::string_view name : kCompiledInSets) {
    const std::int32_t last = name == "cc-pVTZ" || name == "aug-cc-pVTZ" ? 10 : 18;
    expect_baked_matches_computed(name, [last](std::int32_t z) { return z <= last; });
  }
}

} // namespace
