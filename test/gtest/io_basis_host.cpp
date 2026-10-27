// Host-only cases for io/basis: the BSE JSON reader. Its placement onto a
// molecule is io_basis_gpu.cpp's.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.io.base;
import free_breakfast.io.basis;
import free_breakfast.core.nontransferrable;

#include "io_basis_json.h"
#include "scratch_dir.h"

namespace {

namespace fb = free_breakfast;
using fb::test::kBasis;

TEST(IoBasis, ParsesElementsIntoShellStamps) {
  auto stamp = fb::parse_basis(kBasis);
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  EXPECT_EQ(stamp->element_shells().size(), 2U);

  const std::vector<fb::ShellStamp> *h = stamp->shells_for(1);
  ASSERT_NE(h, nullptr);
  ASSERT_EQ(h->size(), 1U);
  EXPECT_EQ((*h)[0].angular_momentum(), 0);
  ASSERT_EQ((*h)[0].nprimitives(), 3U);
  EXPECT_EQ((*h)[0].primitives()[0].exponent, 3.42525091);
  EXPECT_EQ((*h)[0].primitives()[2].coefficient, 0.44463454);
}

TEST(IoBasis, SplitsSpAndGeneralContractionBlocks) {
  auto stamp = fb::parse_basis(kBasis);
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  const std::vector<fb::ShellStamp> &o = *stamp->shells_for(8);
  // s, then sp -> s + p, then the general contraction -> s + s, then d.
  const std::array<std::int32_t, 6> momenta{0, 0, 1, 0, 0, 2};
  ASSERT_EQ(o.size(), momenta.size());
  for (std::size_t i = 0; i < o.size(); ++i) {
    EXPECT_EQ(o[i].angular_momentum(), momenta[i]) << "shell " << i;
  }
  // The sp block's p shell takes the second column over the shared exponents.
  EXPECT_EQ(o[2].primitives()[1].exponent, 1.1695961);
  EXPECT_EQ(o[2].primitives()[1].coefficient, 0.60768372);
  // Each general-contraction column is its own shell.
  EXPECT_EQ(o[3].primitives()[0].coefficient, 1.0);
  EXPECT_EQ(o[4].primitives()[0].coefficient, 0.0);
  EXPECT_EQ(o[4].primitives()[1].coefficient, 1.0);
}

TEST(IoBasis, ReadBasisFromFile) {
  const fb::test::ScratchDir dir("io-basis");
  ASSERT_TRUE(fb::write_file(dir / "basis.json", kBasis).has_value());
  auto stamp = fb::read_basis(dir / "basis.json");
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  EXPECT_EQ(stamp->element_shells().size(), 2U);

  auto missing = fb::read_basis(dir / "absent.json");
  ASSERT_FALSE(missing.has_value());
  EXPECT_EQ(missing.error().kind, fb::ErrorKind::IO);
}

TEST(IoBasis, MalformedJsonIsIoError) {
  auto stamp = fb::parse_basis(R"({"elements": )");
  ASSERT_FALSE(stamp.has_value());
  EXPECT_EQ(stamp.error().kind, fb::ErrorKind::IO);
}

TEST(IoBasis, WrongContentIsInvalidConfig) {
  const std::array<std::string_view, 10> bad{
      R"([])",
      R"({})",
      R"({"elements": []})",
      R"({"elements": {"H": {"electron_shells": []}}})",
      R"({"elements": {"1": {}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": ["1.0x"], "coefficients": [["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": [1.0], "coefficients": [["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": ["1.0", "2.0"], "coefficients": [["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0, 1, 2],
          "exponents": ["1.0"], "coefficients": [["1.0"], ["1.0"]]}]}}})",
      R"({"elements": {"1": {"electron_shells": [{"angular_momentum": [0],
          "exponents": ["1.0"]}]}}})",
  };
  for (const std::string_view text : bad) {
    auto stamp = fb::parse_basis(text);
    ASSERT_FALSE(stamp.has_value()) << "accepted: " << text;
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::InvalidConfig) << text;
  }
}

TEST(IoBasis, OnlySphericalFunctionTypesAreAccepted) {
  auto shell_with = [](std::string_view function_type, std::int32_t l) {
    const std::string tag =
        function_type.empty() ? "" : std::format(R"("function_type": "{}", )", function_type);
    return fb::parse_basis(std::format(
        R"({{"elements": {{"1": {{"electron_shells": [{{{}"angular_momentum": [{}],
            "exponents": ["1.0"], "coefficients": [["1.0"]]}}]}}}}}})",
        tag, l));
  };
  // An explicit spherical tag is accepted at any l.
  for (const std::int32_t l : {0, 2, 3}) {
    auto stamp = shell_with("gto_spherical", l);
    ASSERT_TRUE(stamp.has_value()) << "gto_spherical, l=" << l << ": " << stamp.error().message;
  }
  // Cartesian functions are not supported, at any l.
  for (const std::int32_t l : {0, 1, 2, 3}) {
    auto stamp = shell_with("gto_cartesian", l);
    ASSERT_FALSE(stamp.has_value()) << "gto_cartesian, l=" << l;
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::InvalidConfig);
  }
  // Untagged or bare "gto" is exact as spherical for s and p ...
  EXPECT_TRUE(shell_with("", 1).has_value());
  EXPECT_TRUE(shell_with("gto", 0).has_value());
  // ... and ambiguous from d up.
  for (const std::string_view ambiguous : {"", "gto"}) {
    auto stamp = shell_with(ambiguous, 2);
    ASSERT_FALSE(stamp.has_value()) << "'" << ambiguous << "'";
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::InvalidConfig);
  }
  // Slater-type orbitals are not Gaussian shells.
  EXPECT_FALSE(shell_with("sto", 0).has_value());
}

// The compiled-in sets (data/basis/, wrapped into the binary by
// src/io/basis/CMakeLists.txt) and resolution by name or path.

TEST(IoBasisBuiltins, EveryCompiledInSetParsesAndCoversItsElements) {
  // The element counts pin data/basis/: a set whose range silently changed, or
  // one re-downloaded with Cartesian shells the reader refuses, fails here
  // rather than at someone's first calculation.
  struct Expected {
    std::string_view name;
    std::size_t elements;
  };
  for (const auto &[name, elements] : std::array<Expected, 5>{
           {{"sto-3g", 36}, {"6-31g", 18}, {"cc-pVDZ", 35}, {"cc-pVTZ", 35}, {"minao", 35}}}) {
    auto stamp = fb::resolve_basis(name);
    ASSERT_TRUE(stamp.has_value()) << name << ": " << stamp.error().message;
    EXPECT_EQ(stamp->element_shells().size(), elements) << name;
  }

  // STO-3G runs to krypton; the correlation-consistent sets have no potassium
  // (BSE defines none), which data/basis/README.md records.
  auto sto = fb::resolve_basis("sto-3g");
  ASSERT_TRUE(sto.has_value()) << sto.error().message;
  EXPECT_NE(sto->shells_for(36), nullptr);
  auto dz = fb::resolve_basis("cc-pvdz");
  ASSERT_TRUE(dz.has_value()) << dz.error().message;
  EXPECT_EQ(dz->shells_for(19), nullptr);
  EXPECT_NE(dz->shells_for(20), nullptr);

  // MINAO is minimal: one function per occupied free-atom AO, so C is
  // 1s 2s 2p and Fe 1s-4s 2p 3p 3d. Ca has no 3d, its 4s being the last filled.
  auto minao = fb::resolve_basis("minao");
  ASSERT_TRUE(minao.has_value()) << minao.error().message;
  EXPECT_EQ(minao->name(), "MINAO");
  EXPECT_EQ(minao->shells_for(19), nullptr);
  for (const auto &[z, functions] : std::array<std::pair<std::int32_t, std::size_t>, 5>{
           {{1, 1}, {6, 5}, {20, 10}, {26, 15}, {36, 18}}}) {
    std::size_t n = 0;
    for (const fb::ShellStamp &shell : *minao->shells_for(z)) {
      n += shell.function_count_sphe();
    }
    EXPECT_EQ(n, functions) << "Z=" << z;
  }
}

TEST(IoBasisBuiltins, NamesIgnoreCaseHyphensAndUnderscores) {
  for (const std::string_view spelling : {"STO-3G", "sto-3g", "sto3g", "STO_3G", "s t o-3 g"}) {
    EXPECT_TRUE(fb::is_builtin_basis(spelling)) << spelling;
  }
  for (const std::string_view spelling : {"cc-pVTZ", "ccpvtz", "CC-PVTZ"}) {
    EXPECT_TRUE(fb::is_builtin_basis(spelling)) << spelling;
  }
  // Normalization folds separators, never the characters that distinguish one
  // set from another: 6-31G* is a different basis from 6-31G, and is not
  // compiled in (its d shells are Cartesian -- data/basis/README.md).
  EXPECT_TRUE(fb::is_builtin_basis("6-31g"));
  EXPECT_FALSE(fb::is_builtin_basis("6-31g*"));
  EXPECT_FALSE(fb::is_builtin_basis("cc-pvqz"));
  EXPECT_FALSE(fb::is_builtin_basis(""));
}

TEST(IoBasisBuiltins, AnUnknownBasisReadsAsAPathAndSaysWhatTheNamesAre) {
  auto stamp = fb::resolve_basis("sto-3gg");
  ASSERT_FALSE(stamp.has_value());
  // Whatever the user meant, what failed was opening a path.
  EXPECT_EQ(stamp.error().kind, fb::ErrorKind::IO);
  EXPECT_NE(stamp.error().message.find("STO-3G"), std::string::npos) << stamp.error().message;
}

TEST(IoBasisBuiltins, APathIsStillRead) {
  // The escape hatch: a basis this build does not carry needs no rebuild.
  const fb::test::ScratchDir dir("io_basis_builtins");
  const auto path = dir / "custom.json";
  ASSERT_TRUE(fb::write_file(path, kBasis).has_value());

  auto stamp = fb::resolve_basis(path.string());
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  EXPECT_EQ(stamp->element_shells().size(), 2U);
}

TEST(IoBasisBuiltins, OnlyACompiledInSetIsNamed) {
  // The canonical spelling, whatever spelling asked for it, and kept through
  // the normalization the BasisSet step applies.
  auto sto = fb::resolve_basis("sto3g");
  ASSERT_TRUE(sto.has_value()) << sto.error().message;
  EXPECT_EQ(sto->name(), "STO-3G");
  EXPECT_EQ(sto->primitive_normalized().name(), "STO-3G");

  // A file is unnamed even when its document claims a compiled-in name: the
  // name promises the baked SAD stamps the set's own shells.
  const fb::test::ScratchDir dir("io_basis_named");
  const auto path = dir / "claims-sto-3g.json";
  ASSERT_TRUE(
      fb::write_file(path, R"({"name": "STO-3G", )" + std::string{kBasis}.substr(1)).has_value());
  auto file = fb::resolve_basis(path.string());
  ASSERT_TRUE(file.has_value()) << file.error().message;
  EXPECT_EQ(file->name(), "");
  auto read = fb::read_basis(path);
  ASSERT_TRUE(read.has_value()) << read.error().message;
  EXPECT_EQ(read->name(), "");
}

// The compiled-in fitting sets (data/aux_basis/, their own table) and
// resolve_aux_basis. Their placement over a molecule is io_basis_gpu.cpp's.

TEST(IoBasisAux, BothJkfitSetsResolveByAnySpelling) {
  // Both sets are tools/jkfit_from_pyscf.py's: PySCF's 19 elements, H, B-Ne,
  // Al-Ar, Ga-Kr.
  struct Spelling {
    std::string_view spec;
    std::string_view canonical;
  };
  for (const auto &[spec, canonical] :
       std::array<Spelling, 4>{{{"cc-pVDZ-JKFIT", "cc-pVDZ-JKFIT"},
                                {"ccpvdzjkfit", "cc-pVDZ-JKFIT"},
                                {"cc-pVTZ-JKFIT", "cc-pVTZ-JKFIT"},
                                {"CC_PVTZ_JKFIT", "cc-pVTZ-JKFIT"}}}) {
    auto stamp = fb::resolve_aux_basis(spec);
    ASSERT_TRUE(stamp.has_value()) << spec << ": " << stamp.error().message;
    EXPECT_EQ(stamp->name(), canonical);
    EXPECT_EQ(stamp->element_shells().size(), 19U) << spec;
    EXPECT_NE(stamp->shells_for(36), nullptr) << spec;
    EXPECT_EQ(stamp->shells_for(2), nullptr) << spec;
  }
}

TEST(IoBasisAux, TheFittingAndOrbitalTablesAreDisjoint) {
  // A fitting set is no orbital set: tools/sad and io_sad_host see only the
  // five that carry data/sad/ stamps.
  for (const std::string_view aux : {"cc-pVDZ-JKFIT", "cc-pvtz-jkfit"}) {
    EXPECT_FALSE(fb::is_builtin_basis(aux)) << aux;
    auto stamp = fb::resolve_basis(aux);
    ASSERT_FALSE(stamp.has_value()) << aux;
    EXPECT_EQ(stamp.error().kind, fb::ErrorKind::IO);
  }
  // ... and an orbital set's name is no fitting set's.
  auto orbital = fb::resolve_aux_basis("cc-pvdz");
  ASSERT_FALSE(orbital.has_value());
  EXPECT_EQ(orbital.error().kind, fb::ErrorKind::IO);
}

TEST(IoBasisAux, AnUnknownNameReadsAsAPathAndSaysWhatTheNamesAre) {
  auto stamp = fb::resolve_aux_basis("cc-pvqz-jkfit");
  ASSERT_FALSE(stamp.has_value());
  EXPECT_EQ(stamp.error().kind, fb::ErrorKind::IO);
  EXPECT_NE(stamp.error().message.find("cc-pVDZ-JKFIT"), std::string::npos)
      << stamp.error().message;
  EXPECT_EQ(stamp.error().message.find("STO-3G"), std::string::npos) << stamp.error().message;
}

TEST(IoBasisAux, APathIsStillReadAndUnnamed) {
  const fb::test::ScratchDir dir("io_basis_aux");
  const auto path = dir / "custom-fit.json";
  ASSERT_TRUE(fb::write_file(path, kBasis).has_value());

  auto stamp = fb::resolve_aux_basis(path.string());
  ASSERT_TRUE(stamp.has_value()) << stamp.error().message;
  EXPECT_EQ(stamp->element_shells().size(), 2U);
  EXPECT_EQ(stamp->name(), "");
}

TEST(IoBasisAux, TheTripleZetaSetIsTheDoubleZetaSetPlusItsTopMomentum) {
  // Molpro's cc-pVDZ-JKFIT, which PySCF carries, is cc-pVTZ-JKFIT with each
  // element's highest angular momentum dropped (data/aux_basis/README.md).
  auto dz = fb::resolve_aux_basis("cc-pvdz-jkfit");
  auto tz = fb::resolve_aux_basis("cc-pvtz-jkfit");
  ASSERT_TRUE(dz.has_value()) << dz.error().message;
  ASSERT_TRUE(tz.has_value()) << tz.error().message;
  for (const auto &[z, tz_shells] : tz->element_shells()) {
    SCOPED_TRACE(std::format("Z={}", z));
    const std::vector<fb::ShellStamp> *dz_shells = dz->shells_for(z);
    ASSERT_NE(dz_shells, nullptr);
    std::int32_t top = 0;
    for (const fb::ShellStamp &shell : tz_shells) {
      top = std::max(top, shell.angular_momentum());
    }
    std::size_t kept = 0;
    for (const fb::ShellStamp &shell : tz_shells) {
      if (shell.angular_momentum() == top) {
        continue;
      }
      ASSERT_LT(kept, dz_shells->size());
      const fb::ShellStamp &match = (*dz_shells)[kept++];
      EXPECT_EQ(match.angular_momentum(), shell.angular_momentum());
      ASSERT_EQ(match.nprimitives(), shell.nprimitives());
      for (std::size_t p = 0; p < shell.nprimitives(); ++p) {
        EXPECT_EQ(match.primitives()[p].exponent, shell.primitives()[p].exponent);
        EXPECT_EQ(match.primitives()[p].coefficient, shell.primitives()[p].coefficient);
      }
    }
    EXPECT_EQ(kept, dz_shells->size());
  }
}

} // namespace
