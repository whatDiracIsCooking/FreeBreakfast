// Host cases for ci.strings: the enumeration and its addressing, and the
// single-replacement lists checked against a†_p a_q applied by brute force to
// bit patterns, which shares no code with the module's own sign rule.

#include <gtest/gtest.h>

import std;
import free_breakfast.core.error_handling;
import free_breakfast.ci;

namespace {

namespace fb = free_breakfast;

std::int64_t choose(std::int32_t n, std::int32_t k) {
  std::int64_t c = 1;
  for (std::int32_t i = 0; i < k; ++i) {
    c = c * (n - i) / (i + 1);
  }
  return c;
}

// a_q then a†_p on |s>, with |s> = a†_{o_0} a†_{o_1} ... |vac>, o ascending:
// each operator passes the occupied orbitals below it. Returns the sign and the
// new pattern, or a sign of 0 when the result vanishes.
std::pair<std::int32_t, std::uint64_t> apply_brute(std::uint64_t s, std::int32_t p,
                                                   std::int32_t q) {
  std::int32_t sign = 1;
  if (((s >> q) & 1U) == 0) {
    return {0, 0};
  }
  for (std::int32_t o = 0; o < q; ++o) {
    sign *= ((s >> o) & 1U) != 0 ? -1 : 1;
  }
  s &= ~(std::uint64_t{1} << q);
  if (((s >> p) & 1U) != 0) {
    return {0, 0};
  }
  for (std::int32_t o = 0; o < p; ++o) {
    sign *= ((s >> o) & 1U) != 0 ? -1 : 1;
  }
  s |= std::uint64_t{1} << p;
  return {sign, s};
}

TEST(CiStrings, AddressingIsABijectionOntoTheCount) {
  for (std::int32_t n = 0; n <= 10; ++n) {
    for (std::int32_t n_e = 0; n_e <= n; ++n_e) {
      SCOPED_TRACE(std::format("n = {}, N = {}", n, n_e));
      const auto set = fb::StringSet::create(n, n_e);
      ASSERT_TRUE(set.has_value()) << set.error().message;
      ASSERT_EQ(set->size(), choose(n, n_e));
      EXPECT_EQ(set->n_orbitals(), n);
      EXPECT_EQ(set->n_electrons(), n_e);

      // Every pattern of n bits: those with N set are the strings, each at its
      // own address, ascending; every other pattern has none.
      std::vector<bool> seen(static_cast<std::size_t>(set->size()), false);
      std::int32_t expected = 0;
      for (std::uint64_t s = 0; s < (std::uint64_t{1} << n); ++s) {
        const std::int32_t address = set->address(s);
        if (std::popcount(s) != n_e) {
          EXPECT_EQ(address, -1);
          continue;
        }
        ASSERT_EQ(address, expected) << "pattern " << s;
        EXPECT_FALSE(seen[static_cast<std::size_t>(address)]);
        seen[static_cast<std::size_t>(address)] = true;
        EXPECT_EQ(set->strings()[static_cast<std::size_t>(address)], s);
        ++expected;
      }
      EXPECT_EQ(expected, set->size());

      // And back: address -> string -> address.
      for (std::int32_t a = 0; a < set->size(); ++a) {
        EXPECT_EQ(set->address(set->strings()[static_cast<std::size_t>(a)]), a);
      }
    }
  }
}

TEST(CiStrings, ReplacementsMatchBruteForce) {
  for (std::int32_t n = 0; n <= 8; ++n) {
    for (std::int32_t n_e = 0; n_e <= n; ++n_e) {
      SCOPED_TRACE(std::format("n = {}, N = {}", n, n_e));
      const auto set = fb::StringSet::create(n, n_e);
      ASSERT_TRUE(set.has_value()) << set.error().message;
      ASSERT_EQ(set->replacements_per_string(), n_e * (n - n_e + 1));

      for (std::int32_t a = 0; a < set->size(); ++a) {
        const std::uint64_t s = set->strings()[static_cast<std::size_t>(a)];
        const auto list = set->replacements(a);
        ASSERT_EQ(static_cast<std::int32_t>(list.size()), set->replacements_per_string());

        // Diagonal terms first, over the occupied orbitals ascending.
        std::int32_t d = 0;
        for (std::int32_t q = 0; q < n; ++q) {
          if (((s >> q) & 1U) != 0) {
            ASSERT_LT(d, n_e);
            EXPECT_EQ(list[static_cast<std::size_t>(d)].p, q);
            EXPECT_EQ(list[static_cast<std::size_t>(d)].q, q);
            ++d;
          }
        }

        // Every (p, q) whose a†_p a_q |s> survives appears exactly once, with
        // the brute-force target and sign; nothing else appears.
        std::int32_t nonzero = 0;
        for (std::int32_t p = 0; p < n; ++p) {
          for (std::int32_t q = 0; q < n; ++q) {
            const auto [sign, target] = apply_brute(s, p, q);
            const auto hits = std::ranges::count_if(
                list, [&](const fb::Replacement &r) { return r.p == p && r.q == q; });
            if (sign == 0) {
              EXPECT_EQ(hits, 0) << "p " << p << " q " << q;
              continue;
            }
            ++nonzero;
            ASSERT_EQ(hits, 1) << "p " << p << " q " << q;
            const auto r = *std::ranges::find_if(
                list, [&](const fb::Replacement &e) { return e.p == p && e.q == q; });
            EXPECT_EQ(r.sign, sign) << "p " << p << " q " << q;
            EXPECT_EQ(r.target, set->address(target)) << "p " << p << " q " << q;
            EXPECT_EQ(set->strings()[static_cast<std::size_t>(r.target)], target);
          }
        }
        EXPECT_EQ(nonzero, set->replacements_per_string());
      }
    }
  }
}

TEST(CiStrings, NoElectronsIsTheVacuumAlone) {
  const auto set = fb::StringSet::create(6, 0);
  ASSERT_TRUE(set.has_value()) << set.error().message;
  ASSERT_EQ(set->size(), 1);
  EXPECT_EQ(set->strings()[0], 0U);
  EXPECT_EQ(set->address(0), 0);
  EXPECT_EQ(set->replacements_per_string(), 0);
  EXPECT_TRUE(set->replacements(0).empty());

  const auto none = fb::StringSet::create(0, 0);
  ASSERT_TRUE(none.has_value()) << none.error().message;
  EXPECT_EQ(none->size(), 1);
  EXPECT_TRUE(none->replacements(0).empty());
}

TEST(CiStrings, FullShellHasOnlyTheDiagonal) {
  const auto set = fb::StringSet::create(5, 5);
  ASSERT_TRUE(set.has_value()) << set.error().message;
  ASSERT_EQ(set->size(), 1);
  EXPECT_EQ(set->strings()[0], 0b11111U);
  EXPECT_EQ(set->address(0b11111U), 0);
  const auto list = set->replacements(0);
  ASSERT_EQ(list.size(), 5U);
  for (std::int32_t q = 0; q < 5; ++q) {
    const auto &r = list[static_cast<std::size_t>(q)];
    EXPECT_EQ(r.p, q);
    EXPECT_EQ(r.q, q);
    EXPECT_EQ(r.target, 0);
    EXPECT_EQ(r.sign, 1);
  }
}

TEST(CiStrings, SixtyFourOrbitalsUseEveryBit) {
  // The widest a uint64_t holds: the top bit is an orbital, not out of range.
  const auto one = fb::StringSet::create(64, 1);
  ASSERT_TRUE(one.has_value()) << one.error().message;
  ASSERT_EQ(one->size(), 64);
  EXPECT_EQ(one->address(std::uint64_t{1} << 63), 63);
  EXPECT_EQ(one->strings()[63], std::uint64_t{1} << 63);

  const auto full = fb::StringSet::create(64, 64);
  ASSERT_TRUE(full.has_value()) << full.error().message;
  ASSERT_EQ(full->size(), 1);
  EXPECT_EQ(full->strings()[0], ~std::uint64_t{0});
  EXPECT_EQ(full->address(~std::uint64_t{0}), 0);
}

TEST(CiStrings, AddressRejectsForeignPatterns) {
  const auto set = fb::StringSet::create(4, 2);
  ASSERT_TRUE(set.has_value()) << set.error().message;
  EXPECT_EQ(set->address(0b0111U), -1);  // three electrons
  EXPECT_EQ(set->address(0b10001U), -1); // orbital 4 is outside n = 4
}

TEST(CiStrings, RejectsBadShapes) {
  EXPECT_EQ(fb::StringSet::create(-1, 0).error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_EQ(fb::StringSet::create(65, 1).error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_EQ(fb::StringSet::create(4, 5).error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_EQ(fb::StringSet::create(4, -1).error().kind, fb::ErrorKind::InvalidConfig);
  // C(64, 32) strings: more than an int32_t addresses.
  EXPECT_EQ(fb::StringSet::create(64, 32).error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
