/// @file
/// @brief Skip-in-fast-mode helper for the heavy end-to-end validation tests,
/// and the markers that name a family's representative or exempt it.
///
/// Lives in test/support/ (target free_breakfast::test::support), the one home
/// of the helpers every gtest tree shares; consumers include it by bare name.
///
/// The C++ analogue of the pytest `slow` marker: a full run executes these
/// cases (a bare ctest, `devtools/test.py nightly`), while both gates,
/// `devtools/test.py push` and `devtools/test.py pr`, opt out by exporting
/// FREE_BREAKFAST_FAST_TESTS. Default -- the variable unset -- runs everything,
/// so nothing changes for a bare ctest run.
///
/// The rule, in family terms (docs/testing.md "Families"). The fast set has a
/// budget: both gates must finish in under five minutes on a busy card, where
/// every case runs slower. So a case that takes more than about 1.5 s on an
/// IDLE RTX 3080 (the gtest `(N ms)` of a fast run) goes under
/// FBF_SKIP_IF_FAST_TESTS, unless it is its family's representative.
///
/// - A family is a gtest suite (TEST, TEST_F), or one TEST_P across its params
///   and instantiation prefixes (a TYPED_TEST across its types).
/// - Its representative is the case marked FBF_FAMILY_REPRESENTATIVE, else its
///   cheapest case by the recorded times. `devtools/test_families.py` lists
///   every family and its representative, and flags a family with neither.
/// - Mark a case where the cheapest is the wrong stand-in. Where a family's
///   cheapest case is a refusal, mark its cheapest validation case, which
///   stays in the fast set too. Where one mode of a TEST_P needs a fast case
///   of its own, split the mode off (below).
///
/// The line is set by the budget, not by a measured slowdown: at about ten
/// seconds the batch was 552-592 s on an idle 3080 and ~770 s on the gfx1200;
/// at 1.5 s it is ~289 s (estimated) and 419 s (#977). A busy card can be far
/// slower (CLAUDE.md: 2.3 s alone, 158 s shared).
///
/// Put the macro at the TOP of a test body, after FBF_REQUIRE_GPU():
/// a slow test is also a GPU test, so it should still skip cleanly on a
/// device-less host.
///
/// FBF_FAMILY_REPRESENTATIVE goes at the top of the test body too. It
/// compiles to nothing (a `static_assert(true)`, valid in any scope); the
/// lister reads it from the source, and nothing but the lister checks its
/// arguments, which are string literals:
///
///   FBF_FAMILY_REPRESENTATIVE();             a TEST or TEST_F: this case
///   FBF_FAMILY_REPRESENTATIVE("p");          a TEST_P: its instance p
///   FBF_FAMILY_REPRESENTATIVE("p", "re");    ... of a family of its own: the
///                                            instances whose param matches re
///
/// An instance is named by its param, or `Prefix/param` where two
/// instantiations share one; `re` is a Python regex, searched for.
///
/// A family with no case that could stay fast, such as a measurement that
/// asserts nothing, is declared exempt instead, with the reason, in any one
/// of its cases (#1008). The reason is one string literal, on one line. The lister
/// reports it as exempt rather than flagged:
///
///   FBF_FAMILY_EXEMPT("why");                the family of this case
///
/// Exempt only where no smaller input keeps the family's check: a family takes
/// a representative or an exemption, never both.
///
/// FBF_SKIP_IF_FAST_TESTS also records the gtest property `fbf_slow`, in
/// every mode, so a full run's gtest reports (a GPU batch's JSON, each ctest
/// entry's XML) say which cases a fast run would have skipped: the nightly reads it to tell the fast set from the slow
/// one (docs/testing.md "The slow-test rule, enforced"; #987). A case that
/// reaches the macro only for some params is marked for exactly those.
///
/// std::getenv comes from the including TU's `import std;` -- the macro is
/// expanded at the point of use, like gpu_requirement.h, not resolved here.

#pragma once

#include <gtest/gtest.h>

#define FBF_SKIP_IF_FAST_TESTS()                                                                   \
  do {                                                                                             \
    ::testing::Test::RecordProperty("fbf_slow", "1");                                              \
    if (const char *fb_fast = std::getenv("FREE_BREAKFAST_FAST_TESTS");                            \
        fb_fast != nullptr && fb_fast[0] != '\0' && fb_fast[0] != '0') {                           \
      GTEST_SKIP() << "slow validation test skipped in fast mode "                                 \
                      "(FREE_BREAKFAST_FAST_TESTS is set)";                                        \
    }                                                                                              \
  } while (0)

#define FBF_FAMILY_REPRESENTATIVE(...) static_assert(true)
#define FBF_FAMILY_EXEMPT(reason) static_assert(true)
