/// @file
/// @brief Which rung of a Richardson ladder a derivative check reads: the
/// coarsest settled one (#1153, #1157).
///
/// Lives in test/support/ (target free_breakfast::test::support), the one home
/// of the helpers every gtest tree shares; consumers include it by bare name.
///
/// Include it AFTER `import std;`, like scratch_dir.h: it includes no standard
/// header itself and names `std::` only through the including TU's imports.
///
/// The rule. A ladder's extrapolant R_k at step t_k = t_0 / 2^k is off by
/// c t_k^p plus noise n_k, with p = 4 for one level of Richardson over
/// central differences and p = 6 for two (central first and second
/// differences are both even in t). So
///
///     R_{k+1} - R_k = -(1 - 2^-p) c t_k^p + (n_{k+1} - n_k),
///
/// and a rung that moves at most kSettled bars to the next is off by at most
/// kSettled / (1 - 2^-p) bars, under 0.54 for p >= 4, plus its own noise. The
/// noise is the evaluations' round-off σ over the steps: about 3 σ / t_k for
/// one level and 6.6 σ / t_k for two over first differences, and over t_k^2
/// for second differences. It grows 2- or 4-fold with every rung, so the
/// coarsest settled rung is the most accurate one there is. The rung that
/// moves least over all of them is not: two noisy fine extrapolants can agree
/// by chance. Along rot_01 on h2co_ccpvdz_sa2_weighted, whose evaluations
/// repeat in-process only to 1e-13 to 9e-13 of their scale (6.6 σ / t_6 up to
/// 4e-7), that rule took rung 4 (off 1.1e-7, a nightly failure) and rung 6
/// (off 5.9e-7) where rung 0 had settled, off 5e-9 (#1153). Only where no
/// rung settles does the rule fall back to the one that moves least. A wrong
/// product misses at every rung alike, so the choice of rung cannot hide one,
/// and the bar is the check's own, unchanged.

#pragma once

namespace free_breakfast::test {

/// @brief A settled rung moves at most this many bars to the next.
inline constexpr double kSettled = 0.5;

/// @brief The rung a ladder reads, and how far it moves to the next.
struct SettledRung {
  std::size_t rung = 0;
  double change = std::numeric_limits<double>::infinity();
};

/// @brief The coarsest k < @p count whose @p change(k), how far R_k moves to
/// R_{k+1} in the units of @p bar, is at most kSettled bars; else the k whose
/// change is least.
template <class Change>
SettledRung settled_rung(const std::size_t count, const double bar, Change &&change) {
  SettledRung out;
  for (std::size_t k = 0; k < count; ++k) {
    const double moved = change(k);
    if (moved <= kSettled * bar) {
      return {.rung = k, .change = moved};
    }
    if (moved < out.change) {
      out = {.rung = k, .change = moved};
    }
  }
  return out;
}

} // namespace free_breakfast::test
