/// @file
/// @brief StringSet: enumeration, addressing and the replacement lists.

module free_breakfast.ci.strings;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;

namespace free_breakfast {

// A uint64_t holds the bit pattern, so 64 orbitals is the most a string spans.
constexpr int32_t kMaxStringOrbitals = 64;

Result<StringSet> StringSet::create(const int32_t n_orbitals, const int32_t n_electrons) {
  if (n_orbitals < 0 || n_orbitals > kMaxStringOrbitals) {
    return err_config("StringSet: {} orbitals, outside [0, {}]", n_orbitals, kMaxStringOrbitals);
  }
  if (n_electrons < 0 || n_electrons > n_orbitals) {
    return err_config("StringSet: {} electrons in {} orbitals", n_electrons, n_orbitals);
  }
  const auto n = static_cast<size_t>(n_orbitals);
  const auto n_e = static_cast<size_t>(n_electrons);

  // Pascal's triangle up to C(n, N), saturated one past the int32_t range:
  // C(64, 32) overflows even an int64_t, and only whether a count fits matters
  // once it does not.
  constexpr int64_t kCap = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  std::vector<int64_t> binom((n + 1) * (n_e + 1), 0);
  const auto c = [&](size_t i, size_t j) -> int64_t & { return binom[i * (n_e + 1) + j]; };
  for (size_t i = 0; i <= n; ++i) {
    c(i, 0) = 1;
    for (size_t j = 1; j <= std::min(i, n_e); ++j) {
      c(i, j) = std::min(kCap, c(i - 1, j - 1) + c(i - 1, j));
    }
  }
  const int64_t count = c(n, n_e);
  const auto per = static_cast<int64_t>(n_electrons) * (n_orbitals - n_electrons + 1);
  if (count >= kCap || count * per >= kCap) {
    return err_config("StringSet: {} electrons in {} orbitals is more strings or replacements "
                      "than an int32_t counts",
                      n_electrons, n_orbitals);
  }

  // Every C(o, k + 1) with o < n and k < N is at most C(n, N), so it fits.
  std::vector<int32_t> weights(n_e * n);
  for (size_t k = 0; k < n_e; ++k) {
    for (size_t o = 0; o < n; ++o) {
      weights[k * n + o] = static_cast<int32_t>(c(o, k + 1));
    }
  }

  // Ascending by Gosper's hack: the next larger integer of the same popcount.
  // The step after the last string can overflow, so it is never taken.
  std::vector<uint64_t> strings(static_cast<size_t>(count));
  uint64_t string = n_e == 64 ? ~uint64_t{0} : (uint64_t{1} << n_e) - 1;
  for (size_t i = 0; i < strings.size(); ++i) {
    strings[i] = string;
    if (i + 1 < strings.size()) {
      const uint64_t low = string & (~string + 1);
      const uint64_t ripple = string + low;
      string = (((ripple ^ string) >> 2) / low) | ripple;
    }
  }

  StringSet set{n_orbitals, n_electrons, std::move(weights), std::move(strings), {}};

  std::vector<Replacement> replacements;
  replacements.reserve(static_cast<size_t>(count * per));
  std::vector<int32_t> occupied;
  std::vector<int32_t> empty;
  for (size_t i = 0; i < set.strings_.size(); ++i) {
    const uint64_t s = set.strings_[i];
    const auto address = static_cast<int32_t>(i);
    occupied.clear();
    empty.clear();
    for (int32_t o = 0; o < n_orbitals; ++o) {
      ((s >> o) & 1U ? occupied : empty).push_back(o);
    }
    for (const int32_t q : occupied) {
      replacements.push_back({address, q, q, 1});
    }
    for (const int32_t q : occupied) {
      for (const int32_t p : empty) {
        const int32_t lo = std::min(p, q);
        const int32_t hi = std::max(p, q);
        // Bits lo + 1 .. hi - 1; hi <= 63, so neither shift reaches 64.
        const uint64_t between = ((uint64_t{1} << hi) - 1) & ~((uint64_t{1} << (lo + 1)) - 1);
        const int32_t sign = std::popcount(s & between) % 2 == 0 ? 1 : -1;
        const uint64_t target = (s & ~(uint64_t{1} << q)) | (uint64_t{1} << p);
        replacements.push_back({set.address(target), p, q, sign});
      }
    }
  }
  set.replacements_ = std::move(replacements);
  return set;
}

int32_t StringSet::address(const uint64_t string) const noexcept {
  if (n_orbitals_ < kMaxStringOrbitals && (string >> n_orbitals_) != 0) {
    return -1;
  }
  if (std::popcount(string) != n_electrons_) {
    return -1;
  }
  const auto n = static_cast<size_t>(n_orbitals_);
  int32_t address = 0;
  size_t k = 0;
  for (uint64_t rest = string; rest != 0; rest &= rest - 1) {
    address += weights_[k * n + static_cast<size_t>(std::countr_zero(rest))];
    ++k;
  }
  return address;
}

} // namespace free_breakfast
