/// @file
/// @brief The occupation strings of one spin for a determinant CI: every way
/// of putting N electrons in n orbitals, their lexical addresses, and the
/// single-replacement lists for E_pq.
///
/// Host-side and over plain integers. A determinant is a pair of strings, one
/// per spin, each held by its own StringSet; for N_alpha == N_beta the two are
/// the same set.

export module free_breakfast.ci.strings;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;

export namespace free_breakfast::ci {

/// @brief One term of a string's single-replacement list:
/// a†_p a_q |I> = sign |J>, with J at address @c target.
///
/// The spin-free E_pq is the sum of this over the two spins.
struct Replacement {
  int32_t target; ///< Address of J in the same StringSet
  int32_t p;      ///< The orbital created
  int32_t q;      ///< The orbital annihilated, occupied in I
  int32_t sign;   ///< +1 or -1
};

/// @brief Every occupation string of @c n_electrons() electrons in
/// @c n_orbitals() orbitals, in lexical order, with their replacement lists.
///
/// A string is a bit pattern: bit k set means orbital k occupied. Strings are
/// addressed in ascending order of that pattern read as an integer, which is
/// PySCF's `cistring` order, so a CI vector laid out over two StringSets
/// matches `fci.direct_spin1`'s element for element. The address is
/// sum over the k-th occupied orbital o_k (k from 0, o ascending) of
/// C(o_k, k + 1).
///
/// The sign convention orders creation operators by ascending orbital,
/// |I> = a†_{o_0} a†_{o_1} ... |vac>, so a†_p a_q carries (-1) to the number of
/// orbitals occupied in I strictly between p and q.
class StringSet {
public:
  /// @brief Enumerate the strings of @p n_electrons in @p n_orbitals and
  /// build their replacement lists.
  ///
  /// @return InvalidConfig for @p n_orbitals outside [0, 64], @p n_electrons
  ///         outside [0, n_orbitals], or more strings, or replacements in
  ///         all, than an int32_t counts.
  [[nodiscard]] static Result<StringSet> create(int32_t n_orbitals, int32_t n_electrons);

  /// @brief n, the orbitals a string spans.
  [[nodiscard]] int32_t n_orbitals() const noexcept { return n_orbitals_; }
  /// @brief N, the electrons in every string.
  [[nodiscard]] int32_t n_electrons() const noexcept { return n_electrons_; }
  /// @brief C(n, N), the number of strings; addresses are [0, size()).
  [[nodiscard]] int32_t size() const noexcept { return static_cast<int32_t>(strings_.size()); }

  /// @brief Every string's bit pattern, indexed by address, so ascending.
  [[nodiscard]] std::span<const uint64_t> strings() const noexcept { return strings_; }

  /// @brief The address of @p string, or -1 when it is not one of this set's
  /// (a bit at or above n_orbitals(), or a popcount other than n_electrons()).
  [[nodiscard]] int32_t address(uint64_t string) const noexcept;

  /// @brief N (n - N + 1), the length of every string's replacement list.
  [[nodiscard]] int32_t replacements_per_string() const noexcept {
    return n_electrons_ * (n_orbitals_ - n_electrons_ + 1);
  }

  /// @brief The nonzero a†_p a_q |I> for the string at @p address.
  ///
  /// The N diagonal terms p == q (sign +1, target I) come first, over the
  /// occupied q ascending; then, for each occupied q ascending, each empty p
  /// ascending. That is PySCF's `gen_linkstr_index` order. Empty when N is 0.
  ///
  /// @param address In [0, size()).
  [[nodiscard]] std::span<const Replacement> replacements(int32_t address) const noexcept {
    const auto per = static_cast<size_t>(replacements_per_string());
    return std::span<const Replacement>(replacements_)
        .subspan(static_cast<size_t>(address) * per, per);
  }

private:
  StringSet(int32_t n_orbitals, int32_t n_electrons, std::vector<int32_t> weights,
            std::vector<uint64_t> strings, std::vector<Replacement> replacements)
      : n_orbitals_{n_orbitals}, n_electrons_{n_electrons}, weights_{std::move(weights)},
        strings_{std::move(strings)}, replacements_{std::move(replacements)} {}

  int32_t n_orbitals_;
  int32_t n_electrons_;
  // weights_[k * n + o] = C(o, k + 1): the k-th electron's contribution to the
  // address when it sits in orbital o.
  std::vector<int32_t> weights_;
  std::vector<uint64_t> strings_;
  std::vector<Replacement> replacements_;
};

} // namespace free_breakfast::ci
