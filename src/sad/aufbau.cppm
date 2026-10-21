/// @file
/// @brief The neutral atom's aufbau fill: @ref kAufbauOrder and @ref
/// subshell_occupation, which the atomic SCF seats on its l blocks' orbitals.

// An internal partition of free_breakfast.sad.
module free_breakfast.sad:aufbau;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;

namespace free_breakfast {

/// @brief One subshell in the aufbau filling order.
struct SubshellFill {
  int32_t l;        ///< Angular momentum (0=s, 1=p, 2=d, 3=f).
  int32_t capacity; ///< Electrons that fill it, 2(2l+1).
};

/// @brief The Madelung (n+l, then n) filling order, 1s 2s 2p 3s 3p 4s 3d 4p ...
/// 7p: long enough to seat any Z in [1, 118].
///
/// Written out rather than generated so the order is auditable at a glance.
constexpr std::array<SubshellFill, 19> kAufbauOrder{{
    {0, 2},  // 1s
    {0, 2},  // 2s
    {1, 6},  // 2p
    {0, 2},  // 3s
    {1, 6},  // 3p
    {0, 2},  // 4s
    {2, 10}, // 3d
    {1, 6},  // 4p
    {0, 2},  // 5s
    {2, 10}, // 4d
    {1, 6},  // 5p
    {0, 2},  // 6s
    {3, 14}, // 4f
    {2, 10}, // 5d
    {1, 6},  // 6p
    {0, 2},  // 7s
    {3, 14}, // 5f
    {2, 10}, // 6d
    {1, 6},  // 7p
}};

/// @brief Ground-state electron count per subshell for a neutral atom of atomic
/// number @p z, aligned to @ref kAufbauOrder.
///
/// A plain aufbau fill: full subshells hold their capacity, the last holds the
/// remainder, and trailing empty subshells are omitted, so the entries sum to
/// exactly @p z. The d/f anomalies (Cr, Cu, ...) are not encoded; a spherically
/// averaged guess does not need them.
///
/// @param z Atomic number in [0, 118]; 0 (@c Element::Sentinel) gives an empty
///   result. Anything outside that range aborts -- no @c Element spells it, so
///   it is a caller bug.
[[nodiscard]] std::vector<int32_t> subshell_occupation(int32_t z) {
  if (z < 0 || z > 118) {
    check_or_abort(err_config("subshell_occupation: Z={} is outside [0, 118]", z));
  }
  std::vector<int32_t> occupation;
  int32_t remaining = z;
  for (const SubshellFill &subshell : kAufbauOrder) {
    if (remaining == 0) {
      break;
    }
    const int32_t here = std::min(remaining, subshell.capacity);
    occupation.push_back(here);
    remaining -= here;
  }
  return occupation;
}

} // namespace free_breakfast
