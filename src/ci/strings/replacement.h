/// @file
/// @brief Replacement: one link of a string's single-replacement list, as the
/// host builds it and the device gathers read it.
///
/// This header is the single definition: a launcher reaches it by `#include`
/// (through its bridge header), a module by `import free_breakfast.ci.strings;`,
/// which includes it in its global module fragment and re-exports the name
/// (atom.cppm has the why).

#pragma once

#include "integral_types.h"

namespace free_breakfast::ci {

/// @brief One term of a string's single-replacement list:
/// a†_p a_q |I> = sign |J>, with J at address @c target.
///
/// The spin-free E_pq is the sum of this over the two spins. Eight bytes,
/// aligned to eight, so a gather reads a link in one load; every consumer
/// derives the pair index it wants (packed, p · n + q, q · n + p) from @c p and
/// @c q. An orbital fits an int8_t because a string spans at most 64.
struct alignas(8) Replacement {
  int32_t target; ///< Address of J in the same StringSet
  int16_t sign;   ///< +1 or -1
  int8_t p;       ///< The orbital created
  int8_t q;       ///< The orbital annihilated, occupied in I
};

static_assert(sizeof(Replacement) == 8);

} // namespace free_breakfast::ci
