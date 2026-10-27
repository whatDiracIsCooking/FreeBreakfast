/// @file
/// @brief The DF three-center integrals' limits, declared once for the module
/// that will launch the generated eri3<la, lb, lP> and the device TUs beside
/// them.
///
/// A plain header, with no runtime type in it, so a host module unit can
/// include it in its GLOBAL MODULE FRAGMENT and a .cu directly, for the reason
/// overlap_bridge.h gives. The per-class launch table itself is
/// kernels/eri3.h's `eri3_kernels`, which only a device TU can include;
/// three_center_table.cu checks it against these limits.

#pragma once

#include "integral_types.h"

#include "angular_momentum.h"

namespace free_breakfast::device {

/// @brief (mn|P)'s orbital L limit: the largest momentum of an orbital shell m
/// or n in any class it takes, f. Its kernels are emitted through it
/// (tools/emit_df_kernels.py's MAX_L, which the tool checks against this).
inline constexpr int32_t kThreeCenterMaxMomentum = 3;

/// @brief (mn|P)'s aux L limit: the largest momentum of an auxiliary shell P
/// in any class it takes, g (tools/emit_df_kernels.py's AUX_MAX_L, as
/// kMetricMaxMomentum is).
inline constexpr int32_t kThreeCenterAuxMaxMomentum = 4;

static_assert(kThreeCenterMaxMomentum == kMaxAngularMomentum,
              "(mn|P)'s orbital limit and the tree-wide kMaxAngularMomentum "
              "disagree");

} // namespace free_breakfast::device
