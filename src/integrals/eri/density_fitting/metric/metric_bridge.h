/// @file
/// @brief The DF Coulomb metric's limit, declared once for the module that
/// will launch the generated metric<lP, lQ> and the device TUs beside them.
///
/// A plain header, with no runtime type in it, so a host module unit can
/// include it in its GLOBAL MODULE FRAGMENT and a .cu directly, for the reason
/// overlap_bridge.h gives. The per-class launch table itself is
/// kernels/metric.h's `metric_kernels`, which only a device TU can include;
/// metric_table.cu checks it against this limit.

#pragma once

#include "integral_types.h"

namespace free_breakfast::device {

/// @brief The metric's L limit: the largest momentum of an auxiliary shell in
/// any class (P|Q) it takes, g. Its kernels are emitted through it
/// (tools/emit_df_kernels.py's AUX_MAX_L, which the tool checks against this),
/// and metric_table.cu checks their table against it; the tree-wide
/// kMaxAngularMomentum is the orbital basis's, not the aux basis's.
inline constexpr int32_t kMetricMaxMomentum = 4;

} // namespace free_breakfast::device
