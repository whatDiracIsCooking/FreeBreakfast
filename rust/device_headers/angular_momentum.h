/// @file
/// @brief The angular-momentum limit of the Gaussian basis description:
/// @ref free_breakfast::kMaxAngularMomentum.
///
/// Forked from src/basis/angular_momentum.h, its definition unchanged; the
/// module re-export it documents has no counterpart here. fb_basis's
/// MAX_ANGULAR_MOMENTUM is asserted equal to it by a test that reads this
/// file, so the two cannot drift apart.

#pragma once

#include "integral_types.h"

namespace free_breakfast {

/// @brief The largest angular momentum the GPU kernels currently support (f).
///
/// The device s/p/d/f kernel set covers L in [0, @ref kMaxAngularMomentum]; the
/// GPU integral/Fock engines reject a basis with any higher shell. @ref Shell
/// itself imposes no such cap -- the host reference engines evaluate arbitrary L
/// -- so this is the GPU-kernel limit, not a basis-wide invariant.
inline constexpr int32_t kMaxAngularMomentum = 3;

} // namespace free_breakfast
