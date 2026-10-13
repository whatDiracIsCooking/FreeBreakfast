/// @file
/// @brief The angular-momentum limit of the Gaussian basis description:
/// @ref free_breakfast::kMaxAngularMomentum.
///
/// A center-independent scalar, like the pieces in @c basis.h: it carries no
/// @c wwr::double3, so it lives in a plain header both CUDA
/// `#include`rs and the `basis` module share (included in the module's global
/// module fragment, its name re-exported from within the namespace so importers
/// and CUDA code see one entity).

#pragma once

namespace free_breakfast {

/// @brief The largest angular momentum the GPU kernels currently support (d).
///
/// The device s/p/d kernel set covers L in [0, @ref kMaxAngularMomentum]; the
/// GPU integral/Fock engines reject a basis with any higher shell. @ref Shell
/// itself imposes no such cap -- the host reference engines evaluate arbitrary L
/// -- so this is the GPU-kernel limit, not a basis-wide invariant.
inline constexpr int kMaxAngularMomentum = 2;

} // namespace free_breakfast
