/// @file
/// @brief The trivially-copyable functor concept device code constrains its
/// fold and transform ops with, plus the identity pre-transform.
///
/// DeviceFunctor names the host-checkable half of a device functor's contract:
/// a trivially-copyable class type. The callability probe (`{ f(a, b) } -> T`)
/// is deliberately not here -- in a host context it would reject a
/// `__device__`-only operator(), so each device template carries its own
/// `requires`, where a device entity accepts it.
///
/// It is not wwr's `device_functor` (wwr.extension.parallel_for), which also
/// demands !is_copy_assignable so that functor can ride WWR_GRID_CONSTANT. These
/// functors are plain kernel and device-function parameters, never grid
/// constants, so a functor without const members must still pass.
///
/// The concept is a plain trait, so this header #includes anywhere -- a host
/// TU, a module's GMF, a device .cu. IdentityFunctor is `__host__ __device__`
/// and sits behind the compiler's device-pass macros: absent in a host compile
/// rather than an error, the shape of wwr's complex.h.

#pragma once

#include <type_traits>

namespace free_breakfast::device {

/// @brief A trivially-copyable class functor, passable by value to a kernel
///
/// The callability and signature checks live on the consuming device function;
/// its @tparam docs state the shape each op must have.
template <typename F>
concept DeviceFunctor = std::is_trivially_copyable_v<F> && std::is_class_v<F>;

} // namespace free_breakfast::device

// Only in a device-compile pass, where __host__ __device__ are keywords.
#if defined(__CUDACC__) || defined(__HIP__) || defined(__HIPCC__)

namespace free_breakfast::device {

/// @brief Pre-transform that passes each element through unchanged
///
/// Seeds the identity case of a transform-reduce. `__host__ __device__` so the
/// same type is usable on either side of the host/device line within a device
/// TU.
template <typename T>
struct IdentityFunctor {
  /// @brief @p x, unchanged
  __host__ __device__ T operator()(const T &x) const { return x; }
};

} // namespace free_breakfast::device

#endif // device-compile pass
