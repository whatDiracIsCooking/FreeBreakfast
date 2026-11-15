/// @file
/// @brief The trivially-copyable functor concept device code constrains its
/// fold ops with.
///
/// Forked from src/core/kernel/device_functor.h and trimmed to what rust/'s
/// launch wrappers use: DeviceFunctor, its definition unchanged.
/// IdentityFunctor comes back with the first wrapper that names it.
///
/// DeviceFunctor names the host-checkable half of a device functor's contract:
/// a trivially-copyable class type. The callability probe (`{ f(a, b) } -> T`)
/// is deliberately not here -- in a host context it would reject a
/// `__device__`-only operator(), so each device template carries its own
/// `requires`, where a device entity accepts it.

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
