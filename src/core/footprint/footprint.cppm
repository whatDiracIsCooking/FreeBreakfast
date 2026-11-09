/// @file
/// @brief The footprint vocabulary: the bytes a routine or a step holds, per
///        memory kind and per lifetime, and the two ways footprints compose.
///
/// Every layer uses it: a library routine exports its own footprint next to
/// the code that allocates, and a pipeline step composes its routines'. See
/// README.md.

export module free_breakfast.core.footprint;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;

export namespace free_breakfast {

/// @brief The fragmentation headroom, in bytes: what pre-flight leaves between
///        a plan's device peak and its pool's cap, and what an adaptive
///        routine leaves of the memory budget when it sizes its adaptive part.
///
/// Two of the pool's 32 MiB chunks: a pool whose allocations pack holds past
/// its peak use at most the free top of its last chunk and the holes small
/// frees leave in the chunk its small allocations share, each under a chunk
/// (pipeline/footprint's README, "Fragmentation headroom"). Past its cap an
/// allocation through the pool aborts rather than returns, so this keeps a
/// footprint that fits on the refusing side of it.
inline constexpr size_t kPoolHeadroomBytes = size_t{64} << 20;

/// @brief The bytes of memory kind `K` a routine or step holds, one field per
///        lifetime.
///
/// - @ref persistent: what it returns and keeps (a step's product, to the
///   pipeline's end, whether it was computed or loaded).
/// - @ref scratch: what it allocates and frees before it returns; for an
///   adaptive routine, the least it runs with.
/// - @ref workspace: what it borrows from the `DeviceResources` workspaces.
///   Fixed-size uses only: never sized from a memory budget.
///
/// The kind is in the type, so a device footprint cannot be composed with,
/// or compared against, a pinned one.
template <wwr::extension::MemoryKind K>
struct Footprint {
  size_t persistent = 0;
  size_t scratch = 0;
  size_t workspace = 0;

  /// @brief The most it holds at once: all three fields together.
  [[nodiscard]] constexpr size_t peak() const { return persistent + scratch + workspace; }

  friend constexpr bool operator==(const Footprint &, const Footprint &) = default;
};

/// @brief What a routine or step reports: its device and pinned footprints.
///
/// Pageable host memory is not modelled, and unified memory is unused, so
/// `Host` and `Unified` have no member.
struct MemoryFootprint {
  Footprint<wwr::extension::MemoryKind::Device> device;
  Footprint<wwr::extension::MemoryKind::Pinned> pinned;

  friend constexpr bool operator==(const MemoryFootprint &, const MemoryFootprint &) = default;
};

/// @brief @p a runs and its output is kept, then @p b runs.
///
/// Exact, not a bound: `then(a, b).peak()` is
/// `max(a.persistent + a.scratch, a.persistent + b.persistent + b.scratch)`
/// plus the larger workspace. What of @p a's scratch rises above @p b's
/// persistent is the residue left in scratch, clamped at zero.
template <wwr::extension::MemoryKind K>
[[nodiscard]] constexpr Footprint<K> then(const Footprint<K> &a, const Footprint<K> &b) {
  const size_t residue = a.scratch > b.persistent ? a.scratch - b.persistent : 0;
  return {.persistent = a.persistent + b.persistent,
          .scratch = std::max(residue, b.scratch),
          .workspace = std::max(a.workspace, b.workspace)};
}

/// @brief What @p f returned is freed before the enclosing routine returns:
///        its persistent bytes become scratch, and the peak is unchanged.
template <wwr::extension::MemoryKind K>
[[nodiscard]] constexpr Footprint<K> drop(const Footprint<K> &f) {
  return {.persistent = 0, .scratch = f.persistent + f.scratch, .workspace = f.workspace};
}

/// @brief @ref then for the device and pinned footprints alike.
[[nodiscard]] constexpr MemoryFootprint then(const MemoryFootprint &a, const MemoryFootprint &b) {
  return {.device = then(a.device, b.device), .pinned = then(a.pinned, b.pinned)};
}

/// @brief @ref drop for the device and pinned footprints alike.
[[nodiscard]] constexpr MemoryFootprint drop(const MemoryFootprint &f) {
  return {.device = drop(f.device), .pinned = drop(f.pinned)};
}

} // namespace free_breakfast
