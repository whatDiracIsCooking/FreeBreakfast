/// @file
/// @brief Device: one device's context -- its properties, a ForkStream as its
/// work stream, and a memory pool -- plus the MemPool wrapper it owns

module;

// WWR_SELECTED_HIP, for the backend branches in this tree's host code here:
// the ROCm workaround in pool_props(), and stack_growth(), which is 0 under
// ROCm. Macros only, so safe in an interface's GMF.
#include "selected_backend.h"

export module free_breakfast.core.wwr:device;

import std;
import :policy;      // CreatePolicy / DestroyPolicy
import :fork_stream; // ForkStream, Stream
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import wwr.runtime_api;
import wwr.extension.runtime;

export namespace free_breakfast {

/// @brief A memory pool on one device.
using MemPool =
    wwr::extension::MemPoolWrapper<CreatePolicy<wwr::wwrError_t>, DestroyPolicy<wwr::wwrError_t>,
                                   CreatePolicy<wwr::wwrError_t>>;

/// @brief A memory pool's usage, in bytes: what its live allocations hold
/// (`used_*`) and what it holds from the driver (`reserved_*`, the used bytes
/// plus freed memory it has cached), now and at their high-water marks.
struct PoolUsage {
  uint64_t used_current = 0;
  uint64_t used_high = 0;
  uint64_t reserved_current = 0;
  uint64_t reserved_high = 0;
};

/// @brief One device's context: its properties, an owned ForkStream as the work
/// stream, and an owned memory pool.
///
/// In place of wwr's kit::DeviceHandle, whose stream is a plain Stream: the
/// work stream here is a ForkStream, so everything queued on stream() --
/// allocations, BLAS and solver calls -- can be forked from, and joined back
/// into, a parent stream, including inside a graph capture (fork_stream()).
///
/// Exposes dev_idx(), stream() and pool(), so it satisfies wwr's
/// device_handle_pool: Buffers, BlasHandle and SolverHandle bind to it as they
/// did to kit::DeviceHandle.
///
/// Hold it as a std::shared_ptr -- buffers and library handles retain their
/// device so its stream and pool outlive them. Movable, not copyable; a
/// moved-from Device owns nothing and must not be used.
class Device : private NonCopyable {
public:
  /// @brief Default release threshold of the memory pool: 1 GiB.
  ///
  /// The bytes of freed memory the pool keeps cached across stream syncs
  /// instead of returning to the driver -- not a cap on allocation. wwr's own
  /// default is 1 MiB, which hands nearly every free back at the next sync and
  /// makes the following allocation pay the driver round trip again.
  static constexpr uint64_t kDefaultPoolReleaseThreshold = uint64_t{1} << 30;

  /// @brief Query `dev_idx`'s properties, create its memory pool, and a
  /// ForkStream on it with the default configuration (non-blocking, sync-only
  /// events). A failure aborts (CreatePolicy).
  ///
  /// `pool_cap` is the pool's hard limit in bytes (`maxSize`): an allocation
  /// that would take the pool past it fails. A nonzero cap is also the pool's
  /// release threshold, in place of `pool_release_threshold`, so memory one
  /// step frees stays in the pool for the next instead of going back to the
  /// driver, where another process could take it mid-run. 0, the default, is
  /// no cap and a release threshold of `pool_release_threshold` bytes.
  explicit Device(int32_t dev_idx = 0,
                  uint64_t pool_release_threshold = kDefaultPoolReleaseThreshold,
                  uint64_t pool_cap = 0,
                  std::source_location location = std::source_location::current())
      : props_(query_props(dev_idx, location)),
        pool_(pool_props(dev_idx, pool_cap), pool_cap != 0 ? pool_cap : pool_release_threshold,
              location),
        fork_(dev_idx, location) {}

  /// @brief Index of the device the stream and pool live on (-1 once moved from).
  int32_t dev_idx() const noexcept { return fork_.dev_idx(); }

  /// @brief This device's static properties, queried once at construction.
  const wwr::wwrDeviceProp &props() const noexcept { return props_; }

  /// @brief The work stream: fork_stream().stream().
  Stream &stream() noexcept { return fork_.stream(); }
  /// @copydoc stream()
  const Stream &stream() const noexcept { return fork_.stream(); }

  /// @brief The memory pool buffers allocate from, on stream().
  MemPool &pool() noexcept { return pool_; }
  /// @copydoc pool()
  const MemPool &pool() const noexcept { return pool_; }

  /// @brief The pool's used and reserved bytes, current and high-water. A
  /// failed query aborts (CreatePolicy).
  ///
  /// A read, not a sync: it counts the allocations and frees that have been
  /// queued so far, and memory a free returns to the driver leaves
  /// `reserved_current` only once the stream has synchronized past it.
  PoolUsage pool_usage(std::source_location location = std::source_location::current()) const {
    const uint64_t used = pool_attribute(wwr::wwrMemPoolAttrUsedMemCurrent, location);
    const uint64_t reserved = pool_attribute(wwr::wwrMemPoolAttrReservedMemCurrent, location);
    // A high-water mark is never below the current value. CUDA keeps it so;
    // ROCm (7.2.4) resets it to 0 rather than to the current value, so after
    // a reset it can read below the current value; it is floored here.
    return {.used_current = used,
            .used_high = std::max(used, pool_attribute(wwr::wwrMemPoolAttrUsedMemHigh, location)),
            .reserved_current = reserved,
            .reserved_high =
                std::max(reserved, pool_attribute(wwr::wwrMemPoolAttrReservedMemHigh, location))};
  }

  /// @brief Reset both high-water marks (`used_high`, `reserved_high`) to
  /// their current values. A failure aborts (CreatePolicy).
  void reset_pool_high_water(std::source_location location = std::source_location::current()) {
    // CUDA accepts only 0 here, and reads it as "reset to current"; ROCm
    // stores the 0, which pool_usage() floors to the current value.
    for (const auto attr : {wwr::wwrMemPoolAttrUsedMemHigh, wwr::wwrMemPoolAttrReservedMemHigh}) {
      uint64_t zero = 0;
      CreatePolicy<wwr::wwrError_t>{}.handle_error(wwr::wwrMemPoolSetAttribute(pool_, attr, &zero),
                                                   location);
    }
  }

  /// @brief The device memory, in bytes, the runtime would still take outside
  /// any pool to launch a kernel of `per_thread` bytes of stack: 0 when the
  /// stack it already holds is that large. A failed query aborts
  /// (CreatePolicy).
  ///
  /// CUDA holds the stack limit (`wwrLimitStackSize`) for every thread the
  /// device can keep resident, SMs x maxThreadsPerMultiProcessor (104,448 on
  /// the 3080), and raises the limit at the first launch of a kernel that
  /// needs more, keeping it: so this is the limit's shortfall times that
  /// thread count. ROCm (7.2.4, gfx1200) sizes scratch per dispatch and
  /// neither reserves it device-wide nor raises the limit, so it is 0 there.
  /// See README.md, "Kernel stacks outside the pool".
  uint64_t stack_growth(size_t per_thread,
                        std::source_location location = std::source_location::current()) const {
#if defined(WWR_SELECTED_HIP)
    (void)per_thread;
    (void)location;
    return 0;
#else
    // The limit is the current device's: this one's, for the call.
    const wwr::extension::ScopedDeviceIndex<CreatePolicy<wwr::wwrError_t>> on{dev_idx()};
    size_t limit = 0;
    CreatePolicy<wwr::wwrError_t>{}.handle_error(
        wwr::wwrDeviceGetLimit(&limit, wwr::wwrLimitStackSize), location);
    const uint64_t threads = uint64_t{static_cast<uint32_t>(props_.multiProcessorCount)} *
                             uint64_t{static_cast<uint32_t>(props_.maxThreadsPerMultiProcessor)};
    return per_thread > limit ? uint64_t{per_thread - limit} * threads : 0;
#endif
  }

  /// @brief The ForkStream that owns stream(), for forking it from, and joining
  /// it back into, a parent stream.
  ForkStream &fork_stream() noexcept { return fork_; }
  /// @copydoc fork_stream()
  const ForkStream &fork_stream() const noexcept { return fork_; }

private:
  // wwr's default pool properties with a maxSize of `cap` (0: none), except
  // that under HIP the pool is created
  // exportable (a POSIX file descriptor handle type), which it never uses.
  //
  // A workaround for a ROCm defect, seen on 7.2.4 (#300). ROCm backs a
  // non-exportable pool with a virtual-memory heap (DEBUG_HIP_MEM_POOL_VMHEAP,
  // on by default), which registers each physical chunk it maps in MemObjMap,
  // beside the allocations inside it. After a pool has served many
  // differently sized allocations, hipMemPoolDestroy can look a chunk up,
  // find nothing, and dereference the null (VmHeap::UncommitMemory,
  // vmheap.cpp:86): an intermittent SIGSEGV as the Device is destroyed. An
  // exportable pool is never given the heap. ROCm/rocm-systems 944b5d3c6d
  // (#12671) stops registering the chunks; no release carries it yet. Drop
  // this once the image's ROCm does.
  static wwr::wwrMemPoolProps pool_props(int32_t dev_idx, uint64_t cap) {
    wwr::wwrMemPoolProps props{};
    props.allocType = wwr::wwrMemAllocationTypePinned;
#if defined(WWR_SELECTED_HIP)
    props.handleTypes = wwr::wwrMemHandleTypePosixFileDescriptor;
#else
    props.handleTypes = wwr::wwrMemHandleTypeNone;
#endif
    props.location.type = wwr::wwrMemLocationTypeDevice;
    props.location.id = dev_idx;
    props.maxSize = static_cast<std::size_t>(cap);
    return props;
  }

  // One 64-bit pool attribute. uint64_t, not size_t: the usage attributes are
  // 64-bit on both backends (cuuint64_t / uint64_t).
  uint64_t pool_attribute(wwr::wwrMemPoolAttr attr, std::source_location location) const {
    uint64_t value = 0;
    CreatePolicy<wwr::wwrError_t>{}.handle_error(wwr::wwrMemPoolGetAttribute(pool_, attr, &value),
                                                 location);
    return value;
  }

  static wwr::wwrDeviceProp query_props(int32_t dev_idx, std::source_location location) {
    wwr::wwrDeviceProp prop{};
    CreatePolicy<wwr::wwrError_t>{}.handle_error(wwr::wwrGetDeviceProperties(&prop, dev_idx),
                                                 location);
    return prop;
  }

  wwr::wwrDeviceProp props_{};
  // Declared before fork_, so destroyed after it: Stream synchronizes before
  // destroying, so any free still queued on the stream has run by the time the
  // pool it returns memory to is destroyed.
  MemPool pool_;
  ForkStream fork_;
};

} // namespace free_breakfast
