/// @file
/// @brief Device: one device's context -- its properties, a ForkStream as its
/// work stream, and a memory pool -- plus the MemPool wrapper it owns

module;

// WWR_SELECTED_HIP, for the one backend branch in this tree's host code: the
// ROCm workaround in pool_props(). Macros only, so safe in an interface's GMF.
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

  /// @brief Query `dev_idx`'s properties, create its memory pool with a release
  /// threshold of `pool_release_threshold` bytes, and a ForkStream on it with
  /// the default configuration (non-blocking, sync-only events). A failure
  /// aborts (CreatePolicy).
  explicit Device(int32_t dev_idx = 0,
                  uint64_t pool_release_threshold = kDefaultPoolReleaseThreshold,
                  std::source_location location = std::source_location::current())
      : props_(query_props(dev_idx, location)),
        pool_(pool_props(dev_idx), pool_release_threshold, location), fork_(dev_idx, location) {}

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

  /// @brief The ForkStream that owns stream(), for forking it from, and joining
  /// it back into, a parent stream.
  ForkStream &fork_stream() noexcept { return fork_; }
  /// @copydoc fork_stream()
  const ForkStream &fork_stream() const noexcept { return fork_; }

private:
  // wwr's default pool properties, except that under HIP the pool is created
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
  static wwr::wwrMemPoolProps pool_props(int32_t dev_idx) {
    wwr::wwrMemPoolProps props{};
    props.allocType = wwr::wwrMemAllocationTypePinned;
#if defined(WWR_SELECTED_HIP)
    props.handleTypes = wwr::wwrMemHandleTypePosixFileDescriptor;
#else
    props.handleTypes = wwr::wwrMemHandleTypeNone;
#endif
    props.location.type = wwr::wwrMemLocationTypeDevice;
    props.location.id = dev_idx;
    return props;
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
