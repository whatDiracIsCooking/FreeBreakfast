/// @file
/// @brief DeviceResources: one device's context, the library handles bound to
/// its stream, and the solver scratch they share

export module free_breakfast.core.resources:device_resources;

import std;
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.primitives;
import free_breakfast.core.wwr;

export namespace free_breakfast {

/// @brief One device's working set: a shared Device (ForkStream + memory pool),
/// the BLAS and dense-solver handles bound to that Device's stream, and the
/// solver scratch those calls share: a byte workspace on the device and one on
/// the host, and a device info flag with a pinned slot to read it back into.
///
/// The handles anchor to the Device, never to this class: a handle member
/// holding a shared_ptr back to its own owner would be a reference cycle (wwr
/// docs/architecture.md section 18). Buffers draw from device(), so allocation,
/// library calls and frees are all ordered on the one stream.
///
/// **One driving thread at a time.** The BLAS and solver handles, the stream
/// and the ForkStream's events are not safe to use from two threads at once
/// (cuBLAS / hipBLAS forbid concurrent use of one handle), and nothing here
/// locks. Whichever thread drives an instance must also have its device
/// current: open a ScopedDeviceIndex on dev_idx(), or set it once on a thread
/// that only ever drives this device. Library calls and kernel launches do not
/// select it; only buffer allocation and free do.
///
/// **The workspaces only grow.** device_workspace(bytes) and
/// host_workspace(bytes) reallocate when asked for more than they hold, rounded
/// up to kWorkspaceAlignment, and never shrink. A growth frees the old block, so
/// take the buffer -- or a reinterpret_buffer_view of it -- just before the call
/// that uses it, and never hold its pointer across another request. Pass the
/// routine its own lwork, not the buffer's size, which may be larger.
///
/// **One routine owns the device workspace at a time.** A routine may carve all
/// of its device scratch from it -- a solver's work and its own intermediates
/// alike -- by asking once for the total and splitting the result. So nothing
/// nests: a routine that uses it calls no other that does, and no argument to
/// one may point into it.
///
/// Movable, not copyable. A moved-from DeviceResources holds no Device.
class DeviceResources : private NonCopyable {
public:
  /// @brief Default release threshold of the Device's memory pool: 1 GiB
  /// (Device::kDefaultPoolReleaseThreshold).
  static constexpr uint64_t kDefaultPoolReleaseThreshold = Device::kDefaultPoolReleaseThreshold;

  /// @brief The granularity the workspaces grow in, in bytes: the pool's own
  /// allocation alignment, and a multiple of every solver element type's size,
  /// so a reinterpret_buffer_view of a whole workspace as any of them succeeds.
  static constexpr size_t kWorkspaceAlignment = 256;

  /// @brief Create a Device on `dev_idx` whose pool keeps up to
  /// `pool_release_threshold` bytes cached, and bind a BLAS and a solver handle
  /// to its stream. A failed creation aborts (CreatePolicy).
  explicit DeviceResources(int32_t dev_idx = 0,
                           uint64_t pool_release_threshold = kDefaultPoolReleaseThreshold)
      : device_(std::make_shared<Device>(dev_idx, pool_release_threshold)), blas_(device_),
        solver_(device_), device_workspace_(0, device_), host_workspace_(0), info_(1, device_),
        info_host_(1) {}

  /// @brief The shared Device; pass it to Buffers to allocate from its pool.
  const std::shared_ptr<Device> &device() const noexcept { return device_; }
  /// @brief Index of the device every resource here lives on.
  int32_t dev_idx() const noexcept { return device_->dev_idx(); }
  /// @brief The one work stream: allocations, BLAS and solver calls all queue here.
  wwr::wwrStream_t stream() const noexcept { return device_->stream(); }
  /// @brief The ForkStream that owns stream(): fork it from a parent stream to
  /// order -- or graph-capture -- everything queued here after the parent's work.
  ForkStream &fork_stream() noexcept { return device_->fork_stream(); }
  /// @copydoc fork_stream()
  const ForkStream &fork_stream() const noexcept { return device_->fork_stream(); }

  /// @brief The BLAS handle bound to stream().
  BlasHandle &blas() noexcept { return blas_; }
  /// @copydoc blas()
  const BlasHandle &blas() const noexcept { return blas_; }

  /// @brief The dense-solver handle bound to stream().
  SolverHandle &solver() noexcept { return solver_; }
  /// @copydoc solver()
  const SolverHandle &solver() const noexcept { return solver_; }

  /// @brief The scratch device workspace, grown to hold at least `bytes`.
  ///
  /// A growth allocates align_up(bytes, kWorkspaceAlignment) from device()'s
  /// pool on stream(), and the old block's free is ordered on stream() after
  /// the work already queued against it. View it as a solver's element type
  /// with reinterpret_buffer_view<T>. A failed allocation aborts (CreatePolicy).
  Buffers::device<std::byte> &device_workspace(size_t bytes) {
    if (bytes > device_workspace_.num_elements()) {
      device_workspace_ =
          Buffers::device<std::byte>(math::align_up(bytes, kWorkspaceAlignment), device_);
    }
    return device_workspace_;
  }

  /// @brief The scratch host workspace (a solver's bufferOnHost), grown to hold
  /// at least `bytes`.
  ///
  /// Plain host memory, not pinned: a solver reads and writes it from the CPU
  /// and never copies it to or from the device, so pinning would buy nothing
  /// and cost a page-locking allocation on every growth. A growth frees the old
  /// block at once, not on stream(), so grow it only once the calls that used
  /// it have returned. A failed allocation aborts (CreatePolicy).
  Buffers::host<std::byte> &host_workspace(size_t bytes) {
    if (bytes > host_workspace_.num_elements()) {
      host_workspace_ = Buffers::host<std::byte>(math::align_up(bytes, kWorkspaceAlignment));
    }
    return host_workspace_;
  }

  /// @brief One device int32_t for a solver's devInfo, allocated with the
  /// handles. Read it back with fetch_info().
  Buffers::device<int32_t> &info() noexcept { return info_; }

  /// @brief Copy info() into pinned host memory and return it: the d2h copy is
  /// queued on stream(), after the solver call that wrote it, and stream() is
  /// then synchronized. A GpuRuntime error if either call fails.
  [[nodiscard]] Result<int32_t> fetch_info() {
    if (auto status = gpu_status(wwr::extension::copy(info_host_, info_, stream())); !status) {
      return std::unexpected(std::move(status).error());
    }
    if (auto status = gpu_status(wwr::extension::sync(stream())); !status) {
      return std::unexpected(std::move(status).error());
    }
    return info_host_.data()[0];
  }

private:
  // Declared first: blas_, solver_ and the device buffers are constructed from
  // it.
  std::shared_ptr<Device> device_;
  BlasHandle blas_;
  SolverHandle solver_;
  Buffers::device<std::byte> device_workspace_;
  Buffers::host<std::byte> host_workspace_;
  Buffers::device<int32_t> info_;
  Buffers::pinned<int32_t> info_host_;
};

} // namespace free_breakfast
