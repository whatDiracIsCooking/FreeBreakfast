/// @file
/// @brief DeviceResources: one device's context and the library handles bound
/// to its stream

export module free_breakfast.core.resources:device_resources;

import std;
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.wwr;

export namespace free_breakfast {

/// @brief One device's working set: a shared Device (ForkStream + memory pool) and
/// the BLAS and dense-solver handles bound to that Device's stream.
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
/// Movable, not copyable. A moved-from DeviceResources holds no Device.
class DeviceResources : private NonCopyable {
public:
  /// @brief Default release threshold of the Device's memory pool: 1 GiB
  /// (Device::kDefaultPoolReleaseThreshold).
  static constexpr uint64_t kDefaultPoolReleaseThreshold = Device::kDefaultPoolReleaseThreshold;

  /// @brief Create a Device on `dev_idx` whose pool keeps up to
  /// `pool_release_threshold` bytes cached, and bind a BLAS and a solver handle
  /// to its stream. A failed creation aborts (CreatePolicy).
  explicit DeviceResources(int32_t dev_idx = 0,
                           uint64_t pool_release_threshold = kDefaultPoolReleaseThreshold)
      : device_(std::make_shared<Device>(dev_idx, pool_release_threshold)), blas_(device_),
        solver_(device_) {}

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

private:
  // Declared first: blas_ and solver_ are constructed from it.
  std::shared_ptr<Device> device_;
  BlasHandle blas_;
  SolverHandle solver_;
};

} // namespace free_breakfast
