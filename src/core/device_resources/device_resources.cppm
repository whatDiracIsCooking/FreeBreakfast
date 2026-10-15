/// @file
/// @brief DeviceResources: one device's context and the library handles bound
/// to its stream

export module free_breakfast.core.device_resources;

import std;
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.wwr;

export namespace free_breakfast {

/// @brief One device's working set: a shared Device (stream + memory pool) and
/// the BLAS and dense-solver handles bound to that Device's stream.
///
/// The handles anchor to the Device, never to this class: a handle member
/// holding a shared_ptr back to its own owner would be a reference cycle (wwr
/// docs/architecture.md section 18). Buffers draw from device(), so allocation,
/// library calls and frees are all ordered on the one stream.
///
/// Movable, not copyable. A moved-from DeviceResources holds no Device.
class DeviceResources : private NonCopyable {
public:
  /// @brief Create a Device on `dev_idx` and bind a BLAS and a solver handle to
  /// its stream. A failed creation aborts (CreatePolicy).
  explicit DeviceResources(int32_t dev_idx = 0)
      : device_(std::make_shared<Device>(dev_idx)), blas_(device_), solver_(device_) {}

  /// @brief The shared Device; pass it to Buffers to allocate from its pool.
  const std::shared_ptr<Device> &device() const noexcept { return device_; }
  /// @brief Index of the device every resource here lives on.
  int32_t dev_idx() const noexcept { return device_->dev_idx(); }
  /// @brief The one work stream: allocations, BLAS and solver calls all queue here.
  wwr::wwrStream_t stream() const noexcept { return device_->stream(); }

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
