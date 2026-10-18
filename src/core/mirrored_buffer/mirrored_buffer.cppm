/// @file
/// @brief MirroredBuffer: a device buffer paired with a pinned host buffer of
/// the same size, and the copies between them on the device's work stream

export module free_breakfast.core.mirrored_buffer;

import std;
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.wwr;

export namespace free_breakfast {

/// @brief A `Buffers::device<T>` and a `Buffers::pinned<T>` of equal size, with
/// the h2d and d2h copies between them.
///
/// Both sides are allocated once, at construction, with the same element count,
/// and never resized, so the copies need no bounds of their own. Every copy is
/// queued on the Device's work stream -- the stream the device side was
/// allocated and zeroed on -- so a to_device() straight after construction is
/// ordered after the zero-fill.
///
/// to_pinned() / to_device() only enqueue: the pinned side is not safe to read
/// after a to_pinned(), or to overwrite after a to_device(), until the stream is
/// synchronized. The `_sync` variants enqueue and then synchronize the stream.
/// Every copy returns the runtime's error code, as ForkStream's operations do,
/// so the caller picks the tier: FBF_GPU_MUST or gpu_status.
///
/// ```cpp
/// MirroredBuffer<double> m{n, device};
/// std::ranges::copy(values, m.pinned().data());
/// FBF_GPU_MUST(m.to_device());          // enqueue h2d on device->stream()
/// ... launch on m.device() ...
/// FBF_GPU_MUST(m.to_pinned_sync());     // d2h, then wait for it
/// ```
///
/// Destruction synchronizes the stream first, so a copy still in flight cannot
/// read or write the pinned side after it has been freed. Movable, not
/// copyable; a moved-from MirroredBuffer owns nothing and must not be used.
template <typename T>
class MirroredBuffer : private NonCopyable {
public:
  /// @brief Allocate `num_elements` on `device`'s pool and as pinned host
  /// memory. Both sides start zeroed. A failure aborts (CreatePolicy).
  MirroredBuffer(size_t num_elements, std::shared_ptr<Device> device,
                 std::source_location location = std::source_location::current())
      : device_(std::move(device)), device_buf_(num_elements, device_, location),
        pinned_(num_elements, location) {}

  MirroredBuffer(MirroredBuffer &&) noexcept = default;

  MirroredBuffer &operator=(MirroredBuffer &&other) noexcept {
    if (this != &other) {
      // The buffers' move-assignment frees what they held, so this side's
      // copies in flight need the same wait as in the destructor.
      sync_before_release();
      device_buf_ = std::move(other.device_buf_);
      pinned_ = std::move(other.pinned_);
      device_ = std::move(other.device_);
    }
    return *this;
  }

  ~MirroredBuffer() { sync_before_release(); }

  /// @brief Number of elements on each side.
  size_t size() const noexcept { return device_buf_.num_elements(); }

  /// @brief The device side.
  Buffers::device<T> &device() noexcept { return device_buf_; }
  /// @copydoc device()
  const Buffers::device<T> &device() const noexcept { return device_buf_; }

  /// @brief The pinned host side.
  Buffers::pinned<T> &pinned() noexcept { return pinned_; }
  /// @copydoc pinned()
  const Buffers::pinned<T> &pinned() const noexcept { return pinned_; }

  /// @brief Enqueue a device-to-pinned copy of the whole buffer on the work stream.
  [[nodiscard]] wwr::wwrError_t to_pinned() noexcept {
    return wwr::extension::copy(pinned_, device_buf_, device_->stream());
  }

  /// @brief Enqueue a pinned-to-device copy of the whole buffer on the work stream.
  [[nodiscard]] wwr::wwrError_t to_device() noexcept {
    return wwr::extension::copy(device_buf_, pinned_, device_->stream());
  }

  /// @brief to_pinned(), then block until the work stream -- and so the copy --
  /// has completed: the pinned side is then safe to read.
  [[nodiscard]] wwr::wwrError_t to_pinned_sync() noexcept {
    if (const auto code = to_pinned(); code != wwr::wwrSuccess) {
      return code;
    }
    return wwr::extension::sync(device_->stream());
  }

  /// @brief to_device(), then block until the work stream -- and so the copy --
  /// has completed: the pinned side is then safe to overwrite.
  [[nodiscard]] wwr::wwrError_t to_device_sync() noexcept {
    if (const auto code = to_device(); code != wwr::wwrSuccess) {
      return code;
    }
    return wwr::extension::sync(device_->stream());
  }

private:
  // Wait out any copy still queued against the pinned side before it is freed.
  // The device side needs no such wait -- its free is stream-ordered -- but the
  // pinned free is not, and freeing under an in-flight DMA is a use-after-free.
  // Skipped when moved from. A failure goes to DestroyPolicy, as on every
  // destroy path.
  void sync_before_release() noexcept {
    if (device_) {
      DestroyPolicy<wwr::wwrError_t>{}.handle_error(wwr::extension::sync(device_->stream()),
                                                    std::source_location::current());
    }
  }

  // Declared first: device_buf_ is constructed from it.
  std::shared_ptr<Device> device_;
  Buffers::device<T> device_buf_;
  Buffers::pinned<T> pinned_;
};

} // namespace free_breakfast
