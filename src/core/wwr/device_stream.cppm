/// @file
/// @brief DeviceStream: one more work stream on a Device, allocating from that
/// Device's pool

export module free_breakfast.core.wwr:device_stream;

import std;
import :fork_stream; // ForkStream, Stream
import :device;      // Device, MemPool
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;

export namespace free_breakfast {

/// @brief A Device's pool paired with a ForkStream of its own: a second (third,
/// ...) work stream on the same device.
///
/// Exposes dev_idx(), stream() and pool(), so it satisfies wwr's
/// device_handle_pool, as Device does: a buffer bound to it (StreamBuffers)
/// draws from the Device's pool, ordered on this stream. A bare ForkStream would
/// satisfy only device_handle_stream, and a buffer bound to it would quietly
/// allocate from the device's default pool instead, losing the Device's release
/// threshold.
///
/// The Device never learns about its DeviceStreams: each one retains its Device,
/// so any number can be made from one, and the pool outlives every stream that
/// allocates from it. Work that has to meet the Device's own stream, or another
/// DeviceStream's, crosses over by fork/join (fork_stream()).
///
/// ```cpp
/// auto dev = std::make_shared<Device>(0);
/// auto a = std::make_shared<DeviceStream>(dev);
/// auto b = std::make_shared<DeviceStream>(dev);   // independent of a
/// StreamBuffers::device<double> x(n, a);          // dev's pool, on a's stream
/// ```
///
/// Hold it as a std::shared_ptr, as a Device is held -- buffers retain their
/// handle. Movable, not copyable; a moved-from DeviceStream owns nothing and
/// must not be used.
class DeviceStream : private NonCopyable {
public:
  /// @brief A ForkStream with the default configuration (non-blocking,
  /// sync-only events) on `device`'s device, allocating from `device`'s pool.
  /// A failure aborts (CreatePolicy).
  explicit DeviceStream(std::shared_ptr<Device> device,
                        std::source_location location = std::source_location::current())
      : device_(std::move(device)), fork_(device_->dev_idx(), location) {}

  /// @brief Index of the device the stream lives on (-1 once moved from).
  int32_t dev_idx() const noexcept { return fork_.dev_idx(); }

  /// @brief This handle's work stream: fork_stream().stream().
  Stream &stream() noexcept { return fork_.stream(); }
  /// @copydoc stream()
  const Stream &stream() const noexcept { return fork_.stream(); }

  /// @brief The Device's memory pool, shared with the Device and every other
  /// DeviceStream made from it.
  MemPool &pool() noexcept { return device_->pool(); }
  /// @copydoc pool()
  const MemPool &pool() const noexcept { return device_->pool(); }

  /// @brief The ForkStream that owns stream(), for forking it from, and joining
  /// it back into, a parent stream.
  ForkStream &fork_stream() noexcept { return fork_; }
  /// @copydoc fork_stream()
  const ForkStream &fork_stream() const noexcept { return fork_; }

  /// @brief The Device this stream was made from.
  const std::shared_ptr<Device> &device() const noexcept { return device_; }

private:
  // Declared before fork_, so released after it: Stream synchronizes before
  // destroying, so any free still queued on this stream has returned its block
  // to the pool before this handle can drop the last reference to the Device.
  std::shared_ptr<Device> device_;
  ForkStream fork_;
};

} // namespace free_breakfast
