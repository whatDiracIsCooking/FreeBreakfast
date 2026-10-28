/// @file
/// @brief OutputRing: device arrays written to .npy files through a fixed ring
/// of pinned staging slots, the copies on the device's stream and the file
/// writes on a taskflow executor, so several files are written at once.
///
/// The pinned memory is allocated once, at construction, and held for the
/// ring's life: page-locking is a driver call whose cost grows with the size,
/// and freeing pinned memory synchronizes the device, so a per-write staging
/// buffer would pay both on every dump. An array of any size streams through
/// the ring in slot-sized chunks.

export module free_breakfast.experimental.output_ring;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;

export namespace free_breakfast {

/// @brief How an OutputRing is sized.
struct OutputRingConfig {
  /// @brief Slots in the ring: how many chunks can be in flight at once.
  size_t num_slots = 4;
  /// @brief Bytes per slot, rounded down to whole doubles. 0 aborts.
  size_t slot_bytes = size_t{32} << 20;
};

/// @brief A fixed ring of pinned slots, each with an Event, that streams
/// device arrays into .npy files.
///
/// write_npy() runs on the thread that drives @p res. It writes the header,
/// then for each chunk takes a free slot, queues the device-to-host copy into
/// it on res.stream(), records the slot's event, and hands the slot to a task
/// on the executor. That task waits on the event, writes the chunk at its own
/// offset in the file, and frees the slot. Chunks carry their offsets, so they
/// land in any order, and chunks of different files share the slots: two
/// write_npy() calls back to back are written concurrently, the second's
/// copies queued while the first's chunks are still being written.
///
/// write_npy() returns once its last chunk is queued, not written. wait()
/// blocks until every queued chunk is on disk and returns the first error any
/// of them hit since the last wait(). The source buffer must stay unchanged
/// until its copies have run, which is ordered on res.stream(): a later write
/// or free of it on that stream is safe, one on another stream is not.
///
/// **Who touches a slot.** The driving thread takes a slot, queues its copy and
/// records its event; the writer task waits on that event and frees the slot;
/// only then can the driving thread take it again. No event is used from two
/// threads at once, so DeviceResources' one-driving-thread rule still holds.
///
/// **Blocking inside the executor.** Taking a slot, and wait(), block until a
/// writer task frees one. Called from one of the executor's own workers, as a
/// pipeline step is, they run other tasks while they wait (corun_until), so
/// the writers make progress even with a single worker.
///
/// Neither copyable nor movable: the writer tasks hold its address. The
/// destructor waits for them, and logs rather than returns their error.
class OutputRing : private NonTransferable {
public:
  /// @brief Allocate the slots' pinned memory and create their events on
  /// @p dev_idx, which is left the current device. A failure aborts
  /// (CreatePolicy).
  OutputRing(int32_t dev_idx, tf::Executor &executor, const OutputRingConfig &cfg = {});

  ~OutputRing();

  /// @brief Queue @p device, an array of @p shape in column-major order, to be
  /// written to @p path as .npy.
  ///
  /// A shape whose element count is not device.size() yields
  /// ErrorKind::InvalidConfig, and a header that cannot be written
  /// ErrorKind::IO, before anything is queued. A failed copy is
  /// ErrorKind::GpuRuntime; chunks queued before it are still written.
  [[nodiscard]] Status write_npy(DeviceResources &res, const std::filesystem::path &path,
                                 std::span<const size_t> shape, std::span<const double> device);

  /// @brief Block until every queued chunk is written; the first error any of
  /// them hit since the last wait(), which this clears.
  [[nodiscard]] Status wait();

  /// @brief Doubles per slot.
  [[nodiscard]] size_t slot_elements() const noexcept { return slot_elements_; }

private:
  struct Slot {
    Buffers::pinned<double> host;
    Event ready;
  };

  // A free slot's index, blocking (or co-running the executor) until one is.
  [[nodiscard]] size_t acquire();
  // Return @p index to the free list, keeping @p status if it is the first error.
  void release(size_t index, Status status);

  tf::Executor &executor_;
  size_t slot_elements_;
  std::vector<Slot> slots_;
  std::counting_semaphore<> free_count_;
  std::mutex mutex_;
  std::vector<size_t> free_; // guarded by mutex_
  Status first_error_;       // guarded by mutex_
};

} // namespace free_breakfast
