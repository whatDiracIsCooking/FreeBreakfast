/// @file
/// @brief OutputRing's slots, the chunked copy loop and the writer tasks.

module free_breakfast.experimental.output_ring;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.io.npy;

#include "diagnostics.h"

namespace free_breakfast {

OutputRing::OutputRing(int32_t dev_idx, tf::Executor &executor, const OutputRingConfig &cfg)
    : executor_(executor), slot_elements_(cfg.slot_bytes / sizeof(double)),
      free_count_(static_cast<std::ptrdiff_t>(cfg.num_slots)) {
  if (cfg.num_slots == 0 || slot_elements_ == 0) {
    FBF_LOG_CRITICAL("OutputRing: {} slots of {} bytes hold nothing", cfg.num_slots,
                     cfg.slot_bytes);
    std::abort();
  }
  slots_.reserve(cfg.num_slots);
  free_.reserve(cfg.num_slots);
  for (size_t i = 0; i < cfg.num_slots; ++i) {
    slots_.push_back(Slot{.host = Buffers::pinned<double>(slot_elements_),
                          .ready = Event(dev_idx, wwr::wwrEventDisableTiming)});
    free_.push_back(i);
  }
}

OutputRing::~OutputRing() {
  if (const Status status = wait(); !status) {
    FBF_LOG_ERROR("OutputRing: a write failed and was never waited for: {}",
                  status.error().message);
  }
}

[[nodiscard]] size_t OutputRing::acquire() {
  if (executor_.this_worker_id() >= 0) {
    executor_.corun_until([this] { return free_count_.try_acquire(); });
  } else {
    free_count_.acquire();
  }
  const std::scoped_lock lock{mutex_};
  const size_t index = free_.back();
  free_.pop_back();
  return index;
}

void OutputRing::release(size_t index, Status status) {
  {
    const std::scoped_lock lock{mutex_};
    if (!status && first_error_) {
      first_error_ = std::move(status);
    }
    free_.push_back(index);
  }
  free_count_.release();
}

// @p bytes at @p offset in @p path, which already exists. Each writer opens its
// own stream, so chunks of one file land concurrently at disjoint offsets.
[[nodiscard]] Status write_at(const std::filesystem::path &path, size_t offset, const double *data,
                              size_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::in | std::ios::out);
  if (!out) {
    return err_io("cannot open {} to write a chunk", path.string());
  }
  out.seekp(static_cast<std::streamoff>(offset));
  out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
  if (!out) {
    return err_io("failed writing {} bytes at offset {} of {}", bytes, offset, path.string());
  }
  return {};
}

[[nodiscard]] Status OutputRing::write_npy(DeviceResources &res, const std::filesystem::path &path,
                                           std::span<const size_t> shape,
                                           std::span<const double> device) {
  const size_t count = FBF_TRY(element_count(shape));
  if (count != device.size()) {
    return err_config("OutputRing: shape describes {} elements but the buffer holds {}", count,
                      device.size());
  }
  // Written here, truncating, so the file exists before any chunk opens it.
  const std::string header = FBF_TRY(npy_header(shape));
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!out) {
      return err_io("cannot write the .npy header of {}", path.string());
    }
  }

  for (size_t begin = 0; begin < count; begin += slot_elements_) {
    const size_t n = std::min(slot_elements_, count - begin);
    const size_t index = acquire();
    Slot &slot = slots_[index];
    if (const Status copied = gpu_status(
            wwr::wwrMemcpyAsync(slot.host.data(), device.data() + begin, n * sizeof(double),
                                wwr::wwrMemcpyDeviceToHost, res.stream()));
        !copied) {
      release(index, {});
      return copied;
    }
    if (const Status recorded = gpu_status(wwr::extension::record(slot.ready, res.stream()));
        !recorded) {
      // The copy is queued but nothing marks its end: wait it out on the
      // stream before the slot can be handed out again.
      static_cast<void>(res.sync());
      release(index, {});
      return recorded;
    }
    executor_.silent_async([this, index, path, offset = header.size() + begin * sizeof(double),
                            bytes = n * sizeof(double)] {
      Slot &s = slots_[index];
      Status status = gpu_status(wwr::extension::sync(s.ready));
      if (status) {
        status = write_at(path, offset, s.host.data(), bytes);
      }
      release(index, std::move(status));
    });
  }
  return {};
}

[[nodiscard]] Status OutputRing::wait() {
  // Every in-flight chunk holds a slot, so holding them all means none is left.
  std::vector<size_t> held;
  held.reserve(slots_.size());
  for (size_t i = 0; i < slots_.size(); ++i) {
    held.push_back(acquire());
  }
  Status status;
  {
    const std::scoped_lock lock{mutex_};
    status = std::exchange(first_error_, Status{});
  }
  for (const size_t index : held) {
    release(index, {});
  }
  return status;
}

} // namespace free_breakfast
