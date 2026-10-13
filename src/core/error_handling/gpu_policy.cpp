/// @file
/// @brief The per-thread recording slot behind RecordPolicy and capture

module free_breakfast.core.error_handling;

import free_breakfast.core.log;
import std;

namespace free_breakfast {

namespace {
// Per thread, because a policy has no other way to reach its capture: the
// policies are empty (see :gpu_policy), and the error belongs to whichever
// capture is running on the thread that made the failing call.
struct Slot {
  std::optional<Error> error;
  int depth = 0; // active captures on this thread
};

Slot &slot() noexcept {
  thread_local Slot s;
  return s;
}
} // namespace

void record_error(Error error) noexcept {
  auto &s = slot();
  if (s.depth == 0) {
    try {
      log::error(error.message, error.origin);
    } catch (...) {
    }
  }
  if (!s.error) {
    s.error = std::move(error);
  }
}

std::optional<Error> take_error() noexcept {
  return std::exchange(slot().error, std::nullopt);
}

namespace detail {
CaptureScope::CaptureScope() noexcept : outer_(take_error()) {
  ++slot().depth;
}

CaptureScope::~CaptureScope() {
  auto &s = slot();
  --s.depth;
  s.error = std::move(outer_);
}
} // namespace detail

} // namespace free_breakfast
