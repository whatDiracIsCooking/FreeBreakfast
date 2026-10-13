/// @file
/// @brief The per-thread recording slot behind RecordPolicy and capture

module free_breakfast.core.error_handling;

import free_breakfast.core.log;
import std;

namespace free_breakfast {

// Per thread, because a policy has no other way to reach its capture: the
// policies are empty (see :gpu_policy), and the error belongs to whichever
// capture is running on the thread that made the failing call.
struct CaptureSlot {
  std::optional<Error> error;
  int depth = 0; // active captures on this thread
};

CaptureSlot &capture_slot() noexcept {
  thread_local CaptureSlot s;
  return s;
}

void record_error(Error error) noexcept {
  auto &s = capture_slot();
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
  return std::exchange(capture_slot().error, std::nullopt);
}

CaptureScope::CaptureScope() noexcept : outer_(take_error()) {
  ++capture_slot().depth;
}

CaptureScope::~CaptureScope() {
  auto &s = capture_slot();
  --s.depth;
  s.error = std::move(outer_);
}

} // namespace free_breakfast
