/// @file
/// @brief The stream-level wrappers bound to this tree's policies -- Stream and
/// Event -- and ForkStream, a stream with fork/join events for graph-captured
/// parallel work

export module free_breakfast.core.wwr:fork_stream;

import std;
import :policy; // CreatePolicy / DestroyPolicy
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import wwr.runtime_api;
import wwr.extension.runtime;

export namespace free_breakfast {

/// @brief A non-blocking stream on one device.
using Stream =
    wwr::extension::StreamWrapper<CreatePolicy<wwr::wwrError_t>, DestroyPolicy<wwr::wwrError_t>,
                                  CreatePolicy<wwr::wwrError_t>>;

/// @brief An event on one device; default-constructed it is sync-only (timing disabled).
using Event =
    wwr::extension::EventWrapper<CreatePolicy<wwr::wwrError_t>, DestroyPolicy<wwr::wwrError_t>,
                                 CreatePolicy<wwr::wwrError_t>>;

/// @brief How a ForkStream creates its stream and events.
struct ForkStreamConfig {
  /// @brief Device the stream and both events are created on.
  int32_t dev_idx = 0;
  /// @brief Stream creation flags. Non-blocking by default, as Stream is, so a
  /// fork never serialises against the legacy default stream.
  uint32_t stream_flags = wwr::wwrStreamNonBlocking;
  /// @brief Stream priority (lower is higher); unset means the runtime's default.
  std::optional<int32_t> stream_priority;
  /// @brief Flags for the fork and join events. Timing is disabled by default:
  /// the events only order work, and a timing event is the slower kind.
  uint32_t event_flags = wwr::wwrEventDisableTiming;
};

/// @brief A Stream plus the two Events that fork it from, and join it back
/// into, a parent stream.
///
/// The fork/join protocol is what lets work fan out across streams inside one
/// graph capture: a stream becomes part of a capture by waiting on an event
/// recorded in the capturing stream, and leaves it when the capturing stream
/// waits on an event recorded in it. The same calls order work outside a
/// capture too.
///
/// ```cpp
/// ForkStream side{dev_idx};
/// FBF_GPU_MUST(side.fork_from(parent));   // record fork on parent; side waits
/// ... enqueue work on side.stream() ...
/// FBF_GPU_MUST(side.record_join());       // record join on side
/// FBF_GPU_MUST(side.join_into(parent));   // parent waits on join
/// ```
///
/// Every operation returns the runtime's error code, like wwr's own stream and
/// event free functions, so the caller picks the tier: FBF_GPU_MUST at a site
/// that cannot recover, gpu_status where a Status is wanted. Creation goes
/// through CreatePolicy and aborts on failure, as every wrapper here does.
///
/// Inside a capture the two events are dependency markers, not records: the
/// runtime rejects a synchronize or query on them until they are recorded again
/// outside the capture. Do not wait on fork_event() / join_event() yourself
/// after a captured fork/join; the destructor re-records them so Event's own
/// synchronize-then-destroy succeeds (see rearm_events()).
///
/// Exposes dev_idx() and stream(), so it satisfies wwr's device_handle_stream
/// and can back a buffer suite directly, the way a bare Stream can. Movable,
/// not copyable; a moved-from ForkStream owns nothing and must not be used.
class ForkStream : private NonCopyable {
public:
  /// @brief A non-blocking stream and two sync-only events on `dev_idx`.
  explicit ForkStream(int32_t dev_idx = 0,
                      std::source_location location = std::source_location::current())
      : ForkStream(ForkStreamConfig{.dev_idx = dev_idx}, location) {}

  /// @brief A stream and two events created as `cfg` says, all on `cfg.dev_idx`.
  explicit ForkStream(const ForkStreamConfig &cfg,
                      std::source_location location = std::source_location::current())
      : stream_(make_stream(cfg, location)), fork_event_(cfg.dev_idx, cfg.event_flags, location),
        join_event_(cfg.dev_idx, cfg.event_flags, location) {}

  ForkStream(ForkStream &&) noexcept = default;

  ForkStream &operator=(ForkStream &&other) noexcept {
    if (this != &other) {
      // The wrappers' move-assignment destroys what they held, so the events
      // about to be destroyed need the same treatment as in the destructor.
      rearm_events();
      stream_ = std::move(other.stream_);
      fork_event_ = std::move(other.fork_event_);
      join_event_ = std::move(other.join_event_);
    }
    return *this;
  }

  ~ForkStream() { rearm_events(); }

  /// @brief Index of the device the stream and events live on (-1 once moved from).
  int32_t dev_idx() const noexcept { return stream_.dev_idx(); }

  /// @brief The owned stream. Converts to wwrStream_t, so it is what a launch or
  /// an async copy takes, and what makes this a device_handle_stream.
  Stream &stream() noexcept { return stream_; }
  /// @copydoc stream()
  const Stream &stream() const noexcept { return stream_; }

  /// @brief The event fork_from() records on the parent.
  Event &fork_event() noexcept { return fork_event_; }
  /// @copydoc fork_event()
  const Event &fork_event() const noexcept { return fork_event_; }

  /// @brief The event record_join() records on this stream.
  Event &join_event() noexcept { return join_event_; }
  /// @copydoc join_event()
  const Event &join_event() const noexcept { return join_event_; }

  /// @brief Record the fork event on `parent`, then make this stream wait on it:
  /// work enqueued here afterwards runs after everything already on `parent`.
  /// Inside a capture of `parent`, this is what joins the capture.
  [[nodiscard]] wwr::wwrError_t fork_from(wwr::wwrStream_t parent) noexcept {
    if (const auto code = wwr::extension::record(fork_event_, parent); code != wwr::wwrSuccess) {
      return code;
    }
    return wwr::extension::wait_event(stream_, fork_event_);
  }

  /// @brief Record the join event on this stream, after the work the parent
  /// should wait for has been enqueued.
  [[nodiscard]] wwr::wwrError_t record_join() noexcept {
    return wwr::extension::record(join_event_, stream_);
  }

  /// @brief Make `parent` wait on the join event: work enqueued on `parent`
  /// afterwards runs after everything recorded by record_join().
  [[nodiscard]] wwr::wwrError_t join_into(wwr::wwrStream_t parent) noexcept {
    return wwr::extension::wait_event(parent, join_event_);
  }

  /// @brief Block the host until every operation on this stream has completed.
  [[nodiscard]] wwr::wwrError_t sync() noexcept { return wwr::extension::sync(stream_); }

private:
  // Record both events on the owned stream, for real, before Event destroys them.
  //
  // An event recorded inside a capture is a dependency edge in the graph, not a
  // record, and after the capture ends the runtime rejects wwrEventSynchronize
  // on it (cudaErrorInvalidValue) until it is recorded again. Event::destroy
  // synchronizes before destroying, so after a captured fork/join the plain
  // destructor path would log two errors -- and leave one as the runtime's last
  // error, for an unrelated wwrGetLastError() to report as its own. Recording
  // here, outside any capture, makes the events ordinary again; on a stream
  // that never captured it is a no-op in effect. Skipped when the stream is
  // still capturing (destroying a ForkStream mid-capture is already a bug this
  // cannot fix, and a record now would only be captured too) and when moved
  // from. Failures go to DestroyPolicy, as on every destroy path.
  void rearm_events() noexcept {
    if (!stream_.valid()) {
      return;
    }
    wwr::wwrStreamCaptureStatus status = wwr::wwrStreamCaptureStatusNone;
    if (wwr::wwrStreamIsCapturing(stream_, &status) != wwr::wwrSuccess ||
        status != wwr::wwrStreamCaptureStatusNone) {
      return;
    }
    for (Event *event : {&fork_event_, &join_event_}) {
      if (event->valid()) {
        DestroyPolicy<wwr::wwrError_t>{}.handle_error(wwr::extension::record(*event, stream_),
                                                      std::source_location::current());
      }
    }
  }

  // Stream has one constructor per (flags) / (flags, priority) shape rather
  // than an optional priority, so the choice is made here, once.
  static Stream make_stream(const ForkStreamConfig &cfg, std::source_location location) {
    if (cfg.stream_priority) {
      return Stream(cfg.dev_idx, cfg.stream_flags, *cfg.stream_priority, location);
    }
    return Stream(cfg.dev_idx, cfg.stream_flags, location);
  }

  Stream stream_;
  Event fork_event_;
  Event join_event_;
};

} // namespace free_breakfast
