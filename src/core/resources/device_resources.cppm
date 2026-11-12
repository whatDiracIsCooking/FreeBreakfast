/// @file
/// @brief DeviceResources: one device's context, the library handles bound to
/// its stream, the workspaces they share, sized at construction, and its
/// pinned host memory

export module free_breakfast.core.resources:device_resources;

import std;
import free_breakfast.core.types;
import free_breakfast.core.nontransferrable;
import free_breakfast.core.primitives;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
// Re-exported: the constructor takes the workspaces' sizes as a
// MemoryFootprint, and workspace_high_water() hands one out.
export import free_breakfast.core.footprint;

export namespace free_breakfast {

class DeviceResources;

/// @brief A pinned host buffer counted in the pinned total of the
/// DeviceResources that allocated it (DeviceResources::counted_pinned): its
/// bytes join the total when it is allocated and leave it when it is freed.
///
/// For a product that lives in pinned memory past the call that made it, such
/// as an offloaded block streamed through a ring. Fixed-size staging -- an
/// upload, a readback, a ring's slots -- takes the pinned workspace instead.
///
/// It may outlive its DeviceResources: the total it returns its bytes to is
/// shared with it. Its free is immediate, not stream-ordered, so drain every
/// copy that reads or writes it before it is destroyed or assigned to.
/// Movable, not copyable; a moved-from one owns nothing.
template <typename T>
class CountedPinned : private NonCopyable {
public:
  CountedPinned(CountedPinned &&) noexcept = default;

  CountedPinned &operator=(CountedPinned &&other) noexcept {
    if (this != &other) {
      release();
      buffer_ = std::move(other.buffer_);
      counted_ = std::move(other.counted_);
    }
    return *this;
  }

  ~CountedPinned() { release(); }

  /// @brief The first element.
  T *data() noexcept { return buffer_.data(); }
  /// @copydoc data()
  const T *data() const noexcept { return buffer_.data(); }
  /// @brief Number of elements.
  size_t num_elements() const noexcept { return buffer_.num_elements(); }

private:
  friend class DeviceResources;

  // Allocate count elements and add their bytes to *counted. A failed
  // allocation aborts (CreatePolicy).
  CountedPinned(const size_t count, std::shared_ptr<size_t> counted)
      : buffer_(count), counted_(std::move(counted)) {
    *counted_ += buffer_.num_elements() * sizeof(T);
  }

  // Take this buffer's bytes back out of the total; skipped when moved from.
  void release() noexcept {
    if (counted_) {
      *counted_ -= buffer_.num_elements() * sizeof(T);
      counted_.reset();
    }
  }

  Buffers::pinned<T> buffer_;
  std::shared_ptr<size_t> counted_;
};

/// @brief One device's working set: a shared Device (ForkStream + memory pool),
/// the BLAS and dense-solver handles bound to that Device's stream, the
/// workspaces those calls share (a byte workspace on the device and one in
/// pinned host memory, both sized at construction), a device info flag with a
/// pinned slot to read it back into, and counted pinned buffers, the pinned
/// total held under one optional cap.
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
/// **The workspaces are sized at construction and held for its life.** The
/// constructor allocates both, and nothing resizes them: device_workspace,
/// device_regions, pinned_workspace and pinned_regions borrow from them and
/// carve them, and a borrow past a workspace's size returns Unsupported
/// naming the caller and both sizes, allocating nothing. So what goes in a
/// workspace may depend on a routine's dimensions and options, never on a
/// memory budget: inside a pipeline the sizes are pre-flight's plan maximum,
/// and a refused borrow is a footprint that undercounted.
///
/// **One routine owns each workspace at a time.** A routine may carve all of
/// its device scratch from the device workspace -- a solver's work and its own
/// intermediates alike -- by asking once for the total and splitting the
/// result. So nothing nests: a routine holds no region of it across a call to
/// another that uses it, and no argument to one may point into it. A callback
/// into caller code (an iterative solver's operator) counts as such a call:
/// assume it takes the workspace. The README has why, and davidson_solve the
/// pattern. The pinned workspace follows the same rule, and one more: the CPU
/// writes it, so a routine drains the copies that read or write a region (a
/// sync, or an event the reader waits on) before it rewrites the region or
/// returns.
///
/// **The memory budget.** memory_budget() is the device memory a refusal
/// check may count on: what the pool can still hand out, counting memory it
/// holds cached but unused as available. Past it an allocation through the
/// pool would abort (CreatePolicy), so a site that can refuse compares its
/// need against this, not against the driver's free memory.
///
/// **The pinned cap.** pinned_cap() bytes (0: none) bound the pinned total,
/// pinned_bytes(): the pinned workspace, from construction, plus every live
/// CountedPinned. A counted allocation that would take the total past it
/// returns Unsupported, and allocates nothing.
///
/// Movable, not copyable. A moved-from DeviceResources holds no Device.
class DeviceResources : private NonCopyable {
public:
  /// @brief Default release threshold of the Device's memory pool: 1 GiB
  /// (Device::kDefaultPoolReleaseThreshold).
  static constexpr uint64_t kDefaultPoolReleaseThreshold = Device::kDefaultPoolReleaseThreshold;

  /// @brief The granularity of the workspaces' sizes and of every borrow, in
  /// bytes: the pool's own allocation alignment, and a multiple of every
  /// solver element type's size, so a region carved at a multiple of it starts
  /// aligned for any of them.
  static constexpr size_t kWorkspaceAlignment = 256;

  /// @brief kWorkspaceAlignment in elements of T: the step a routine carving
  /// the workspace into T-typed regions rounds each region up to. Constrained
  /// to a T whose size divides kWorkspaceAlignment, which rules out a step
  /// that truncates (regions off the boundary) or is 0 (align_up divides by
  /// it).
  template <typename T>
    requires(kWorkspaceAlignment % sizeof(T) == 0)
  static constexpr size_t kWorkspaceAlignmentIn = kWorkspaceAlignment / sizeof(T);

  /// @brief What a process holds on a device that no pool counts -- the
  /// context, and the BLAS and solver handles' own workspaces -- in bytes:
  /// 256 MiB, memory_budget()'s allowance for it on an uncapped pool.
  ///
  /// The 249 MiB measured on the 3080 (src/core/wwr/README.md, "The fixed
  /// reserve outside the pool"), rounded up: it is a floor, which grows as
  /// lazily loaded kernel modules are first launched. The kernels' stack,
  /// which grows far more, is kKernelStackBytes'. A pool cap is set below
  /// the device's share by this much, so it is not taken again from the cap.
  static constexpr uint64_t kDeviceReserve = uint64_t{256} << 20;

  /// @brief The largest per-thread stack of any kernel the tree launches, in
  /// bytes: kinetic_hessian<3,3>'s 23,192 (`cuobjdump
  /// --dump-resource-usage`), a generated integral kernel.
  ///
  /// CUDA holds a kernel's stack for every resident thread from its first
  /// launch on, outside the pool (Device::stack_growth: 2.2 GiB on the 3080
  /// at this size), so an uncapped memory_budget() leaves room for it before
  /// the kernel that needs it has run. The ctest core_kernel_stack_bound
  /// checks it against the built binary; raise it with a regenerated kernel
  /// that needs more. src/core/wwr/README.md, "Kernel stacks outside the
  /// pool", has the measurements.
  static constexpr size_t kKernelStackBytes = 23192;

  /// @brief What a DeviceResources itself holds in its pool from
  /// construction on besides the device workspace, in bytes: the info() flag,
  /// one int32_t. The workspace is a footprint's (its workspace field, which
  /// the plan's peak counts); this no step's footprint can count, so a refusal
  /// check that takes a pool's whole cap (pre-flight's device room) takes this
  /// off it too.
  static constexpr size_t kOwnPoolBytes = sizeof(int32_t);

  /// @brief Create a Device on `dev_idx` whose pool keeps up to
  /// `pool_release_threshold` bytes cached and is capped at `pool_cap` bytes
  /// (0: no cap; a cap is also the release threshold, see Device), bind a
  /// BLAS and a solver handle to its stream, allocate the device and pinned
  /// workspaces at `workspaces.device.workspace` and
  /// `workspaces.pinned.workspace` bytes, each rounded up to
  /// kWorkspaceAlignment (its other fields are unread), and hold the pinned
  /// total to `pinned_cap` bytes (0: no cap).
  ///
  /// The device workspace comes from the pool, so it counts against
  /// `pool_cap`; the pinned one counts against `pinned_cap` from here on. A
  /// failed creation or allocation aborts (CreatePolicy), and so does a pinned
  /// workspace past `pinned_cap`: pre-flight sizes both from a plan it has
  /// held to the caps.
  explicit DeviceResources(int32_t dev_idx, const MemoryFootprint &workspaces,
                           uint64_t pool_release_threshold = kDefaultPoolReleaseThreshold,
                           size_t pinned_cap = 0, uint64_t pool_cap = 0)
      : device_(std::make_shared<Device>(dev_idx, pool_release_threshold, pool_cap)),
        blas_(device_), solver_(device_),
        device_workspace_(math::align_up(workspaces.device.workspace, kWorkspaceAlignment),
                          device_),
        info_(kOwnPoolBytes / sizeof(int32_t), device_), info_host_(1),
        pinned_workspace_(
            capped_pinned_workspace(dev_idx, workspaces.pinned.workspace, pinned_cap)),
        pinned_counted_(std::make_shared<size_t>(0)), pinned_cap_(pinned_cap), pool_cap_(pool_cap) {
  }

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

  /// @brief Borrow the first `bytes` of the device workspace, for `caller`.
  ///
  /// Returns them; a span of device memory, for its data(). Unsupported
  /// naming `caller`, the request rounded up to kWorkspaceAlignment and the
  /// workspace's size when the request is past it. Nothing is allocated
  /// either way.
  [[nodiscard]] Result<std::span<std::byte>> device_workspace(const std::string_view caller,
                                                              const size_t bytes) {
    return borrow(caller, "device", device_workspace_.data(), device_workspace_.num_elements(),
                  bytes, device_high_water_);
  }

  /// @brief The device workspace carved into one region per count: a region
  /// of count elements of its T, each rounded up to kWorkspaceAlignment so
  /// every one starts aligned for its type, from a single device_workspace
  /// borrow of the total for `caller`.
  ///
  /// One T per count pairs them up, `device_regions<double, int32_t>(who, n,
  /// m)`; a single T is every region's, `device_regions<double>(who, a, b,
  /// c)`. Returns the regions' starts in order, or device_workspace's
  /// refusal.
  template <typename... Ts, std::same_as<size_t>... Counts>
    requires(sizeof...(Ts) == sizeof...(Counts) || sizeof...(Ts) == 1) &&
            ((kWorkspaceAlignment % sizeof(Ts) == 0) && ...)
  [[nodiscard]] auto device_regions(const std::string_view caller, const Counts... counts) {
    if constexpr (sizeof...(Ts) != sizeof...(Counts)) {
      using T = std::tuple_element_t<0, std::tuple<Ts...>>;
      return device_regions<std::conditional_t<true, T, Counts>...>(caller, counts...);
    } else {
      return carve<Ts...>(device_workspace(caller, region_bytes<Ts...>(counts...)), counts...);
    }
  }

  /// @brief Borrow the first `bytes` of the pinned host workspace, for
  /// `caller`.
  ///
  /// For fixed-size staging: an upload's source, a small readback, the bounce
  /// buffer of a load from disk, the slots of an offload ring. Returns them,
  /// or Unsupported naming `caller`, the request rounded up to
  /// kWorkspaceAlignment and the workspace's size when the request is past
  /// it. Nothing is allocated either way.
  [[nodiscard]] Result<std::span<std::byte>> pinned_workspace(const std::string_view caller,
                                                              const size_t bytes) {
    return borrow(caller, "pinned", pinned_workspace_.data(), pinned_workspace_.num_elements(),
                  bytes, pinned_high_water_);
  }

  /// @brief The pinned workspace carved into one region per count, as
  /// device_regions carves the device workspace: each region rounded up to
  /// kWorkspaceAlignment, from a single pinned_workspace borrow of the total
  /// for `caller`.
  ///
  /// Returns the regions' starts in order, or pinned_workspace's refusal.
  /// Drain the copies that use them before returning.
  template <typename... Ts, std::same_as<size_t>... Counts>
    requires(sizeof...(Ts) == sizeof...(Counts) || sizeof...(Ts) == 1) &&
            ((kWorkspaceAlignment % sizeof(Ts) == 0) && ...)
  [[nodiscard]] auto pinned_regions(const std::string_view caller, const Counts... counts) {
    if constexpr (sizeof...(Ts) != sizeof...(Counts)) {
      using T = std::tuple_element_t<0, std::tuple<Ts...>>;
      return pinned_regions<std::conditional_t<true, T, Counts>...>(caller, counts...);
    } else {
      return carve<Ts...>(pinned_workspace(caller, region_bytes<Ts...>(counts...)), counts...);
    }
  }

  /// @brief The workspaces' sizes, in bytes, as the constructor took them
  /// (rounded up): a MemoryFootprint with only its two workspace fields set.
  [[nodiscard]] MemoryFootprint workspaces() const noexcept {
    return {.device = {.workspace = device_workspace_.num_elements()},
            .pinned = {.workspace = pinned_workspace_.num_elements()}};
  }

  /// @brief The largest borrow of each workspace since construction, in
  /// bytes, rounded up to kWorkspaceAlignment as the borrows are: a
  /// MemoryFootprint with only its two workspace fields set. What a routine's
  /// footprint states as its workspace, so a test holds the footprint tight
  /// against it.
  [[nodiscard]] MemoryFootprint workspace_high_water() const noexcept {
    return {.device = {.workspace = device_high_water_},
            .pinned = {.workspace = pinned_high_water_}};
  }

  /// @brief A pinned buffer of `count` elements whose bytes are counted in
  /// pinned_bytes() for as long as it lives: for a product that stays in
  /// pinned memory past one call.
  ///
  /// One that would take pinned_bytes() past pinned_cap() returns Unsupported
  /// naming `caller`, and allocates nothing. A failed allocation aborts
  /// (CreatePolicy).
  template <typename T>
  [[nodiscard]] Result<CountedPinned<T>> counted_pinned(const std::string_view caller,
                                                        const size_t count) {
    if (auto admitted = admit_pinned(caller, count * sizeof(T)); !admitted) {
      return std::unexpected(std::move(admitted).error());
    }
    return CountedPinned<T>(count, pinned_counted_);
  }

  /// @brief The pinned total, in bytes: the pinned workspace, held from
  /// construction, plus every live CountedPinned this allocated.
  size_t pinned_bytes() const noexcept {
    return pinned_workspace_.num_elements() + *pinned_counted_;
  }

  /// @brief The cap on pinned_bytes(), in bytes; 0 for none.
  size_t pinned_cap() const noexcept { return pinned_cap_; }

  /// @brief The cap on device()'s pool, in bytes; 0 for none.
  uint64_t pool_cap() const noexcept { return pool_cap_; }

  /// @brief The device memory, in bytes, the pool can still give: its limit
  /// less what its live allocations use (UsedMemCurrent). The limit is
  /// pool_cap(), or with no cap the device's total less kDeviceReserve; and
  /// never more than the driver can supply, its free memory plus what the
  /// pool holds cached (ReservedMemCurrent), since another process may hold
  /// part of the device. With no cap, the supply also leaves out what the
  /// kernels' stack can still take (Device::stack_growth(kKernelStackBytes)),
  /// which the driver takes outside the pool at a launch after this is read;
  /// under a cap the pool already holds its cap (pre-flight's pre-warm), and
  /// the stack comes from what the cap leaves of the device. 0 when the pool
  /// already uses the limit. A GpuRuntime error if the driver's free memory
  /// cannot be read; a failed pool or limit query aborts (CreatePolicy).
  ///
  /// Unlike the driver's free memory, it counts what the pool holds reserved
  /// but unused as available: a freed block the pool keeps cached is the next
  /// allocation's. A read, not a sync, as Device::pool_usage() is.
  [[nodiscard]] Result<size_t> memory_budget() const {
    const ScopedDeviceIndex on{dev_idx()};
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    // pool-bypass-ok: the budget authority itself, bounding the pool by the driver.
    if (auto status = gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)); !status) {
      return std::unexpected(std::move(status).error());
    }
    const PoolUsage usage = device_->pool_usage();
    const uint64_t total = total_bytes;
    const uint64_t uncapped = total > kDeviceReserve ? total - kDeviceReserve : 0;
    const uint64_t limit = pool_cap_ != 0 ? pool_cap_ : uncapped;
    const uint64_t driver = uint64_t{free_bytes} + usage.reserved_current;
    const uint64_t stack = pool_cap_ != 0 ? 0 : device_->stack_growth(kKernelStackBytes);
    const uint64_t supply = driver > stack ? driver - stack : 0;
    const uint64_t bound = std::min(limit, supply);
    return static_cast<size_t>(bound > usage.used_current ? bound - usage.used_current : 0);
  }

  /// @brief One device int32_t for a solver's devInfo, allocated with the
  /// handles. Read it back with fetch_info().
  Buffers::device<int32_t> &info() noexcept { return info_; }

  /// @brief Wait on the host for everything queued on stream(). A GpuRuntime
  /// error if the wait fails.
  [[nodiscard]] Status sync() const { return gpu_status(wwr::extension::sync(stream())); }

  /// @brief Copy info() into pinned host memory and return it: the d2h copy is
  /// queued on stream(), after the solver call that wrote it, and stream() is
  /// then synchronized. A GpuRuntime error if either call fails.
  [[nodiscard]] Result<int32_t> fetch_info() {
    if (auto status = gpu_status(wwr::extension::copy(info_host_, info_, stream())); !status) {
      return std::unexpected(std::move(status).error());
    }
    if (auto status = sync(); !status) {
      return std::unexpected(std::move(status).error());
    }
    return info_host_.data()[0];
  }

private:
  // The pinned workspace's size, `bytes` rounded up; past a nonzero `cap` it
  // aborts, as a failed creation does.
  static size_t capped_pinned_workspace(const int32_t dev_idx, const size_t bytes,
                                        const size_t cap) {
    const size_t held = math::align_up(bytes, kWorkspaceAlignment);
    if (cap != 0 && held > cap) {
      check_or_abort(Status{err_config("DeviceResources: a pinned workspace of {} bytes is past "
                                       "device {}'s pinned cap of {}",
                                       held, dev_idx, cap)});
    }
    return held;
  }

  // The first `bytes` of a workspace of `size` bytes at `base`, or `caller`'s
  // refusal; `high_water` keeps the largest borrow.
  [[nodiscard]] Result<std::span<std::byte>> borrow(const std::string_view caller,
                                                    const std::string_view kind,
                                                    std::byte *const base, const size_t size,
                                                    const size_t bytes, size_t &high_water) const {
    const size_t rounded = math::align_up(bytes, kWorkspaceAlignment);
    if (rounded > size) {
      return err_unsupported("{}: borrows {} bytes of device {}'s {} workspace, which holds {}",
                             caller, rounded, dev_idx(), kind, size);
    }
    high_water = std::max(high_water, rounded);
    return std::span<std::byte>(base, bytes);
  }

  // The bytes the regions of `counts` elements of their Ts take, each
  // rounded up to kWorkspaceAlignment.
  template <typename... Ts, typename... Counts>
  static size_t region_bytes(const Counts... counts) {
    return (math::align_up(counts * sizeof(Ts), kWorkspaceAlignment) + ... + size_t{0});
  }

  // The regions' starts in a borrow of region_bytes<Ts...>(counts...), or its
  // refusal.
  template <typename... Ts, typename... Counts>
  static Result<std::tuple<Ts *...>> carve(Result<std::span<std::byte>> block,
                                           const Counts... counts) {
    if (!block) {
      return std::unexpected(std::move(block).error());
    }
    size_t bytes = 0;
    const std::array<size_t, sizeof...(Ts)> at{
        std::exchange(bytes, bytes + math::align_up(counts * sizeof(Ts), kWorkspaceAlignment))...};
    std::byte *const base = block->data();
    size_t i = 0;
    return std::tuple<Ts *...>{reinterpret_cast<Ts *>(base + at[i++])...};
  }

  // Refuse a counted pinned request of `bytes` when the total would pass the
  // cap.
  [[nodiscard]] Status admit_pinned(const std::string_view caller, const size_t bytes) const {
    const size_t total = pinned_bytes();
    if (const size_t after = total + bytes; pinned_cap_ != 0 && after > pinned_cap_) {
      return err_unsupported("{}: {} pinned bytes would take device {}'s pinned total from {} "
                             "to {} bytes, past its cap of {}",
                             caller, bytes, dev_idx(), total, after, pinned_cap_);
    }
    return {};
  }

  // Declared first: blas_, solver_ and the device buffers are constructed from
  // it.
  std::shared_ptr<Device> device_;
  BlasHandle blas_;
  SolverHandle solver_;
  Buffers::device<std::byte> device_workspace_;
  Buffers::device<int32_t> info_;
  // Four bytes, fixed, read by fetch_info() at any time: not counted.
  Buffers::pinned<int32_t> info_host_;
  Buffers::pinned<std::byte> pinned_workspace_;
  // The bytes of the live CountedPinned buffers, shared with them so one can
  // outlive this.
  std::shared_ptr<size_t> pinned_counted_;
  size_t pinned_cap_;
  uint64_t pool_cap_;
  // The largest borrows, for workspace_high_water().
  size_t device_high_water_ = 0;
  size_t pinned_high_water_ = 0;
};

} // namespace free_breakfast
