/// @file
/// @brief The allocation permission: AllocKey, minted only by a root in this
/// module, and KeyedBuffer, a device buffer whose constructor takes one.

export module free_breakfast.experimental.routine:key;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;

export namespace free_breakfast::routine {

/// @brief Whether a routine may allocate device memory: Yes takes the
/// caller's AllocKey, No takes none and so cannot pass one on.
enum class Alloc : bool {
  No,  ///< takes no key
  Yes, ///< takes the caller's key and passes it to invoke
};

/// @brief The root that mints the key here: the step-shaped caller of the
/// root_rotation port. Declared before AllocKey so its friend declaration
/// names this exported entity rather than introducing an unexported one.
///
/// The key's only friend, and attached to this module, so no other module or
/// TU can define it: clang rejects that as "declaration of 'RootRotationStep'
/// in the global module follows declaration in module ..." (README.md "Where
/// the key's friend lives", and the must-not-compile case forged_root).
struct RootRotationStep;

/// @brief Permission to allocate device memory, minted only by a root.
///
/// Its default constructor is private, so only a friend makes one; its copy
/// constructor is deleted, so a routine holds a reference to its caller's and
/// never a key of its own. The user-provided destructor is load-bearing: with
/// it defaulted the class is trivially copyable, and std::bit_cast<AllocKey>
/// of any byte forges one.
class AllocKey {
  AllocKey() = default;
  friend struct RootRotationStep;

public:
  AllocKey(const AllocKey &) = delete;
  AllocKey &operator=(const AllocKey &) = delete;
  /// @brief Not trivial, so AllocKey is not trivially copyable.
  constexpr ~AllocKey() {} // NOLINT(modernize-use-equals-default): see the class comment
};

static_assert(!std::is_trivially_copyable_v<AllocKey>);
static_assert(!std::is_default_constructible_v<AllocKey>);

/// @brief A Buffers::device whose constructor needs an AllocKey: the only
/// way an Alloc::Yes routine's own allocations are made in this spike.
/// Everything else is Buffers::device's, so it passes wherever one does.
template <typename T>
class KeyedBuffer : public Buffers::device<T> {
public:
  /// @brief @p count elements from @p device's pool, with the caller's key.
  KeyedBuffer(const AllocKey & /*key*/, const size_t count, const std::shared_ptr<Device> &device)
      : Buffers::device<T>(count, device) {}
};

} // namespace free_breakfast::routine
