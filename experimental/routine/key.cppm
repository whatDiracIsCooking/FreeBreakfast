/// @file
/// @brief The allocation permission: AllocKey, minted only by a root, and
/// KeyedBuffer, a device buffer whose constructor takes one.

export module free_breakfast.experimental.routine:key;

import std;
import free_breakfast.core.types;
import free_breakfast.core.wwr;

// The fixture's minter, attached to the global module so a test TU, which is
// not a unit of this module, can define it. A friend named from this module
// would be attached here, and a class attached to one module cannot be defined
// in another (clang: "declaration of 'X' in module B follows declaration in
// module A"). The price is that any TU may define it: README.md "Friends
// across modules".
extern "C++" {
namespace free_breakfast::test {
struct RoutineKeyMinter;
} // namespace free_breakfast::test
}

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
  friend struct ::free_breakfast::test::RoutineKeyMinter;

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
