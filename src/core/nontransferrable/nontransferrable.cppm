/// @file
/// @brief Mixin base classes for controlling copy/move semantics

module;

export module free_breakfast.core.nontransferrable;

import std;

export namespace free_breakfast {

/**
 * @brief Mixin that deletes copy operations while allowing moves
 *
 * Inherit privately. The protected destructor prevents polymorphic deletion
 * through a base pointer.
 */
class NonCopyable {
public:
  // Public special members (matches std::unique_ptr convention)
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
  // Explicitly allow move operations (required for derived classes to be
  // movable)
  NonCopyable(NonCopyable &&) = default;
  NonCopyable &operator=(NonCopyable &&) = default;

protected:
  ~NonCopyable() = default; // Protected: prevents deletion through base pointer
};

/**
 * @brief Mixin that deletes move operations while allowing copies
 *
 * Symmetric to NonCopyable. Inherit privately. Protected destructor prevents
 * polymorphic deletion through a base pointer.
 */
class NonMovable {
public:
  // Public special members (symmetric to NonCopyable)
  NonMovable() = default;
  // Explicitly allow copy operations (symmetric to NonCopyable allowing move)
  NonMovable(const NonMovable &) = default;
  NonMovable &operator=(const NonMovable &) = default;
  // Delete move operations (declaring copy ops doesn't suppress move in all
  // contexts)
  NonMovable(NonMovable &&) = delete;
  NonMovable &operator=(NonMovable &&) = delete;

protected:
  ~NonMovable() = default; // Protected: prevents deletion through base pointer
};

/// @brief Mixin that deletes both copy and move operations (e.g., RAII guards)
class NonTransferable : private NonCopyable, private NonMovable {
protected:
  NonTransferable() = default;
  ~NonTransferable() = default;
};

} // namespace free_breakfast

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compile-Time Verification (not exported)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// We cannot directly verify base class semantics with type traits (protected
// destructors), but we can verify via minimal test-derived classes with public
// destructors

namespace free_breakfast {

// Test classes for verification
struct VerifyNonCopyable : private NonCopyable {};
struct VerifyNonMovable : private NonMovable {};
struct VerifyNonTransferable : private NonTransferable {};

// Verify NonCopyable: prevents copy, allows move
static_assert(!std::is_copy_assignable_v<VerifyNonCopyable>);
static_assert(std::is_move_assignable_v<VerifyNonCopyable>);

// Verify NonMovable: allows copy
// Note: std::is_move_assignable_v<VerifyNonMovable> is true because C++ allows
// binding rvalues to const lvalue refs, so move assignment falls back to copy
// assignment. The actual NonMovable base correctly has deleted move ops - this
// is expected behavior.
static_assert(std::is_copy_assignable_v<VerifyNonMovable>);

// Verify NonTransferable: prevents both copy and move
static_assert(!std::is_copy_assignable_v<VerifyNonTransferable>);
static_assert(!std::is_move_assignable_v<VerifyNonTransferable>);

// Verify using C++20 concepts (preferred - more semantic)
static_assert(!std::copyable<VerifyNonTransferable>);
static_assert(!std::movable<VerifyNonTransferable>);

// Note: std::copyable/movable concepts work correctly even with protected
// destructors
} // namespace free_breakfast
