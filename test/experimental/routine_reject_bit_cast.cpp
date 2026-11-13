// Must not compile (#1246 case c): the key forged with std::bit_cast. Its
// user-provided destructor makes AllocKey not trivially copyable, which
// bit_cast requires of its target. CMakeLists.txt registers a ctest entry
// that builds this TU and passes on that diagnostic.

import std;
import free_breakfast.experimental.routine;

namespace {

namespace fb = free_breakfast;

static_assert(sizeof(fb::routine::AllocKey) == sizeof(char));

} // namespace

void routine_reject_bit_cast() {
  [[maybe_unused]] const auto key = std::bit_cast<fb::routine::AllocKey>(char{});
}
