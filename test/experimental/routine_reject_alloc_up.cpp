// Must not compile (#1246 case a): an Alloc::No routine calls an Alloc::Yes
// one. It holds no key, so the only call it can spell passes none, and the
// Yes routine's operator() takes two arguments. CMakeLists.txt registers a
// ctest entry that builds this TU and passes on that diagnostic.

import std;
import free_breakfast.experimental.routine;

namespace {

namespace fb = free_breakfast;

struct CallsUp : fb::routine::Routine<CallsUp, fb::routine::Alloc::No> {
  struct In {
    const fb::routine::RootRotationRoutine::In &inner;
  };
  static void check(const In & /*in*/) {}
  static auto invoke(const In &in) { return fb::routine::root_rotation(in.inner); }
};

} // namespace

auto routine_reject_alloc_up(const CallsUp::In &in) {
  return CallsUp{}(in);
}
