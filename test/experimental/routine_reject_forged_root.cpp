// Must not compile (#1324 case e): a key forged by a TU outside the root's
// module. AllocKey's one friend, RootRotationStep, is declared in the key's
// module and so attached to it; this TU defines a class of that name in the
// global module to mint a key with, and clang refuses the declaration before
// it reaches the constructor. CMakeLists.txt registers a ctest entry that
// builds this TU and passes on that diagnostic.

import std;
import free_breakfast.experimental.routine;

namespace free_breakfast::routine {

struct RootRotationStep {
  static std::size_t forge() {
    const AllocKey key;
    return sizeof(key);
  }
};

} // namespace free_breakfast::routine

std::size_t routine_reject_forged_root() {
  return free_breakfast::routine::RootRotationStep::forge();
}
