// Must not compile (#1246 case b): an Alloc::No routine constructs the keyed
// buffer. It has no key to pass, and cannot make one: AllocKey's default
// constructor is private. CMakeLists.txt registers a ctest entry that builds
// this TU and passes on that diagnostic.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.experimental.routine;

namespace {

namespace fb = free_breakfast;

struct Allocates : fb::routine::Routine<Allocates, fb::routine::Alloc::No> {
  struct In {
    fb::DeviceResources &res;
    std::size_t n;
  };
  static void check(const In & /*in*/) {}
  static std::size_t invoke(const In &in) {
    const fb::routine::KeyedBuffer<double> buffer(fb::routine::AllocKey{}, in.n, in.res.device());
    return buffer.num_elements();
  }
};

} // namespace

std::size_t routine_reject_key_buffer(const Allocates::In &in) {
  return Allocates{}(in);
}
