/// @file linkprobe.cpp
/// @brief Links fb_overlap's archive the way the Rust binary must, so CMake's
/// link line for it can be recorded (rust/gemm's record_link.sh) and replayed
/// by build.rs. Never run for its effect: the call below is only there to pull
/// the archive's object, and with it every library behind it, into the link.
#include "fb_overlap.h"

int main(const int argc, char ** /*argv*/) { // int-type-ok: main's signature is fixed
  if (argc > 1'000'000) {
    fb_overlap(nullptr, nullptr, nullptr, nullptr, nullptr);
  }
  return 0;
}
