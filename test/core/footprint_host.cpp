// then / drop against hand-computed peaks: each case's footprint is composed
// from made-up parts, and its peak is held to the value worked out beside it,
// exactly. Host-only: a footprint is arithmetic over byte counts.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;

using Device = fb::Footprint<wwr::extension::MemoryKind::Device>;
using Pinned = fb::Footprint<wwr::extension::MemoryKind::Pinned>;

// a's scratch is above b's product, so the residue (50 - 20 = 30) is what
// stays in scratch: while a runs, 10 + 50 = 60; while b runs, 10 + 20 + 15 =
// 45; plus the larger workspace, 8.
TEST(CoreFootprint, ThenScratchAboveNextPersistent) {
  const Device a{.persistent = 10, .scratch = 50, .workspace = 5};
  const Device b{.persistent = 20, .scratch = 15, .workspace = 8};
  const Device ab = fb::then(a, b);
  EXPECT_EQ(ab, (Device{.persistent = 30, .scratch = 30, .workspace = 8}));
  EXPECT_EQ(ab.peak(), std::size_t{60 + 8});
}

// a's scratch is below b's product: 5 - 20 clamps at 0, and b's own scratch
// sets it. While a runs, 10 + 5 = 15; while b runs, 10 + 20 + 7 = 37; plus 3.
TEST(CoreFootprint, ThenScratchBelowNextPersistentClampsAtZero) {
  const Device a{.persistent = 10, .scratch = 5, .workspace = 0};
  const Device b{.persistent = 20, .scratch = 7, .workspace = 3};
  const Device ab = fb::then(a, b);
  EXPECT_EQ(ab, (Device{.persistent = 30, .scratch = 7, .workspace = 3}));
  EXPECT_EQ(ab.peak(), std::size_t{37 + 3});

  // With no scratch of b's own, nothing is left in scratch at all.
  const Device b_bare{.persistent = 20, .scratch = 0, .workspace = 0};
  EXPECT_EQ(fb::then(a, b_bare), (Device{.persistent = 30, .scratch = 0, .workspace = 0}));
  EXPECT_EQ(fb::then(a, b_bare).peak(), std::size_t{30});
}

// a's scratch equals b's product: the residue is exactly 0, and both phases
// hold 4 + 20 = 24 (b's scratch adds nothing above it).
TEST(CoreFootprint, ThenScratchEqualToNextPersistent) {
  const Device a{.persistent = 4, .scratch = 20, .workspace = 0};
  const Device b{.persistent = 20, .scratch = 0, .workspace = 0};
  const Device ab = fb::then(a, b);
  EXPECT_EQ(ab, (Device{.persistent = 24, .scratch = 0, .workspace = 0}));
  EXPECT_EQ(ab.peak(), std::size_t{24});
}

// drop moves the product into scratch and leaves the peak alone; of a then,
// the peak is the then's (60 + 8, from the first case).
TEST(CoreFootprint, DropOfThen) {
  const Device a{.persistent = 10, .scratch = 50, .workspace = 5};
  const Device b{.persistent = 20, .scratch = 15, .workspace = 8};
  const Device dropped = fb::drop(fb::then(a, b));
  EXPECT_EQ(dropped, (Device{.persistent = 0, .scratch = 60, .workspace = 8}));
  EXPECT_EQ(dropped.peak(), std::size_t{68});

  // The README's example: B (100, built with 40 of scratch) freed before J and
  // K (30, with 20) return. B's build is the peak, 140; J and K are kept.
  const Device b_mo{.persistent = 100, .scratch = 40, .workspace = 0};
  const Device jk{.persistent = 30, .scratch = 20, .workspace = 0};
  const Device routine = fb::then(fb::drop(b_mo), jk);
  EXPECT_EQ(routine, (Device{.persistent = 30, .scratch = 110, .workspace = 0}));
  EXPECT_EQ(routine.peak(), std::size_t{140});
}

// The workspace is one block held throughout, so a composition holds the
// larger of the two, in either order and through a drop.
TEST(CoreFootprint, WorkspaceIsTheMaximum) {
  const Device a{.persistent = 1, .scratch = 2, .workspace = 100};
  const Device b{.persistent = 3, .scratch = 4, .workspace = 40};
  EXPECT_EQ(fb::then(a, b).workspace, std::size_t{100});
  EXPECT_EQ(fb::then(b, a).workspace, std::size_t{100});
  EXPECT_EQ(fb::drop(fb::then(b, a)).workspace, std::size_t{100});
  // max(1 + 2, 1 + 3 + 4) + 100 and max(3 + 4, 3 + 1 + 2) + 100.
  EXPECT_EQ(fb::then(a, b).peak(), std::size_t{108});
  EXPECT_EQ(fb::then(b, a).peak(), std::size_t{107});
}

// MemoryFootprint's overloads compose each kind by itself: the device half is
// the first case, the pinned half the second, unchanged by the other.
TEST(CoreFootprint, MemoryFootprintComposesPerKind) {
  const fb::MemoryFootprint a{.device = {.persistent = 10, .scratch = 50, .workspace = 5},
                              .pinned = {.persistent = 10, .scratch = 5, .workspace = 0}};
  const fb::MemoryFootprint b{.device = {.persistent = 20, .scratch = 15, .workspace = 8},
                              .pinned = {.persistent = 20, .scratch = 7, .workspace = 3}};
  const fb::MemoryFootprint ab = fb::then(a, b);
  EXPECT_EQ(ab.device, (Device{.persistent = 30, .scratch = 30, .workspace = 8}));
  EXPECT_EQ(ab.pinned, (Pinned{.persistent = 30, .scratch = 7, .workspace = 3}));
  EXPECT_EQ(ab.device.peak(), std::size_t{68});
  EXPECT_EQ(ab.pinned.peak(), std::size_t{40});

  const fb::MemoryFootprint dropped = fb::drop(ab);
  EXPECT_EQ(dropped.device, (Device{.persistent = 0, .scratch = 60, .workspace = 8}));
  EXPECT_EQ(dropped.pinned, (Pinned{.persistent = 0, .scratch = 37, .workspace = 3}));
  EXPECT_EQ(dropped.device.peak(), ab.device.peak());
  EXPECT_EQ(dropped.pinned.peak(), ab.pinned.peak());
}

} // namespace
