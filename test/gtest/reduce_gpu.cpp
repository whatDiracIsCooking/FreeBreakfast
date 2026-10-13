// GPU cases for core/kernel: block_reduce -- warp_reduce underneath -- run on
// the device through test/gpu_kernels/reduce_kernels.cu and compared against a
// left fold on the host, at 1, 2, 4 and the most warps a block holds.
//
// The inputs are small integers (as doubles too), so every sum is exact
// whatever the association, and the comparisons are equality. First and Last
// are associative but not commutative: they return the first or last ACTIVE
// input, so a reduction that reorders operands or folds in a thread past
// nactive fails them. Every thread's output is checked, active or not, since
// every thread is promised the result.

#include <gtest/gtest.h>

#include "gpu_requirement.h"
#include "reduce_kernels.h"

import std;
import free_breakfast.core;
import wwr.runtime_api;

namespace {

namespace fb = free_breakfast;
namespace tk = free_breakfast::test_kernels;

using tk::FoldOp;

constexpr std::array kOps{FoldOp::Add, FoldOp::MaxNan, FoldOp::First, FoldOp::Last};

const char *op_name(const FoldOp op) {
  switch (op) {
  case FoldOp::Add:
    return "Add";
  case FoldOp::MaxNan:
    return "MaxNan";
  case FoldOp::First:
    return "First";
  case FoldOp::Last:
    return "Last";
  }
  return "?";
}

// Distinct-looking small integers of both signs.
template <typename T>
std::vector<T> make_inputs(const std::size_t count) {
  std::vector<T> v(count);
  for (std::size_t t = 0; t < count; ++t) {
    v[t] = static_cast<T>(static_cast<std::int32_t>((t * 37) % 101) - 50);
  }
  return v;
}

// The left fold of the first nactive inputs: the order the device promises.
template <typename T>
T reference(const FoldOp op, const std::vector<T> &v, const std::uint32_t nactive) {
  T acc = v[0];
  for (std::uint32_t t = 1; t < nactive; ++t) {
    switch (op) {
    case FoldOp::Add:
      acc = acc + v[t];
      break;
    case FoldOp::MaxNan:
      // NaN wins, as MaxNanOp does on the device with `a != a`.
      acc = (acc != acc || acc > v[t]) ? acc : v[t];
      break;
    case FoldOp::First:
      break;
    case FoldOp::Last:
      acc = v[t];
      break;
    }
  }
  return acc;
}

// Every thread received @p expected; reports only the first disagreement.
template <typename T>
void expect_all_equal(const std::vector<T> &out, const T expected) {
  for (std::size_t t = 0; t < out.size(); ++t) {
    const bool same = (expected != expected) ? (out[t] != out[t]) : (out[t] == expected);
    if (!same) {
      ADD_FAILURE() << "thread " << t << " got " << out[t] << ", expected " << expected;
      return;
    }
  }
}

// nactive values worth checking for a block of @p threads: a lone thread, the
// edges of the first warp, and the full block.
std::vector<std::uint32_t> nactive_cases(const std::uint32_t threads) {
  std::vector<std::uint32_t> cases;
  for (const std::uint32_t n :
       {1U, 2U, fb::kWarpSize - 1, fb::kWarpSize, fb::kWarpSize + 1, threads - 1, threads}) {
    if (n >= 1 && n <= threads && std::ranges::find(cases, n) == cases.end()) {
      cases.push_back(n);
    }
  }
  return cases;
}

template <typename T>
void check_block_reduce() {
  for (const std::uint32_t num_warps : {1U, 2U, 4U, 1024 / fb::kWarpSize}) {
    const std::uint32_t threads = num_warps * fb::kWarpSize;
    const std::vector<T> in = make_inputs<T>(threads);
    for (const std::uint32_t nactive : nactive_cases(threads)) {
      for (const FoldOp op : kOps) {
        SCOPED_TRACE(std::format("{} warps, nactive {}, {}", num_warps, nactive, op_name(op)));
        std::vector<T> out(threads);
        ASSERT_TRUE(tk::block_reduce_on_device(op, num_warps, in.data(), nactive, out.data()));
        expect_all_equal(out, reference(op, in, nactive));
      }
    }
  }
}

TEST(ReduceGpu, Double) {
  FBF_REQUIRE_GPU();
  check_block_reduce<double>();
}

TEST(ReduceGpu, Int32) {
  FBF_REQUIRE_GPU();
  check_block_reduce<std::int32_t>();
}

// A NaN in an active thread wins the maximum; one past nactive is ignored.
TEST(ReduceGpu, MaxNanSeesOnlyActiveNaN) {
  FBF_REQUIRE_GPU();
  const std::uint32_t num_warps = 4;
  const std::uint32_t threads = num_warps * fb::kWarpSize;
  const std::uint32_t nactive = threads - 3;
  std::vector<double> in = make_inputs<double>(threads);
  std::vector<double> out(threads);

  in[threads - 1] = std::numeric_limits<double>::quiet_NaN();
  ASSERT_TRUE(
      tk::block_reduce_on_device(FoldOp::MaxNan, num_warps, in.data(), nactive, out.data()));
  expect_all_equal(out, reference(FoldOp::MaxNan, in, nactive));
  EXPECT_FALSE(std::isnan(out[0]));

  in[fb::kWarpSize + 1] = std::numeric_limits<double>::quiet_NaN();
  ASSERT_TRUE(
      tk::block_reduce_on_device(FoldOp::MaxNan, num_warps, in.data(), nactive, out.data()));
  EXPECT_TRUE(std::isnan(out[0]));
  expect_all_equal(out, out[0]);
}

} // namespace
