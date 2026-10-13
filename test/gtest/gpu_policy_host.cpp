// Host-only cases for core/gpu_policy. The policies only ever look at an error
// code, so every case here drives them with codes constructed on the host and
// names RecordPolicy / AbortPolicy explicitly -- the cases mean the same thing
// whichever mode this build resolved to. The one mode-dependent case checks
// that the CMake switch actually reached the module.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import wwr.runtime_api;

namespace {

namespace fb = free_breakfast;
namespace gpu = free_breakfast::gpu;
namespace ext = wwr::extension;
using wwr::wwrError_t;

// Every policy is empty and fits every wwr slot, including the destructor ones.
static_assert(std::is_empty_v<gpu::RecordPolicy<wwrError_t>>);
static_assert(std::is_empty_v<gpu::AbortPolicy<wwrError_t>>);
static_assert(std::is_empty_v<gpu::LogPolicy<wwrError_t>>);
static_assert(ext::typed_error_policy<gpu::CreatePolicy<wwrError_t>>);
static_assert(ext::nothrow_error_policy<gpu::DestroyPolicy<wwrError_t>, wwrError_t>);
static_assert(ext::nothrow_error_policy<gpu::RecordPolicy<wwrError_t>, wwrError_t>);
static_assert(ext::policy_map<gpu::PolicyMap, wwrError_t>);
static_assert(ext::policy_map<gpu::PolicyMap, ext::stdHostMemoryError_t>);

// The aliases instantiate (checked here rather than only on a GPU box).
static_assert(sizeof(gpu::Buffers::device<float>) > 0);
static_assert(sizeof(gpu::BlasHandle) > 0);
static_assert(sizeof(gpu::SolverHandle) > 0);

void record(wwrError_t code) {
  gpu::RecordPolicy<wwrError_t>{}.handle_error(code, std::source_location::current());
}

TEST(GpuPolicy, SwitchReachedTheModule) {
  EXPECT_EQ(gpu::aborts_on_error, FREE_BREAKFAST_TEST_EXPECT_GPU_ABORTS != 0);
  if constexpr (gpu::aborts_on_error) {
    EXPECT_TRUE((std::same_as<gpu::CreatePolicy<wwrError_t>, gpu::AbortPolicy<wwrError_t>>));
    EXPECT_TRUE((std::same_as<gpu::DestroyPolicy<wwrError_t>, gpu::AbortPolicy<wwrError_t>>));
  } else {
    EXPECT_TRUE((std::same_as<gpu::CreatePolicy<wwrError_t>, gpu::RecordPolicy<wwrError_t>>));
    EXPECT_TRUE((std::same_as<gpu::DestroyPolicy<wwrError_t>, gpu::LogPolicy<wwrError_t>>));
  }
}

TEST(GpuPolicy, CaptureReturnsValueWhenNothingFailed) {
  auto r = gpu::capture([] { return 7; });
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 7);

  fb::Status s = gpu::capture([] { record(wwr::wwrSuccess); });
  EXPECT_TRUE(s.has_value());
}

TEST(GpuPolicy, CaptureReturnsRecordedErrorWithCallSite) {
  int line = 0;
  auto r = gpu::capture([&] {
    line = std::source_location::current().line() + 1;
    auto loc = std::source_location::current();
    gpu::RecordPolicy<wwrError_t>{}.handle_error(wwr::wwrErrorInvalidValue, loc);
    return 7;
  });
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, fb::ErrorKind::GpuRuntime);
  EXPECT_FALSE(r.error().message.empty());
  EXPECT_EQ(r.error().origin.line(), static_cast<std::uint_least32_t>(line));
}

TEST(GpuPolicy, FirstErrorWins) {
  auto s = gpu::capture([] {
    record(wwr::wwrErrorInvalidValue);
    record(wwr::wwrErrorMemoryAllocation);
  });
  ASSERT_FALSE(s.has_value());
  auto expected = fb::gpu_status(wwr::wwrErrorInvalidValue);
  EXPECT_EQ(s.error().message, expected.error().message);
}

TEST(GpuPolicy, NestedCaptureNeitherLeaksNorWipes) {
  auto outer = gpu::capture([] {
    record(wwr::wwrErrorInvalidValue);
    auto inner = gpu::capture([] { return 1; });
    EXPECT_TRUE(inner.has_value()); // the outer error is not the inner's
    auto failing = gpu::capture([] { record(wwr::wwrErrorMemoryAllocation); });
    EXPECT_FALSE(failing.has_value());
  });
  ASSERT_FALSE(outer.has_value()); // ...and is still there afterwards
  EXPECT_EQ(outer.error().message, fb::gpu_status(wwr::wwrErrorInvalidValue).error().message);
  EXPECT_FALSE(gpu::take_error().has_value()); // nothing left behind
}

TEST(GpuPolicy, ThrowInsideCaptureRestoresOuterState) {
  auto outer = gpu::capture([] {
    record(wwr::wwrErrorInvalidValue);
    try {
      (void)gpu::capture([]() -> int {
        record(wwr::wwrErrorMemoryAllocation);
        throw std::runtime_error("boom");
      });
    } catch (const std::runtime_error &) {
    }
  });
  ASSERT_FALSE(outer.has_value());
  EXPECT_EQ(outer.error().message, fb::gpu_status(wwr::wwrErrorInvalidValue).error().message);
}

TEST(GpuPolicy, RecordOutsideCaptureIsKeptForTakeError) {
  record(wwr::wwrErrorInvalidValue);
  auto e = gpu::take_error();
  ASSERT_TRUE(e.has_value());
  EXPECT_EQ(e->kind, fb::ErrorKind::GpuRuntime);
  EXPECT_FALSE(gpu::take_error().has_value());
}

TEST(GpuPolicy, LogPolicyNeitherRecordsNorAborts) {
  auto s = gpu::capture([] {
    gpu::LogPolicy<wwrError_t>{}.handle_error(wwr::wwrErrorInvalidValue,
                                              std::source_location::current());
  });
  EXPECT_TRUE(s.has_value());
}

// As in diagnostics_host.cpp: the critical line goes to stdout, which a death
// test does not match, so this asserts the abort only.
TEST(GpuPolicyDeathTest, AbortPolicyAbortsOnFailureOnly) {
  gpu::AbortPolicy<wwrError_t>{}.handle_error(wwr::wwrSuccess, std::source_location::current());
  EXPECT_DEATH(gpu::AbortPolicy<wwrError_t>{}.handle_error(wwr::wwrErrorInvalidValue,
                                                           std::source_location::current()),
               "");
}

} // namespace
