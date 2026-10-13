// Host-only cases for core/error_handling, core/log and diagnostics.h. None
// of these touch a device: gpu_status is exercised with codes constructed on
// the host, which is all it ever looks at.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import wwr.runtime_api;

// After the imports, because the macros expand against names they make visible.
#include "diagnostics.h"

namespace {

namespace fb = free_breakfast;

fb::Result<int> half(int n) {
  if (n % 2 != 0) {
    return fb::err_config(std::format("{} is odd", n));
  }
  return n / 2;
}

fb::Status require_positive(int n) {
  if (n <= 0) {
    return fb::err_numerical("not positive");
  }
  return {};
}

fb::Result<int> quarter_of_positive(int n) {
  FBF_TRY(require_positive(n));
  auto h = FBF_TRY(half(n));
  return FBF_TRY(half(h));
}

TEST(ErrorHandling, FactoriesSetKindAndMessage) {
  auto e = fb::Error::io("missing");
  EXPECT_EQ(e.kind, fb::ErrorKind::IO);
  EXPECT_EQ(e.message, "missing");
  EXPECT_EQ(fb::err_gpu("x").error().kind, fb::ErrorKind::GpuRuntime);
  EXPECT_EQ(fb::err_max_iterations_exceeded("x").error().kind,
            fb::ErrorKind::MaxIterationsExceeded);
}

TEST(ErrorHandling, OriginIsTheCallSite) {
  auto const line = std::source_location::current().line() + 1;
  auto e = fb::Error::allocation("oom");
  EXPECT_EQ(e.origin.line(), line);
  EXPECT_TRUE(std::string_view(e.origin.file_name()).ends_with("diagnostics_host.cpp"));
}

TEST(ErrorHandling, ExitCodesArePinnedAndDistinct) {
  using K = fb::ErrorKind;
  std::array const pinned{
      std::pair{K::InvalidConfig, 1}, std::pair{K::Allocation, 2},
      std::pair{K::GpuRuntime, 3},    std::pair{K::IO, 4},
      std::pair{K::Divergence, 5},    std::pair{K::MaxIterationsExceeded, 6},
      std::pair{K::Unsupported, 7},   std::pair{K::Numerical, 8},
  };
  for (auto [kind, code] : pinned) {
    EXPECT_EQ(fb::to_exit_code(kind), code);
  }
}

TEST(Diagnostics, TryUnwrapsOnSuccess) {
  auto r = quarter_of_positive(12);
  ASSERT_TRUE(r.has_value());
  EXPECT_EQ(*r, 3);
}

TEST(Diagnostics, TryPropagatesResultErrorUnchanged) {
  auto r = quarter_of_positive(6); // 6 -> 3, then 3 is odd
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig);
  EXPECT_EQ(r.error().message, "3 is odd");
  // The origin is where the error was made, not the FBF_TRY that forwarded it.
  EXPECT_EQ(std::string_view(r.error().origin.function_name()).find("quarter_of_positive"),
            std::string_view::npos);
}

TEST(Diagnostics, TryPropagatesStatusError) {
  auto r = quarter_of_positive(-4);
  ASSERT_FALSE(r.has_value());
  EXPECT_EQ(r.error().kind, fb::ErrorKind::Numerical);
}

TEST(Diagnostics, GpuStatusMapsSuccessAndFailure) {
  EXPECT_TRUE(fb::gpu_status(wwr::wwrSuccess).has_value());

  auto s = fb::gpu_status(wwr::wwrErrorInvalidValue);
  ASSERT_FALSE(s.has_value());
  EXPECT_EQ(s.error().kind, fb::ErrorKind::GpuRuntime);
  EXPECT_FALSE(s.error().message.empty());
}

TEST(Diagnostics, CheckOrAbortUnwrapsSuccess) {
  EXPECT_EQ(fb::check_or_abort(half(8)), 4);
  fb::check_or_abort(require_positive(1));
  FBF_GPU_MUST(wwr::wwrSuccess);
}

// The critical line goes to the logger's stdout sink, and a death test matches
// stderr only -- so these assert the abort, not the message.
TEST(DiagnosticsDeathTest, CheckOrAbortAbortsOnError) {
  EXPECT_DEATH(fb::check_or_abort(half(3)), "");
  EXPECT_DEATH(FBF_GPU_MUST(wwr::wwrErrorInvalidValue), "");
}

TEST(Log, ShouldLogFollowsSetLevel) {
  using L = fb::log::LogLevel;
  fb::log::set_level(L::warn);
  EXPECT_FALSE(fb::log::should_log(L::info));
  EXPECT_TRUE(fb::log::should_log(L::warn));
  EXPECT_TRUE(fb::log::should_log(L::critical));
  fb::log::set_level(L::info);
}

TEST(Log, WarnCountsEvenWhenFiltered) {
  fb::log::reset_warning_count();
  fb::log::set_level(fb::log::LogLevel::off);
  fb::log::warn("one");
  FBF_LOG_WARN("{}", "two");
  fb::log::set_level(fb::log::LogLevel::info);
  // log::warn counts unconditionally; the macro short-circuits before calling
  // it when warn is filtered, so only the direct call counts.
  EXPECT_EQ(fb::log::warning_count(), 1U);
  fb::log::reset_warning_count();
}

TEST(Log, MacrosDoNotFormatWhenFiltered) {
  fb::log::set_level(fb::log::LogLevel::info);
  int evaluated = 0;
  auto const touch = [&] { return ++evaluated; };
  FBF_LOG_DEBUG("{}", touch());
  EXPECT_EQ(evaluated, 0);
  FBF_LOG_INFO("{}", touch());
  EXPECT_EQ(evaluated, 1);
}

} // namespace
