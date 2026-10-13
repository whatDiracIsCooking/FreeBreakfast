// Host-only cases for core/wwr's :policy. The policies only ever look
// at an error code, so every case here drives them with codes constructed on
// the host.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;

namespace {

namespace fb = free_breakfast;
namespace ext = wwr::extension;
using wwr::wwrError_t;

// Every policy is empty and fits every wwr slot, including the destructor ones.
static_assert(std::is_empty_v<fb::AbortPolicy<wwrError_t>>);
static_assert(std::is_empty_v<fb::LogPolicy<wwrError_t>>);
static_assert(ext::typed_error_policy<fb::CreatePolicy<wwrError_t>>);
static_assert(ext::nothrow_error_policy<fb::DestroyPolicy<wwrError_t>, wwrError_t>);
static_assert(ext::policy_map<fb::PolicyMap, wwrError_t>);
static_assert(ext::policy_map<fb::PolicyMap, ext::stdHostMemoryError_t>);

// Creation aborts; destruction logs.
static_assert(std::same_as<fb::CreatePolicy<wwrError_t>, fb::AbortPolicy<wwrError_t>>);
static_assert(std::same_as<fb::DestroyPolicy<wwrError_t>, fb::LogPolicy<wwrError_t>>);

// The aliases instantiate (checked here rather than only on a GPU box).
static_assert(sizeof(fb::Buffers::device<float>) > 0);
static_assert(sizeof(fb::BlasHandle) > 0);
static_assert(sizeof(fb::SolverHandle) > 0);

TEST(GpuPolicy, LogPolicyDoesNotAbort) {
  fb::LogPolicy<wwrError_t>{}.handle_error(wwr::wwrSuccess, std::source_location::current());
  fb::LogPolicy<wwrError_t>{}.handle_error(wwr::wwrErrorInvalidValue,
                                           std::source_location::current());
}

// As in diagnostics_host.cpp: the critical line goes to stdout, which a death
// test does not match, so this asserts the abort only.
TEST(GpuPolicyDeathTest, AbortPolicyAbortsOnFailureOnly) {
  fb::AbortPolicy<wwrError_t>{}.handle_error(wwr::wwrSuccess, std::source_location::current());
  EXPECT_DEATH(fb::AbortPolicy<wwrError_t>{}.handle_error(wwr::wwrErrorInvalidValue,
                                                          std::source_location::current()),
               "");
}

} // namespace
