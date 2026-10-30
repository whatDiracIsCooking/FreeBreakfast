# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------
# The headers every gtest tree #includes by bare name: gpu_requirement.h
# (FBF_REQUIRE_GPU), slow_test.h (FBF_SKIP_IF_FAST_TESTS), scratch_dir.h
# (ScratchDir), and io_basis_json.h -- the BSE fixture test/io's host cases and
# test/gtest's GPU ones both read, so neither tree reaches into the other.
# Header-only, so an INTERFACE target that puts this directory on a consumer's
# include path. GTest::gtest is linked because the two macros expand to
# GTEST_SKIP and both headers include gtest.h.
#
# The macros expand at the point of use, so they need the consumer's imports,
# not anything linked here: each header's own comment says which.
free_breakfast_add_interface_library(
  NAME free_breakfast.test.support LINK_PUBLIC GTest::gtest
)
