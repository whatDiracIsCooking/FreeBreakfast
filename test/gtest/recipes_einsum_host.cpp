// Host cases for recipes.einsum: every refusal. einsum checks its subscripts
// before it reads or launches anything, so each call here goes to a null
// stream with null data and never reaches a device; a call that got past the
// checks would launch on no device and fail, not pass. The limits, 3 operands,
// rank 6, 6 summed and 8 distinct labels, are written out here rather than
// read from the plan's header, so a change to them shows up as a test change.

#include <gtest/gtest.h>

import std;
import free_breakfast.core;
import free_breakfast.recipes.einsum;

namespace {

namespace fb = free_breakfast;

// A packed operand of the given extents, at no address.
fb::EinsumOperand shaped(std::vector<std::int64_t> dims) {
  return {nullptr, std::move(dims), {}};
}

// The refusal of einsum(subscripts, operands) -> out_dims: InvalidConfig, and
// the message, which the caller checks names the reason.
std::string einsum_refusal(const std::string_view subscripts,
                           const std::vector<fb::EinsumOperand> &operands,
                           const std::vector<std::int64_t> &out_dims) {
  const fb::Status status = fb::einsum(nullptr, subscripts, operands, nullptr, out_dims);
  EXPECT_FALSE(status.has_value()) << subscripts << " was accepted";
  if (status.has_value()) {
    return {};
  }
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  return status.error().message;
}

std::string transpose_refusal(const fb::EinsumOperand &x, const std::vector<std::int32_t> &axes) {
  const fb::Status status = fb::transpose(nullptr, x, axes, nullptr);
  EXPECT_FALSE(status.has_value()) << "the axes were accepted";
  if (status.has_value()) {
    return {};
  }
  EXPECT_EQ(status.error().kind, fb::ErrorKind::InvalidConfig);
  return status.error().message;
}

TEST(RecipesEinsumRefusals, MalformedSubscripts) {
  const auto m = shaped({2, 3});
  const auto n = shaped({3, 4});
  EXPECT_NE(einsum_refusal("ij,jk", {m, n}, {2, 4}).find("no '->'"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij,jk->i k", {m, n}, {2, 4}).find("output labels"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij,j1->ik", {m, n}, {2, 4}).find("input labels"), std::string::npos);
  EXPECT_NE(einsum_refusal("...j,jk->ik", {m, n}, {2, 4}).find("input labels"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij->ji->ij", {m}, {3, 2}).find("output labels"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij,jk,kl->il", {m, n}, {2, 4}).find("3 label groups for 2"),
            std::string::npos);
  EXPECT_NE(einsum_refusal("ij->ii", {m}, {2, 2}).find("'i' repeats"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij->k", {m}, {2}).find("'k' is on no operand"), std::string::npos);
}

TEST(RecipesEinsumRefusals, SubscriptsThatDoNotFitTheOperands) {
  const auto m = shaped({2, 3});
  const auto n = shaped({3, 4});
  EXPECT_NE(einsum_refusal("ijk,jk->i", {m, n}, {2}).find("operand 0 3 labels"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij,ik->jk", {m, n}, {3, 4}).find("label 'i' is 3"), std::string::npos);
  EXPECT_NE(einsum_refusal("ii->i", {m}, {2}).find("label 'i' is 3"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij->ij", {shaped({2, -1})}, {2, 0}).find("extent -1"),
            std::string::npos);
  EXPECT_NE(einsum_refusal("ij->ij", {{nullptr, {2, 3}, {3}}}, {2, 3}).find("1 strides"),
            std::string::npos);
  EXPECT_NE(einsum_refusal("ij,jk->ik", {m, n}, {2, 4, 1}).find("rank-2 output"),
            std::string::npos);
  EXPECT_NE(einsum_refusal("ij,jk->ik", {m, n}, {2, 5}).find("axis 1 4 long"), std::string::npos);
  EXPECT_NE(einsum_refusal("ij,ij->", {m, m}, {1}).find("rank-0 output"), std::string::npos);
}

TEST(RecipesEinsumRefusals, OperandCount) {
  const auto v = shaped({2});
  EXPECT_NE(einsum_refusal("->", {}, {}).find("0 operands"), std::string::npos);
  EXPECT_NE(einsum_refusal("i,i,i,i->i", {v, v, v, v}, {2}).find("4 operands"), std::string::npos);
}

TEST(RecipesEinsumRefusals, PastEachLimit) {
  // Rank 7, on an operand and on the output alone.
  EXPECT_NE(einsum_refusal("abcdefg->a", {shaped({1, 1, 1, 1, 1, 1, 1})}, {1})
                .find("operand 0 has rank 7"),
            std::string::npos);
  EXPECT_NE(einsum_refusal("abcd,efg->abcdefg", {shaped({1, 1, 1, 1}), shaped({1, 1, 1})},
                           {1, 1, 1, 1, 1, 1, 1})
                .find("output has rank 7"),
            std::string::npos);
  // Seven summed labels, seven distinct in all.
  EXPECT_NE(einsum_refusal("abcd,efg->", {shaped({1, 1, 1, 1}), shaped({1, 1, 1})}, {})
                .find("sums 7 labels"),
            std::string::npos);
  // Nine distinct labels, six of them summed.
  EXPECT_NE(
      einsum_refusal("abcdef,ghi->ghi", {shaped({1, 1, 1, 1, 1, 1}), shaped({1, 1, 1})}, {1, 1, 1})
          .find("9 distinct labels"),
      std::string::npos);
}

TEST(RecipesEinsumRefusals, TransposeAxesThatAreNotAPermutation) {
  const auto x = shaped({2, 3, 4});
  EXPECT_NE(transpose_refusal(x, {1, 0}).find("2 axes for an x of rank 3"), std::string::npos);
  EXPECT_NE(transpose_refusal(x, {0, 0, 1}).find("not a permutation"), std::string::npos);
  EXPECT_NE(transpose_refusal(x, {0, 1, 3}).find("not a permutation"), std::string::npos);
  EXPECT_NE(transpose_refusal(x, {0, -1, 2}).find("not a permutation"), std::string::npos);
  EXPECT_NE(transpose_refusal(shaped({1, 1, 1, 1, 1, 1, 1}), {0, 1, 2, 3, 4, 5, 6}).find("rank 7"),
            std::string::npos);
}

} // namespace
