/// @file
/// @brief einsum: the subscripts checked against the operands and recorded as
/// the plan einsum.cu's kernel reads; transpose as the one-operand einsum.

module;

// The plan and its launcher, defined in einsum.cu: declared in the GMF so the
// launcher keeps external linkage (see the header).
#include "einsum_bridge.h"

module free_breakfast.recipes.einsum;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr; // gpu_status

#include "diagnostics.h"

namespace free_breakfast {

bool is_einsum_label(const char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

// The plan for @p subscripts over @p operands, and the output's extents, or
// why there is none. Every label's extent is taken where it first appears and
// checked everywhere else; the summed labels take their slots in that same
// first-appearance order, so a call's plan, and its launch, never depends on
// a hash.
Result<std::pair<device::EinsumPlan, std::vector<int64_t>>>
einsum_plan(const std::string_view subscripts, const std::vector<EinsumOperand> &operands) {
  const size_t arrow = subscripts.find("->");
  if (arrow == std::string_view::npos) {
    return err_config("einsum: '{}' has no '->'", subscripts);
  }
  const std::string_view lhs = subscripts.substr(0, arrow);
  const std::string_view rhs = subscripts.substr(arrow + 2);
  for (const char c : lhs) {
    if (c != ',' && !is_einsum_label(c)) {
      return err_config("einsum: '{}' has '{}' among its input labels", subscripts, c);
    }
  }
  for (const char c : rhs) {
    if (!is_einsum_label(c)) {
      return err_config("einsum: '{}' has '{}' among its output labels", subscripts, c);
    }
  }

  std::vector<std::string_view> groups;
  for (size_t begin = 0;;) {
    const size_t comma = lhs.find(',', begin);
    groups.push_back(lhs.substr(begin, comma - begin));
    if (comma == std::string_view::npos) {
      break;
    }
    begin = comma + 1;
  }
  if (operands.empty() || std::ssize(operands) > device::kEinsumMaxOperands) {
    return err_config("einsum: '{}' is given {} operands, and takes 1 to {}", subscripts,
                      operands.size(), device::kEinsumMaxOperands);
  }
  if (groups.size() != operands.size()) {
    return err_config("einsum: '{}' has {} label groups for {} operands", subscripts, groups.size(),
                      operands.size());
  }

  std::string labels;
  std::vector<int64_t> extents;
  for (size_t o = 0; o < operands.size(); ++o) {
    const EinsumOperand &op = operands[o];
    if (groups[o].size() != op.dims.size()) {
      return err_config("einsum: '{}' gives operand {} {} labels, and it has rank {}", subscripts,
                        o, groups[o].size(), op.dims.size());
    }
    if (std::ssize(op.dims) > device::kEinsumMaxRank) {
      return err_config("einsum: '{}' operand {} has rank {}, past the {} it takes", subscripts, o,
                        op.dims.size(), device::kEinsumMaxRank);
    }
    if (!op.strides.empty() && op.strides.size() != op.dims.size()) {
      return err_config("einsum: '{}' operand {} has {} strides for rank {}", subscripts, o,
                        op.strides.size(), op.dims.size());
    }
    for (size_t x = 0; x < op.dims.size(); ++x) {
      const char c = groups[o][x];
      const int64_t d = op.dims[x];
      if (d < 0) {
        return err_config("einsum: '{}' operand {} has extent {} on axis {}", subscripts, o, d, x);
      }
      const size_t at = labels.find(c);
      if (at == std::string::npos) {
        labels.push_back(c);
        extents.push_back(d);
      } else if (extents[at] != d) {
        return err_config("einsum: '{}' label '{}' is {} on operand {} and {} before it",
                          subscripts, c, d, o, extents[at]);
      }
    }
  }

  if (std::ssize(rhs) > device::kEinsumMaxRank) {
    return err_config("einsum: '{}' output has rank {}, past the {} it takes", subscripts,
                      rhs.size(), device::kEinsumMaxRank);
  }
  std::string slots;
  for (const char c : rhs) {
    if (labels.find(c) == std::string::npos) {
      return err_config("einsum: '{}' output label '{}' is on no operand", subscripts, c);
    }
    if (slots.find(c) != std::string::npos) {
      return err_config("einsum: '{}' output label '{}' repeats", subscripts, c);
    }
    slots.push_back(c);
  }
  for (const char c : labels) {
    if (rhs.find(c) == std::string_view::npos) {
      slots.push_back(c);
    }
  }
  const auto contracted = std::ssize(slots) - std::ssize(rhs);
  if (contracted > device::kEinsumMaxContracted) {
    return err_config("einsum: '{}' sums {} labels, past the {} it takes", subscripts, contracted,
                      device::kEinsumMaxContracted);
  }
  if (std::ssize(slots) > device::kEinsumMaxLabels) {
    return err_config("einsum: '{}' has {} distinct labels, past the {} it takes", subscripts,
                      slots.size(), device::kEinsumMaxLabels);
  }

  // Every count is bounded by the limits above, so each narrows exactly.
  device::EinsumPlan plan;
  plan.operand_count = static_cast<int32_t>(operands.size());
  plan.out_rank = static_cast<int32_t>(rhs.size());
  plan.contracted_count = static_cast<int32_t>(contracted);
  std::vector<int64_t> out_dims;
  for (size_t s = 0; s < slots.size(); ++s) {
    const int64_t d = extents[labels.find(slots[s])];
    if (s < rhs.size()) {
      plan.out_dims[s] = d;
      plan.out_size *= d;
      out_dims.push_back(d);
    } else {
      plan.contracted_dims[s - rhs.size()] = d;
      plan.contracted_size *= d;
    }
  }
  for (size_t o = 0; o < operands.size(); ++o) {
    const EinsumOperand &op = operands[o];
    const auto rank = static_cast<int32_t>(op.dims.size());
    plan.operand_rank[o] = rank;
    int64_t packed = 1;
    for (int32_t x = rank - 1; x >= 0; --x) {
      const auto ux = static_cast<size_t>(x);
      plan.axis_label[o][x] = static_cast<int32_t>(slots.find(groups[o][ux]));
      plan.axis_stride[o][x] = op.strides.empty() ? packed : op.strides[ux];
      packed *= op.dims[ux];
    }
  }
  return std::pair{plan, std::move(out_dims)};
}

Status einsum(const wwr::wwrStream_t stream, const std::string_view subscripts,
              const std::vector<EinsumOperand> &operands, double *out,
              const std::vector<int64_t> &out_dims, const double alpha) {
  const auto [plan, dims] = FBF_TRY(einsum_plan(subscripts, operands));
  if (dims != out_dims) {
    if (dims.size() != out_dims.size()) {
      return err_config("einsum: '{}' makes a rank-{} output, and out has rank {}", subscripts,
                        dims.size(), out_dims.size());
    }
    const auto axis = static_cast<size_t>(std::ranges::mismatch(dims, out_dims).in1 - dims.begin());
    return err_config("einsum: '{}' makes output axis {} {} long, and out's is {}", subscripts,
                      axis, dims[axis], out_dims[axis]);
  }
  const auto operand = [&](const size_t o) {
    return o < operands.size() ? operands[o].data : nullptr;
  };
  return gpu_status(
      device::einsum_contract(stream, plan, operand(0), operand(1), operand(2), out, alpha));
}

Status transpose(const wwr::wwrStream_t stream, const EinsumOperand &x,
                 const std::vector<int32_t> &axes, double *out, const double alpha) {
  const size_t rank = x.dims.size();
  if (std::ssize(x.dims) > device::kEinsumMaxRank) {
    return err_config("transpose: x has rank {}, past einsum's {}", rank, device::kEinsumMaxRank);
  }
  if (axes.size() != rank) {
    return err_config("transpose: {} axes for an x of rank {}", axes.size(), rank);
  }
  // x's axes are labelled a, b, c, ... in order, and the output reads them in
  // the order axes names: "abc->cab" for axes {2, 0, 1}.
  std::string subscripts;
  for (size_t i = 0; i < rank; ++i) {
    subscripts.push_back(static_cast<char>('a' + i));
  }
  subscripts += "->";
  std::vector<int64_t> out_dims;
  for (const int32_t axis : axes) {
    const char label = static_cast<char>('a' + axis);
    if (axis < 0 || std::cmp_greater_equal(axis, rank) ||
        subscripts.find(label, rank) != std::string::npos) {
      return err_config("transpose: axes are not a permutation of x's {} axes", rank);
    }
    subscripts.push_back(label);
    out_dims.push_back(x.dims[static_cast<size_t>(axis)]);
  }
  return einsum(stream, subscripts, {x}, out, out_dims, alpha);
}

} // namespace free_breakfast
