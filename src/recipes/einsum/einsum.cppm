/// @file
/// @brief A NumPy-style einsum on the GPU, accumulating into its output: one
/// generic contraction kernel driven by the subscript string, and the
/// transpose as the one-operand case of it.
///
/// The README has the limits, the error tier and why this is not
/// cuTENSOR / hipTensor.

export module free_breakfast.recipes.einsum;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;

export namespace free_breakfast {

/// @brief One operand of einsum: device data and its extents.
struct EinsumOperand {
  /// @brief The operand's first element, on the device.
  const double *data = nullptr;
  /// @brief Its extents, one per subscript label, outermost first.
  std::vector<int64_t> dims;
  /// @brief Its strides in elements, one per axis; empty means packed
  /// row-major over @ref dims. A view along any axis of a larger array is an
  /// operand with that array's strides.
  std::vector<int64_t> strides;
};

/// @brief @p out += @p alpha * einsum(@p subscripts, @p operands...), queued on
/// @p stream.
///
/// @p subscripts is NumPy's explicit form, "ij,jk->ik": one label group per
/// operand, then "->" and the output's labels. Labels are ASCII letters. A
/// label in the output is kept; one that is not is summed over. A label may
/// repeat within an operand (a diagonal, "ii->i") but not in the output, and
/// an empty output ("ij,ij->") is a scalar. There is no implicit form and no
/// "...". At most 3 operands, rank 6 for any operand or the output, 6 summed
/// labels and 8 distinct labels in all.
///
/// @p out is packed row-major over @p out_dims, which must be the extents the
/// output labels take ({} for a scalar). It is accumulated into, never
/// cleared: for a plain product, zero it first. It must not overlap an
/// operand. Every term is added with an atomicAdd, so the sum's order varies
/// from run to run, and two runs agree to rounding, not bit for bit.
///
/// The caller's device must be current, and the call does not synchronize.
/// Nothing is read or launched until the subscripts are checked, so a refusal
/// touches no device.
///
/// @return InvalidConfig for malformed subscripts, ones that do not fit the
///         operands' ranks, extents or @p out_dims, or a call past the limits;
///         else the launch status.
[[nodiscard]] Status einsum(wwr::wwrStream_t stream, std::string_view subscripts,
                            const std::vector<EinsumOperand> &operands, double *out,
                            const std::vector<int64_t> &out_dims, double alpha = 1.0);

/// @brief @p out += @p alpha * @p x transposed by @p axes, queued on
/// @p stream: NumPy's `x.transpose(axes)`, axis i of the result being axis
/// axes[i] of @p x.
///
/// It is einsum with one operand and nothing summed, so it takes einsum's
/// limits and conventions: @p out is packed row-major, accumulated into, and
/// must not overlap @p x.
///
/// @return InvalidConfig when @p axes is not a permutation of @p x's axes or
///         @p x's rank is past einsum's; else the launch status.
[[nodiscard]] Status transpose(wwr::wwrStream_t stream, const EinsumOperand &x,
                               const std::vector<int32_t> &axes, double *out, double alpha = 1.0);

} // namespace free_breakfast
