# Einsum

`einsum(stream, "ij,jk->ik", {{a, {m, k}}, {b, {k, n}}}, c, {m, n}, alpha)`:
`c += alpha * einsum(...)`, NumPy's explicit subscripts on the GPU, and
`transpose(stream, x, axes, out, alpha)`: `out += alpha * x.transpose(axes)`,
which is the one-operand einsum `"abc->cab"`. Link
`free_breakfast::recipes::einsum`.

Ported from the NEVPT2 demo's `src/einsum` (`nevpt2-gpu-demo`), so that its
SC-NEVPT2 energy ports as literal transcriptions of the NumPy reference's
subscript strings rather than re-derived index arithmetic per call.

## What it takes

- **Subscripts**: one label group per operand, `->`, the output's labels.
  Labels are ASCII letters. A label in the output is kept, any other is summed.
  A label may repeat within an operand (a diagonal, `"ii->i"`, `"mjjn->mn"`),
  never in the output; an empty output is a scalar. No implicit form, no `...`.
- **Operands**: an `EinsumOperand` is the data, its extents and, for a view,
  its strides (empty: packed row-major). A slice along any axis of a larger
  array is an operand with that array's strides.
- **Output**: packed row-major over `out_dims`, which must be the extents the
  output labels take. It is **accumulated into**, never cleared: zero it first
  for a plain product. It must not overlap an operand.
- **Limits**: 3 operands, rank 6 (operands and output), 6 summed labels and 8
  distinct labels in one call. These are the demo's measured maxima over
  every call site SC-NEVPT2 has, not a round number; the plan
  (`EinsumPlan`, `einsum_bridge.h`) is a fixed-size POD sized by them.

A call outside any of that is refused with `InvalidConfig`, naming the
subscripts, before anything is read or launched. That is the value tier
(`docs/style.md` "Errors") rather than the demo's abort: the subscripts are
literals, so a refusal is the caller's bug, but the caller decides what to do
with it, and the refusals are testable on a host with no device. That is also
why the call takes a bare stream rather than a `DeviceResources`; it needs no
library handle, and the caller's device must be current.

## Kernel

One thread per (output element, summed tuple), each decoding its slots
row-major, multiplying its operands' elements and adding the term to its
output element with `atomicAdd`, in blocks of `kFourWarps`. One thread per
output element with a loop over the sum would idle nearly the whole card on
SC-NEVPT2's small-output, large-sum calls (Sr's `"ipqr,pqrabc,iabc->i"` has an
output of nvirt and six summed active labels); spreading the sum over threads
keeps it busy whichever side dominates.

The atomics reassociate the sum, and the order varies from run to run, so two
runs agree to rounding, not bit for bit, and every test compares at a stated
tolerance.

## Why not cuTENSOR / hipTensor

wwr wraps both (`wwr.tensor`), and the demo probed them: on the HIP side
hipTensor has no fp64 contraction on gfx12 (this tree's gfx1200), and linking
`libhiptensor` aborts the process at load. A routine that exists on one
backend only is not one this tree can use, so the kernel here is
home-brewed until that changes.

## Tests

- `test/gtest/recipes_einsum_gpu.cpp`: matmul, diagonal, trace, a full
  reduction to a scalar, a strided view, transpose, a three-operand
  contraction, the small-output / large-sum shape `"ipqr,pqrabc,iabc->i"`, and
  the accumulate / scale semantics, each against a host loop at a tolerance.
- `test/gtest/recipes_einsum_host.cpp`: the refusals -- malformed subscripts,
  subscripts that do not fit their operands or `out_dims`, a transpose that is
  not a permutation, and a call past each limit.
