# Norm

A chosen norm of a real device vector: ℓ₁ (`asum`), ℓ₂ (`nrm2`) or ℓ∞ (`iamax`, then that element's magnitude), picked by a `Norm` template argument. It goes through WarpWraps, so one source builds for CUDA and HIP.

## Module

`free_breakfast.recipes.norm`, also re-exported by `free_breakfast.recipes`.

## Design

`norm<which, T>(res, n, x, incx, result)` writes the `which` norm over the `n` elements of `x` (stride `incx`) to `result`. An `n` of zero or less writes 0.

**`result` follows the handle's pointer mode, as a BLAS reduction's does.** In host mode (`res.blas()`'s default) `result` is a host pointer and the call has waited for it. In device mode it is a device pointer, and nothing synchronizes. The routine reads the mode and never changes it.

**ℓ∞ is the one norm that is not a single BLAS call.** `iamax` returns a 1-based *index*, and the value still has to be read and its sign dropped. In host mode the index is on the host, so the element is copied back and the stream synced once more. In device mode the index lands in an `int32_t` from `res.device()`'s pool, and `abs_at.cu` reads `|x[(*at - 1) * incx]|` into `result` on the device. BLAS has no call that reads an element back by a device-held index, so the kernel fills that gap. Neither path touches `res.device_workspace()`, so `x` may point into it.

**Real types only.** For a complex vector, `iamax` ranks elements by |Re| + |Im| rather than by modulus, so the element it picks need not have the largest |z|. `asum` has the same flaw: it sums |Re| + |Im|, not |z|. And the result's type would be the real component type, not `T`. No caller needs the complex case, so it is left out rather than made correct.

Ported from calaman's `diff_norm`, without its `y := y − x` axpy (and so without the host-mode guard that axpy's scalar needed), plus the `set_element_abs` kernel behind its device-mode ℓ∞. calaman wrote nothing for `n == 0`; this writes 0.

## Public API

| Symbol | Role |
|---|---|
| `Norm` | `l1`, `l2`, `inf` |
| `norm<which, T>(res, n, x, incx, result)` | `Status`: the `which` norm of `x` into `result`, host or device per the pointer mode. Instantiated for each `Norm` × `float`/`double`. |

## Usage

```cpp
import free_breakfast.recipes;
namespace fb = free_breakfast;

double top = 0.0;  // host pointer mode: the call has waited
fb::Status status = fb::norm<fb::Norm::inf>(res, n, d_x.data(), 1, &top);
```

## Consumers

`OrbitalHessian::capped_density` (`mcscf/hessian`): ℓ∞ of a transition density, for its prune cap.

## Tests

`test/gtest/norm_gpu.cpp`: every `Norm` in both precisions and both pointer modes, contiguous and strided (with a larger element off the stride), against a host reference. ℓ∞ is checked exactly, ℓ₁ and ℓ₂ to a tolerance. Also an empty vector writing 0 in each mode, and the handle's mode left as it was.

## Files

| File | Role |
|------|------|
| `norm.cppm` | Module interface: `Norm` and `norm` |
| `norm.cpp` | Implementation, and the `Norm` × `float`/`double` instantiations |
| `abs_at.cu` | The one-item kernel reading the magnitude at a device-held 1-based index |
| `abs_at_bridge.h` | The launcher's declaration, shared by the `.cu` and the implementation unit's GMF |
| `CMakeLists.txt` | The device library `free_breakfast.recipes.norm.device` and the module |
