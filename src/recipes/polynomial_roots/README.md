# Polynomial roots

Every root of a batch of polynomials of one degree, by the simultaneous Aberth-Ehrlich iteration, on the GPU. One warp solves one polynomial, one lane per root, so the degree is at most `kWarpSize`. It goes through WarpWraps, so one source builds for CUDA and HIP.

## Module

`free_breakfast.recipes.polynomial_roots`, also re-exported by `free_breakfast.recipes`.

## Design

| Call | Effect |
|---|---|
| `polynomial_roots<T, CT>(res, n, batch, coeffs, roots, iters, tol, max_iter)` | roots[b*n, b*n + n) ← the n roots of Σ_k coeffs[b*(n+1) + k] z^k; iters[b] ← the iterations taken, or -1 |

`T` is `float`, `double`, `wwrFloatComplex` or `wwrDoubleComplex`, and `CT` the complex type of the same precision. Coefficients are ascending with a_n ≠ 0, and 1 ≤ n ≤ `kWarpSize` (`free_breakfast.core.kernel`): 32 on NVIDIA and RDNA, 64 on CDNA. The call is one launch on `res.stream()`, with `res.dev_idx()` current for the call, and returns without synchronizing. An `n` or `batch` of zero launches nothing and succeeds.

**Non-convergence is per polynomial, not a `Status`.** A batch can hold polynomials that settle and some that do not (a NaN coefficient never settles), so `iters[b] == -1` is the report and the call still succeeds. The roots of an unsettled polynomial are its last estimates. A `Status` error means the arguments were wrong or the launch failed.

**The iteration.** Lane i owns root i, in a register. The estimates start on the circle of radius |a_0 / a_n|^(1/n), turned off the real axis so a real polynomial's conjugate pairs are not started symmetric. Each iteration every lane reads every other estimate by `shfl`, a tile-wide collective, so each lane has read them all before any lane moves its own: every root moves at once (Jacobi), the result does not depend on lane scheduling, and there is no shared memory and no barrier. A root settles when its relative correction is ≤ `tol` or |p(z)| is inside Horner's rounding bound n·ε·Σ|a_k||z|^k. The worst correction over the n owner lanes is folded with `core/kernel`'s `warp_reduce` (its `nactive` is n), NaN winning so a NaN never reads as converged. For |z| > 1 the polynomial is evaluated through its reversal in 1/z, so no |z|^n is formed.

**Arithmetic on a local (re, im) pair.** The vendor complex types have no portable operators, so `aberth.cu` reads coefficients into a file-local `Complex<R>` and writes roots back through `complex.h`'s accessors. Division and modulus are scaled, so |z|² is never formed, and both are the same algorithm on CUDA and HIP.

**One lane per root caps the degree at `kWarpSize`.** calaman held each warp's 2n estimates in dynamic shared memory, which allowed n up to 1536 but needed a `dynamic-smem-ok` exception and a ping-pong buffer. Registers and shuffles need neither. A block holds `kFourWarps / kWarpSize` polynomials.

Ported from calaman's `polynomial_roots` (`aberth`). Its `aberth_warp` was a public `.cuh` for a kernel solving a polynomial inside its own launch. That has no caller here, so it is file-local to `aberth.cu`. Lift it into a header with an interface target when a second kernel wants it.

## Tests

`test/gtest/polynomial_roots_gpu.cpp`, over all four coefficient types. Random Gaussian batches at degrees up to the full warp are certified without a reference solver: each root's backward error |p(z)| / Σ|a_k||z|^k is held to O(n·ε), and the Weierstrass inclusion disks |z − z_i| ≤ n·|p(z_i) / (a_n ∏_{j≠i}(z_i − z_j))| must be pairwise disjoint. Disjoint disks each hold exactly one root, so this proves that every root was found exactly once, and the radius bounds each one's forward error. Exact cases (roots of unity, a zero root) pin the forward error directly. Further cases cover non-convergence, an empty call and argument validation.

## Files

| File | Role |
|------|------|
| `polynomial_roots.cppm` | Module interface: `polynomial_roots` |
| `polynomial_roots.cpp` | Argument checks, and the four instantiations |
| `aberth.cu` | The kernel: one warp per polynomial, one lane per root, over `aberth_warp` |
| `aberth_bridge.h` | The launcher's declaration, shared by the `.cu` and the implementation unit's GMF |
| `CMakeLists.txt` | The device library `free_breakfast.recipes.polynomial_roots.device` and the module |
