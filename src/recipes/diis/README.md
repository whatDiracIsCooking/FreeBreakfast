# DIIS

Pulay DIIS (direct inversion of the iterative subspace) extrapolation on the GPU: a ring of iterates and residuals, an incrementally updated residual Gram, a one-block coefficient solve, and a `gemv` extrapolation. It goes through WarpWraps, so one source builds for CUDA and HIP.

## Module

`free_breakfast.recipes.diis`, also re-exported by `free_breakfast.recipes`.

## Design

DIIS accelerates a fixed-point iteration by replacing the current iterate with a combination `F* = sum_i c_i F_i` of recent ones. The coefficients minimize the norm of the same combination of their residuals `e_i`, subject to `sum_i c_i = 1`, which is the bordered system

```
[ <e_i|e_j>  -1 ] [ c ]   [ 0  ]
[   -1^T      0 ] [ l ] = [ -1 ]
```

For SCF, `F_i` is the Fock matrix and `e_i` the orthogonal-basis commutator `Xᵀ (F D S − S D F) X`, which vanishes at self-consistency. `Diis` sees neither: it holds opaque length-`len` vectors. That is also why `len` is an element count rather than a matrix order. One n×n Fock is `n*n`. An alpha/beta pair concatenated into one vector is `2*n*n`, and both spins are then extrapolated with one shared coefficient set.

**Ring storage.** The iterate and residual histories are each one column-major `(len × cap)` device buffer, one pair per column, with leading dimension `len`. A push appends while the history grows and, once it is full, overwrites the *oldest* column and advances a `head_` cursor, so eviction moves no data. Eviction starts only once the ring is full, so the valid columns are always the prefix `[0, size)` or the whole buffer, never a wrapped range. Every BLAS call below therefore sees one base pointer and a fixed stride. The columns are in physical (rotated) order, not age order.

**Incremental Gram.** `B_ij = <e_i, e_j>` lives in a `cap × cap` buffer with fixed leading dimension `cap`, indexed by physical column. A push changes only the products that involve the column it wrote (`slot`), so two `gemv`s with `trans = T` recompute that row and column directly into the upper triangle the solve reads, at `min(j, slot)·cap + max(j, slot)`:

- `j = 0..slot`: base `gram + slot`, `incy = cap`;
- `j = slot..m−1`: base `gram + slot·cap + slot`, `incy = 1`.

Both write the diagonal, with the same value. This runs on every push, including the first. So each entry `[a][c]` is rewritten by the push that wrote column `max(a, c)`, and `reset()`, which keeps the buffers, can never expose a stale entry.

**Coefficient solve.** The `(m+1)`-square bordered system is solved on the device by `diis_solve` (`diis_solve.cu`). It is one block of `kWarpSize` threads that builds the system in shared memory, striding by `blockDim.x` so a 32-wide warp and a 64-wide wavefront take the same path. Thread 0 then runs Gauss–Jordan elimination with partial pivoting. The system is the history depth plus one, far too small to parallelize. The shared memory is static, sized for the largest system: `kDiisMaxHistoryLimit` (64) bounds it at `65·66` doubles, under CUDA's 48 KiB static per-block limit, and `diis_solve_bridge.h`'s `kDiisSolveMaxDim` is that bound on the device side (`diis.cpp` asserts the two agree). The constructor clamps `max_history` to `1..64`.

**Singular subspace.** If a pivot falls below `1e-14` the kernel writes the unit selector at the newest physical column, so the extrapolation reproduces the newest iterate, which is the plain, unextrapolated step. Under the ring that column is `slot`, not `m − 1`. No host branch is needed for this.

**Extrapolation.** One `gemv`, `out = F_hist · c` with `beta = 0`, over physical columns `0..m−1`. The sum is order-invariant, so the rotated column order needs no unscrambling. `out` may alias the current iterate, because that has already been copied into the ring. Since the full-phase basis is a symmetric permutation of the age-ordered one, the result agrees with an age-ordered solve to rounding, not bit for bit.

**Pointer mode.** Every `alpha`/`beta` is a host constant, and the coefficients are `gemv`'s device *data* operand, which neither pointer mode governs. So the step runs in the BLAS handle's default host pointer mode with no save and restore.

**One host round-trip.** With two or more pairs, the push copies back the kernel's singular flag and synchronizes the stream once, so `last_extrapolation_singular()` can feed a numerical-health warning. Nothing else leaves the device. An SCF iteration already synchronizes for its energy and convergence norms, so this adds a few bytes to a round-trip the caller pays anyway.

**Errors.** A `res` on another device, or a `len` past `INT32_MAX` (the BLAS dimension type), is `InvalidConfig` and leaves the history untouched. A failed copy, BLAS call or launch is `GpuRuntime` and resets the history, because a column or Gram entry may be half-written.

**Device.** The constructor allocates every buffer on `res.device()`. Each push makes `res.dev_idx()` current for its duration (`ScopedDeviceIndex`) and restores the previous device afterwards.

**`double` only.** An SCF Fock is double, and the `1e-14` singular threshold is a double-precision value. A `float` variant would need its own threshold.

## Public API

| Symbol | Role |
|---|---|
| `kDiisMaxHistoryLimit` | 64: the largest history kept. |
| `Diis(res, len, max_history)` | Allocate the ring, Gram and scratch on `res`'s device. A failed allocation aborts. |
| `push_and_extrapolate(res, cur_fock, cur_residual, out_fock)` | `Status`: push the pair and, with two or more held, write `F*` to `out_fock`; with one, leave it untouched. |
| `reset()` | Drop every pair and keep the buffers. Call it at the start of each run, so repeated runs on one object match a fresh one. |
| `size()`, `capacity()`, `length()` | Pairs held, the clamped cap, and `len`. |
| `last_extrapolation_singular()` | Whether the last extrapolating push fell back to the newest iterate. |

## Usage

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;
namespace fb = free_breakfast;

fb::DeviceResources res{0};
fb::Diis diis{res, n * n, 8};
diis.reset();                       // once per SCF run
for (/* each SCF iteration */;;) {
  // ... build fock and residual (n*n device doubles) on res.stream() ...
  fb::Status status = diis.push_and_extrapolate(res, fock, residual, fock);
  if (diis.last_extrapolation_singular()) { /* log a numerical-health warning */ }
  // ... diagonalize fock ...
}
```

## Tests

`test/gtest/diis_gpu.cpp` covers the capacity clamp, the untouched output on the first push, agreement with a host solve of the age-ordered system through eleven pushes into a ring of four, the singular fallback, `reset()`, an aliased output, and a linear fixed point `x = A x + b`. On a linear map DIIS is GMRES, so it must converge within `len + 1` steps.

## Files

| File | Role |
|------|------|
| `diis.cppm` | Module interface: `Diis` and `kDiisMaxHistoryLimit` |
| `diis.cpp` | The constructor and `push_and_extrapolate` |
| `diis_solve.cu` | The bordered-solve kernel and its launcher |
| `diis_solve_bridge.h` | The launcher's declaration, shared by the `.cu` and the implementation unit's GMF |
| `CMakeLists.txt` | The device library `free_breakfast.recipes.diis.device` and the module |
