# LU

`lu_solve(res, n, nrhs, a, b)`: A X = B for a general square A, by the LU
factorization with partial pivoting, P A = L U. One dense-solver `getrf`
through WarpWraps, its flag read, then a solve of the tree's own: `lu.cu`'s
kernel applies the pivots' row interchanges to B, and two calls of the
`../trsm/` recipe solve L (unit diagonal) and then U. One source builds for
CUDA and HIP. It returns `getrf`'s flag: 0, or the 1-based index of the first
exactly zero U(i, i), in which case B is left as it was. Link
`free_breakfast::recipes::lu`.

## Why

The tree's other factorization, `recipes::cholesky`, needs A symmetric
positive definite. The RIS-J preconditioner's capacitance matrix
(`../../cis/ris_precondition/`) is symmetric but indefinite, so it needs a
pivoted one; an LU is the one both dense solvers carry.

**The legacy getrf.** This calls the int-based `getrf`, as `diagonalize`
calls the int-based `syevd`: its pivots are `int32_t`, its work one device
buffer, and no params object or host workspace is needed.

**Why the solve is ours.** On HIP, hipSOLVER's `getrs` allocates about B's
size plus 42 MiB outside its size query, so outside the caller's budget, and
keeps it (`experimental/rust/blas_workspace_scaling/README.md`, "Solvers").
`getrf` stays inside its query on both backends. So `getrs` is replaced by
what it does: B <- P B (LAPACK's `laswp`, `lu.cu`: one thread per column of
B walking the 1-based pivots in order, row i swapped with row ipiv[i] - 1),
then `trsm` left, lower, unit diagonal, then `trsm` left, upper, non-unit,
both the tree's recipe, which takes nothing past the BLAS handle's fixed gemm
workspace. It is one code path on both backends.

**The race checked for.** rocSOLVER's small-matrix `potf2` kernel raced
between its waves under contention on another process's load (PR #410, and
`../cholesky/README.md`), which is why the tree does not call `potrf`. Its LU
was given the same check before anything used it, on ROCm 7.2.4 and the
gfx1200 card: 4 concurrent processes of 50 repeats each, then 8 of 25, of
`Lu.*`, `CisRisPrecondition.*` and `CisStep.RisMatchesPyscf`, every LU
compared against a host elimination and bitwise against its first run. All
3600 case runs passed. Were it ever to race, the fallback is a small pivoted
LU of the tree's own, as `cholesky` is.

## Workspace and flags

`getrf`'s work for n, queried per call, and the n pivots are
`res.device_workspace()`'s, carved by `device_regions`, so neither `a` nor `b`
may point into it; growing it is the one allocation the call can make. The
solve after `getrf` takes no scratch of its own. `res.info()` takes `getrf`'s
flag, and reading it synchronizes `res.stream()`; the solve is queued after
it and not waited for.
An `info < 0` (a rejected argument, so a bug here) is `GpuRuntime`.

`getrf_lwork(res, n)` is that query on its own, and `lu_solve_footprint(n,
lwork)` what a solve at order n takes of the workspace: getrf's work and the
pivots, nothing for the solve, each region rounded up to `DeviceResources::kWorkspaceAlignment`. The
footprint takes the size, not a handle, so a plan's footprint is computed
before the run (`FootprintDimensions::getrf_lwork`, `pipeline/shape`'s query
at the RIS basis's order).

## Tests

`test/recipes/lu_gpu.cpp`: against a host pivoted elimination on random
symmetric indefinite and general matrices at orders 1 to 130 with three
right-hand sides, to 1e-12 relative; a general matrix at order 300 with 40
right-hand sides, past the trsm recipe's 64-row block and its 16-column
tile, whose pivoting moves most rows, held to a backward-error bound; bitwise repeatable over 20 repeats; a
zero column flagged at its pivot with B untouched; the empty and negative
shapes; and `Lu.Footprint`, the workspace a solve grows held to
`lu_solve_footprint`'s, and its pool peak to nothing.

## Files

| File | Role |
|------|------|
| `lu.cppm` | Module interface: `lu_solve`, `getrf_lwork`, `lu_solve_footprint` |
| `lu.cpp` | Implementation |
| `lu.cu` | The row-interchange kernel |
| `lu_bridge.h` | Its launcher's declaration, shared by `lu.cpp` and `lu.cu` |
| `CMakeLists.txt` | The device library `free_breakfast.recipes.lu.device` and the module `free_breakfast.recipes.lu` |
