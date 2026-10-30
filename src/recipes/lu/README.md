# LU

`lu_solve(res, n, nrhs, a, b)`: A X = B for a general square A, by the LU
factorization with partial pivoting, P A = L U. One dense-solver `getrf`, its
flag read, then one `getrs`, through WarpWraps, so one source builds for CUDA
and HIP. It returns `getrf`'s flag: 0, or the 1-based index of the first
exactly zero U(i, i), in which case B is left as it was. Link
`free_breakfast::recipes::lu`.

## Why

The tree's other factorization, `recipes::cholesky`, needs A symmetric
positive definite. The RIS-J preconditioner's capacitance matrix
(`../../cis/ris_precondition/`) is symmetric but indefinite, so it needs a
pivoted one; an LU is the one both dense solvers carry.

**The legacy pair.** This calls the int-based `getrf`/`getrs`, as
`diagonalize` calls the int-based `syevd`: their pivots are `int32_t`, their
work one device buffer, and no params object or host workspace is needed.

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
may point into it; growing it is the one allocation the call can make.
`res.info()` takes both flags, and reading each synchronizes `res.stream()`.
An `info < 0` (a rejected argument, so a bug here) is `GpuRuntime`.

## Tests

`test/gtest/lu_gpu.cpp`: against a host pivoted elimination on random
symmetric indefinite and general matrices at orders 1 to 130 with three
right-hand sides, to 1e-12 relative; bitwise repeatable over 20 repeats; a
zero column flagged at its pivot with B untouched; the empty and negative
shapes.

## Files

| File | Role |
|------|------|
| `lu.cppm` | Module interface: `lu_solve` |
| `lu.cpp` | Implementation |
| `CMakeLists.txt` | The module `free_breakfast.recipes.lu` |
