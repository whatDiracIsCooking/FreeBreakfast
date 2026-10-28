# Symmetrize

A square device matrix made symmetric in place, on the GPU. It goes through WarpWraps, so one source builds for CUDA and HIP.

## Module

`free_breakfast.recipes.symmetrize`, also re-exported by `free_breakfast.recipes`.

## Design

| Call | Effect |
|---|---|
| `symmetrize(res, n, a, lda)` | A ← (A + Aᵀ) / 2 |
| `symmetrize(res, fill, n, a, lda)` | The triangle opposite `fill` ← the transpose of `fill`'s |

Both take the column-major `n` × `n` block at `a` with leading dimension `lda` (≥ `n`); padding rows are neither read nor written. Each is one launch on `res.stream()`, with `res.dev_idx()` current for the call, and returns without synchronizing. `n = 0` launches nothing and succeeds, whatever `lda`. `fill` is `WWRBLAS_FILL_MODE_LOWER` or `_UPPER`, BLAS's own vocabulary, so a syrk's fill mode passes straight through.

**Why not geam.** A ← (A + Aᵀ) / 2 is one geam, but geam may not write over an operand it transposes, so in place it needs a second n² matrix and a copy back. And BLAS has no call that fills the triangle a syrk leaves; the workaround is a zeroed scratch, L + Lᵀ by geam, and a strided copy to undo the doubled diagonal. Each of those is one call here, with no scratch.

**Both results are exactly symmetric.** The mirror is a copy. The average writes (i, j) and (j, i) from two threads adding the same pair in opposite orders, (a + b) · ½ and (b + a) · ½, which are the same bits: IEEE addition commutes, and a sum times a constant has no product to contract into an FMA. A symmetric input comes back unchanged.

**Tiles, not warps.** Each block owns a 32 × 32 tile below the diagonal and, off it, its mirror above, both staged in shared memory (32 × 33 doubles each, the extra column keeping a row read off one bank) so global reads and writes both run down columns. The grid is the full T × T over tiles, and the blocks above the diagonal return at once. A diagonal tile is its own mirror. The tile is fixed at 32 rather than `kWarpSize`: two 64-wide tiles on a wave64 build would be 66 KiB, past CDNA's 64 KiB of LDS. Eight warps per block (32 × 8 threads, four columns each) keep enough loads in flight; one warp per block would be held to a third of an sm_86 SM by its 16-resident-block limit.

## Tests

`test/gtest/symmetrize_gpu.cpp`, at orders around the tile's edges (1, 2, 31–33, 63–65, 100, 129) in a padded block: the average against the host's own sum and halving bit for bit, a symmetric input unchanged, both mirrors as exact copies over a NaN-filled destination, the padding untouched throughout, and the argument checks.

## Files

| File | Role |
|------|------|
| `symmetrize.cppm` | Module interface: both `symmetrize` overloads |
| `symmetrize.cpp` | Implementation: the argument checks and one launch each |
| `symmetrize.cu` | The two tile kernels |
| `symmetrize_bridge.h` | The launchers' declarations, shared by the `.cu` and the implementation unit's GMF |
| `CMakeLists.txt` | The device library `free_breakfast.recipes.symmetrize.device` and the module |
