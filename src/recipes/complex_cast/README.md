# Complex cast

Splitting a complex device matrix into real and imaginary planes, and merging two planes back, on the GPU. This is the packing a complex routine needs when its operands arrive, or its results must leave, as separate real arrays, for instance a complex product formed from real gemms. It goes through WarpWraps, so one source builds for CUDA and HIP.

## Module

`free_breakfast.recipes.complex_cast`, also re-exported by `free_breakfast.recipes`.

## Design

| Call | Effect |
|---|---|
| `split_planes<Z>(res, rows, cols, a, lda, re, im, ldp)` | re(i,j) ← Re(a(i,j)), im(i,j) ← Im(a(i,j)) |
| `merge_planes<Z>(res, rows, cols, re, im, ldp, c, ldc)` | c(i,j) ← (re(i,j), im(i,j)) |

Each walks a column-major `rows` × `cols` block, the complex side with its own leading dimension and both planes sharing `ldp`. Padding rows are neither read nor written, so a block of a larger matrix splits in place. A contiguous array of `count` elements is the block `count` × 1.

`Z` is `wwrFloatComplex` or `wwrDoubleComplex` (`wwr::complex_fp`), and the planes are `wwr::ComplexToRealType<Z>`: `float` or `double`. Each call is one `parallel_for` launch on `res.stream()`, with `res.dev_idx()` current for the call, and returns without synchronizing. A `rows` or `cols` of zero or less launches nothing and succeeds. Nothing alias-checks.

**Components go through `complex.h`'s accessors, never `.x`/`.y`.** `cuFloatComplex` is a `float2`, but `hipFloatComplex` is a class, and `wwrCreal*` / `wwrCimag*` / `make_wwr*Complex` are the spelling both backends share. The `.cu` overloads them per precision as `real_part`, `imag_part` and `make_complex`, so the two functors stay generic.

**The launchers are generic in Z and R**, so the bridge header names no complex type and needs only `runtime.h` in the implementation unit's GMF. The `.cu` instantiates the two pairs.

Ported from calaman's `complex_cast`, keeping only its strided pair: the four flat `get_*_part` / `set_*_part` calls are this pair on a `count` × 1 block, or a read-modify-write of one component.

## Tests

`test/gtest/complex_cast_gpu.cpp`, in both precisions: split against the host components with the planes' padding untouched, merge with the matrix's padding untouched, an exact split-merge round trip, and an empty block that launches nothing.

## Files

| File | Role |
|------|------|
| `complex_cast.cppm` | Module interface: `split_planes` and `merge_planes` |
| `complex_cast.cpp` | Implementation, and the `wwrFloatComplex` / `wwrDoubleComplex` instantiations |
| `complex_cast.cu` | The two kernels, over wwr's `parallel_for` |
| `complex_cast_bridge.h` | The launchers' declarations, shared by the `.cu` and the implementation unit's GMF |
| `CMakeLists.txt` | The device library `free_breakfast.recipes.complex_cast.device` and the module |
