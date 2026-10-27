# Sort

The stable merge sort of `double` keys, optionally carrying `uint32_t` values,
and the column gather that applies its permutation to a matrix, on the GPU.

## Module

`free_breakfast.recipes.sort` (`free_breakfast::recipes::sort`):

```cpp
// keys (and optionally values) are device pointers, sorted in place.
FBF_TRY(free_breakfast::merge_sort(res, n, keys, values));
// column j of dst = column perm[j] of src, rows x cols column-major.
FBF_TRY(free_breakfast::gather_columns(res, rows, cols, perm, src, dst));
```

Pass the indices `0 .. n-1` as `values` to get the stable sorting permutation,
then `gather_columns` with it reorders a matrix's columns to match. The sort's
scratch is a second copy of the keys, and of the values if any, carved from
`res.device_workspace()`, so neither array may live there. Everything is
queued on `res.stream()` and nothing synchronizes.

## Device library

`free_breakfast.recipes.sort.device` (`free_breakfast::recipes::sort::device`).
The integral builds' device libraries link it directly, and their `.cu`
include `merge_sort_bridge.h` and call:

```cpp
// keys ascending in IEEE-754 total order, stably, values carried (or null);
// the result ends in keys/values, the _alt arrays are n-long scratch.
int32_t merge_sort_passes(wwrStream_t stream, uint32_t n, double *keys, double *keys_alt,
                          uint32_t *values, uint32_t *values_alt);
```

It returns the first launch's `wwrError_t` as an integer and never
synchronizes. Up to 2816 keys (one tile) it is a single block sort in 1 to 8
warps, and the `_alt` scratch is never touched. Above that it runs merge
passes over global memory. Descending order is the ascending sort of negated
keys: total order puts `-0.0` after every negative key.

## Users

| User | Sorts | Key |
|---|---|---|
| [`mmd`](../../integrals/mmd/README.md) | each class's records, once per basis | `-zeta` |
| [`coulomb/sort_bounds`](../../integrals/eri/integral_direct/coulomb/sort_bounds/README.md) | each ket class, every J build | `-(Iq · Dmax)` |
| [`exchange/pairs`](../../integrals/eri/integral_direct/exchange/pairs/README.md) | each ordered class, once per basis, twice: by bound, then by group (stable, so each group ends sorted by bound) | `-I_rec`, then the group index |
| [`pipeline/localization`](../../pipeline/localization/README.md) | the localized orbitals, then `gather_columns` on C and U | `<i\|F\|i>` |
| [`experimental/sort/merge`](../../../experimental/sort/merge/README.md) | re-exports the module; its benchmark and tests | any |

## Where it came from, and what checks it

It was developed in `experimental/sort/merge`. That directory's README has the
algorithm and its timings against `radix_sort` and CUB's merge and radix sorts.
Its tests (`test/experimental/merge_sort_gpu.cpp`, the `experimental` preset)
hold it bit for bit to `std::strong_order` and `std::stable_sort`, and to
`radix_sort` and the vendor sort. `test/gtest/sort_gpu.cpp` checks the module
in the default build: `merge_sort`'s permutation against `std::stable_sort`
past one tile, and `gather_columns` against a host gather. Through its users
it is also covered by `test/gtest/sort_bounds_gpu.cpp` and
`test/gtest/mmd_gpu.cpp`, each past one tile.

## Files

| File | Role |
|------|------|
| `sort.cppm` / `.cpp` | The module: `merge_sort` carves its scratch and calls its launcher; `gather_columns` checks its arguments and calls its launcher |
| `merge_sort_bridge.h` | `merge_sort_passes`, declared once for its callers and the `.cu` |
| `merge_sort.cu` | The block-sort and merge-pass kernels, and the launch sequence |
| `gather_columns_bridge.h` | `gather_columns`' launcher, declared once for the module's GMF and the `.cu` |
| `gather_columns.cu` | The column gather, one thread per element |
| `CMakeLists.txt` | The device library and the module |
