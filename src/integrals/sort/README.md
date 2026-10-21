# Sort

The stable merge sort of `double` keys, optionally carrying `uint32_t` values,
that the integral builds sort with on the GPU.

## Target

Device code only, no module: `free_breakfast.integrals.sort.device`
(`free_breakfast::integrals::sort::device`). A device library links it, and its
`.cu` includes `merge_sort_bridge.h` and calls:

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
| [`mmd`](../mmd/README.md) | each class's records, once per basis | `-zeta` |
| [`coulomb/sort_bounds`](../coulomb/sort_bounds/README.md) | each ket class, every J build | `-(Iq · Dmax)` |
| [`experimental/sort/merge`](../../../experimental/sort/merge/README.md) | its module wrapper, benchmark and tests | any |

## Where it came from, and what checks it

It was developed in `experimental/sort/merge`. That directory's README has the
algorithm and its timings against `radix_sort` and CUB's merge and radix sorts.
Its tests (`test/experimental/merge_sort_gpu.cpp`, the `experimental` preset)
hold it bit for bit to `std::strong_order` and `std::stable_sort`, and to
`radix_sort` and the vendor sort. Through its users it is also covered by
`test/gtest/sort_bounds_gpu.cpp` and `test/gtest/mmd_gpu.cpp`, each past one
tile.

## Files

| File | Role |
|------|------|
| `merge_sort_bridge.h` | `merge_sort_passes`, declared once for its callers and the `.cu` |
| `merge_sort.cu` | The block-sort and merge-pass kernels, and the launch sequence |
| `CMakeLists.txt` | The device library |
