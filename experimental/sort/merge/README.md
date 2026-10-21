# Merge sort (experimental)

A stable merge sort of `double` keys on the GPU, in IEEE-754 total order,
optionally carrying a `uint32_t` value per key. It has
[`radix_sort`](../radix/README.md)'s contract and order exactly, and it is faster
than `radix_sort` and than CUB's `DeviceMergeSort` at every measured size.
Built only under
`FREE_BREAKFAST_BUILD_EXPERIMENTAL` (the `experimental` preset).

## Module

`free_breakfast.experimental.sort.merge`, target
`free_breakfast::experimental::sort::merge`:

```cpp
import free_breakfast.experimental.sort.merge;

// keys (and optionally values) are device pointers, sorted in place.
FBF_TRY(free_breakfast::merge_sort(res, n, keys, values));
```

Like `radix_sort`, pass the indices `0 .. n-1` as `values` to get the stable
sorting permutation. The scratch is a second copy of the keys, and of the
values if any, taken from `res.device_workspace()`. Unlike `radix_sort`, it
needs no digit counts. Everything is queued on `res.stream()` and nothing
synchronizes.

The order is `radix_sort`'s: `std::strong_order` on `double`, with `-0.0`
before `+0.0` and NaNs past the infinities on their sign's side. The two sorts
give bit-identical keys and identical permutations
(`AgreesWithRadixAndVendorMergeSorts`).

## Algorithm

Keys are compared as `ordered_bits` of their 64 raw bits (`radix_sort.cu`'s
mapping, whose unsigned order is total order). They are remapped on load and
mapped back on store.

| Kernel | Grid | What it does |
|--------|------|--------------|
| `block_sort_kernel` | one block per tile of 256 threads × 11 keys (2816) | each thread sorts its 11 keys in registers by odd-even transposition, then 8 rounds of merge-path merges in shared memory double the runs until one covers the tile |
| `merge_pass_kernel` | one block per 2816 outputs | merges adjacent sorted runs, doubling the run each pass: `ceil(log2(n / 2816))` passes |

- **Up to 2816 keys** the sort is one launch, in the fewest warps that hold the
  keys (1 warp up to 352 keys), so a small input pays fewer merge rounds.
- **Merge pass:** two warps find where the block's slice of outputs starts and
  ends on the merge path, by a 32-way search over global memory (each step
  probes a warp's worth of points, so about 5 dependent loads instead of 25).
  The block then loads that slice of both runs into shared memory, coalesced,
  and each thread merges 11 outputs. It writes them back through shared
  memory, coalesced.
- **Stability:** every merge takes from the left run on a tie, and
  odd-even transposition swaps only strictly out-of-order neighbours.
- **Padding:** a partial tile is padded with the all-ones ordered key (the
  `+NaN` with every payload bit set). It sorts after every real key, stably,
  so after a real key with those bits too, and is never written out.
- **11 keys a thread** is odd, so neighbouring threads' runs in shared memory
  start on different banks.
- **Warp barriers:** while a block-sort round's pair of runs fits in one warp
  (5 of the 8 rounds on 32-lane warps), the round syncs the warp, not the
  block.

**What didn't work.** Merging by rank: every key binary-searches the other run
for its own output slot and is written straight there. A thread's 11 searches
are independent, so their loads overlap. But they make about 6× the
shared-memory loads of one merge-path search plus a serial merge, and both
kernels ran about 2× slower (1024 keys: 26 µs against 13).

## Against radix_sort and the vendor sorts

`cub_merge_sort(res, n, keys, values)` takes the same arguments and gives the
same in-place result as `merge_sort`, through the vendor's stable
`DeviceMergeSort` (`StableSortKeys` / `StableSortPairs`): CUB on CUDA, hipCUB
(rocPRIM) on HIP, reached as `wwr::wwrcub` from `cub_merge_sort.cu`. It is the
reference this sort is checked and timed against, not a second implementation.

Unlike `DeviceRadixSort`, the vendor merge sort takes a comparator, so it is
handed total order itself: the keys go in as their `uint64_t` bits, compared
as `ordered_bits`. So it agrees with `merge_sort` everywhere, `-0.0` before
`+0.0` included, and `AgreesWithRadixAndVendorMergeSorts` requires identical
key bits and identical permutations from all three.

```sh
build-experimental/experimental/sort/merge/free_breakfast_merge_sort_bench [max_log2=26] [reps=10]
```

This uses `radix_sort_bench`'s method and input: the median of `reps` runs
timed with GPU events, the input restored outside the timed region, and
finite doubles of both signs over 40 binary decades. It times all four sorts
on the same input. "match" means `merge_sort` agreed bit for bit, keys and
values, with both `radix_sort` and `cub_merge_sort`. Every row below matched.

### Measured (RTX 3080, CUDA 13 CUB, 2026-10-21)

These are medians of 20 runs, given as the range over two back-to-back runs.
An outside job shares the card, so read the ratios, which were measured
side by side, rather than the absolute times.

| n | merge ms, keys / pairs | merge / radix_sort | merge / CUB merge | merge / CUB radix |
|---|---:|---:|---:|---:|
| 32 | 0.010 / 0.010–0.011 | 0.67–0.77 / 0.71–0.79 | 0.93–1.00 / 0.83–0.92 | 0.44–0.48 / 0.40–0.44 |
| 512 | 0.012 / 0.013 | 0.86 / 0.87 | 0.86–0.92 / 0.81 | 0.54–0.55 / 0.50 |
| 1024 | 0.014 / 0.015 | 0.82 / 0.83 | 0.93–1.00 / 0.88 | 0.61–0.63 / 0.56–0.58 |
| 2048 | 0.016–0.017 / 0.019 | 0.76–0.81 / 0.83 | 0.94 / 0.95 | 0.67–0.71 / 0.68 |
| 4096 | 0.023 / 0.026 | 0.85–0.88 / 0.89 | 0.88–0.92 / 0.90–0.93 | 0.31–0.32 / 0.32–0.33 |
| 16384 | 0.033 / 0.037–0.038 | 0.94 / 0.95–0.97 | 0.82–0.84 / 0.84–0.86 | 0.41 / 0.42–0.43 |
| 2^16 | 0.044–0.045 / 0.051 | 0.10 / 0.11 | 0.78 / 0.81–0.82 | 0.49–0.50 / 0.53 |
| 2^18 | 0.069–0.070 / 0.090–0.091 | 0.05 / 0.06 | 0.71–0.72 / 0.79 | 0.58 / 0.62 |
| 2^20 | 0.32 / 0.47 | 0.05 / 0.07 | 0.85–0.86 / 0.91 | 1.09–1.10 / 1.18 |
| 2^22 | 1.34–1.35 / 2.29–2.30 | 0.05 / 0.08 | 0.90 / 0.95–0.96 | 1.41–1.42 / 1.68–1.69 |
| 2^24 | 7.1–7.4 / 10.3–10.7 | 0.06–0.07 / 0.09–0.10 | 0.93–0.95 / 0.93–0.95 | 1.82–1.88 / 1.67–1.80 |
| 2^26 | 34.1–34.2 / 47.8–48.8 | 0.08 / 0.11 | 0.94–0.95 / 0.90–0.95 | 2.03–2.05 / 1.91–2.03 |

- **Against CUB's merge sort** this sort is faster at every size, by 5–29%.
  It is at parity only at 32 and 1024 keys, keys only, in one of the two runs.
  The widest margin is 2^16 to 2^18 (0.71–0.82), where the tile, the 32-way
  pass search and the in-warp barriers count most. From 2^22 up both sorts
  are bandwidth bound and the margin narrows to 5–10%.
- **Against `radix_sort`** it is faster at every size: 0.67–0.97× below
  16384 keys, where `radix_sort` runs its tile-sort path, and 9–20× faster
  above it. The margin below 16384 varies between sessions. An earlier
  session on the same build measured 0.71–0.93× there (0.80–0.81× at 16384)
  against today's 0.67–0.97×, while `radix_sort`'s own times barely moved.
- **Against CUB's radix sort** it wins below 2^20 keys, where onesweep pays
  ~10 launches, and loses by 1.1–2× from 2^20 up. That isn't a tuning gap: a
  two-way merge reads and writes every key once per pass, `ceil(log2(n /
  2816))` passes, so 2^24 keys take 14 sweeps (block sort included) against
  onesweep's 8 radix passes. CUB's own merge sort sits in the same place for
  the same reason. The passes themselves run near the card's bandwidth. Under
  `ncu`, each kernel run alone, each keys-only pass at 2^24 took 400–470 µs
  for ~270 MB of DRAM traffic, about 580–680 GB/s of the card's 760. Closing
  the gap would take fewer passes (a larger block-sort tile, or a multiway
  merge), not faster ones.

### Where the time goes (`nsys`, n ≤ 16384)

| Kernel | GPU µs, keys / pairs |
|--------|---:|
| block sort, 1 warp (≤ 352 keys) | 8.2 / 9.1 |
| block sort, 2 warps | 9.5 / 10.4 |
| block sort, 4 warps | 10.8 / 11.8 |
| block sort, 8 warps (full tile, any number of tiles up to 6) | 13.4 / 15.2 |
| merge pass, 2–6 blocks | 3.8–4.1 / 4.4–4.7 |

Below 16384 keys the cost is mostly the block sort's chain of dependent
shared-memory loads: each round is a binary search, then 11 serial merge
steps. That is why a lone tile runs in the fewest warps that hold it (each
halving saves a round), and it is the place to look for more speed at these
sizes. Making the in-warp rounds sync the warp, not the block, did not
measurably change the 8-warp time.

## Build cost

`merge_sort.cu` instantiates 10 kernels: the block sort in 1, 2, 4 and 8
warps, each with and without values, plus the merge pass with and without.
`radix_sort.cu` instantiates 12. In the CUDA objects (sm_86, uncached
Release build):

| | `merge_sort.cu` | `cub_merge_sort.cu` |
|---|---:|---:|
| kernels | 10 | 6 |
| device code (`__nv_relfatbin`) | 35 KiB | 39 KiB |
| whole object | 50 KiB | 114 KiB |

The HIP compile was checked but not measured.

## Tests

`test/experimental/merge_sort_gpu.cpp`, in the same binary as the radix tests
(`free_breakfast_tests_experimental_gpu`, ctest labels `gpu;experimental`).
Every case compares bits:

- random bit patterns against a sort under `std::strong_order`, at the edges of
  the lone-tile block sizes (352, 704, 1408 keys), of the 2816-key tile, of an
  odd tile count (a lone run carried through a pass), and up to 2^20;
- finite values over 40 binary decades, against `std::sort`;
- every special value, including the NaNs of largest payload at both ends
  (the larger one is the pad's bits), against the order written out by hand;
- sorted, reversed and all-equal inputs;
- stability: duplicate-heavy keys (`-0.0`, `+0.0` and the pad's bits among
  them) must give exactly `std::stable_sort`'s permutation;
- agreement with `radix_sort` and with `cub_merge_sort`, keys and
  permutation, on random bits with zeros of both signs mixed in.

## Files

| File | Role |
|------|------|
| `merge_sort.cppm` / `.cpp` | The module: each sort carves its scratch and calls its launcher |
| `merge_sort_bridge.h` | Launcher declarations, shared by the module's GMF and the two `.cu` files |
| `merge_sort.cu` | The block-sort and merge-pass kernels, and the launch sequence |
| `cub_merge_sort.cu` | The vendor reference: stable `DeviceMergeSort` through `wwr::wwrcub`, with a total-order comparator |
| `merge_sort_bench.cpp` | `free_breakfast_merge_sort_bench`: this sort, `radix_sort`, and the vendor's merge and radix sorts side by side |
| `CMakeLists.txt` | The device library (`kernel_headers` PUBLIC, which also brings `cub.h`), the module and the benchmark |
