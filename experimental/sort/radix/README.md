# Radix sort (experimental)

A stable LSD radix sort of `double` keys on the GPU, in IEEE-754 total order,
optionally carrying a `uint32_t` value per key. Built on `core/kernel`'s
`block_reduce`. Built only under `FREE_BREAKFAST_BUILD_EXPERIMENTAL` (the
`experimental` preset).

## Module

`free_breakfast.experimental.sort.radix`, target
`free_breakfast::experimental::sort::radix`:

```cpp
import free_breakfast.experimental.sort.radix;

// keys (and optionally values) are device pointers, sorted in place.
FBF_TRY(free_breakfast::radix_sort(res, n, keys, values));
```

Pass the indices `0 .. n-1` as `values` to get the stable sorting permutation.
The scratch (a second copy of the keys, and of the values if any, plus the
digit counts) is taken from `res.device_workspace()`. Everything is queued on
`res.stream()` and nothing synchronizes.

## Order

The order is IEEE-754 total order, which is what `std::strong_order` gives on
`double`:

```
-NaN < -inf < -max < ... < -denorm_min < -0.0 < +0.0 < +denorm_min < ... < +max < +inf < +NaN
```

Every key has a place, NaNs included: a NaN sorts to the end on its sign's
side, and NaNs order among themselves by payload. `-0.0` sorts before `+0.0`.
For keys with no NaNs and no zeros of both signs, this agrees with `std::sort`
and `operator<`. The sort never computes with a key, so each comes out
bit-for-bit as it went in.

## Algorithm

The kernels treat each key as its 64 raw bits. They take digits from
`ordered_bits(bits)`, which sets a non-negative key's sign bit and flips every
bit of a negative one. Comparing those as unsigned integers gives total order.
The mapping is applied on each read, not stored, so the data never needs an
encode or decode pass.

`radix_sort` takes one of two paths by size.

### Up to 16384 keys: tile sort, then merge

| Kernel | Grid | What it does |
|--------|------|--------------|
| `radix_sort_tile_kernel` | one block per 1024-key tile (256 threads × 4 keys) | sorts its tile on chip: 8 passes of 8-bit digits with the keys and values held in registers, each pass ranking them stably and exchanging them through shared memory. Global memory is read once and written once |
| `merge_runs_kernel` | 128-thread blocks, 8 outputs a thread | merges adjacent sorted runs, doubling the run each round (at most 4 rounds). Each thread binary-searches the merge path for its first output; ties go to the left run, so the merge is stable |

Up to 1024 keys this is a single launch, in the fewest warps that hold the
keys (1 warp up to 128 keys).

**Ranking.** Ranking a pass is a *warp multisplit*. Eight ballots give each
lane the mask of lanes in its warp holding the same digit. A key's rank is a
running per-(digit, warp) count in shared memory plus the number of lower
lanes in that mask, so there are no atomics and the result is deterministic.
One block scan of the per-(digit, warp) counts then gives each warp's start
per digit.

**Other details:**
- **Uniform passes:** a pass where every key in the tile has the same digit
  moves nothing, so it is skipped.
- **Padding:** a partial tile is padded with the all-ones key, which sorts to
  the end and is never written out.
- **Shared memory:** the histogram's digit rows are padded by one slot, so a
  warp ranking different digits doesn't serialize on one bank.

**Tile size.** 1024 keys measured fastest. A 2048-key tile took 24 µs where
two 1024-key tiles and a merge took 20 µs, because a pass's fixed cost
(zeroing, scanning and syncing the histogram) grows with the warps in a block.

**What didn't work.** An earlier version sorted all 16384 keys in one
1024-thread block, bouncing them through L2 every pass. It was 1.7–1.9×
slower than CUB at 12k–16k keys. One SM can't keep up there, and the
1024-thread variants spilled registers.

### Above 16384 keys: 16 passes of 4 bits, three kernels each

| Kernel | Grid | What it does |
|--------|------|--------------|
| count | one block per `kFourWarps`-key tile | per digit, `block_reduce` of "is my digit", with `nactive` covering the partial last tile; writes `counts[digit * tiles + tile]` |
| scan | one block of 1024 threads | exclusive prefix sum over `counts` in that digit-major order, in block-wide chunks with a carry |
| scatter | as count | per digit, `block_exclusive_sum` of "is my digit" gives each key its rank among its tile's keys of that digit; writes to the slot's start plus that rank |

The rank keeps input order within a digit, so each pass is stable, and LSD
sorting needs that. The pass count is even, so the result ends up back in the
caller's arrays.

`block_scan.cuh` holds `block_exclusive_sum`, the scan counterpart of
`block_reduce`, with the same structure: a shuffle ladder per warp, per-warp
totals through caller-provided shared memory, and one barrier. It is
`uint32_t`-only. It belongs in `core/kernel` once something else needs it.

## Against the vendor sort

`cub_radix_sort(res, n, keys, values)` takes the same arguments and gives the
same in-place result as `radix_sort`, through the vendor's `DeviceRadixSort`:
CUB on CUDA, hipCUB on HIP, reached as `wwr::wwrcub` (WarpWraps' `cub.h`), so
`cub_radix_sort.cu` has no backend `#if`. It is the reference this sort is
checked and timed against, not a second implementation.

The orders agree except at zero. The vendor sorts `-0.0` and `+0.0` as one key,
so mixed zeros keep their input order, while `radix_sort` puts every `-0.0`
first. `VendorTreatsSignedZerosAsEqual` pins that. Elsewhere, NaNs included,
`AgreesWithVendorSort` requires identical key bits and identical stable
permutations.

```sh
build-experimental/experimental/sort/radix/free_breakfast_radix_sort_bench [max_log2=26] [reps=10]
```

The benchmark times both sorts on the same input for n = 2^10, 2^12, …,
2^max_log2, keys only and with values. The input is finite doubles of both
signs over 40 binary decades. Each figure is the median of `reps` runs timed
with GPU events; the input is restored before every run, outside the timed
region. It prints both times, their ratio, keys per second, and whether the
two outputs matched bit for bit. It is a plain executable, not a ctest entry,
for `bench/`'s reason: read the numbers, don't gate on them.

### Measured below 16384 keys: the tile-sort path beats CUB (RTX 3080, 2026-10-20)

This is the median of 20 runs, and the two runs matched to the microsecond.
The card was still shared with an outside job (74–99% utilization before the
runs). Every row matched CUB bit for bit.

| n | ours ms, keys / pairs | CUB ms, keys / pairs | ours / CUB |
|---|---:|---:|---:|
| 32 | 0.014 / 0.014 | 0.022 / 0.026 | 0.64 / 0.56 |
| 512 | 0.013 / 0.013 | 0.022 / 0.026 | 0.59–0.62 / 0.52 |
| 1024 | 0.015 / 0.016 | 0.022 / 0.026 | 0.68 / 0.64 |
| 2048 | 0.020 / 0.022 | 0.023 / 0.027 | 0.91 / 0.81 |
| 4096 | 0.026 / 0.027 | 0.067 / 0.073 | 0.38 / 0.37 |
| 8192 | 0.031 / 0.033 | 0.072 / 0.079 | 0.43 / 0.42 |
| 12000 | 0.037 / 0.040 | 0.068 / 0.074 | 0.55 / 0.54 |
| 16384 | 0.036–0.037 / 0.038–0.039 | 0.073 / 0.079 | 0.49–0.51 / 0.48–0.50 |

**Why it wins.** `nsys` shows the gap is in the GPU work, not host overhead:

- **Up to 2048 keys:** CUB runs one `DeviceRadixSortSingleTileKernel` of about
  20 µs. Ours is one tile launch, or two tiles and a merge.
- **4096 to 16384 keys:** CUB switches to onesweep: a histogram kernel, 8
  onesweep passes and a scan, about 10 launches at roughly 5 µs each. Ours is
  one tile launch over up to 16 SMs plus at most 4 short merge launches.

The tightest margin is n = 2048: 0.91 keys-only.

### Measured above 16384 keys (RTX 3080, CUDA 13 CUB, 2026-10-20)

This table predates the tile-sort path. Its 2^10 row (0.24 ms, about 10×
slower than CUB) was the old multi-kernel path, which now runs only above
16384 keys; the table above has current figures for small n.

This is a range over two runs. Another job outside the container was keeping
the card busy (about 96% utilization, up to 300 W, before the benchmark
started), so the absolute times are noisy. The ratios held within about ±20%
between runs, and every row matched bit for bit.

| n | kind | ours ms | CUB ms | ours / CUB |
|---|------|--------:|-------:|-----------:|
| 2^16 | keys | 0.45 | 0.07–0.09 | 5–6× |
| 2^20 | keys | 5.7–6.5 | 0.29–0.30 | 20–22× |
| 2^20 | pairs | 5.8–6.4 | 0.39–0.40 | 15–16× |
| 2^24 | keys | 113 | 3.9–4.1 | 27–29× |
| 2^26 | keys | 447–450 | 14–21 | 21–31× |
| 2^26 | pairs | 482–483 | 24.5–25.5 | ~19× |

At large n, CUB sorts about 4.3 billion keys/s (2.7 billion with values). This
sort flattens at about 0.15 billion keys/s from 2^18 keys up, with or without
values. In memory traffic, CUB moves about 135 bytes per key at ~4.4 billion
keys/s, close to the card's bandwidth. This sort moves about 384 bytes per key
(16 passes, each reading the keys twice and writing them once) at 0.15 billion
keys/s, about 58 GB/s. So it is limited by latency and barriers, not
bandwidth. Just above 16384 keys the gap is launch overhead: 48 launches per
sort against CUB's handful.

`nsys` at n ≤ 2^22 splits this sort's GPU time roughly evenly between count
(39%), the single-block scan (31%) and scatter (30%). CUB's is about 92% in
`DeviceRadixSortOnesweepKernel`. The differences, roughly in order of what
they cost:

- **8-bit digits.** CUB makes 8 passes over a double; this sort makes 16.
- **One kernel per pass.** CUB builds every pass's histogram in a single kernel
  up front. Each onesweep pass then ranks, scans across tiles with decoupled
  look-back, and scatters in one launch. This sort launches three kernels per
  pass, and its scan is a single block walking every count serially.
- **Many keys per thread.** CUB ranks a tile of thousands of keys in shared
  memory and writes each digit's run contiguously. Here each 128-key tile
  needs 32 block-wide collectives per pass (16 reductions and 16 scans), and
  its writes are scattered.

The "Known costs" below are the same list, so this measures them rather than
adding new ones.

### What the vendor sort costs to build and ship (2026-10-20)

Each `.cu` was compiled with the build's own command, uncached, three times;
the table gives the median. The build targets one GPU architecture per backend
(`-arch=native`, sm_86, on CUDA; gfx1200 on HIP). Both `.cu` files instantiate
the same two sorts: keys only, and `double` keys with `uint32_t` values.

| | `radix_sort.cu` before the tile path | `radix_sort.cu` now | `cub_radix_sort.cu` |
|---|---:|---:|---:|
| CUDA compile (nvcc) | 4.1 s | 4.9 s | 8.6 s |
| CUDA kernels / device code in the linked binary | 3 / 18 KiB | 12 / 207 KiB | 9 / 135 KiB |
| HIP compile (clang `-x hip`) | 2.0 s | 2.7 s | 9.5 s |
| HIP kernels / embedded device code (`.hip_fatbin`) | 3 / 19 KiB | 12 / 87 KiB | 274 / 1.97 MiB |
| Host code (`.text`), CUDA / HIP | 2.8 / 1.5 KiB | 4.4 / 5.5 KiB | 2.5 / 11.3 KiB |

- **CUDA:** CUB compiles about 3.7 s slower than this sort. Since the tile
  path, though, this sort carries *more* device code than CUB: 207 KiB against
  135. The tile kernel is instantiated 8 times (1, 2, 4 or 8 warps, each with
  and without values), about 23 KiB each with its rank and exchange loops
  fully unrolled. Dropping the 2- and 4-warp variants, so 129 to 512 keys use
  the 8-warp tile, would cut about 90 KiB. Neither the saving nor its cost in
  speed (those sizes would pay the 8-warp tile's ~15 µs instead of ~13 µs) has
  been measured.
- **HIP:** hipCUB is much heavier. rocPRIM, its backend, instantiates 274
  kernels for the two sorts. Their machine code is modest (about 160 KiB of
  `.text`). Most of the listed size is per-kernel metadata notes (334 KiB) and
  mangled symbol names (223 KiB), and the device code object comes to 1.96 MiB.
  This sort's 87 KiB is under a twentieth of that.
- **Both costs grow** with every extra GPU architecture in the build and every
  extra key/value type pair instantiated.
- **In practice:** the extra compile time is one translation unit, cached by
  sccache and `devtools/cuda-object-cache.sh` after the first build. It
  lengthens a full build only if this TU ends up on the critical path.

## Known costs

These apply to the path above 16384 keys. The tile-sort path below it avoids
all of them. It keeps 4 keys a thread on chip, has no separate scan kernel,
writes global memory once, and skips uniform passes within a tile.

That path aims at correctness, not speed:

- One key per thread. Each pass makes 16 block-wide reductions in count and 16
  block-wide scans in scatter, each with its own barrier, over 16 passes.
- The scan is a single block, so it is serial in `16 * n / kFourWarps / 1024`
  chunks.
- Scatter writes are uncoalesced. A tuned sort would rank in shared memory
  first and then write each digit's run contiguously.
- Every pass runs, even one where every key has the same digit (for example
  the top exponent bits of keys all within a few decades). Skipping such a pass
  needs the counts on the host, which means a synchronization.

## Tests

`test/experimental/radix_sort_gpu.cpp` (binary
`free_breakfast_tests_experimental_gpu`, ctest labels `gpu;experimental`;
`ctest -L experimental` runs them). Every case compares bits, not values, so
`-0.0` and NaN payloads are checked exactly. The cases:

- random bit patterns (every kind of double), against a sort under
  `std::strong_order`, for sizes around the tile edges up to 2^20;
- finite values over 40 binary decades, against `std::sort`;
- every special value, shuffled, against the order written out by hand;
- sorted, reversed and all-equal inputs;
- stability: duplicate-heavy keys (`-0.0` and `+0.0` among them) carrying
  their indices must give exactly `std::stable_sort`'s permutation.

## Files

| File | Role |
|------|------|
| `radix_sort.cppm` / `.cpp` | The module: carves the scratch and calls the launcher |
| `radix_sort_bridge.h` | Launcher declaration and scratch sizing, shared by the module's GMF and the `.cu` |
| `radix_sort.cu` | `ordered_bits`; the tile-sort and merge kernels for up to 16384 keys; the three per-pass kernels and the pass loop above that |
| `cub_radix_sort.cu` | The vendor reference: `DeviceRadixSort` through `wwr::wwrcub` |
| `radix_sort_bench.cpp` | `free_breakfast_radix_sort_bench`: the two sorts side by side |
| `block_scan.cuh` | `block_exclusive_sum`, device only |
| `CMakeLists.txt` | The device library (`kernel_headers` PUBLIC, for the bridge), the module and the benchmark |
