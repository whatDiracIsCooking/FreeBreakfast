# fp16 GEMM (experimental)

A probe of half-precision (`__half`) GEMM on both backends: whether cuBLAS and
hipBLAS take it, how accurately, and how fast against fp32 `Sgemm`. Built only
under `FREE_BREAKFAST_BUILD_EXPERIMENTAL` (the `experimental` preset).

## Module

`free_breakfast.experimental.fp16_gemm`, target
`free_breakfast::experimental::fp16_gemm`:

```cpp
import free_breakfast.experimental.fp16_gemm;

// C = A B, column-major m x k, k x n, m x n, every one __half, on res.stream().
const auto status = free_breakfast::gemm_fp16(res, free_breakfast::Fp16Accumulate::kFloat,
                                              m, n, k, a, b, c);
```

| `Fp16Accumulate` | Call | Sum in (CUDA) | Sum in (HIP) |
|---|---|---|---|
| `kHalf` | `wwrblasHgemm` | `__half` | `float` |
| `kFloat` | `wwrblasGemmEx`, `WWRBLAS_COMPUTE_32F` | `float` | `float` |

A, B and C are `__half` either way, and alpha = 1, beta = 0. The two
`Hgemm`s share a name, not numerics: hipBLAS's accumulates in fp32 (measured
on gfx1200, bit-identical to `GemmEx`-32F), cuBLAS's in fp16.

## Tests and bench

`test/experimental/fp16_gemm_gpu.cpp` runs both accumulators over odd and
tile-aligned shapes and holds every entry to the worst-case summation bound in
that accumulator's unit roundoff, against the exact product of the rounded
inputs:

```sh
devtools/cpp-tier.sh --preset experimental -- -L experimental -R Fp16
```

`free_breakfast_fp16_gemm_bench [max_n] [reps]` prints TFLOP/s for `Hgemm`,
`GemmEx`-f32 and `Sgemm` on square sizes from 512.

## Findings

Measured 2026-10-27 on this machine's two cards: CUDA 13.0 and ROCm 7.2.4.

**Both backends support fp16 GEMM, under both accumulators.** Every case
passes its worst-case bound on both cards, and the half path runs well above
`Sgemm` on both, so each library takes a real half-precision matrix path
(tensor cores / WMMA), not an emulation.

The worst entry error, relative to `S = sum |a_il b_lj|`, at the last shape
(1024 x 1024 x 64):

| | `kHalf` (`Hgemm`) | `kFloat` (`GemmEx` 32F) |
|---|---|---|
| RTX 3080 (sm_86) | 7.9e-4 | 2.7e-4 |
| gfx1200 | 2.7e-4 | 2.7e-4 |

On gfx1200 the two accumulators give **bit-identical errors** and the same rate.
`hipblasHgemm` there accumulates in fp32 internally: compute type 16F
does not buy speed and does not lose accuracy. On the 3080 it really is a
half-precision sum: about 3x the error, at twice the rate.

TFLOP/s, square n, median of 10 (`free_breakfast_fp16_gemm_bench 8192 10`,
both of the card's GPU-lock slots held):

| n | 3080 `Hgemm` | 3080 `GemmEx`-f32 | 3080 `Sgemm` | gfx1200 `Hgemm` | gfx1200 `GemmEx`-f32 | gfx1200 `Sgemm` |
|---|---|---|---|---|---|---|
| 512  | 21.9  | 14.6 | 9.4  | 13.3 | 13.3 | 1.7  |
| 1024 | 72.3  | 34.4 | 14.8 | 43.4 | 43.7 | 6.3  |
| 2048 | 91.7  | 52.9 | 21.3 | 65.1 | 67.0 | 7.4  |
| 4096 | 106.0 | 54.7 | 22.0 | 70.9 | 77.7 | 8.7  |
| 8192 | 122.4 | 63.3 | 22.7 | 41.2 | 40.9 | 0.54 |

- **3080:** fp32 accumulation runs at half the fp16-accumulate rate. That is
  GeForce Ampere's tensor-core rating (about 119 vs 60 dense TFLOP/s), not a
  library choice.
- **gfx1200:** n = 8192 drops for all three calls. `Sgemm` falls to 0.54
  TFLOP/s, and this repeated on every run. The 4096 row is noisy (one rerun
  gave 49 fp16 TFLOP/s). Both point at the vendor library's kernel choice or
  the card's clocks rather than at this code; neither was investigated.
- The bench feeds constant matrices, and some cards clock higher on them than
  on random data, so read the rates as upper bounds.
