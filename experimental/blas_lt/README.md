# BLAS Lt fp64 GEMM (experimental)

A probe of the cuBLASLt / hipBLASLt matmul surface in **fp64**, on the one
shape that matters to SC-NEVPT2: the three digest GEMMs of
[`nevpt2::rdms`](../../src/nevpt2/rdm/README.md). Built only under
`FREE_BREAKFAST_BUILD_EXPERIMENTAL` (the `experimental` and
`hip-experimental` presets).

## Why

The digest GEMMs are most of an SC-NEVPT2 run, and their shape is lopsided: a
short `m = n_act^2`, a long `n = n_act^4`, a long `k` (the tile width). On the
RTX 3080, cuBLAS runs that shape well below the `m = 3 n_act^2` one a fused
digest gives — the same FLOPs, three times the width, measurably faster — which
says the library's algorithm choice, not the card, is part of the limit.

CUDA has a bigger lever than algorithm choice (CUDA 13's fp64 fixed-point
emulation). hipBLASLt implements the same idea on INT8 matrix cores, but it is
**MI350X/MI355X-only and absent from ROCm 7.2.4**, so it does not reach this
tree's card; and a hand-written int8 Ozaki digest measured *slower* than plain
`rocBLAS` fp64 on gfx1200. So on AMD the Lt
heuristic is the cheapest remaining thing to rule in or out, and the first
question is blunter than speed: **does hipBLASLt offer an fp64 algorithm for
this shape at all?**

## Module

`free_breakfast.experimental.blas_lt`, target
`free_breakfast::experimental::blas_lt`:

```cpp
import free_breakfast.experimental.blas_lt;

free_breakfast::BlasLtHandle lt{};
// C += op(A) op(B), column-major, wwr::gemm's argument order, alpha = beta = 1.
const auto result = free_breakfast::gemm_lt(res, lt, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N,
                                            m, n, k, a, k, b, k, c, m, nullptr, 0);
```

`LtResult::algorithms` is the heuristic's returned count and **0 is an answer,
not a failure**: the backend has no fp64 algorithm for that shape.
`LtResult::status` is the matmul's, or the heuristic's when the count is 0 —
returned rather than aborted, as `gemm_fp16` does.

Each call asks the heuristic for one algorithm, so the search is inside the
timing exactly as a caller doing no caching would pay it. That is deliberate:
if the Lt path only wins with a cached algorithm, the port has to cache one,
and this bench should say so.

## Bench

```sh
devtools/cpp-tier.sh --preset experimental          # CUDA
devtools/cpp-tier.sh --preset hip-experimental      # ROCm
build-experimental/experimental/blas_lt/free_breakfast_blas_lt_bench [reps] [budget_gib]
```

It walks the demo's measured cases — CAS(10,10) at 3 tiles, CAS(12,12) at 80
and 40 — each unfused (`m = n_act^2`) and fused (`m = 3 n_act^2`), and prints
GFLOP/s for `Dgemm` and for `gemm_lt`, their ratio, and the two results'
elementwise agreement against `max|C|`. A shape whose operands exceed
`budget_gib` (default 6) is skipped and said to be; `B` alone is
`n_act^4 * width` doubles, 3.5 GB at CAS(12,12) with 40 tiles.

Agreement is a check, not a bound: a different algorithm is free to sum in a
different order.

There is **no ctest entry**. There is nothing to assert about which algorithm a
vendor heuristic picks, and a test that pinned today's choice would fail on the
next library bump for no fault of this tree.

## The one piece of plumbing this added

`wwr`'s neutral BLAS surface named `WWRBLAS_R_16F`/`R_32F` and
`WWRBLAS_COMPUTE_16F`/`COMPUTE_32F` but not their fp64 partners, so
`deps/WarpWraps/src/detail/blas_names.h` gains two `WWR_VALUE_RAW` lines:
`WWRBLAS_R_64F` and `WWRBLAS_COMPUTE_64F`. **That is a submodule change**:
landing this needs a WarpWraps PR and a pin bump, not just a commit here. The
alternative was naming `CUBLAS_COMPUTE_64F` / `HIPBLAS_COMPUTE_64F` behind a
source-level backend conditional, which this tree has none of anywhere.

## Findings

Measured 2026-10-29 on this machine's two cards: CUDA 13.0 (RTX 3080, sm_86)
and ROCm 7.2.4 (RX 9060 XT, gfx1200), the whole card held through
`with_gpu_card`. Each table is `free_breakfast_blas_lt_bench 11 6`, and every
ratio below reproduced to two decimals against a `… 5 6` run in a separate
session — these are not noise.

**Both backends support fp64 through the Lt surface.** The heuristic returned
an algorithm for all six shapes on both cards; no row reported `no algo`. For
hipBLASLt that was the open question, and the answer is yes.

### gfx1200: hipBLASLt is rocBLAS, and there is nothing here

| n_act | k | fused | `Dgemm` GF/s | `gemm_lt` GF/s | lt/dgemm | max rel |
|---|---|---|---|---|---|---|
| 10 | 21168 | no  | 359.2 | 373.8 | 1.04 | 0.00e+00 |
| 10 | 21168 | yes | 438.6 | 438.7 | 1.00 | 0.00e+00 |
| 12 | 10673 | no  | 422.0 | 423.8 | 1.00 | 0.00e+00 |
| 12 | 10673 | yes | 427.9 | 427.5 | 1.00 | 0.00e+00 |
| 12 | 21345 | no  | 422.6 | 423.6 | 1.00 | 0.00e+00 |
| 12 | 21345 | yes | 428.0 | 428.1 | 1.00 | 0.00e+00 |

**The two paths agree bitwise on every shape** (`max rel` exactly 0), which is
the finding: for fp64 hipBLASLt dispatches the same kernels rocBLAS does, so
there is no second algorithm family to reach for. And rocBLAS is already at
423–429 GF/s on five of the six — the card's fp64 ceiling, matching the
~425 GF/s the demo measured for its digest GEMM. **So the Lt route on AMD is
closed**: not unsupported, just the same thing under another name.

The one shape below the ceiling is CAS(10,10) unfused (359), and Lt does not
fix it (374); fusing does (439).

### RTX 3080: Lt recovers the ceiling, it does not raise it

| n_act | k | fused | `Dgemm` GF/s | `gemm_lt` GF/s | lt/dgemm | max rel |
|---|---|---|---|---|---|---|
| 10 | 21168 | no  | 319.5 | 269.7 | **0.84** | 1.79e-14 |
| 10 | 21168 | yes | 412.5 | 405.6 | 0.98 | 1.85e-14 |
| 12 | 10673 | no  | 400.5 | 405.3 | 1.01 | 0.00e+00 |
| 12 | 10673 | yes | 376.6 | 436.1 | **1.16** | 1.31e-14 |
| 12 | 21345 | no  | 336.5 | 406.1 | **1.21** | 1.80e-14 |
| 12 | 21345 | yes | 436.8 | 434.8 | 1.00 | 1.80e-14 |

- **Every path caps at ~435 GF/s**, the 3080's fp64 ceiling. Lt wins exactly
  where cuBLAS had picked a poor algorithm and loses nothing where it had not.
  It is a ceiling-*recoverer*, never a ceiling-raiser — so it cannot compete
  with CUDA 13's fp64 emulation, which takes the demo's CAS(12,12) RDM build
  from 38.9s to 10.85s (~1550 GF/s-equivalent). **Lt is not the CUDA answer.**
- **Lt closes most of the gap that fusing closes.** At k = 21345 unfused,
  336 → 406 against fused `Dgemm`'s 437: ~70% of the fused-digest win, with
  none of the fused plumbing.
- **"Always fuse" is not safe on CUDA.** At k = 10673 the fused `Dgemm` (377)
  is *slower* than the unfused one (400) — fusing changes which algorithm
  cuBLAS picks, sometimes for the worse. Lt repairs that case (436). The two
  levers interact; a port cannot treat them as independent.
- **Lt is a 16% loss at CAS(10,10) unfused**, reproducibly. "Use Lt
  everywhere" would be a regression at small active spaces.

### What this means for the digest

Nothing to port on either card, for different reasons: on gfx1200 Lt *is*
rocBLAS, and on the 3080 it is strictly dominated by the emulated digest. Its
only standing use is diagnostic — the 3080 rows separate "the card is out of
fp64 throughput" from "cuBLAS chose badly", and they say the digest shape
suffers from both.
