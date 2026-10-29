# Split fp32 GEMM (experimental)

fp64 GEMM composed from the vendor's own `Sgemm` on a hi/lo fp32 split, on the
shapes [`nevpt2::rdms`](../../src/nevpt2/rdm/README.md)' digest runs: how much
of a consumer card's fp32 throughput can be spent buying fp64 accuracy, and
what accuracy survives. Built only under `FREE_BREAKFAST_BUILD_EXPERIMENTAL`
(the `experimental` and `hip-experimental` presets).

## Why

The digest GEMMs are most of an SC-NEVPT2 run, and
[`blas_lt`](../blas_lt/README.md) closed the last library-side lever: both
cards already run that shape at their fp64 ceiling, and the Lt heuristic
recovers a badly-chosen algorithm without ever raising the ceiling. Probe 1
here confirms the ceiling is the card's and not the shape's — square `Dgemm`,
which has far better reuse, tops out at the same place.

So the only lever left is to stop using the fp64 units. CUDA 13 and hipBLASLt
both do that with int8 matrix cores, and the demo measured the CUDA one taking
a CAS(12,12) RDM build from 38.9s to 10.85s. Neither reaches this tree's cards:
hipBLASLt's is MI350X/MI355X-only and absent from ROCm 7.2.4.

This is the same idea with fp32 instead of int8, and — unlike the demo's
hand-written int8 WMMA digest, which lost to rocBLAS on kernel quality — it
writes no GEMM. The products are the vendor's `Sgemm`, which both cards run
20–58× faster than `Dgemm`.

## Module

`free_breakfast.experimental.split_gemm`, target
`free_breakfast::experimental::split_gemm`:

```cpp
import free_breakfast.experimental.split_gemm;

// C += op(A) op(B), column-major, wwr::gemm's argument order.
const auto status = free_breakfast::gemm_split_fp32(
    res, wwr::WWRBLAS_OP_T, wwr::WWRBLAS_OP_N, m, n, k, a, k, b, k, c, m,
    k_chunk, terms, scratch);
```

`split_scratch_floats(m, n, k)` sizes the caller-owned `SplitScratch`:
`2(mk + kn)` floats for the two operands' splits — exactly as many bytes as the
fp64 operands they came from, so the split costs no memory over the inputs —
plus `mn` for one chunk's fp32 product.

The operands must be **packed**; the call returns `Unsupported` otherwise,
because the K chunking slices them in place.

Two knobs, and the measurement is about which of them matters:

- `terms` — 1 forms `A_hi B_hi` alone (a plain fp32 GEMM), 3 adds
  `A_hi B_lo + A_lo B_hi` (the double-single product, ~2^-48 in the inputs).
- `k_chunk` — the K extent of one fp32 accumulation. Each chunk runs `terms`
  `Sgemm` calls and one `accumulate_f32` pass folding the result into fp64 `C`.

## Bench

```sh
devtools/cpp-tier.sh --preset experimental          # CUDA
devtools/cpp-tier.sh --preset hip-experimental      # ROCm
build-experimental/experimental/split_gemm/free_breakfast_split_gemm_bench [reps] [budget_gib]
```

Probe 1 is square `Dgemm` and `Sgemm`, n = 1024…8192. Probe 2 is the digest's
shape through `gemm_split_fp32` at `terms` 1 and 3 over a `k_chunk` sweep,
against `Dgemm`, reporting fp64-equivalent GF/s (2mnk either way) and the max
elementwise error against `Dgemm`'s answer.

Read the two columns together: **a row is only interesting if its error is
inside what the caller needs.** No ctest entry — this is a measurement, not an
assertion, for [`bench/README.md`](../../bench/README.md)'s reason.

Run it line-buffered to a file under a timeout, never piped:

```sh
timeout 900 stdbuf -oL …/free_breakfast_split_gemm_bench 5 6 > out.txt 2>&1
```

Both hazards below are why. A pipe makes stdout block-buffered, and the whole
table is under one 4 KiB block, so a kill discards every row it had already
measured.

## Findings

Measured 2026-10-29, CUDA 13.0 (RTX 3080, sm_86) and ROCm 7.2.4 (RX 9060 XT,
gfx1200), the whole card held through `with_gpu_card`, `… 5 6`.

### Probe 1: the ceiling is the card's

| n | 3080 `Dgemm` | 3080 `Sgemm` | ratio | gfx1200 `Dgemm` | gfx1200 `Sgemm` | ratio |
|---|---|---|---|---|---|---|
| 1024 | 343.8 | 14563.6 | 42.4 | 224.7 | 4481.4 | 19.9 |
| 2048 | 403.5 | 23696.6 | 58.7 | 265.4 | 6974.4 | 26.3 |
| 4096 | 414.3 | 21585.4 | 52.1 | 397.2 | **8602.3** | 21.7 |
| 8192 | **426.5** | 20678.7 | 48.5 | **432.5** | 542.8 | 1.3 |

GF/s. Square `Dgemm` tops out at 426 GF/s on the 3080 and 432 on gfx1200 —
the same place the digest's own lopsided shapes reach, so **no shape, tiling or
library change can help the digest**. The two cards' fp64 ceilings are within
1.5% of each other; gfx1200's is marginally the higher of the two.

The fp32 column is the headroom probe 2 tries to spend, and it is 20–58× the
fp64 one.

### Probe 2: `terms = 3` buys nothing, on either card

CAS(12,12) unfused, m = 144, n = 20736, k = 10673. `vs d` is against that
card's own `Dgemm` on the same shape (377.7 GF/s on the 3080, 409.7 on
gfx1200 — both at the ceiling).

| `k_chunk` | 3080 t=1 | 3080 t=3 | gfx1200 t=1 | gfx1200 t=3 | rel err |
|---|---|---|---|---|---|
| 10673 | 8.04× | 3.19× | 3.89× | 1.49× | 5.7e-06 |
| 4096 | 12.73× | 5.79× | 4.16× | 1.74× | 2.3e-06 |
| 1024 | **13.33×** | 6.38× | 3.95× | 1.73× | 6.7e-07 |
| 256 | 10.54× | 5.35× | 3.49× | 1.63× | 1.7e-07 |
| 64 | 5.53× | 3.32× | 2.48× | 1.32× | 4.5e-08 |

**One error column serves both `terms`, and that is the finding.** At every
`k_chunk`, on both cards, `terms = 3` reproduces `terms = 1`'s error to three
significant figures. The hi/lo split is pure waste: ~2.4× the cost for no
accuracy at all.

The reason is in the module header — `Sgemm` accumulates in **fp32**, so
splitting the *inputs* cannot touch the sum, and the sum is the entire error.
`k_chunk` is the only knob that matters, and `terms = 1` is the only setting
worth running.

### The error is linear in `k_chunk`, not `sqrt`

Measured, both cards: ×4.0 in `k_chunk` gives ×3.7–4.0 in error from 64 up to
4096, tapering to ×2.4–2.6 at the top. A random-walk model predicts ×2.0 and
is wrong — `rel` is a worst-case statistic over 2.9M outputs, and round-off
bias accumulates, so it grows with the term count rather than its square root.

Practically, `rel ≈ 7e-10 · k_chunk` below ~4096. That is the number that
decides whether this route is usable at all.

### What this means for the digest

Usable, and only against an energy-level gate:

- **`nevpt2_rdm_gpu` is permanently out of reach.** It checks `dm3` to 1e-12 of
  max|dm3|. Linear scaling puts that at `k_chunk ≈ 0.001` — not a small chunk,
  an impossible one. No configuration of this route passes that test, so taking
  it means replacing an RDM-level gate with an `E_corr`-level one.
- **At ~1.7e-07 the speedup is 10.5× on the 3080 and 3.5× on gfx1200.** For
  SC-NEVPT2's `E_corr` the gate is 1e-7, so this sits right at its edge, and
  whether 1.7e-07 on the RDM lands inside 1e-7 on `E_corr` is not answerable
  from this bench — it needs an end-to-end run through `class_energies`.
- **The backends diverge by 3×, and not because AMD is weak at fp64.**
  gfx1200's `Dgemm` ceiling is the higher of the two; it simply has far less
  fp32 headroom to spend at these shapes (21.7× against the 3080's 52.1×). So
  this route is worth roughly three times as much on CUDA, which is an awkward
  property for a single-path tree.
- CUDA's fused rows (`m = 3 n_act^2`) are non-monotonic in `k_chunk` — 28.2×,
  10.8×, 17.9×, 14.9×, 7.6× — because cuBLAS picks a different `Sgemm` kernel
  per chunk width. The broad 5–18× is solid at 5 reps; individual fused rows
  are not. The unfused rows above are the ones to quote.

## Two measurement hazards on gfx1200

**`Sgemm` at 8192³ collapses to 543 GF/s**, against 8602 at 4096³ — a 16×
cliff over one power of two, while fp64 rises smoothly across the same step. It
reproduces in a standalone program calling `hipblasSgemm_64` directly (530.9
GF/s), so it is rocBLAS/Tensile solution selection and nothing to do with this
tree. Harmless here — no digest shape is near it — but **do not read a
square-`Sgemm` ceiling off n = 8192 on this card**; 4096 is the real peak.

**One unreproduced hang.** This bench once wedged for 2h13m on gfx1200 and had
to be killed; the same binary, card and arguments then completed in **61s**.
While wedged: host thread pinned at 100% CPU (HIP spin-waits on sync, so it
looks like work), `rocm-smi` at 2% GPU busy with **CU occupancy 0** and an edge
temperature of 30°C. A staged probe afterwards cleared hipBLAS entirely —
`hipMalloc`, memcpy, `hipblasCreate`, `hipblasDaxpy`, and every
`hipblasDgemm_64`/`hipblasSgemm_64` this bench issues at the exact digest
shapes, all passing in 24s. So it is not the BLAS, there is no reproducer, and
nothing was filed. Run under `timeout` and treat a second occurrence as the
datum that makes it diagnosable.
