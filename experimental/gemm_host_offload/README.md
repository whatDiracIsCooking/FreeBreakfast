# `gemm_host_offload/`

The DF exchange build `K = Xᵀ X`, `X_m = B_m C`, cut into batches, with B
resident on the device or streamed from pinned host memory through a ring of
device slots. The question: **what does each batch cost on top of the one-call
build?** The model is

```
T(n_b) = T_1 + n_b·δ                         (B resident, cut into n_b batches)
T_pipe ≈ max(T_1, T_copy) + T_copy/n_b + n_b·δ   (B streamed: + the unhidden first copy)
```

and the bench measures δ, and what it is made of: CPU launch overhead, the
cross-stream event waits, or the GEMMs' own tail and tile fill.

```sh
cmake --build --preset experimental --target free_breakfast_gemm_host_offload_bench
free_breakfast_gemm_host_offload_bench [--reps 10] [--warmup 3] [--nao 256,448,640]
    [--only resident|pipeline] [--ring-gib 6] [--csv gemm_host_offload.csv]
```

Hold the whole card (`with_gpu_card`, `devtools/lib.sh`) while it runs.

## What it runs

The calls are `df_exchange`'s
([`jk.cpp`](../../src/integrals/eri/density_fitting/jk/jk.cpp)): B is
`[m][n][P]` with P fastest, X_m = B_m C is one strided-batched gemm over m, and
K = Xᵀ X is one syrk. B and C are random (a hash of the index, B symmetric in
m and n), not a real fit: dense fp64 GEMM time does not depend on the values.

Shapes: `nao ∈ {256, 448, 640}`, `naux = 3 nao` (the JKFIT ratio), and
`nocc ∈ {64, 128}`, all of B on the card so the one-call build is there to
compare against. The pipeline runs on `nao = 640` only, at `nocc = 128`
(gemms limit on the 3080: copies ~0.26 s, gemms ~0.65 s) and `nocc = 32`
(copies limit).

Two cuts:

| cut | a batch is | per batch | after the last |
| --- | --- | --- | --- |
| **m** | b consecutive slabs B_m, contiguous in the host layout | the batched gemm over b slabs, into its part of a device-resident X | one syrk |
| **P** | b consecutive aux functions; streamed, B is laid out in P blocks `[block][m][n][p]` | the batched gemm, b × nocc per slab, and a syrk into K (beta = 1 after the first) | — |

Batch sizes: m over `1, 2, 4, …, nao`; P over `8, 12, 16, 24, …, 1536` (powers
of two and the 1.5× points between, so tile quantization shows as steps) and
`naux`. n_b = 1 is the one-call build in both.

Variants, each a curve over n_b:

| variant | B | what it adds to the one-call build |
| --- | --- | --- |
| `chopped` | resident | the batches, back to back on one stream: the GEMMs' tail and tile fill |
| `events` | resident | `chopped` plus the ring's event protocol on a second stream, R = 2, no copies: the waits |
| `graph` | resident | `chopped` captured once into a graph: no per-call CPU launch. `graph ≈ chopped` rules launch overhead out of δ |
| `pipeline` | pinned | copies of batch i+1 overlap the gemms of batch i, ring depth R ∈ {2, 3} |
| `copies` | pinned | `pipeline` with the gemms removed: the copies alone, same chunks, same waits |

A pipeline config whose ring does not fit `--ring-gib` is skipped and said to
be.

## Method

Every config runs `warmup` times untimed, then `reps` times, GPU-event timed.
Within a group the configs are round-robin, one run of each per pass, so clock
and thermal drift spreads over all of them. The pipeline's P cut is the
exception: each batch size needs its own host layout, so it is round-robin per
batch size only.

Columns: median, min and interquartile range (% of the median); `excess/b` =
(median − T_1) / (n_b − 1), δ read off one point; for `pipeline`, `over` =
median − max(T_1, `copies` at that batch), what the overlap failed to hide. Per
curve, a least-squares fit of median against n_b gives δ with its standard
error. T_1 is the resident group's own `chopped m` n_b = 1 point; the pipeline
group, which runs after B is freed, uses a standalone T_1 timed beforehand.

K is checked once per config against the one-call build. The m cut keeps every
product identical, so it should agree bitwise unless cuBLAS picks a different
algorithm for a different batch count; P sums in another order, so it is a
relative difference.

Every rep goes to the CSV (`group,nao,naux,nocc,variant,mode,depth,batch,n_b,rep,ms`).

## Results

**The full sweep has not been run.** Only a smoke run so far: RTX 3080,
`--nao 256 --reps 2 --warmup 1 --only resident`, 2026-11-08. Two reps is too
few to fit δ with confidence; read these as the shape of the answer, not the
numbers.

- **K agrees everywhere.** The m cut is bitwise against the one-call build at
  every batch size, in all three resident variants; the P cut agrees to
  ~2e-14 relative. Graph capture of the cuBLAS calls works.
- **Launch overhead is not δ.** `chopped`, `events` and `graph` agree within
  noise: m-cut δ ≈ 31, 30 and 25 µs/batch (±7) at nocc 64. Capturing the
  launches away and adding the ring's event waits both change nothing
  measurable.
- **The P cut shows tile quantization.** b = 32, 64, 128 are cheap; the 1.5×
  points b = 24, 48, 96 each cost 15–20% more than their power-of-two
  neighbours, and below b = 16 time climbs steeply (b = 8 is 2.6× T_1). So δ
  for P is not a constant, and a straight-line fit over it (~475 µs/batch) is
  not meaningful.
- **The m cut is flat down to ~8 slabs per batch, then steps up ~20%** (b ≤ 4
  at nocc 64, b ≤ 2 at nocc 128): a threshold more than a linear δ.
- **The reps spread 1–3% of the median** with clocks unpinned. On the 640
  shape that hides a δ below ~10 µs.

T_1 at nao 256, naux 768: 29.5 ms at nocc 64 and 59.4 ms at nocc 128, about
330 GF/s, ~70% of the 3080's fp64 peak.

### Still to do

1. **The full sweep is ~1 hour of a held card, not ~15 min**: on the 640
   shape one build is 0.3–0.65 s. The resident groups are ~40 min, the
   pipeline ~20. Consider dropping `graph` on nao 448 and 640, now that it
   has answered its question (saves ~¼ of the resident time).
2. Run it under `with_gpu_card cuda` (`devtools/lib.sh`), through a script
   file rather than an inline `bash -c`, from the build dir's `.log/`.
3. Fill in this section with the fits, the pipeline's `over` column against
   `T_copy/n_b + n_b·δ`, and whether the m cut's step is per-call tile fill.
4. Then the same on gfx1200 (`hip-experimental`).
