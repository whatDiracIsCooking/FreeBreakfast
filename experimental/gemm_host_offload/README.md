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
(gemms limit on the 3080: copies 0.47 s, gemms 0.75 s) and `nocc = 32`
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

RTX 3080 (PCIe 4 x16), clocks unpinned, `--reps 10 --warmup 3`, all
defaults, 2026-11-08; 65 min of a held card. Times are medians, ms.

**K agrees everywhere.** The m cut is bitwise against the one-call build at
every batch size but b = 1 (~1e-15, see below); the P cut agrees to ≤ 6e-14
relative, resident and streamed.

### δ is not launch overhead or event waits

`chopped`, `events` and `graph` agree within their fit errors at every shape,
on both cuts. At nao 640, nocc 128, m cut: δ = −8.3, −7.5 and −10.4 ± 46
µs/batch. Neither capturing the launches away nor adding the ring's waits
changes anything measurable, so whatever a batch costs is the GEMMs' own.

### The m cut: no linear δ, a step at small b, and b = 1 is fastest

δ fits to zero within error at nao ≥ 448 (it is 30–55 ± 8 µs at nao 256).
What there is instead is a step at small b, the slabs per batch (`chopped`):

| nao, nocc | T_1 | b = 32 | b = 8 | b = 4 | b = 2 | b = 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 256, 128 | 64.0 | 64.9 | 65.7 | 65.6 | 75.6 | 76.5 |
| 448, 128 | 369.2 | 371.5 | 371.3 | 371.6 | 403.5 | **357.5** |
| 640, 64 | 397.2 | 394.7 | 415.7 | 415.8 | 413.4 | **377.5** |
| 640, 128 | 749.7 | 753.9 | 777.7 | 809.8 | 810.6 | **719.2** |

So at the 640 shape the cost starts below ~32 slabs per batch and reaches 8%
at b ≤ 4. And at nao ≥ 448, **b = 1 beats the one-call build by 3–5%**, with
K at ~1e-15 instead of bitwise: cuBLAS evidently takes another algorithm for
a batch count of one. That says the one strided-batched call `df_exchange`
makes now is not the fastest form of its own gemm at these shapes. Worth an
A/B in `jk.cpp` itself (one gemm per slab, or the batched call with a
different count) before trusting it beyond this card.

### The P cut: tile quantization, not δ

The P cut's straight-line fit (0.5–5 ms/batch) means nothing: its cost is a
function of b, not of n_b. At nao 640, nocc 128 (`chopped`, T_1 = 749.7):

| b | 1024…128 | 96 | 64 | 48 | 32 | 24 | 16 | 12 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms | 749.7–753.8 | 904.0 | 754.0 | 904.5 | 755.7 | 906.4 | 1207.8 | 1509.5 | 2112.5 |

Every power of two from 32 up costs ≤ 1%; the 1.5× points 96, 48 and 24 cost
20%, the same 150 ms each, and below 32 the time grows like 1/b. nocc 64 and
nao 448 show the same pattern. **A P batch should be a power of two ≥ 32.**

### Streaming B: the overlap works, and the model holds

The copies alone take 467.1 ms for B's 5.86 GiB (13.5 GB/s, ~half of PCIe 4
x16's peak), whatever the chunking: `copies` δ = 1.4 µs/chunk. Ring depth R =
3 is indistinguishable from R = 2 everywhere, so R = 2 is enough.

**Gemms limit (nocc 128, standalone T_1 = 765.9):** at P b = 32 and 64 the
pipeline takes 758.1 and 764.6 ms, `over` ≤ 0: all 5.86 GiB of copies are
hidden and the build costs what it does resident. Above that, `over` is the
unhidden first copy: near the model's T_copy/n_b at small n_b (166 ms at
n_b = 3 against 156) and falling faster than it (14 at n_b = 15 against 31). At 96, 48, 24 and below 32 the
P cut's tile penalty shows up unchanged. The m cut's best is b = 1 (770.1
ms, `over` 4.3) or b = 8 (778.0): the same step as resident.

**Copies limit (nocc 32, T_1 = 200.2):** the P cut gets to 469.7–470.5 ms for
b = 12…32, within 3 ms of the copies alone. Above that, `over` is exactly
the compute of the *last* batch, the part that cannot overlap a copy (40 ms
when it is 384 aux functions, 13 at 128, 19 at 192: T_1 × its share of
naux). The m cut never gets below 543 ms, `over` 76–98 ms at every b: that
cut's one syrk runs only after the last slab lands, and nothing hides it.
For a streamed build **the P cut is the one to use**: its syrk goes with
each batch.

### Noise

At nao 640 the resident reps spread 4–7% (IQR) while the minima sit within
0.1% of the median: a few slow reps per config, consistent with clock or
thermal excursions over a 40 min resident run. The pipeline group's IQR is
≤ 0.1%. The conclusions above are all 3% effects or larger, or zero.

### Still to do

1. The same sweep on gfx1200 (`hip-experimental`).
2. The m cut's b = 1 result: A/B one gemm per slab against the one batched
   call in `df_exchange` itself, on both cards.
