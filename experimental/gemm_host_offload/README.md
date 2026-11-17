# `gemm_host_offload/`

The DF exchange build `K = Xᵀ X`, `X_m = B_m C`, cut into batches, with B
resident on the device, or streamed through a ring of device slots from pinned
host memory or from the heap (packed into pinned staging slots first). The question: **what does each batch cost on top of the one-call
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
    [--only resident|pipeline] [--ring-gib 6] [--pack-threads 1[,2,4,8]]
    [--aux-batches 32,64] [--csv gemm_host_offload.csv]
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
| `heap` | heap | two hops: the enqueuing CPU thread packs each batch into a pinned staging slot (once that slot's previous upload has landed), then the copy stream uploads it as `pipeline` does. The pack of batch i+1 overlaps the upload and the gemms of batch i |
| `heap-copies` | heap | `heap` with the gemms removed: the two hops alone |

`heap` is the Rust design's path (`rust/docs/memory.md`, "Pinned is staging;
large flat host data is a heap row"): B in a heap row, streamed through a
pinned window. The staging ring has the device ring's depth R and slot size.
`--pack-threads` splits each pack over N CPU threads (default 1, the
one CPU node of the design). It takes a list: each N is its own `heap` and
`heap-copies` curve (`heap P R2 N4`), in the same passes as `pipeline`. The
threads are a pool kept for the whole run: the enqueuing thread copies slice 0
and N − 1 workers, parked on a barrier between packs, copy the rest, so a
batch pays two barrier crossings rather than N thread starts.
`--aux-batches` narrows the pipeline group to those P batch sizes and drops
its m cut. Both host copies of B are written before any
timing, so no first-touch fault is in a timed pack.

A pipeline config whose ring does not fit `--ring-gib` (the device ring and
the pinned staging ring alike) is skipped and said to be.

## Method

Every config runs `warmup` times untimed, then `reps` times, GPU-event timed.
Within a group the configs are round-robin, one run of each per pass, so clock
and thermal drift spreads over all of them. The pipeline's P cut is the
exception: each batch size needs its own host layout, so it is round-robin per
batch size only.

Columns: median, min and interquartile range (% of the median); `excess/b` =
(median − T_1) / (n_b − 1), δ read off one point; for `pipeline`, `over` =
median − max(T_1, `copies` at that batch), what the overlap failed to hide
(for `heap`, against `heap-copies`). Per
curve, a least-squares fit of median against n_b gives δ with its standard
error. T_1 is the resident group's own `chopped m` n_b = 1 point; the pipeline
group, which runs after B is freed, uses a standalone T_1 timed beforehand.

K is checked once per config against the one-call build. The m cut keeps every
product identical, so it should agree bitwise unless cuBLAS picks a different
algorithm for a different batch count; P sums in another order, so it is a
relative difference.

Every rep goes to the CSV (`group,nao,naux,nocc,variant,mode,depth,threads,batch,n_b,rep,ms`;
`threads` is N for the heap variants, 0 otherwise).

## Results

Two cards. The RTX 3080 sections below were the whole sweep (resident and
pinned streaming, before `heap` existed). The RTX A6000 section reruns the
pipeline group, pinned and heap side by side, so that comparison is all on
one card.

### RTX 3080

RTX 3080 (PCIe 4 x16), clocks unpinned, `--reps 10 --warmup 3`, all
defaults, 2026-11-08; 65 min of a held card. Times are medians, ms.

**K agrees everywhere.** The m cut is bitwise against the one-call build at
every batch size but b = 1 (~1e-15, see below); the P cut agrees to ≤ 6e-14
relative, resident and streamed.

#### δ is not launch overhead or event waits (RTX 3080)

`chopped`, `events` and `graph` agree within their fit errors at every shape,
on both cuts. At nao 640, nocc 128, m cut: δ = −8.3, −7.5 and −10.4 ± 46
µs/batch. Neither capturing the launches away nor adding the ring's waits
changes anything measurable, so whatever a batch costs is the GEMMs' own.

#### The m cut: no linear δ, a step at small b, and b = 1 is fastest (RTX 3080)

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

#### The P cut: tile quantization, not δ (RTX 3080)

The P cut's straight-line fit (0.5–5 ms/batch) means nothing: its cost is a
function of b, not of n_b. At nao 640, nocc 128 (`chopped`, T_1 = 749.7):

| b | 1024…128 | 96 | 64 | 48 | 32 | 24 | 16 | 12 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ms | 749.7–753.8 | 904.0 | 754.0 | 904.5 | 755.7 | 906.4 | 1207.8 | 1509.5 | 2112.5 |

Every power of two from 32 up costs ≤ 1%; the 1.5× points 96, 48 and 24 cost
20%, the same 150 ms each, and below 32 the time grows like 1/b. nocc 64 and
nao 448 show the same pattern. **A P batch should be a power of two ≥ 32.**

#### Streaming B: the overlap works, and the model holds (RTX 3080)

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

#### Noise (RTX 3080)

At nao 640 the resident reps spread 4–7% (IQR) while the minima sit within
0.1% of the median: a few slow reps per config, consistent with clock or
thermal excursions over a 40 min resident run. The pipeline group's IQR is
≤ 0.1%. The conclusions above are all 3% effects or larger, or zero.

### RTX A6000: B streamed from the heap against pinned

RTX A6000 (sm_86, 48 GB, PCIe 4 x16), clocks unpinned, 2026-11-17,
`--only pipeline --nao 640 --reps 10 --warmup 3 --pack-threads 1`, ring
6 GiB, the card held alone; 40 min. Host: AMD EPYC (Rome), one thread's
memcpy measured separately at 16.1 GB/s (heap to pinned, same rate). B is
5.86 GiB (6.29 GB). Times are medians, ms, R = 2 (R = 3 matches it again).

**K agrees as before.** `heap` gives the same K as `pipeline` at every
config: the m cut bitwise (1.1e-15 at b = 1), the P cut ≤ 5.5e-14 relative.

**The rates.** Pinned `copies`: 240.3 ms at any chunking, **26.2 GB/s**
(near PCIe 4 x16's practical ceiling; the 3080 reached only 13.5). The
heap's two hops, `heap-copies`, are bound by the one-thread pack: 392–410 ms
at b = 8…128 (P) and b = 2…16 (m), **15.5–16 GB/s**, about the measured
memcpy rate. The pack overlaps the upload, so the upload adds nothing on top.
Small batches add a per-batch cost: the m cut at b = 1 (640 batches of
9.4 MB) takes 701 ms, about 0.47 ms per batch for the host wait on the slot
and the handoff. Large batches lose the overlap: the first pack and upload
are not hidden (P b = 768: 476 ms).

**Gemms limit (nocc 128, standalone T_1 = 680.3):** the overlap hides
nearly all of the 400 ms of the two hops.

| P b | 256 | 128 | 64 | 32 | m b | 16 | 8 | 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pipeline` | 717.7 | 702.1 | 699.3 | 708.0 | | 694.0 | 705.7 | 706.3 |
| `heap` | 770.5 | 728.0 | 712.9 | 715.3 | | 730.1 | 725.2 | 723.2 |

At P b = 64 the heap costs 13.6 ms (2%) more than pinned. `over` is again
the unhidden first batch, now pack plus upload (32.7 ms at P b = 64 against
pinned's 19.0). The tile penalty at 96, 48 and 24 is unchanged (844, 834,
842). The m cut's b = 1 trick (pinned 655.0) does not carry over: 640
batches cost the heap 1035 ms.

**Copies limit (nocc 32, T_1 = 169.2):** the heap build is the two hops'
time: 409–425 ms at P b = 12…128, within 5–15 ms of `heap-copies` (pinned:
243 ms). The m cut is 484–508 ms at b = 4…32, with `over` 78–86 ms as
pinned has: its one syrk after the last slab.

**Where the copies and the GEMMs cross.** T_1 is linear in nocc on this
card: 169.2 ms at 32, 680.3 at 128, about 5.3 ms per occupied orbital at
nao 640, naux 1920. The copies cost as much as the GEMMs at **nocc ≈ 45
pinned** (240 ms) and **nocc ≈ 75 from the heap** with one pack thread
(about 400 ms). Above that the stream is hidden; below it a build costs the
two hops. The batch to use does not change: a P batch of 32 or 64 is the
best or near it for heap and pinned, at both nocc.

**Four spawned pack threads** (`--pack-threads 4` as it was then, each pack
starting its own threads; same day and card, stopped after its nocc 32
group). `heap-copies` fell to 266–275 ms at P b = 64…128, about 23 GB/s, but
each batch cost about 1 ms more (fit δ 0.8–1.0 ms), so small batches lost:
P b = 8 took 543 ms, m b = 1 820. The pooled sweep below replaces it.

### RTX A6000: a pooled multi-threaded heap pack

RTX A6000 (sm_86, 48 GB, PCIe 4 x16), CUDA card 1 held alone, clocks
unpinned, 2026-11-17, `--only pipeline --nao 640 --reps 10 --warmup 3
--pack-threads 1,2,4,8 --aux-batches 32,64`, ring 6 GiB; 7.9 min. Same host
as above (AMD EPYC Rome). The pack threads are now a pool kept for the whole
run (N copy threads: the enqueuing thread and N − 1 workers on a barrier),
and every N runs in the same passes as `pipeline` and `copies`. Times are
medians, ms, R = 2 (R = 3 matches it within 1%); rates are B's 6.29 GB over
the time.

**K agrees as before**: every `heap` config at every N gives the
`pipeline` K, 2.0e-14 (nocc 32) and 4.3e-14 (nocc 128) relative to the
one-call build, inside 6e-14.

**The two hops** (`heap-copies`, against pinned `copies`):

| | pinned | N = 1 | N = 2 | N = 4 | N = 8 |
| --- | --- | --- | --- | --- | --- |
| nocc 32, P b = 32 | 239.7 | 386.8 | 243.0 | **242.2** | 250.1 |
| nocc 32, P b = 64 | 239.6 | 397.9 | 246.6 | 245.8 | 252.1 |
| nocc 128, P b = 32 | 240.3 | 401.4 | 251.0 | **243.6** | 250.0 |
| nocc 128, P b = 64 | 240.3 | 400.8 | 257.2 | 249.9 | 251.8 |
| best rate, GB/s | 26.2 | 16.3 | 25.9 | **26.0** | 25.2 |

**A pooled pack reaches pinned's rate at N = 4**: 242.2 ms at P b = 32,
26.0 GB/s, within 1% of pinned's 26.2 (the PCIe ceiling), at both nocc. N =
2 is nearly there (25.9 GB/s at nocc 32, but 251 ms in the nocc 128 group);
N = 8 is slightly worse than 4 (25.2 GB/s), with a wider spread; why (memory
contention, scheduling against the enqueuing thread) was not investigated.

**Per-batch overhead: none resolvable.** For every N ≥ 2, P b = 32 (60
batches) is 2–6 ms *faster* than b = 64 (30 batches), as the smaller
unhidden first batch predicts, so 30 more batches add no measurable cost.
The spawning pack paid about 1 ms per batch; the pool's two barrier
crossings per batch are lost in the noise.

**Builds** (`heap` against `pipeline`):

| | T_1 | pipeline | N = 1 | N = 2 | N = 4 | N = 8 |
| --- | --- | --- | --- | --- | --- | --- |
| nocc 32, P b = 32 | 169.2 | 242.8 | 393.6 | 256.9 | **245.2** | 252.5 |
| nocc 128, P b = 64 | 677.5 | 699.1 | 712.0 | 708.9 | 706.1 | **705.2** |

Copies limit (nocc 32): the heap build at N = 4 costs 245.2 ms, 1% over
pinned, against 393.6 at N = 1. Gemms limit (nocc 128): the overlap already
hid one thread's two hops (+1.8%); four or eight threads leave +1.0%, the
unhidden first batch (`over` 28 ms against pinned's 22). At nocc 128 P b =
64 beats b = 32 for every variant (pinned 699.1 against 708.2), so the best
batch is still 32 or 64: 32 when the copies limit, 64 when the GEMMs do.

**The crossover.** T_1 = 169.2 + 5.29 (nocc − 32) ms here. The copies and
the GEMMs cross at nocc ≈ 46 from the heap with a pooled pack of N ≥ 4 (242
ms), the same as pinned's 45; nocc ≈ 75 with one thread (about 395 ms) and
46–49 with two.

So for the Rust heap row: splitting `upload_heap`'s CPU stage over a pool of
four threads kept across batches brings the heap path to pinned's rate on
this host, at no per-batch cost.

### Still to do

1. The same sweep on gfx1200 (`hip-experimental`).
2. The m cut's b = 1 result: A/B one gemm per slab against the one batched
   call in `df_exchange` itself, on both cards.
3. The pooled pack on other hosts: four threads saturate PCIe 4 x16 on this
   EPYC; a host with less memory bandwidth per thread may need more.
