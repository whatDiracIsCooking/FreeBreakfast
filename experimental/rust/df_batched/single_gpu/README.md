# experimental/rust/df_batched/single_gpu — the batched DF factor B on one GPU

A driver and benchmark for building the density-fitting factor

```text
B[Q][mn] = sum_P L^-1[Q][P] (P|mn),   M = (P|Q) = L L^T
```

for a molecule whose B does not fit on the card but does fit in host memory.
It copies no production code. It calls rust/'s
`fb_integrals_eri_density_fitting_three_center::add_packed_b` unchanged,
from crates it depends on by path, and measures it.

## The scheme under study

Everything O(N²) stays resident on the card: the orbital records, the aux
tables, U for both bases, the metric and its factor L. The O(N³) tensors
(mn|P) and B never are. The loop runs over **batches of shell pairs**,
never over aux functions. Each column mn of B depends on every P, so a
P-batch cannot be fitted on its own. Each trip:

1. `packed three-center`: (P|mn) for every P and the batch's pairs
   (FelizJueves' packed `peri3<la, lb, lP>` kernels, one launch per class)
2. `spherical transform`: Cartesian to spherical on both indices (GEMMs)
3. `density fit`: the trsm against L, in place
4. `B (la, lb) batch transpose`: one `geam`, aux-fastest to pair-fastest
5. `download` + `unstage`: a pitched copy into B's heap row, `naux` runs
   at a pitch of `n_columns`

A batch never spans two classes `(la, lb)`. Each class is a counted loop
with ring depth 2, so trip k+1's launches run under trip k's download. The
batch width is the planner's adaptive part `packed B batch`: the widest
that fits the device room.

(mn|P) never reaches the host. Compared with building all of (mn|P) and
then fitting it in a second pass, this halves host memory and cuts PCIe
traffic from 3·|B| to |B|. `report.html` has the numbers.

## Build and run

The CUDA kernel libraries come first, from `rust/`:

```bash
cd rust && cmake --preset cuda && cmake --build --preset cuda && cd -
cargo build --release --manifest-path experimental/rust/df_batched/single_gpu/Cargo.toml
```

`FB_RUST_BUILD_DIR` points at another CMake build directory.

```bash
df-batched --xyz gly10.xyz                     # the card's free memory as the room
df-batched --xyz gly10.xyz --room-mib 300 --room-mib 2400 --compare   # a width sweep
```

| flag | default | |
|---|---|---|
| `--xyz` | required | the molecule |
| `--basis`, `--aux` | `cc-pvdz`, `cc-pvdz-jkfit` | compiled-in sets |
| `--schwarz` | `1e-12` | the pair screen, as `[eri] schwarz_threshold`; 0 keeps every pair |
| `--room-mib` | free memory less 512 MiB | device room for the plan (the BLAS workspace on top); repeat to sweep |
| `--reps` | 3 | timed runs after one warm-up, median reported |
| `--compare` | off | copy B out and hold each run's against the first's |
| `--plan-out`, `--record-out` | | the plan text and the last run's `timing.json` |

The JSON on stdout carries the sizes, the plan's width and trips, device
and pinned bytes, per-task host and device time, and B's build time. That
time is the full plan's wall less a prefix plan's (records, metric and
Cholesky), both warm. Hold both CUDA locks while timing, since other
worktrees share the card:

```bash
L=/home/ubuntu/.sccache/gpu-locks
flock $L/cuda.0.lock flock $L/cuda.1.lock df-batched --xyz gly10.xyz
```

## The report

`report.html` is generated from `report/data/`, which holds two series:

- `before/`: the Cartesian-aux path (`peri3` plus a dense U_aux GEMM), at
  `cb30cc80`.
- `after/`: #1736's spherical-aux kernels (`speri3`), at `40afb73b`.

Each holds the JSON of the runs, `timeline_gly10.json` (extracted by
`report/timeline.py` from an nsys trace) and `kernels_gly10.json`, the
per-class device time of the three-center kernels (extracted by
`report/kernels.py` from the same trace). Every number on the page is read
from those files.

```bash
nsys export -t sqlite -o run.sqlite run.nsys-rep
python3 report/timeline.py run.sqlite report/data/after/timeline_gly10.json
python3 report/kernels.py run.sqlite report/data/after/kernels_gly10.json
python3 report/make_report.py report/data report.html
```

What it found, on the RTX 3080 with glycine chains in cc-pVDZ /
cc-pVDZ-JKFIT:

- The build is compute-bound on one stream. At gly20, B is 11.6 GB, built
  on a 10 GB card, and the download is 1.1% of device time before #1736
  and 3.3% after.
- Before #1736, the spherical transform was two thirds of the time, nearly
  all of it the dense `U_auxᵀ X` GEMM against a block-diagonal U_aux. With
  #1736's `speri3` kernels that GEMM is gone: gly20's B builds in 31.5 s
  instead of 90.2 s (2.86×), gly10's in 5.2 s instead of 12.4 s (2.40×).
  The gain grows with size.
- The trsm against L is now about four fifths of device time, and the next
  lever.
- `speri3`, low-stack only, makes the integrals 1.11× slower overall.
  Several low classes lost to the old unrolled `peri3` by up to 2.3×.
  That is small beside the GEMM it replaced.
- Width stops mattering past about 20 trips. Width 1 costs 5.6× before and
  11.6× after, because each trip's fixed cost now weighs more.
- Compared with two passes ((mn|P) to the host, then fitted), the fused
  loop's gain is mostly host memory, |B| against 2·|B|. The extra PCIe
  traffic would add only a few percent to a compute-bound build.
