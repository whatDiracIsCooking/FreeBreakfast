# Pipeline footprint

Footprints as the pipeline sees them: the plan dimensions every step's
footprint is computed from, one step's footprint, and the plan fold that finds
the peak a plan reaches and the step that reaches it. The vocabulary itself —
`Footprint<K>`, `MemoryFootprint`, `then`, `drop` — is
[`core/footprint`](../../core/footprint/README.md).

## Module

`free_breakfast.pipeline.footprint`, target
`free_breakfast::pipeline::footprint`. It imports `core.types`, and
re-exports `core.footprint` and `basis`, so a step module importing it has
`MemoryFootprint`, `then` and `drop`, and can read the bases the dimensions
carry. `basis` is host data; nothing GPU-side, nothing from `pipeline.step`
and no method layer: every step module imports it. Templates and `constexpr`
functions only, so there is no implementation unit.

| Export | What |
|---|---|
| `FootprintDimensions` | The plan's dimensions, before the first step: the placed orbital basis and, under density fitting, the aux basis, and under AVAS MINAO (const pointers), the molecule's atoms, nao, naux, nocc / n_alpha / n_beta, natoms, mmd's record counts per (la, lb), syevd's lwork at every order through the plan's largest eigensolve, and worst-case ncas / nelecas (0 when no active space is planned). [`pipeline/shape`](../shape/README.md)'s `plan_shape` fills it |
| `step_footprint(step, dims)` | A step's footprint, or none for a step that has no footprint member yet |
| `PlanFootprint`, `plan_footprint(steps)` | The plan fold: `then` over one device's step footprints in plan order, with the step at which each kind's peak is reached |

## The dimensions

**The bases are carried, not counted.** mmd's, K's and the three-center
tensor's footprints, and the spherical transform's, take the host `BasisSet`
(and the aux one), their routines' own argument; their tables are sized per
shell and per primitive, which no count of AOs fixes. So `basis` and `aux`
point at the bases `plan_shape` placed, and a step footprint passes them
straight on. Counts the bases already say are not kept beside them: there is
no Cartesian AO count and no table of K's blocks. `nao` and `naux` stay, as
the scalars most footprints are written in. `mmd_records` stays because J's
footprint, `coulomb_pair_classes_footprint`, takes it.

**syevd's lwork is a dimension, by order.** `diagonalize`, `orthogonalize`,
`orthonormalize` and the lindep fit size their workspace by the solver's own
query, a host computation through a handle. `syevd_lwork[k]` is that query at
order k, for every k from 0 through the plan's largest eigensolve, which
`plan_shape` makes through the plan's `DeviceResources`, so a step's
`footprint(dims)` holds no handle. The RHF chain reads `syevd_lwork[nao]`.
The CI steps diagonalize at orders of their own: AVAS's labelled MINAO
overlap and its projectors, the natural orbitals at ncas, and the Davidson's
Rayleigh-Ritz problem, at every subspace size up to 16 nroots columns, which
can pass nao (FCI over 7 orbitals, 16 columns). A footprint reads each order
it diagonalizes at, and the Davidson's largest over its subspace: the most it
can take, reached when the subspace fills before it converges. One table,
one name: the largest order is nao, a CI's 16 nroots, a CASSCF's augmented
Hessian (ah_max_subspace + 1) or MINAO's AO count under AVAS. There is none
at naux: no step fits with lindep.

**MINAO and the atoms are carried for AVAS.** The selection's footprint sizes
MINAO's overlaps from the MINAO basis `plan_shape` places, and the labelled
AOs from the labels matched against the atoms (`avas_minao_aos`), so neither
is counted beside them.

## A step's footprint

```cpp
struct ScfStep {
  ...
  MemoryFootprint footprint(const FootprintDimensions &dims) const;
};
```

It is `const` because a step's options (`nroots`, …) set sizes. It unpacks
the dimensions into its routines' arguments and composes the footprints those
routines export (`core/footprint`'s library pattern) with `then` and `drop`,
so it reads like the routine it describes:
`return then(drop(b_mo_fp), drop(jk_fp));`.

The RHF chain's steps carry theirs, from the Molecule to the Scf, and the CI
steps CASCI and FCI theirs (#1049; CASSCF's is #1151).
`test/gtest/pipeline_step_footprint_gpu.cpp` is the step test's pattern: the
dimensions are `plan_shape`'s over a parsed config, the step is the one it
planned, and each runs on a fresh `DeviceResources`, once to grow the
workspaces to exactly the footprint's, then again, measured, against the rest
by `expect_footprint`'s two bars.

**No public trait.** Every step will carry a footprint, and #1122 makes it a
requirement of `PipelineStep`, so a "has a footprint" trait like
`LoadableStep` would have nothing to sort. Until then an unexported detection
recognises the member, and only `step_footprint` uses it: a member that is
non-const, static, overloaded, a template or of another signature is not a
footprint, and `step_footprint` returns `std::nullopt`, so the fold's caller
can name the step rather than count it as zero. #1122 deletes both.

## The peak model

`plan_footprint` folds a plan's step footprints with `then`, in plan order, per
kind. **At step k the device holds every step's persistent bytes up to and
including k, step k's scratch, and the plan's maximum workspace**:

```
peak(k) = Σ_{j ≤ k} persistent(j) + scratch(k) + max_j workspace(j)
```

`footprint.device.peak()` is the largest `peak(k)` and `device_peak_step` the
first k that reaches it; pinned alike. The workspace term is the maximum at
every step because the workspace is one block per device, grown to the plan's
maximum before the first step and held through every one.

- **Device and pinned fold alike.** Device-shaped steps run on one serial
  chain (`Pipeline::run`), so device scratch is one step's at a time. Pinned
  memory is owned per device by `DeviceResources`, and only steps that hold
  one can take it, so pinned folds per device along the same chain. Host-shaped
  steps, which may run beside the chain, take neither. The fold is one
  device's chain; a caller with several devices folds each one's.
- **Pageable host memory is not modelled.** The limits come from device and
  pinned memory. Every pageable host allocation in the tree is O(nao²) or
  smaller, and large host-side data is pinned. A step that would move
  something big to pageable memory moves it to pinned memory instead, where
  the footprint counts it.
- **Nested pipelines** (`[optimize]`, semi-numerical `[hessian]`): the outer
  step's footprint is the outer persistent plus the inner plan's peak. The
  inner steps share the outer's `DeviceResources`, so their workspace is in
  the outer maximum.
- **No step is exempt**, test-only ones included. A trivial step writes its
  exact small footprint, and a host fake writes `return {};`. "Negligible"
  has no threshold, and a step that grows would go unseen.
- **Placement.** The device workspace is for what many steps use at similar
  sizes (a library's `lwork`, solver work). A one-step intermediate goes in
  named scratch, because the workspace holds its maximum through every step;
  pre-flight logs the gap (#1051). Fixed-size staging, bounce buffers and ring
  slots go in the pinned workspace. A product kept in pinned memory (an
  offloaded B) is `pinned.persistent`.
- **A loaded product is persistent too.** A `Load`ed step's product is as
  resident as a computed one.

## Fragmentation headroom

A plan whose footprint peak fits under a cap can still be refused by the pool:
the pool takes memory from the driver in chunks, and what it has reserved, not
what is in use, is what `maxSize` bounds. Pre-flight (#1051) therefore leaves
**64 MiB** (two pool chunks) between a plan's device peak and the device's
budget, and an adaptive routine sizes its adaptive part to the budget less its
own allocations still to come and this headroom (`core/footprint`).

**How the pool reserves.** Measured on the RTX 3080 (CUDA 13.0, driver
580.126.20, the default 1 GiB release threshold, no cap), with a probe of
bare `cudaMallocFromPoolAsync` calls and with the chain's own allocations
logged one by one:

- It reserves in **32 MiB chunks**. A small request is carved from a chunk
  already held; a request no free extent holds is placed at the top of the
  pool's address range, its first bytes in the free top of the last chunk and
  the rest in new chunks, so a large allocation's reservation is its size
  rounded up to the chunk.
- A chunk whose free has completed and that holds nothing live is remapped
  into such a request (a 100 MiB request after 96 MiB of freed chunks grew the
  reservation by one chunk). A chunk whose free is **still pending** on the
  stream, the kernels that read it not yet done, is not: the request takes
  new chunks instead. Nor is the free part of a chunk that also holds a live
  allocation.

**The bound.** The reserved high-water mark is set when the reservation
grows. What the pool then holds past its use is the free part of every chunk
it holds: the top of the last chunk, under one chunk; and below it, each chunk
that holds both live and free bytes, and each held free chunk. A plan whose
allocations **pack** -- nothing live above a freed extent at the peak, and a
large allocation made only once the frees before it have completed -- leaves
below the top only the holes its small frees leave in the chunk its small
allocations share (the RHF chain's setup steps allocate 18 MB of them, under
one chunk) and the 512 B alignment of each allocation. So a packed plan's
gap is under **two chunks**, and the headroom is two chunks, not the largest
gap measured, because each part can be nearly a whole chunk in another run.
Each break in the packing costs more and is the allocation pattern's to fix,
not the headroom's to cover: a hole, a freed extent under a live allocation,
strands the free part of the two chunks at its edges, and a large allocation
made while frees are pending strands the whole freed extent (a probe that
frees 117 MiB under a running kernel and then allocates 868 MiB holds the
117 MiB, and at the peak, 2.7 GB later, still 35 MiB of it; with the frees
complete, 3 MiB).

**The measurement.** `test/gtest/pipeline_footprint_gpu.cpp` runs the RHF
chain, every step from the Molecule to the Scf, by hand on a fresh
`DeviceResources`, and reads its pool's `ReservedMemHigh` against its
`UsedMemHigh`, held to the headroom:

| Chain | UsedMemHigh | ReservedMemHigh | Reserved past used |
|---|---|---|---|
| water, cc-pVDZ, integral-direct | 1,724,192 | 33,554,432 | 31,830,240 |
| water, cc-pVDZ, density-fitted | 2,419,948 | 33,554,432 | 31,134,484 |
| water, cc-pVTZ, integral-direct | 6,863,968 | 33,554,432 | 26,690,464 |
| water, cc-pVTZ, density-fitted | 42,317,820 | 67,108,864 | 24,791,044 |
| benzene, cc-pVTZ, density-fitted | 3,856,607,988 | 3,858,759,680 | 2,151,692 |

Water's gaps are its one partly used chunk. Benzene's 2,151,692 bytes are
the small allocations' holes in the first chunk (1,458,428), the top of the
last (692,736) and two allocations' alignment (528). The chain's per-class
streams (mmd's, J's and K's classes), which allocate beside `res.stream()`,
leave no more.

**#1153: the DF Eri broke the packing.** Once #1148 freed the metric's pair
and the aux U before B, benzene's gap was 35,706,124 bytes, past the one
chunk then taken as the headroom: the device workspace, grown to metric's
staging inside the block, sat above the pair and outlived it, and B was
allocated while the block's frees were still pending. B went above both,
the three-center scratch refilled 85 MiB of the block's 117, and the 31.5 MiB
left straddled two chunks (4,602,624 and 28,446,208 bytes free) beside the
first chunk's and the top's. The Eri now grows the workspace before the
block and synchronizes before B (`pipeline/eri`'s README); B takes the
block's place, and the chain packs.

**Reuse rounding (HIP).** ROCm's pool also serves a request whole from a
block an earlier allocation freed, when the block is at most 9/8 of the
request, and counts the block: up to request / 8 more than asked, on reuse
only (`core/wwr`'s README has the measurement). A step that frees one phase
before the next allocates, as CASCI and FCI free the Hamiltonian before the
CI, can then peak a few bytes past its footprint on the gfx1200 (12 B and
64 B measured, #1049). The footprint still counts requested bytes, and the
footprint tests' sound bar allows the pool its rounding there: at most 1/8
of the footprint, the sum of the requests live at the peak
(`test/support/pool_footprint.h`), logged when used, and nothing on CUDA. At
run time pre-flight's 64 MiB headroom covers the overcounts measured, as it
covers the chunk rounding above. The bound itself, 1/8 of a footprint, is
larger than 64 MiB for a step past 512 MiB, so a HIP plan whose peak sits
within that of its cap could still be refused by the pool: pre-flight's to
weigh (#1051).

**HIP is not measured here.** ROCm's pool reserves in its own steps and resets
a high-water mark to 0 (`core/wwr`'s README); its gap needs its own run on the
gfx1200 before a HIP pre-flight leans on this number.
