# Pipeline footprint

Footprints as the pipeline sees them: the plan dimensions every step's
footprint is computed from, one step's footprint, and the plan fold that finds
the peak a plan reaches and the step that reaches it. The vocabulary itself —
`Footprint<K>`, `MemoryFootprint`, `then`, `drop` — is
[`core/footprint`](../../core/footprint/README.md).

## Module

`free_breakfast.pipeline.footprint`, target
`free_breakfast::pipeline::footprint`. It imports `core.types` and
`core.error_handling`, and re-exports `core.footprint` and `basis`, so a step
module importing it has `MemoryFootprint`, `then` and `drop`, and can read
the bases the dimensions carry. `basis` is host data; nothing GPU-side,
nothing from `pipeline.step` and no method layer: every step module imports
it. The implementation unit is `check_plan`'s and `check_pinned_node`'s, for
their messages and log lines.

| Export | What |
|---|---|
| `FootprintDimensions` | The plan's dimensions, before the first step: the placed orbital basis and, under density fitting, the aux basis, and under AVAS MINAO (const pointers), the molecule's atoms, nao, naux, nocc / n_alpha / n_beta, natoms, mmd's record counts per (la, lb), syevd's lwork at every order through the plan's largest eigensolve, skew_eig's at nao for a planned CASSCF, and worst-case ncas / nelecas (0 when no active space is planned). [`pipeline/shape`](../shape/README.md)'s `plan_shape` fills it |
| `step_footprint(step, dims)` | A step's footprint, or none for a step that has no footprint member yet |
| `PlanFootprint`, `plan_footprint(steps)` | The plan fold: `then` over one device's step footprints in plan order, with the step at which each kind's peak is reached |
| `step_adaptive_unit(step, dims)` | An adaptive step's bytes for one more unit of its adaptive part, or none for a step that is not adaptive |
| `StepFootprint` | One planned step as pre-flight sees it: its name, footprint and adaptive unit. `Pipeline::footprints(dims)` lists them in plan order |
| `PlanCaps` | The `[memory]` caps as the fold is held to them: `device_room`, the most the device peak may reach, and the two caps |
| `check_plan(steps, caps)` | The fold held to the caps: refused naming the step with no footprint, or the step reaching a peak past its cap; logs the peaks, the workspace gap and each adaptive step's room |
| `check_pinned_node(cap, devices, mem_available)` | The machine check: the devices times the pinned cap within MemAvailable |

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
at naux: no step fits with lindep. `skew_eig_lwork`, beside it, is the CASSCF
orbital step's other eigensolver query, `skew_eig`'s heevd at nao, filled
only when a CASSCF is planned.

**The CASSCF's roots are carried for its readers.** `casscf_nroots` and
`casscf_unequal_weights` are the planned CASSCF step's roots and whether two
of its weights differ. The CasscfGradient and Nac steps read the CASSCF's
product, so their footprints are sized over its CI vectors (dim × nroots)
and, for a state average, the Z-vector, whose rotations among the roots are
allocated only at unequal weights. Neither step carries the CASSCF's
options, so `plan_shape` copies these two. Only these two are carried, not
the weights themselves: nothing a footprint counts depends on their values.

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
steps CASCI, FCI and CASSCF theirs (#1049, #1151).
`test/gtest/pipeline_step_footprint_gpu.cpp` is the step test's pattern: the
dimensions are `plan_shape`'s over a parsed config, the step is the one it
planned, and each runs on a fresh `DeviceResources`, once to grow the
workspaces to exactly the footprint's, then again, measured, against the rest
by `expect_footprint`'s two bars.

**A nested step's footprint is optional, for now.** `OptimizeStep` and
`HessianStep` rerun a chain whose shape their configuration picks, and every
configuration of both is footprinted: the RHF and the CASSCF, NEVPT2 and
MECI chains under `[optimize]` (#1054, #1212: each evaluation's chain, what
it runs past it, and a followed state's C and CI between evaluations), and
every `[hessian]` -- the semi-numerical RHF one (#1054), the analytic
DF-RHF (#1053), DF-CASSCF (#1176) and SC-, PC- and QD-NEVPT2 (#1186) ones,
and the semi-numerical CASSCF and NEVPT2 / QD-NEVPT2 ones (#1204: the
reference's chain, then each displacement's from its carried orbitals, and
under QD the state's gradient and the ensemble's edge beside each). Their
member is still
`std::optional<MemoryFootprint> footprint(const FootprintDimensions &) const`,
which always has a value; #1122 makes it a plain `MemoryFootprint`.
`step_footprint` passes it through.
This is transitional: once those chains are footprinted, the member returns
a `MemoryFootprint` like every other step's, and #1122 deletes the optional
form with the detection.

**No public trait.** Every step will carry a footprint, and #1122 makes it a
requirement of `PipelineStep`, so a "has a footprint" trait like
`LoadableStep` would have nothing to sort. Until then an unexported detection
recognises the member, and only `step_footprint` uses it (and its twin
`step_adaptive_unit`, the optional adaptive unit): a member that is
non-const, static, overloaded, a template or of another signature is not a
footprint, and `step_footprint` returns `std::nullopt`, so the fold's caller
can name the step rather than count it as zero. #1122 deletes both.

## An adaptive step's unit

An adaptive step's footprint scratch is the least it runs with
(`core/footprint`): the cap is checked against that, and the step takes what
is left at run time. A step that says how much one more unit of its adaptive
part takes,

```cpp
size_t adaptive_unit(const FootprintDimensions &dims) const;
```

gets a line from pre-flight: what the cap leaves it above its least at its
point in the fold, at warn level when that is under one unit, so "fits, but
runs at width 1" shows before the run starts. The member is optional, and
recognised as `footprint` is (`step_adaptive_unit`); no step carries one
yet, so the warning is exercised by the made-up steps of
`test/pipeline/preflight_host.cpp`. FCI's sigma blocks and the NEVPT2 tiles
are the first that should.

## Holding the fold to the caps

`check_plan(steps, caps)` folds a plan's `StepFootprint`s and holds each
kind's peak to its cap: the device peak to `caps.device_room` (the pool's
cap less the headroom and any reuse rounding; `pipeline/shape` computes it),
the pinned peak to `max_pinned_mem_per_gpu` exactly, since pinned memory is
not pooled. Past either it is `InvalidConfig` naming the step that reaches
the peak, the peak and the cap. **A step with no footprint is refused** under
any cap, naming it, rather than counted as zero, which could pass a plan that
does not fit: until every step has one (#1122), a capped plan is one of
footprinted steps. Without a cap it is not called at all. Its log lines are
each kind's peak and step, the workspace gap (the plan's maximum workspace,
held throughout, against what the peak step uses of it) and each adaptive
step's room. `pipeline/README.md`'s "Memory caps" has the whole of
pre-flight.

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
  inner plan is folded by `plan_footprint` over its own steps' footprints,
  listed by the very `Pipeline` the step runs (`footprints(dims)`), so its
  order is the run's; it is freed whole before the next geometry, so it is
  `drop`ped, and every geometry has the same dimensions, so one fold holds
  for all of them. The inner steps share the outer's `DeviceResources`, so
  their workspace is in the outer maximum, and pre-flight's pre-grow covers
  every geometry. `OptimizeStep` holds the start's Molecule beside the chain
  and keeps the optimum's; `HessianStep` keeps nothing on the device, and
  runs `geometry::harmonic_analysis` after the last displacement. The
  optimizer's history and trajectory and the Hessian's 6N gradients are
  pageable host memory, not modelled.
- **No step is exempt**, test-only ones included. A trivial step writes its
  exact small footprint, and a host fake writes `return {};`. "Negligible"
  has no threshold, and a step that grows would go unseen.
- **Placement.** The device workspace is for what many steps use at similar
  sizes (a library's `lwork`, solver work). A one-step intermediate goes in
  named scratch, because the workspace holds its maximum through every step;
  pre-flight logs the gap. Fixed-size staging, bounce buffers and ring
  slots go in the pinned workspace. A product kept in pinned memory (an
  offloaded B) is `pinned.persistent`.
- **A loaded product is persistent too.** A `Load`ed step's product is as
  resident as a computed one.

## Fragmentation headroom

A plan whose footprint peak fits under a cap can still be refused by the pool:
the pool takes memory from the driver in chunks, and what it has reserved, not
what is in use, is what `maxSize` bounds. Pre-flight therefore leaves
**64 MiB** (two pool chunks, `kPoolHeadroomBytes` in `core/footprint`) between a plan's device peak and the device's
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
(`test/support/pool_footprint.h`), logged when used, and nothing on CUDA. The
bound, 1/8 of a peak, is larger than 64 MiB for a peak past 512 MiB, so on a
pool that rounds, pre-flight leaves it beside the headroom: a peak P must fit
the pool's cap less 64 MiB and ⌈P/8⌉ (`pipeline/shape`'s README, "The HIP
margin", has why the two add).

**HIP's gap is not measured here.** ROCm's pool reserves in its own steps
and resets a high-water mark to 0 (`core/wwr`'s README); its reserved gap
needs its own run on the gfx1200. Until then a HIP pre-flight leans on this
number for the chunks and adds the reuse bound above for the rounding.
