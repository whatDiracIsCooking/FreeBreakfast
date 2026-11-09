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
| `FootprintDimensions` | The plan's dimensions, before the first step: the placed orbital basis and, under density fitting, the aux basis (const pointers), nao, naux, nocc / n_alpha / n_beta, natoms, mmd's record counts per (la, lb), syevd's lwork at nao, and worst-case ncas / nelecas (0 when no active space is planned). [`pipeline/shape`](../shape/README.md)'s `plan_shape` fills it |
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

**syevd's lwork is a dimension.** `diagonalize`, `orthogonalize` and the
lindep fit size their workspace by the solver's own query, a host computation
through a handle. `plan_shape` makes it once, at nao, through the plan's
`DeviceResources`, so a step's `footprint(dims)` holds no handle. There is
none at naux: no step fits with lindep.

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

The RHF chain's steps carry theirs, from the Molecule to the Scf.
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
**32 MiB** (one pool chunk) between a plan's device peak and the device's
budget, and an adaptive routine sizes its adaptive part to the budget less its
own allocations still to come and this headroom (`core/footprint`).

**The measurement.** `test/gtest/pipeline_footprint_gpu.cpp` runs the RHF
chain, every step from the Molecule to the Scf, by hand on a fresh
`DeviceResources`, and reads its pool's `ReservedMemHigh` against its
`UsedMemHigh`. On the RTX 3080 (CUDA 13.0, driver 580.126.20, the default
1 GiB release threshold, no cap):

| Chain | UsedMemHigh | ReservedMemHigh | Reserved past used |
|---|---|---|---|
| water, cc-pVDZ, integral-direct | 1,724,192 | 33,554,432 | 31,830,240 |
| water, cc-pVDZ, density-fitted | 2,778,052 | 33,554,432 | 30,776,380 |
| water, cc-pVTZ, integral-direct | 6,863,968 | 33,554,432 | 26,690,464 |
| water, cc-pVTZ, density-fitted | 46,946,660 | 67,108,864 | 20,162,204 |
| benzene, cc-pVTZ, density-fitted | 3,957,730,548 | 3,959,422,976 | 1,692,428 |

**The derivation.** Every gap is under one 32 MiB chunk (33,554,432 bytes),
however large the run: the pool sub-allocates small requests from 32 MiB
chunks and reserves a large one at its own size (benzene's B and three-center
scratch, each past a chunk, leave 1.7 MB), so what it holds past its peak use
is the unfilled part of the chunk its last small allocations opened. The
chain's per-class streams (mmd's, J's and K's classes), which allocate beside
`res.stream()`, leave no more. The
headroom is that one chunk, not the largest gap measured, because the gap is
the chunk's remainder and any run can leave nearly all of it. The test holds
each chain's gap to it.

**HIP is not measured here.** ROCm's pool reserves in its own steps and resets
a high-water mark to 0 (`core/wwr`'s README); its gap needs its own run on the
gfx1200 before a HIP pre-flight leans on this number.
