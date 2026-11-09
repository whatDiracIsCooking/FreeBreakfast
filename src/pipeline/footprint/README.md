# Pipeline footprint

Footprints as the pipeline sees them: the plan dimensions every step's
footprint is computed from, one step's footprint, and the plan fold that finds
the peak a plan reaches and the step that reaches it. The vocabulary itself —
`Footprint<K>`, `MemoryFootprint`, `then`, `drop` — is
[`core/footprint`](../../core/footprint/README.md).

## Module

`free_breakfast.pipeline.footprint`, target
`free_breakfast::pipeline::footprint`. It imports `core.types` and
`core.footprint`, and re-exports the latter, so a step module importing it
has `MemoryFootprint`, `then` and `drop`. Nothing from `pipeline.step` or any
method layer: every step module imports it. Templates and `constexpr`
functions only, so there is no implementation unit.

| Export | What |
|---|---|
| `FootprintDimensions` | The plan's dimensions, before the first step: nao (spherical and Cartesian), naux, nocc / n_alpha / n_beta, natoms, mmd's record counts per (la, lb), K's block counts per (la, lb), and worst-case ncas / nelecas (0 when no active space is planned). #1046 fills it |
| `step_footprint(step, dims)` | A step's footprint, or none for a step that has no footprint member yet |
| `PlanFootprint`, `plan_footprint(steps)` | The plan fold: `then` over one device's step footprints in plan order, with the step at which each kind's peak is reached |

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
