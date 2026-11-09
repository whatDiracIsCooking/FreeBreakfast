# Pipeline shape

A planned `Calculation`'s shape: its bases placed and its
[`FootprintDimensions`](../footprint/README.md) filled, before the first step
runs. Pre-flight (#1051) folds the step footprints over it.

## Module

`free_breakfast.pipeline.shape`, target `free_breakfast::pipeline::shape`. It
re-exports `pipeline.config` (for `Calculation`) and `pipeline.footprint`.
Not a step, and not part of `pipeline.config` either: the parser is one
concern and the shape is another. It cannot live in `pipeline.footprint`,
because every step module imports that one, and it would then pull in the
whole registry.

| Export | What |
|---|---|
| `PlanShape` | The placed bases, owned (`basis`, `aux`), and `dims`, whose `basis` and `aux` point at them. The bases sit behind `unique_ptr`, so a move leaves the pointers valid |
| `plan_shape(calc, res)` | The shape of `calc`, on `res`'s device |

## What it reads

- **The molecule** from the Molecule step. When an `[optimize]` loads it,
  `plan_shape` reads the chain's start instead: the loaded `final.xyz` does
  not exist before the run, and must hold the same atoms. `read_xyz` uploads
  the nuclei, so a `DeviceResources` is needed even before syevd's query.
  The nuclei are freed on return.
- **The bases**, placed as the BasisSet and Eri steps place them, primitives
  normalized. They come from the top-level steps, or else from the copy an
  `[optimize]`'s or `[hessian]`'s chain holds (the config reads those from
  the same tables). An empty `aux_basis` falls back to the orbital set's
  carried `-acCD-6-6`, by the Eri step's rule.
- **mmd's records**: for each class (la, lb), the sum of nprim_i * nprim_j over
  the shell pairs i <= j. mmd screens nothing, so this is exactly what
  `primitive_pair_classes` builds (`test/gtest/pipeline_shape_gpu.cpp` checks
  the two class by class).
- **syevd's lwork** at nao, from `syevd_bufferSize` through `res.solver()`,
  with diagonalize's eigenvector mode and lower triangle. The test pins this
  to `diagonalize_footprint`'s workspace.
- **The active space** at its largest. An explicit `ncas` / `nelecas` is used
  as written. Under AVAS it is `avas_max_ncas` when set, and otherwise twice
  the labelled MINAO AOs: the occupied projector and the virtual one each have
  at most that rank. That count is held to nao, and nelecas is set to ncas
  (half filling, the most determinants for that many orbitals), capped at the
  electron count. FCI's space is nao orbitals and every electron. When
  several steps plan one, the space with the most orbitals wins, and a tie
  goes to the nelecas nearest half filling. Chain copies count only where
  their chain runs a CASSCF.

AVAS's label rule is copied from `pipeline/active_space`'s `avas_minao_aos`,
which is private to that module. Keep the two in step.
