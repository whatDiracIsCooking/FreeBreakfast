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
| `PlanShape` | The placed bases, owned (`basis`, `aux`, `minao`), and `dims`, whose `basis`, `aux` and `minao` point at them. The bases sit behind `unique_ptr`, so a move leaves the pointers valid |
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
  carried `-acCD-6-6`, through the Eri step's own `default_aux_basis`.
- **mmd's records**: for each class (la, lb), the sum of nprim_i * nprim_j over
  the shell pairs i <= j. mmd screens nothing, so this is exactly what
  `primitive_pair_classes` builds (`test/gtest/pipeline_shape_gpu.cpp` checks
  the two class by class).
- **syevd's lwork** at every order from 0 through the plan's largest
  eigensolve, `recipes/diagonalize`'s own `syevd_lwork` query through
  `res.solver()`, one host call an order. The largest is nao, or a planned
  CI's subspace past it: ci::solve's Davidson diagonalizes at up to 16 nroots
  columns (CASCI, FCI and CASSCF, the chain copies included) and a CASSCF's
  augmented Hessian at ah_max_subspace + 1. The RHF chain reads
  `syevd_lwork[nao]`; the CI steps' small eigensolves (AVAS's, the subspace's)
  read their own orders.
- **MINAO**, placed and primitive-normalized as AVAS places it, when any
  planned active space is AVAS's, and the molecule's **atoms**, which AVAS's
  labels are matched against: what the selection's footprint sizes MINAO's
  overlaps and the labelled count from.
- **The active space** at its largest. An explicit `ncas` / `nelecas` is used
  as written. Under AVAS it is `avas_max_ncas` when set, and otherwise twice
  the MINAO AOs the labels name (`pipeline/active_space`'s own
  `avas_minao_aos`): the occupied projector and the virtual one each have
  at most that rank. That count is held to nao, and nelecas is set to ncas
  (half filling, the most determinants for that many orbitals), capped at the
  electron count. FCI's space is nao orbitals and every electron. When
  several steps plan one, the space with the most orbitals wins, and a tie
  goes to the nelecas nearest half filling. Chain copies count only where
  their chain runs a CASSCF.
