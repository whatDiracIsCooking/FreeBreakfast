# Pipeline shape

A planned `Calculation`'s shape: its bases placed and its
[`FootprintDimensions`](../footprint/README.md) filled, before the first step
runs; and pre-flight, which folds the step footprints over it and holds the
plan to the `[memory]` caps.

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
| `preflight(calc, node)` | `node`'s devices rebuilt with their workspaces at `calc`'s plan maximum before the first step; under a `[memory]` cap, `calc` held to it first, and the devices built under it |

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
  augmented Hessian at ah_max_subspace + 1. Under a `quasi_degenerate`
  `[nevpt2_gradient]` a `[hessian]` also solves the ensemble's edge, a CI
  over one more root than its `[casscf]`, so through 16 (nroots + 1). A CIS step's Davidson
  diagonalizes at up to its clamped subspace, max(2 n_states,
  min(max_subspace, n_ov)), n_ov = nocc (nao - nocc), which passes nao with
  the default 40 columns on water in cc-pVDZ (nao 24, n_ov 95). The RHF chain
  reads `syevd_lwork[nao]`; the CI steps' small eigensolves (AVAS's, the
  subspace's) read their own orders. A planned `[hessian]` takes the table
  through 3 natoms, where its harmonic analysis diagonalizes (and at its
  internal modes below). With an NEVPT2 step planned (`[nevpt2]`,
  `[nevpt2_gradient]` or `[nevpt2_nac]`) it runs on through n_act³: PC
  diagonalizes each class's metric at its basis size, n_act³ for Sr and Si.
- **getrf's lwork** at the RIS basis's order, `recipes/lu`'s own
  `getrf_lwork` query, when a CIS step's `ris` preconditioner is planned: its
  capacitance matrix is solved by `lu_solve`. The RIS basis is `cis::ris_basis`
  over the atoms at the step's `ris` θ and shells, whose error `plan_shape`
  returns. 0 otherwise.
- **skew_eig's lwork** at nao, `recipes/expm`'s own `skew_eig_lwork` query,
  when a CASSCF is planned (the step's, or a chain copy's): its orbital step
  diagonalizes κ through `skew_eig` at nmo = nao. 0 otherwise.
- **The CASSCF's roots**: the top-level CASSCF step's `nroots`, and whether
  two of its `weights` differ (`response::rotating_pairs`; none given is
  equal). These are what the CasscfGradient and Nac steps' footprints are
  sized over. They are 0 and false without a CASSCF step.
- **MINAO**, placed and primitive-normalized as AVAS places it, when any
  planned active space is AVAS's, or a `[localization]` builds IAOs over it
  (`mode = "ibo"` or `livvo = true`), and the molecule's **atoms**, which
  AVAS's labels are matched against: what the selection's footprint sizes
  MINAO's overlaps and the labelled count from, and the Localization step's
  its IAOs and LIVVOs. syevd's table then reaches MINAO's AO count, where the
  IAOs orthonormalize and the LIVVOs diagonalize.
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

## Pre-flight

`preflight(calc, node)` is what `main` calls between building the node and
running the plan. Its steps, and what is and is not capped, are
[`pipeline/README.md`](../README.md)'s "Memory caps"; the fold and its log
lines are `check_plan`'s ([`footprint/README.md`](../footprint/README.md)).
What it does that only this module can:

- **It rebuilds the node's devices with the workspaces and the caps.**
  `main` builds the node with no workspaces and uncapped, since neither is
  known before the plan is shaped on a device, and both are set at a
  `DeviceResources`' creation, so each is move-assigned a new one: its
  workspaces at the plan's maximum (`check_plan`'s fold under a cap; the
  most any step borrows without one), the pool capped at
  `max_device_mem_per_gpu` less `DeviceResources::kDeviceReserve`, the
  pinned total at `max_pinned_mem_per_gpu`.
- **Uncapped**, nothing is checked but the device: an AVAS worst case past
  its budget less the headroom is narrowed as under a cap, and a device
  workspace past its budget is refused (`Allocation`) rather than aborting
  in the constructor.
- **The device room.** The fold's device peak P must fit the pool's cap
  less `kPoolHeadroomBytes` and less `DeviceResources::kOwnPoolBytes`, and
  on a pool that rounds a reused block, P + ⌈P/8⌉ must. `kOwnPoolBytes` is
  what the `DeviceResources` holds in its pool from construction, before
  pre-flight and outside every step: its solver `info` flag, one `int32_t`.
  No footprint can count it, so the room does. Whether it does is probed on the pool itself (a 1024-byte
  block freed, then 1000 bytes asked for: ROCm's counts 1024, CUDA's 1000),
  not read off the backend. Below, "The HIP margin".
- **AVAS.** `narrow_avas`: when every planned active space is AVAS's (a
  `[casci]` or `[casscf]` with `avas`, and no `[fci]`) and the worst case
  `plan_shape` took does not fit, the most orbitals that fit, by bisection
  over the fold at each count (nelecas at half filling, as `plan_shape`
  takes it), become each AVAS step's `avas_max_ncas`, and the plan is checked
  at that space. Bisection needs the fold to grow with the space, which
  every active-space footprint does.
- **The pre-warm**, which needs the plan's peak and the pool: the pool's
  cap less the headroom and less what it already holds is allocated and
  freed through the pool. Whether the device can supply the cap is read off
  `memory_budget()`, `min(cap, free + reserved) − used`, so no runtime call
  outside `core/wwr` is made; the refusal derives the free memory from it.

`test/gtest/pipeline_preflight_gpu.cpp` holds it to water's RHF chain, a
cap below the fold's peak refused and one above it run inside the pool's cap
in workspaces held at the plan's maximum; to a pre-warm refused with half the card
held outside the pool; and to a CASCI under AVAS narrowed to the space its
cap fits. The nested `[optimize]` and `[hessian]` cases pin the room to the
byte: a cap whose room is the peak exactly runs, one byte less is refused,
and the pool holds exactly the workspace plus `kOwnPoolBytes`
before the first step. `test/examples/`'s `example_memory` runs
`examples/memory.toml` through the binary.

### The HIP margin

ROCm's pool serves a request whole from a freed block of up to 9/8 its size
and counts the block (`core/wwr/README.md`, "Reuse rounding (HIP)"), so each
allocation live at the peak may be counted up to an eighth past its request.
Their requests sum to at most the peak P, so the rounding is at most P/8,
which is unbounded in P: 64 MiB covers it only for a peak under 512 MiB. And
after the pre-warm every allocation is a reuse, of the warm block. The
rounding is not the chunk gap the headroom covers, but beside it: a block
served whole leaves no hole of its own, and the chunks' free tops and the
small allocations' holes are still there. So the two are added, not
maxed: a pool that rounds leaves the headroom and ⌈P/8⌉. That is the sound
bound; it costs a HIP plan up to an eighth of its peak, which the measured
overcounts (12 and 64 B, #1049) never come near, and a measurement of HIP's
reserved gap at a large peak may later justify less.
