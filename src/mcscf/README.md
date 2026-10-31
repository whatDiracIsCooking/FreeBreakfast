# MCSCF

CASSCF on the GPU, integral-direct (or with its orbital Hessian
density-fitted) and state-averaged: the loop that optimizes the orbitals and
the CI together. It sits over `core`, `recipes`, `integrals`
(J, K and `active_space`), `ci` (the loop's CI and RDMs) and `response` (the
energy, gradient and orbital Hessian it steps over), and **never imports
`pipeline`**: a step that runs a CASSCF wraps this layer, not the other way
round. `mcscf.cppm` (`free_breakfast.mcscf`) re-exports `solve`; it does not
re-export `response`, so a consumer that wants both imports both. Nor does it
re-export `relaxed_hessian`, which is internal to the layer: the CI-relaxed
Hessian product its sub-modules share, linked PRIVATE by each.

Everything it exports is in **`namespace free_breakfast::mcscf`**, spelled bare
inside it — `mcscf::Options`, `mcscf::solve` — and `ah_precondition`, the one
launcher its bridge declares, is in **`free_breakfast::mcscf::device`**. The
namespace is the layer, so the `Casscf` prefix these used to carry is simply
gone. `CasscfIntegrals` and `casscf_integrals` are the exception and stay as
they are — they belong to CASSCF but live in `integrals/active_space/`, so
neither namespace is their home. `docs/style.md` "Namespaces, and when a name
needs a prefix" has the rule.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `relaxed_hessian/` | `free_breakfast.mcscf.relaxed_hessian` | **Internal**, not re-exported: the orbital Hessian with the CI relaxed, H̃ = H + R (`mcscf::RelaxedHessian`), and the orbital–CI coupling (`ResponseIntegrals`, `response_integrals`, `response_gradient`). `solve`'s augmented Hessian steps over it. See `relaxed_hessian/README.md`. |
| `solve/` | `free_breakfast.mcscf.solve` | The two-step state-averaged loop (`mcscf::solve`): per macro iteration the integrals, the CI, the weighted RDMs, `response::fock`'s energy and gradient, and an augmented-Hessian step over `response::Hessian` under a trust radius. Its defaults and what each settles are in `solve/README.md`. |
| `gradient/` | `free_breakfast.mcscf.gradient` | The analytic nuclear gradient of a converged DF-CASSCF (`mcscf::df_gradient`), state-specific or any root of a state average: the one-electron terms at its 1-RDM and generalized Fock, and one Γ and one W of the fitted two-electron energy against the three-center and metric derivatives, each term on its own. For a state average, over `zvector`'s multipliers (linked PRIVATE). Its derivation is in `gradient/README.md`. |
| `derivative/` | `free_breakfast.mcscf.derivative` | The DF-CASSCF Hessian's per-coordinate derivatives at fixed parameters (`mcscf::df_derivatives`): the derivative generalized Fock F^[x] and the CP-CASSCF right-hand sides (orbital, CI and the roots' rotations), every coordinate at once, at the RDMs of any weighting of any roots, so SS, a state average's averaged right-hand side and one root's own. Layouts and signs in `derivative/README.md`. |
| `zvector/` | `free_breakfast.mcscf.zvector` | **Internal**, not re-exported: the Lagrange multipliers of one root's state-averaged gradient (`mcscf::zvector`), z_o by preconditioned conjugate gradients over `RelaxedHessian` and z_c from its CI response; and the same equations on many right-hand sides at once (`mcscf::zvectors`, the CP-CASSCF solve of a Hessian), by one of three `CpSolver`s. See `zvector/README.md`. |

The orbital order, the packed pairs and the gradient and Hessian conventions
the loop steps in are [`../response/README.md`](../response/README.md)'s.

## Tests

`test/gtest/mcscf_solve_gpu.cpp`, linked into `free_breakfast_tests_gpu`,
against PySCF's converged CASSCF through `test/gtest/data/casscf/`
(`tools/emit/casscf_reference.py`). The test sets its active spaces up with
`pipeline.active_space` and the `Casci` step from the test executable; this
layer's own targets link nothing of `pipeline`.

`test/gtest/mcscf_gradient_gpu.cpp` holds `gradient/`'s against PySCF's
`df.grad.casscf` and `df.grad.sacasscf` (`test/gtest/data/casscf_gradient/`),
from the same setup.

`test/gtest/mcscf_derivative_gpu.cpp` holds `derivative/`'s against
`test/gtest/data/casscf_hessian/`, its orbitals and roots loaded from the
fixture.

## Files

| File | Role |
|------|------|
| `mcscf.cppm` | Aggregate module interface; `export import`s `solve`, `gradient` and `derivative` |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` of `relaxed_hessian`, `solve`, `zvector`, `gradient` and `derivative`, in that order, and the `free_breakfast.mcscf` module target |
