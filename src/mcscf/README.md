# MCSCF

CASSCF on the GPU, integral-direct and state-averaged: the loop that optimizes
the orbitals and the CI together. It sits over `core`, `recipes`, `integrals`
(J, K and `active_space`), `ci` (the loop's CI and RDMs) and `response` (the
energy, gradient and orbital Hessian it steps over), and **never imports
`pipeline`**: a step that runs a CASSCF wraps this layer, not the other way
round. `mcscf.cppm` (`free_breakfast.mcscf`) re-exports `solve`; it does not
re-export `response`, so a consumer that wants both imports both.

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
| `solve/` | `free_breakfast.mcscf.solve` | The two-step state-averaged loop (`mcscf::solve`): per macro iteration the integrals, the CI, the weighted RDMs, `response::fock`'s energy and gradient, and an augmented-Hessian step over `response::Hessian` under a trust radius. Its defaults and what each settles are in `solve/README.md`. |

The orbital order, the packed pairs and the gradient and Hessian conventions
the loop steps in are [`../response/README.md`](../response/README.md)'s.

## Tests

`test/gtest/mcscf_solve_gpu.cpp`, linked into `free_breakfast_tests_gpu`,
against PySCF's converged CASSCF through `test/gtest/data/casscf/`
(`tools/emit_casscf_reference.py`). The test sets its active spaces up with
`pipeline.active_space` and the `Casci` step from the test executable; this
layer's own targets link nothing of `pipeline`.

## Files

| File | Role |
|------|------|
| `mcscf.cppm` | Aggregate module interface; `export import`s `solve` |
| `CMakeLists.txt` | Build configuration: `add_subdirectory(solve)`, and the `free_breakfast.mcscf` module target |
