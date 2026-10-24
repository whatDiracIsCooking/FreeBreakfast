# CI

Determinant configuration interaction over plain arrays. CASCI, CASSCF and FCI
all call it, so it never sees a basis, a molecule or an `Scf`: what it takes is
a number of orbitals, electron counts and, in the modules to come, integrals as
arrays. It depends only on `core` (and `recipes`, once there is a solver).
`ci.cppm` (`free_breakfast.ci`) re-exports every sub-module, so one import
covers the layer, and each sub-module stays importable on its own.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `strings/` | `free_breakfast.ci.strings` | The occupation strings of one spin (`StringSet`): enumeration, lexical addressing onto [0, C(n, N)), and the single-replacement lists for E_pq. Host-side. |

A determinant is a pair of strings, alpha and beta, each from its own
`StringSet`. The strings are in PySCF's `cistring` order, so a CI vector laid
out alpha-major over two sets matches `fci.direct_spin1`'s element for element.

## Tests

Host, in `test/gtest/ci_<name>_host.cpp`, linked into `free_breakfast_tests_host`.

## Files

| File | Role |
|------|------|
| `ci.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.ci` module target |
