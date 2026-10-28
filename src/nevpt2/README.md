# NEVPT2

Strongly contracted NEVPT2 on the GPU, ported from
[nevpt2-gpu-demo](https://github.com/whatDiracIsCooking/nevpt2-gpu-demo). It
sits over `core` and `integrals` and **never imports `pipeline`**.
`nevpt2.cppm` (`free_breakfast.nevpt2`) re-exports the sub-modules.

Everything it exports is in **`namespace free_breakfast::nevpt2`**, spelled
bare inside it, and its launchers are in **`free_breakfast::nevpt2::device`**
(`docs/style.md` "Namespaces, and when a name needs a prefix").

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `density_fitting/` | `free_breakfast.nevpt2.density_fitting` | The MO blocks of the fitted B (B_aa, B_ca, B_va, B_cv, B_vc), the active h2e, and the ten external blocks built slab by slab: one strided-batched gemm and a permute each. |

## Tests

`test/gtest/nevpt2_*_gpu.cpp`, linked into `free_breakfast_tests_gpu`, against
the demo's density-fitted goldens in `test/gtest/data/nevpt2/`
(`tools/emit/nevpt2_golden.py`).
