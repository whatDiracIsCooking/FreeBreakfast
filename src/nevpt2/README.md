# NEVPT2

Strongly and partially contracted NEVPT2 on the GPU, ported from
[nevpt2-gpu-demo](https://github.com/whatDiracIsCooking/nevpt2-gpu-demo). It
sits over `core`, `integrals` (the density-fitted MO transform), `ci`
(the reference state's strings) and, for the nuclear gradient alone,
`response` and `mcscf` (the CASSCF's Fock, Hessian, Z-vector and DF
gradient), and **never imports `pipeline`**: a step that runs an NEVPT2 wraps this layer, not the other way
round. `nevpt2.cppm` (`free_breakfast.nevpt2`) re-exports every sub-module, so
one import covers the layer, and each sub-module stays importable on its own.

Everything it exports is in **`namespace free_breakfast::nevpt2`**, spelled
bare inside it, and the `.cu` launchers its bridge headers declare are in
**`free_breakfast::nevpt2::device`**. `docs/style.md` "Namespaces, and when a
name needs a prefix" has the rule.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `density_fitting/` | `free_breakfast.nevpt2.density_fitting` | The MO blocks of the fitted B (B_aa, B_ca, B_va, B_cv, B_vc), the active h2e, and the ten external blocks built slab by slab: one strided-batched gemm and a permute each. |
| `rdm/` | `free_breakfast.nevpt2.rdm` | `nevpt2::rdms`: dm3 and the integral-contracted 4-RDMs f3ca and f3ac of one CI vector, n_act⁶ each, on the device, tiled over the determinants against a byte budget. |
| `energy/` | `free_breakfast.nevpt2.energy` | The eight SC class norms and energies, each class walking its external index in batches over the density-fitted slabs: the demo's einsums, subscripts unchanged, and PySCF's `_norm_to_energy`. And the eight PC class energies over the same metrics and Dyall Hamiltonians, each class solved once. |
| `gradient/` | `free_breakfast.nevpt2.gradient` | `nevpt2::sc_energy` and `nevpt2::sc_adjoint`: one root's SC correlation energy built from (h, B, c), every class input with it (the orbital energies from (h, B, γ), not re-canonicalized), and its adjoint h̄, Λ and y, with the orbital source term X and A = X − Xᵀ as an AO half-transform. `nevpt2::zc_multipliers` and `nevpt2::zc_adjoint`: the inactive-rotation multipliers z^c from any such A and ε, one division per core–core and virtual–virtual pair (a degenerate one zero or refused), and the term ⟨z^c, f⟩ they add with its adjoints h̄, Λ, y^c and A^{z^c}. `nevpt2::sc_gradient` and `nevpt2::lagrangian_gradient`: the SC-NEVPT2 nuclear gradient of a single-state DF-CASSCF, the Z-vector over the CI-relaxed Hessian, the gate that the Lagrangian's orbital source term is symmetric in every block, and one `mcscf::df_gradient` pass over the multipliers and the correlation's MO adjoint. |

## Tests

`test/gtest/nevpt2_<name>_gpu.cpp`, linked into `free_breakfast_tests_gpu`,
against PySCF through `test/gtest/data/nevpt2/` (`tools/emit/nevpt2_golden.py`).

## Files

| File | Role |
|------|------|
| `nevpt2.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.nevpt2` module target |
