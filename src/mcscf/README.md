# MCSCF

CASSCF on the GPU, integral-direct and state-averaged. It sits over `core`,
`recipes`, `integrals` (J, K and `active_space`) and `ci` (the loop's CI and
RDMs), takes the RDMs `ci.rdm` writes as plain host arrays, and **never imports `pipeline`**: a step that runs
a CASSCF wraps this layer, not the other way round. `mcscf.cppm`
(`free_breakfast.mcscf`) re-exports every sub-module, so one import covers the
layer, and each sub-module stays importable on its own.

Everything it exports is in **`namespace free_breakfast::mcscf`**, spelled bare
inside it — `mcscf::Fock`, `mcscf::Hessian`, `mcscf::Options`, `mcscf::solve` —
and `ah_precondition`, the one launcher its bridge declares, is in
**`free_breakfast::mcscf::device`**. The namespace is the layer, so the
`Casscf` prefix every one of these used to carry is simply gone: this layer is
CASSCF, and the namespace says so once instead of each name saying it again.
`CasscfIntegrals` and `casscf_integrals` are the exception and stay as they
are — they belong to CASSCF but live in `integrals/active_space/`, so neither
namespace is their home. `docs/style.md` "Namespaces, and when a name needs a
prefix" has the rule.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `fock/` | `free_breakfast.mcscf.fock` | At fixed orbitals and RDMs: the energy, the inactive and active Fock matrices F^I and F^A, the generalized Fock F and the orbital gradient (`mcscf::fock`); and the packed order of the non-redundant orbital pairs (`pack_orbital_pairs`, `unpack_orbital_pairs`). |
| `hessian/` | `free_breakfast.mcscf.hessian` | At the same point: the exact orbital Hessian at fixed CI as a `LinearOperator` over the packed pairs, and its diagonal (`mcscf::Hessian`). |
| `solve/` | `free_breakfast.mcscf.solve` | The two-step state-averaged loop over both (`mcscf::solve`): per macro iteration the integrals, the CI, the weighted RDMs, the energy and gradient, and an augmented-Hessian step under a trust radius. Its defaults and what each settles are in `solve/README.md`. |

## Orbitals and the packed order

C's columns are the core [0, ncore), the active space [ncore, nocc),
nocc = ncore + ncas, then the virtuals, as `integrals.active_space` takes them.
A rotation is C exp(κ) with κ real antisymmetric, and only the pairs that
change the energy are kept: **core–active, core–virtual and active–virtual**.
Rotations within one space are redundant for a CASSCF and are not carried.

The non-redundant pairs are x_pq, **p > q**, with κ_pq = x_pq and
κ_qp = −x_pq, packed **column by column of κ's lower triangle**: q outer,
p ascending within a column.

| Packed | Pairs | Count |
|---|---|---|
| [0, ncore · (nmo − ncore)) | q core, p in [ncore, nmo): rows [ncore, nmo) of columns [0, ncore) | ncore (nmo − ncore) |
| then | q active, p in [nocc, nmo): rows [nocc, nmo) of columns [ncore, nocc) | ncas (nmo − nocc) |

So x_pq for q core is at q (nmo − ncore) + (p − ncore), and for q active at
ncore (nmo − ncore) + (q − ncore)(nmo − nocc) + (p − nocc). The packed vector
is two column-major blocks of an nmo × nmo column-major matrix, so a pack or an
unpack is two 2D copies, with no gather kernel. This is the order of
`mcscf::Fock::gradient`, of the orbital Hessian's products and diagonal, and of
the optimizer's step: everything over the pairs shares it. It is PySCF's
`uniq_var_indices` mask read in column-major order, where PySCF's own packing
reads it row-major; a test comparing with PySCF matches pairs by (p, q), not by
position.

The gradient is the first derivative in that coordinate, at fixed CI:

    E(C exp(κ(x))) = E + g · x + O(x²),   g_pq = 2 (F_pq − F_qp),

with F the generalized Fock, and the Hessian (`hessian/`) the second
derivative in the same coordinate, E = E + g · x + ½ xᵀ H x + O(x³): both are
twice PySCF's `gen_g_hop` g, `h_op` and `h_diag`. `unpack_orbital_pairs`
writes the full κ that `recipes.expm` exponentiates.

## Tests

`test/gtest/mcscf_fock_gpu.cpp` and `test/gtest/mcscf_hessian_gpu.cpp`,
linked into `free_breakfast_tests_gpu`, against PySCF's `mc1step.gen_g_hop` through `test/gtest/data/casscf/`, and `test/gtest/mcscf_solve_gpu.cpp` against its converged CASSCF
(`tools/emit_casscf_reference.py`). The test sets its active spaces up with
`pipeline.active_space` and the `Casci` step from the test executable; this
layer's own targets link nothing of `pipeline`.

## Files

| File | Role |
|------|------|
| `mcscf.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.mcscf` module target |
