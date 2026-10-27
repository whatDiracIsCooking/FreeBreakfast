# Response

The derivatives of a (state-averaged) CASSCF energy with respect to the
orbital rotation κ, at fixed orbitals and fixed RDMs, on the GPU: the energy
and its gradient (`fock/`) and the orbital Hessian (`hessian/`). They are what
an optimizer steps over (`mcscf/solve`), and what a response solve inverts: a
CASSCF nuclear gradient's Z-vector, E⁽²⁾ z = −∂L/∂κ, and CPHF for MP2
gradients, which is the ncas = 0 RHF limit `fock/` already covers. The mature
MCSCF codes split the two the same way: OpenMolcas `RASSCF` / `MCLR`, Dalton
`SIRIUS` / `RSP`, BAGEL `multi/casscf` / `cpcasscf`.

It sits over `core`, `recipes` and `integrals` (J, K and `active_space`),
takes the RDMs `ci.rdm` writes as plain host arrays, and **never imports
`pipeline`** (nor `ci`, nor `mcscf`). `response.cppm`
(`free_breakfast.response`) re-exports both sub-modules, and each stays
importable on its own. `free_breakfast.mcscf` does not re-export this layer:
a consumer that wants both imports both.

Everything it exports is in **`namespace free_breakfast::response`**, spelled
bare inside it: `response::Fock`, `response::fock`, `response::Hessian`,
`response::pack_orbital_pairs`, `response::unpack_orbital_pairs`.
`docs/style.md` "Namespaces, and when a name needs a prefix" has the rule.

**"Response" also names something else nearby.** `mcscf/solve` calls the
CI's reaction to each Hessian column, R, the CI's "response". That is part of
the optimizer's coupled step and stays in `solve`; it is not this layer.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `fock/` | `free_breakfast.response.fock` | At fixed orbitals and RDMs: the energy, the inactive and active Fock matrices F^I and F^A, the generalized Fock F and the orbital gradient (`response::fock`); and the packed order of the non-redundant orbital pairs (`pack_orbital_pairs`, `unpack_orbital_pairs`). |
| `hessian/` | `free_breakfast.response.hessian` | At the same point: the exact orbital Hessian at fixed CI as a `LinearOperator` over the packed pairs, and its diagonal (`response::Hessian`). |

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
`response::Fock::gradient`, of the orbital Hessian's products and diagonal, and of
`mcscf::solve`'s step: everything over the pairs shares it. It is PySCF's
`uniq_var_indices` mask read in column-major order, where PySCF's own packing
reads it row-major; a test comparing with PySCF matches pairs by (p, q), not by
position.

The gradient is the first derivative in that coordinate, at fixed CI:

    E(C exp(κ(x))) = E + g · x + O(x²),   g_pq = 2 (F_pq − F_qp),

with F the generalized Fock, and the Hessian (`hessian/`) the second
derivative in the same coordinate, E = E + g · x + ½ xᵀ H x + O(x³): both are
twice PySCF's `gen_g_hop` g, `h_op` and `h_diag`. `unpack_orbital_pairs`
writes the full κ that `recipes.expm` exponentiates.

## Not done

- **Active–active rotations.** The packed pairs hard-code CAS's three blocks;
  RAS and GAS need rotations between active subspaces too.
- **The CI blocks of E⁽²⁾**: orbital–CI coupling over transition RDMs, and
  CI–CI (H − E, projected). The SA-CASSCF Z-vector needs them, and they would
  make this layer import `ci`.
- **The Z-vector solve** itself, with CASSCF nuclear gradients.
- **E⁽³⁾ contractions**, E⁽³⁾(a, b) as a vector over κ and c, for NEVPT2
  analytic Hessians.

## Tests

`test/gtest/response_fock_gpu.cpp` and `test/gtest/response_hessian_gpu.cpp`
(suites `ResponseFock*` and `ResponseHessian*`), linked into
`free_breakfast_tests_gpu`, against PySCF's `mc1step.gen_g_hop` through
`test/gtest/data/casscf/`. They set their active spaces up with
`pipeline.active_space` and the `Casci` step from the test executable
(`test/gtest/mcscf_setup.h`, shared with `mcscf_solve_gpu.cpp`); this layer's
own targets link nothing of `pipeline`.

## Files

| File | Role |
|------|------|
| `response.cppm` | Aggregate module interface; `export import`s both sub-modules |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.response` module target |
