# Population analysis

Atomic populations and partial charges over an orthonormal, atom-owned
minimal basis — in practice the intrinsic atomic orbitals, after Knizia,
J. Chem. Theory Comput. 9, 4834 (2013), eq 3. Unlike Mulliken's, these
populations barely move with the basis set (the paper's Table 1).

No kernels of its own: two wwr `gemm`s, and host sums over an m x n_orb
matrix.

## Module

`free_breakfast.analysis.population`, target
`free_breakfast::analysis::population`.

| Name | Role |
|---|---|
| `orbital_weights(res, basis, minimal, n_orb, s, c, iao)` | Each orbital's weight on each minimal-basis function, (A^T S C)_ri^2, m x n_orb on the host; `Result` |
| `orbital_populations(minimal, n_orb, weights)` | Those weights summed over each atom's functions, natoms x n_orb; host only; `Result` |
| `orbital_populations(res, basis, minimal, n_orb, s, c, iao)` | The two above in one call; `Result` |
| `partial_charges(atoms, populations, occupation = 2)` | q_A = Z_A − occupation · Σ_i populations[A, i]; `Result` |

Double precision. `s` (n x n), `c` (n x n_orb) and `iao` (n x m) are
column-major device `double *`, packed, over `basis`'s spherical AOs,
n = `basis.nao_sphe()`, m = `minimal.nao_sphe()`. Column r of `iao` belongs
to the atom owning `minimal`'s spherical AO r, which the placed `minimal`
records. With the IAOs of `localization/ibo`:

```cpp
FBF_TRY(intrinsic_atomic_orbitals(res, basis, minao, n_occ, s, c, iao.data()));
const std::vector<double> populations =
    FBF_TRY(orbital_populations(res, basis, minao, n_occ, s, c, iao.data()));
const std::vector<double> charges = FBF_TRY(partial_charges(molecule.atoms(), populations));
```

## The formulas

Over an S-orthonormal set A (A^T S A = I) spanning the orbitals C, orbital i
is written exactly as A (A^T S C)_i, and its population on atom A is

```
n_A(i) = sum_{r on A} (A^T S C)_ri^2
```

which sums to one over the atoms for an S-orthonormal orbital. The squares
before the sum are `orbital_weights`, kept for a caller that wants more than
the per-atom sum: `analysis/bond` splits each atom's share by angular
momentum from them. The partial
charge of eq 3 is q_A = Z_A − 2 Σ_i n_A(i) over the occupied orbitals of a
closed shell; the charges sum to the molecule's.

## Independent of the IBOs, and usable on them

This module takes the IAOs as an argument, so it does not depend on
`localization/ibo` and serves any orthonormal atom-owned set. And it takes
any orbitals:

- **The charges** are a property of the density, Σ_i n_A(i) being the trace
  of the density over atom A's functions, so they are the same for the
  canonical orbitals and for the IBOs, or any other rotation of them.
- **Each orbital's populations** are not: over the IBOs they are the bonding
  picture (a core or lone pair ~1 on one atom, a bond split over two), and
  their fourth powers summed are the IBO functional L.

`pipeline/localization` does both in IBO mode, over the IBOs and the IAOs
`ibo_localize` hands back.

## Tests

`test/gtest/pipeline_localization_gpu.cpp`, against PySCF's IAO charges
(`test/gtest/data/localization/ibo.json`): from the canonical orbitals, each
orbital's populations summing to one, the charges to PySCF's to 1e-7 and
summing to the molecule's charge; through the step, the IBOs' populations
reproducing L, and the same charges. `partial_charges`' arithmetic and its
refusal of a ragged population array are checked on the host.

## Files

| File | Role |
|---|---|
| `population.cppm` | Module interface: `orbital_populations` and `partial_charges` |
| `population.cpp` | A^T S C on the device, the per-atom sums and charges on the host |
| `CMakeLists.txt` | The module |
