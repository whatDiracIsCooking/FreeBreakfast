# Valence virtual orbitals

The valence virtual orbitals (VVOs) on the GPU, after Derricotte & Evangelista,
J. Chem. Theory Comput. 13, 5984 (2017): the part of the intrinsic atomic
orbitals' span orthogonal to the occupied orbitals. The IAOs are a minimal
basis that spans the occupied space exactly, so what is left of them is
m - n_occ virtual orbitals of valence character, the antibonding partners of
the bonds. Localizing them (the paper's LIVVOs) is a later step; this module
only builds the space.

No kernels of its own: four wwr `gemm`s and `recipes/diagonalize`.

## Module

`free_breakfast.localization.livvo`, target
`free_breakfast::localization::livvo`.

| Name | Role |
|---|---|
| `valence_virtual_orbitals(res, basis, minimal, n_occ, s, c, iao, vvo)` | The n x (m - n_occ) VVOs; `Status` |

Double precision only. Every matrix is a column-major device `double *`,
packed, over the spherical AOs: `s` (n x n), `c` (n x n_occ, S-orthonormal),
`iao` (n x m) and `vvo` (n x (m - n_occ)), n = `basis.nao_sphe()` and
m = `minimal.nao_sphe()`.

**The IAOs are an argument.** They come from `intrinsic_atomic_orbitals`, or
from `ibo_localize`'s `iao` out-parameter, so a caller wanting IBOs and VVOs
builds them once. That also keeps this module off `ibo` and the integrals:
`basis` and `minimal` are read only for their sizes and atom counts.

```cpp
Buffers::device<double> iao(n * m, res.device());
Buffers::device<double> vvo(n * (m - n_occ), res.device());
FBF_TRY(intrinsic_atomic_orbitals(res, basis, minao, n_occ, s, c, iao.data()));
FBF_TRY(valence_virtual_orbitals(res, basis, minao, n_occ, s, c, iao.data(), vvo.data()));
```

## The formulas

With A the orthonormal IAOs (A^T S A = I) and C the occupied orbitals, the
occupied orbitals over the IAOs are

```
C_iao = A^T S C          (m x n_occ, orthonormal columns: C = A C_iao)
```

since the IAOs span them. The VVOs are A times an orthonormal basis U_0 of the
complement of C_iao's columns in R^m:

```
P   = C_iao C_iao^T      (m x m, eigenvalues: n_occ ones, m - n_occ zeros)
P U = U diag(w),  w ascending
V   = A U_0,      U_0 = U's first m - n_occ columns (w = 0)
```

V^T S V = U_0^T A^T S A U_0 = I, and C^T S V = C_iao^T U_0 = 0. This is the
paper's eqs 3-5 in null-space form: its SVD of the virtual/IAO overlap picks
out the same space, and no SVD is needed here because the IAOs are
orthonormal and already contain the occupied orbitals. Taking P's zero
eigenvectors is the same as taking I - P's eigenvalue-one ones, without
forming I. The eigenvalues are 0 and 1 to rounding, so `syevd` separates the
two blocks cleanly; within a block it returns some orthonormal basis, so V is
fixed only up to a rotation among its own columns. The span is unchanged by a
rotation of the occupied orbitals, since the IAOs and the occupied space are.

n_occ = 0 copies the IAOs, and n_occ = m writes nothing.

## Cost

Four gemms (S C, A^T (S C), C_iao C_iao^T and A U_0), none wider than n x n
by n_occ or n x m by m, and one m x m `syevd`. S C and C_iao are carved from `res.device_workspace()`; P and the
eigenvalues are a pool buffer, since `diagonalize` takes the workspace while
they are live. `diagonalize` synchronizes the stream once, for its flag.

## Tests

`test/gtest/livvo_gpu.cpp`, water and N2 in cc-pVDZ over MINAO, from a
pipeline SCF:

- m - n_occ VVOs (2 for water, 3 for N2);
- V^T S V = I, C_occ^T S V = 0, and ||C_vir^T S V||_F^2 = n_vvo against the
  canonical virtuals, so V lies wholly in the virtual space;
- the span, as V V^T, unchanged when the occupied orbitals are rotated among
  themselves before the IAOs are built;
- InvalidConfig for n_occ outside 0..m, a minimal basis over a different atom
  count, and each null pointer.

## Files

| File | Role |
|---|---|
| `livvo.cppm` | Module interface: `valence_virtual_orbitals` |
| `livvo.cpp` | The null-space construction |
| `CMakeLists.txt` | The module |
