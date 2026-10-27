# Valence virtual orbitals and LIVVOs

The valence virtual orbitals (VVOs) on the GPU, after Derricotte & Evangelista,
J. Chem. Theory Comput. 13, 5984 (2017): the part of the intrinsic atomic
orbitals' span orthogonal to the occupied orbitals. The IAOs are a minimal
basis that spans the occupied space exactly, so what is left of them is
m - n_occ virtual orbitals of valence character, the antibonding partners of
the bonds. `localization::livvo_localize` localizes them over the IAO populations (the
paper's LIVVOs), and `valence_character` measures how much of any orbital lies
in their span.

No kernels of its own: `recipes/blas`'s `gemm`, `recipes/diagonalize` and
`localization::pm_localize`.

## Module

`free_breakfast.localization.livvo`, target
`free_breakfast::localization::livvo`.

| Name | Role |
|---|---|
| `valence_virtual_orbitals(res, basis, minimal, n_occ, s, c, iao, vvo)` | The n x (m - n_occ) VVOs; `Status` |
| `localization::livvo_localize(res, basis, minimal, n_occ, s, iao, vvo, livvo, u = nullptr, options = {.exponent = Four})` | The n x n_vvo LIVVOs and the n_vvo x n_vvo rotation; `Result<localization::Info>` |
| `valence_character(res, basis, n_orb, n_vvo, s, c, livvo)` | t_val of each of n_orb orbitals; `Result<std::vector<double>>` |

The module re-exports `free_breakfast.localization.pipek_mezey`, for
`localization::PmOptions` and `localization::Info`.

Double precision only. Every matrix is a column-major device `double *`,
packed, over the spherical AOs: `s` (n x n), `c` (n x n_occ, S-orthonormal),
`iao` (n x m), `vvo` and `livvo` (n x n_vvo, n_vvo = m - n_occ), n =
`basis.nao_sphe()` and m = `minimal.nao_sphe()`.

**The IAOs are an argument.** They come from `intrinsic_atomic_orbitals`, or
from `localization::ibo_localize`'s `iao` out-parameter, so a caller wanting IBOs and VVOs
builds them once. That also keeps this module off `ibo` and the integrals:
`basis` and `minimal` are read only for their sizes and atom counts.

```cpp
Buffers::device<double> iao(n * m, res.device());
Buffers::device<double> vvo(n * (m - n_occ), res.device());
FBF_TRY(intrinsic_atomic_orbitals(res, basis, minao, n_occ, s, c, iao.data()));
FBF_TRY(valence_virtual_orbitals(res, basis, minao, n_occ, s, c, iao.data(), vvo.data()));
Buffers::device<double> livvo(n * (m - n_occ), res.device());
FBF_TRY(localization::livvo_localize(res, basis, minao, n_occ, s, iao.data(), vvo.data(), livvo.data()));
std::vector<double> t_val = FBF_TRY(valence_character(res, basis, n - n_occ, m - n_occ, s,
                                                      c + n * n_occ, livvo.data()));
```

Every function synchronizes the stream before it returns, as `ibo` and
`analysis/population` do, so its outputs are ready to read from any stream.
Its pool buffers need no sync of their own: they are freed with
`wwrFreeAsync` on the same stream, after the work that reads them.

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

### LIVVOs

As `localization::ibo_localize` does for the occupied orbitals: the VVOs written over the
orthonormal IAOs,

```
V_iao = A^T S V          (m x n_vvo, lossless: V = A V_iao)
```

are localized by `localization::pm_localize` over the identity overlap, with
`minimal` saying which atom owns each IAO, so its Mulliken populations are the
IAO ones. Its rotation U gives the LIVVOs as V U, without the round trip
through A. The default exponent is Knizia's four. `localization::pm_localize` is
called directly; there is no IBO entry point for virtuals.

### Valence character

For an orbital psi, t_val = sum_l |<psi|phi_l>|^2 over the LIVVOs phi_l (the
paper's eqs 11-12): the squared column norms of L^T S C. It depends only on
the LIVVOs' span, so the VVOs give the same numbers. For S-orthonormal C each
t_val is in [0, 1], and over all n - n_occ canonical virtuals they sum to
n_vvo, since the VVOs lie wholly in the virtual space.

## Measured: water's sigma*_OH

Water in cc-pVDZ over MINAO (O at z = 0.1173 A, H at y = +-0.7572 A,
z = -0.4692 A), RHF from the pipeline: the two LIVVOs are mirror images, each
on O and one H, by `orbital_populations` over the IAOs:

| | O | its H | the other H |
|---|---|---|---|
| each LIVVO | 0.3245 | 0.6755 | 8.3e-07 |

So the antibonding partner of the O-polarized O-H bond leans toward H, about
two to one. The test checks the two centres and the mirror symmetry, not
which atom carries more.

## Cost

Four gemms (S C, A^T (S C), C_iao C_iao^T and A U_0), none wider than n x n
by n_occ or n x m by m, and one m x m `syevd`. S C and C_iao are carved
from the workspace with `res.device_regions`; P and the eigenvalues are a pool
buffer, since `diagonalize` takes the workspace while they are live. `diagonalize` synchronizes the stream once, for its flag, and
the function once more before freeing P.

`localization::livvo_localize`: two gemms for V_iao (S V from the workspace), the sweep, one
n x n_vvo by n_vvo gemm for V U. V_iao, its localized copy, U and the m x m
identity are pool buffers, held across the sweep, which takes the workspace.
One sync after uploading the identity, the sweep's one per sweep, one before
returning.

`valence_character`: two gemms (S C and L^T S C) from the workspace, one
download and one sync; the sums are on the host.

## Tests

`test/gtest/livvo_gpu.cpp`, water and N2 in cc-pVDZ over MINAO, from a
pipeline SCF:

- m - n_occ VVOs (2 for water, 3 for N2);
- V^T S V = I, C_occ^T S V = 0, and ||C_vir^T S V||_F^2 = n_vvo against the
  canonical virtuals, so V lies wholly in the virtual space;
- the span, as V V^T, unchanged when the occupied orbitals are rotated among
  themselves before the IAOs are built;
- InvalidConfig for n_occ outside 0..m, a minimal basis over a different atom
  count, and each null pointer;
- water's LIVVOs: two, L^T S L = I and C_occ^T S L = 0, each with more than
  0.999 of its IAO population on O and one H, a different H each, and the two
  populations equal to 1e-8 (mirror images);
- t_val over water's canonical virtuals: each in [0, 1], summing to n_vvo,
  and the same from the VVOs as from the LIVVOs;
- InvalidConfig from `localization::livvo_localize` for n_occ outside 0..m, a mismatched
  minimal basis, each null pointer and `livvo` aliasing `vvo`, and from
  `valence_character` for a negative `n_orb` or `n_vvo` and each null pointer.

## Files

| File | Role |
|---|---|
| `livvo.cppm` | Module interface: `valence_virtual_orbitals`, `localization::livvo_localize`, `valence_character` |
| `livvo.cpp` | The null-space construction, the localization over the IAOs, t_val |
| `CMakeLists.txt` | The module |
