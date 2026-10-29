# Intrinsic bond orbitals

Intrinsic atomic orbitals (IAOs) and intrinsic bond orbitals (IBOs) on the GPU,
after Knizia, J. Chem. Theory Comput. 9, 4834 (2013). The IAOs are a minimal
basis, one function per free-atom AO, polarized by the molecule just enough to
span its occupied orbitals exactly. The IBOs are the occupied orbitals
localized over the IAOs' atomic populations: Pipek-Mezey with chemically
sound populations in place of Mulliken's.

No kernels of its own: `recipes/blas`'s `gemm`, `recipes/cholesky`, wwr's
`potrs`, `recipes/orthogonalize`, `integrals/one_electron/overlap_mixed`, and
`localization/pipek_mezey`'s sweep.

## Module

`free_breakfast.localization.ibo`, target `free_breakfast::localization::ibo`.
It re-exports `free_breakfast.localization.pipek_mezey`, whose options and info
it takes and returns.

| Name | Role |
|---|---|
| `intrinsic_atomic_orbitals(res, basis, minimal, n_occ, s, c, iao)` | The orthonormal IAOs, n_ao x n_minimal; `Status` |
| `localization::ibo_localize(res, basis, minimal, n_occ, s, c, c_local, u, iao, options)` | The IBOs, and the IAOs if `iao` is not null; returns `Result<localization::Info>`, exponent 4 by default |
| `localization::iao_localize(res, basis, minimal, n_orb, s, iao, c, c_local, u, options)` | The same sweep over IAOs the caller built, for orbitals in their span: what `ibo_localize` runs after building them |

Double precision only. Every matrix is a column-major device `double *`,
packed, over the spherical AOs: `s` (n x n) and `c` (n x n_occ) as for
`localization::pm_localize`, `iao` n x m with m = `minimal.nao_sphe()`.
`minimal` is the free-atom minimal basis placed over the same atoms, which
for the paper's IAOs is MINAO (`data/basis/minao.json`):

```cpp
const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
const BasisSet minao = FBF_TRY(place(stamp.primitive_normalized(), molecule));
const localization::Info info = FBF_TRY(localization::ibo_localize(res, basis, minao, n_occ, overlap.s.data(),
                                                 scf.c.data(), c_local.data(), u.data()));
```

The module takes the minimal basis rather than resolving MINAO itself, so it
needs no `io` and any minimal basis serves. `pipeline/localization` resolves
MINAO.

## The IAOs (Appendix C)

With S1 the basis's overlap, S2 the minimal basis's, S12 the one between them,
and C the occupied orbitals:

```
P12 = S1^-1 S12
C~  = orth(S1^-1 S12 S2^-1 S12^T C)
A   = C C^T S1 C~ C~^T S12 + (1 - C C^T S1)(1 - C~ C~^T S1) P12
IAO = orth(A),   orth(X) = X (X^T S1 X)^-1/2
```

C~ is the occupied space depolarized: projected onto the free-atom AOs and
back. A splits each free-atom AO by it, the occupied part polarized as the
molecule's orbitals are and the rest kept orthogonal to them, which is why the
IAOs span the occupied orbitals.

**No n x n product is formed.** S1 P12 = S12 exactly, so expanding A leaves
n_occ-wide products only:

```
A = P12 - C E - C~ F + 2 C O F,   E = C^T S12,  F = C~^T S12,  O = C^T S1 C~
```

**S^-1 is a Cholesky solve**, never an inverse, as the paper advises:
`cholesky` once per overlap (not the solver's `potrf`, whose small-matrix
kernel races on ROCm 7.2: `recipes/cholesky/README.md`), `potrs` per
right-hand side. S2, a free-atom minimal basis,
failing to be positive definite is `Numerical`.

**A near-singular S1 falls back to S^-1/2.** A basis read from a path can be
near-linearly dependent (diffuse sets, near-duplicate exponents), and a
Cholesky solve over S1 then amplifies rounding by the inverse of its smallest
pivot. So S1's factor is used only if `cholesky` succeeds and every pivot L_jj^2
clears `kOrthogonalizeEigenvalueFloor` (1e-10). Every pivot bounds S1's
smallest eigenvalue from above, so a pivot under the floor means a mode
`orthogonalize` would drop. The pivot test is what does the work: on the
test's exactly singular S1, the factorization can succeed with a pivot at
rounding (`potrf`'s was ~3e-18), so a
fallback keyed on its failure alone could not be relied on. Otherwise S1^-1 B is
applied as X (X B), with
X = S1^-1/2 from `recipes/orthogonalize`: the floored modes drop, so X X^T is
the pseudo-inverse on the rest. That is the paper's "twice multiplying with
S^-1/2". It logs at warn, and `orthogonalize` logs cond(S1) alongside. The
pivot test is one-sided: an S1 whose pivots all clear the floor can still have
an eigenvalue under it, and such an S1 keeps the Cholesky path.

PySCF falls back only when `cho_factor` fails, to canonical orthogonalization
at 1e-8. On a set both take the fallback for, the two agree where they drop
the same modes. X is built here rather than taken from the pipeline's
`OrthogonalizerOutput`, so the module needs no pipeline type. The cost is one
extra `syevd`, paid only on this path.

**The two orth()s are Loewdin**, `recipes/orthogonalize` on the small metric
(n_occ x n_occ, then m x m) and one gemm. The paper's procedure is invariant to
the kind of orthogonalization for C~, but not for A: a symmetric
orthogonalization depends on the columns' scale. So **the minimal basis's
functions are normalized** first (S12 <- S12 D and S2 <- D S2 D, with
D = diag(S2)^-1/2, three `dgmm`s) — this tree does not force a contraction to
unit norm, and PySCF's normalized MINAO is what the oracle builds its IAOs
from.

**The overlaps** S12 and S2 are `overlap_mixed_spherical`'s
(`integrals/one_electron/overlap_mixed`).

## The IBOs (Appendix D)

The occupied orbitals over the orthonormal IAOs are C_IAO = IAO^T S1 C,
lossless because the IAOs span them. Over an orthonormal basis S is the
identity, so `localization::pm_localize(res, minimal, n_occ, I, C_IAO, ...)`
computes exactly the IAO populations, each IAO owned by its minimal-basis AO's
atom (which the placed `minimal` records). Its p = 4 angle is the paper's
Appendix D one, so this is Knizia's IBO functional and iteration, run on the
existing sweep.

The localized orbitals are written back as C U, not IAO C_IAO U: the same
orbitals, without a round trip through the IAOs' rounding.

**`iao_localize` is that sweep alone**, over IAOs the caller passes, and
`ibo_localize` is `intrinsic_atomic_orbitals` then `iao_localize`. It is
for a subspace of the occupied orbitals the IAOs were built from: an open
shell's doubly and singly occupied orbitals are each localized among
themselves (`pipeline/localization`), over the IAOs of both together, which
span each. IAOs built from one space alone would not span the other, and
would be a different set for each.

## Charges and populations

The IAO partial charges (eq 3) and each orbital's IAO population per atom are
not here: they are `analysis/population`, which takes the IAOs as an argument
and so does not depend on this module. `localization::ibo_localize`'s optional `iao` hands
back the IAOs it ran over, so a caller wanting both builds them once.

## Cost

The integrals and the n^2 m-sized gemms are the cost; everything else is
n_occ wide. Scratch is pool buffers, since the solvers, `orthogonalize` and
the sweep take `res.device_workspace()` in turn. The host synchronizes at each
Cholesky flag and each orthogonalization.

## Tests

`test/gtest/pipeline_localization_gpu.cpp`, against PySCF's `lo.iao` and
`lo.ibo.PipekMezey` (`tools/emit/localization_reference.py`, writing
`test/gtest/data/localization/ibo.json`), for water in STO-3G and cc-pVDZ and
NH4+ in cc-pVDZ:

- through the step (`mode = "ibo"`), both exponents: L over the IAO
  populations to 1e-8, every bond's Mulliken population on every atom matched
  to a PySCF orbital to 1e-5, and the IBOs S-orthonormal and reproducing the
  SCF's density. Orbitals wholly on one atom (water's O 1s and lone pairs) are
  compared as a set, by their summed populations: rotating among them moves L
  by ~1e-13 and their populations by ~1e-3, and PySCF's own two IBO optimizers
  (`lo.ibo.PipekMezey` and Knizia's `ibo_loc`) disagree by that much, so only
  the set is determined;
- the IAOs alone, from the ScfOutput's orbitals: A^T S A = I, each occupied orbital
  of unit weight on them, and the IAO partial charges (eq 3) matching PySCF's
  to 1e-7;
- the same, over cc-pVDZ read from a path with oxygen's outer s function
  given twice: the same space over an exactly singular S1, so the fallback
  runs and the charges are still the plain cc-pVDZ ones.

## Files

| File | Role |
|---|---|
| `ibo.cppm` | Module interface: `intrinsic_atomic_orbitals`, `localization::ibo_localize` and `localization::iao_localize` |
| `ibo.cpp` | The overlaps, the IAO construction and the IBO driver |
| `CMakeLists.txt` | The module |
