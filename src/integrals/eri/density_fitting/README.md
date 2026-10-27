# Density fitting

The density-fitting (resolution-of-the-identity) pieces of J and K:

    (mn|ls) ≈ Σ_PQ (mn|P) (P|Q)⁻¹ (Q|ls) = Σ_Q B_Q,mn B_Q,ls,   B = L⁻¹ (P|mn),  M = L Lᵀ

over an auxiliary `BasisSet` placed as the orbital one is,
`place(resolve_aux_basis(name)->primitive_normalized(), molecule)`.

## Module

`free_breakfast.integrals.eri.density_fitting`, the aggregate, re-exports
`metric`, `three_center`, `fit` and `jk`; it is itself re-exported by `free_breakfast.integrals`.
Link `free_breakfast::integrals::eri::density_fitting`. The sub-modules remain
importable individually.

```
metric/        free_breakfast.integrals.eri.density_fitting.metric: the Coulomb metric (P|Q) of an aux basis on the GPU, Cartesian and spherical, over the generated metric<lP, lQ>, 28 classes lP <= lQ through i
fit/           free_breakfast.integrals.eri.density_fitting.fit: the fit B = L^-1 (mn|P), M = L L^T, one potrf and one trsm in place
jk/            free_breakfast.integrals.eri.density_fitting.jk: density-fitted J and K over B, spherical
three_center/  free_breakfast.integrals.eri.density_fitting.three_center: the spherical three-center (mn|P) on the GPU, [m][n][P] with P fastest (what density_fit takes), over the generated eri3<la, lb, lP>, 70 classes la <= lb through f, lP through i
```

## How the pieces fit

```
resolve_aux_basis --place--> BasisSet aux          (io.basis, primitive_normalized, over the orbital basis's molecule)
aux --metric-->              MatrixAOPair M        (one launch per class into the Cartesian M, U^T M U, made exactly symmetric)
basis, mmd classes, aux --three_center--> (P|mn)   (one launch per class into the Cartesian tensor, spherical on all three indices, made exactly (m, n)-symmetric)
M.spherical(), (P|mn) --density_fit--> L, B        (potrf of M, trsm of (P|mn), in place)
B, D --jk-->                 J, K                  (jk/'s README has the products)
```

**Generated kernels are committed, not built.** `tools/emit/df_kernels.py`
writes `metric/kernels/` and `three_center/kernels/` from the FelizJueves
submodule; rerun it after bumping that submodule, and never edit the emitted
text by hand.
