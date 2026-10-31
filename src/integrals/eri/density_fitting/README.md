# Density fitting

The density-fitting (resolution-of-the-identity) pieces of J and K:

    (mn|ls) ≈ Σ_PQ (mn|P) (P|Q)⁻¹ (Q|ls) = Σ_Q B_Q,mn B_Q,ls,   B = L⁻¹ (P|mn),  M = L Lᵀ

over an auxiliary `BasisSet` placed as the orbital one is,
`place(resolve_aux_basis(name)->primitive_normalized(), molecule)`.

## Module

`free_breakfast.integrals.eri.density_fitting`, the aggregate, re-exports
`metric`, `three_center`, `fit`, `jk`, `mo` and `gradient`; it is itself re-exported by `free_breakfast.integrals`.
Link `free_breakfast::integrals::eri::density_fitting`. The sub-modules remain
importable individually.

```
metric/        free_breakfast.integrals.eri.density_fitting.metric: the Coulomb metric (P|Q) of an aux basis on the GPU, Cartesian and spherical, over the generated metric<lP, lQ>, 28 classes lP <= lQ through i
gradient/      free_breakfast.integrals.eri.density_fitting.gradient: the nuclear gradients Σ W d(P|Q)/dR and Σ Γ d(mn|P)/dR for spherical W and Γ, n_atom x 3 on the host, over the generated gmetric<lP, lQ> (28 classes) and geri3<la, lb, lP> (70); the derivative integrals themselves, d(P|Q)/dA and (∇m n|P) one aux atom at a time, over ipmetric<lP, lQ> and xeri3<la, lb, lP>; and over those, the DF J and K differentiated per nuclear coordinate (df_jk_derivative)
fit/           free_breakfast.integrals.eri.density_fitting.fit: the fit B = L^-1 (mn|P), M = L L^T, one Cholesky factorization (recipes/cholesky) and one trsm in place
jk/            free_breakfast.integrals.eri.density_fitting.jk: density-fitted J and K over B, spherical
mo/            free_breakfast.integrals.eri.density_fitting.mo: B in the MO basis over arbitrary column ranges of C, the half transform b_half (one batched gemm) and b_mo (that and one gemm)
three_center/  free_breakfast.integrals.eri.density_fitting.three_center: the spherical three-center (mn|P) on the GPU, [m][n][P] with P fastest (what density_fit takes), over the generated eri3<la, lb, lP>, 70 classes la <= lb through f, lP through i
```

## How the pieces fit

```
resolve_aux_basis --place--> BasisSet aux          (io.basis, primitive_normalized, over the orbital basis's molecule)
aux --metric-->              MatrixAOPair M        (one launch per class into the Cartesian M, U^T M U, made exactly symmetric)
basis, mmd classes, aux --three_center--> (P|mn)   (one launch per class into the Cartesian tensor, spherical on all three indices, made exactly (m, n)-symmetric)
M.spherical(), (P|mn) --density_fit--> L, B        (Cholesky of M, trsm of (P|mn), in place)
B, D --jk-->                 J, K                  (jk/'s README has the products)
B, C, ranges --mo-->         B_P,pq                (b_half, then a gemm; mo/'s README has the layouts)
aux, W --metric_gradient-->  Σ W d(P|Q)/dR         (W made Cartesian, one launch per class into G; gradient/'s README)
basis, classes, aux, Γ --three_center_gradient--> Σ Γ d(mn|P)/dR   (likewise)
aux --metric_derivative-->   d(P|Q)/dA                   (one launch per class, U_aux^T M1 U_aux)
basis, classes, aux, atom --three_center_derivative--> (∇m n|P), P on atom   (likewise, per aux atom)
B, L, C --df_jk_derivative--> J^(x), K^(x), u^x - M^x d   (both, one aux atom at a time; gradient/'s README)
```

**Generated kernels are committed, not built.** `tools/emit/df_kernels.py`
writes `metric/kernels/` and `three_center/kernels/`, and
`tools/emit/df_gradient_kernels.py` writes `gradient/kernels/`, from the
FelizJueves submodule; rerun both after bumping that submodule, and never edit
the emitted text by hand.
