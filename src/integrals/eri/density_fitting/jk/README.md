# `free_breakfast.integrals.eri.density_fitting.jk`

J and K over the fitted factor B of
[`density_fitting.fit`](../fit/README.md), all spherical, all on
`res.stream()`. Link
`free_breakfast::integrals::eri::density_fitting::jk`.

    df_coulomb(res, nao, naux, b, density, j)
        gamma = B vec(D)            (gemv)
        vec(J) = B^T gamma          (gemv)

    df_exchange(res, nao, naux, nocc, b, c, k)      D = 2 C C^T
        X_m = B_m C                 (one gemmStridedBatched over m)
        K / 2 = X^T X               (one syrk, lower)
        K = L + L^T, diagonal L's   (a geam and a strided copy)

    df_exchange(res, nao, naux, rank, b, left, right, k)    D = A B^T
        X_m = B_m A                 (one gemmStridedBatched over m)
        Y_m = B_m B                 (another)
        K = X^T Y                   (one gemm)

Here A is `left` and B is `right`, nao x rank factors, not the fitted B.

**K is built from a factor, never from D.** `C` is `C_occ` in an SCF, or any
factor of a positive semidefinite density (the SAD guess's). Two factors of
one D give the same K to rounding: K depends on C only through C C^T.

**K is exactly symmetric.** syrk writes one triangle, into a zeroed scratch;
`L + L^T` adds an exact 0 to each off-diagonal element, so (m, n) and (n, m)
carry the same bits, and the doubled diagonal is copied back from L.

**The two-factor form is for every other D.** The one-factor form's D is
`2 C Cᵀ`, symmetric and positive semidefinite; a response or transition
density (the CASSCF orbital Hessian's `C_core κ C_rotᵀ`) is neither. The
two-factor K is `K_mn = Σ_ls (ml|ns) (A Bᵀ)_ls` exactly as written, so no
symmetry is imposed or produced: `K[A Bᵀ]ᵀ = K[B Aᵀ]`. Hence

- a symmetrized `D = A Bᵀ + B Aᵀ` has `K = K[A Bᵀ] + K[A Bᵀ]ᵀ`, one call and a
  `geam`;
- a sum `A₁ B₁ᵀ + A₂ B₂ᵀ` is one call over the factors side by side,
  `[A₁ A₂]` and `[B₁ B₂]`, rank `r₁ + r₂`: two packed nao-row matrices placed
  one after the other in memory already are that concatenation;
- `K[C Cᵀ]` is half the one-factor `df_exchange(C)`, which is `K[2 C Cᵀ]`.

It costs one more batched gemm than the one-factor form, and a gemm where that
form has a syrk (twice the flops of K's product), with twice the scratch.

## Layout

Every matrix is column-major and packed.

| Operand | Shape | Element | Size |
| --- | --- | --- | --- |
| `b` | B, `naux` x `nao²` | `Q + naux (n + nao m)`: `[Q][n][m]`, Q fastest | `naux · nao²` doubles (the caller's) |
| `density`, `j`, `k` | `nao` x `nao` | `n + nao m` | `nao²` doubles each |
| `c` | C, `nao` x `nocc` | `n + nao i` | `nao · nocc` doubles |
| `left`, `right` | A, B, `nao` x `rank` | `n + nao i` | `nao · rank` doubles each |
| gamma (scratch, J) | `naux` | `Q` | `naux` doubles |
| X (scratch, K) | `(naux · nocc)` x `nao` | `Q + naux (i + nocc m)`: `[Q][i][m]` | `naux · nocc · nao` doubles |
| L (scratch, K) | `nao` x `nao`, lower | `n + nao m` | `nao²` doubles |
| X, Y (scratch, two-factor K) | `(naux · rank)` x `nao` | `Q + naux (i + rank m)` | `naux · rank · nao` doubles each |

B, at `naux · nao²` doubles, is still the largest thing DF holds; X is B's
size times `nocc / nao`, so below it whenever the occupied space is smaller
than the basis (H₂O in cc-pVTZ / cc-pVTZ-JKFIT: B 139 x 58² is 3.7 MB, X
139 · 5 · 58 is 0.32 MB). The scratch is `res.device_workspace()`'s, carved by
`device_regions`, so none of the operands may live there.

## Limits

J's gemvs run over `nao²` columns, K's syrk over `naux · nocc` rows and the
two-factor K's gemm over `naux · rank` in the `int32_t` BLAS dispatch both
backends carry, so any of them past `INT32_MAX` is `InvalidConfig`, as is a
negative count. An empty `naux` (or `nocc`, or `rank`) gives J (or K) = 0;
`nao = 0` writes nothing.

## X is `mo`'s `b_half`, kept as its own copy

The one-factor K's X is [`mo/`](../mo/README.md)'s `b_half` over the occupied
block: the same call on the same operands. jk does not call it, deliberately:
X here is private scratch, laid out for the syrk that follows, on the SCF's
per-iteration hot path. `mo/`'s README has the reasons, and `df_mo_gpu` pins
the two to the same K.
