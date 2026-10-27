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

**K is built from a factor, never from D.** `C` is `C_occ` in an SCF, or any
factor of a positive semidefinite density (the SAD guess's). Two factors of
one D give the same K to rounding: K depends on C only through C C^T.

**K is exactly symmetric.** syrk writes one triangle, into a zeroed scratch;
`L + L^T` adds an exact 0 to each off-diagonal element, so (m, n) and (n, m)
carry the same bits, and the doubled diagonal is copied back from L.

## Layout

Every matrix is column-major and packed.

| Operand | Shape | Element | Size |
| --- | --- | --- | --- |
| `b` | B, `naux` x `nao²` | `Q + naux (n + nao m)`: `[Q][n][m]`, Q fastest | `naux · nao²` doubles (the caller's) |
| `density`, `j`, `k` | `nao` x `nao` | `n + nao m` | `nao²` doubles each |
| `c` | C, `nao` x `nocc` | `n + nao i` | `nao · nocc` doubles |
| gamma (scratch, J) | `naux` | `Q` | `naux` doubles |
| X (scratch, K) | `(naux · nocc)` x `nao` | `Q + naux (i + nocc m)`: `[Q][i][m]` | `naux · nocc · nao` doubles |
| L (scratch, K) | `nao` x `nao`, lower | `n + nao m` | `nao²` doubles |

B, at `naux · nao²` doubles, is still the largest thing DF holds; X is B's
size times `nocc / nao`, so below it whenever the occupied space is smaller
than the basis (H₂O in cc-pVTZ / cc-pVTZ-JKFIT: B 139 x 58² is 3.7 MB, X
139 · 5 · 58 is 0.32 MB). The scratch is `res.device_workspace()`'s, carved by
`device_regions`, so none of the operands may live there.

## Limits

J's gemvs run over `nao²` columns and K's syrk over `naux · nocc` rows in the
`int32_t` BLAS dispatch both backends carry, so either past `INT32_MAX` is
`InvalidConfig`. An empty `naux` (or `nocc`) gives J (or K) = 0; `nao = 0`
writes nothing.
