# `free_breakfast.integrals.eri.density_fitting.fit`

`density_fit(res, nao, naux, metric, b)`: the density-fitting factor

    M = L L^T            (recipes/cholesky, lower)
    B = L^-1 (P|mn)      (one trsm: left, lower, no transpose)

in place, so that `(mn|ls) ~ sum_Q B_Q,mn B_Q,ls = (mn|P) M^-1 (P|ls)`. Link
`free_breakfast::integrals::eri::density_fitting::fit`.

## Layout

| Operand | Shape | Element | Size |
| --- | --- | --- | --- |
| `metric` in | (P\|Q), `naux` x `naux` | `P + naux Q` | `naux²` doubles |
| `metric` out | L, its lower triangle | `P + naux Q`, P >= Q | the same buffer |
| `b` in | (P\|mn), `naux` x `nao²` | `P + naux (n + nao m)`: `[P][n][m]`, P fastest | `naux · nao²` doubles |
| `b` out | B, the same | `Q + naux (n + nao m)`: `[Q][n][m]`, Q fastest | the same buffer |

B's `[Q][n][m]` is the layout J and K read: a gemv over `vec(D)` for J, and
for K the `naux` x `nao` blocks `B_m` at `naux · nao · m`, one per m. (mn|P)
is symmetric in m and n, so the caller picks which is slower when it writes
the tensor, and the fit keeps it.

B is the caller's buffer, not scratch: at `naux · nao²` doubles it is the
largest thing a DF calculation holds (H₂O in cc-pVTZ / cc-pVTZ-JKFIT, 139 x
58², is 3.7 MB). The fit takes no scratch.

## Failure

A metric that is not positive definite is `Numerical`, naming the pivot
(the 1-based leading minor the factorization stopped at). There is no eigen-decomposition
fallback: no basis in the tree needs one yet, and a regularized metric would
change what B means for every caller.

## Why not potrf

The factorization was the dense solver's `potrf` until #404: on ROCm 7.2,
rocSOLVER's small-matrix kernel races, and the metric came out "not positive
definite", or its factor slightly wrong, whenever other processes shared the
AMD card. `recipes/cholesky/README.md` has the race.
