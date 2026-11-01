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
        upper = lower^T             (recipes' symmetrize, in place)

    df_exchange(res, nao, naux, rank, b, left, right, k)    D = A B^T
        X_m = B_m A                 (one gemmStridedBatched over m)
        Y_m = B_m B                 (another)
        K = X^T Y                   (one gemm)

    df_coulomb(res, nao, naux, count, b, density, j)              D_0 ... D_count-1
        Gamma = B [vec(D_d)]        (gemm, naux x count)
        [vec(J_d)] = B^T Gamma      (gemm)

    df_exchange(res, nao, naux, rank, count, b, left, right, k)   D_d = A_d B_dᵀ
        X_m = B_m [A_0 ... A_count-1]     (one gemmStridedBatched over m)
        Y_m = B_m [B_0 ... B_count-1]     (another)
        K_d = X_dᵀ Y_d                    (one gemmStridedBatched over d)

    df_exchange_symmetrized(res, nao, naux, rank, count, b, left, right, k)
                                          D_d = A B_dᵀ, K of (D_d + D_dᵀ) / 2
        X_m = B_m A                       (one gemmStridedBatched over m, once)
        Y_m = B_m [B_0 ... B_count-1]     (another)
        K_d = Xᵀ Y_d                      (one gemmStridedBatched over d, X shared)
        K_d <- (K_d + K_dᵀ) / 2           (recipes' symmetrize, per d)

Here A is `left` and B is `right`, nao x rank factors, not the fitted B.

**K is built from a factor, never from D.** `C` is `C_occ` in an SCF, or any
factor of a positive semidefinite density (the SAD guess's). Two factors of
one D give the same K to rounding: K depends on C only through C C^T.

**K is exactly symmetric.** syrk writes the lower triangle of `k`, and
[`recipes.symmetrize`](../../../../recipes/symmetrize/README.md) copies its
transpose onto the upper one, so (m, n) and (n, m) carry the same bits.

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

**The block forms are the one-density forms over many densities.** For a
solver whose every iteration builds J and K of several densities over one B
(CPHF's columns, `response/cphf/`): `count` densities packed one after
another, `count` J or K out the same way. J's two gemvs become two gemms of
width `count`; the two-factor K's batched gemms over m take every density's
factors side by side, width `count · rank`, and its product is one gemm
batched over the densities, each X_d and Y_d a band of `naux · rank` rows of
the `(naux · count · rank)` x nao X and Y. Each J_d and K_d is the
one-density call's to rounding (`df_jk_batched_gpu`); the scratch is
`count` times the one-density call's. There is no block form of the
one-factor K: an SCF builds one K per iteration.

**The symmetrized form is the block K at half the rank.** The CP-CASSCF
response densities (Wang et al.'s D_CV and D_Gen) and the CI responses'
transition densities have nonzero rows only in the occupied block, so they
come factored with no SVD: `D = C_o (C D_o·ᵀ)ᵀ`, one left factor `C_o` for
every density and `R_d = C D_d,o·ᵀ`, rank n_occ. Only `D`'s symmetric part
reaches an energy, and since `K[A Bᵀ]ᵀ = K[B Aᵀ]` its K is `(K[A Bᵀ] +
K[A Bᵀ]ᵀ) / 2`. The block form gets the same K from `[R_d A]` and `[A R_d]`,
rank 2 n_occ, with A's columns repeated in every density; this one builds
A's X once and each Y_d at rank n_occ, so its batched gemms over m carry
`(count + 1) n_occ` columns where the block form's carry `4 count n_occ`, its
product half the inner dimension, and its scratch is about a quarter. Its
K_d is half the block form's of `[R_d A]`, `[A R_d]` (that pair's `L Rᵀ` is
`D + Dᵀ`), to rounding (`df_jk_symmetrized_gpu`, 1e-12 of max |K| on random
factors and on D_CV and D_Gen at the casscf_hessian fixture's λ^x), and
exactly symmetric. Nothing calls it yet: CPHF and `mcscf::df_derivatives`
keep the block form. An SVD compression of the factors (Wang et al.'s) is
not built.

`bench/df_exchange_symmetrized.cpp` times it against the block form of
`[R_d A]`, `[A R_d]` on the same K, over a random B (best of 10 after a
warm-up, RTX 3080; the gfx1200 was not timed):

| nao | naux | n_occ | count | block form (ms) | symmetrized (ms) | speed-up |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 116 | 7 | 9 | 0.727 | 0.375 | 1.94× |
| 58 | 260 | 9 | 9 | 4.017 | 1.527 | 2.63× |
| 120 | 540 | 15 | 27 | 103.3 | 37.05 | 2.79× |
| 200 | 900 | 25 | 24 | 693.4 | 247.0 | 2.81× |
| 250 | 1100 | 41 | 12 | 962.4 | 350.2 | 2.75× |
| 300 | 1350 | 40 | 8 | 1113.2 | 409.4 | 2.72× |

The ratio follows the batched gemms' columns, `4 count` against
`count + 1` per n_occ, less the per-density symmetrize launches at small
sizes.

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
| X, Y (scratch, two-factor K) | `(naux · rank)` x `nao` | `Q + naux (i + rank m)` | `naux · rank · nao` doubles each |
| `density`, `j`, `k` (block forms) | `count` x (`nao` x `nao`) | `n + nao m + nao² d` | `count · nao²` doubles each |
| `left`, `right` (block K) | `nao` x `count · rank` | `n + nao (i + rank d)` | `nao · count · rank` doubles each |
| Gamma (scratch, block J) | `naux` x `count` | `Q + naux d` | `naux · count` doubles |
| X, Y (scratch, block K) | `(naux · count · rank)` x `nao` | `Q + naux (i + rank d + count rank m)` | `naux · count · rank · nao` doubles each |
| `left` (symmetrized K) | A, `nao` x `rank` | `n + nao i` | `nao · rank` doubles |
| X (scratch, symmetrized K) | `(naux · rank)` x `nao` | `Q + naux (i + rank m)` | `naux · rank · nao` doubles |
| Y (scratch, symmetrized K) | as the block K's Y | | `naux · count · rank · nao` doubles |

B, at `naux · nao²` doubles, is still the largest thing DF holds; X is B's
size times `nocc / nao`, so below it whenever the occupied space is smaller
than the basis (H₂O in cc-pVTZ / cc-pVTZ-JKFIT: B 139 x 58² is 3.7 MB, X
139 · 5 · 58 is 0.32 MB). The scratch is `res.device_workspace()`'s, carved by
`device_regions`, so none of the operands may live there.

## Limits

J's gemvs run over `nao²` columns, K's syrk over `naux · nocc` rows and the
two-factor K's gemm over `naux · rank` (the block and symmetrized K's
leading dimension `naux · count · rank`) in the `int32_t` BLAS dispatch both
backends carry, so any of them past `INT32_MAX` is `InvalidConfig`, as is a
negative count. An empty `naux` (or `nocc`, or `rank`) gives J (or K) = 0;
`nao = 0`, or `count = 0` in a block form, writes nothing.

## X is `mo`'s `b_half`, kept as its own copy

The one-factor K's X is [`mo/`](../mo/README.md)'s `b_half` over the occupied
block: the same call on the same operands. jk does not call it, deliberately:
X here is private scratch, laid out for the syrk that follows, on the SCF's
per-iteration hot path. `mo/`'s README has the reasons, and `df_mo_gpu` pins
the two to the same K.
