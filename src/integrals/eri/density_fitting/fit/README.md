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
(the 1-based leading minor the factorization stopped at). The default fit
never regularizes it; `lindep`, below, is the opt-in alternative.

## A near-singular metric: `lindep`

`density_fit(res, nao, naux, metric, b, lindep)` with `lindep > 0` fits over
M's eigenmodes above `lindep` instead (#630), PySCF's DF `lindep` route
(`df.incore._eig_decompose`, which keeps `w > lindep`):

    M = U Λ Uᵀ                         (recipes/orthogonalize, a syevd)
    X = U Λ⁻¹ᐟ² Uᵀ, a mode with λ <= lindep contributing 0
    B = X (P|mn)                       (one gemm per m slab)

So BᵀB = (mn|P) M⁺ (P|ls) over the kept modes. PySCF's B is Λ⁻¹ᐟ² Uᵀ (P|mn)
over the kept modes alone, the same up to the rotation U: J, K and the
energy cannot tell them apart. Here B keeps its `naux` x `nao²` shape (a
dropped mode is a direction of zero weight, not a missing row), so a consumer
that reads only B and `naux` takes it unchanged. On a metric that drops
nothing it is a rotation of the Cholesky path's B, and BᵀB agrees to
rounding. `metric` comes out as X, the whole matrix, not L.

`lindep = 0`, the default, is the Cholesky path, bit for bit: the branch is
taken before anything else is queued. A negative or NaN `lindep` is
`InvalidConfig`. Its cost is a syevd of M, an `naux²` pool copy of M, and an
`naux` x `nao` product in the workspace for the slab gemms.

No basis the tree carries is known to need it, though the diffuse fitting sets
are much worse conditioned than the plain ones. cond(M) over ethylene at
Park's S0 and C3v MECI (#845), from libcint's `int2c2e` over the carried
files:

| Fitting set | cond(M) |
| --- | --- |
| cc-pVDZ/TZ-JKFIT | about 4e6 |
| aug-cc-pVDZ-JKFIT | 3.7e7 to 6.1e7 |
| aug-cc-pVTZ-JKFIT | 1.5e8 to 1.6e8 |
| aug-cc-pVDZ-acCD-6-6 | 4.2e9 to 4.6e9 |
| aug-cc-pVTZ-acCD-6-6 | 9.3e10 to 1.3e11 |

Over each of the four aug sets, the Cholesky path's DF-RHF energy at S0 still
matches PySCF's over the same set to 1e-10 Eh (CUDA, 2026-11-04;
`data/aux_basis/README.md` has the runs). That bounds energies only: no DF
gradient, which solves against L, has been compared over the aug sets yet.
It is not a pipeline option: the Eri step always takes the default, so every
product the pipeline hands on carries a Cholesky L.

### Which consumers take a `lindep` fit

| Consumer | Reads | `lindep` fit |
| --- | --- | --- |
| `jk/` (`df_coulomb`, `df_exchange`), the Scf step | B | accepts |
| `mo/`, NEVPT2's fitted integrals, CASCI/CASSCF energies | B | accepts |
| CIS-RIS (`cis/ris_precondition`) | its own fit | not offered: it calls `density_fit` with the default |
| `gradient::df_rhf`, `density_fitting/gradient`, `hessian::*` DF, `mcscf` gradients, NACs and derivatives, `nevpt2/gradient` | B **and L** | refuses: L is what they solve against, and X is not L |

"Refuses" is by contract, not by a check: those take L as a bare pointer,
which cannot say how it was made. The fit that would feed them a lindep
fit's X does not exist in the tree (the Eri step never sets it), and
extending them, to apply X with a gemm where they trsm with L and to
differentiate the eigendecomposition, is the work a basis that needs this
would bring with it.

`df_fit_lindep_gpu.cpp` has the cases: `lindep = 0` against the default bit
for bit; on H₂O cc-pVDZ / cc-pVDZ-JKFIT, BᵀB to 1e-12 and the DF-RHF energy
to 1e-11 of the Cholesky path's; and over the same set with three O shells
duplicated (`test/gtest/data/df_lindep/`), the default refused and the lindep
energy PySCF's eigen-route DF-RHF at the same threshold.

## Why not potrf

The factorization was the dense solver's `potrf` until #404: on ROCm 7.2,
rocSOLVER's small-matrix kernel races, and the metric came out "not positive
definite", or its factor slightly wrong, whenever other processes shared the
AMD card. `recipes/cholesky/README.md` has the race.
