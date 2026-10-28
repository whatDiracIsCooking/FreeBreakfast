# `free_breakfast.integrals.eri.density_fitting.mo`

The fitted factor B of [`density_fitting.fit`](../fit/README.md) in the MO
basis, over arbitrary column ranges of the orbitals C, all on `res.stream()`.
Link `free_breakfast::integrals::eri::density_fitting::mo`.

    b_half(res, naux, nao, nmo, b, c, rows, out)          [Q][n][m] -> [Q][p][m]
        X_m = B_m C_rows            (one gemmStridedBatched over m)

    b_mo(res, naux, nao, nmo, b, c, rows, cols, out)      [Q][n][m] -> [Q][p][q]
        X_m = B_m C_rows            (b_half, into the workspace)
        B_MO = X C_cols             (one gemm)

so that

    B_P,pq = Σ_mn B_P,mn C_mp C_nq,     B_P,mn = L⁻¹ (mn|P),     (pq|rs) ≈ Σ_P B_P,pq B_P,rs

`rows` and `cols` are `OrbitalRange{first, count}`, the columns
`[first, first + count)` of C, so `(ij|P)`, `(ia|P)`, `(ab|P)` and any other
block are the same call. **Not a new approximation:** `L⁻¹` touches only P and
C only m and n, so fitting and transforming commute, and these read B exactly
as `density_fit` leaves it. The fast AO index is contracted first, which is
legal because `(mn|P)` is symmetric in m and n.

No device code: two BLAS calls, no `.cu`, nothing for
`cross-backend-check.sh`. `strideB = 0`, C's block shared by the batch, is
legal in both cuBLAS and hipBLAS, and `df_exchange` already relies on it.

## Layout

The output layouts are the contract. Every matrix is column-major.

| Operand | Shape | Element | Size |
| --- | --- | --- | --- |
| `b` | B, `naux` x `nao²` | `Q + naux (n + nao m)`: `[Q][n][m]` | `naux · nao²` doubles (the caller's) |
| `c` | C, `nao` x `nmo`, packed | `n + nao p` | `nao · nmo` doubles |
| `b_half`'s `out` | X, `(naux · \|rows\|)` x `nao` | `Q + naux (p + \|rows\| m)`: `[Q][p][m]` | `naux · \|rows\| · nao` doubles |
| `b_mo`'s `out` | `(naux · \|rows\|)` x `\|cols\|` | `Q + naux (p + \|rows\| q)`: `[Q][p][q]` | `naux · \|rows\| · \|cols\|` doubles |
| X (scratch, `b_mo`) | as `b_half`'s `out` | | `naux · \|rows\| · nao` doubles |

`b_half`'s X is **`df_exchange`'s X exactly** (`Q + naux (i + nocc m)`): with
`rows` the occupied block it is the same call on the same operands.

`b_mo`'s slice at fixed p is the `naux` x `|cols|` matrix at `naux p` with
leading dimension `naux · |rows|`, and consecutive p are `naux` apart. That is
what makes DF-MP2's pair products one `gemmStridedBatched`
(`(ia|jb)` blocks as `B_iᵀ B_j`, `lda = ldb = naux · |rows|`, stride `naux`)
rather than a transpose, and what a caller wanting another order permutes
from.

The second contraction is one gemm a caller could write over `b_half`'s
output, so it is not exported on its own.

## Memory

X, `b_mo`'s scratch, is `res.device_workspace()`'s, carved by
`device_regions`, so none of the operands may live there. When the workspace
would have to grow, X's bytes are checked against the device's free memory
first: past it is `Unsupported`, naming the bytes, since the allocation would
abort (`CreatePolicy`) rather than return. `b_half` takes no scratch.

## Limits

Both gemms run in the `int32_t` BLAS dispatch, and the second's rows and
leading dimension are `naux · |rows|`, so that past `INT32_MAX` is
`InvalidConfig`, as is a negative count or a range outside `[0, nmo)`. An empty
output (`naux`, `|rows|` or `|cols|` zero) succeeds and writes nothing; for
`b_mo`, `nao = 0` with a non-empty output writes zeros.

## `df_exchange` keeps its own copy, deliberately

`b_half` is the same operation as `df_exchange`'s X, and
[`jk/`](../jk/README.md) does **not** call it:

- X there is module-private scratch carved from `res.device_workspace()`,
  never a product the caller sees;
- its index order is chosen for the syrk that follows it, over
  `naux · nocc` rows; MP2's pair gemms want the other order, and forcing one
  layout pessimizes one consumer;
- it is on the SCF's per-iteration hot path, and `devtools/cpp-tier.sh` is the
  only gate that would catch a regression there.

Consolidate only if a third consumer turns up wanting jk's layout. Until then
`df_mo_gpu` pins the two to the same answer: `b_half` over the occupied range,
followed by the test's own syrk and symmetrization, reproduces
`df_exchange`'s K to a few ULP, and its X matches the element formula above.

## Why here

`integrals/active_space` is the precedent for MO-basis integrals under
`integrals/`, and `fit/` for a pure-BLAS module here. The matrix AO↔MO
transform is `recipes::blas::gemmwich`; this is the three-index one, for
DF-MP2 and any other method that reads B over orbital blocks.
