# Trsm

`trsm<T>(res, side, fill, op, diag, m, n, alpha, a, lda, b, ldb)`, T `float`
or `double`: BLAS's triangular solve, B <- alpha op(A)^-1 B on the left or
B <- alpha B op(A)^-1 on the right, in place. Every argument means what it does
in `wwr::trsm` (wwr's own enums, leading dimensions and all), so moving a call
over changes only the handle for a `DeviceResources` and the pointer to alpha
for its value. It returns a `Status` and does not synchronize. Link
`free_breakfast::recipes::trsm`.

## Why not the BLAS library's trsm

On ROCm, rocBLAS's trsm takes workspace that grows with the right-hand sides:
about a copy of B when the triangular order is not a multiple of 128, and 1/32
of it when it is (`experimental/rust/blas_workspace_scaling`,
and the v0.0.4 prototype's page, "workspace", at `b0935805`). So no fixed handle workspace covers
it, and a call past it allocates its own memory, outside any budget. This
recipe needs nothing but the gemm the handle's fixed workspace already serves.

It is one source for both backends, so CUDA runs it too: one code path,
tested on both cards.

## Algorithm

One right-hand side (n = 1 on the left, m = 1 on the right) with a nonzero
alpha is a vector: a `scal` when alpha is not 1, then the library's `trsv`,
which takes no workspace on either backend. The rest is the blocked solve.

Every case is put in one frame, M Y = alpha C with M of order k: on the left
M is op(A) and C is B, on the right M is op(A)^T and C is B^T. A lower M is
solved from the top block down, an upper one from the bottom up,
`kTrsmBlock` (64) rows at a time:

1. the diagonal block, by `trsm.cu`'s kernel: one thread block per
   `kTrsmColumns` (48) columns of C, `kTrsmLanes` (4) threads to a column,
   the triangle packed and the tile in shared memory. The triangle is scaled
   to a unit one first (each column times the reciprocal of its diagonal), so
   the substitution has no division and one barrier a row: once row k is
   final every column's threads take their multiple of it from the rows
   below. The diagonal's reciprocals are applied on the way out. An upper
   block is reversed into a lower one on its way into shared memory, so one
   loop serves both;
2. the rows of C still to come lose M's off-diagonal block times the rows just
   solved: one gemm, written in terms of A and B as they are stored, so
   neither is ever transposed or copied.

alpha is applied once: the first diagonal block scales by it, and the first
gemm scales every other row by its beta. With alpha zero B is set to zero
block by block and neither A nor B is read, as the reference BLAS does.

Off-diagonal blocks lie wholly in A's stored triangle, and the kernel reads
only the stored triangle (and not the diagonal under `WWRBLAS_DIAG_UNIT`), so
whatever the other triangle holds never reaches the result.

## Against the libraries' trsm

`bench/trsm_vendor.cpp` (`free_breakfast_trsm_vendor`), 2026-11-24, doubles,
the best of 3 after a warm-up, both solves in one process on the same A and B
with the card held; RTX 3080 (cuBLAS, CUDA 13) and RX 9060 XT (rocBLAS 5.6.0).
Milliseconds; "before" is the recipe as it first stood (a 16-column tile, a
pivot division and two barriers a row, no `trsv` path), "ratio" the recipe
now over the library. The shape is side, fill, op, diag; `LLNN` with
n = nao^2 is the density-fitting solve, the `L...` rows at 1, 32 and 1024
right-hand sides are `cholesky_solve`'s and `lu_solve`'s, and `RLTN` with
n = 64 is the Cholesky factor's panel.

| Shape | m | n | 3080 before | 3080 now | cuBLAS | ratio | 9060 XT before | 9060 XT now | rocBLAS | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| `LLNN` | 116 | 576 | 0.166 | 0.136 | 0.145 | 0.94 | 0.229 | 0.181 | 0.47 | 0.39 |
| `LLNN` | 540 | 14400 | 16.5 | 13.7 | 21.7 | 0.63 | 20.6 | 14.8 | 16 | 0.92 |
| `LLNN` | 1100 | 62500 | 225 | 199 | 249 | 0.80 | 281 | 228 | 227 | 1.00 |
| `LLNN` | 1350 | 90000 | 478 | 425 | 510 | 0.83 | 552 | 470 | 452 | 1.04 |
| `LLNN` | 2000 | 193600 | 2112 | 1914 | 2220 | 0.86 | 2247 | 1972 | 1983 | 0.99 |
| `LLNN` | 1280 | 90000 | 406 | 353 | 461 | 0.76 | 477 | 390 | 374 | 1.04 |
| `LLNN` | 2048 | 193600 | 2138 | 1941 | 2255 | 0.86 | 2257 | 1976 | 1954 | 1.01 |
| `LLNN` | 512 | 1 | 0.365 | 0.077 | 0.077 | 1.00 | 0.721 | 0.077 | 0.074 | 1.04 |
| `LLNN` | 512 | 32 | 0.67 | 0.641 | 0.782 | 0.82 | 0.893 | 0.784 | 0.963 | 0.81 |
| `LLNN` | 512 | 1024 | 1.2 | 1.26 | 1.17 | 1.08 | 1.85 | 1.22 | 1.13 | 1.07 |
| `LLNN` | 2000 | 1 | 1.38 | 0.274 | 0.274 | 1.00 | 2.95 | 0.168 | 0.167 | 1.01 |
| `LLNN` | 2000 | 32 | 2.84 | 2.69 | 3.48 | 0.77 | 3.64 | 3.42 | 3.86 | 0.88 |
| `LLNN` | 2000 | 1024 | 13 | 14.3 | 14.6 | 0.98 | 14.7 | 13.1 | 12.3 | 1.07 |
| `LLTN` | 512 | 1 | 0.362 | 0.121 | 0.121 | 1.00 | 0.745 | 0.079 | 0.079 | 1.00 |
| `LLTN` | 512 | 32 | 0.669 | 0.637 | 0.776 | 0.82 | 0.871 | 0.485 | 0.431 | 1.13 |
| `LLTN` | 512 | 1024 | 1.18 | 1.25 | 1.16 | 1.08 | 1.84 | 1.26 | 1.35 | 0.93 |
| `LLTN` | 2000 | 1 | 1.43 | 0.46 | 0.459 | 1.00 | 2.83 | 0.154 | 0.154 | 1.00 |
| `LLTN` | 2000 | 32 | 2.75 | 2.65 | 3.23 | 0.82 | 3.55 | 2.17 | 1.32 | 1.64 |
| `LLTN` | 2000 | 1024 | 12.4 | 13.8 | 12.9 | 1.07 | 15.4 | 12.1 | 11.8 | 1.03 |
| `LLNU` | 512 | 1 | 0.379 | 0.059 | 0.058 | 1.02 | 0.72 | 0.07 | 0.07 | 1.00 |
| `LLNU` | 512 | 32 | 0.686 | 0.654 | 0.445 | 1.47 | 0.891 | 0.776 | 0.92 | 0.84 |
| `LLNU` | 512 | 1024 | 1.21 | 1.28 | 0.832 | 1.53 | 1.84 | 1.15 | 1.16 | 0.99 |
| `LLNU` | 2000 | 1 | 1.44 | 0.21 | 0.21 | 1.00 | 2.93 | 0.157 | 0.156 | 1.01 |
| `LLNU` | 2000 | 32 | 2.89 | 2.74 | 2.18 | 1.26 | 3.63 | 3.41 | 3.86 | 0.88 |
| `LLNU` | 2000 | 1024 | 13.1 | 14.3 | 12 | 1.19 | 15.7 | 13.9 | 13.2 | 1.05 |
| `LUNN` | 512 | 1 | 0.364 | 0.077 | 0.076 | 1.01 | 0.713 | 0.079 | 0.076 | 1.04 |
| `LUNN` | 512 | 32 | 0.671 | 0.641 | 0.775 | 0.83 | 0.893 | 0.753 | 0.916 | 0.82 |
| `LUNN` | 512 | 1024 | 1.2 | 1.26 | 1.16 | 1.09 | 1.84 | 1.16 | 1.19 | 0.98 |
| `LUNN` | 2000 | 1 | 1.39 | 0.272 | 0.271 | 1.00 | 2.95 | 0.166 | 0.165 | 1.01 |
| `LUNN` | 2000 | 32 | 2.83 | 2.68 | 3.25 | 0.82 | 3.64 | 3.4 | 3.82 | 0.89 |
| `LUNN` | 2000 | 1024 | 12.3 | 13.7 | 14.1 | 0.98 | 15.4 | 13.3 | 12.9 | 1.04 |
| `RLTN` | 500 | 64 | 0.07 | 0.064 | 0.114 | 0.56 | 0.082 | 0.068 | 0.216 | 0.31 |
| `RLTN` | 2000 | 64 | 0.073 | 0.065 | 0.12 | 0.54 | 0.232 | 0.097 | 0.196 | 0.49 |
| `RLTN` | 8000 | 64 | 0.272 | 0.174 | 0.452 | 0.38 | 0.784 | 0.278 | 0.347 | 0.80 |

At the density-fitting solve the recipe is 0.76 to 0.86 of cuBLAS's time
and 0.99 to 1.04 of rocBLAS's from 1 GB of B up; before, 0.84 to 0.91 and
1.13 to 1.28. The difference is the diagonal blocks: at m = 2000 their share
of the recipe's time (its time less its gemms replayed alone) fell from 190
to about 65 ms on the 3080 and from 417 to about 143 on the 9060 XT. The
gemms alone take 1.83 s there against rocBLAS's 1.98, so what is left to gain
is in their inner dimension of 64, not in the kernel.

One right-hand side is the library's own time, since it is the library's
`trsv` (5 to 19 times slower before). The gaps that remain are small
solves, a few milliseconds at most, and are accepted: a unit lower triangle
on the 3080 (1.19 to 1.53: cuBLAS is faster on a unit diagonal than on a
general one, the recipe the same on both), and `LLTN` at 2000 by 32 on the
9060 XT (1.64, 0.85 ms). The sub-millisecond rows move by about a tenth
between runs.

## Tests

`test/recipes/trsm_gpu.cpp`: every side, fill, op and unit flag, float and
double, against a host substitution at orders around the block edges, with
padded leading dimensions whose padding and A's other triangle hold NaN; the
same against the library's trsm at a size past many blocks (one right-hand
side is the `trsv` path); alpha zero over a NaN B; and the refusals.
