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
`docs/rust-redesign.html` "workspace"). So no fixed handle workspace covers
it, and a call past it allocates its own memory, outside any budget. This
recipe needs nothing but the gemm the handle's fixed workspace already serves.

It is one source for both backends, so CUDA runs it too: one code path,
tested on both cards.

## Algorithm

Every case is put in one frame, M Y = alpha C with M of order k: on the left
M is op(A) and C is B, on the right M is op(A)^T and C is B^T. A lower M is
solved from the top block down, an upper one from the bottom up,
`kTrsmBlock` (64) rows at a time:

1. the diagonal block, by `trsm.cu`'s kernel: one thread block per
   `kTrsmColumns` (16) columns of C, the triangle and the tile in shared
   memory, forward substitution row by row with a barrier after the pivot
   division and after each update. An upper block is reversed into a lower
   one on its way into shared memory, so one loop serves both;
2. the rows of C still to come lose M's off-diagonal block times the rows just
   solved: one gemm, written in terms of A and B as they are stored, so
   neither is ever transposed or copied.

alpha is applied once: the first diagonal block scales by it, and the first
gemm scales every other row by its beta. With alpha zero B is set to zero
block by block and neither A nor B is read, as the reference BLAS does.

Off-diagonal blocks lie wholly in A's stored triangle, and the kernel reads
only the stored triangle (and not the diagonal under `WWRBLAS_DIAG_UNIT`), so
whatever the other triangle holds never reaches the result.

## Tests

`test/recipes/trsm_gpu.cpp`: every side, fill, op and unit flag, float and
double, against a host substitution at orders around the block edges, with
padded leading dimensions whose padding and A's other triangle hold NaN; the
same against the library's trsm at a size past many blocks; alpha zero over a
NaN B; and the refusals.
