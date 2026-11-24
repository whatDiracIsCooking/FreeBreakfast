# Cholesky

`cholesky(res, n, a)`: A = L L^T for a symmetric positive definite A, in
place, L in the lower triangle. It is LAPACK's `potrf` with `uplo = L`, and it
returns `potrf`'s flag: 0, or the 1-based order of the first leading minor
that is not positive.

`cholesky_solve(res, n, nrhs, l, b)`: A X = B in place over that factor,
B <- L^-T L^-1 B. It is LAPACK's `potrs` with `uplo = L`, as two of
`recipes/trsm`'s solves (left, lower, non-unit; `OP_N`, then `OP_T`). It
returns a `Status`, takes no flag and does not synchronize.

Link `free_breakfast::recipes::cholesky`.

## Why not the dense solver's potrf

On ROCm 7.2, rocSOLVER factors every matrix of order 127 or less, and the
first 127-column block of a larger one, in one kernel,
`potf2_kernel_small` (`library/src/specialized/roclapack_potf2_specialized_kernels.hpp`).
Each step of its loop has every thread read the pivot `A[k][k]` from shared
memory, and thread 0 then overwrite it with its square root, with no barrier
between the two. A wave that reaches the read late gets `l_kk` instead of
`a_kk`, and divides its rows of column k by `sqrt(l_kk)`. Whether that happens
depends on how far apart the block's waves drift, so alone the result was
always right, and with other processes on the card it was not: the DF metric
came out "not positive definite" at a pivot that changed from run to run
(#404), or its factor, and with it B, came out slightly wrong (#379). In every
captured failure the bad rows were exactly the correct ones times
`l_kk^{1/2}`. Upstream rewrote the kernel in 2026; nothing here can choose
the version a ROCm install carries, so the tree does not call `potrf` at all.

`cholesky` is one source for both backends, so CUDA runs it too: one code path,
tested on both cards.

## Why not the dense solver's potrs

On HIP, hipSOLVER's `potrs` allocates about B's size plus ~42 MiB of its own
on a call, outside any budget the caller set, and keeps it
(`experimental/rust/blas_workspace_scaling/README.md`, "Solvers"). Its two
triangular solves are `recipes/trsm`'s here, which take nothing but the gemm
the handle's fixed workspace already serves, and nothing from
`res.device_workspace()`, so `l` or `b` may sit in it.

## Algorithm

Right-looking and blocked, `kCholeskyBlock` (64) columns at a time:

1. the 64 x 64 diagonal block, in shared memory, by `cholesky.cu`'s one-block
   kernel: per column, read the pivot, barrier, scale the column, barrier,
   rank-1 update of the trailing triangle, barrier;
2. the panel below it, `A21 <- A21 L11^-T` (a trsm);
3. the trailing matrix, `A22 <- A22 - L21 L21^T` (a syrk, lower).

The flag lives in `res.info()`. A diagonal block that finds it already set
leaves itself alone, so the flag names the first failing minor, as `potrf`'s
does. The routine takes no scratch, so `a` may sit in `res.device_workspace()`.

## Tests

`test/recipes/cholesky_gpu.cpp`: against a host Cholesky at orders around the
block edges, bitwise repeatable over repeated runs, and the flag for a matrix
that is not positive definite; `cholesky_solve` against a host solve at orders
around the block edges with one and several right-hand sides, and its empty
and negative sizes.
