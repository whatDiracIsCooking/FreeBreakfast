# faer_blas_lapack — symmetric eigendecomposition on the CPU: faer against LAPACK

Would the AO overlap S or the DF metric (P|Q) be worth diagonalizing on the
CPU, and if so, through faer (pure Rust, no C library) or LAPACK? This crate
times the full symmetric eigendecomposition of real S and (P|Q) matrices with
both. faer 0.24's `self_adjoint_evd` runs against OpenBLAS 0.3.29's `dsyevd`
(divide and conquer, the same algorithm as faer's) and `dsyevr` (MRRR). Each
is timed with eigenvalues alone and with eigenvectors, at 1, 4 and 12 threads.
CPU only, with no GPU anywhere.

## Verdict

**OpenBLAS's `dsyevd` is the faster routine at every size and thread count
measured, and faer never wins.** With eigenvectors, which canonical
orthogonalization and a metric pseudo-inverse both need, faer is 1.2–1.3×
slower at one thread, level with `dsyevd` (1.01–1.17×) at four, and
1.3–1.6× slower at twelve. With eigenvalues alone it is 1.5–3× slower.

faer's thread scaling stops at four. From 4 to 12 threads it gets
*slower* at every n up to 2154 and gains only 13–23% at 3606. That points
at the tridiagonal reduction: the eigenvalues-only time, which is almost
all reduction, is where faer falls furthest behind. No profile was taken
to confirm it. A rerun reproduced
this (S_gly30: 0.22 s at 4 threads and 0.34 s at 12, eigenvalues only).

**All three routines are accurate.** Every residual and orthogonality error
is below n·ε. The eigenvalues agree to 2e-13 of the largest. For every
matrix, every routine and thread count puts the same number of eigenvalues
below each cutoff (1e-6, 1e-8, 1e-10 of the largest), so the choice of
library never changes what a threshold keeps. `dsyevr`'s eigenvectors are
the least orthogonal, up to 6.6e-12 at n = 3606 (8× n·ε). That is the
known cost of MRRR, and it is also slower than `dsyevd` with eigenvectors,
so there is no case for using it here.

So if S or (P|Q) is ever diagonalized on the CPU, the answer is LAPACK
`dsyevd` at four or more threads. faer would cost 1.0–1.3× at four threads
in exchange for no C dependency. That is a reasonable trade only if a
LAPACK link is otherwise unwanted, and only at four threads or fewer.

## Results

AMD Ryzen 9 5900X (12 cores, AVX2), each run in its own container pinned
to CPUs `0..t-1`, which are distinct physical cores on this machine. Times
are medians of at least 3 runs and a 3 s budget (at most 15 runs), after
one warm-up. The input copy and the workspace allocation are outside the
timed region for both libraries. Matrices are cc-pVDZ S and cc-pVDZ-JKFIT
(P|Q) of the glycine chains in `.temp/gly`.

Times in seconds. "ratio" is faer over the faster LAPACK routine.

| matrix | n | vectors | faer t1 | dsyevd t1 | dsyevr t1 | ratio t1 | faer t4 | dsyevd t4 | dsyevr t4 | ratio t4 | faer t12 | dsyevd t12 | dsyevr t12 | ratio t12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S_gly5 | 379 | no | 0.00573 | 0.00388 | 0.00382 | 1.50 | 0.00544 | 0.00279 | 0.0027 | 2.02 | 0.00635 | 0.00333 | 0.00325 | 1.95 |
| S_gly5 | 379 | yes | 0.0109 | 0.00898 | 0.0111 | 1.21 | 0.0073 | 0.00622 | 0.00888 | 1.17 | 0.0104 | 0.00778 | 0.0104 | 1.33 |
| S_gly10 | 734 | no | 0.0285 | 0.019 | 0.0189 | 1.51 | 0.0196 | 0.0109 | 0.0106 | 1.85 | 0.027 | 0.0118 | 0.0115 | 2.35 |
| S_gly10 | 734 | yes | 0.0572 | 0.0465 | 0.0579 | 1.23 | 0.0275 | 0.0253 | 0.0401 | 1.09 | 0.036 | 0.028 | 0.0417 | 1.28 |
| S_gly15 | 1089 | no | 0.0807 | 0.0552 | 0.0536 | 1.51 | 0.0439 | 0.0266 | 0.0258 | 1.70 | 0.0668 | 0.0266 | 0.0261 | 2.56 |
| S_gly15 | 1089 | yes | 0.166 | 0.133 | 0.159 | 1.25 | 0.0676 | 0.065 | 0.0985 | 1.04 | 0.0879 | 0.065 | 0.101 | 1.35 |
| S_gly20 | 1444 | no | 0.173 | 0.112 | 0.112 | 1.54 | 0.0796 | 0.051 | 0.0498 | 1.60 | 0.126 | 0.0486 | 0.048 | 2.63 |
| S_gly20 | 1444 | yes | 0.362 | 0.289 | 0.326 | 1.25 | 0.134 | 0.131 | 0.197 | 1.03 | 0.171 | 0.122 | 0.187 | 1.41 |
| S_gly30 | 2154 | no | 0.566 | 0.34 | 0.341 | 1.66 | 0.219 | 0.141 | 0.138 | 1.59 | 0.335 | 0.116 | 0.114 | 2.93 |
| S_gly30 | 2154 | yes | 1.16 | 0.907 | 0.995 | 1.28 | 0.388 | 0.385 | 0.541 | 1.01 | 0.488 | 0.308 | 0.474 | 1.58 |
| PQ_gly5 | 1861 | no | 0.37 | 0.228 | 0.227 | 1.63 | 0.155 | 0.0968 | 0.0966 | 1.61 | 0.239 | 0.0873 | 0.0852 | 2.80 |
| PQ_gly5 | 1861 | yes | 0.735 | 0.562 | 0.596 | 1.31 | 0.249 | 0.241 | 0.311 | 1.03 | 0.333 | 0.205 | 0.281 | 1.63 |
| PQ_gly10 | 3606 | no | 3.21 | 1.96 | 1.99 | 1.64 | 1.34 | 0.902 | 0.899 | 1.49 | 1.17 | 0.388 | 0.381 | 3.07 |
| PQ_gly10 | 3606 | yes | 5.85 | 4.4 | 4.58 | 1.33 | 2.02 | 1.99 | 2.43 | 1.02 | 1.56 | 1.14 | 1.63 | 1.36 |

Accuracy, the worst value over the three thread counts. The residual is
`max_j ‖A v_j − λ_j v_j‖₂ / max|λ|` and orthogonality is `max |VᵀV − I|`:

| matrix | n | n·ε | faer residual / orth. | dsyevd residual / orth. | dsyevr residual / orth. |
|---|---|---|---|---|---|
| S_gly5 | 379 | 8.3e-14 | 8.4e-15 / 1.1e-14 | 1.4e-15 / 3.1e-15 | 1.8e-15 / 3.8e-13 |
| S_gly30 | 2154 | 4.7e-13 | 5.6e-15 / 1.2e-14 | 1.4e-15 / 5.6e-15 | 1.8e-15 / 4.0e-12 |
| PQ_gly5 | 1861 | 4.1e-13 | 3.3e-15 / 2.2e-14 | 7.8e-16 / 4.8e-15 | 9.0e-16 / 7.0e-13 |
| PQ_gly10 | 3606 | 7.9e-13 | 4.5e-15 / 2.1e-14 | 9.4e-16 / 8.0e-15 | 8.9e-16 / 6.6e-12 |

The spectra are mild. S has λ in [1.33e-3, 7.07] at every size, so cc-pVDZ
on these chains has no near-linear dependence. (P|Q) has λ in
[5.5e-5, 1.04e3], and 77 (gly5) and 171 (gly10) of its eigenvalues fall
below 1e-6 of the largest. Every routine and thread count agrees on those
counts. None of the matrices puts an eigenvalue near a cutoff, so this
does not test a case where a threshold decision is marginal. A diffuse
basis (aug-cc-pVDZ) would.

## How

```sh
# 1. Matrices (inside the devcontainer, which has cargo): the packed lower
#    triangles from the two existing host generators.
cd experimental/rust
G=/path/to/main/.temp/gly   # the main checkout's; any XYZ works
M=../faer_blas_lapack/results/matrices
mkdir -p $M
for g in gly5 gly10 gly15 gly20 gly30; do
  cargo run --release -q --manifest-path s_overlap_host/Cargo.toml -- $G/$g.xyz cc-pvdz $M/S_$g.npy
done
for g in gly5 gly10; do
  cargo run --release -q --manifest-path df_metric/Cargo.toml -- $G/$g.xyz cc-pvdz-jkfit $M/PQ_$g.npy
done

# 2. Both images, every matrix at 1, 4, 12 threads (on the host: docker only).
experimental/faer_blas_lapack/run.sh          # THREADS="1 4" to narrow

# 3. The tables above (standard library only).
python3 experimental/faer_blas_lapack/compare.py
```

- **One crate, two images.** `eigh-bench` is built with exactly one of
  `--features faer` or `--features lapack`, and `main.rs` refuses both or
  neither. The harness (reading the input, warm-up, timing, the accuracy
  products, output) is shared, so only the diagonalizer call differs.
  `Dockerfile.faer` and `Dockerfile.lapack` use the same
  `rust:1.99.1-slim-trixie` base and rustc as the devcontainer, and the
  LAPACK image adds only Debian's `libopenblas-pthread-dev` (LP64,
  DYNAMIC_ARCH, which picked its Zen kernels here).
- **LAPACK is bound by hand.** `src/lapack_backend.rs` declares `dsyevd_`,
  `dsyevr_`, `dgemm_` and `openblas_set_num_threads` itself, with LP64
  integers and the hidden string-length arguments. This avoids the
  `lapack-sys` crate and its link-provider features. The host's own
  `libopenblas64` is ILP64, with `64_`-suffixed symbols, and would not link
  against those bindings. That is one reason the LAPACK build lives in an
  image.
- **faer is called at the low level.** `faer::linalg::evd::self_adjoint_evd`
  gets its scratch sized once, as LAPACK's workspace query does. The
  high-level `Mat::self_adjoint_eigen` allocates U, S and the scratch inside
  every call. Parallelism is `Par::Seq` at one thread and `Par::rayon(t)`
  otherwise, on a global rayon pool of exactly t threads. OpenBLAS gets
  `openblas_set_num_threads(t)`.
- **The input is a packed lower triangle**, as `s-overlap-host` and
  `df-metric-host` write it: 1-D, `n (n + 1) / 2`, `(p, q)` with `q <= p` at
  `p (p + 1) / 2 + q`. It is unpacked to a full column-major n × n. Both
  libraries read the lower triangle, and the residual product reads all of
  it. `src/npy.rs` reads and writes 1-D `<f8` itself, so neither image needs
  the `rust/` workspace in its build context.
- **The accuracy products use each image's own GEMM** (faer's `matmul`, or
  OpenBLAS's `dgemm`). This runs after timing, on the last run's output.
- **Outputs.** Each row goes to `results/run/<backend>.jsonl`, and each
  eigenvalue set to `results/run/eigvals/`. `compare.py` sets each one
  against `dsyevd`'s (with eigenvectors, one thread). `results/` is
  gitignored, matrices included (about 100 MB).
