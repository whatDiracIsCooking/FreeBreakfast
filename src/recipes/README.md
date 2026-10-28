# Recipes

Dense linear algebra on the GPU, built over WarpWraps' BLAS and dense-solver dispatch (`gemm<T>`, `syevd<T>`, …, re-exported by `free_breakfast.core.wwr`). Each sub-directory is one module. `recipes.cppm` (`free_breakfast.recipes`) re-exports all of them, so one import covers the layer, and each sub-module stays importable on its own.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `blas/` | `free_breakfast.recipes.blas` | BLAS conveniences over wwr's dispatch: `gemm` for packed operands, and the sandwiches `gemmwich` (`Xᵀ F X`) and `gmmwich` (`Xᵀ diag(d) X`). |
| `cg_unitary/` | `free_breakfast.recipes.cg_unitary` | Riemannian conjugate gradient on U(n) (O(n) for a real T), with the paper's polynomial and DFT geodesic line searches (`cg_unitary`), over a caller's `UnitaryCostFunction`. |
| `cholesky/` | `free_breakfast.recipes.cholesky` | The Cholesky factorization A = L Lᵀ in place, blocked over a shared-memory diagonal-block kernel, trsm and syrk (`cholesky`): `potrf` with its flag, without rocSOLVER's small-matrix race. |
| `complex_cast/` | `free_breakfast.recipes.complex_cast` | Split a complex matrix into real and imaginary planes (`split_planes`), and merge them back (`merge_planes`). |
| `davidson/` | `free_breakfast.recipes.davidson` | Block Davidson for the lowest eigenpairs of a symmetric `LinearOperator` (`davidson_solve`). |
| `diagonalize/` | `free_breakfast.recipes.diagonalize` | The symmetric eigendecomposition A = U diag(w) Uᵀ, one `syevd` with its work and flag from `DeviceResources`. |
| `diis/` | `free_breakfast.recipes.diis` | Pulay DIIS extrapolation over a device-resident ring of iterates and residuals. |
| `einsum/` | `free_breakfast.recipes.einsum` | A NumPy-style einsum, accumulating `alpha` times the contraction into its output (`einsum`), and the transpose as its one-operand case (`transpose`): one generic kernel driven by the subscript string, up to 3 operands. |
| `expm/` | `free_breakfast.recipes.expm` | The exponential of a skew-Hermitian H, exp(tH) = U diag(e^{-itw}) U^H, from one Hermitian eigendecomposition of iH (`skew_eig`) and evaluated at any t (`skew_expm`). |
| `linear_operator/` | `free_breakfast.recipes.linear_operator` | The `LinearOperator` concept: a matrix known only by `op.apply(res, k, x, y)`, Y = A X. |
| `matrix_ao/` | `free_breakfast.recipes.matrix_ao` | AO-basis matrices tagged Cartesian or spherical, and a pair holding both that converts through a shared transform. |
| `norm/` | `free_breakfast.recipes.norm` | A chosen norm (`Norm::l1`, `l2`, `inf`) of a real device vector, in either pointer mode (`norm`). |
| `orthogonalize/` | `free_breakfast.recipes.orthogonalize` | The symmetric (Löwdin) orthogonalizer X = S^{-1/2}, and `orthonormalize`, a block of vectors orthonormalized in place with dependent columns dropped, both over `diagonalize`. |
| `polynomial_roots/` | `free_breakfast.recipes.polynomial_roots` | Every root of a batch of polynomials by the Aberth-Ehrlich iteration, one warp per polynomial and one lane per root (`polynomial_roots`). |
| `sort/` | `free_breakfast.recipes.sort` | Stable merge sort of `double` keys in IEEE-754 total order, optionally carrying `uint32_t` values (`merge_sort`), and the column gather that applies its permutation to a matrix (`gather_columns`). Its device library is also linked directly by the integral builds' kernels. |

Routines take a `DeviceResources` (`free_breakfast.core.resources`) for their stream and library handles, so the BLAS and solver calls are ordered on one stream by construction. Each routine returns `Result`/`Status`. `blas/` is the exception: its building blocks take a BLAS handle and return the library status, the way `wwr::gemm` does. `einsum/` takes a bare stream instead: it needs no library handle, so the caller can issue it on any stream, and a refused call touches no device.

## Tests

GPU, in `test/gtest/<name>_gpu.cpp`, linked into `free_breakfast_tests_gpu`. `einsum`'s refusals need no device, so they are in `test/gtest/recipes_einsum_host.cpp`, in `free_breakfast_tests_host`.

## Files

| File | Role |
|------|------|
| `recipes.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.recipes` module target |
