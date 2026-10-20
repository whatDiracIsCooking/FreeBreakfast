# Linalg

Dense linear algebra on the GPU, built over WarpWraps' BLAS and dense-solver dispatch (`gemm<T>`, `syevd<T>`, …, re-exported by `free_breakfast.core.wwr`). Each sub-directory is one module. `linalg.cppm` (`free_breakfast.linalg`) re-exports all of them, so one import covers the layer, and each sub-module stays importable on its own.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `blas/` | `free_breakfast.linalg.blas` | BLAS conveniences over wwr's dispatch: `gemm` for square operands, and the sandwiches `gemmwich` (`Xᵀ F X`) and `gmmwich` (`Xᵀ diag(d) X`). |
| `diis/` | `free_breakfast.linalg.diis` | Pulay DIIS extrapolation over a device-resident ring of iterates and residuals. |
| `matrix_ao/` | `free_breakfast.linalg.matrix_ao` | AO-basis matrices tagged Cartesian or spherical, and a pair holding both that converts through a shared transform. |
| `orthogonalize/` | `free_breakfast.linalg.orthogonalize` | The symmetric (Löwdin) orthogonalizer X = S^{-1/2}, its intermediates carved from `DeviceResources`' workspace. |

Routines take a `DeviceResources` (`free_breakfast.core.resources`) for their stream and library handles, so the BLAS and solver calls are ordered on one stream by construction. Each routine returns `Result`/`Status`. `blas/` is the exception: its building blocks take a BLAS handle and return the library status, the way `wwr::gemm` does.

## Tests

GPU, in `test/gtest/<name>_gpu.cpp`, linked into `free_breakfast_tests_gpu`.

## Files

| File | Role |
|------|------|
| `linalg.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.linalg` module target |
