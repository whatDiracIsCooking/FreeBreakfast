# Linalg

Dense linear algebra on the GPU, built over WarpWraps' BLAS and dense-solver dispatch (`gemm<T>`, `syevd<T>`, …, re-exported by `free_breakfast.core.wwr`). Each sub-directory is one module. `linalg.cppm` (`free_breakfast.linalg`) re-exports all of them, so one import covers the layer, and each sub-module stays importable on its own.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `orthogonalize/` | `free_breakfast.linalg.orthogonalize` | The symmetric (Löwdin) orthogonalizer X = S^{-1/2}, with every intermediate caller-owned. |

Routines take a `DeviceResources` (`free_breakfast.core.resources`) for their stream and library handles, so the BLAS and solver calls are ordered on one stream by construction. Each routine returns `Result`/`Status`.

## Tests

GPU, in `test/gtest/<name>_gpu.cpp`, linked into `free_breakfast_tests_gpu`.

## Files

| File | Role |
|------|------|
| `linalg.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration: `add_subdirectory` per module, and the `free_breakfast.linalg` module target |
