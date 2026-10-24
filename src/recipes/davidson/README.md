# Davidson

Block Davidson for the lowest eigenpairs of a symmetric operator known only as a [`LinearOperator`](../linear_operator/README.md): one dense matrix in a test, a CIS or CASCI sigma build in a calculation. Ported from calaman's `calaman.davidson`, Euclidean path only.

## Module

`free_breakfast.recipes.davidson`, also re-exported by `free_breakfast.recipes`. It re-exports `free_breakfast.recipes.linear_operator`.

## Design

`davidson_solve<T>(res, op, precondition, n, n_roots, max_subspace, guess, guess_count, eigenvalues, eigenvectors, options)` converges the `n_roots` lowest eigenpairs of `op` from `guess`'s orthonormal columns. Each iteration runs these steps:

1. **Expand A V over the new columns only.** `op.apply` runs once per iteration, on the columns V gained since the last, so the operator is never reapplied to a direction already in the subspace. Vᵀ A V gains those columns' rows, V_newᵀ (A V), by one gemm.
2. **Rayleigh–Ritz.** H = Vᵀ A V copied out packed at the current dimension, then `diagonalize` (syevd), which syncs the stream.
3. **Ritz pairs.** X = V S and A X = (A V) S from the same leading columns of S, so A is never applied to X.
4. **Residuals.** R = A X − X diag(θ). A root converges when ‖rᵢ‖ ≤ `residual_tolerance` · max(|θᵢ|, ‖H‖₂), ‖H‖₂ being max |θ| over the subspace spectrum.
5. **Corrections.** `precondition(res, theta, residual, correction)` maps the residual block to a correction block. The diagonal correction r / (θ − diag(A)) is the usual choice, and it is the caller's, because only the caller knows the diagonal.
6. **Collapse.** When a full new block would not fit, V and A V reset to X and A X, and Vᵀ A V to Xᵀ A X = Sᵀ H S = diag(θ). `max_subspace ≥ 2 · n_roots` leaves room for that block afterwards.
7. **Re-orthogonalization.** The corrections are projected off V twice (block Gram–Schmidt, since one pass loses orthogonality as V grows). Then Gram–Schmidt runs among the corrections of the unconverged roots, each projected off the ones already accepted by two gemvs, twice. A correction at or below `linear_dependence_floor` is dropped.

**Non-convergence is a value-tier error**, as `docs/style.md` "Errors" has it, not a success with a reason attached:

| Outcome | Return |
|---|---|
| Every root converged | `DavidsonInfo`: iterations, final max residual norm |
| `max_iterations` ran out | `MaxIterationsExceeded` |
| No correction survived re-orthogonalization, so the subspace cannot grow | `Divergence` |
| A shape outside the documented bounds | `InvalidConfig` |
| A BLAS, runtime, `op` or `precondition` failure | that error, passed up |

After `MaxIterationsExceeded` or `Divergence`, `eigenvalues` and `eigenvectors` hold the last iteration's Ritz pairs, for a caller that wants to log or restart from them.

**None of the subspace is the device workspace.** V, A V, H, the Ritz values and the residual, correction and projection blocks are held across `op.apply`, `precondition` and `diagonalize`, and each of those may take `res.device_workspace()`. A CIS sigma build will want it for J/K scratch. A workspace growth frees the old block, and even without one, the callee's scratch lands on the same bytes. So these are the call's own pool buffers, allocated on entry (`docs/style.md` "Scratch memory", "Scratch held across a call to another workspace user"). syevd's work is still the workspace's: `diagonalize` takes it and is finished with it before the next callback.

**Host pointer mode for the whole call.** The kept corrections' `nrm2`s write host results and every scalar is a host value, so `res.blas()` is held in host pointer mode by `ScopedPointerMode` and restored on return; only the residual norms are taken in a scoped device mode. `op` and `precondition` run in that mode too, which is also the handle's default.

**Vᵀ A V grows by rows, not recomputed.** A column of V and of A V never changes until a collapse, so neither does any element of Vᵀ A V already formed. Each iteration forms only the new columns' rows, V_newᵀ (A V): with the rows before them they hold every element of its lower triangle, which is all syevd reads. Recomputing the whole of it was a gemm with a dim × dim output over a reduction of length n, a shape BLAS runs far below peak, and at CASCI sizes it was a third of the 3080's kernel time (see `../../ci/solver/`). The incremental form costs dim · k instead of dim² per iteration, k the new columns. It is held at leading dimension `max_subspace` in its own buffer, because syevd overwrites the copy it is handed with S.

**Few host round trips.** Each iteration syncs for the Ritz values, right after `diagonalize` has already synced, so on an idle stream; once for every residual norm, written to the device by `nrm2` in device pointer mode and read back together; and once per kept correction, for its norm. The Gram–Schmidt overlaps among the corrections are gemvs into a device vector, so they never come to the host. Before, each `nrm2` and `dot` was its own round trip: with three unconverged roots, 11 syncs an iteration, `diagonalize`'s included, against 6 now.

**No metric path.** calaman's generalized (`sygvd`, M-inner-product) problem was left out because nothing here needs it yet: CIS and CASCI are standard symmetric problems in an orthonormal basis. It enters as a defaulted callback, so adding it back (for instance for the product form of TDHF/RPA) adds an argument and breaks no call site.

**Device.** The call makes `res.dev_idx()` current for its duration and restores the previous device.

## Public API

| Symbol | Role |
|---|---|
| `davidson_solve<T>(…)` | `Result<DavidsonInfo>`: the `n_roots` lowest eigenpairs of `op`. Any `real_fp` `T`; a template instantiated by the caller. |
| `DavidsonOptions` | `residual_tolerance` (1e-8), `max_iterations` (100), `linear_dependence_floor` (1e-12) |
| `DavidsonInfo` | `iterations`, `max_residual_norm` |
| `DavidsonPreconditioner<F, T>` | `f(res, theta, residual, correction)` returns something convertible to `Status`; `theta` is a host `std::span<const T>` of the current Ritz values, one per column |

## Usage

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;
namespace fb = free_breakfast;

fb::DeviceResources res{0};
Sigma op{...};                     // any fb::LinearOperator<Sigma, double>
const auto precondition = [&](fb::DeviceResources &r, std::span<const double> theta,
                              const double *residual, double *correction) -> fb::Status { ... };

// guess: n x n_roots, orthonormal columns (unit vectors on the lowest diagonal entries)
fb::Buffers::device<double> values(n_roots, res.device());
fb::Buffers::device<double> vectors(static_cast<size_t>(n) * n_roots, res.device());
const auto info = fb::davidson_solve<double>(res, op, precondition, n, n_roots, 8 * n_roots,
                                             guess.data(), n_roots, values.data(), vectors.data());
```

## Tests

`test/gtest/davidson_gpu.cpp`, against `diagonalize`'s full spectrum of a dense diagonally dominant A held as a one-gemm operator. It covers:

- the lowest pairs in `double` and `float`: the values, unit-norm vectors, and host-recomputed residuals;
- convergence at the smallest subspace, `2 · n_roots`, which collapses every iteration;
- one apply per iteration, every apply made in host pointer mode;
- an operator that grows and overwrites the device workspace on every apply;
- `MaxIterationsExceeded` with the first step's Ritz values written, and `Divergence` from zero corrections;
- shape rejection;
- the concepts, accepting and rejecting models.

## Files

| File | Role |
|------|------|
| `davidson.cppm` | Module interface: `davidson_solve`, its options, info and preconditioner concept |
| `CMakeLists.txt` | The module `free_breakfast.recipes.davidson` |
