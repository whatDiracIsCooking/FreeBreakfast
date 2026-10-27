# Orthogonalize

The symmetric (Löwdin) orthogonalizer X = S^{-1/2}, and `orthonormalize`, which orthonormalizes a block of vectors in place and drops the dependent ones (below), on the GPU. The orthogonalizer is an eigendecomposition (`diagonalize`, one dense-solver `syevd`), the diagonal sandwich `gmmwich` (a BLAS `dgmm` and `gemm`), and a small `inverse_sqrt` kernel, through WarpWraps so one source builds for CUDA and HIP.

## Module

`free_breakfast.recipes.orthogonalize`, also re-exported by `free_breakfast.recipes`.

## Design

Given a real-symmetric, positive-(semi)definite operand S, it builds

```
X = S^{-1/2} = U * Lambda^{-1/2} * U^T
```

where S = U Λ Uᵀ is the symmetric eigendecomposition (the columns of U are eigenvectors, Λ the ascending eigenvalues). For the SCF overlap this turns the generalized eigenproblem `F C = S C ε` into a standard one: with `F' = Xᵀ F X` and `C = X C'`, `F' C' = C' ε`. Nothing here is specific to SCF. A density-fitting Coulomb metric or a localization overlap goes through the same eigendecomposition, eigenvalue map and recombination, which is why this lives in `recipes`.

The computation, all queued on `res.stream()`:

1. `diagonalize(S)` (`free_breakfast.recipes.diagonalize`) → ascending eigenvalues λ_j and eigenvectors U. It overwrites its input, so S is first copied into U, and S itself is left untouched.
2. `inverse_sqrt` kernel: Λ ← diag(λ_j^{-1/2}), in place, with eigenvalues at or below `eigenvalue_floor` set to 0 (see Conditioning).
3. X = U diag(Λ^{-1/2}) Uᵀ, as one `gmmwich<WWRBLAS_OP_T>(res.blas(), n, U, Λ^{-1/2}, M, X)` (`free_breakfast.recipes.blas:gmmwich`), which is two library calls:
   - M = U diag(Λ^{-1/2}) (`dgmm`, right side) into M: scales column j of U by λ_j^{-1/2}, in O(n²) where a `gemm` against a dense diagonal would be O(n³);
   - X = M Uᵀ (`gemm`, `transb = T`), symmetric by construction.

Every matrix is a raw column-major device pointer with packed leading dimension n. There is no matrix type in between.

**The caller passes S and X; U and the eigenvalues are the routine's own, M is `res`'s.** U and the eigenvalues are `diagonalize`'s outputs and live to the end, held across a call that takes `res.device_workspace()` for syevd's work. So they cannot be carved from the workspace too: `diagonalize`'s request would hand back the same block, or grow it and free theirs. They are pool buffers on `res.device()` (`Buffers::device`), allocated on `res.stream()` and freed there after the gmmwich that reads them. M is pure scratch first written after `diagonalize` returns, so it is the workspace's, requested then for n² elements of T. syevd's work is usually larger, so that request rarely grows it. Neither S nor X may point into the workspace (see `DeviceResources`' "One routine owns the device workspace at a time").

**Two host waits, one of them on queued work.** `diagonalize` syncs the stream to read syevd's flag. A non-converged eigendecomposition becomes `ErrorKind::Numerical` instead of quietly corrupting X; an `info < 0` (a rejected argument, so a bug here) is `GpuRuntime`. The routine then copies back the extreme eigenvalues (first and last, since they are ascending) before step 2 overwrites them, and syncs again; the stream is idle by then, so that wait is two scalar copies. The read logs cond(S) = λ_max / λ_min, at `debug` always and at `warn` above 1e10 or for a non-positive λ_min. 1e10 is the threshold GPU4PySCF's `check_linear_dependency` warns at. Steps 2-3 never synchronize.

**Conditioning.** The kernel *floors*: an eigenvalue at or below `eigenvalue_floor` (default `kOrthogonalizeEigenvalueFloor<T>`, 1e-10) contributes 0 instead of λ^{-1/2}, which drops that mode. The test comes before the square root, so X is free of NaN and Inf even for a zero or negative eigenvalue. The trade-off: `Xᵀ S X == I` only when every eigenvalue clears the floor (a positive-definite, well-conditioned S). For a singular or indefinite S the dropped modes make `Xᵀ S X` a rank-deficient projector, which is expected, because such an S has no true S^{-1/2}.

**Not canonical orthogonalization.** The floor drops near-null modes, but X stays square (n×n). Canonical orthogonalization returns a rectangular n_ao × n_mo X. For a well-conditioned S nothing is dropped and X is exactly the symmetric root. The rectangular variant belongs in a separate entry.

**Device.** The call makes `res.dev_idx()` current for its duration (`ScopedDeviceIndex`) and restores the previous device, because library calls and kernel launches run on whichever device the calling thread has current.

## orthonormalize

`orthonormalize<T>(res, n, k, v, floor)` orthonormalizes the n × k device block `v` in place, compacts the kept columns to the front, and returns how many it kept. With `DavidsonOptions::canonical_orthonormalization`, `davidson_solve` (`../davidson/`) orthonormalizes the corrections it appends to V with it, and `ci::solve`'s `ci_guesses` (`../../ci/solver/`) its guesses, in place of their Gram–Schmidt. The option is off by default; the davidson README has why. Canonical orthogonalization, then a second pass. One pass over m columns:

1. G = Vᵀ V, m × m, by one `gemv` per column. Neither `syrk` nor `gemm`: for an m × m output over n ≈ 10⁵ rows, cuBLAS's syrk tiles only the output, with no split over the reduction, and took 10–20 ms a call on the 3080 at CAS(10,10) (`ci::solve` 6–9× slower); its gemm runs one output tile at about 200 µs, which left `ci::solve` 5–7% slower than Gram–Schmidt. A gemv is about 4 µs.
2. D = diag(G)^{-1/2} by the `inverse_sqrt` kernel, with a column of norm at or below `floor` scaled to zero, and G' = D G D by two `dgmm`s: the Gram matrix of the unit-normalized columns.
3. G' = U Λ Uᵀ by `diagonalize`, which syncs; Λ comes back to the host on the idle stream. The kept modes are the top of the ascending spectrum: λ > max(floor², m ε λ_max).
4. U_kept ← D U_kept Λ_kept^{-1/2} (`inverse_sqrt` on the kept λ, two `dgmm`s), V U_kept by `gemm` into the workspace, copied back over V's leading columns.

Then a second pass, since one leaves Qᵀ Q = I + E with ‖E‖ of order κ ε, κ = λ_max / λ_min over the kept modes:

5. When κ ε ≤ 10⁻² √ε (κ up to 7·10⁵ in `double`, 29 in `float`), no mode can drop again, and one Newton–Schulz step, Q ← Q (3 I − Qᵀ Q) / 2, takes E to O(E²), which is rounding. It is m gemvs for Qᵀ Q, a gemm for 3/2 Q − 1/2 Q (Qᵀ Q) into the product Q is still in, and a copy, with no sync and no syevd.
6. Otherwise steps 1–4 again, over the kept columns.

**Why the Newton–Schulz step.** Two canonical passes cost `davidson_solve` two syevds and four syncs an iteration, and made `ci::solve` about 7% slower than Gram–Schmidt at CAS(10,10) on the 3080; one pass alone was even with it. Davidson's correction blocks are seldom near-dependent once projected off V: over a CAS(10,10) solve κ had a median of 16 and a 99th percentile of 2.6·10⁴, so in `double` the step is nearly always the polish. The tests' near-duplicate block (κ ≈ 4·10⁸) takes the canonical second pass, so both are exercised.

**Why the unit diagonal.** λ is a squared norm along a mode, and syevd resolves it only to about ε λ_max. Davidson's corrections differ in norm by the ratio of their roots' residuals, easily 10⁶ near convergence, so in G itself the short column's mode sits at 10⁻¹² λ_max with an error near its own size, and its direction would come out as noise. Scaled to unit columns first, the modes measure dependence alone. For columns of comparable norm the floor is still a norm, as `linear_dependence_floor` was under Gram–Schmidt, and a column of norm at or below it is dropped outright, as Gram–Schmidt's norm test dropped it.

**Why the m ε λ_max cut.** A Gram matrix cannot see a direction below √ε of the columns: an exact duplicate comes out of syevd as λ ≈ ε, not 0. Kept, that mode is rounding noise scaled up by λ^{-1/2}, a column of no meaning and orthogonal to nothing. The cut drops it, so a floor below √ε (Davidson's 1e-12) still drops exactly dependent columns; a floor above it, as in the tests, is the cut that decides.

**Rejected:** Householder QR (wwr wraps `geqrf` but not `orgqr`, and unpivoted QR cannot drop a dependent column) and CholQR2 (`potrf` + `trsm`: fine for `ci_guesses`' near-orthonormal columns, fails on a rank-deficient block). One algorithm serves both.

**Buffers.** G (then U), Λ and D are held across `diagonalize`, which takes the workspace for syevd, so they are one pool buffer of the call's own; the n × kept gemm product is the workspace's, carved by `device_regions` after `diagonalize` returns. `res.blas()` is held in host pointer mode for the call and restored. Each canonical pass synchronizes twice, both on the stream `diagonalize` has just drained; the Newton–Schulz step not at all.

## Public API

| Symbol | Role |
|---|---|
| `kOrthogonalizeEigenvalueFloor<T>` | The default floor, 1e-10. |
| `orthogonalize<T>(res, n, s, x, floor)` | `Status`: X = S^{-1/2} into `x`. Instantiated for `float` and `double`. |
| `orthonormalize<T>(res, n, k, v, floor)` | `Result<int32_t>`: `v`'s columns orthonormalized in place, the kept count returned. Instantiated for `float` and `double`. |

## Usage

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;
namespace fb = free_breakfast;

fb::DeviceResources res{0};

const auto nn = static_cast<size_t>(n) * n;
fb::Buffers::device<double> s(nn, res.device());  // filled with the operand
fb::Buffers::device<double> x(nn, res.device());

fb::Status status = fb::orthogonalize<double>(res, n, s.data(), x.data());
```

## Tests

`test/gtest/orthogonalize_gpu.cpp` covers an empty operand, one `res` growing its workspace across a small `float` and a larger `double` operand, a diagonal S (X is the elementwise inverse root), a dense well-conditioned S in `double` and `float` (X symmetric, `Xᵀ S X == I`, S unchanged), and a singular S (X finite, and `Xᵀ S X` an idempotent of trace 2).

`test/gtest/orthonormalize_gpu.cpp` covers, in `double` and `float`: a random tall block (Qᵀ Q = I and span(Q) = span(V), to 1e-13 in `double`); a block with an exact duplicate, a near duplicate above the floor, one below it and a column below it, which keeps exactly the four directions above the floor; a column 10¹² shorter than its neighbour, kept with its direction; one column, kept normalized or dropped below the floor; no columns; and a negative count rejected.

## Files

| File | Role |
|------|------|
| `orthogonalize.cppm` | Module interface: `orthogonalize`, the default floor, and `orthonormalize` |
| `orthogonalize.cpp` | Both implementations, the cond(S) log, and the `float`/`double` instantiations |
| `inverse_sqrt.cu` | The `inverse_sqrt` kernel, over wwr's `parallel_for` |
| `inverse_sqrt_bridge.h` | The launcher's declaration, shared by the `.cu` and the implementation unit's GMF |
| `CMakeLists.txt` | The device library `free_breakfast.recipes.orthogonalize.device` and the module |
