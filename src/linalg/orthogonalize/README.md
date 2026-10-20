# Orthogonalize

The symmetric (Löwdin) orthogonalizer X = S^{-1/2}, on the GPU: a dense-solver `syevd`, the diagonal sandwich `gmmwich` (a BLAS `dgmm` and `gemm`), and a small `inverse_sqrt` kernel, through WarpWraps so one source builds for CUDA and HIP.

## Module

`free_breakfast.linalg.orthogonalize`, also re-exported by `free_breakfast.linalg`.

## Design

Given a real-symmetric, positive-(semi)definite operand S, it builds

```
X = S^{-1/2} = U * Lambda^{-1/2} * U^T
```

where S = U Λ Uᵀ is the symmetric eigendecomposition (the columns of U are eigenvectors, Λ the ascending eigenvalues). For the SCF overlap this turns the generalized eigenproblem `F C = S C ε` into a standard one: with `F' = Xᵀ F X` and `C = X C'`, `F' C' = C' ε`. Nothing here is specific to SCF. A density-fitting Coulomb metric or a localization overlap goes through the same eigendecomposition, eigenvalue map and recombination, which is why this lives in `linalg`.

The computation, all queued on `res.stream()`:

1. `syevd(S)` → ascending eigenvalues λ_j and eigenvectors U. syevd overwrites its input, so S is first copied into U, and S itself is left untouched.
2. `inverse_sqrt` kernel: Λ ← diag(λ_j^{-1/2}), in place, with eigenvalues at or below `eigenvalue_floor` set to 0 (see Conditioning).
3. X = U diag(Λ^{-1/2}) Uᵀ, as one `gmmwich(res.blas(), WWRBLAS_OP_T, n, U, Λ^{-1/2}, M, X)` (`free_breakfast.linalg.blas:gmmwich`), which is two library calls:
   - M = U diag(Λ^{-1/2}) (`dgmm`, right side) into M: scales column j of U by λ_j^{-1/2}, in O(n²) where a `gemm` against a dense diagonal would be O(n³);
   - X = M Uᵀ (`gemm`, `transb = T`), symmetric by construction.

Every matrix is a raw column-major device pointer with packed leading dimension n. There is no matrix type in between.

**The caller passes S and X; every intermediate is `res`'s.** U, M, the eigenvalues and syevd's workspace are all carved from `res.device_workspace()`, and syevd's flag is `res.info()`. The routine queries syevd's size for n, with the same eigenvector mode and triangle the call uses, then asks for the whole layout in one request:

```
[ U: n² | eigenvalues: n | syevd's work: lwork, then M: n² ]
```

U and the eigenvalues are syevd's outputs and live to the end. M is first written by gmmwich, after syevd is done with its work on the same stream, so the two share a region: n² + n + max(lwork, n²) elements of T, each region rounded up to `kWorkspaceAlignment`. Growing the workspace to that is the only allocation the routine can make, and a `res` that has already served an n this large makes none. The workspace then stays with `res` for the next solver call, such as the SCF loop's own syevd of the same n. Neither S nor X may point into the workspace (see `DeviceResources`' "One routine owns the device workspace at a time").

**One host round-trip.** After syevd, the routine copies back the extreme eigenvalues (first and last, since they are ascending) and then the convergence flag through `res.fetch_info()`, whose stream sync completes all three: one synchronization. A non-converged eigendecomposition becomes `ErrorKind::Numerical` instead of quietly corrupting X; an `info < 0` (a rejected argument, so a bug here) is `GpuRuntime`. The same read logs cond(S) = λ_max / λ_min, at `debug` always and at `warn` above 1e10 or for a non-positive λ_min. 1e10 is the threshold GPU4PySCF's `check_linear_dependency` warns at. Steps 2-3 never synchronize.

**Conditioning.** The kernel *floors*: an eigenvalue at or below `eigenvalue_floor` (default `kOrthogonalizeEigenvalueFloor<T>`, 1e-10) contributes 0 instead of λ^{-1/2}, which drops that mode. The test comes before the square root, so X is free of NaN and Inf even for a zero or negative eigenvalue. The trade-off: `Xᵀ S X == I` only when every eigenvalue clears the floor (a positive-definite, well-conditioned S). For a singular or indefinite S the dropped modes make `Xᵀ S X` a rank-deficient projector, which is expected, because such an S has no true S^{-1/2}.

**Not canonical orthogonalization.** The floor drops near-null modes, but X stays square (n×n). Canonical orthogonalization returns a rectangular n_ao × n_mo X. For a well-conditioned S nothing is dropped and X is exactly the symmetric root. The rectangular variant belongs in a separate entry.

**The legacy syevd.** This calls the int-based `syevd`, not cuSOLVER's generic `Xsyevd` with its params object and separate host workspace. hipSOLVER has no generic eigensolver API at all, so the legacy one is the only `syevd` wwr offers on both backends.

**Device.** The call makes `res.dev_idx()` current for its duration (`ScopedDeviceIndex`) and restores the previous device, because library calls and kernel launches run on whichever device the calling thread has current.

## Public API

| Symbol | Role |
|---|---|
| `kOrthogonalizeEigenvalueFloor<T>` | The default floor, 1e-10. |
| `orthogonalize<T>(res, n, s, x, floor)` | `Status`: X = S^{-1/2} into `x`. Instantiated for `float` and `double`. |

## Usage

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.linalg;
namespace fb = free_breakfast;

fb::DeviceResources res{0};

const auto nn = static_cast<size_t>(n) * n;
fb::Buffers::device<double> s(nn, res.device());  // filled with the operand
fb::Buffers::device<double> x(nn, res.device());

fb::Status status = fb::orthogonalize<double>(res, n, s.data(), x.data());
```

## Tests

`test/gtest/orthogonalize_gpu.cpp` covers an empty operand, one `res` growing its workspace across a small `float` and a larger `double` operand, a diagonal S (X is the elementwise inverse root), a dense well-conditioned S in `double` and `float` (X symmetric, `Xᵀ S X == I`, S unchanged), and a singular S (X finite, and `Xᵀ S X` an idempotent of trace 2).

## Files

| File | Role |
|------|------|
| `orthogonalize.cppm` | Module interface: `orthogonalize` and the default floor |
| `orthogonalize.cpp` | Implementation, the cond(S) log, and the `float`/`double` instantiations |
| `inverse_sqrt.cu` | The `inverse_sqrt` kernel, over wwr's `parallel_for` |
| `inverse_sqrt_bridge.h` | The launcher's declaration, shared by the `.cu` and the implementation unit's GMF |
| `CMakeLists.txt` | The device library `free_breakfast.linalg.orthogonalize.device` and the module |
