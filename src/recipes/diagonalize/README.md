# Diagonalize

The symmetric eigendecomposition A = U diag(w) U^T on the GPU: one dense-solver `syevd` through WarpWraps, with its scratch and convergence flag taken from `DeviceResources`, so one source builds for CUDA and HIP.

## Module

`free_breakfast.recipes.diagonalize`, also re-exported by `free_breakfast.recipes`.

## Design

`diagonalize<T>(res, n, a, w)` overwrites the n×n column-major `a` with its eigenvectors (column j belongs to `w[j]`) and writes the ascending eigenvalues into `w`. Only the lower triangle of `a` is read. It is the one place the tree calls `syevd`: `orthogonalize` diagonalizes the overlap through it, and the SCF step the transformed Fock matrix F'.

**The caller owns `a` and `w`; `syevd`'s work is `res`'s.** The routine queries `syevd`'s size for n per call (a host computation), with the same eigenvector mode and triangle the call uses, takes that much of `res.device_workspace()`, and uses `res.info()` for the flag. Growing the workspace is the only allocation it can make, and a `res` that has already served an n this large makes none. Neither `a` nor `w` may point into the workspace (see `DeviceResources`' "One routine owns the device workspace at a time"). A caller that needs scratch live across the call, as `orthogonalize` does for U and the eigenvalues, holds it in its own buffers.

**One synchronization.** The call queues the solve on `res.stream()` and reads the flag through `res.fetch_info()`, which syncs the stream. A non-converged eigendecomposition is `ErrorKind::Numerical`; an `info < 0` (a rejected argument, so a bug in the caller) is `GpuRuntime`. The error names `diagonalize`, not the operand: a caller that wants it to say which matrix failed wraps it.

**The legacy syevd.** This calls the int-based `syevd`, not cuSOLVER's generic `Xsyevd` with its params object and separate host workspace. hipSOLVER has no generic eigensolver API at all, so the legacy one is the only `syevd` wwr offers on both backends.

**Device.** The call makes `res.dev_idx()` current for its duration (`ScopedDeviceIndex`) and restores the previous device.

## Public API

| Symbol | Role |
|---|---|
| `diagonalize<T>(res, n, a, w)` | `Status`: A = U diag(w) U^T, U into `a`, ascending eigenvalues into `w`. Instantiated for `float` and `double`. |

## Usage

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;
namespace fb = free_breakfast;

fb::DeviceResources res{0};

fb::Buffers::device<double> a(static_cast<size_t>(n) * n, res.device());  // filled with A
fb::Buffers::device<double> w(n, res.device());

fb::Status status = fb::diagonalize<double>(res, n, a.data(), w.data());  // a now holds U
```

## Tests

`test/gtest/diagonalize_gpu.cpp` covers an empty operand, the workspace growing from empty, a diagonal A (w its sorted diagonal, U the unit vectors up to sign), and a dense symmetric A in `double` and `float` (w ascending, Uᵀ U == I, U diag(w) Uᵀ == A).

## Files

| File | Role |
|------|------|
| `diagonalize.cppm` | Module interface: `diagonalize` |
| `diagonalize.cpp` | Implementation, and the `float`/`double` instantiations |
| `CMakeLists.txt` | The module `free_breakfast.recipes.diagonalize` |
