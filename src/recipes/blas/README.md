# BLAS

Three BLAS conveniences over WarpWraps' typed dispatch (`wwr::gemm<T>`,
`wwr::dgmm<T>`, re-exported by [`free_breakfast.core.wwr`](../../core/wwr/README.md)):

- `gemm`, `wwr::gemm` for square, contiguous operands;
- `gemmwich`, the sandwich product `Xᴴ F X` (and its general rectangular form)
  as two gemms;
- `gmmwich`, the same sandwich with a diagonal middle, `Xᴴ diag(d) X`, as a
  dgmm and a gemm.

`ᴴ` is the adjoint of the element type: the conjugate transpose on a complex
`T`, the plain transpose on a real one, where the sandwiches are `Xᵀ F X` and
`Xᵀ diag(d) X`. All three take any `wwr::usual_fp` (float, double and the two
complex types).

They fix arguments and compose calls; they do not rename them. `gemm` forwards
to `wwr::gemm` with arguments filled in, and every other gemm still calls
`wwr::gemm` itself. `gemmwich` is two `wwr::gemm` calls under wwr's
`ScopedPointerMode`, and `gmmwich` the same composition with a `wwr::dgmm` in
place of the first gemm.

Unlike the rest of `recipes`, these take a BLAS handle rather than a
`DeviceResources` and return the library's `wwrblasStatus_t` rather than a
`Status`: they are building blocks, called the way `wwr::gemm` is.

## Module

`free_breakfast.recipes.blas`, target `free_breakfast::recipes::blas`, also
re-exported by `free_breakfast.recipes`. Four partitions under the primary
interface, all re-exported:

| Partition | Exports |
|-----------|---------|
| `:shared` | nothing: the helpers below |
| `:gemm` | `gemm` |
| `:gemmwich` | `gemmwich` (two overloads) |
| `:gmmwich` | `gmmwich` (two overloads) |

`:shared` is a private interface partition holding what the other three share,
unexported: `raw_blas_handle` unwraps a `BlasHandle` or passes a raw handle
through, and two concepts constrain the ops. `BlasOp<Op>` is `N`, `T` or `C`,
what `gemm` takes. `SandwichOp<Op, T>` is `N` or `kAdjointOp<T>`, `T`'s adjoint
in its one spelling, what the sandwiches take:

| `T` | `kAdjointOp<T>` | Left out, and why |
| --- | --- | --- |
| `float`, `double` | `WWRBLAS_OP_T` | `C`: on a real `T` it is `T` under another name |
| `wwrFloatComplex`, `wwrDoubleComplex` | `WWRBLAS_OP_C` | `T`: the sandwich's other factor would be `(Xᵀ)ᴴ = conj(X)`, which no BLAS op spells |

They sit in an interface partition, not an internal one, because the
exported templates that use them are instantiated in the importer's TU
(`docs/style.md`, "Module-private code").

Every op is a **template argument**, not a function argument, and each one not
given is `WWRBLAS_OP_N`. So an op outside the concept is a compile error rather
than a status, and the ops come first in the template parameter list, ahead of
the deduced `T` and handle type, so they can be spelled alone:
`gemmwich<WWRBLAS_OP_T>(…)`. For an op known only at run time, call
`wwr::gemm`.

```cpp
import free_breakfast.core.wwr;
import free_breakfast.recipes.blas;   // or free_breakfast.recipes
namespace fb = free_breakfast;
```

## gemm

`free_breakfast::gemm` is `wwr::gemm<T, int32_t>` for square, contiguous
operands: A, B and C all `n x n`, column-major, `m = n = k = n` and every
leading dimension `n`. One function:

| Call | Computes |
| --- | --- |
| `gemm<OpA, OpB>(handle, n, alpha, a, b, beta, c)` | `C = alpha*OpA(A)*OpB(B) + beta*C` |
| `gemm<OpA>(handle, n, alpha, a, b, beta, c)` | `C = alpha*OpA(A)*B + beta*C` |
| `gemm(handle, n, alpha, a, b, beta, c)` | `C = alpha*A*B + beta*C` |

`OpA` / `OpB` are `WWRBLAS_OP_N`, `_T` or `_C` (`BlasOp`; `_C` is the
conjugate transpose, the same as `_T` for a real `T`). `gemm<WWRBLAS_OP_T>` is
`AᵀB`: an op not given is `N`, not a copy of the one before it.

```cpp
fb::BlasHandle blas(dev);
FBF_GPU_MUST(fb::gemm(blas, n, &kOne<double>, a, b, &kZero<double>, c));
FBF_GPU_MUST(fb::gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, n, &kOne<double>, m, u,
                                                            &kZero<double>, x));
```

- `T` is any `wwr::usual_fp` (float, double and the two complex types); sizes
  are `int32_t`.
- `handle` is anything satisfying `wwr::extension::blas_handle`: a `BlasHandle`
  or a raw `wwrblasHandle_t`, so there is no `.get()` to write.
- `alpha` / `beta` are pointers, read as the handle's pointer mode says (host by
  default). There is deliberately no by-value overload: on a handle in device
  pointer mode it would hand the library a host address to read as a device
  one.
- It returns the `wwrblasStatus_t`, like `wwr::gemm`, so the caller picks
  `FBF_GPU_MUST` or `gpu_status`. Like every BLAS call it runs on the calling
  thread's current device (see `ScopedDeviceIndex` in `core/wwr`).
- The name is `wwr::gemm`'s too. They never collide: wwr's takes 14 arguments
  and is reached qualified, and this takes 7. Inside `free_breakfast`,
  write `wwr::gemm` for the general call and plain `gemm` for these.

For a rectangular shape or a padded leading dimension, call `wwr::gemm`.

## gemmwich

`gemmwich` is the sandwich product, two `wwr::gemm`s through a caller-supplied
work matrix. Two overloads, the square one the general one with `m = n`:

| Call | Computes |
| --- | --- |
| `gemmwich<OpX, OpF>(handle, m, n, x, f, work, c)` | `C = OpX(X)ᴴ OpF(F) OpX(X)` |
| `gemmwich<OpX, OpF>(handle, n, x, f, work, c)` | the same, `m = n` |
| `gemmwich(handle, n, x, f, work, c)` | `C = Xᴴ F X` |
| `gemmwich<kAdjointOp<T>>(handle, n, x, f, work, c)` | `C = X F Xᴴ` |

The last row is `gemmwich<WWRBLAS_OP_T>` on a real `T` and
`gemmwich<WWRBLAS_OP_C>` on a complex one (`kAdjointOp` itself is not
exported). In the general form `OpX(X)` is `n x m`, F is `n x n` and C is
`m x m`, through an `n x m` work matrix: `work = OpF(F) OpX(X)`, then
`C = OpX(X)ᴴ work`.
Every matrix is column-major and contiguous, so its leading dimension is its
stored row count:

| Matrix | Stored as | Leading dimension |
| --- | --- | --- |
| X, `OpX = N` | `n x m` | `n` |
| X, `OpX` = the adjoint | `m x n` | `m` |
| F | `n x n` | `n` |
| work | `n x m` | `n` |
| C | `m x m` | `m` |

`OpX = N` with X stored `n x m` is `Xᴴ OpF(F) X`, the projection of F onto
X's `m` columns. `OpX` and `OpF` are each `WWRBLAS_OP_N` or the adjoint
(`SandwichOp`).

```cpp
fb::Buffers::device<double> work(n * n, dev);
FBF_GPU_MUST(fb::gemmwich(blas, n, x, f, work.data(), c));         // C = X^T F X

fb::Buffers::device<double> rect_work(n * m, dev);                  // X stored n x m
FBF_GPU_MUST(fb::gemmwich(blas, m, n, x, f, rect_work.data(), c));  // C (m x m) = X^T F X
```

- `T` is any `wwr::usual_fp`. On a complex `T` the sandwich is `Xᴴ F X`, the
  one that keeps a Hermitian F Hermitian; the complex-symmetric `Xᵀ F X` is not
  offered.
- `work` is scratch, ignored on entry and left holding `OpF(F) OpX(X)`. It
  must not alias X, F or C, and C must not alias X or F. C is overwritten
  (`beta = 0`), not accumulated into.
- The scalars are `kOne` / `kZero` in host memory (`1 + 0i` / `0` on a
  complex `T`; see `core/constants`), so the handle is held in
  host pointer mode for the call by wwr's `ScopedPointerMode` (bound to
  `CreatePolicy`) and handed back in whatever mode it had. A failure to read or
  set the mode means a bad handle and aborts, as `ScopedDeviceIndex` does.
- It returns the first failing gemm's `wwrblasStatus_t`; there is no host
  synchronisation between the two gemms.

## gmmwich

`gmmwich` is `gemmwich` with F = diag(d), where the diagonal is never formed:
a `wwr::dgmm` scales X in O(nm), where a gemm against a dense diagonal would
be O(n²m), and one `wwr::gemm` finishes. There is no `OpF`, since a diagonal
is its own transpose. Two overloads, as for `gemmwich`:

| Call | Computes |
| --- | --- |
| `gmmwich<OpX>(handle, m, n, x, d, work, c)` | `C = OpX(X)ᴴ diag(d) OpX(X)` |
| `gmmwich<OpX>(handle, n, x, d, work, c)` | the same, `m = n` |
| `gmmwich(handle, n, x, d, work, c)` | `C = Xᴴ diag(d) X` |

`OpX(X)` is `n x m`, d has `n` elements (stride 1) and C is `m x m`. `work`
always has X's stored shape, because the diagonal goes on whichever side of X
needs no adjoint:

| `OpX` | X stored as | `work =` | `C =` |
| --- | --- | --- | --- |
| `N` | `n x m`, ld `n` | `diag(d) X` | `Xᴴ work` |
| the adjoint (`T` / `C`) | `m x n`, ld `m` | `X diag(d)` | `work Xᴴ` |

`OpX` = the adjoint with an eigenvector matrix U is the spectral recombination
`U f(Λ) Uᴴ`, which is how `recipes.orthogonalize` assembles `S^{-1/2}`:

```cpp
// X = U diag(lambda^{-1/2}) U^T; m_scratch is left holding U diag(lambda^{-1/2})
FBF_TRY(fb::gpu_status(fb::gmmwich<wwr::WWRBLAS_OP_T>(res.blas(), n, u, inv_sqrt_lambda,
                                                      m_scratch, x)));
```

- `T` is any `wwr::usual_fp`, as for `gemmwich`. d is a `T` too, as the
  complex dgmm takes it, so on a complex `T` a real diagonal (eigenvalues) is
  stored with zero imaginary parts. C is Hermitian up to rounding when d is
  real.
- `work` is scratch, ignored on entry and left as in the table. It must not
  alias X, d or C, and C must not alias X or d. C is overwritten (`beta = 0`).
- The dgmm takes no scalars. The gemm's `kOne` / `kZero` are in host memory,
  under the same `ScopedPointerMode` guard as `gemmwich`.
- It returns the first failing status; there is no host synchronisation
  between the two calls.

## Tests

GPU, in `test/gtest/{gemm,gemmwich,gmmwich}_gpu.cpp`, linked into
`free_breakfast_tests_gpu`.

## Files

| File | Role |
|------|------|
| `blas.cppm` | Primary interface: re-exports the four partitions |
| `shared.cppm` | `:shared`: `raw_blas_handle`, `BlasOp`, `SandwichOp`, `kAdjointOp`, all unexported |
| `gemm.cppm` | `:gemm`: `gemm` |
| `gemmwich.cppm` | `:gemmwich`: `gemmwich` |
| `gmmwich.cppm` | `:gmmwich`: `gmmwich` |
| `CMakeLists.txt` | The module library |

## Dependencies

All PUBLIC: every export is a template instantiated in the consumer's TU.

- `free_breakfast::core::types`: `int32_t`
- `free_breakfast::core::constants`: `kOne` / `kZero`, the sandwiches' scalars,
  real or complex
- `free_breakfast::core::wwr`: wwr's BLAS dispatch and `ScopedPointerMode`, and
  `CreatePolicy`
