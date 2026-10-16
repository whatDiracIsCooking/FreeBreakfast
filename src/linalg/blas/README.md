# BLAS

Three BLAS conveniences over WarpWraps' typed dispatch (`wwr::gemm<T>`,
`wwr::dgmm<T>`, re-exported by [`free_breakfast.core.wwr`](../../core/wwr/README.md)):

- `gemm`, `wwr::gemm` for square, contiguous operands;
- `gemmwich`, the sandwich product `Xᵀ F X` (and its general rectangular form)
  as two gemms;
- `gmmwich`, the same sandwich with a diagonal middle, `Xᵀ diag(d) X`, as a
  dgmm and a gemm.

They fix arguments and compose calls; they do not rename them. `gemm` forwards
to `wwr::gemm` with arguments filled in, and every other gemm still calls
`wwr::gemm` itself. `gemmwich` is two `wwr::gemm` calls under wwr's
`ScopedPointerMode`, and `gmmwich` the same composition with a `wwr::dgmm` in
place of the first gemm.

Unlike the rest of `linalg`, these take a BLAS handle rather than a
`DeviceResources` and return the library's `wwrblasStatus_t` rather than a
`Status`: they are building blocks, called the way `wwr::gemm` is.

## Module

`free_breakfast.linalg.blas`, target `free_breakfast::linalg::blas`, also
re-exported by `free_breakfast.linalg`. Three partitions under the primary
interface, all re-exported:

| Partition | Exports |
|-----------|---------|
| `:gemm` | `gemm` (three overloads) |
| `:gemmwich` | `gemmwich` (three overloads) |
| `:gmmwich` | `gmmwich` (three overloads) |

Two unexported helpers are shared between partitions: `raw_blas_handle`
(`:gemm`) unwraps a `BlasHandle` or passes a raw handle through, and
`is_n_or_t` (`:gemmwich`) is the sandwiches' argument check. They sit
unexported in the interface partitions that use them because the exported
templates that call them are instantiated in the importer's TU (`docs/style.md`,
"Module-private code").

```cpp
import free_breakfast.core.wwr;
import free_breakfast.linalg.blas;   // or free_breakfast.linalg
namespace fb = free_breakfast;
```

## gemm

`free_breakfast::gemm` is `wwr::gemm<T, int32_t>` for square, contiguous
operands: A, B and C all `n x n`, column-major, `m = n = k = n` and every
leading dimension `n`. Three overloads, each the one above it with arguments
filled in:

| Call | Computes |
| --- | --- |
| `gemm(handle, op_a, op_b, n, alpha, a, b, beta, c)` | `C = alpha*op_a(A)*op_b(B) + beta*C` |
| `gemm(handle, op_both, n, alpha, a, b, beta, c)` | `C = alpha*op(A)*op(B) + beta*C`, one `op` for both |
| `gemm(handle, n, alpha, a, b, beta, c)` | `C = alpha*A*B + beta*C`: `op_both = WWRBLAS_OP_N` |

`op_a` / `op_b` are `WWRBLAS_OP_N`, `_T` or `_C` (conjugate transpose, the same
as `_T` for a real `T`). With `op_both = WWRBLAS_OP_T` the product is
`A^T B^T = (B A)^T`, the transpose of the product in the other order.

```cpp
fb::BlasHandle blas(dev);
FBF_GPU_MUST(fb::gemm(blas, n, &kOne<double>, a, b, &kZero<double>, c));
FBF_GPU_MUST(fb::gemm(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, &kOne<double>, m, u,
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
  and is reached qualified, and these take 7, 8 or 9. Inside `free_breakfast`,
  write `wwr::gemm` for the general call and plain `gemm` for these.

For a rectangular shape or a padded leading dimension, call `wwr::gemm`.

## gemmwich

`gemmwich` is the sandwich product, two `wwr::gemm`s through a caller-supplied
work matrix. Three overloads, each the one above it with arguments filled in:

| Call | Computes |
| --- | --- |
| `gemmwich(handle, op_x, op_f, m, n, x, f, work, c)` | `C = op_x(X)ᵀ op_f(F) op_x(X)` |
| `gemmwich(handle, op, n, x, f, work, c)` | `C = op(X)ᵀ F op(X)`: `op_x = op`, `op_f = N`, `m = n` |
| `gemmwich(handle, n, x, f, work, c)` | `C = Xᵀ F X`: `op = WWRBLAS_OP_N` |

In the general form `op_x(X)` is `n x m`, F is `n x n` and C is `m x m`, through
an `n x m` work matrix: `work = op_f(F) op_x(X)`, then `C = op_x(X)ᵀ work`.
Every matrix is column-major and contiguous, so its leading dimension is its
stored row count:

| Matrix | Stored as | Leading dimension |
| --- | --- | --- |
| X, `op_x = N` | `n x m` | `n` |
| X, `op_x = T` | `m x n` | `m` |
| F | `n x n` | `n` |
| work | `n x m` | `n` |
| C | `m x m` | `m` |

`op_x = N` with X stored `n x m` is `Xᵀ op_f(F) X`, the projection of F onto
X's `m` columns. `op_x` and `op_f` are each `WWRBLAS_OP_N` or `_T`; anything
else returns `WWRBLAS_STATUS_INVALID_VALUE` without enqueuing anything.

```cpp
fb::Buffers::device<double> work(n * n, dev);
FBF_GPU_MUST(fb::gemmwich(blas, n, x, f, work.data(), c));  // C = X^T F X

fb::Buffers::device<double> rect_work(n * m, dev);           // X stored n x m
FBF_GPU_MUST(fb::gemmwich(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, m, n, x, f,
                          rect_work.data(), c));             // C (m x m) = X^T F X
```

- `T` is `float` or `double` (`wwr::real_fp`): for complex operands the useful
  sandwich is `Xᴴ F X`, which this does not offer.
- `work` is scratch, ignored on entry and left holding `op_f(F) op_x(X)`. It
  must not alias X, F or C, and C must not alias X or F. C is overwritten
  (`beta = 0`), not accumulated into.
- The scalars are `kOne` / `kZero` in host memory, so the handle is held in
  host pointer mode for the call by wwr's `ScopedPointerMode` (bound to
  `CreatePolicy`) and handed back in whatever mode it had. A failure to read or
  set the mode means a bad handle and aborts, as `ScopedDeviceIndex` does.
- It returns the first failing gemm's `wwrblasStatus_t`; there is no host
  synchronisation between the two gemms.

## gmmwich

`gmmwich` is `gemmwich` with F = diag(d), where the diagonal is never formed:
a `wwr::dgmm` scales X in O(nm), where a gemm against a dense diagonal would
be O(n²m), and one `wwr::gemm` finishes. There is no `op_f`, since a diagonal
is its own transpose. Three overloads, as for `gemmwich`:

| Call | Computes |
| --- | --- |
| `gmmwich(handle, op_x, m, n, x, d, work, c)` | `C = op_x(X)ᵀ diag(d) op_x(X)` |
| `gmmwich(handle, op, n, x, d, work, c)` | `C = op(X)ᵀ diag(d) op(X)`: `op_x = op`, `m = n` |
| `gmmwich(handle, n, x, d, work, c)` | `C = Xᵀ diag(d) X`: `op = WWRBLAS_OP_N` |

`op_x(X)` is `n x m`, d has `n` elements (stride 1) and C is `m x m`. `work`
always has X's stored shape, because the diagonal goes on whichever side of X
needs no transpose:

| `op_x` | X stored as | `work =` | `C =` |
| --- | --- | --- | --- |
| `N` | `n x m`, ld `n` | `diag(d) X` | `Xᵀ work` |
| `T` | `m x n`, ld `m` | `X diag(d)` | `work Xᵀ` |

`op_x = T` with an eigenvector matrix U is the spectral recombination
`U f(Λ) Uᵀ`, which is how `linalg.orthogonalize` assembles `S^{-1/2}`:

```cpp
// X = U diag(lambda^{-1/2}) U^T; m_scratch is left holding U diag(lambda^{-1/2})
FBF_TRY(fb::gpu_status(fb::gmmwich(res.blas(), wwr::WWRBLAS_OP_T, n, u, inv_sqrt_lambda,
                                   m_scratch, x)));
```

- `T` is `float` or `double` (`wwr::real_fp`), as for `gemmwich`.
- `work` is scratch, ignored on entry and left as in the table. It must not
  alias X, d or C, and C must not alias X or d. C is overwritten (`beta = 0`).
- The dgmm takes no scalars. The gemm's `kOne` / `kZero` are in host memory,
  under the same `ScopedPointerMode` guard as `gemmwich`.
- `op_x` other than `N` / `T` returns `WWRBLAS_STATUS_INVALID_VALUE` without
  enqueuing anything. Otherwise it returns the first failing status; there is no
  host synchronisation between the two calls.

## Tests

GPU, in `test/gtest/{gemm,gemmwich,gmmwich}_gpu.cpp`, linked into
`free_breakfast_tests_gpu`.

## Files

| File | Role |
|------|------|
| `blas.cppm` | Primary interface: re-exports the three partitions |
| `gemm.cppm` | `:gemm`: `gemm` |
| `gemmwich.cppm` | `:gemmwich`: `gemmwich` |
| `gmmwich.cppm` | `:gmmwich`: `gmmwich` |
| `CMakeLists.txt` | The module library |

## Dependencies

All PUBLIC: every export is a template instantiated in the consumer's TU.

- `free_breakfast::core::types`: `int32_t`
- `free_breakfast::core::constants`: `kOne` / `kZero`, the sandwiches' scalars
- `free_breakfast::core::wwr`: wwr's BLAS dispatch and `ScopedPointerMode`, and
  `CreatePolicy`
