# Linear operator

The `LinearOperator<Op, T>` concept: a matrix known only by what it does to a block of device vectors, Y = A X. An iterative eigensolver applies its operator through it and never learns what the operator is: a dense matrix in a test, or a CIS or CASCI sigma build that never forms one.

## Module

`free_breakfast.recipes.linear_operator`, also re-exported by `free_breakfast.recipes`. Its own module rather than part of `davidson` so that an operator can name the concept without importing a solver, and a second solver can take the same operators.

## Design

A model has one member, `Status apply(DeviceResources &res, int32_t k, const T *x, T *y)`. It writes Y = A X for the n×k column-major device blocks `x` and `y`, both with leading dimension n.

**n is the operator's.** The concept passes only the block width k. The solver is given n separately and the two must agree.

**`res` is the solver's.** The operator queues its work on `res.stream()` with `res.blas()`, and may take `res.device_workspace()` for its scratch. That is why a solver holds none of the workspace across `apply`: its subspace lives in its own buffers (see `DeviceResources`' "One routine owns the device workspace at a time").

**Non-const, by lvalue reference.** A model may keep state, such as an apply count or a cached intermediate, so `apply` is not required to be `const` and a solver takes the operator as `Op &`.

## Public API

| Symbol | Role |
|---|---|
| `LinearOperator<Op, T>` | `op.apply(res, k, x, y)` returns something convertible to `Status` |

## Usage

```cpp
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.recipes;
namespace fb = free_breakfast;

// A dense n x n device matrix as an operator: one gemm per apply.
struct Dense {
  std::int32_t n;
  const double *a;
  fb::Status apply(fb::DeviceResources &res, std::int32_t k, const double *x, double *y) {
    const double one = 1.0;
    const double zero = 0.0;
    return fb::gpu_status(wwr::gemm<double, std::int32_t>(res.blas().get(), wwr::WWRBLAS_OP_N,
                                                          wwr::WWRBLAS_OP_N, n, k, n, &one, a, n,
                                                          x, n, &zero, y, n));
  }
};
static_assert(fb::LinearOperator<Dense, double>);
```

## Tests

`test/gtest/davidson_gpu.cpp` checks the concept against models that satisfy it and ones that do not, and drives `davidson_solve` through a dense model.

## Files

| File | Role |
|------|------|
| `linear_operator.cppm` | Module interface: `LinearOperator` |
| `CMakeLists.txt` | The module `free_breakfast.recipes.linear_operator` |
