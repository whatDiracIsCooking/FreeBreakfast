# Primitives

Element-wise math (`sqrt`, `log`, `sin`, `cos`, …) templated on `float` /
`double`, and small exact combinatorics (`factorial`, `double_factorial_odd`,
`binomial`), under the same names in host module code and in device code.

## Module

`free_breakfast.core.primitives` — target `free_breakfast::core::primitives`,
also re-exported by `free_breakfast.core`.

The names live in `free_breakfast::math`, not directly in `free_breakfast`:
`free_breakfast::log` is already the logging facade's namespace, and a function
cannot share a name with it.

## Functions

### Element-wise

Every element-wise function is a template accepting exactly `float` or `double`;
an `int` argument is rejected at overload resolution rather than silently
promoted. Each dispatches to the C library's `f`-suffixed name for `float`
(`::sqrtf`) and the bare one for `double` (`::sqrt`).

| Group | Functions |
|-------|-----------|
| Exponential / logarithmic | `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`, `log1p` |
| Powers and roots | `pow`, `sqrt`, `rsqrt` |
| Trigonometric / hyperbolic | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh` |
| Rounding | `floor`, `ceil`, `trunc`, `round`, `rint` |
| Magnitude, sign, min/max | `fabs`, `hypot`, `copysign`, `fmod`, `fmin`, `fmax`, `fma` |
| Classification | `isnan`, `isinf`, `isfinite` |

`rsqrt` is the one function whose two sides differ: the host computes
`1 / sqrt(x)` (two roundings; the C library has no rsqrt), the device uses the
vendor intrinsic, so results may differ in the last bit.

### Combinatorics

`int32_t` in, `double` out, `constexpr`, exact for the small arguments a basis
set reaches (angular momenta).

| Function | Value |
|----------|-------|
| `double_factorial_odd(n)` | n!! over odd n; 1 for n <= 1 |
| `factorial(n)` | n!; 1 for n <= 1 |
| `binomial(n, k)` | C(n, k); 0 outside 0 <= k <= n |

## One definition, both sides

Every function is written **once**, in `primitives.h`, marked
`__host__ __device__ __forceinline__` in a device pass and plain C++ otherwise.
The element-wise bodies call the global C names (`::sqrtf`, `::sin`, ...), which
both CUDA and HIP declare host+device: a device pass resolves them to the
vendor's device implementation, a host compile to libm. `primitives.cppm`
GMF-includes the header and re-exports the names, so the host module and a
kernel run the same source. WarpWraps' `wrappers/math/math.cuh` is not reused
because its templates are `__device__`-only.

## Usage

Host module code:

```cpp
import free_breakfast.core.primitives;

double r = free_breakfast::math::sqrt(dx * dx + dy * dy + dz * dz);
```

Device code (`.cu`, linking `free_breakfast::core::primitives_header`):

```cpp
#include "primitives.h"

__device__ float f(float x) { return free_breakfast::math::exp(-x * x); }
```

## Files

| File | Role |
|------|------|
| `primitives.h` | Every definition, host+device. Include it from a `.cu` |
| `primitives.cppm` | Host module: GMF-includes `primitives.h` and re-exports its names. Must list everything the header defines |
| `CMakeLists.txt` | The module, and `free_breakfast::core::primitives_header` (links `wwr::device` for the vendor runtime header, and `types_headers`) for device code |
