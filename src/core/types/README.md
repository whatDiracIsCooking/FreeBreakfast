# Types

Three things, all reached through one `import free_breakfast.core.types;`:

- **Builtin GPU vector types.** Maps a scalar type and an order `N` onto the
  corresponding builtin vector type: `Vec<double, 4>` is `double4`,
  `Vec<float, 3>` is `float3`, and so on for every scalar family
  (`signed char` … `double`), `N` in `1..4`. `make_vec` constructs one from its
  components in either host or device code. The builtin identifiers cannot be
  synthesised from `(scalar, N)` in the language, so each scalar row names its
  four builtins explicitly; `detail::nth_type` keeps the `N` axis DRY without a
  macro.
- **Fixed-width integer types.** The `:integral` partition re-exports
  `int8_t` … `uint64_t` and `size_t` into the global namespace, so a module can
  write bare `int32_t` / `size_t` after importing this module.
- **Scalar concepts.** The `:concepts` partition re-exports WarpWraps'
  `real_fp` and `int_type` (and `std::int64_t` / `std::size_t`) into
  `free_breakfast`. See [Concepts](#concepts).

## Module

`free_breakfast.core.types` (with the `:integral` and `:concepts` interface
partitions) — target
`free_breakfast::core::types`, also re-exported by `free_breakfast.core`.

## Backend neutrality

The vector types are spelled identically on CUDA and HIP (`::float3` is
`::float3` on both), so the mapping needs no backend `#if`. What differs is the
vendor header that declares them, and `vec.h` does not pick it itself: it
includes WarpWraps' `src/vector_types.h`, wwr's one neutral include point for
them, reached through `wwr::vector_types::host`.

`Vec` maps onto the global builtins rather than wwr's `wwr::float3` aliases,
because wwr deliberately leaves out the 64-bit 4-vectors (`double4`, `long4`,
`longlong4` and their unsigned forms) and `Vec` keeps them. That is the one
place this module goes further than wwr. CUDA 13 deprecates those five, whose
replacements (`double4_16a` / `_32a`) do not exist on HIP, so the plain name
is still the only portable spelling. `vec.h` silences the deprecation at the
mapping itself, so naming `Vec<double, 4>` is warning-free under clang. A
caller that wraps it in its own alias (`using D4 = Vec<double, 4>;`) gets the
warning back at each use of `D4`; spell `Vec<double, 4>` instead.

The builtin names themselves (`::float3`, `::double4`) are **not** visible to
an importer: they arrive through `types.cppm`'s global module fragment, which
exports nothing. Module code spells them `Vec<float, 3>`, or imports
`wwr.vector_types` for `wwr::float3`.

`vec.h` is not named `vector_types.h` on purpose. It includes wwr's header by
that name in angle brackets, and a same-named file here would make that include
find itself.

## Concepts

**These are WarpWraps' concepts, re-exported, not a second definition.**
`wwr.wrappers.common` already defines `real_fp` and `int_type`; the `:concepts`
partition `using`-declares them into `free_breakfast`. Two definitions sharing
a name would be two different concepts, and a template constrained on one could
disagree with a template constrained on the other about the same type.

- `real_fp<T>`: `T` is `float` or `double`.
- `int_type<T>`: `T` is **exactly** `int` or `int64_t`. There is no
  `convertible_to` arm: that arm let `std::size_t` through, but it let
  `double`, `float` and `bool` through too. A `size_t` caller casts.
- `std::int64_t` and `std::size_t` are re-exported alongside them, as
  `free_breakfast::int64_t` / `free_breakfast::size_t` (the `:integral`
  partition's are the global-namespace spellings of the same types).

The GPU-bound concepts built on `real_fp` (`complex_fp`, `half_fp`,
`usual_fp`, `ComplexToRealType`, ...) are not re-exported here; import
`wwr.wrappers.common` for those. The concepts have no header counterpart:
device code that needs them is out of reach of `import` either way.

## Files

| File | Role |
|------|------|
| `vec.h` | The vector mapping: `vector_type` trait, `Vec` alias, `make_vec` factory. Includable from a `.cu`. |
| `integral.h` | `<cstdint>` + `<cstddef>` for the fixed-width integer types and `size_t`. Includable from a `.cu`. |
| `types.cppm` | Primary module interface; GMF-includes `vec.h`, re-exports its names, and `export import :integral`. |
| `integral.cppm` | The `:integral` partition; GMF-includes `integral.h` and re-exports the integer types. |
| `concepts.cppm` | The `:concepts` partition; re-exports wwr's `real_fp` and `int_type`, plus `int64_t`/`size_t`, into `free_breakfast`. |
| `CMakeLists.txt` | Build definitions for the module and the headers' interface target |

## Build targets

- **`free_breakfast::core::types`**: the C++23 module library, for host code
  that `import`s it. Links `wwr::vector_types::host` and
  `wwr::wrappers::common` PUBLIC.
- **`free_breakfast::core::types_headers`**: an INTERFACE library exposing
  `vec.h` and `integral.h` to device translation units, which `#include` them
  directly (a named module cannot be imported into a `.cu`).

## Usage

- **Host / module code:** `import free_breakfast.core.types;` (or
  `free_breakfast.core`) and use `free_breakfast::Vec` /
  `free_breakfast::make_vec`, the bare integer types (`uint32_t`, `size_t`), or
  the concepts (`free_breakfast::real_fp`, `free_breakfast::int_type`).
- **Device code:** link `free_breakfast::core::types_headers` from the
  `free_breakfast_add_gpu_device_library` target and `#include "vec.h"` and/or
  `#include "integral.h"`.

```cpp
import free_breakfast.core.types;

free_breakfast::Vec<double, 3> r = free_breakfast::make_vec<double>(0, 0, 1.4);

template <free_breakfast::real_fp T>
void axpy(T alpha, const T* x, T* y, free_breakfast::int_type auto n);
```
