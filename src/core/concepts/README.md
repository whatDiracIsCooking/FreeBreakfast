# Concepts

The scalar type-trait concepts, under the `free_breakfast` namespace.

## Module

`free_breakfast.core.concepts` — target `free_breakfast::core::concepts`, also
re-exported by `free_breakfast.core`.

## Design

**These are WarpWraps' concepts, re-exported, not a second definition.**
`wwr.wrappers.common` already defines `real_fp` and `int_type`; this module
`using`-declares them into `free_breakfast`. Two definitions sharing a name
would be two different concepts, and a template constrained on one could
disagree with a template constrained on the other about the same type.

- `real_fp<T>`: `T` is `float` or `double`.
- `int_type<T>`: `T` is **exactly** `int` or `int64_t`. There is no
  `convertible_to` arm: that arm let `std::size_t` through, but it let
  `double`, `float` and `bool` through too. A `size_t` caller casts.
- `std::int64_t` and `std::size_t` are re-exported alongside them.

The GPU-bound concepts built on `real_fp` (`complex_fp`, `half_fp`,
`usual_fp`, `ComplexToRealType`, ...) are not re-exported here; import
`wwr.wrappers.common` for those.

## Usage

```cpp
import free_breakfast.core.concepts;
using namespace free_breakfast;

template <real_fp T>
void axpy(T alpha, const T* x, T* y, int_type auto n);
```

## Files

| File | Role |
|------|------|
| `concepts.cppm` | Module interface: re-exports wwr's `real_fp` and `int_type`, plus `int64_t`/`size_t` |
| `CMakeLists.txt` | Build definition; links `wwr::wrappers::common` PUBLIC |
