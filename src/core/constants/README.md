# Constants

Type-generic physical constants and SI unit prefixes for host and device code.

## Module

`free_breakfast.core.constants` — target `free_breakfast::core::constants`, also
re-exported by `free_breakfast.core`.

## Constants

### Numeric literals

Typed `0`/`1`/`2`/`-1`, for scalars that must be passed by address (e.g. the
BLAS / solver `alpha`/`beta` host pointers).

| Name | Value |
|------|-------|
| `kZero<T>` | 0 |
| `kOne<T>` | 1 |
| `kTwo<T>` | 2 |
| `kNegOne<T>` | -1 |

### Physical

| Name | Symbol | Value |
|------|--------|-------|
| `kBoltzmann<T>` | k_B | 1.380649e-23 |
| `kEpsilon0<T>` | epsilon_0 | 8.8541878188e-12 |
| `kElementaryCharge<T>` | e | 1.602176634e-19 |
| `kAvogadro<T>` | N_A | 6.02214076e+23 |
| `kFaraday<T>` | F | 96485.33212331001 |
| `kGasConstant<T>` | R | 8.31446261815324 |
| `kPi<T>` | pi | 3.141592653589793 |

### Length conversions (atomic units)

| Name | Value |
|------|-------|
| `kAngstromPerBohr<T>` | 0.529177210903 (CODATA a_0 in Angstrom) |
| `kBohrPerAngstrom<T>` | 1 / 0.529177210903 |

Multiply a length in bohr by `angstrom_per_bohr` to get Angstrom.

### SI Prefixes

`kGiga<T>`, `kMega<T>`, `kKilo<T>`, `kMilli<T>`, `kMicro<T>`, `kNano<T>`, `kPico<T>`, `kFemto<T>`

## Usage

```cpp
import free_breakfast.core.constants;

auto kT = free_breakfast::kBoltzmann<double> * 300.0;
auto length = 5.0 * free_breakfast::kNano<double>;
```

## Files

| File | Role |
|------|------|
| `constants.h` | Header-based definitions; plain C++, so a device `.cu` can `#include` it |
| `constants.cppm` | Module interface re-exporting all constants |
| `CMakeLists.txt` | Build definitions: the module, and `free_breakfast::core::constants_header` exposing `constants.h` to device code |
