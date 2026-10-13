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
| `zero<T>` | 0 |
| `one<T>` | 1 |
| `two<T>` | 2 |
| `neg_one<T>` | -1 |

### Physical

| Name | Symbol | Value |
|------|--------|-------|
| `boltzmann<T>` | k_B | 1.380649e-23 |
| `epsilon_0<T>` | epsilon_0 | 8.8541878188e-12 |
| `elementary_charge<T>` | e | 1.602176634e-19 |
| `avogadro<T>` | N_A | 6.02214076e+23 |
| `faraday<T>` | F | 96485.33212331001 |
| `gas_constant<T>` | R | 8.31446261815324 |
| `pi<T>` | pi | 3.141592653589793 |

### Length conversions (atomic units)

| Name | Value |
|------|-------|
| `angstrom_per_bohr<T>` | 0.529177210903 (CODATA a_0 in Angstrom) |
| `bohr_per_angstrom<T>` | 1 / 0.529177210903 |

Multiply a length in bohr by `angstrom_per_bohr` to get Angstrom.

### SI Prefixes

`giga<T>`, `mega<T>`, `kilo<T>`, `milli<T>`, `micro<T>`, `nano<T>`, `pico<T>`, `femto<T>`

## Usage

```cpp
import free_breakfast.core.constants;

auto kT = free_breakfast::boltzmann<double> * 300.0;
auto length = 5.0 * free_breakfast::nano<double>;
```

## Files

| File | Role |
|------|------|
| `constants.h` | Header-based definitions; plain C++, so a device `.cu` can `#include` it |
| `constants.cppm` | Module interface re-exporting all constants |
| `CMakeLists.txt` | Build definitions: the module, and `free_breakfast::core::constants_header` exposing `constants.h` to device code |
