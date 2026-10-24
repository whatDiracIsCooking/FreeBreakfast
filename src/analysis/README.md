# Analysis

Wavefunction analysis: properties read off orbitals a calculation already has,
without changing them.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `population/` | `free_breakfast.analysis.population` | Orbitals' weights on an orthonormal atom-owned minimal basis such as the IAOs (`orbital_weights`), their atomic populations (`orbital_populations`), and partial charges from them (`partial_charges`) |
| `bond/` | `free_breakfast.analysis.bond` | Each orbital's centers and their angular character from those weights (`bond_orbitals`): core or lone pair, bond, multi-center bond. Host only |

## Files

| File | Role |
|------|------|
| `CMakeLists.txt` | Adds the sub-directories |
