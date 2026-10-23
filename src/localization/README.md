# Localization

Orbital localization: post-SCF rotations of the occupied orbitals, within their
own span, into a chemically readable form. The density, the energy and every
observable are invariant under them; only the individual orbitals change. So a
localization runs after the SCF, not inside it.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `pm/` | `free_breakfast.localization.pm` | Pipek-Mezey on the GPU (`pipek_mezey_localize`), maximizing the sum of squared (or fourth powers of) Mulliken atomic populations |

Pipek-Mezey came first because it needs nothing the SCF does not already have:
the overlap and the occupied orbitals. Foster-Boys (dipole integrals) and
Edmiston-Ruedenberg (the two-electron integrals in the MO basis) are not here.

The pipeline step over it is `pipeline/pipek_mezey`.

## Files

| File | Role |
|------|------|
| `CMakeLists.txt` | Adds the sub-directories |
