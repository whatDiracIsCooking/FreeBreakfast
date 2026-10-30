# `test/sad/data/`

Golden reference data `test/sad/sad_host.cpp` reads, through
`FBF_TEST_DATA_DIR`, an absolute path compiled into the test binary. A copy,
never hand-edited: the file names the tool that wrote it and the source it came
from.

## `atomic_scf.json`

PySCF's spherically averaged atomic RHF (`AtomSphAverageRHF`) for H, He, Li,
Be, C, N, O, Ne, Na, Ar and Zn in STO-3G, H, He, C, O and Ne in cc-pVDZ, and C
and O in cc-pVTZ, each with its basis in BSE form. Written by
`tools/emit/sad_reference.py`, which runs PySCF.

| Key | Meaning |
|---|---|
| `provenance` | PySCF and numpy versions, date, host, the command that wrote it |
| `bases` | `{name: {elements: {Z: {electron_shells}}}}` in Basis Set Exchange JSON form: raw coefficients over primitive-normalised Gaussians, so a test reads it with `parse_basis` and `primitive_normalized` as a production run would |
| `atoms` | One per (basis, element): `basis`, `z`, `symbol`, `energy` (hartree, for a reader; no test compares it), and `density_blocks` |
| `density_blocks` | Per l, the spin-summed density over that momentum's shells for one m (the same for every m, zero between different m), over PySCF's **unit-normalised** spherical AOs. A test divides element (i, j) by N_i N_j, N_i the norm of this tree's function i |

Every element is one whose per-l electron counts in PySCF's
`NRSRHF_CONFIGURATION` equal the plain Madelung fill `src/sad` uses. Regenerate
with:

```sh
uv run --group tools tools/emit/sad_reference.py
```
