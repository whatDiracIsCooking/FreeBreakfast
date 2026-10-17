# `test/gtest/data/`

Golden reference data the gtests read, through `FBF_TEST_DATA_DIR`, an absolute
path compiled into the test binary. These are copies, never hand-edited: each
file names the tool that wrote it and the source it came from.

| Directory | What | Written by |
|---|---|---|
| `one_electron/` | Cartesian basis shells, the nuclei, libcint's one-electron matrices (`int1e_ovlp`, `int1e_kin`, `int1e_nuc`) and PySCF's `hcore` for H₂O in STO-3G and cc-pVDZ, recorded by PySCF | `tools/extract_one_electron_golden.py`, from FelizJueves's `test/oracle/fixtures/` |

## `one_electron/*.json`

| Key | Meaning |
|---|---|
| `source` | The FelizJueves commit and fixture this was copied from |
| `provenance` | The fixture's own: PySCF, libcint and numpy versions, date, the molecule as PySCF was given it, the basis name |
| `nao` | Cartesian AO count |
| `atoms` | The nuclei, in PySCF's atom order: `symbol`, `charge` (Z), `coord` (bohr) |
| `shells` | One per contraction, in AO order: `l`, `centre` (bohr), `exponents`, `coefficients`. The coefficients multiply **unnormalised** Cartesian primitives `x^a y^b z^c exp(-alpha r^2)`, so they go into a `Shell` as given, with no `primitive_normalized` |
| `arrays` | `{name: {shape, values}}`, `values` row-major in AO order |

Regenerate after bumping `deps/FelizJueves`:

```sh
tools/extract_one_electron_golden.py
```
