# `data/aux_basis/`

The density-fitting (auxiliary) basis sets compiled into the `free-breakfast`
binary, as [Basis Set Exchange](https://www.basissetexchange.org/) (BSE) JSON,
the same format as `data/basis/` and read by the same reader (`src/io/basis`).

They are **build input** in the same way: `src/io/basis/CMakeLists.txt` wraps
each file verbatim in a raw string literal, and `resolve_aux_basis("cc-pvdz-jkfit")`
parses it out of the binary. `resolve_aux_basis` also takes a plain file path,
so a fitting set this build does not carry needs no rebuild.

**They are kept apart from `data/basis/` on purpose.** They go into a table of
their own, which only `resolve_aux_basis` looks in. Every set in `data/basis/`
is one the SAD guess carries baked densities for (`tools/sad`, `data/sad/`,
checked by `test/gtest/io_sad_host.cpp`), and a SAD density in a fitting basis
means nothing. `is_builtin_basis` and `resolve_basis` never see these names.

## What is here

| File | Name | Elements | O |
|---|---|---|---|
| `cc-pVDZ-JKFIT.json` | `cc-pVDZ-JKFIT` | H, B–Ne, Al–Ar, Ga–Kr (19) | 10s 7p 5d 2f |
| `cc-pVTZ-JKFIT.json` | `cc-pVTZ-JKFIT` | the same 19 | 10s 7p 5d 2f 1g |

Weigend's JK fitting sets for Coulomb and exchange (F. Weigend, Phys. Chem.
Chem. Phys. 4, 4285 (2002)), the ones that pair with cc-pVDZ and cc-pVTZ.
Placed over H2O they give 116 and 139 fitting functions, PySCF's `naux`. As
with the orbital sets, the name a user writes is the document's `name` field
with `-`, `_` and spaces dropped and case ignored, so `ccpvdzjkfit` works.

Most shells are one uncontracted primitive. Ga–Br also carry one contracted
shell at each l (an s of three primitives, two-primitive p, d and f, and in
TZ a four-primitive g). Shells at l ≥ 2 are tagged `gto_spherical`; s and p
are bare `gto`, which the reader takes as spherical. cc-pVTZ-JKFIT runs past
the orbital ceiling: every element but H goes to g, and H goes to f. Holding
those shells is supported; an integral kernel that cannot take one rejects
it.

## Where they come from: PySCF, not BSE

Both files are written by `tools/jkfit_from_pyscf.py` from PySCF's
`pyscf/gto/basis/cc-pvdz-jkfit.dat` and `cc-pvtz-jkfit.dat`. Those files hold
Molpro's `weigend_jkfit.libmol` (11 February 2010).

- **BSE has no cc-pVDZ-JKFIT.** The double-zeta set was never published.
  Molpro forms it by dropping each element's highest angular momentum from
  the triple-zeta set, and PySCF's file is that truncation, exactly. The
  host test `IoBasisAux.TheTripleZetaSetIsTheDoubleZetaSetPlusItsTopMomentum`
  holds the two files to it.
- **PySCF is the oracle.** Its density fitting uses these tables, so a PySCF
  reference and this tree fit over one auxiliary basis, bit for bit. Taking
  TZ from BSE would break that: BSE's cc-pVTZ-JKFIT is the same set, but
  rounded differently. Its exponents differ from PySCF's by up to 4e-8
  relative (8e-9 on O). BSE also lacks Ne, Ar and Kr, which in Molpro's
  table are extrapolated rather than published values.
- **Provenance travels in the file.** `revision_description` names the PySCF
  version (2.14.0, the one `uv.lock` pins), and every element's `references`
  entry names the source table. PySCF is Apache-2.0.
- **The values are PySCF's doubles exactly**, each written as its shortest
  round-trip decimal. Running the tool twice gives the same bytes.

## Regenerating

```bash
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/jkfit_from_pyscf.py
```

Re-run it after a PySCF bump. Nothing else is derived from these files, so
unlike `data/basis/` there is no `data/sad/` to regenerate.

## Adding a set

Fitting sets BSE does carry (`def2-universal-jkfit`, `cc-pvqz-jkfit`, ...)
can come from `tools/bse_download.py <name> -o data/aux_basis`. Check the
harmonic types first, as `data/basis/README.md` describes: one
`gto_cartesian` shell makes the whole file unreadable. Then extend the host
tests in `test/gtest/io_basis_host.cpp`, and reconfigure.
