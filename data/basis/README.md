# `data/basis/`

The basis sets compiled into the `free-breakfast` binary, as
[Basis Set Exchange](https://www.basissetexchange.org/) (BSE) JSON — the one
on-disk basis format this tree reads (`src/io/basis`).

These files are **input to the build**, not data installed beside the
executable: `src/io/basis/CMakeLists.txt` wraps each one verbatim in a raw
string literal in a generated module implementation unit, and
`resolve_basis("sto-3g")` parses it out of the binary. The reasoning is in
`src/io/basis/builtin_basis.cpp.in`; the short version is that this project
installs nothing of its own, so a runtime data directory would mean inventing
an install rule and a search path, and a name that resolves inside the binary
cannot be misplaced. Adding a set is a rebuild — and `resolve_basis` also
takes a plain file path, so a basis this build does not carry needs no rebuild
at all.

The density-fitting sets are compiled in the same way but live in
`data/aux_basis/`, not here: every set in this directory is one the SAD guess
carries baked densities for (`data/sad/`), and a fitting set needs none.

## What is here

| File | BSE name | Elements | Notes |
|---|---|---|---|
| `sto-3g.json` | `STO-3G` | Z 1–36 (H–Kr) | |
| `6-31g.json` | `6-31G` | Z 1–36 (H–Kr) | d relabelled spherical from K up; see below |
| `cc-pVDZ.json` | `cc-pVDZ` | Z 1–18, 20–36 | no K |
| `cc-pVTZ.json` | `cc-pVTZ` | Z 1–18, 20–36 | no K |
| `minao.json` | `MINAO` | Z 1–18, 20–36 | **not from BSE**: converted from PySCF; see below |

Filenames are whatever `tools/basis/bse_download.py` (or, for MINAO,
`tools/minao_from_pyscf.py`) writes, so the regeneration commands below
reproduce this directory byte for byte. The name a user writes
in a config is **not** the filename: it is the `name` field inside the
document, matched with `-`, `_` and spaces dropped and case ignored, so
`cc-pVDZ`, `cc-pvdz` and `ccpvdz` are one name.

## Regenerating

```bash
tools/basis/bse_download.py sto-3g -o data/basis --elements 1-36 --force
tools/basis/bse_download.py 6-31g -o data/basis --elements 1-36 --force
tools/basis/cartesian_to_spherical.py data/basis/6-31g.json
tools/basis/bse_download.py cc-pVDZ cc-pVTZ -o data/basis --elements 1-18,20-36 --force
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/minao_from_pyscf.py
```

The four BSE sets are BSE version 1 (`sto-3g` and `6-31g` revised 2018-06-19, the two
correlation-consistent sets 2018-10-03), and all but `6-31g.json` are written as
BSE serves them. Each document keeps its BSE header —
`description`, `references`, `revision_date` — so provenance travels with the
data rather than living only here.

## Why the cc ranges skip potassium

**`cc-pVDZ` and `cc-pVTZ` define no potassium.** BSE simply has no Z = 19 for
either set, and asking for it 404s the whole request. Everything else through
Kr is present.

## 6-31G from potassium on: Cartesian, carried spherical

BSE defines 6-31G through Kr, but from K (Z = 19) up its two d shells are
tagged `gto_cartesian`, and this tree is spherical-only: every `ShellStamp`
means `2l+1` real solid harmonics, and `src/io/basis` rejects
`gto_cartesian` at any l by design. `parse_basis` walks the whole document,
so one Cartesian shell anywhere makes the entire file unreadable — not just
the element that carries it.

So `tools/basis/cartesian_to_spherical.py` relabels those shells
`gto_spherical` after the download, keeping every exponent and coefficient:
the "5D" re-contraction other codes do on request. BSE coefficients multiply
normalized primitives, whose radial part is the same in either convention,
so one contraction serves both. What changes is the space: each d shell's
six Cartesians hold x²+y²+z², an s-like function, and the five pure d drop
it. K–Kr each have 27 functions here where BSE's 6D set has 29.

**This is not the basis BSE publishes**, and energies from K up will not
match a program running 6-31G Cartesian (Gaussian's default for Pople sets)
or this same file read 6D. H–Ar are unaffected: they have no shell above p,
and the tool changes nothing there. The document keeps the name `6-31G`, and
its `revision_description` records the relabelling, so the change travels
with the file. Running the tool again is a no-op.

**`6-31G*` is still not here**, though the same tool would now admit it: its
d polarization functions are `gto_cartesian` for every element from Li up, and
its 6D form is what its exponents were optimized in and what most published
6-31G* numbers mean. Carrying it 5D is a decision of its own, not a side
effect of this one.

## MINAO, the one set taken from PySCF

`minao.json` is the free-atom minimal basis of the intrinsic atomic orbitals
(Knizia, J. Chem. Theory Comput. 9, 4834 (2013), Appendix B): for each element,
the first n_occ contracted functions of each angular momentum of cc-pVTZ,
which are spherically averaged ground-state Hartree–Fock AOs. Molpro
introduced it under that name. It is there to be the IAOs' B2, not to run an
SCF in, though nothing stops one.

**The Basis Set Exchange does not serve it.** Its `mini` is Huzinaga's MINI,
a different set. So `tools/minao_from_pyscf.py` converts PySCF's copy of
Molpro's table, `pyscf/gto/basis/minao.py`, into the same BSE JSON the other
files are. PySCF's own `lo.iao` defaults to that table, so a PySCF oracle and
this tree build their IAOs over one B2, bit for bit.

- **Provenance travels in the file**: `revision_description` names the PySCF
  version it came from (2.14.0, the one `uv.lock` pins), and every element's
  `references` entry names the table. PySCF is Apache-2.0.
- **The values are PySCF's doubles exactly**, each written as its shortest
  round-trip decimal, so the reader parses back what PySCF holds. Running the
  tool twice gives the same bytes.
- **The element range is the cc sets'**. PySCF's table has no potassium,
  since cc-pVTZ has none. From rubidium on, it is cut from cc-pVTZ-PP, a
  valence set for an effective core potential, and this all-electron tree has
  no use for that.
- **Against `cc-pVTZ.json`**, the BSE copy in this directory, MINAO is exactly
  its first n_occ columns for every element but five. These five match only
  to rounding, because Molpro's cc-pVTZ table and BSE's are rounded
  differently: one Ca coefficient differs by 5e-8, and Ga, Ge, As and Kr carry
  exponents with fewer digits, at most 4.4e-7 relative. That is why this file
  is converted from PySCF rather than cut from `cc-pVTZ.json`: cutting would
  give a B2 that differs from the oracle's in those elements.

After a PySCF bump, re-run the tool. If the diff is not empty, regenerate
`data/sad/` too.

## Adding a set

1. `tools/basis/bse_download.py <name> -o data/basis --elements <range>`
2. **Check the harmonic types**, or the build will compile in a file that
   never parses. Every shell must be `gto_spherical`, or bare `gto`/untagged
   at l ≤ 1. A Cartesian set can be relabelled with
   `tools/basis/cartesian_to_spherical.py`, but that makes it a different
   basis (see 6-31G above): decide that, and say so here, rather than doing
   it to get a file to parse. `test/gtest/io_basis_host.cpp`'s
   `IoBasisBuiltins.EveryCompiledInSetParsesAndCoversItsElements` catches it,
   which is the point of that case.
3. Reconfigure. The glob is `CONFIGURE_DEPENDS`, so a new file is picked up on
   the next build without a manual `cmake` invocation.
4. **Regenerate `data/sad/`** (its README has the commands), so the new set's
   SAD guess is baked rather than computed on every run. The same goes for
   re-downloading a set: the stamps were computed over the old shells, and
   `test/gtest/io_sad_host.cpp` fails until they are regenerated.
