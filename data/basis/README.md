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

## What is here

| File | BSE name | Elements | Notes |
|---|---|---|---|
| `sto-3g.json` | `STO-3G` | Z 1–36 (H–Kr) | |
| `6-31g.json` | `6-31G` | Z 1–18 (H–Ar) | Cartesian d from K up; see below |
| `cc-pVDZ.json` | `cc-pVDZ` | Z 1–18, 20–36 | no K |
| `cc-pVTZ.json` | `cc-pVTZ` | Z 1–18, 20–36 | no K |

Filenames are whatever `tools/bse_download.py` writes, so the regeneration
commands below reproduce this directory byte for byte. The name a user writes
in a config is **not** the filename: it is the `name` field inside the
document, matched with `-`, `_` and spaces dropped and case ignored, so
`cc-pVDZ`, `cc-pvdz` and `ccpvdz` are one name.

## Regenerating

```bash
tools/bse_download.py sto-3g -o data/basis --elements 1-36 --force
tools/bse_download.py 6-31g -o data/basis --elements 1-18 --force
tools/bse_download.py cc-pVDZ cc-pVTZ -o data/basis --elements 1-18,20-36 --force
```

All four are BSE version 1 (`sto-3g` and `6-31g` revised 2018-06-19, the two
correlation-consistent sets 2018-10-03). Each document keeps its BSE header —
`description`, `references`, `revision_date` — so provenance travels with the
data rather than living only here.

## Why the element ranges are not all `1-36`

**`cc-pVDZ` and `cc-pVTZ` define no potassium.** BSE simply has no Z = 19 for
either set, and asking for it 404s the whole request. Everything else through
Kr is present.

**`6-31G` stops at argon because of the harmonic type, not coverage.** BSE
defines 6-31G through Kr, but from K (Z = 19) up its d shells are tagged
`gto_cartesian`, and this tree is spherical-only: every `ShellStamp` means
`2l+1` real solid harmonics, and `src/io/basis` rejects `gto_cartesian` at any
l by design. `parse_basis` walks the whole document, so a single Cartesian
shell anywhere makes the entire file unreadable — not just the element that
carries it. Cut at Z = 18, 6-31G is fully spherical and parses.

That same rule is why **`6-31G*` is not here at all.** Its d polarization
functions — the whole point of the star — are `gto_cartesian` for every
element from Li up, so the only elements it could offer are H and He, where
the star adds nothing. A Cartesian Pople set would need either a Cartesian
reader (a deliberate reversal of the design in `src/basis/README.md`) or a
spherical re-contraction, which is a different basis from the one BSE
publishes. Pick that fight separately; do not quietly ship data the reader
refuses.

## Adding a set

1. `tools/bse_download.py <name> -o data/basis --elements <range>`
2. **Check the harmonic types**, or the build will compile in a file that
   never parses. Every shell must be `gto_spherical`, or bare `gto`/untagged
   at l ≤ 1. `test/gtest/io_basis_host.cpp`'s
   `IoBasisBuiltins.EveryCompiledInSetParsesAndCoversItsElements` catches it,
   which is the point of that case.
3. Reconfigure. The glob is `CONFIGURE_DEPENDS`, so a new file is picked up on
   the next build without a manual `cmake` invocation.
4. **Regenerate `data/sad/`** (its README has the commands), so the new set's
   SAD guess is baked rather than computed on every run. The same goes for
   re-downloading a set: the stamps were computed over the old shells, and
   `test/gtest/io_sad_host.cpp` fails until they are regenerated.
