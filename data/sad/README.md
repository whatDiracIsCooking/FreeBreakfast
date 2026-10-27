# `data/sad/`

The SAD initial guess's atomic densities, computed ahead of time. Like
`data/basis/`, this is **build input, not installed data**: `src/io/sad`
compiles every file into the binary, and the Guess step takes an element's
density from there instead of running its atomic SCF.

## Layout

```
<basis>/<Z>.npy     e.g. cc-pVDZ/026.npy
```

- `<basis>` is the set's canonical name, the BSE `name` field of its
  `data/basis/*.json` document. That is what `BasisSet::name()` holds, and the
  lookup compares it exactly. A directory naming no `data/basis` document fails
  the configure.
- `<Z>` is the atomic number, three digits, so a listing sorts by element.
- Each file is one atom's converged spherically averaged density over its own
  spherical AOs: n x n float64 in Fortran order, n the atom's spherical AO
  count. It is what `sad_density` computes for that atom alone. Load one with
  a plain `np.load`.

Every element of every `data/basis` set is here, but one row: 379 files,
about 3.8 MB. The exception is aug-cc-pVTZ's Sc–Zn, at 119 Cartesian AOs
each, past the 96 the atomic SCF holds its n⁴ tensor for (`src/sad/README.md`).
The generator skips them because the runtime refuses them the same way, so a
molecule with one of them fails its Guess step in aug-cc-pVTZ, stamp or no
stamp.

## Why

The atomic SCF holds its whole n⁴ two-electron tensor and has no screening. It
runs in milliseconds for the first rows, but takes minutes for a 3d element in
cc-pVDZ or cc-pVTZ. On the commit that added this directory, FeO in cc-pVDZ
took 2 min 17 s end to end with the SCF, and 0.43 s with the stamp. The
densities depend only on the element and its shells, so they are computed
once here, not on every run.

## When it is used, and when it is not

Only for a basis that `resolve_basis` took from the binary by name. A basis
read from a path computes every element, even when its document calls itself
`cc-pVDZ`, because only the compiled-in set is known to have the shells the
stamps were computed over (`src/basis/README.md`, "The name"). An element with
no file here is computed too.

## Regenerating

After a change to a `data/basis/` set, regenerate that set; after a change to
the atomic SCF in `src/sad/`, regenerate them all. Then rebuild, because
configure is what reads the files:

```sh
cmake --build build --target free_breakfast_sad_stamps
build/tools/sad/free-breakfast-sad-stamps data cc-pVDZ   # one set, or several
build/tools/sad/free-breakfast-sad-stamps data --all     # every set
cmake --build build
```

A set is named as `resolve_basis` takes it, so `cc-pvdz` is `cc-pVDZ`. Naming
none is a usage error rather than a quiet `--all`: the full run is the
expensive one, and it is rarely what a change needs.

The generator (`tools/sad/main.cpp`) resolves each set **through the binary**,
so build first: a set the binary does not carry yet is refused with "not
compiled into this binary". It removes each regenerated set's directory before
writing, so an element that is gone leaves nothing behind, and leaves the
other sets' directories alone. `--all` removes every subdirectory here, so a
set that is gone leaves nothing behind either. It runs every SCF in parallel.
On a 24-thread machine `--all` took 9.5 minutes when it covered five sets,
nearly all of it the cc-pVDZ and cc-pVTZ 3d elements; the Pople sets, STO-3G,
def2-SVP or MINAO alone take seconds. The six sets added after that, run on
their own, were dominated by aug-cc-pVTZ's Ga–Kr at about three minutes each.

The files are binary, so `git diff` shows only that they changed. The review
is `test/gtest/io_sad_host.cpp`: it recomputes H, C and O of every set in the
fast tier, and every element through argon (neon for cc-pVTZ) in the slow
tier. It compares with `EXPECT_EQ`, so both must match **bitwise**, and on
this toolchain they do. A build with different floating-point flags would
converge to the same density only to within roundoff, and that test would say
so.
