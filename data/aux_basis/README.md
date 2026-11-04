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
checked by `test/io/sad_host.cpp`), and a SAD density in a fitting basis
means nothing. `is_builtin_basis` and `resolve_basis` never see these names.

## What is here

| File | Name | Elements | O |
|---|---|---|---|
| `cc-pVDZ-JKFIT.json` | `cc-pVDZ-JKFIT` | H, B–Ne, Al–Ar, Ga–Kr (19) | 10s 7p 5d 2f |
| `cc-pVTZ-JKFIT.json` | `cc-pVTZ-JKFIT` | the same 19 | 10s 7p 5d 2f 1g |
| `aug-cc-pVDZ-JKFIT.json` | `aug-cc-pVDZ-JKFIT` | the same 19 | 11s 8p 6d 3f |
| `aug-cc-pVTZ-JKFIT.json` | `aug-cc-pVTZ-JKFIT` | the same 19 | 11s 8p 6d 3f 2g |
| `def2-universal-JKFIT.json` | `def2-universal-JKFIT` | H–Kr (36) | 10s 8p 4d 2f 1g |
| `cc-pVDZ-acCD.json` | `cc-pVDZ-acCD` | all of `data/basis/cc-pVDZ.json`'s 35, H–Kr but K | 8s 5p 5d 2f 1g, contracted |
| `<orbital>-acCD-6-6.json` | `<orbital>-acCD-6-6` | each its orbital set's | contracted, to i at most; [below](#the-accd-6-6-sets) |

Weigend's JK fitting sets for Coulomb and exchange (F. Weigend, Phys. Chem.
Chem. Phys. 4, 4285 (2002)), the ones that pair with cc-pVDZ and cc-pVTZ.
Placed over H2O they give 116 and 139 fitting functions, PySCF's `naux`.
`aug-cc-pVDZ-JKFIT` and `aug-cc-pVTZ-JKFIT` are the same two sets with one
diffuse function added per angular momentum, for aug-cc-pVDZ and
aug-cc-pVTZ: 150 and 196 over H2O, 300 and 392 over ethylene
([below](#the-aug-jkfit-sets-and-molpros-library)). As
with the orbital sets, the name a user writes is the document's `name` field
with `-`, `_` and spaces dropped and case ignored, so `ccpvdzjkfit` works.

`def2-universal-JKFIT` is Weigend's one JK fitting set for every def2
orbital set (F. Weigend, J. Comput. Chem. 29, 167 (2008)); PySCF pairs it
with def2-SVP and def2-TZVP alike. It has a few contracted shells (H's first
s is three primitives), and from Sc on it runs to h or i: Fe is
16s 13p 11d 9f 5g 3h 1i.

In the cc sets most shells are one uncontracted primitive. Ga–Br also carry one contracted
shell at each l (an s of three primitives, two-primitive p, d and f, and in
TZ a four-primitive g). Shells at l ≥ 2 are tagged `gto_spherical`; s and p
are bare `gto`, which the reader takes as spherical. cc-pVTZ-JKFIT runs past
the orbital ceiling: every element but H goes to g, and H goes to f. Holding
those shells is supported; an integral kernel that cannot take one rejects
it.

`cc-pVDZ-acCD` is not a published set: `tools/atomic_cd` derived it from
`data/basis/cc-pVDZ.json` itself (below). It covers every element the
orbital set does, including the Li–Be, Na–Mg, Ca and Sc–Zn that JKFIT lacks,
in 101 functions over H2O. Each element's shells are general contractions, one
per angular momentum, of up to 22 primitives (40 on the 3d metals), and no
shell passes g.

**No set here passes i, the DF kernels' ceiling.** `(P|Q)` and `(mn|P)`
carry aux shells through **i** (`kMetricMaxMomentum`,
`kThreeCenterAuxMaxMomentum` — the aux ladder is a separate, higher knob from
the orbital `kMaxAngularMomentum`, because a fitting set is expected to run
past the set it fits). The JKFIT cc sets and `cc-pVDZ-acCD` stop at g;
`def2-universal-JKFIT` and the acCD-6-6 sets reach h and i by name, beside
the hand-built `test/gtest/data/df_high_aux/`.

## Where the JKFIT sets come from: PySCF, not BSE

All five files are written by `tools/jkfit_from_pyscf.py` from PySCF's
`pyscf/gto/basis/cc-pvdz-jkfit.dat`, `cc-pvtz-jkfit.dat`,
`aug-cc-pvdz-jkfit.dat`, `aug-cc-pvtz-jkfit.dat` and
`def2-universal-jkfit.dat`. The cc files hold Molpro's
`weigend_jkfit.libmol` (11 February 2010), the aug ones Molpro's
`weigend_jkfit_aug.libmol` (16 February 2010); the def2 one is BSE's copy of
Turbomole 7.3's, as PySCF ships it, so taking it from PySCF rather than BSE
keeps the one-oracle rule below without costing any provenance.

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

## The aug JKFIT sets, and Molpro's library

Park's ethylene MECIs (J. W. Park, JCTC (2020), acs.jctc.9b00919) fit
aug-cc-pVDZ and aug-cc-pVTZ with aug-cc-pV{D,T}Z-JKFIT, "the coefficients
for the tempering functions ... adopted from the MOLPRO basis set library"
(Supporting Information, page S2). These two files are PySCF's tables for
those names. What was checked, and against what (2026-11-04):

- **Molpro's own library was not reached.** The URL Park cites,
  `https://www.molpro.net/info/basis.php`, answers 404, and no public copy
  of `weigend_jkfit_aug.libmol` was found. So the comparison below is with
  Molpro's documented construction, not with its file, and that is a gap.
- **BSE carries neither set** (its metadata lists cc-pV{T,Q,5}Z-JKFIT and
  def2 ones, no aug-JKFIT), so it gives no independent check.
- **PySCF's provenance** is its file header: "from MOLPRO
  weigend_jkfit_aug.libmol files 16 February 2010", merged from
  `cc-pvdz-jkfit.gbs` and `molpro-diffuse-aug-cc-pvdz-jkfit.gbs`. It quotes
  the libmol note on how the diffuse functions were made: one even-tempered
  function per angular momentum, e1²/e2 from the most diffuse exponent e1
  and the next e2, or e1/2.5 where there was only one, "not actually
  optimized for the indicated basis sets".
- **Psi4's `aug-cc-pv{d,t}z-jkfit.gbs`** (its `master` branch) carry the
  same header and are the same numbers bit for bit, all 19 elements: one
  lineage, so this confirms the transcription, not the source.
- **The construction holds** for the elements Park's runs need. Each aug
  set is its cc set with every shell kept and one primitive added per l.
  For H, B–F and Al–Cl every added exponent is the rule's value to
  ≤ 1e-7 relative, the eight printed decimals' rounding.
  `IoBasisAux.EachAugJkfitSetIsItsCcSetPlusDiffuseFunctions` holds H and C
  to it. Ne, Ar and Kr follow it to the four decimals their exponents are
  printed with. **Ga–Br do not.** They carry one more f than the rule, and
  in TZ two g where it gives one (Ga's f 0.26038, g 0.27382 and 0.15095,
  against the rule's g 0.21042). Whether Molpro's library has the same is
  unverified.

So for ethylene, the difference from the paper is only this: the sets were
read from PySCF, not from Molpro, and Molpro's file could not be compared.
For H and C they are what the libmol note says Molpro built.

### The metric's conditioning

The fit is Cholesky with no regularization
(`src/integrals/eri/density_fitting/fit/README.md`). The `lindep` route is
refused by every consumer that solves against L, gradients and MECI walks
included, so how close (P|Q) is to singular matters. Its eigenvalues over
Park's SI ethylene geometries, from libcint's `int2c2e` over these files
(the tree's metric kernels match it to 1e-12 of max|M|, by
`df_metric_gpu.cpp` on cc-pV{D,T}Z-JKFIT):

| Fitting set | naux | S0: λmin | S0: cond(M) | C3v MECI: λmin | C3v MECI: cond(M) |
|---|---|---|---|---|---|
| cc-pVDZ-JKFIT | 232 | 8.6e-5 | 3.8e6 | 8.8e-5 | 3.7e6 |
| aug-cc-pVDZ-JKFIT | 300 | 1.6e-5 | 6.1e7 | 2.6e-5 | 3.7e7 |
| aug-cc-pVDZ-acCD-6-6 | 438 | 2.5e-8 | 4.2e9 | 2.3e-8 | 4.6e9 |
| cc-pVTZ-JKFIT | 278 | 8.6e-5 | 3.9e6 | 8.3e-5 | 4.0e6 |
| aug-cc-pVTZ-JKFIT | 392 | 5.9e-6 | 1.6e8 | 6.6e-6 | 1.5e8 |
| aug-cc-pVTZ-acCD-6-6 | 1184 | 1.1e-9 | 9.3e10 | 8.1e-10 | 1.3e11 |

The diffuse functions cost the JKFIT sets one to two decades, to 6e7 and
1.6e8. That is still one to three decades better conditioned than the aug
acCD-6-6 sets the tree already carried for the same orbital sets.
**Cholesky holds its 1e-10 over all of them.** `free-breakfast`'s GPU DF-RHF
(CUDA, RTX 3080, `--preset default`, 2026-11-04) matched PySCF's DF-RHF over
the same files (`conv_tol` 1e-12) to the 1e-10 Eh it prints: ethylene at S0
in aug-cc-pVDZ / aug-cc-pVDZ-JKFIT (-78.0425992057) and aug-cc-pVTZ /
aug-cc-pVTZ-JKFIT (-78.0648685041), at the C3v MECI in aug-cc-pVTZ /
aug-cc-pVTZ-JKFIT (-77.8938918675), and at S0 over both aug acCD-6-6 sets,
cond(M) 4e9 and 9e10. So the agreement is still intact three decades above
the aug JKFIT sets' 1.6e8. Where it does degrade was not reached. Cholesky
itself fails near 1/ε ≈ 4.5e15, seven decades up. This bounds energies, not
gradients: no DF gradient has been compared over these sets yet.

The C3v MECI in aug-cc-pVDZ is not in that list because the tree's RHF
does not converge there (#865): it stalls at |[F,D]| ≈ 2.9e-6 after 400
iterations, drifting toward PySCF's -77.8731835362.

## Where cc-pVDZ-acCD comes from: generated here

Lehtola's automatic acCD with SVD contraction (JCTC 17, 6886 (2021) and 19,
6242 (2023)), run by `tools/atomic_cd/aux_set.py` at its default thresholds
(1e-7 Cholesky, 1e-5 contraction) over this tree's cc-pVDZ, not PySCF's.
`--max-l 4` keeps it within the C++ kernels' g; that binds only on Sc–Zn,
where cc-pVDZ has f and the uncapped set would reach l = 6.
`tools/atomic_cd/README.md` has the method and what the cap costs. The
settings and PySCF version are in the file's `revision_description`.

## The acCD-6-6 sets

Seven more generated sets, one per orbital set this tree carries but the
minimal two and the two smallest Pople ones, all at one setting: Cholesky
and contraction thresholds both 1e-6 (the `6-6`), and `--max-l 6`, the
kernels' i.

| Orbital set | acCD-6-6 on C | its JKFIT on C | CO dE, acCD-6-6 | CO dE, JKFIT |
|---|---|---|---|---|
| cc-pVDZ | 79 | 70 | +2.0e-5 | +2.4e-4 |
| cc-pVTZ | 199 | 79 | -2.9e-7 | +1.3e-4 |
| aug-cc-pVDZ | 145 | 86 (aug-cc-pVDZ-JKFIT) | +9.5e-6 | +2.3e-4 |
| aug-cc-pVTZ | 324 | 104 (aug-cc-pVTZ-JKFIT) | +1.4e-7 | +1.2e-4 |
| def2-SVP | 78 | 75 (universal) | +2.0e-5 | +1.6e-4 |
| def2-TZVP | 221 | 75 (universal) | +1.1e-7 | +9.4e-5 |
| 6-31G** | 82 | — | +1.5e-5 | — |

dE is E(DF-RHF) - E(RHF) in hartree on CO, over this tree's own orbital and
fitting files, PySCF's RHF; the aug JKFIT columns were measured over
PySCF's own tables, before this tree carried them. CH4 and H2O agree in order: every acCD-6-6
set is within 2e-5 Eh on all three, the TZ ones within 5e-7. So against
JKFIT on CO, about one order more accurate at double zeta at its size, and two to
three orders at triple zeta for 2.5-3x the functions.

**`--max-l 6` binds on cc-pVTZ's and aug-cc-pVTZ's Sc–Zn only**, whose g
shells would reach l = 8: it drops about 47 functions per atom (Fe 533 to
486 in cc-pVTZ). What that costs those elements has not been measured;
cc-pVDZ's g cap (`tools/atomic_cd/README.md`) suggests integrals much worse,
energies barely. aug-cc-pVTZ's Sc–Zn have no SAD stamp either
(`data/sad/README.md`), so no molecule with one runs in that set yet.

**6-31G and 6-31G\* are left out on purpose.** Their H has only s
functions, so its acCD set is three s functions, and no one-centre function
fits the p-like part of an H–X bond product: CH4's dE was -1.4e-3 Eh
(6-31G) and +2.0e-4 (6-31G\*). The tool cannot add the missing l, since
s x s products give no l = 1 candidate. 6-31G\*\* gives H a p shell and fits
like the rest.

`IoBasisAux.EveryAccdSixSixSetCoversItsOrbitalSetAndStopsAtI` holds each
file to its orbital set's elements and to l <= 6. By hand, `free-breakfast`'s
GPU DF-RHF on H2O over each of the seven, and over def2-TZVP /
def2-universal-JKFIT, matched PySCF's DF-RHF over the same files to the 1e-10
Eh it prints (CUDA, 2026-10-27); no ctest case pins that yet.

## Regenerating

```bash
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools tools/jkfit_from_pyscf.py
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    python -m tools.atomic_cd.aux_set data/basis/cc-pVDZ.json cc-pVDZ-acCD --max-l 4
for b in cc-pVDZ cc-pVTZ aug-cc-pVDZ aug-cc-pVTZ def2-SVP def2-TZVP; do
  UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
      python -m tools.atomic_cd.aux_set data/basis/$b.json $b-acCD-6-6 \
      --threshold 1e-6 --contract-threshold 1e-6 --max-l 6
done
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    python -m tools.atomic_cd.aux_set data/basis/6-31gstarstar.json '6-31G**-acCD-6-6' \
    --threshold 1e-6 --contract-threshold 1e-6 --max-l 6 \
    -o data/aux_basis/6-31gstarstar-acCD-6-6.json
```

Re-run them after a PySCF bump, and the acCD ones after any change to
their orbital file in `data/basis/` or to `tools/atomic_cd`. Each writes the same bytes
on a rerun. Nothing in `data/` is derived from these files, so unlike
`data/basis/` there is no `data/sad/` to regenerate; but
`tools/emit/df_reference.py` fits over them, so re-run it too.

## Adding a set

Fitting sets BSE does carry (`cc-pvqz-jkfit`, ...)
can come from `tools/basis/bse_download.py <name> -o data/aux_basis`. Check the
harmonic types first, as `data/basis/README.md` describes: one
`gto_cartesian` shell makes the whole file unreadable. Then extend the host
tests in `test/io/basis_host.cpp`, and reconfigure.
