# Ethylene's four S1/S0 MECI starts (#846)

Park optimizes four S1/S0 conical intersections of ethylene (his p. 18):
twisted-pyramid (**Py**), ethylidene (**Et**), C3v-ethylidene (**C3v**) and
hydrogen-migration (**Hm**). Before any NEVPT2 is spent on them (#847), each
start is walked to its MECI over a bare DF-SA3-CASSCF(2,2)/cc-pVDZ
(cc-pVDZ-JKFIT), `[optimize] meci = [0, 1]`. The CASSCF is cheap, its
basins are the ones a QD-NEVPT2 walk starts in, and a start that fails here
would otherwise waste a full QD-PC-NEVPT2 walk.

## The starts are Park's geometries

Each `geometries/ethylene_{py,et,c3v,hm}_start.xyz` is Park's converged
QD-PC-NEVPT2/aug-cc-pVDZ MECI for that basin, from J. W. Park, *J. Chem.
Theory Comput.* (2020), acs.jctc.9b00919, Supporting Information,
pp. S2–S4 ("Ethylene, (2e,2o), SA3, aug-cc-pVDZ"). The SI prints them in
bohr, and they were converted to Å at 1 bohr = 0.529177210903 Å, with the
atom order kept (C, C, then the four H). His aug-cc-pVTZ geometries
(pp. S5–S7) are not used here.

The four configs, `ethylene_{py,et,c3v,hm}_casscf_meci.toml`, differ only in
`xyz`: SA3 over all three singlets at equal weights, `[casscf]
conv_tol_grad = 1e-10`, `max_gap = 1e-7` Eh, Opt=Tight's four criteria, and
`max_steps = 200`. They run like every config in this directory
([README.md](README.md), "How the runs are made").

## How a structure is identified

- **RMSD** is all-atom, after Kabsch alignment, minimized over the
  permutations of the two C and of the four H, *and over a reflection*: a
  MECI and its mirror image have the same energy, and the Hm walk below lands
  on Py's mirror image.
- **The heavy-atom RMSD carries almost no information here.** It is reported
  because the issue asks for it, but ethylene has two heavy atoms, so after
  alignment it is half the difference of the two C–C lengths: Park's Et and
  C3v are 0.0016 Å apart on it, and 0.159 Å apart all-atom.
- Each structure is named by its connectivity (how many H each carbon holds
  within 1.3 Å) and the angle that defines it: the angle sum at a CH2 carbon
  (360° is planar) for the pyramidalization, and the C–C–H angle of the
  carbene H for the two ethylidenes.

## Results

CUDA, RTX 3080 (10 GB), `--preset default` (`build/`), each run holding
the card (`with_gpu_card cuda`). Py, Et and C3v come from the build over
main at e1868b8 (#865); the Hm walks below come from the build at a0e7b9c.
Py and Et were also run at a0e7b9c, with identical evaluation counts,
energies, gaps and overlaps. Every walk below converged with
`completed: 0 errors, 0 warnings`, with a two-dimensional branching space
at every evaluation.

| | Py | Et | C3v |
|---|---|---|---|
| evaluations | 21 | 18 | 12 |
| mean E (Eh) | −77.8413115329 | −77.8946737354 | −77.8916036866 |
| `gap` (Eh) | 1.36e-8 | 1.74e-9 | 5.70e-12 |
| `gradient_norm` (last; max) | 2.14e-12; 8.96e-11 | 4.34e-12; 5.12e-11 | 9.39e-11; 9.39e-11 |
| `ci_residual_norm` (last; max) | 2.62e-14; 9.99e-14 | 6.62e-14; 1.06e-13 | 5.27e-14; 9.64e-14 |
| min `pair_overlap` (evaluation) | 0.99842 (6) | 0.99950 (1) | 0.99999985 (8) |
| max / rms force (Eh/bohr) | 1.11e-5 / 5.05e-6 | 2.29e-6 / 7.87e-7 | 1.01e-5 / 3.19e-6 |
| max / rms displacement (bohr) | 5.74e-5 / 2.96e-5 | 2.71e-5 / 1.11e-5 | 3.17e-5 / 1.42e-5 |
| CASSCF root 2 at the MECI (Eh) | −77.6342434 | −77.8408142 | −77.8418402 |
| all-atom RMSD to Park's (Å) | 0.137 | 0.035 | 0.008 |

The `pair_overlap` floor is `kStateOverlapFloor` = 0.9. No walk came near
it: the lowest value on any of the three trails is 0.99842, and so no walk slid through to
the third root.

Pairwise RMSD of the converged CASSCF MECIs (Å, all-atom / heavy-atom):

| | Py | Et | C3v |
|---|---|---|---|
| Py | 0 | 0.511 / 0.039 | 0.562 / 0.041 |
| Et | 0.511 / 0.039 | 0 | 0.125 / 0.0016 |
| C3v | 0.562 / 0.041 | 0.125 / 0.0016 | 0 |

Converged geometries:
[`geometries/ethylene_py_casscf_meci.xyz`](geometries/ethylene_py_casscf_meci.xyz),
[`geometries/ethylene_et_casscf_meci.xyz`](geometries/ethylene_et_casscf_meci.xyz)
and [`geometries/ethylene_c3v_casscf_meci.xyz`](geometries/ethylene_c3v_casscf_meci.xyz).

### Py: twisted-pyramidalized

The walk keeps two H on each carbon. One CH2 stays planar (angle sum 359.2°).
The other is strongly pyramidalized (301.2°), with one of its H bent down to
C–C–H 78.2°, 1.167 Å from its own carbon and 1.618 Å from the other. The C–C
bond is 1.383 Å. Park's QD-NEVPT2 Py has the same shape, more pyramidalized
(295.8°, bent H at 92.3°, 1.881 Å from the far carbon). The PT2 lengthening
of that H's reach is most of the 0.137 Å RMSD.

### Et: ethylidene

The walk gives CH3–CH: three H on one carbon (1.095–1.101 Å) and one on the
other. The carbene H is bent to C–C–H 157.3°, and the C–C bond is 1.461 Å.
It is 0.035 Å from Park's Et (C–C–H 152.6°). The methyl is not C3v: one of
its H sits at C–C–H 107.6° against 113.2° for the other two, and that
asymmetry is what separates Et from C3v. Root 2 lies 0.054 Eh above the pair here.

### C3v: C3v-ethylidene

The walk again gives CH3–CH, now exactly C3v: the carbene H lies on the
C–C axis (C–C–H 180.0°), and the three methyl H are equal at 1.096 Å and
111.6°. The C–C bond is 1.464 Å. The pair here is the linear carbene's E
pair, degenerate by symmetry, and the walk settles onto the C3v line:
from Park's start it took 12 evaluations. The structure is 0.008 Å from
Park's C3v.

Against Et, the difference is the carbene H: 157.3° there, 180° here.
That makes C3v 0.125 Å and 3.07 mEh (0.084 eV) above Et. Root 2 is
0.050 Eh above the pair.

Two fixes made this walk possible:
- #856: SA weights are carried, not recovered from the roots' 1-RDMs,
  which coincide near C3v. Before it, `df_intersection` refused the walk
  near the axis.
- #865: the final fresh RHF at the MECI now converges. Before it, the run
  exited 6 there.

## Hm has no basin of its own at SA3-CASSCF(2,2), or at QD-PC-NEVPT2

**Finding: no walk found an Hm MECI distinct from Py and Et.** Three walks
were run on the same card and build, aimed at an Hm basin:

| start | evaluations | lands on | RMSD to it (Å) |
|---|---|---|---|
| Park's Hm (`ethylene_hm_casscf_meci.toml`) | 61 | Py's mirror image | 0.0000 |
| Park's Hm with the migrating H moved over the C–C midpoint, 1.0 Å off the axis | 54 | Py | 0.0000 |
| hand-built: CH2 on one carbon, CH on the other, the fourth H bridging at 1.15 / 1.31 Å | 26 | Et | 0.0000 |

All three converged cleanly:
- from Park's Hm: mean E −77.8413115312, min `pair_overlap` 0.99975;
- from the bridged variant: −77.8413115321, min `pair_overlap` 0.99713;
- from the hand-built start: −77.8946737373, min `pair_overlap` 0.93356 at
  evaluation 3, the closest any walk came to the 0.9 floor.

Each matches Py's or Et's energy to 2e-9 Eh.

Park's Hm is a CH2–CH2 structure with both CH2 groups planar and twisted
90°, and one H swung to C–C–H 75.5° (1.188 / 1.582 Å from the two carbons).
At the CASSCF level that H relaxes back onto its own carbon, and the walk
ends at Py: Park's Hm is 0.257 Å from the Py MECI.

The hand-built and bridged starts were scratch inputs and are not checked in.
The hand-built one, in Å:

```
C   0.0000000000   0.0000000000   0.0000000000
C   0.0000000000   0.0000000000   1.4200000000
H  -0.8870394727  -0.3228559647  -0.5450000000
H   0.8870394727  -0.3228559647  -0.5450000000
H  -0.0000000000  -0.9353074361   1.9600000000
H   0.1000000000   1.0000000000   0.5700000000
```

### And at QD-PC-NEVPT2

So that #847 does not commit to an Hm start blind, one QD-PC-NEVPT2 walk was
run from Park's Hm geometry. The config is
[`tier0_ethylene_pc.toml`](tier0_ethylene_pc.toml) with `xyz =
"geometries/ethylene_hm_start.xyz"`: SA3-CASSCF(2,2)/cc-pVDZ
(cc-pVDZ-JKFIT), `[nevpt2_gradient] quasi_degenerate = true, variant =
"pc"`, and `meci = [0, 1]` on the same criteria.

**It lands on Py too.** The walk converged in 130 evaluations to a mean of
−78.1573045308 Eh, with:
- `heff_gap` 7.0e-10;
- `gradient_norm` 1.06e-11 at the end (max 6.9e-11) and `ci_residual_norm`
  2.43e-14 (max 1.45e-13);
- min `pair_overlap` 0.99812 (evaluation 108);
- the third QD state at least 0.125 Eh above the pair, and the CASSCF roots
  at least 0.020 Eh apart.

Its geometry is tier 0's PC-QD MECI ([README.md](README.md), "Tier 0",
reached from `examples/ethylene.xyz`) to 0.0000 Å RMSD and 1.7e-8 Eh. It is
0.020 Å from Park's QD-NEVPT2/aug-cc-pVDZ Py and 0.310 Å from his Hm.

| | Park's Hm | Park's Py | CASSCF Py | CASSCF Et | tier 0 PC MECI |
|---|---|---|---|---|---|
| all-atom RMSD of the Hm QD-PC MECI (Å) | 0.310 | 0.020 | 0.119 | 0.606 | 0.0000 |

So from Park's own geometry, in cc-pVDZ, our Hm falls into Py at the PT2
level as well as at CASSCF.

### And with diffuse functions

Two more walks test whether the diffuse functions are what make Hm a seam
minimum: all-electron QD-PC-NEVPT2/aug-cc-pVDZ (aug-cc-pVDZ-JKFIT, #845), on
tier 0's criteria. Frozen core is not an option here, since its gradient
does not exist yet (#875) and `[nevpt2_gradient]` refuses `frozen_core`.

- [`ethylene_hm_qdpc_augdz_meci.toml`](ethylene_hm_qdpc_augdz_meci.toml)
  starts from Park's aug-cc-pVDZ Hm.
- [`ethylene_hm_qdpc_augdz_py_reference.toml`](ethylene_hm_qdpc_augdz_py_reference.toml)
  starts from his aug-cc-pVDZ Py, and gives the reference energy.

| | from Park's Hm | from Park's Py |
|---|---|---|
| evaluations | 47 | 8 |
| mean E at the start (Eh) | −78.1522934435 | −78.1771099719 |
| `heff_gap` at the start (Eh) | 3.44e-5 | 5.84e-5 |
| mean E at the MECI (Eh) | −78.1771297191 | −78.1771297166 |
| `heff_gap` (Eh) | 8.3e-10 | 1.4e-9 |
| `gradient_norm` (last; max) | 2.60e-12; 9.35e-11 | 2.94e-12; 2.74e-11 |
| `ci_residual_norm` (last; max) | 3.45e-14; 1.14e-13 | 1.82e-14; 6.79e-14 |
| min `pair_overlap` (evaluation) | 0.99896 (23) | 0.99999991 (1) |
| min third-state margin; min CASSCF root gap (Eh) | 0.1165; 0.0210 | 0.1648; 0.0278 |
| max / rms force (Eh/bohr) | 5.82e-7 / 1.87e-7 | 5.41e-6 / 3.08e-6 |

All-atom RMSD of the converged MECI from Park's Hm, reflection allowed:

| | our aug-cc-pVDZ Py | Park's Py | Park's Hm | tier 0's PC MECI (cc-pVDZ) |
|---|---|---|---|---|
| RMSD (Å) | 0.0001 | 0.0012 | 0.2996 | 0.019 |

**Verdict: Park's Hm is a seam point, not a seam minimum, at our level.**
At all-electron QD-PC/aug-cc-pVDZ, Hm slides to Py, as it does at cc-pVDZ
and at SA3-CASSCF. The
walk from Park's Hm ends 0.0001 Å and 2.5e-9 Eh from the Py MECI, and our
Py is 0.0012 Å from his. At his Hm geometry the pair is already nearly
degenerate in our calculation: a gap of 3.44e-5 Eh against his 3.55e-5.
That point lies 0.676 eV above our Py MECI, against his 0.68 eV
(−78.1468050753 against −78.1717632797 Eh, state 0). So his Hm *is* on
the seam at this level. It is just not a minimum of the seam here: the walk
descends from it to Py.

Park's Hm is converged more loosely than his other three. Its state 0–1
gap is 3.55e-5 Eh at aug-cc-pVDZ and 2.19e-5 at aug-cc-pVTZ, against at
most 7e-6 for Py, Et and C3v (SI pp. S2–S7). That fits a point his
optimizer stopped at on a slope of the seam rather than at a minimum.

One difference is not ruled out. His frozen core moves every energy here
by a nearly constant amount, so it is not expected to create a seam
minimum. But the frozen-core MECI gradient does not exist yet (#875), so
that cannot be checked directly.

**Consequence for tier 1**, per the ruling on
[#847](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/847)
(milestone orchestrator, under the owner's delegation):
- Tier 1's Hm row is reported, not gated, and no Hm optimization is run.
- The row is given as frozen-core single points at Park's aug-cc-pV{D,T}Z Hm
  geometries (`single_points/`, #860).
- The gated per-MECI basis shift covers Py, Et and C3v.

The two walks' absolute energies sit 5.4 mEh below his (−78.1771297
against −78.1717633 for Py, state 0). That offset is the correlated core
([README.md](README.md), "What cannot match").
