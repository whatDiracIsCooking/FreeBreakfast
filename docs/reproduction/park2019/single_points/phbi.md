# Single points at Park's SI geometries: pHBI

#904. pHBI is the GFP chromophore model that milestone 26 left out of scope
([../REPRODUCTION.md](../REPRODUCTION.md), "What could not be done"). Each
point below is an SA2-QD-PC-NEVPT2 single point at one of Park's own SI
geometries, with the core frozen as his was, compared with the absolute
energies he lists beside it. It is the method of
[frozen_core.md](frozen_core.md), which matched ethylene and PSB3 to
≤ 0.52 µEh, taken to (12e,12o) and to 805 basis functions.

**Six of the SI's ten pHBI points match his to ≤ 0.14 µEh in both
states.** These are (4e,3o) at all three structures, and (12e,11o) and
(12e,12o) at the S0 minimum in cc-pVTZ. The sixth is (12e,11o) at the S0
minimum in aug-cc-pVTZ, 805 basis functions, in 136 s on the 40 GB A100
(it needed #905's fix). The other four did not match or did not run:

- **(14e,12o)/cc-pVTZ** ran, but **misses by mEh**. Three all-π choices of
  the space give +0.53, −6.05 and −4.24 mEh for state 0. Three starts with
  an in-plane lone pair as the seventh occupied orbital all converge to the
  second of those. So his (14e,12o) is none of them. Below, "(14e,12o)",
  says why it stops there.
- **(12e,11o)/cc-pVTZ at the I- and P-twisted MECIs** ran, but **misses by
  mEh**: +3.35 and +3.68 mEh for state 0. The start is the RHF orbitals
  that best carry a π-projected space. It converges to other CASSCF
  solutions than his: his two states are degenerate there and ours are
  not. **Carrying the S0 minimum's converged space there (#939) does not
  reach his either**: walked in eight steps, it lands at the P-twisted MECI
  on the same solution as the π-projected start (+3.68 mEh), and at the
  I-twisted one on a third (+3.48 mEh), its states 2.8 mEh apart where his
  are 20 µEh. So his twisted (12e,11o) solutions are not the ones
  continuously connected to the S0 π space along this path.
- **(14e,12o)/aug-cc-pVTZ** was **not run**. It needs the cc-pVTZ
  (14e,12o) space settled first, and that is not settled.

| point | Δ₀ (µEh) | Δ₁ (µEh) | status |
|---|---|---|---|
| (4e,3o)/cc-pVTZ S0 min | −0.00 | +0.04 | reproduced |
| (4e,3o)/cc-pVTZ I-twisted MECI | −0.05 | −0.08 | reproduced |
| (4e,3o)/cc-pVTZ P-twisted MECI | −0.14 | −0.03 | reproduced |
| (12e,11o)/cc-pVTZ S0 min | +0.04 | +0.11 | reproduced; the scale test |
| (12e,12o)/cc-pVTZ S0 min | +0.02 | +0.02 | reproduced |
| (12e,11o)/aug-cc-pVTZ S0 min | +0.02 | +0.01 | reproduced; 805 basis functions |
| (14e,12o)/cc-pVTZ S0 min | +526.00 | +1775.32 | ran; not his active space |
| (12e,11o)/cc-pVTZ I-twisted MECI | +3353.40 | +11218.40 | ran; not his CASSCF solution; carried from S0 (#939): +3478.76, +6278.65, not his either |
| (12e,11o)/cc-pVTZ P-twisted MECI | +3683.21 | +5065.05 | ran; not his CASSCF solution; carried from S0 (#939): the same solution |
| (14e,12o)/aug-cc-pVTZ S0 min | – | – | not run: the (14e,12o) space is unsettled |

## The points

[`phbi/geometries/`](phbi/geometries/) holds his ten pHBI structures,
converted from bohr at 1 bohr = 0.529177210903 Å, with no other change and
his atom order. That order is C1 N2 C3 N4 C5 (imidazolinone), C6 (bridge),
C7…C12 (phenoxy ring), O13 (phenoxide), O14 (carbonyl), then the seven H.
Each `.xyz` comment line cites its pages: J. W. Park, *J. Chem. Theory
Comput.* (2020), acs.jctc.9b00919, Supporting Information, pp. S24–S39.
pHBI here is the **anion**: there is no phenolic H, so 97 electrons
neutral, and his energies (−644.06 Eh) are those of the 98-electron
closed shell. Every config has `charge = -1`. In cc-pVTZ it has 518 basis
functions, his count (acs.jctc.9b00919 §4).

| file stem | structure | basis / aux | SI pages |
|---|---|---|---|
| `phbi_tz43_{s0,itw,ptw}` | S0 min, I-twisted MECI, P-twisted MECI, each optimized at (4e,3o) | cc-pVTZ / cc-pVTZ-JKFIT | S24–S27 |
| `phbi_tz1211_s0` | S0 min, (12e,11o) | cc-pVTZ / cc-pVTZ-JKFIT | S28–S29 |
| `phbi_tz1211_{itw,ptw}` | I-, P-twisted MECIs, (12e,11o) | cc-pVTZ / cc-pVTZ-JKFIT | S29–S31 |
| `phbi_tz1212_s0` | S0 min, (12e,12o) | cc-pVTZ / cc-pVTZ-JKFIT | S32–S33 |
| `phbi_tz1412_s0` | S0 min, (14e,12o) | cc-pVTZ / cc-pVTZ-JKFIT | S34–S35 |
| `phbi_atz1211_s0` | S0 min, (12e,11o) | aug-cc-pVTZ / aug-cc-pVTZ-JKFIT | S36–S37 |
| `phbi_atz1412_s0` (geometry only) | S0 min, (14e,12o) | aug-cc-pVTZ | S38–S39 |

The method in each config is DF-RHF (`max_iterations = 400`), then
DF-SA2-CASSCF at `conv_tol_grad = 1e-9` with `active` named, then
`[nevpt2] quasi_degenerate = true` and `frozen_core = "auto"`. Every log
reports `NEVPT2: 14 frozen core orbitals`, the C, N and O 1s. Both
JKFIT sets are carried in `data/aux_basis/`. Run from the root of the tree:

```sh
bash -c '. devtools/lib.sh; with_gpu_card cuda build/src/free-breakfast \
  -i docs/reproduction/park2019/single_points/phbi -o <out> \
  docs/reproduction/park2019/single_points/phbi/<stem>.toml'
```

## Δ = E_ours − E_his, PC-QD-NEVPT2

E_ours is from the step's `PC-QD-NEVPT2 energies` log line, at 12
decimals. E_his is from the SI, at 10 decimals. Δ is in µEh. In every log
below, `main` printed both `PC-QD-NEVPT2 state` lines, with no
`PC-QD-NEVPT2: refused` and no PC-refusal warning, and ended
`completed: 0 errors, 0 warnings`.

| point | E₀ ours (Eh) | E₀ his (Eh) | Δ₀ | E₁ ours (Eh) | E₁ his (Eh) | Δ₁ |
|---|---|---|---|---|---|---|
| (4e,3o) S0 min | −644.064695575599 | −644.0646955749 | −0.00 | −643.961393345907 | −643.9613933828 | +0.04 |
| (4e,3o) I-twisted | −643.963739377602 | −643.9637393315 | −0.05 | −643.963723110395 | −643.9637230351 | −0.08 |
| (4e,3o) P-twisted | −643.956720598305 | −643.9567204580 | −0.14 | −643.956712716009 | −643.9567126891 | −0.03 |
| (12e,11o) S0 min | −644.064703927419 | −644.0647039716 | +0.04 | −643.956437060214 | −643.9564371682 | +0.11 |
| (12e,12o) S0 min | −644.065072191798 | −644.0650722075 | +0.02 | −643.952497471475 | −643.9524974918 | +0.02 |
| (12e,11o)/aug-cc-pVTZ S0 min | −644.121526103953 | −644.1215261274 | +0.02 | −644.016838273742 | −644.0168382832 | +0.01 |
| (14e,12o) S0 min, all-π (not his) | −644.063112439751 | −644.0636384427 | +526.00 | −643.954903747327 | −643.9566790714 | +1775.32 |
| (12e,11o) I-twisted (not his) | −643.977522723378 | −643.9808761194 | +3353.40 | −643.969637336607 | −643.9808557371 | +11218.40 |
| (12e,11o) P-twisted (not his) | −643.948980156666 | −643.9526633687 | +3683.21 | −643.947586918569 | −643.9526519725 | +5065.05 |
| (12e,11o) I-twisted, carried from S0 (not his) | −643.977397355278 | −643.9808761194 | +3478.76 | −643.974577090650 | −643.9808557371 | +6278.65 |
| (12e,11o) P-twisted, carried from S0 (not his) | −643.948980156658 | −643.9526633687 | +3683.21 | −643.947586918567 | −643.9526519725 | +5065.05 |

The vertical gaps, E₁ − E₀ at the S0 minimum, as the paper quotes them
(acs.jctc.9b00919 §4, pHBI): (4e,3o) 2.8110 eV ours, 2.8110 his ("2.81");
(12e,11o) 2.9461 / 2.9461 ("2.95"); (12e,12o) 3.0633 / 3.0633 ("3.06");
(12e,11o)/aug-cc-pVTZ 2.8487 / 2.8487 ("2.85"). At the MECIs the (4e,3o)
gaps are 0.4 meV (I) and 0.2 meV (P), his to the digit.

## Active spaces, and how they were checked

The check is the π-character metric of [README.md](README.md), "Active
spaces": for each MO, the p weight along each heavy atom's local normal,
over its total p weight on heavy atoms. It is extended to the two oxygens,
which have one bonded neighbour; each takes its neighbour's normal. pHBI's
π system has 14 centres (10 C, 2 N, 2 O) and 16 π electrons (the phenoxide
O and the N–H nitrogen give two each). At the RHF S0 minimum the metric
finds exactly 8 occupied and 6 valence virtual orbitals of π character
1.000. In 0-based indices these are 30 (or 31), 37, 39, 42, 43, 46, 47, 48
and 49, 53, 55, 57, 62, 65. The full valence π space is (16e,14o). Park's
spaces are cuts of it:

- **(4e,3o)** is "the minimal (4e,3o) active space" of his ref. 77 (Olsen
  and Smith): the allyl-like set over the bridge. That is the two π
  orbitals with weight on the bridge carbon C6 and its neighbours C5 and
  C7, plus the bridge π*. At the S0 minimum and the I-twisted MECI, RHF
  orbitals 47, 48 and 49 are those. At the P-twisted MECI, RHF orbital 47
  is the phenoxy ring's nonbonding π (C8, C9, C10 only, no weight on C6 or
  C7), and the bridge π is 45, so `active = [45, 48, 49]`. In all three
  `casscf.molden` files the three active natural orbitals have π character
  0.92–1.00, with weight on C5, C6 and C7, and occupations 1.88–1.95,
  1.49–1.50 and 0.56–0.62.
- **(12e,11o)** is "6 π-orbitals and 5 π*-orbitals" (§4): the six highest
  occupied π and the five lowest π* at RHF, `[39, 42, 43, 46, 47, 48, 49,
  53, 55, 57, 62]`. It drops the two deepest occupied π (30 and 37) and the
  highest π* (65). All eleven active natural orbitals of `casscf.molden` have
  π character 1.000. Their occupations are 1.965, 1.957, 1.946, 1.923,
  1.907, 1.480, 0.572, 0.085, 0.072, 0.054 and 0.040.
- **(12e,12o)** is (12e,11o) plus the sixth π*, 65: six occupied π and every
  valence π*. All twelve active natural orbitals have π character 1.000,
  and the twelfth's occupation is 0.026.

The µEh comparison confirms each of these choices. A space off by one
orbital is an mEh effect, as the next section measures.

### (14e,12o): not his space, and why it stops here

(14e,12o) is twelve orbitals with seven doubly occupied, so it is six π
plus one more occupied orbital, with five virtuals. Three all-π choices
were run at his geometry:

| `active` (0-based RHF) | added to (12e,11o) | Δ₀ (mEh) | Δ₁ (mEh) | E₁ − E₀ (eV) |
|---|---|---|---|---|
| 37, 39–48 π, 49–62 π* (the config) | occupied π 37 (phenoxide O π) | +0.526 | +1.775 | 2.9445 |
| 31, 39–48 π, 49–62 π* | the deepest occupied π, 31 (N2) | −6.048 | −5.082 | 2.9368 |
| 37, 39–48 π, 49–57 and 65 π* | π 37, with π* 65 in place of 62 | −4.241 | −2.185 | 2.9665 |
| his | | | | 2.9105 |

None is his. In the first, the added orbital stays a near-spectator
(natural occupation 1.997), and the gap is (12e,11o)'s to 2 meV. He reports
that (14e,12o) lowers the gap by 0.04 eV (§4: "decreases the gap by 0.04 eV
(2.91 eV)"), so his seventh orbital is doing something ours is not. The
other two are other spaces again; in the third, the CASSCF trades π 37
for N2's π (occupation 1.998). The paper says only "(14e,12o)", and
the XMCQDPT2(14e,12o) it compares with (its ref. 81) is another
program's choice, not stated in the paper.

The seventh occupied orbital need not be π. pHBI is planar, so a σ orbital
cannot mix with the π ones by symmetry. The highest σ orbitals at RHF are
in-plane lone pairs, and those are the chemically natural candidates. Each
was added to the (12e,11o) space in turn, with the same five π*:

| `active` adds (0-based RHF) | its character (RHF) | converged space | Δ₀ (mEh) | Δ₁ (mEh) |
|---|---|---|---|---|
| 45 | phenoxide O13 in-plane lone pair (0.65 on O13) | all π, N2's π added | −6.048 | −5.082 |
| 44 | carbonyl O14 lone pair, with N4 (0.48, 0.20) | the same | −6.048 | −5.082 |
| 40 | imine N4 lone pair, with O14 (0.40, 0.21) | the same | −6.048 | −5.082 |

None of the three stays. In each, the CASSCF stalls for two macro
iterations near E = −641.6716 Eh (|g| about 1e-2), then goes on to a lower
solution, E = −641.675127118 Eh in all three. That solution is the all-π
space with N2's π as the seventh occupied orbital (natural occupation
1.995, 0.75 of its π weight on N2). Its Δ is the second row of the table
above, to the µEh. In the plane's symmetry, the gradient that would rotate
a σ orbital into a π one is zero. So the optimizer can leave a lone-pair
space only along a direction where the energy curves down, from noise
that grows. That is what these runs show, and it means the lone-pair
spaces are not CASSCF minima here. The optimizer here keeps no symmetry.
One that kept it could converge with the lone pair active. Whether Park's
did cannot be tested here: the tree has no symmetry-constrained CASSCF,
and no way to start from orbitals other than the RHF ones (#939).

So six spaces were tried and none is his. This page stops there. The config
is checked in as the record of the first row of the all-π table, and its
header says it is not his space. The aug-cc-pVTZ (14e,12o) point was not
run: it needs this space first.

### aug-cc-pVTZ

His two aug-cc-pVTZ points are (12e,11o) and (14e,12o). In aug-cc-pVTZ the
canonical RHF virtuals of π character are diffuse. The valence π* is
spread over many of them, so "the five lowest π*" names nothing. For
(12e,11o) the space itself is confirmed at cc-pVTZ, and only its start
needs choosing. So a cc-pVTZ SA2-CASSCF(12e,11o) was run at the
aug-cc-pVTZ geometry. The aug RHF virtuals were then projected onto its
five virtual natural orbitals through the exact cross overlap (cc-pVTZ is
a subset of aug-cc-pVTZ). The aug virtual with the largest projection
onto each was taken: 59, 87, 89, 93 and 108 (projections 0.84, 0.28,
0.35, 0.35 and 0.24). The six occupied π are 39–48 as in cc-pVTZ, each
projecting 0.57–1.00 onto the cc-pVTZ CASSCF's.

**It reproduces his to +0.02 and +0.01 µEh.** The first try of this
start was refused for device memory before CASSCF's first iteration
(#905). The DF three-center build had left its scratch in the grow-only
workspace, so the process held the whole card. #905's fix (PR #935) frees
that scratch, and the run now completes on the 40 GB A100. The CASSCF took
10 macro iterations to |g| 3.8e-10. All eleven active natural orbitals of
`casscf.molden` have π character 1.000 (diffuse shells left out). Their
occupations are 1.965, 1.957, 1.947, 1.924, 1.907, 1.480, 0.571, 0.084,
0.071, 0.054 and 0.040, the cc-pVTZ ones to within 0.001. The scale numbers
are in the table below. (14e,12o) in aug-cc-pVTZ inherits the cc-pVTZ
(14e,12o) question, so it was not run.

### (12e,11o) at the twisted MECIs

At his (12e,11o) I- and P-twisted geometries, the RHF virtuals do not
separate into π and σ. Between MOs 49 and 109, the only virtual with π
character above 0.3 is 49, and the rest are ≤ 0.26. So no single canonical
virtual is a π*, and "the five lowest π*" names no RHF orbital.

A start can still be chosen without guessing. Project the RHF orbitals onto
a minimal π basis: MINAO's 2p on each heavy atom, along that atom's local
normal (as in the π metric). Do this within the occupied orbitals and
within the virtual ones, as AVAS does. At all three geometries that gives
exactly 8 occupied and 6 virtual orbitals of weight 0.93 or more (the next
is at most 0.07). Diagonalize the RHF Fock matrix within each set and
leave out the orbitals the S0 space leaves out. Then take the RHF orbitals
that carry most of what is kept. At the S0 minimum the space leaves out the
two deepest occupied π (N2's, and the phenoxy ring's with O13) and the top
π* (the phenoxy ring's). There the rule gives `[39, 42, 43, 46, 47, 48]`
and `[49, 53, 55, 57, 62]`, the space confirmed to µEh above. At the
twisted geometries the orbitals left out are matched by their per-atom π
populations.

| point | `active` (0-based RHF) | principal overlaps with the target, occupied / virtual | converged CASSCF | Δ₀, Δ₁ (mEh) | E₁ − E₀, ours / his |
|---|---|---|---|---|---|
| I-twisted | 40, 41, 44, 45, 47, 48, 49, 50, 56, 59, 60 | 0.94–1.00 / 0.73–0.98 | 11 macro; all eleven natural orbitals π (0.94–1.00) | +3.35, +11.22 | 7.9 mEh / 20 µEh |
| P-twisted, first try | 39, 41, 44, 45, 47, 48, 49, 52, 54, 57, 62 | 0.84–1.00 / 0.63–0.97 | 15 macro; a σ/σ* pair takes two places (π 0.04, occupations 1.978 and 0.021) | +13.05, +16.25 | 3.2 mEh / 11 µEh |
| P-twisted | 36, 39, 44, 45, 47, 48, 49, 52, 54, 57, 62 | 0.83–1.00 / 0.63–0.97 | 9 macro; all eleven natural orbitals π (0.92–1.00) | +3.68, +5.07 | 1.4 mEh / 11 µEh |

At the I-twisted MECI the two deepest occupied π by energy are the S0
ones in character, and the top π* is the phenoxy ring's, so the energy
order and the character agree. At the P-twisted MECI they do not. The
first try left out the two deepest occupied π by energy. The second of
those is the carbonyl O14's π, which the S0 space keeps. The second try
leaves out the phenoxy ring's π with O13 in its place, as at S0. Its
SA-CASSCF energy is 0.32 mEh below the first try's.

All three miss by mEh, so none is his. The degeneracy says so on its own:
these are his MECIs at this method, and his two states are 20 and 11 µEh
apart there. Ours are 7.9 and 1.4 mEh apart. Both final spaces are π
throughout, yet they are different CASSCF solutions from his. In the
first P-twisted try, the virtuals' σ* content (57 and 62 have π character
0.08 and 0.10) pulled a σ pair into the space.

The start that would test this is one carried from a converged planar
solution, as his MECI search presumably carried its orbitals along the
walk. One alternative was considered and not used: projecting the
S0 minimum's natural orbitals through the AO overlap between the two
geometries. At a twisted ring that overlap measures the ring's rotation,
not its orbitals' character. The π projection above uses each atom's own
normal instead. The I-twisted config and the P-twisted second try are
checked in as the record of what was run. Their headers say they are not
his solutions.

#### Carried from the S0 minimum (#939)

#939 added that start: `[casscf] write_npy = true` writes a converged run's
orbitals and CI as `.npy`, and `orbitals` and `ci` start a later run from
them, orthonormalized in the new geometry's overlap
([`src/pipeline/casscf/README.md`](../../../../src/pipeline/casscf/README.md),
"Starting from another run's orbitals"). The S0 minimum's run is
`phbi_tz1211_s0.toml` with `write_npy = true`: its CASSCF converges as
before (8 macro), and its QD-NEVPT2 is the µEh-confirmed one above to
1e-11 Eh.

A p function points along the lab frame, not the ring, so a start carried
straight to a ring twisted by ~100° offers its π combinations partly
in-plane. So the twist was **walked**: eight geometries from the S0
minimum to each MECI (`phbi/geometries/walk/`), each CASSCF started from the
one before. They interpolate a Z-matrix built over the S0 bonds (bond
lengths and angles linearly, dihedrals along the shorter arc) rather than
the Cartesians: a Cartesian line through a 100° ring rotation shortens the
ring's distances to the axis by up to a third midway, where the Z-matrix
keeps every bond length between its two end values. The MECI is first
moved rigidly onto the S0 minimum (Kabsch, all atoms), and each step placed
by Kabsch onto the Cartesian blend, so the frame turns no more than the
molecule does. Step 8 is his geometry so moved; the energies do not depend
on the frame (the P-twisted point below reproduces the checked-in config's,
at his own frame, to 1e-11 Eh).

| walk | per step | max \|AᵀSA − I\| per step | SA-CASSCF at the MECI | Δ₀, Δ₁ (mEh) | E₁ − E₀, ours / his |
|---|---|---|---|---|---|
| P-twisted, 8 steps | 6 macro each, 61–126 s | 0.037–0.048 | −641.600091835673 | +3.68, +5.07 | 1.39 mEh / 11 µEh |
| P-twisted, one jump | 37 macro, 660 s | 0.37 | −641.600091835674 | +3.68, +5.07 | 1.39 mEh / 11 µEh |
| I-twisted, 8 steps | 6 macro each, 56–120 s | 0.043–0.049 | −641.633744174915 | +3.48, +6.28 | 2.82 mEh / 20 µEh |
| I-twisted, one jump | 33 macro, 848 s | 0.51 | −641.633744174911 | +3.48, +6.28 | 2.82 mEh / 20 µEh |

(A is the carried core and active orbitals; the step logs the largest
element of Aᵀ S A − I, how far they moved.) **Neither matches his.** Every
step of both walks converged in 6 macro iterations, its SA-CASSCF energy
moving smoothly (P: −641.6725 at S0, then −641.6708, −641.6655, … ,
−641.6162, −641.6001), so each walk followed one solution the whole way.
At the P-twisted MECI it is the π-projected second try's above, to 1e-11
Eh in both QD-NEVPT2 states: three different starts, one solution. At the
I-twisted MECI it is a third solution, 0.13 mEh above the π-projected
start's in state 0 and 4.9 mEh below it in state 1. The one jump straight
from S0 lands where the walk does, in five to six times the macro
iterations. So his two degenerate pairs are not the solutions continuously
connected to the S0 minimum's π space along this path; his MECI search
reached them by a route this does not reproduce (a walk at (12e,11o) from
another start, or a state average that differed along the way). Neither
twisted config is changed: the carried start does not reach his solution,
so they stay the record of #904's runs.

To regenerate, from the root of the tree, with `<out>` a fresh directory
(the `.npy` files, 2.1 MB of orbitals and 3.4 MB of CI each, stay out of
git):

```sh
P=docs/reproduction/park2019/single_points/phbi
sed 's/conv_tol_grad = 1e-9/&\nwrite_npy = true/' $P/phbi_tz1211_s0.toml > <out>/s0.toml
cp $P/geometries/phbi_tz1211_s0.xyz $P/geometries/walk/*.xyz <out>/
sed -i 's|geometries/||' <out>/s0.toml
bash -c ". devtools/lib.sh; with_gpu_card cuda build/src/free-breakfast -i <out> -o <out>/s0 <out>/s0.toml"
prev=s0
for k in 01 02 03 04 05 06 07 08; do  # ptw, then the same with itw
  sed -e "s|phbi_tz1211_s0.xyz|phbi_tz1211_ptw_walk$k.xyz|" \
      -e '/^active = /d' \
      -e "s|^write_npy = true|&\norbitals = \"$prev/casscf_orbitals.npy\"\nci = \"$prev/casscf_ci.npy\"|" \
      <out>/s0.toml > <out>/ptw$k.toml
  bash -c ". devtools/lib.sh; with_gpu_card cuda build/src/free-breakfast -i <out> -o <out>/ptw$k <out>/ptw$k.toml"
  prev=ptw$k
done
```

The steps before 08 need no `[nevpt2]` (delete that table to save a
minute each); the runs above had none.

## The scale test: (12e,11o) and up, on the A100

| | (12e,11o) | (12e,12o) | (14e,12o) | (12e,11o)/aug-cc-pVTZ |
|---|---|---|---|---|
| basis functions / fitting functions | 518 / 1,316 | 518 / 1,316 | 518 / 1,316 | 805 / 1,778 |
| determinants | 462² = 213,444 | 924² = 853,776 | 792² = 627,264 | 462² = 213,444 |
| CASSCF | 8 macro, \|g\| 9.1e-11, 44.0 s | 8 macro, \|g\| 5.5e-11, 126.8 s | 8 macro, \|g\| 4.5e-11, 122.5 s | 10 macro, \|g\| 3.8e-10, 82.1 s |
| NEVPT2 per root (RDMs + energy) | 18.6 s, 18.6 s | 31.0 s, 31.0 s | 30.3 s, 30.3 s | 9.8 s, 8.7 s |
| QD: diagonal walk over 2 roots, 1 pair walk | 39.5 s, 18.0 s | 61.3 s, 30.0 s | 60.9 s, 29.4 s | 23.3 s, 7.0 s |
| **wall, the whole run** | **141 s** | **283 s** | **276 s** | **136 s** |
| **RDM tiles** (`rdms`, per root) | **2** | **11** | **8** | **2** |
| **peak VRAM** (`nvidia-smi`) | **26.0 GiB** | **27.3 GiB** | **27.3 GiB** | **38.8 GiB** (39,713 MiB) |

The first three columns are the tree at 7960956, sampled every 0.5 s. The
aug-cc-pVTZ column is the tree at 33f2d63 (#905's fix included), sampled
every 0.1 s. The tree changed in between: at 33f2d63 the cc-pVTZ
(12e,11o) twisted points take 77–167 s for the whole run, and the
(14e,12o) lone-pair starts 215–237 s with 7 tiles. So the aug column's
NEVPT2 times are not comparable with the others.

- **The peak is a budget, not a need.** The NEVPT2 step gives the RDM
  build half the device's free memory (`rdm_budget = free_bytes / 2`,
  `src/pipeline/nevpt2/nevpt2.cpp`). The tile count is what fits in that
  budget, and the peak falls at the start of the RDM build in every
  cc-pVTZ run.
  A smaller card takes more tiles of the same build
  ([`src/nevpt2/rdm/README.md`](../../../../src/nevpt2/rdm/README.md),
  "The tile width is planned from bytes").
- **aug-cc-pVTZ peaks elsewhere.** Its 38.8 GiB peak lasts about a
  second, inside the DF three-center build before SCF iteration 1, and
  leaves 0.7 GiB of the card's 39.5 GiB. The CASSCF then holds 14.7 GiB
  and the NEVPT2 step at most 26.9 GiB, the RDM budget as above. So an
  aug-cc-pVTZ point at this size is at the edge of a 40 GB card, and the
  three-center build is the edge.
- **`rdms_adjoint` is not reached.** A single point takes no gradient, so
  only the forward `rdms` tiles are reported.
  [../REPRODUCTION.md](../REPRODUCTION.md) records that "neither the forward
  nor the adjoint has been timed at pHBI's (12e,11o)". The forward now has
  been: 18.6 s per root, CASSCF and RDMs included in under 2.5 minutes. The
  adjoint is still untimed.
- **Wall** is the binary's own run, from the `with_gpu_card` hold to its
  exit, with no wait for the card. SCF is 1.5 s of it in cc-pVTZ.
- The (4e,3o) points take 9–10 s each.

## What was run

- **Card and build.** NVIDIA A100-SXM4-40GB (CUDA, 39.49 GiB),
  the C++ PR gate (now `devtools/test.py pr`) `--preset default --no-test`,
  passing at `rev=7960956 preset=default`, the tree at 7960956. The
  aug-cc-pVTZ (12e,11o) point, the twisted (12e,11o) points and the
  (14e,12o) lone-pair starts were run on the same card at 33f2d63
  (passing at `rev=33f2d63 preset=default`, built `--no-test`). Every
  run held the card with `with_gpu_card cuda`, one at a time. Each
  checked-in config's numbers are from a run of that file as checked in.
- **Orbital checks.** The π-character metric, the aug-cc-pVTZ projection,
  the π projection for the twisted starts and the per-atom π populations
  were scratch scripts over the runs' `scf.molden` and `casscf.molden`.
  They read them with PySCF's `tools.molden.load`, and the projections
  used `gto.intor_cross` against PySCF's MINAO. None is checked in, as in
  [README.md](README.md).
- **The carried starts (#939)** ran on the same card at the tree of
  #939's PR (the PR gate, `--preset default`), one at a time under
  `with_gpu_card cuda`: the S0 minimum with `write_npy` (67 s), the two
  walks and the two one-step jumps, each final step with `[nevpt2]`. Each
  log ended `completed: 0 errors, 0 warnings`, with no PC refusal.
- Run outputs (logs, Molden files, `.npy` files, VRAM traces) stay out of
  git. The numbers above are quoted from them.
