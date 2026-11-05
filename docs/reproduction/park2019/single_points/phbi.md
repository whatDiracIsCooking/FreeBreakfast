# Single points at Park's SI geometries: pHBI

#904. pHBI is the GFP chromophore model that milestone 26 left out of scope
([../REPRODUCTION.md](../REPRODUCTION.md), "What could not be done"). Each
point below is an SA2-QD-PC-NEVPT2 single point at one of Park's own SI
geometries, with the core frozen as his was, compared with the absolute
energies he lists beside it. It is the method of
[frozen_core.md](frozen_core.md), which matched ethylene and PSB3 to
≤ 0.52 µEh, taken to (12e,12o) and 518 basis functions.

**Five of the SI's ten pHBI points match his to ≤ 0.14 µEh in both
states.** These are (4e,3o) at all three structures, and (12e,11o) and
(12e,12o) at the S0 minimum, all in cc-pVTZ. The other five did not match
or did not run:

- **(14e,12o)/cc-pVTZ** ran, but **misses by mEh**. Three all-π choices of
  the space give +0.53, −6.05 and −4.24 mEh for state 0, so his (14e,12o)
  is not one of them. Below, "(14e,12o)", says why it was not pursued
  further.
- **(12e,11o)/aug-cc-pVTZ** is **refused for device memory** at CASSCF on
  the 40 GB A100 (#905).
- **(14e,12o)/aug-cc-pVTZ**, and **(12e,11o)/cc-pVTZ at the I- and
  P-twisted MECIs**, were **not run**. In both cases the active space could
  not be named from the paper and orbital character, and a guessed space
  would only measure the guess.

| point | Δ₀ (µEh) | Δ₁ (µEh) | status |
|---|---|---|---|
| (4e,3o)/cc-pVTZ S0 min | −0.00 | +0.04 | reproduced |
| (4e,3o)/cc-pVTZ I-twisted MECI | −0.05 | −0.08 | reproduced |
| (4e,3o)/cc-pVTZ P-twisted MECI | −0.14 | −0.03 | reproduced |
| (12e,11o)/cc-pVTZ S0 min | +0.04 | +0.11 | reproduced; the scale test |
| (12e,12o)/cc-pVTZ S0 min | +0.02 | +0.02 | reproduced |
| (14e,12o)/cc-pVTZ S0 min | +526.00 | +1775.32 | ran; not his active space |
| (12e,11o)/aug-cc-pVTZ S0 min | – | – | refused at CASSCF for device memory (#905) |
| (14e,12o)/aug-cc-pVTZ S0 min | – | – | not run: space unsettled, and #905 |
| (12e,11o)/cc-pVTZ I-, P-twisted MECIs | – | – | not run: π* not identifiable at twisted geometries |

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
| `phbi_tz1211_{itw,ptw}` (geometries only) | I-, P-twisted MECIs, (12e,11o) | cc-pVTZ | S29–S31 |
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
| (14e,12o) S0 min, all-π (not his) | −644.063112439751 | −644.0636384427 | +526.00 | −643.954903747327 | −643.9566790714 | +1775.32 |

The vertical gaps, E₁ − E₀ at the S0 minimum, as the paper quotes them
(acs.jctc.9b00919 §4, pHBI): (4e,3o) 2.8110 eV ours, 2.8110 his ("2.81");
(12e,11o) 2.9461 / 2.9461 ("2.95"); (12e,12o) 3.0633 / 3.0633 ("3.06").
At the MECIs the (4e,3o) gaps are 0.4 meV (I) and 0.2 meV (P), his to
the digit.

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
program's choice, not stated in the paper. This page does not guess
further. The config is checked in as the record of the first row,
and its header says it is not his space.

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

That run did not reach CASSCF's first iteration (#905):

```
[error] [casscf.cpp:177] casscf: the fitted integrals over 805 orbitals, 11 active and 1778 fitting functions need 2133771640 bytes of device memory past B's 9217507600, and device 0 has 760807424 free
completed: 1 errors, 0 warnings
```

B is 9.2 GB and CASSCF asks for 2.1 GB more, but the process already
holds 39.7 GB of the card's 39.5 GiB. An RHF-only run fills the card
before SCF iteration 1. #905 has the measurements. The start above is
therefore untested. (14e,12o) in aug-cc-pVTZ inherits both this and the
cc-pVTZ (14e,12o) question.

### (12e,11o) at the twisted MECIs

At his (12e,11o) I- and P-twisted geometries, the RHF virtuals do not
separate into π and σ. Between MOs 49 and 109, the only virtual with π
character above 0.3 is 49, and the rest are ≤ 0.26. So the five π* cannot
be named from orbital character, and PSB3's twisted-virtual check (pick,
then verify on `casscf.molden`) has nothing to pick from. These points were
not run. The geometries are checked in for whoever settles it.

## The scale test: (12e,11o) and up, on the A100

| | (12e,11o) | (12e,12o) | (14e,12o) |
|---|---|---|---|
| determinants | 462² = 213,444 | 924² = 853,776 | 792² = 627,264 |
| CASSCF | 8 macro, \|g\| 9.1e-11, 44.0 s | 8 macro, \|g\| 5.5e-11, 126.8 s | 8 macro, \|g\| 4.5e-11, 122.5 s |
| NEVPT2 per root (RDMs + energy) | 18.6 s, 18.6 s | 31.0 s, 31.0 s | 30.3 s, 30.3 s |
| QD: diagonal walk over 2 roots, 1 pair walk | 39.5 s, 18.0 s | 61.3 s, 30.0 s | 60.9 s, 29.4 s |
| **wall, the whole run** | **141 s** | **283 s** | **276 s** |
| **RDM tiles** (`rdms`, per root) | **2** | **11** | **8** |
| **peak VRAM** (`nvidia-smi`, 0.5 s) | **26.0 GiB** | **27.3 GiB** | **27.3 GiB** |

- **The peak is a budget, not a need.** The NEVPT2 step gives the RDM
  build half the device's free memory (`rdm_budget = free_bytes / 2`,
  `src/pipeline/nevpt2/nevpt2.cpp`). The tile count is what fits in that
  budget, and the peak falls at the start of the RDM build in every run.
  A smaller card takes more tiles of the same build
  ([`src/nevpt2/rdm/README.md`](../../../../src/nevpt2/rdm/README.md),
  "The tile width is planned from bytes").
- **`rdms_adjoint` is not reached.** A single point takes no gradient, so
  only the forward `rdms` tiles are reported.
  [../REPRODUCTION.md](../REPRODUCTION.md) records that "neither the forward
  nor the adjoint has been timed at pHBI's (12e,11o)". The forward now has
  been: 18.6 s per root, CASSCF and RDMs included in under 2.5 minutes. The
  adjoint is still untimed.
- **Wall** is the binary's own run, from the `with_gpu_card` hold to its
  exit, with no wait for the card. SCF is 1.5 s of it.
- The (4e,3o) points take 9–10 s each.

## What was run

- **Card and build.** NVIDIA A100-SXM4-40GB (CUDA, 39.49 GiB),
  `devtools/cpp-tier.sh --preset default --no-test`
  (`cpp-tier PASS rev=7960956 preset=default`), the tree at 7960956. Every
  run held the card with `with_gpu_card cuda`, one at a time.
- **Orbital checks.** The π-character metric and the aug-cc-pVTZ
  projection were scratch scripts over the runs' `scf.molden` and
  `casscf.molden`. The projection read them with PySCF's
  `tools.molden.load` and `gto.intor_cross`. Neither is checked in, as in
  [README.md](README.md).
- Run outputs (logs, Molden files, VRAM traces) stay out of git. The
  numbers above are quoted from them.
