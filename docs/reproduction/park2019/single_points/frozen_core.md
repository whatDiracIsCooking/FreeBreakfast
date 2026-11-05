# Single points at Park's SI geometries: frozen core

Phase 2 of #860, run on #859's `[nevpt2] frozen_core`. These are the same
SA3-QD-PC-NEVPT2 single points as [phase 1](README.md), at Park's own
geometries, with one change: the C and N 1s are frozen
(`frozen_core = "auto"`, 2 orbitals for ethylene and 6 for PSB3). He froze
the core too ("The frozen core approximation is invoked for all the
calculations", acs.jctc.9b00919 §4). So Δ = E_ours − E_his no longer
includes his missing core correlation. What is left is aux-set, density
fitting and convergence differences.

**The prediction was that |Δ| would fall to the aux and convergence level
and become nearly constant. It does more than that: Δ is zero to within
0.52 µEh at every point and every state.** The largest |Δ| is 0.52 µEh
(PSB3 cc-pVTZ 11–12 CI, state 1). The largest spread over the geometries,
for one state, is 0.014 meV. In phase 1, all-electron, the mean was
−5.4 to −92.9 mEh and the spread was 3.7 to 46 meV.

## Δ against the all-electron floor

Δ is PC-QD − his PC-QD at all 17 points. That includes ethylene's S0 minimum,
which is PC since #874. The all-electron columns are phase 1's
([README.md](README.md)): its floor rows (PC MECIs only, no S0), and for
ethylene's S0 the PC-QD Δ that `src/nevpt2/energy/README.md` "Poles, not
signs" measured after #874.

| molecule, basis | points | mean Δ, states 0 / 1 / 2 (µEh) | spread of Δ (meV) | all-electron mean Δ (mEh) | all-electron spread (meV) |
|---|---|---|---|---|---|
| ethylene, aug-cc-pVDZ | S0, Py, Et, Hm | −0.02 / +0.02 / +0.02 | 0.002 / 0.003 / 0.009 | −5.42 / −5.40 / −5.47 | 3.7 / 5.0 / 3.7 |
| ethylene, aug-cc-pVTZ | S0, Py, Et, C3v, Hm | +0.00 / −0.00 / −0.01 | 0.0005 / 0.002 / 0.004 | −35.12 / −35.10 / −35.17 | 19.4 / 21.5 / 18.8 |
| PSB3, cc-pVDZ | S0, S1, 11–12 CI, 13–14 CI | −0.06 / −0.06 / −0.02 | 0.008 / 0.002 / 0.003 | −14.94 / −14.96 / −14.96 | 8.0 / 8.1 / 7.9 |
| PSB3, cc-pVTZ | S0, S1, 11–12 CI, 13–14 CI | −0.07 / −0.14 / −0.06 | 0.005 / 0.014 / 0.011 | −92.74 / −92.81 / −92.88 | 39.8 / 46.1 / 42.7 |

Over all states of a molecule and basis, the spread is 0.009 meV (ethylene
aug-cc-pVDZ), 0.004 meV (aug-cc-pVTZ), 0.008 meV (PSB3 cc-pVDZ) and
0.016 meV (cc-pVTZ). The all-electron S0 point, which phase 1 could not
include, had Δ = −5.515, −5.505, −5.497 mEh (aug-cc-pVDZ) and −35.548,
−35.446, −35.374 mEh (aug-cc-pVTZ). With the core frozen it is
−0.00, +0.09, +0.21 µEh and +0.00, −0.04, −0.11 µEh.

Ethylene aug-cc-pVDZ C3v is not run: its RHF is blocked by #865, as in
phase 1. The other 17 of the 18 points are here.

## The points

E₀ is ours, from the step's `PC-QD-NEVPT2 energies` log line at 12
decimals. E₀ his is from the SI at 10 decimals, as in each config's header.
Δ is in µEh.

### Ethylene, aug-cc-pVDZ

| geometry | E₀ ours (Eh) | E₀ his (Eh) | Δ₀ | Δ₁ | Δ₂ |
|---|---|---|---|---|---|
| S0 min | −78.3344531016 | −78.3344531014 | −0.00 | +0.09 | +0.21 |
| Py MECI | −78.1717633544 | −78.1717632797 | −0.07 | +0.02 | −0.13 |
| Et MECI | −78.1616020793 | −78.1616020687 | −0.01 | −0.02 | +0.01 |
| C3v MECI | blocked by #865 | – | – | – | – |
| Hm MECI | −78.1468050794 | −78.1468050753 | −0.00 | −0.01 | −0.00 |

### Ethylene, aug-cc-pVTZ

| geometry | E₀ ours (Eh) | E₀ his (Eh) | Δ₀ | Δ₁ | Δ₂ |
|---|---|---|---|---|---|
| S0 min | −78.4113893792 | −78.4113893792 | +0.00 | −0.04 | −0.11 |
| Py MECI | −78.2456878964 | −78.2456878972 | +0.00 | +0.02 | +0.03 |
| Et MECI | −78.2371466542 | −78.2371466703 | +0.02 | +0.01 | −0.01 |
| C3v MECI | −78.2329478860 | −78.2329478858 | −0.00 | −0.00 | +0.00 |
| Hm MECI | −78.2229819408 | −78.2229819395 | −0.00 | +0.01 | +0.01 |

### PSB3, cc-pVDZ

| geometry | E₀ ours (Eh) | E₀ his (Eh) | Δ₀ | Δ₁ | Δ₂ |
|---|---|---|---|---|---|
| S0 min (FC) | −249.0491104086 | −249.0491104000 | −0.01 | −0.03 | +0.03 |
| S1 min | −249.0354435779 | −249.0354436267 | +0.05 | −0.02 | −0.01 |
| 11–12 CI | −248.9329876960 | −248.9329874544 | −0.24 | −0.10 | −0.03 |
| 13–14 CI | −248.9403471047 | −248.9403470698 | −0.03 | −0.09 | −0.08 |

### PSB3, cc-pVTZ

| geometry | E₀ ours (Eh) | E₀ his (Eh) | Δ₀ | Δ₁ | Δ₂ |
|---|---|---|---|---|---|
| S0 min (FC) | −249.2912816419 | −249.2912816246 | −0.02 | −0.02 | +0.08 |
| S1 min | −249.2840999920 | −249.2840999522 | −0.04 | −0.04 | +0.05 |
| 11–12 CI | −249.1781979902 | −249.1781978062 | −0.18 | −0.52 | −0.32 |
| 13–14 CI | −249.1754770828 | −249.1754770559 | −0.03 | −0.00 | −0.03 |

The two largest misses are at the PSB3 11–12 CI, in both bases, and are
still under 0.6 µEh. Phase 1's E₁ − E₀ there, 0.235 meV (cc-pVDZ) and 1.981 meV (cc-pVTZ)
against his 0.070 and 0.426 meV, is now 0.074 and 0.417 meV.

## The tiers' quantities at his geometries

Phase 1's table, again with the core frozen. A MECI's energy is the mean of
states 0 and 1.

| quantity | ours | his | ours − his |
|---|---|---|---|
| tier 3, E(13–14 CI) − E(11–12 CI), cc-pVDZ | −0.20015 eV | −0.20016 eV | +0.01 meV |
| tier 3, E(13–14 CI) − E(11–12 CI), cc-pVTZ | +0.07392 eV | +0.07392 eV | 0.00 meV |
| tier 1, S0-referenced shift aug-cc-pVDZ → aug-cc-pVTZ, Py | +0.08187 eV | +0.08187 eV | +0.001 meV |
| the same, Et | +0.03791 eV | +0.03790 eV | +0.001 meV |
| the same, Hm | +0.02048 eV | +0.02048 eV | +0.000 meV |
| the same, C3v | – (#865 at aug-cc-pVDZ) | +0.0162 eV | – |
| MECI − Py MECI shift, Et | −0.04397 eV | −0.04397 eV | 0.00 meV |
| MECI − Py MECI shift, Hm | −0.06139 eV | −0.06139 eV | 0.00 meV |

Phase 1's +31.4 meV on tier 3's cc-pVTZ value, and its −4.7 and −16.2 meV
on the MECI − Py shifts, were the core floor. With the core frozen, what is
left is under 0.01 meV. Every S0-referenced gap in phase 1's "Relative
energies beside his" tables (E₀ − E₀(S0 min)) now matches his to the
fourth decimal in eV. The verdicts on the tiers belong to the milestone
owner, as before.

## What was run

- **Configs.** Each `<stem>.toml` of this directory but
  `ethylene_adz_c3v`, copied with `frozen_core = "auto"` appended to its
  `[nevpt2]` table, nothing else changed. The committed configs stay
  all-electron, phase 1's. The run log of each point reports
  `NEVPT2: 2 frozen core orbitals, 5 correlated` (ethylene) or
  `6 frozen core orbitals, 13 correlated` (PSB3). Run from the root of
  the tree:

  ```sh
  for f in docs/reproduction/park2019/single_points/*.toml; do
    { cat "$f"; echo 'frozen_core = "auto"'; } > "<dir>/$(basename "$f")"
  done
  bash -c '. devtools/lib.sh; with_gpu_card cuda build/src/free-breakfast \
    -i docs/reproduction/park2019/single_points -o <out> <dir>/<stem>.toml'
  ```

- **Card and build.** RTX 3080 (CUDA, 10 GiB), `devtools/cpp-tier.sh
  --preset default --no-test`, the tree at #859's branch (8516b9f plus the
  change). All 17 points ran under one `with_gpu_card cuda` hold.
- **Status.** All 17 logs end `completed: 0 errors, 0 warnings`, with the
  three `PC-QD-NEVPT2 state` lines and no refusal.
- **Wall time.** 1–4 s for each ethylene point, 4–5 s (cc-pVDZ) or 13–17 s
  (cc-pVTZ) for each PSB3 point.
- Run outputs (logs, Molden files) stay out of git. The numbers above are
  quoted from them.
