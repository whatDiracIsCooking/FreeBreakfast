# Tier 2b: Figure 5, the g–h loop around the 13–14CI

Park's Figure 5 tests the method rather than a number. A loop of radius
0.002 Å is drawn in the branching plane of PSB3's 13–14CI, and the S1–S0 gap
is followed around it. His claim is that there is no artifact. The gap
oscillates by about 0.005 eV (~0.1 kcal/mol) around a maximum of about
0.014 eV (~0.3 kcal/mol), which is how MRCISD, XMCQDPT2 and XMS-CASPT2
behave. This tier repeats that loop around **our** converged 13–14CI from
[tier 2a](tier2a.md) (#848), with our own g and h (#849).

Every number here is from one card, CUDA (an RTX 3080, 9.6 GiB), from
`--preset default` at 446cf30 ([README.md](README.md), "How the runs are
made").

## The MECI and its branching space

The loop's centre x₀ is
[`geometries/psb3_1314_meci_ccpvdz.xyz`](geometries/psb3_1314_meci_ccpvdz.xyz).
It sits 0.0034 Å (Kabsch, all 14 atoms) from Park's 13–14 CI (SI pp.
S10–S11). That is more than the loop's radius, so the loop is drawn around
our MECI, not his.

g_P, g_Q and h_PQ come from the head of that walk's `optimize.json`, which
holds the last evaluation's vectors
([`../../../src/pipeline/optimize/README.md`](../../../src/pipeline/optimize/README.md),
"The QD-NEVPT2 MECI"). That run's `final.xyz` is identical, digit for digit,
to the checked-in geometry, so the vectors belong to x₀. They are copied,
unchanged, into [`tier2b/branching_space.json`](tier2b/branching_space.json).
The file also carries that evaluation's `gradient_norm`, `ci_residual_norm`,
s_o and s_c, so the scan can be regenerated from git alone.

| at x₀ | |
|---|---|
| ‖g_Q − g_P‖ (Eh/bohr) | 0.10520 |
| ‖h_PQ‖ (Eh/bohr) | 0.05362 |
| cos ∠(g_Q − g_P, h_PQ) | 0.277 |
| gap (Eh) | 3.83e-11 |
| s_o, s_c | 0.4264, 5.951 |

Both vectors sum to zero over the atoms (to 6e-13), so neither carries a
net translation.

## The plane

[`gh_loop.py`](gh_loop.py) writes the 24 points
x(θ) = x₀ + r (cos θ ĝ + sin θ ĥ), with r = 0.002 Å and θ = 0°, 15°, …,
345°, over the 3N Cartesians:

- **Orthonormalization.** ĝ = (g_Q − g_P)/‖g_Q − g_P‖. Then ĥ is h_PQ with
  its ĝ component removed and the result normalized: Gram–Schmidt, ĝ first.
  The two vectors are 74° apart, so the projection removes 28% of h_PQ's
  length along ĝ. At the crossing, u_P and u_Q are any orthonormal pair of
  the two-state space, so g_P, g_Q and h_PQ are each one choice of it. The
  plane they span is not (`src/nevpt2/intersection/README.md`), and the loop
  depends only on that plane, its orientation and r.
- **Signs.** ĝ points from P to Q: moving along +ĝ raises E_Q − E_P to first
  order. h_PQ is taken as the intersection signs it. Its sign follows the
  CI phases through R (each eigenvector is signed so that its largest
  component is positive), so it carries no physical sign. Flipping ĥ maps
  θ to −θ: the same loop, traversed the other way. The gap curve is then
  read backwards, and its amplitude and maximum do not change.

Run `python gh_loop.py` from this directory to regenerate
[`tier2b/`](tier2b/)'s `theta_NNN.xyz` and `theta_NNN.toml` byte for byte.

## Each point

Each point is one run of the binary, from the root of the tree:

```sh
bash -c '. devtools/lib.sh; with_gpu_card cuda build/src/free-breakfast \
  -i docs/reproduction/park2019 -o <out> docs/reproduction/park2019/tier2b/theta_NNN.toml'
```

- **The settings are the MECI's.** The point uses the same `[casscf]`
  (SA3-CASSCF(6,6), `conv_tol_grad = 1e-9`), DF basis and SCF as
  [`tier2a_psb3_1314_qdpc.toml`](tier2a_psb3_1314_qdpc.toml). Like it, the
  point names no `active`: the frontier 19–24 are the π set there.
- **The observable** is the `[nevpt2] quasi_degenerate = true` step's PC-QD
  H_eff eigenvalues. The step logs them as `PC-QD-NEVPT2 energies [...]` to
  12 decimals, 1e-12 Eh. That is eight orders below the 2e-4 Eh scale being
  measured, so those are the numbers used. (The per-state summary lines print
  10 decimals, which would also have been enough.)
- **Why each config also has an `[optimize]`.** The bound needs each point's
  CASSCF ‖g‖ and ρ. The `[nevpt2]` step writes no JSON, and the log carries
  ‖g‖ (to 4 digits) but not ρ, the Davidson residual. In the tree, only
  `optimize.json` records ρ. So each config also has an `[optimize] meci = [0, 1]`
  beside `[nevpt2_gradient] quasi_degenerate = true, variant = "pc"`, with
  every bar at 1 and `max_steps = 0`. It converges at its first evaluation,
  at x(θ) itself, and moves nothing. Its `optimize.json` entry holds that
  evaluation's `gradient_norm`, `ci_residual_norm`, s_o and s_c. The outer
  `[nevpt2]` then runs at `final.xyz`, which is the same geometry.
  - The two CASSCFs are separate solves with the same inputs. Their PC-QD
    gaps agree to 3.0e-11 Eh at every point (`heff_gap` against the outer
    step's eigenvalues), and their final ‖g‖ to within 2.1e-11 (the outer
    one read from its log). So the ‖g‖ and ρ quoted below describe the CASSCF under
    the reported gap to well within the bound they enter.
  - That evaluation is the intersection's, so each point also gets its own
    s_o and s_c. They are reported below as a check on the issue's
    assumption, not as a replacement for it.

Each point ran in about 16 s.

## The checks at every point

- **No PC refusal.** All 24 logs end `completed: 0 errors, 0 warnings`.
  None contains "PC refused", "SC kept" or "refused" anywhere. Each prints
  exactly one `PC-QD-NEVPT2 energies` line. So `pc_qd_refusal` was empty at
  every point, and every gap below is PC-QD's, not an SC-QD gap standing in
  silence. The intersection evaluation would also have stopped the run on a
  refusal, and none did.
- **The π character.** #860's script was run over each point's
  `casscf.molden`. The six active natural orbitals are π with minimum
  character 0.965 at every point, the MECI's value. Their natural
  occupations are 1.962–0.045.
- **The other states.** The third QD state is 0.1233–0.1238 Eh above the
  pair. The CASSCF roots are at least 0.0549 Eh apart.

## Results

| θ (°) | E₀ (Eh) | E₁ (Eh) | gap (Eh) | gap (meV) | CASSCF ‖g‖ | ρ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | −248.955020974579 | −248.954607289270 | 4.13685309e-4 | 11.257 | 6.83e-11 | 1.82e-10 |
| 15 | −248.954942952255 | −248.954504954147 | 4.37998108e-4 | 11.919 | 6.39e-11 | 1.75e-10 |
| 30 | −248.954892071978 | −248.954439951593 | 4.52120385e-4 | 12.303 | 6.12e-11 | 1.73e-10 |
| 45 | −248.954871187177 | −248.954417858064 | **4.53329113e-4** | **12.336** | 7.59e-11 | 1.99e-10 |
| 60 | −248.954881799491 | −248.954440369632 | 4.41429859e-4 | 12.012 | 6.22e-11 | 1.71e-10 |
| 75 | −248.954923903750 | −248.954505166663 | 4.18737087e-4 | 11.394 | 6.82e-11 | 1.70e-10 |
| 90 | −248.954996086508 | −248.954605991904 | 3.90094604e-4 | 10.615 | 6.20e-11 | 1.80e-10 |
| 105 | −248.955095743000 | −248.954733059467 | 3.62683533e-4 | 9.869 | 6.22e-11 | 1.62e-10 |
| 120 | −248.955218994859 | −248.954874142669 | 3.44852190e-4 | 9.384 | 6.52e-11 | 1.57e-10 |
| 135 | −248.955359775977 | −248.955016761194 | 3.43014783e-4 | 9.334 | 6.93e-11 | 1.78e-10 |
| 150 | −248.955508629855 | −248.955150790472 | 3.57839383e-4 | 9.737 | 5.01e-11 | 1.78e-10 |
| 165 | −248.955653177503 | −248.955269399883 | 3.83777620e-4 | 10.443 | 7.19e-11 | 2.02e-10 |
| 180 | −248.955780506964 | −248.955367953804 | 4.12553160e-4 | 11.226 | 7.40e-11 | 1.99e-10 |
| 195 | −248.955879450567 | −248.955442826518 | 4.36624049e-4 | 11.881 | 6.65e-11 | **2.45e-10** |
| 210 | −248.955941755101 | −248.955491079260 | 4.50675841e-4 | 12.264 | 7.57e-11 | 1.96e-10 |
| 225 | −248.955962526351 | −248.955510603354 | 4.51922997e-4 | 12.298 | 7.67e-11 | 2.11e-10 |
| 240 | −248.955940391381 | −248.955500293004 | 4.40098377e-4 | 11.976 | 7.88e-11 | 2.12e-10 |
| 255 | −248.955877551847 | −248.955460089462 | 4.17462385e-4 | 11.360 | 8.10e-11 | 2.19e-10 |
| 270 | −248.955779745301 | −248.955390892036 | 3.88853265e-4 | 10.581 | 8.04e-11 | 2.40e-10 |
| 285 | −248.955655984279 | −248.955294470130 | 3.61514149e-4 | 9.837 | 8.06e-11 | 2.43e-10 |
| 300 | −248.955517676265 | −248.955173768578 | 3.43907687e-4 | 9.358 | 8.28e-11 | 2.10e-10 |
| 315 | −248.955376624330 | −248.955034103761 | **3.42520569e-4** | **9.321** | **9.34e-11** | 1.80e-10 |
| 330 | −248.955242557293 | −248.954884609118 | 3.57948175e-4 | 9.740 | 7.35e-11 | 1.90e-10 |
| 345 | −248.955122312015 | −248.954737839228 | 3.84472787e-4 | 10.462 | 8.99e-11 | 2.28e-10 |

The gap is E₁ − E₀ over the PC-QD H_eff eigenvalues (eV = 27.211386245988
Eh). ‖g‖ and ρ are each point's CASSCF `gradient_norm` and
`ci_residual_norm`.

The curve is the cone's. It has two maxima (near 45° and 225°) and two
minima (near 135° and 315°), half a turn apart, and it is smooth: no
cusp, no jump, no near-zero. Over the loop, a linear two-state model built
from x₀'s own g and h, gap(θ) = r √((Δg·d)² + 4 (h·d)²) with
d = cos θ ĝ + sin θ ĥ, gives an extreme range of 12.35–9.29 meV. The scan
measures 12.34–9.32 meV, so the PC-QD surface over 0.002 Å is the cone that
x₀'s branching space predicts. The 15° sampling can miss the true extrema
by at most 1 − cos 15° ≈ 3% of the half amplitude, about 0.05 meV.

| | ours | Park (Figure 5) |
|---|---|---|
| maximum gap | **0.0123 eV** (4.533e-4 Eh) | ~0.014 eV |
| minimum gap | 0.0093 eV (3.425e-4 Eh) | – |
| oscillation amplitude, half peak-to-peak | **0.0015 eV** (5.54e-5 Eh) | ~0.005 eV |
| oscillation, peak-to-peak | 0.0030 eV (1.108e-4 Eh) | – |

The paper does not say whether "amplitude" means peak-to-peak or half of
it, so both are given. Each gate below is applied to the reading that is
harder to pass.

## The gates

**The bound.** The reported gap's distance from the exact surface's is at
most s_o ‖g‖ + s_c √n ρ (`src/nevpt2/intersection/README.md`, "The gap
sensitivities"), with n = 3 roots.

- It is formed from the MECI's s_o = 0.4264 and s_c = 5.951, and the worst
  scan point's values: ‖g‖ = 9.34e-11 (θ = 315°) and ρ = 2.45e-10
  (θ = 195°), each taken separately. That gives

      0.4264 × 9.34e-11 + 5.951 × √3 × 2.45e-10 = 2.56e-9 Eh (7.0e-8 eV)

- 20% of the smaller amplitude reading (half peak-to-peak, 5.54e-5 Eh) is
  1.11e-5 Eh. **The bound is 2.3e-4 of that: passed**, by more than three
  orders.
- **The assumption.** s_o and s_c were taken as the MECI's, assumed to vary
  negligibly over the 0.002 Å loop. Here that is not only assumed. Each
  point's own intersection gives s_o of 0.4234–0.4293 and s_c of
  5.751–6.061, within 2% and 3% of the MECI's. The worst bound formed point
  by point from each point's own s_o, s_c, ‖g‖ and ρ is 2.57e-9 Eh.
- So no tightening of `conv_tol_grad` below 1e-9 was needed, and that
  tightening stalls at PSB3 anyway (#867).

**The amplitude below 0.02 eV.** Even the larger reading, peak-to-peak
0.0030 eV, is under the bar: **passed**. The bar is 4× his 0.005 eV, the
slack our own MECI geometry needs and no more. The amplitude is a property
of the local topology over a 0.002 Å loop, where the frozen-core offset
(tier 2a: −14.9 mEh, the same to 0.24 mEh at all three PSB3 structures) is constant to high order.

So the claim under test holds: **the PC-QD-NEVPT2 surface is artifact-free
around the 13–14CI.**

- **The maximum gap,** 0.0123 eV against his ~0.014 eV, agrees to the
  precision a figure is read at.
- **Our oscillation is smaller than his.** It is 0.0015 eV half
  peak-to-peak (0.0030 eV peak-to-peak) against his ~0.005 eV. The
  oscillation is set by how unequal the cone's two axes are in the plane,
  that is by the lengths of g_Q − g_P and 2 h_PQ and the 74° between them.
  A 0.0034 Å difference in MECI geometry and all-electron against
  frozen-core correlation can each move that ratio. It is a property of the cone's shape,
  not a sign of an artifact, which would appear as a cusp or a collapse of
  the gap at some θ. Nothing of the kind is seen.

## Why the bar is not "an order below MS-CASPT2"

The ~1 eV MS-CASPT2 number in Park's discussion is not an artifact
amplitude that this scan could be compared against. At these PT2 MECIs he
found no method unphysical, MS-CASPT2 included: "the differential
correlation effect of ~1.0 eV moved the MECI to the region that the
non-invariance does not affect the PES" (p. 22). That ~1 eV is a shift from
correlation, and the artifacts of his Figure 6(b) sit at the **CASSCF**
MECIs, not at the PT2 ones. A bar such as 0.05 eV, an order below that
scale, would discriminate against nothing he measured. The claim tested
here is "artifact-free", not "equals 0.005 eV". It is tested by the
amplitude bar (4× his number) together with the bound that shows the
amplitude is resolved.

## Files

- [`gh_loop.py`](gh_loop.py) is the generator, stdlib only.
- [`tier2b/branching_space.json`](tier2b/branching_space.json) holds the
  MECI's g_P, g_Q, h_PQ and its last evaluation's norms and sensitivities.
- [`tier2b/theta_NNN.xyz`](tier2b/) and `theta_NNN.toml` are the 24 points
  and their configs, as `gh_loop.py` writes them.

**Rerun after #874.** The tables are from a build at 446cf30. Main then
took #874 (8516b9f), which replaces PC's rule against non-positive
denominators with one against poles. All 24 points were rerun on that
build at 97207e7:

- every one completed with no refusal and the same 0.965 minimum π
  character;
- every gap agrees with the table above to 5.2e-11 Eh;
- the bound formed from that rerun's worst ‖g‖ and ρ is 2.39e-9 Eh.

Run outputs (logs, molden files, each point's `optimize.json`) stay out of
git. No C++ changed, and no ctest was run for this tier.
