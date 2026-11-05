# Reproducing Park's QD-NEVPT2 MECIs

The reproducible part of J. W. Park, *Analytical Gradient Theory for
Quasidegenerate N-Electron Valence State Perturbation Theory (QD-NEVPT2)*,
*J. Chem. Theory Comput.*,
[DOI 10.1021/acs.jctc.9b00919](https://doi.org/10.1021/acs.jctc.9b00919):
his minimum-energy conical intersections (MECIs), reoptimized here. This
directory holds the configs, the starting and converged geometries and the
results; run outputs (molden files, logs, `optimize.json`) stay out of git,
and the numbers that matter are quoted below.

## The method is his

His "QD-NEVPT2" is **SA-QD-PC-NEVPT2** (his p. 11): PC contraction, the
*state-averaged* generalized Fock in Dyall's H, a state-specific projection
manifold, `H_eff = diag(E_M) + ½ (C + Cᵀ)`, and the analytic gradient by a
Lagrangian with a Z-vector, density-fitted throughout. That is what this tree
computes: QD is built on the CASSCF's own orbitals, core and virtuals
canonical over the state-averaged Fock
([`src/pipeline/nevpt2/README.md`](../../../src/pipeline/nevpt2/README.md)),
with `H_IJ = E_I δ_IJ + ½ (C_IJ + C_JI)`; `qd_pc_gradient`'s W form is his
eq. 20, and `nevpt2::qd_pc_intersection` builds his g_P, g_Q and h_PQ from
one set of H_eff eigenvectors
([`src/nevpt2/intersection/README.md`](../../../src/nevpt2/intersection/README.md)).
A config asks for it with `[nevpt2_gradient] quasi_degenerate = true,
variant = "pc"` beside `[optimize] meci = [P, Q]`
([`src/pipeline/optimize/README.md`](../../../src/pipeline/optimize/README.md),
"The QD-NEVPT2 MECI").

## What cannot match, and so what is compared

- **Frozen core.** He freezes the core everywhere (p. 18); this tree has no
  such option, and correlates every inactive orbital. That leaves a floor of
  about 0.02–0.05 eV under any absolute MECI energy, so each tier's bar is a
  trend, a sign or a bound that cancels a constant offset, never an absolute
  energy.
- **The f ceiling.** The orbital basis stops at f (`kMaxAngularMomentum = 3`,
  `src/basis/angular_momentum.h`), and `data/basis/` carries nothing past
  cc-pVTZ: Table 2's cc-pVQZ and cc-pV5Z columns, and his cc-pV5Z PSB3
  result, are out of reach.
- **No reference geometries.** The paper has no Supporting Information, so
  every MECI here is our own optimization, compared through the energies of
  his Tables 1–2 and the amplitude of his Figure 5.

## How the runs are made

Every number in this directory comes from one card, CUDA (an RTX 3080), from
the tree's `--preset default` build (`build/`): the later tiers' bars are
assembled across runs, and a table whose rows came from two backends would
carry a cross-backend difference inside the gated quantity. A run holds the
whole card (`devtools/lib.sh`'s `with_gpu_card cuda`). From the root of the
tree:

```sh
bash -c '. devtools/lib.sh; with_gpu_card cuda build/src/free-breakfast \
  -i docs/reproduction/park2019 -o <out> docs/reproduction/park2019/<config>.toml'
```

Each config's `xyz` resolves against `-i`, and every file the run writes
lands in `-o`.

## Tier 0: does PC-QD survive a MECI walk at (2e,2o)?

[`tier0_ethylene_pc.toml`](tier0_ethylene_pc.toml) is
[`examples/qd_meci.toml`](../../../examples/qd_meci.toml) with
`variant = "pc"`: the MECI of QD states 0 and 1 over ethylene's
DF-SA3-CASSCF(2,2)/cc-pVDZ (cc-pVDZ-JKFIT), from the twisted (80°) and
pyramidalized start of [`examples/ethylene.xyz`](../../../examples/ethylene.xyz),
converged on `max_gap = 1e-7` Eh and Opt=Tight's four criteria, with
`[casscf] conv_tol_grad = 1e-10`.

The question it answers gates the rest of this directory. PC refuses a class
whose metric has no gap at its cut, or a non-positive denominator
([`src/nevpt2/energy/README.md`](../../../src/nevpt2/energy/README.md), "The
cut, and refusals"), and at (2e,2o) the class bases are tiny, so whether that
rule bites along a walk is an empirical question. In the `[optimize]` path a
refusal is loud: `qd_pc_intersection` has no SC to fall back on, and the
walk stops with the step's error. (`pc_qd_refusal`, the silent SC fallback,
is the `[nevpt2]` energy step's, and `optimize.json` has no such key.) So the
walk converging *is* the statement that PC was refused at no geometry.

**Finding: PC was not refused.** The walk converged, `completed: 0 errors,
0 warnings`, in 83 evaluations (about a minute of wall clock), with no
refusal at any of them.

| | at the start (evaluation 0) | at the MECI (evaluation 82) |
|---|---|---|
| mean E of states 0, 1 (Eh) | −78.1186616688 | **−78.1573045477** |
| `heff_gap` (Eh) | 4.10e-2 | **8.07e-8** (`max_gap` 1e-7) |
| `gradient_norm` (CASSCF) | 5.23e-12 | **2.96e-11** |
| `ci_residual_norm` (CASSCF) | 3.32e-14 | **4.61e-14** |
| `pair_overlap` | – | 0.99999999999 |
| max / rms force (Eh/bohr) | 5.11e-2 / 2.56e-2 | 5.67e-7 / 2.51e-7 |
| max / rms displacement (bohr) | – | 1.00e-5 / 4.86e-6 |
| third-state margin E₂ − E₁ (Eh) | 0.152 | 0.181 |
| CASSCF smallest root gap (Eh) | 0.0547 | 0.0294 |

Over the whole trail: `pair_overlap` never fell below 0.99522 (evaluation 1;
the floor is 0.9), `gradient_norm` stayed at or below 9.8e-11 and
`ci_residual_norm` at or below 1.0e-13, the third QD state stayed at least
0.152 Eh above the pair, the CASSCF roots at least 0.0236 Eh apart, and the
branching space was two-dimensional at every evaluation. The gap fell to
1.4e-6 Eh by evaluation 9; from evaluation 16 to 72 the trust radius sat at
0.004–0.03 bohr while the mean energy crept down along the seam (gap
5e-6–4e-5 Eh) before the last ten steps converged it.

The last evaluation's `gradient_norm` and `ci_residual_norm` are recorded so
the reported gap's distance from the exact surface's, `s_o ‖g‖ + s_c √n ρ`
([`src/nevpt2/intersection/README.md`](../../../src/nevpt2/intersection/README.md)),
can be stated once the sensitivities s_o and s_c reach the output (#844).

Beside it, the same walk over SC (`examples/qd_meci.toml` unchanged, same
card and build) converged in 97 evaluations to a mean of −78.1572756343 Eh,
gap 7.80e-8: the PC MECI lies 2.89e-5 Eh (0.79 meV) below the SC one, and
the two start 6.4e-5 Eh apart, which also shows the `variant` took effect.

Converged geometry (Å, `final.xyz`, mean E −78.157304547696 Eh, gap
8.073e-8):

```
C      0.046370034846     0.018453948351    -0.102635276489
C      0.065084191963     0.100905211926     1.325390063514
H      0.989650725466    -0.023331787445    -0.674831958634
H     -0.856561421789    -0.111165527967    -0.732730319866
H     -0.788546674608     0.705034699075     1.725508806900
H     -0.550562395078    -0.882897980640     1.383319889575
```

Card and build: RTX 3080 (CUDA, 9.6 GiB), `--preset default`, at 79b2bc7.

[`single_points/`](single_points/README.md) (#860) measures the all-electron core floor under tiers 1 and 3 with single points at his SI geometries.

[`tier2a.md`](tier2a.md) (#848) converges PSB3's FC geometry, 13–14CI and 11–12CI at QD-PC-NEVPT2/cc-pVDZ.

[`tier3.md`](tier3.md) (#850) reproduces Table 2 in cc-pVDZ and cc-pVTZ: E(13–14CI) − E(11–12CI) flips sign, −0.195 eV to +0.108 eV.

[`tier2b.md`](tier2b.md) (#849) runs Figure 5's 0.002 Å g–h loop around the 13–14CI: maximum gap 0.0123 eV, oscillation 0.0015 eV (half peak-to-peak), no PC refusal, bound 2.6e-9 Eh.
