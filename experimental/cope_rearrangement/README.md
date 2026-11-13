# `cope_rearrangement/`

A study of the chair Cope rearrangement of 1,5-hexadiene. The question is
whether it goes through one concerted C2h TS once dynamic correlation is
included. CASSCF(6,6) alone overstabilises the cyclohexane-1,4-diyl; NEVPT2
is expected to remove that artifact. The deciding test is the sign of the
a_u curvature at the C2h stationary point.

Steps 1 and 2 are PySCF (and block2) references. From step 4 on, the PT2-level
work runs on **free-breakfast's own DF-PC-NEVPT2 gradient and Hessian**.

Nothing here is built: each step is Python generators, TOML configs and
their output. The PySCF steps run under `uv run --group tools`, which carries
PySCF, geomeTRIC and block2.

## What the study is for

The chemistry is settled: the chair Cope goes through one concerted TS
([Staroverov & Davidson 2001](#references), [Graulich 2011](#references)).
This study adds no new insight into the mechanism. What it adds is a
demonstration that PC-NEVPT2, through our GPU implementation, gets a known
hard case right. Every geometry, frequency and the IRC is done at the
PT2 level, which most studies skipped because PT2 gradients were expensive.

Two claims, each with its own evidence:

| Claim | Evidence |
|---|---|
| **The implementation is correct** | Agreement with an independent code at fixed geometries: PC-NEVPT2 energies against step 2's block2 numbers (same DF-CASSCF(6,6)/cc-pVDZ, same cc-pVDZ-JKFIT), and the PC-NEVPT2 gradient against BAGEL or ORCA at the final TS. The Cope result itself cannot show this: a wrong but smooth surface looks just as plausible. |
| **The method recovers the concerted mechanism** | At the PT2-optimised C2h stationary point, exactly one imaginary frequency, of a_u symmetry; the IRC from it reaches reactant and product with nothing in between. Contrasted with CASSCF's two C2h stationary points. |

The sentence the study is meant to support:

> With analytic PC-NEVPT2 gradients and Hessians from our GPU implementation
> (agreeing with BAGEL/ORCA to X at the TS), PC-NEVPT2 recovers the concerted
> chair mechanism that CASSCF(6,6) misses: a single C2h stationary point with
> one imaginary a_u frequency, connected directly to reactant and product.

**"Concerted" is not enough on its own.** CASPT2 also found one concerted TS,
but with R ≈ 1.75 Å, too short
([Hrovat, Morokuma & Borden 1994](#references); [Kozlowski, Dupuis & Davidson 1995](#references)).
MR-CISD and MR-AQCC put it nearer 1.9–2.0 Å ([Ventura et al. 2003](#references)).
So the study also reports:
- R at the TS and the electronic barrier, against CCSD(T) at the same geometry;
- ΔH‡ against experiment (≈ 33.5 kcal/mol).

A concerted TS at too short an R, as with CASPT2, is still a publishable
answer.

**Prior GPU NEVPT2 derivative work exists**
([Guerrero 2026](#references): DMRG-QD-NEVPT2 gradients and NACs on a
consumer GPU). Check the literature for a published NEVPT2-gradient TS
optimisation of the Cope before claiming that part is new.

## Steps

| Step | What | Code | State |
|---|---|---|---|
| [`step_1/`](step_1/README.md) | C2h chair guess, symmetry-labelled CAS(6,6) (2a_g + 2b_u + a_u + b_g), relaxed C2h DF-CASSCF(6,6)/cc-pVDZ scan of R16 = R34, 1.50–2.60 Å | PySCF | done: diyl minimum at 1.65 Å, "aromatic" minimum at 2.30 Å |
| [`step_2/`](step_2/README.md) | PC-NEVPT2 (block2) and SC-NEVPT2 along the scan; the a_u Hessian (CASSCF full, NEVPT2 in a 3-D subspace) | PySCF, block2 | partial: E(R) at all 23 points, one PC-NEVPT2 minimum at 1.95 Å; a_u curvature negative there (concerted), run at 1.95 and 1.65 Å only |
| [`step_3/`](step_3/README.md) | 2D (R16, R34) surface under C2, relaxed at PC-NEVPT2 and at CASSCF (both 1.50–2.80 Å at 0.05 Å), with `[optimize]` distance constraints | free-breakfast | done: PC has one stationary point, a saddle on the diagonal at 1.922 Å (step 4b's TS to 1e-3 Å, 0.002 kcal/mol), and no diyl well; CASSCF has the diyl minimum (1.65 Å, 4b's to 0.009 kcal/mol) flanked by two off-diagonal saddles (1.67/1.82 Å, 1.07 kcal/mol above it), a diagonal maximum (1.89 Å) and the aromatic saddle (2.28 Å). Step 4d's IRC runs down the PC valley. All 756 points converged; 6.9 h on a 3080. [`comparison.svg`](step_3/comparison.svg) has both surfaces and the diagonal cut side by side |
| [`step_4/`](step_4/README.md) 4a | implementation check: free-breakfast DF-PC-NEVPT2 energies at step 1's geometries against step 2's block2 numbers | free-breakfast | done: CASSCF to 7e-12, PC to 1.1e-7 against step 2 (its convergence); to 4e-10 on the same orbitals and CI. ~1.5 s a point on a 3080 |
| [`step_4/`](step_4/README.md) 4b | PT2-level C2h stationary point, analytic Hessian, mode-displaced minimisations (the IRC check); the CASSCF contrast | free-breakfast | done: PC-NEVPT2 has one C2h stationary point, R = 1.922 Å, one imaginary frequency (475i cm⁻¹, a_u); ± along it reaches both mirror-image 1,5-hexadiene minima; barrier 31.2 kcal/mol. CASSCF: diyl minimum at 1.642 Å, a_u saddle at 2.284 Å. The CASSCF 2.30 Å walk left C2h (negative a_u curvature amplifying round-off) |
| step 4c | PC-NEVPT2 gradient against BAGEL or ORCA at step 4b's TS | BAGEL / ORCA | not started |
| [`step_4/`](step_4/README.md#step-4d-the-irc) 4d | the IRC from 4b's TS, both sides, at DF-PC-NEVPT2 (`[irc]`, GS2), replacing 4b's mode-displaced walks; each end then polished | free-breakfast | done: E falls monotonically by 31.2 kcal/mol on both sides into the hexadiene minima (0.002 and 0.010 kcal/mol short, 2 and 5 steps, the sphere threshold's resolution), \|g\| has a single maximum, max(R16, R34) ≥ 1.922 Å throughout (no diyl region), C2 kept to 3e-4 bohr; each end polishes to 4b's `pc_plus`/`pc_minus` (RMSD ≤ 1.2e-4 Å). Final run on main after #1300; three runs as `[irc]` was fixed (#1278, #1294, #1295) |
| step 5 | CCSD(T) at the PT2 TS; cc-pVTZ single points on the barrier | PySCF | not started |
| [`step_6/`](step_6/README.md) | comparison to experiment: the end points' Hessians, the reference conformer, ΔH‡/ΔS‡/ΔG‡, boat vs chair; and two checks of the TS (energy-only curvature, PC walks from the CASSCF C2h points). Secondary KIEs dropped | free-breakfast | done: end points minima (numerical Hessians; the analytic one refuses them, #1268); reference the ensemble of ten conformers, lowest 0.25 kcal/mol below step 4b's; chair ΔH‡ 29.3 kcal/mol, ΔS‡ −15.1 eu at 505 K against 33.5 and −13.8; boat C2v TS (one B1 imaginary mode) ΔΔH‡ 10.2 against 11.2 ± 2.0; one C2h stationary point at PC-NEVPT2 |

## Step 4 with free-breakfast

free-breakfast has a quasi-Newton minimiser (`[optimize]`), the analytic
single-state DF-PC-NEVPT2 gradient and Hessian, and harmonic frequencies.
It has no point-group symmetry and no TS search. When step 4 was planned it
also had no constraints and no IRC; it now has both, `[optimize]`'s distance
`constraints` (#1260) and `[irc]` (#1269), and step 4d's IRC uses the latter.
**None of those is needed for the deciding test.**

**Under C2h the TS is a minimum.** The reaction mode is a_u, which is not
totally symmetric. The gradient at an exactly C2h geometry is totally
symmetric, so BFGS steps from an exactly C2h start stay C2h in exact
arithmetic. The only a_u component comes from noise. The DF energy is exactly
symmetric (the aux basis is atom-centred, and there is no DFT grid), so
that noise is GPU atomics and solver thresholds, around 1e-10. That is far
below the ~1e-5 gradient convergence, even amplified by negative a_u
curvature over the dozen or two iterations a walk takes. **So no symmetry
projection is used**: a plain `[optimize]` from a symmetric start converges on
the symmetric saddle, and the Hessian decides.

*Step 4b's measurement:* that held for PC-NEVPT2's walk, but not for CASSCF's
from 2.30 Å. There the a_u departure grew about 2× a step from 2e-11, so
20 steps took it to the threshold, and the walk left the saddle for
1,5-hexadiene. Its saddle was reached by a restart from a symmetrised
geometry. The optimiser's identity-started BFGS also overshoots stiff
non-symmetric directions, which grows round-off in sign-alternating steps
until BFGS learns them ([step 4b](step_4/README.md#step-4b-the-c2h-stationary-points-their-hessians-the-irc-stand-in)).

The plan:
1. **PC-NEVPT2.** `[optimize]` over `[nevpt2_gradient]` (PC), from step 1's
   R = 1.95 Å geometry, the minimum of step 2's PC-NEVPT2 E(R).
2. **CASSCF contrast.** `[optimize]` over `[casscf_gradient]` from step 1's
   1.65 Å and 2.30 Å geometries. The expectation is two distinct C2h stationary
   points at CASSCF against one at PC-NEVPT2.
3. **Hessian.** The analytic `[hessian]` at each stationary point.
   - One imaginary frequency, of a_u symmetry: an a_u saddle, **concerted**.
   - All a_u modes real: a diyl intermediate, **stepwise**.
4. **The IRC check.** Displace ± along the imaginary mode and run `[optimize]`
   again. Reaching reactant and product, with no minimum in between, is the
   connectivity check. It stood in for an IRC, and is now replaced by one
   ([step 4d](step_4/README.md#step-4d-the-irc)).

**Two checks replace the symmetry projection.** Both are read from the
`[optimize]` history and trajectory.
- **Geometry drift.** Log |R16 − R34| and the gradient's a_u component
  (its projection onto the inversion-antisymmetric displacements; step 2's
  `au.py` builds that basis) at every step. Noise-level values mean symmetry
  held. If either grows, the result is not trusted.
- **Wavefunction symmetry breaking, the real risk.** In the diradical region,
  CASSCF can converge to orbitals that break the symmetry, for example
  localised on one allyl, even at an exactly C2h geometry. Projecting the
  nuclear gradient would not stop that. A nonzero a_u gradient at a symmetric
  geometry is how it shows up, so the drift check above catches it too.

**The active space.** free-breakfast has no irrep labels, so 2a_g + 2b_u +
a_u + b_g cannot be asked for by name. Two routes:
- carry step 1's converged orbitals in through `[active_space]`'s `orbitals`
  .npy, **projected first** (see *Steps 1–2's basis is not this tree's*);
- or pick the six allyl π orbitals by character, as step 1's
  `cope.pi_character` does.

Either way, track the natural occupations of both diradical pairs along the
walk. Step 1 found that the pair that opens up is b_u/a_g′ at short R and
a_u/b_g at long R.

**PC, not SC.** SC-NEVPT2's gradient is singular at a degenerate inactive
pair, and free-breakfast refuses one. In C₆H₁₀ the in-phase and out-of-phase
combinations of distant C–H σ orbitals come close to that. PC avoids the
issue.

**Comparisons must use the same approximation.** free-breakfast's NEVPT2
gradients and Hessians are density-fitted only. Every external comparison
(block2, BAGEL, ORCA) is DF over the same aux set, cc-pVDZ-JKFIT, so the DF
error is not counted as an implementation error. Both sides correlate every
core orbital: step 2 freezes none, and `[nevpt2]`'s `frozen_core` defaults
to 0.

### Steps 1–2's basis is not this tree's

Steps 1 and 2 ask PySCF for `"cc-pvdz"` and `"cc-pvdz-jkfit"` by name, not
for this tree's `data/basis/cc-pVDZ.json` and
`data/aux_basis/cc-pVDZ-JKFIT.json`. Checked at step 1's R = 1.95 Å geometry:

| | PySCF built-in vs this tree's JSON |
|---|---|
| AO order | the same: 134 AOs with identical labels; p as x, y, z, then m = −l…+l, each contraction's components together, as `src/basis` orders them |
| contractions | **different.** PySCF's built-in cc-pVDZ splits the outer primitive of each carbon s and p off into its own shell; the BSE JSON is the general contraction. The overlap between same-label AOs is as low as 0.915 (C 2s) |
| span | **identical**, orbital and aux: every function of each set lies in the other's span to 2e-15 |

Two consequences:
- **Energies and gradients carry over exactly.** CASSCF, DF and NEVPT2 depend
  only on the spans, so step 2's block2 energies are a valid reference for
  step 4a at the same geometries, with no rerun.
- **Orbital files do not.** The same columns over different AOs are not the
  same orbitals: step 1's `orbitals/R*.npy` read as-is has an
  orthonormality defect of 0.58 in this tree's overlap. `[active_space]`'s
  `orbitals` would Löwdin it away without complaint and start from the wrong
  orbitals. Project first, C_tree = S_tree⁻¹ S_tree,pyscf C_pyscf, where
  S_tree,pyscf is the cross overlap (`gto.intor_cross`). Because the spans are
  equal, this is exact: the projected orbitals are orthonormal to 1e-13.

Atom order and units need nothing. The points' geometries are Å, C1…C6 then
the hydrogens, and PySCF's Bohr (CODATA 2010) differs from this tree's
(CODATA 2018) by 3e-11 relative.

## Step 3, after step 4

Done: [step 3](step_3/README.md) has the result. This section is the plan
as it was written.

The 2D (R16, R34) surface is the clearest figure: one saddle on the diagonal
against a diagonal well flanked by two saddles. It is not needed for the
verdict, since step 4's Hessian already decides it. It is still worth
doing, and free-breakfast is the right tool for it:
- **Not done in PySCF.** PySCF has no PC-NEVPT2 gradient (step 2's PC is
  block2 energies), so a PySCF grid would be relaxed at CASSCF with PT2
  single points. That keeps the CASSCF diradical bias the study is about.
- **Blocked on one feature**, since added (#1260). Relaxing at fixed R16
  and R34 needs distance constraints in `[optimize]`.
  C2 needs nothing: as with C2h, a symmetric start stays symmetric.
- **Affordable.** Half of a 1.50–2.60 Å grid at 0.05 Å is ~280 relaxed
  PC-NEVPT2 optimisations of a 134-function system. That is the throughput
  case the GPU implementation is for.

## References

The mechanism and its computational history:
- Y. Osamura, S. Kato, K. Morokuma, D. Feller, E. R. Davidson, W. T. Borden,
  *J. Am. Chem. Soc.* **106**, 3362 (1984). [doi:10.1021/ja00323a055](https://doi.org/10.1021/ja00323a055).
  The first CASSCF chair TS.
- M. Dupuis, C. Murray, E. R. Davidson, *J. Am. Chem. Soc.* **113**, 9756
  (1991). [doi:10.1021/ja00026a007](https://doi.org/10.1021/ja00026a007).
  CASSCF's diyl and "aromatic" C2h structures.
- D. A. Hrovat, K. Morokuma, W. T. Borden, *J. Am. Chem. Soc.* **116**, 1072
  (1994). [doi:10.1021/ja00082a032](https://doi.org/10.1021/ja00082a032).
  CASPT2N: one concerted chair TS, at too short an R.
- P. M. Kozlowski, M. Dupuis, E. R. Davidson, *J. Am. Chem. Soc.* **117**, 774
  (1995). [doi:10.1021/ja00107a021](https://doi.org/10.1021/ja00107a021).
  Multireference MP2 on the chair region.
- S. Sakai, *Int. J. Quantum Chem.* **80**, 1099 (2000).
  [doi:10.1002/1097-461X(2000)80:4/5<1099::AID-QUA59>3.0.CO;2-C](https://doi.org/10.1002/1097-461X(2000)80:4/5%3C1099::AID-QUA59%3E3.0.CO;2-C).
  CASSCF chair, boat and Dewar-type TSs by CiLC-IRC.
- V. N. Staroverov, E. R. Davidson, *J. Am. Chem. Soc.* **122**, 7377 (2000).
  [doi:10.1021/ja001259k](https://doi.org/10.1021/ja001259k).
  C2h cuts through the PES, with cyano substituents.
- V. N. Staroverov, E. R. Davidson, *J. Mol. Struct. THEOCHEM* **573**, 81
  (2001). [doi:10.1016/S0166-1280(01)00536-X](https://doi.org/10.1016/S0166-1280(01)00536-X).
  Review: why generally reliable methods fail in the transition region.
- E. Ventura, S. Andrade do Monte, M. Dallos, H. Lischka, *J. Phys. Chem. A*
  **107**, 1175 (2003). [doi:10.1021/jp0259014](https://doi.org/10.1021/jp0259014).
  MR-CISD and MR-AQCC geometry optimisations of the chair and boat TSs.
- N. Graulich, *WIREs Comput. Mol. Sci.* **1**, 172 (2011).
  [doi:10.1002/wcms.17](https://doi.org/10.1002/wcms.17). Review of the
  experimental and theoretical record.
- J. L. Mackey, Z. Yang, K. N. Houk, *Chem. Phys. Lett.* **683**, 253 (2017).
  [doi:10.1016/j.cplett.2017.03.011](https://doi.org/10.1016/j.cplett.2017.03.011).
  Dynamically concerted and stepwise trajectories.

NEVPT2 analytic derivatives:
- J. W. Park, *J. Chem. Theory Comput.* (2019).
  [doi:10.1021/acs.jctc.9b00762](https://doi.org/10.1021/acs.jctc.9b00762);
  [arXiv:1907.10180](https://arxiv.org/abs/1907.10180). Analytic SC- and
  PC-NEVPT2 gradients (BAGEL), the reference for step 4c.
- J. W. Park, *J. Chem. Theory Comput.* (2019).
  [doi:10.1021/acs.jctc.9b00919](https://doi.org/10.1021/acs.jctc.9b00919).
  Analytic QD-NEVPT2 gradients.
- C. Kollmar, K. Sivalingam, Y. Guo, F. Neese, *J. Chem. Phys.* **155**, 234104
  (2021). [doi:10.1063/5.0072129](https://doi.org/10.1063/5.0072129).
  NEVPT2 and CASPT2 without the 4-RDM (ORCA).
- Y. Nishimoto, ChemRxiv (2026).
  [doi:10.26434/chemrxiv.15002579](https://doi.org/10.26434/chemrxiv.15002579).
  QD-NEVPT2 analytic derivatives in OpenMolcas.
- R. D. Guerrero, arXiv:2609.09990 (2026).
  [arXiv:2609.09990](https://arxiv.org/abs/2609.09990). DMRG-QD-NEVPT2
  gradients and NACs on a consumer GPU: the prior GPU work.
