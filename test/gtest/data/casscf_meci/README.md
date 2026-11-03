# `casscf_meci/`

`casscf_meci.json`, written by `tools/emit/meci_reference.py` and never
hand-edited: the S0/S1 minimum-energy conical intersection (MECI) of
ethylene's density-fitted SA2-CASSCF(2,2), the reference the `[optimize]
meci` test of milestone 22 reads (#774).

**It is this tree's own loop over PySCF's quantities, not another
program's MECI:** no available program finds a DF-SA-CASSCF MECI. PySCF
supplies, at each geometry, the two roots' gradients (`df.grad.sacasscf`)
and the coupling h_01 (`../casscf_nac/`'s `DFNonAdiabaticCouplings`, the
`mult_ediff=True`, `use_etfs=True` numerator, the same quantity as
`Nac::numerator`, no CSF term); the tool's loop does the rest.

## The case

Ethylene, cc-pVDZ over `data/aux_basis/cc-pVDZ-JKFIT.json`, CAS(2,2) (π,
π*), two singlet roots at equal weights, ncore 7. The start has the second
CH₂ twisted 80° about C–C and pyramidalized (C–C 1.40 Å, C–H 1.08 Å, H–C–H
117°, that CH₂'s bisector bent 100° off the C–C axis toward its own p
orbital), near the twisted-pyramidalized MECI, and has no symmetry: from
the same start twisted 90° (Cs) the loop keeps Cs and finds a **saddle** on
the seam 6.6 mEh higher (P (∂ḡ/∂R) P positive there, at least 0.0122, but
the seam's own curvature, below, −0.028 along a Cs-breaking mode). Each
geometry's DF-SA-CASSCF is built, converged, polished (|g| < 1e-10), canonicalized and its CI
re-solved densely as in `../casscf_nac/`; the start's orbitals are the exact
RHF's, every later geometry's the previous one's, Löwdin-carried.

## The loop

Each step is the Bearpark–Robb–Schlegel composite gradient's two parts,
each taken as a Newton step:

- **in the branching plane** {x = g_1 − g_0, y = h_01}: the least
  displacement d in span{x, y} with x·d = −gap and y·d = 0. On a linear
  cone, elliptic or tilted, this lands on the seam: the gap is homogeneous
  of degree one in the distance from it (so x·δ = gap), and y is tangential
  to it.
- **in the seam**: Newton on the projected gradient P ḡ, ḡ = (g_0 +
  g_1)/2, at the geometry the plane step reaches, over its Jacobian along
  the seam, each mode at its |curvature| (at least 5e-3) so a step never
  climbs. The value and Jacobian come from central differences (step 1e-3
  bohr, recomputed at every iterate) of the projected force along the
  seam's ten and the plane's two orthonormal directions, each displaced
  point projected over its own plane.

That Jacobian is P (∂ḡ/∂R) P plus the plane's turn along the seam acting
on ḡ's in-plane part, i.e. the mean energy's second derivative along the
seam itself (the Lagrangian's reduced Hessian). The cone here is steeply
tilted (`gbar_plane` 0.066 Eh/bohr against in-plane slopes of 0.14 to
0.23), so the turn matters: with
P (∂ḡ/∂R) P alone the loop contracted by only about 0.4 per step and then
diverged near 2e-6 Eh/bohr. With the full Jacobian it converges
quadratically.

The projector is #771's: the three translations and three rotations about
the centroid are removed, {x, y} is orthonormalized in that order after
them, and P projects off all eight. The **projected force** is −P ḡ. The
loop stops when the gap is under **1e-8 Eh** and the largest projected force
under **1e-6 Eh/bohr**.

## What each derived quantity is for

#774 derives its bars from these, with F its max-force threshold plus this
reference's 1e-6 and Δ its gap threshold plus this reference's 1e-8:

| Key | What it is | What it is for |
|---|---|---|
| `meci.k_min` | The least eigenvalue of P (∂ḡ/∂R) P on the 3N − 8 = 10 seam directions, ḡ's central difference above along the seam's orthonormal directions at the MECI, symmetrized (all ten in `seam_curvatures`); the tool refuses a k_min ≤ 0 | The seam term of the geometry bar, F/k_min: a residual force F leaves the geometry at most F/k_min from the seam minimum. Positive, with `reduced_curvatures` (below), means a minimum on the seam, not a saddle |
| `meci.x_norm` | \|x\| = \|g_1 − g_0\| at the MECI | With `h_norm`, the branching-plane term Δ/min(x_norm, 2·h_norm): a residual gap Δ leaves the geometry about that far off the seam |
| `meci.h_norm` | \|h_01\| at the MECI | As `x_norm`; the gap rises at least 2·\|h_01\| per bohr along y |
| `meci.gbar_plane` | The norm of ḡ's component in the orthonormalized plane (the cone's tilt) | The mean-energy bar's plane term, gbar_plane·Δ/min(x_norm, 2·h_norm): the mean energy is linear in the plane error through the tilt (and quadratic along the seam, F·F/k_min) |
| `s2_margin` | min(E_2 − E_1) over every geometry the tool evaluated (the iterates and every difference's displaced point), E_2 the third singlet root of a CASCI at that geometry's SA2 orbitals; the tool refuses one under 10 mEh | A margin, not a bound: the second zwitterion is outside the average and must stay clear of S1, or a root-following chain could flip roots on the way |

`meci.reduced_curvatures` are the eigenvalues of the symmetrized Jacobian of
P ḡ along the seam at the MECI, the seam's own curvature (the plane's turn
included). The tool refuses the fixture unless all ten are positive: that,
not k_min alone, is what makes the point a minimum on the seam. Its least
(0.0311) differs from k_min (0.0338); both are written, and k_min is the
issue's definition.

**`x_norm` and `h_norm` are not reproducible at a converged MECI. Their
invariants are.** Near a cone the pair (x/2, h_01) is a rotation, within
the plane, of one fixed pair, by an angle set by the direction the point
was approached from. At a gap of 1e-13 that angle is arbitrary: the
committed run has x_norm 0.1766, h_norm 0.1015, and its rerun 0.2222,
0.0759. Both are written as the issue defines them. The invariants are the
singular values of the 2 × 3N matrix [x/2; h_01]: twice them are the gap's
greatest and least slope in the plane, **0.2290 and 0.1413 Eh/bohr** here,
equal in both runs to 1e-12, as is |x/2|² + |h_01|². Each of x_norm and
2·h_norm is at least the least slope, so min(x_norm, 2·h_norm) ≥ 0.1413
always, and a branching-plane term over 0.1413 is the conservative one.
`test/pytest/test_meci_reference.py` recomputes both slopes from the
written vectors.

## Keys

| Key | Meaning |
|---|---|
| `name`, `basis`, `charge`, `electrons`, `nao`, `nmo`, `ncore`, `ncas`, `nelecas`, `weights` | As in `../casscf_nac/` |
| `aux_basis` | `name`, `file` (from the repository root), `sha256`, `naux` |
| `start` | At the start: `atoms` (`symbol`, `z`, `coord` in bohr), `energies` E_0, E_1, `gradients` g_0, g_1 (n_atom × 3, hartree/bohr), `h` h_01 (bra 0, ket 1, as `../casscf_nac/`'s `h`), `ci` (each root's CI vector, n_alpha × n_beta strings, signed by `../casscf_nac/`'s rule) and `natural_occupations` |
| `meci` | At the MECI: `atoms`, `energies`, `gap`, `mean_energy`, `gradients`, `h`, `projected_force` (−P ḡ, n_atom × 3), `max_projected_force`, `k_min`, `seam_curvatures`, `reduced_curvatures`, `x_norm`, `h_norm`, `gbar_plane` |
| `s2_margin` | As above |
| `fresh` | The independent single point at `meci.atoms`: a fresh RHF (second-order SCF where DIIS stalls), the SA-CASSCF from it by mc1step (whose `energies`, `gap` and `max_projected_force`, over its own x and h_01, are written) and by mc2step (`mc2step_agreement`, the largest root difference). The tool requires gap < 1e-8 and force < 1e-6 |
| `path` | One record per iterate: `energies`, `gap`, `max_projected_force`, `s2_margin`, `coords` (bohr) |
| `optimizer` | The loop's tolerances, Hessian step, step cap and margin floor |

The start's h_01 sign is fixed by the CI sign rule and PySCF's orbital
signs, as in `../casscf_nac/`; a test in its own orbitals compares it up to
s_0 s_1. The MECI's `h` is written for completeness; only its norm is a
reference.

## Precision

The MECI geometry is the true seam minimum to within δ = 1e-6/k_min +
1e-8/min(x_norm, 2·h_norm) = 3.0e-5 bohr (the loop's tolerances, not the
gap and force it reached, 3e-13 and 2e-13). Its energies are each converged
to better than 1e-10 Eh. The mean energy at the recorded point is the true
MECI's to within gbar_plane·1e-8/min(x_norm, 2·h_norm) + (1e-6)²/k_min =
3.7e-9 Eh. A rerun agrees with the committed file well within that: the
last regeneration's rerun, by `--out`, differed by 3.9e-8 bohr in the
geometry, 1.4e-13 Eh in the energies, 8e-8 in k_min and 2e-13 in
gbar_plane. `x_norm` and `h_norm` are the exception, as above.

## Regenerating

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/emit/meci_reference.py
```

About 16 minutes. PySCF's OpenMP runs at 8 threads (the tool's `THREADS`)
and BLAS at one (PySCF calls BLAS inside its OpenMP loops, so a threaded
BLAS would nest past 8 cores), and no date or host is written; threaded
reductions make a rerun agree to the precision above, not byte for byte.
`test/pytest/test_meci_reference.py` re-derives the projected force, the
norms, the in-plane slopes and the margins with the standard library alone, and (slow, needs the
`tools` group) reruns the independent single point.
