# Intersection

What a minimum-energy conical intersection (MECI) search takes at each
geometry of a converged DF-SA-CASSCF, from one call: `mcscf::df_intersection`,
the nuclear gradients g_I and g_J of two roots of equal weight and their
coupling h_IJ, the numerator of `df_nac`, computed so that nothing divides by
the gap E_I − E_J and the call is as good at the seam as away from it.

## Module

`free_breakfast.mcscf.intersection`, target
`free_breakfast::mcscf::intersection`, re-exported by `free_breakfast.mcscf`.

```cpp
Result<mcscf::Intersection> mcscf::df_intersection(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t i,
    int32_t j, const mcscf::IntersectionOptions &options = {});
// x->g_i, x->g_j, x->h_ij: each n_atom x 3, row-major, element 3 A + k;
// x->iterations, x->residual: the block Z-vector solve's
```

Every argument through `ncas` is `df_gradient`'s and `df_nac`'s
([`../gradient/`](../gradient/README.md), [`../nac/`](../nac/README.md)).
`options.tolerance` is each Z-vector column's, 1e-10 by default, the one
`df_gradient`'s own solve takes; `options.max_iterations` caps the block
solve's products, 200 by default, `zvectors`' own.

## What it is

The derivative of the pair's 2 × 2 Hamiltonian, turned to its eigenvectors:

- `g_i` and `g_j` are dE_I/dR and dE_J/dR, the adiabatic gradients.
- `h_ij` is h_IJ = ⟨I|∂H/∂R|J⟩ with the orbitals and CI relaxed, as
  `df_nac`'s `numerator`: symmetric in I and J, and changing sign with
  either root's CI phase as `solution` carries it. No CSF term and no
  division by the gap.

**Away from a crossing** each equals its own call — `df_gradient(…, i)`,
`df_gradient(…, j)`, `df_nac(…, i, j).numerator` — to the Z-vector solves'
convergence (Tests). **Near one**, the two roots' CI vectors are fixed only
to the CI solver's convergence within their 2-dimensional space, and so are
g_I, g_J and h_IJ: at a gap of 3e-13 Eh the pair (x/2, h_IJ), x = g_J − g_I,
is a rotation within the branching plane by an angle the noise sets. What is
fixed there, and what a search reads, is the **mean gradient**
ḡ = (g_I + g_J)/2 and the **branching plane** span{x, h_IJ}: both are
invariant to that rotation, and neither is computed through a 1/gap.

## The derivation

**The 2 × 2.** Let c_I and c_J be the solution's two roots, at equal weight
w in the state average, and H_KL(R) = ⟨c_K|Ĥ(R)|c_L⟩, K, L ∈ {I, J}, with
the orbitals and the CI relaxed with the geometry. The state-averaged energy
E_SA = Σ_r w_r ⟨c_r|Ĥ|c_r⟩ is stationary in the orbital rotations κ and in
the CI rotations P out of the averaged space, and *not* in the rotation
between I and J: at equal weights it leaves E_SA unchanged, so it is no
parameter of the state average at all (redundant). The relaxation of H_KL is
therefore the state average's alone. Its Lagrangian,

    L_KL = H_KL + z_KL · ∂E_SA/∂(κ, P),

is stationary in (κ, P) when A z_KL = −∂H_KL/∂(κ, P), A the SA Hessian, and
dH_KL/dR = ∂L_KL/∂R. The CI part of the right-hand side is zero: every root
is an eigenvector of the same active-space H₀, so P Ĥ c_L = 0 for K = L and
K ≠ L alike. Eliminating the CI leaves the CI-relaxed orbital Hessian H̃
([`../relaxed_hessian/`](../relaxed_hessian/README.md)) and its Z-vector
H̃ z_o = −g_KL, g_KL the orbital gradient of H_KL at the solution
([`../zvector/`](../zvector/README.md)). This is the SA-CASSCF coupling
numerator of Galván et al. (I. F. Galván, M. G. Delcey, T. B. Pedersen, F.
Aquilante and R. Lindh, JCTC 12, 3636 (2016)) for K ≠ L, and the SA-CASSCF
gradient's Lagrangian for K = L, at the same multiplier equations.

**Three columns, not four.** H_KL is linear in the pair's RDMs, so the
derivatives are taken in the combinations a search reads:

| Column | Combination | RDMs | Orbital right-hand side | Derivative call |
|---|---|---|---|---|
| 0 | (H_II + H_JJ)/2 | the mix (½, ½) | `response::fock`'s gradient, core included | `df_gradient` at the mix (½, ½) |
| 1 | (H_II − H_JJ)/2 | ½(γ_I − γ_J) | `transition_orbital_gradient` (the core cancels) | `df_gradient` at the sum-0 mix (½, −½) |
| 2 | H_IJ | the symmetrized transition RDMs | `transition_orbital_gradient` | `df_transition_gradient` |

The three are solved together, one `zvectors` (Schur) over the one H̃, and
each column's multipliers go to its derivative call, which solves nothing.
The mix calls are #741's (`df_gradient(…, mix, …)`): a sum-0 mix is the
transition gradient at the mix's RDMs, so column 1 is the half difference of
the two roots' derivatives with no core or nuclear term, as it must be.

**The turn to the eigenvectors.** At the solution the 2 × 2 is diagonal to
the CI's convergence. It is formed from the sigma (H_II, H_JJ and H_IJ =
½(c_I · H₀ c_J + c_J · H₀ c_I)) and diagonalized: its eigenvectors are c_I
and c_J turned by φ, tan 2φ = H_IJ / ((H_II − H_JJ)/2), |φ| ≤ π/4, the
smallest turn, so root I stays root I. The eigenvalues' derivatives are the
turned matrix's diagonal and the coupling its off-diagonal (Hellmann–Feynman
within the 2 × 2, the eigenvector held: J. W. Park, JCTC,
doi:10.1021/acs.jctc.9b00919 (2019), whose QD-NEVPT2 gradient is
E_P = Σ R_PPMN H_MN at a fixed eigenvector R, Eqs. 17–20, and whose
interstate coupling replaces R_PPMN by R_PQMN, Eqs. 46–47: the same
construction over a model space),
with ḡ = dH_mean, x̃ = dH_diff and h̃ = dH_IJ the three columns:

    g_I = ḡ + cos 2φ x̃ + sin 2φ h̃,
    g_J = ḡ − cos 2φ x̃ − sin 2φ h̃,
    h_IJ = −sin 2φ x̃ + cos 2φ h̃.

Away from a crossing φ is H_IJ over a gap of hundredths of a hartree, a
turn of roundoff. At the seam it is whatever the CI's noise makes it, and any
value is right: ḡ does not depend on φ, and (x/2, h_IJ) is (x̃, h̃) turned by
2φ, the same plane — the branching plane of Bearpark, Robb and Schlegel
(CPL 223, 269 (1994)), along which the gap rises at the two slopes
twice the singular values of [x/2; h_IJ] (`../../../test/gtest/data/casscf_meci/`).

**What the previous version lost, and where.** #773's version solved g_I's,
g_J's and h_IJ's own right-hand sides; in exact arithmetic those are the
same three derivatives (the I–J rotation is no parameter there either). It
diverged in the CI-relaxed Hessian's product: the CI response δ_r =
−(H₀ − E_r)⁻¹ (1 − c_r c_rᵀ) H′ c_r carries every other root's part exactly,
−(c_s · H′ c_r)/(E_s − E_r) c_s, and the I–J pair's adds (w_I − w_J) times
that, over the transition RDMs, to H̃ x. That is zero at equal weights, but
weights recovered from the solution's averaged RDMs by least squares
(`sa_gradient_weights`, then the only route) are equal only to about 1e-8
relative, and the
term is an O(1/gap) number cancelling through the transition RDMs (at #774
the difference of two RDMs of c ± δ, since #691 `ci::Rdm::trans_rdm12`
summed over the roots). Measured in #774 near ethylene's seam, the true
residual of every column grew as 5e-12 / gap from a gap of 1e-5 Eh down to
the RelaxedHessian's floor of 1e-8 (where the mixing is dropped outright),
reaching 1e-3, while the conjugate gradients' recurrence still met 1e-10 and
the multipliers came back with no error. Here H̃ is created with `RelaxedHessian::create`'s `mixing`
mask from `rotating_pairs(weights)`: a pair of equal weights, I and J among
them, carries no mixing, which is the state average's response exactly, and
a pair of unequal weights (another root's against I or J) keeps its own, as
`df_gradient` has it.

**An unconverged Z-vector is an error.** After the block solve each column's
true residual, ||H̃ z + g|| from one more product, is checked against the
tolerance; a column above it is `Numerical`, naming the column and the
residual. A solve that runs out of products is `zvectors`' own
`MaxIterationsExceeded`. Multipliers short of the tolerance never reach the
derivative calls.

## What the three share

Three separate calls build the CI objects and the CI-relaxed Hessian H̃ at
the solution three times, and solve three Z-vectors one after another. Here:

1. The CI strings, sigma, RDMs, the fitted integrals, the averaged Fock, the
   orbital Hessian and H̃ are built once, with the weights `df_gradient`
   takes (`sa_gradient_weights`: the solution's own, recovered from its RDMs
   only when it carries none), the equal-weight pairs' mixing left out.
2. The three right-hand sides of "The derivation" are solved together, one
   `zvectors` over that H̃, the Schur solver (its default), and each
   column's true residual is checked.
3. Each column goes, as given multipliers, to its own derivative call:
   `df_gradient` at the mixes (½, ½) and (½, −½), `df_transition_gradient`
   at the transition RDMs. Those calls solve nothing.
4. The 2 × 2 from two sigmas, and the turn of the three to its
   eigenvectors, on the host.

**The derivative contractions are not stacked.** The issue (#773) asked for
the three-center and metric contractions to be done once over the three
density sets. Measured on the RTX 3080 (the skewed H₂CO SA-3, cc-pVDZ /
cc-pVDZ-JKFIT), each derivative call is 57 ms, of which
`three_center_gradient` is 53 ms, and of that the geri3 kernels are 50 ms
(the staging 2.6 ms, the Cartesian transform 0.7 ms); `metric_gradient` and
the one-electron terms are under 1 ms each. So the only stacking that pays is
a geri3 that contracts several Γ against each integral it evaluates, a
generator change in `deps/FelizJueves` (`tools/emit/df_gradient_kernels.py`)
and a new `three_center_gradient` overload, outside this module. The
public route that does exist, the materialized derivative integrals
(`three_center_derivative` over every aux atom, then one contraction per set),
costs 167 ms for the integrals alone, more than three geri3 passes (150 ms),
so it is not taken.

## Refusals

- **Unequal weights on I and J** (`Unsupported`, naming the reason): two
  weights differing by more than `kRotationWeightTolerance` of the larger.
  The rotation between I and J is then a parameter of the Lagrangian, and its
  multiplier divides by E_I − E_J, which is singular at an intersection.
  Other roots may carry other weights. The weights are the solution's own,
  so equal ones are equal exactly: before #856 they were recovered from the
  RDMs, and at ethylene's C3v-ethylidene MECI, where the roots' 1-RDMs are
  nearly collinear, equal weights came back unequal past the tolerance and
  every walk there was refused.
- `InvalidConfig`: fewer than two roots, I = J, an index outside the average,
  a tolerance not positive or `max_iterations` below 1, and `df_nac`'s checks
  of the solution, the orbitals and the fit.
- **An unconverged Z-vector**: `MaxIterationsExceeded` (`zvectors`', a
  column still above the tolerance after `max_iterations` products) or
  `Numerical` (a column's true residual above it, the recurrence having
  passed it). Never multipliers short of the tolerance.

## Timing

The wall clock against the three separate calls (`df_gradient(I)`,
`df_gradient(J)`, `df_nac(I, J)`), on each case's roots (0, 1): one warm-up of
each, then the median of three (`McscfIntersectionTiming.AgainstThreeCalls`,
slow tier only), on the **RTX 3080 (CUDA)**, the card held. The gfx1200 was
not measured.

| Case | One call | Three calls | |
|---|---|---|---|
| H₂CO SA-2, (0, 1) | 0.209 s | 0.260 s | 1.25× |
| skewed H₂CO SA-3, (0, 1) | 0.248 s | 0.317 s | 1.28× |

What it saves is two of the three H̃ builds and the serial solves: its setup
and block solve take about 70 ms on the SA-3, where the three calls' own take
about 140 ms. The three derivative calls, 57 ms each, are most of what is
left. It is faster, so it is not behind an
option.

## Memory and stream

`df_gradient`'s, with H̃, the CI objects and the three columns of multipliers
held beside them. Queued on `res.stream()`; every call it makes synchronizes,
so it returns with everything done.

## Tests

`test/gtest/mcscf_intersection_gpu.cpp`.

**Away from a crossing** (`McscfIntersection`), on
`test/gtest/data/casscf_nac/`'s two cases at their equal weights, every pair
I < J: g_I, g_J and h_IJ against `df_gradient(I)`, `df_gradient(J)` and
`df_nac(I, J)`'s numerator, each **1e-9 relative to the largest element of
the two vectors compared**. Every one of the three compares independently
converged Z-vector solves, so each is solver-limited, and 1e-9 is the bar the
ruling on #767 sets for such a check; none is a round-off-only path
agreement, so none is held at 1e-10. Then the refusals: I = J and a root past
the average (`InvalidConfig`), unequal weights (`Unsupported`, the weighted γ
re-mixed so that it is no longer the carried weights' sum and the recovered
weights differ), and the block solve capped at one product
(`MaxIterationsExceeded`).

**At the seam**, on `test/gtest/data/casscf_meci/`'s ethylene
SA2-CASSCF(2,2), cc-pVDZ / cc-pVDZ-JKFIT:

- `McscfIntersectionSeam.MeciAgainstReference`: a fresh evaluation at the
  reference's MECI. Every column's true residual at most 1e-10; −P ḡ (off the
  translations, the rotations about the centroid and our plane, as the
  reference's projector) within **1e-6 Eh/bohr** of the reference's
  projected force, element by element; and the gap's in-plane slopes, twice
  the singular values of [x/2; h_IJ], against those recomputed from the
  reference's vectors (0.2290 and 0.1413 to four places) within
  **2 √(6 n_atom) · 1e-8** of the largest element of its g₀, g₁ and h₀₁:
  the tree's 1e-8 bar against PySCF's gradients and couplings carried
  through Weyl's inequality (a singular value moves by at most the change's
  spectral norm, at most its Frobenius norm).
- `McscfIntersectionNearSeam.CarriedAgreesWithFresh/{Meci, AlongX,
  AlongH}`: at the MECI and at the MECI moved 1e-7 bohr along the
  reference's x and h₀₁ (gaps of about 2e-8 Eh), a fresh-orbital evaluation
  against a carried-orbital one (the CASSCF's orbitals at the reference's
  iterate 2.4e-3 bohr away, Löwdin-carried as `[optimize]` carries them).
  Every column's true residual at most 1e-10; ḡ within **1e-9 of the larger
  element** (#767's bar, two independent solves), and the two planes'
  largest principal angle below **1e-6 rad**. `AlongX` stands for the three
  in the fast tier: the MECI's own gap is under the RelaxedHessian's 1e-8
  floor, below which #773's version already dropped the mixing.

Measured on the RTX 3080 (CUDA):

| Check | Bar | Measured |
|---|---|---|
| True residual, every evaluation (15 products) | 1e-10 | 2.5e-11 to 7.7e-11 |
| MECI: −P ḡ against the reference's projected force | 1e-6 Eh/bohr | 2.0e-12 |
| MECI: in-plane slopes 0.2290407634, 0.1412706023 against the reference's | 7.98e-9 | 1.2e-11, 1.4e-11 |
| `Meci` (gap 1.2e-11): carried vs fresh ḡ, plane | 1e-9, 1e-6 rad | 2.6e-10, 6.8e-11 rad |
| `AlongX` (gap 1.97e-8): carried vs fresh ḡ, plane | 1e-9, 1e-6 rad | 5.5e-10, 1.5e-10 rad |
| `AlongH` (gap 2.17e-8): carried vs fresh ḡ, plane | 1e-9, 1e-6 rad | 3.0e-10, 7.3e-11 rad |
| Away: g_I, g_J, h_IJ against the separate calls (worst, skewed SA-3 (0, 1)) | 1e-9 | 1.8e-10, 1.0e-10, 2.2e-10 |

(the slow tier's run; the fast tier's agreed to the same digits).

With the mixing kept for every pair (#773's H̃, everything else as here),
`AlongX` stops at `Numerical`: column 1's true residual is 1.6e-5 after 16
products, the recurrence having met 1e-10.

## Files

| File | Role |
|------|------|
| `intersection.cppm` | Module interface: `IntersectionOptions`, `Intersection`, `df_intersection` |
| `intersection.cpp` | Implementation: the shared H̃ (equal-weight pairs unmixed), the three right-hand sides, the block solve and its residual check, the three derivative calls, the 2 × 2's turn |
| `CMakeLists.txt` | The `free_breakfast.mcscf.intersection` module library |
